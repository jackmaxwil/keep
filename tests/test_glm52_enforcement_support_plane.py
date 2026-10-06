from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class _CloudFormationLoader(yaml.SafeLoader):
    pass


def _cloudformation_tag(
    loader: _CloudFormationLoader,
    suffix: str,
    node: yaml.Node,
):
    if isinstance(node, yaml.ScalarNode):
        return f"!{suffix} {loader.construct_scalar(node)}"
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


_CloudFormationLoader.add_multi_constructor("!", _cloudformation_tag)


def _gpu_teacher_template() -> dict[str, object]:
    path = ROOT / "aws/glm52-gpu/cfn/gpu-teacher-stack.yaml"
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=_CloudFormationLoader)
    assert isinstance(value, dict)
    return value


def _support_deletion_inventory() -> dict[str, object]:
    account = "246813579024"
    region = "us-west-2"
    activation = "act-20260728-0001"
    ec2_ids = {
        "instance": ["i-00000000000000001"],
        "volume": ["vol-00000000000000002"],
        "natgateway": ["nat-00000000000000003"],
        "network-interface": ["eni-00000000000000004"],
        "route-table": [
            "rtb-00000000000000005",
            "rtb-00000000000000006",
            "rtb-00000000000000007",
        ],
        "subnet": [
            "subnet-00000000000000008",
            "subnet-00000000000000009",
            "subnet-0000000000000000a",
        ],
        "security-group": [
            "sg-0000000000000000b",
            "sg-0000000000000000c",
            "sg-0000000000000000d",
            "sg-0000000000000000e",
            "sg-0000000000000000f",
            "sg-00000000000000010",
        ],
        "vpc-endpoint": [
            "vpce-00000000000000011",
            "vpce-00000000000000012",
            "vpce-00000000000000013",
        ],
        "elastic-ip": ["eipalloc-00000000000000014"],
    }
    ec2 = {
        kind: [
            f"arn:aws:ec2:{region}:{account}:{kind}/{resource_id}"
            for resource_id in resource_ids
        ]
        for kind, resource_ids in ec2_ids.items()
    }
    function_names = [
        f"keep-glm52-h1g-support-{name}"
        for name in (
            "tls-handler",
            "decision",
            "attestation",
            "launch-admission",
            "numeric-binding",
            "retained-cancellation",
            "deadline",
        )
    ]
    functions = [
        f"arn:aws:lambda:{region}:{account}:function:{name}" for name in function_names
    ]
    versions = [f"{arn}:7" for arn in functions]
    role_names = [
        f"keep-glm52-h1g-support-{name}"
        for name in (
            "combined-host",
            "decision",
            "attestation",
            "launch-admission",
            "numeric-binding",
            "retained-cancellation",
            "tls-handler",
            "workflow",
            "deadline",
            "schedule-invoke",
        )
    ] + [
        "keep-glm52-h1g-closure-session-"
        + hashlib.sha256(activation.encode("ascii")).hexdigest()[:16]
    ]
    roles = [f"arn:aws:iam::{account}:role/{name}" for name in role_names]
    profile_name = "keep-glm52-h1g-support-combined-host"
    profile = f"arn:aws:iam::{account}:instance-profile/{profile_name}"
    secret_arns = [
        (
            f"arn:aws:secretsmanager:{region}:{account}:secret:"
            f"/keep/glm52/glm52-sky-20260724/{activation}/{purpose}-ABC123"
        )
        for purpose in (
            "raw-sky-token",
            "sky-bootstrap-hash",
            "attestation-client-tls",
            "launch-admission-client-tls",
            "numeric-binding-client-tls",
            "retained-cancellation-client-tls",
            "combined-host-server-tls",
            "ca-issuance",
        )
    ]
    logs = [
        (f"arn:aws:logs:{region}:{account}:log-group:/aws/lambda/{name}:*")
        for name in function_names
    ]
    alarms = [
        f"arn:aws:cloudwatch:{region}:{account}:alarm:"
        f"keep-glm52-h1g-{activation}-alarm-{index:02d}"
        for index in range(12)
    ]
    bucket = f"arn:aws:s3:::keep-glm52-h1g-rehearsal-{account}-{region}"
    queue = f"arn:aws:sqs:{region}:{account}:keep-glm52-h1g-support-dlq-{activation}"
    state_machine = (
        f"arn:aws:states:{region}:{account}:stateMachine:"
        f"keep-glm52-h1g-support-{activation}"
    )
    schedules = [
        f"arn:aws:scheduler:{region}:{account}:schedule/default/"
        f"keep-glm52-h1g-{activation}-{name}"
        for name in ("work-stop", "delete-request", "absence-deadline")
    ]
    event_rule = (
        f"arn:aws:events:{region}:{account}:rule/keep-glm52-h1g-{activation}-egress"
    )
    kms = f"arn:aws:kms:{region}:{account}:key/12345678-1234-4234-8234-1234567890ab"
    root_volume_id = "vol-00000000000000015"
    snapshot_id = "snap-00000000000000016"
    root_volume_arn = f"arn:aws:ec2:{region}:{account}:volume/{root_volume_id}"
    snapshot_arn = f"arn:aws:ec2:{region}:{account}:snapshot/{snapshot_id}"
    action_resources = {
        "ec2:CreateSnapshot": ec2["volume"],
        "ec2:DeleteNatGateway": ec2["natgateway"],
        "ec2:DeleteNetworkInterface": ec2["network-interface"],
        "ec2:DeleteRoute": ec2["route-table"],
        "ec2:DeleteRouteTable": ec2["route-table"],
        "ec2:DisassociateRouteTable": [
            *ec2["route-table"],
            *ec2["subnet"],
        ],
        "ec2:RevokeSecurityGroupEgress": ec2["security-group"],
        "ec2:RevokeSecurityGroupIngress": ec2["security-group"],
        "ec2:DeleteSecurityGroup": ec2["security-group"],
        "ec2:DeleteSubnet": ec2["subnet"],
        "ec2:DeleteVolume": [*ec2["volume"], root_volume_arn],
        "ec2:DetachVolume": [*ec2["instance"], *ec2["volume"]],
        "ec2:DeleteVpcEndpoints": ec2["vpc-endpoint"],
        "ec2:ReleaseAddress": ec2["elastic-ip"],
        "ec2:TerminateInstances": ec2["instance"],
        "lambda:InvokeFunction": [versions[0]],
        "lambda:DeleteFunction": [*functions, *versions],
        "lambda:DeleteFunctionEventInvokeConfig": functions,
        "lambda:RemovePermission": [*functions, *versions],
        "iam:DeleteRolePolicy": roles,
        "iam:DeleteRole": roles,
        "iam:RemoveRoleFromInstanceProfile": [*roles, profile],
        "iam:DeleteInstanceProfile": [profile],
        "s3:ListBucket": [bucket],
        "s3:DeleteObject": [f"{bucket}/*"],
        "s3:DeleteObjectVersion": [f"{bucket}/*"],
        "s3:GetBucketPolicy": [bucket],
        "s3:DeleteBucketPolicy": [bucket],
        "s3:DeleteBucket": [bucket],
        "logs:DeleteLogGroup": logs,
        "cloudwatch:DeleteAlarms": alarms,
        "sqs:DeleteQueue": [queue],
        "secretsmanager:DeleteResourcePolicy": secret_arns,
        "secretsmanager:DeleteSecret": secret_arns,
        "states:DeleteStateMachine": [state_machine],
        "states:DeleteStateMachineVersion": [f"{state_machine}:1"],
        "scheduler:DeleteSchedule": schedules,
        "events:RemoveTargets": [event_rule],
        "events:DeleteRule": [event_rule],
        "kms:CreateGrant": [kms],
        "kms:RetireGrant": [kms],
        "kms:RevokeGrant": [kms],
        "ec2:DescribeAddresses": ["*"],
        "ec2:DescribeInstances": ["*"],
        "ec2:DescribeNatGateways": ["*"],
        "ec2:DescribeNetworkInterfaces": ["*"],
        "ec2:DescribeRouteTables": ["*"],
        "ec2:DescribeSecurityGroups": ["*"],
        "ec2:DescribeSnapshots": ["*"],
        "ec2:DescribeSubnets": ["*"],
        "ec2:DescribeVolumes": ["*"],
        "ec2:DescribeVpcEndpoints": ["*"],
        "lambda:GetFunction": [*functions, *versions],
        "lambda:ListVersionsByFunction": functions,
        "iam:GetRole": roles,
        "iam:GetInstanceProfile": [profile],
        "iam:ListRolePolicies": roles,
        "s3:GetBucketLocation": [bucket],
        "s3:ListBucketVersions": [bucket],
        "logs:DescribeLogGroups": ["*"],
        "cloudwatch:DescribeAlarms": ["*"],
        "sqs:GetQueueAttributes": [queue],
        "secretsmanager:DescribeSecret": secret_arns,
        "states:DescribeStateMachine": [state_machine, f"{state_machine}:1"],
        "scheduler:GetSchedule": schedules,
        "events:DescribeRule": [event_rule],
        "events:ListTargetsByRule": [event_rule],
    }
    readback_resources = [
        {
            "LogicalResourceId": "CombinedHostRootVolume",
            "ResourceType": "AWS::EC2::Volume",
            "PhysicalResourceId": root_volume_id,
            "ResourceArn": root_volume_arn,
            "SourceApi": "DescribeInstances",
        },
        {
            "LogicalResourceId": "CombinedHostDataVolume",
            "ResourceType": "AWS::EC2::Volume",
            "PhysicalResourceId": ec2_ids["volume"][0],
            "ResourceArn": ec2["volume"][0],
            "SourceApi": "ListStackResources",
        },
        {
            "LogicalResourceId": "CombinedHostDataVolumeDeletionSnapshot",
            "ResourceType": "AWS::EC2::Snapshot",
            "PhysicalResourceId": snapshot_id,
            "ResourceArn": snapshot_arn,
            "SourceApi": "DescribeSnapshots",
        },
        {
            "LogicalResourceId": "NatGateway",
            "ResourceType": "AWS::EC2::NatGateway",
            "PhysicalResourceId": ec2_ids["natgateway"][0],
            "ResourceArn": ec2["natgateway"][0],
            "SourceApi": "ListStackResources",
        },
        *[
            {
                "LogicalResourceId": logical_id,
                "ResourceType": "AWS::SecretsManager::Secret",
                "PhysicalResourceId": secret_arn,
                "ResourceArn": secret_arn,
                "SourceApi": "ListStackResources",
            }
            for logical_id, secret_arn in zip(
                (
                    "RawSkyTokenSecret",
                    "SkyBootstrapSecret",
                    "AttestationClientTlsSecret",
                    "LaunchAdmissionClientTlsSecret",
                    "NumericBindingClientTlsSecret",
                    "RetainedCancellationClientTlsSecret",
                    "CombinedHostTlsSecret",
                    "CaIssuanceSecret",
                ),
                secret_arns,
            )
        ],
        {
            "LogicalResourceId": "TlsHandlerVersion",
            "ResourceType": "AWS::Lambda::Version",
            "PhysicalResourceId": versions[0],
            "ResourceArn": versions[0],
            "SourceApi": "ListStackResources",
        },
    ]
    resource_arns = sorted(
        {
            resource
            for resources in action_resources.values()
            for resource in resources
            if resource != "*"
        }
    )
    resource_ids = [
        resource_id for resource_ids in ec2_ids.values() for resource_id in resource_ids
    ] + [root_volume_id, snapshot_id]
    resource_names = sorted(
        {
            *function_names,
            *role_names,
            profile_name,
            f"keep-glm52-h1g-rehearsal-{account}-{region}",
            f"keep-glm52-h1g-support-dlq-{activation}",
        }
    )
    deletion_inventory_projection = {
        "resource_arns_sha256": hashlib.sha256(
            json.dumps(
                resource_arns,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
        "resource_ids_sha256": hashlib.sha256(
            json.dumps(
                resource_ids,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
        "resource_names_sha256": hashlib.sha256(
            json.dumps(
                resource_names,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
        "action_resources_sha256": hashlib.sha256(
            json.dumps(
                action_resources,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
    }
    readback_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_resource_readback_v1",
        "activation_id": activation,
        "support_stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "stack/keep-glm52-h1g-support/"
            "11111111-2222-4333-8444-555555555555"
        ),
        "observed_at": "2026-07-28T00:05:00Z",
        "resources": readback_resources,
        "deletion_inventory_projection": deletion_inventory_projection,
    }
    readback_body["canonical_body_sha256"] = hashlib.sha256(
        json.dumps(
            readback_body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    readback_evidence: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_resource_readback_evidence_v1",
        "support_stack_id": readback_body["support_stack_id"],
        "list_stack_resources_request_id": ("11111111-2222-4333-8444-555555555555"),
        "list_stack_resources_response_sha256": "a" * 64,
        "describe_instances_request_id": ("22222222-3333-4444-8555-666666666666"),
        "describe_instances_response_sha256": "b" * 64,
        "describe_snapshots_request_id": ("33333333-4444-4555-8666-777777777777"),
        "describe_snapshots_response_sha256": "c" * 64,
        "observed_at": "2026-07-28T00:05:00Z",
        "resource_readback_identity_sha256": (readback_body["canonical_body_sha256"]),
    }
    readback_evidence["canonical_body_sha256"] = hashlib.sha256(
        json.dumps(
            readback_evidence,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_deletion_inventory_v1",
        "activation_id": activation,
        "support_stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "stack/keep-glm52-h1g-support/"
            "11111111-2222-4333-8444-555555555555"
        ),
        "resource_arns": resource_arns,
        "resource_ids": resource_ids,
        "resource_names": resource_names,
        "action_resources": action_resources,
        "stack_resource_readback": readback_body,
        "stack_resource_readback_evidence": readback_evidence,
        "custom_resource_delete_callbacks": [
            {
                "LogicalResourceId": "SkyBootstrapCustomResource",
                "HandlerVersionArn": versions[0],
            },
            {
                "LogicalResourceId": "TlsBundleCustomResource",
                "HandlerVersionArn": versions[0],
            },
        ],
    }
    body["canonical_body_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return body


def _rehash_support_deletion_inventory(
    value: dict[str, object],
) -> dict[str, object]:
    readback = value["stack_resource_readback"]
    assert isinstance(readback, dict)
    projection = readback["deletion_inventory_projection"]
    assert isinstance(projection, dict)
    for field, source in (
        ("resource_arns_sha256", value["resource_arns"]),
        ("resource_ids_sha256", value["resource_ids"]),
        ("resource_names_sha256", value["resource_names"]),
        ("action_resources_sha256", value["action_resources"]),
    ):
        projection[field] = hashlib.sha256(
            json.dumps(
                source,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
    readback_body = dict(readback)
    readback_body.pop("canonical_body_sha256", None)
    readback["canonical_body_sha256"] = hashlib.sha256(
        json.dumps(
            readback_body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    evidence = value["stack_resource_readback_evidence"]
    assert isinstance(evidence, dict)
    evidence["resource_readback_identity_sha256"] = readback["canonical_body_sha256"]
    evidence_body = dict(evidence)
    evidence_body.pop("canonical_body_sha256", None)
    evidence["canonical_body_sha256"] = hashlib.sha256(
        json.dumps(
            evidence_body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    body = dict(value)
    body.pop("canonical_body_sha256", None)
    value["canonical_body_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return value


_FENCE_SLOTS = (
    "PREPARE_GENESIS_LIVE_STATE",
    "BATCH_FIVE_SOURCE_ACTIVATION",
    "RESERVATION_ONLY",
    "CLOSED_SOURCE",
    "TERMINAL",
)


def _fence_change_set_name(slot: str) -> str:
    body = {
        "run_id": "glm52-sky-20260724",
        "activation_id": "act-20260728-0001",
        "generation": 1,
        "slot": slot,
    }
    digest = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()[:16]
    return f"glm52-h1g-00000001-{slot.lower().replace('_', '-')}-{digest}"


def _fence_template_inventory() -> list[dict[str, object]]:
    base = (
        "campaigns/glm52-sky-20260724/authorities/fence/templates/"
        "act-20260728-0001/00000001"
    )
    bucket = "keep-glm52-model-evidence-fixture"
    return [
        {
            "slot": slot,
            "template_key": f"{base}/{slot}.json",
            "template_url": (
                f"https://{bucket}.s3.us-west-2.amazonaws.com/"
                f"{base}/{slot}.json?versionId=fence-template-v{index:04d}"
            ),
            "version_id": f"fence-template-v{index:04d}",
            "file_sha256": f"{index + 5:x}" * 64,
            "body_sha256": f"{index + 6:x}" * 64,
            "policy_sha256": f"{index + 7:x}" * 64,
            "change_set_name": _fence_change_set_name(slot),
            "change_set_arn": (
                "arn:aws:cloudformation:us-west-2:246813579024:"
                f"changeSet/{_fence_change_set_name(slot)}/"
                f"00000000-0000-4000-8000-{index:012d}"
            ),
        }
        for index, slot in enumerate(_FENCE_SLOTS, start=1)
    ]


def _fence_manifest_coordinate() -> dict[str, object]:
    value = {
        "input_kind": "FENCE_EXECUTION_REQUEST",
        "bucket": "keep-glm52-model-evidence-fixture",
        "key": (
            "campaigns/glm52-sky-20260724/authorities/fence/manifests/"
            "act-20260728-0001/00000001/FENCE_TEMPLATE_MANIFEST.json"
        ),
        "version_id": "fence-manifest-v0001",
        "file_sha256": "4" * 64,
        "body_sha256": "5" * 64,
    }
    return {
        **value,
        "canonical_identity_sha256": hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
    }


_TASK11_WRITER_IDS = (
    "SourceGpuSpend",
    "SourceSubmissionIntent",
    "SourceControllerBaseline",
    "SourceControlPlaneReadiness",
    "SourceSubmissionAcquisition",
    "FenceSuccessor",
    "ClaimWriter",
    "DecisionWriter",
    "TerminalV1Writer",
    "ClosureHandoff",
)


def _task11_writer_bindings() -> list[dict[str, object]]:
    sha = {index: f"{index:x}" * 64 for index in range(1, 7)}
    outputs = (
        f"campaigns/glm52-sky-20260724/spend-snapshots/{sha[1]}/"
        "GPU_SPEND_SNAPSHOT.json",
        "campaigns/glm52-sky-20260724/submissions/production/intents/"
        f"{sha[2]}/SKYPILOT_SUBMISSION_INTENT.json",
        "campaigns/glm52-sky-20260724/production/controller-baselines/"
        f"{sha[3]}/CONTROLLER_BASELINE.json",
        "campaigns/glm52-sky-20260724/monitor/must-start/production/"
        f"{sha[4]}/control-plane-ready/{sha[5]}/CONTROL_PLANE_READY.json",
        "campaigns/glm52-sky-20260724/submissions/production/acquisitions/"
        f"{sha[6]}/SUBMISSION_ACQUIRED.json",
        "campaigns/glm52-sky-20260724/authorities/fence/successors/"
        f"{sha[1]}/FENCE_SUCCESSOR.json",
        "campaigns/glm52-sky-20260724/submissions/production/generations/"
        "00000001/GENERATION_CLAIM.json",
        "campaigns/glm52-sky-20260724/submissions/production/generations/"
        "00000001/START_DECISION.json",
        "campaigns/glm52-sky-20260724/submissions/production/generations/"
        "00000001/GENERATION_TERMINAL.json",
        "campaigns/glm52-sky-20260724/submissions/production/generations/"
        "00000001/handoff/SKY_POST_HANDOFF.json",
    )
    manifest_key = _fence_manifest_coordinate()["key"]
    result = []
    for index, (role_id, output_key) in enumerate(
        zip(_TASK11_WRITER_IDS, outputs, strict=True)
    ):
        input_key = (
            "campaigns/glm52-sky-20260724/authorities/task11/"
            f"act-20260728-0001/00000001/{index:02d}-{role_id}/REQUEST.json"
        )
        result.append(
            {
                "role_id": role_id,
                "activation_id": "act-20260728-0001",
                "generation": 1,
                "index": index,
                "action_key": (
                    f"ACTIVATION#act-20260728-0001#TASK11#{index:02d}#{role_id}"
                ),
                "input_key": input_key,
                "output_key": output_key,
                "read_keys": [input_key, manifest_key],
                "ledger_leading_keys": ["RUN#glm52-sky-20260724"],
            }
        )
    return result


def _support_inputs() -> dict[str, object]:
    user_data = "#!/bin/sh\nset -eu\nexit 64\n"
    retained_function_version_bindings = [
        {
            "output_key": "Task12TerminalV2VersionArn",
            "export_name": "KeepGlm52Task12TerminalV2VersionArn",
            "version_arn": (
                "arn:aws:lambda:us-west-2:246813579024:function:"
                "keep-glm52-h1g-terminal-v2-writer:7"
            ),
            "function_name": "keep-glm52-h1g-terminal-v2-writer",
            "version": "7",
            "code_sha256": "3" * 64,
        },
        {
            "output_key": "Task12WorkerDrainVersionArn",
            "export_name": "KeepGlm52Task12WorkerDrainVersionArn",
            "version_arn": (
                "arn:aws:lambda:us-west-2:246813579024:function:"
                "keep-glm52-h1g-worker-drain-signal:7"
            ),
            "function_name": "keep-glm52-h1g-worker-drain-signal",
            "version": "7",
            "code_sha256": "3" * 64,
        },
        {
            "output_key": "Task9LiabilityWatcherVersionArn",
            "export_name": "KeepGlm52Task9LiabilityWatcherVersionArn",
            "version_arn": (
                "arn:aws:lambda:us-west-2:246813579024:function:"
                "keep-glm52-h1g-worker-launch-custody:7"
            ),
            "function_name": "keep-glm52-h1g-worker-launch-custody",
            "version": "7",
            "code_sha256": "3" * 64,
        },
    ]
    return {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_inputs_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "activation_id": "act-20260728-0001",
        "nat_gateway_id": "nat-00000000000000003",
        "support_stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "stack/keep-glm52-h1g-support/"
            "11111111-2222-4333-8444-555555555555"
        ),
        "fence_stack_name": "keep-glm52-h1g-fence",
        "fence_stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "stack/keep-glm52-h1g-fence/"
            "22222222-3333-4444-8555-666666666666"
        ),
        "fence_service_role_arn": (
            "arn:aws:iam::246813579024:role/keep-glm52-h1g-fence-service"
        ),
        "fence_manifest_coordinate": _fence_manifest_coordinate(),
        "fence_template_inventory": _fence_template_inventory(),
        "task11_writer_bindings": _task11_writer_bindings(),
        "organizations_id": "o-08ddnqdzd3",
        "retained_stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "stack/keep-glm52-gpu/"
            "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        ),
        "retained_vpc_id": "vpc-0123456789abcdef0",
        "retained_vpc_cidr": "10.20.0.0/16",
        "existing_subnet_cidrs": [
            "10.20.0.0/24",
            "10.20.1.0/24",
            "10.20.2.0/24",
        ],
        "existing_secondary_cidrs": ["10.30.0.0/16"],
        "primary_az": "us-west-2a",
        "alternate_az": "us-west-2b",
        "primary_public_subnet_id": "subnet-0123456789abcdef0",
        "retained_public_s3_endpoint_id": "vpce-0123456789abcdef0",
        "retained_kms_key_arn": (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "12345678-1234-4234-8234-1234567890ab"
        ),
        "retained_kms_key_id": "12345678-1234-4234-8234-1234567890ab",
        "ledger_table_name": "keep-glm52-h1g-ledger-v1",
        "ledger_table_arn": (
            "arn:aws:dynamodb:us-west-2:246813579024:table/keep-glm52-h1g-ledger-v1"
        ),
        "model_bucket_name": "keep-glm52-model-evidence-fixture",
        "model_bucket_arn": "arn:aws:s3:::keep-glm52-model-evidence-fixture",
        "model_prefix": "campaigns/glm52-sky-20260724/",
        "host_ami_id": "ami-0123456789abcdef0",
        "host_private_ip": "10.20.101.10",
        "host_user_data": user_data,
        "host_user_data_sha256": hashlib.sha256(user_data.encode("utf-8")).hexdigest(),
        "host_boot_identity_sha256": "1" * 64,
        "root_volume_gib": 30,
        "root_volume_type": "gp3",
        "root_volume_iops": 3000,
        "root_volume_throughput_mibps": 125,
        "cryptography_layer_arn": (
            "arn:aws:lambda:us-west-2:246813579024:layer/h1g-cryptography:7"
        ),
        "cryptography_layer_sha256": "2" * 64,
        "lambda_code_bucket": "keep-glm52-code-fixture",
        "lambda_code_key": "h1g/support/support-v1.zip",
        "lambda_code_version_id": "fixture-version-0001",
        "lambda_code_sha256": "3" * 64,
        "attestation_port": 18443,
        "launch_admission_port": 18444,
        "numeric_binding_port": 18445,
        "retained_cancellation_port": 18446,
        "activation_started_at": "2026-07-28T00:00:00Z",
        "runtime_credential_cutoff_at": "2026-07-28T00:15:00Z",
        "support_deletion_inventory": _support_deletion_inventory(),
        "price_card_identity_sha256": _mechanical_price_card()[
            "price_card_identity_sha256"
        ],
        "retained_function_version_bindings": (retained_function_version_bindings),
        "retained_export_names": [
            "KeepGlm52VpcId",
            "KeepGlm52PrimaryPublicSubnetId",
            "KeepGlm52CampaignKmsKeyArn",
            "KeepGlm52H1gLedgerArn",
            "KeepGlm52Task12TerminalV2VersionArn",
            "KeepGlm52Task12WorkerDrainVersionArn",
            "KeepGlm52Task9LiabilityWatcherVersionArn",
            "keep-glm52-gpu-ModelBucket",
            "keep-glm52-gpu-LaunchTemplate",
        ],
    }


def _bootstrap_fence_entries_v2() -> list[dict[str, object]]:
    from glm52_enforcement.canonical import canonical_json_bytes
    from glm52_enforcement.fence_artifacts import (
        FenceSlot,
        build_fence_entry,
    )
    from glm52_enforcement.fence_policy_renderer import (
        PolicyLedgerSummary,
        RenderedPolicy,
    )

    policy_bytes = canonical_json_bytes({"Version": "2012-10-17", "Statement": []})
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    component_names = (
        "global_guards",
        "legacy_fragment",
        "permanent_lineage",
        "source_family_closures",
        "active_writer_cohort",
        "object_reader_guards",
        "bucket_list_reader_guards",
        "terminal_closure",
    )
    component_bytes = dict.fromkeys(component_names, 0)
    component_bytes["global_guards"] = len(policy_bytes)
    component_statement_counts = dict.fromkeys(component_names, 0)
    rendered = RenderedPolicy(
        policy_bytes=policy_bytes,
        policy_sha256=policy_sha256,
        rendered_policy_bytes=len(policy_bytes),
        statement_ledger=(),
        component_bytes=component_bytes,
        ledger_summary=PolicyLedgerSummary(
            policy_prefix_bytes=14,
            policy_suffix_bytes=25,
            statement_count=0,
            max_policy_bytes=18_432,
            working_limit_bytes=17_920,
            reserved_design_headroom_bytes=512,
            working_headroom_bytes=17_920 - len(policy_bytes),
            design_headroom_bytes=18_432 - len(policy_bytes),
            s3_headroom_bytes=20_480 - len(policy_bytes),
            component_statement_counts=component_statement_counts,
        ),
    )
    stack_id = (
        "arn:aws:cloudformation:us-west-2:246813579024:"
        "stack/keep-glm52-h1g-fence/"
        "00000000-0000-4000-8000-000000000001"
    )
    migration_role = (
        "arn:aws:iam::246813579024:role/keep-glm52-h1g-cloudformation-deployment"
    )
    fence_role = "arn:aws:iam::246813579024:role/keep-glm52-h1g-fence-service"
    seed_sha256 = policy_sha256
    entries = []
    for index, slot in enumerate(
        (
            FenceSlot.PREPARE_GENESIS_LIVE_STATE,
            FenceSlot.RESERVATION_ONLY,
            FenceSlot.CLOSED_SOURCE,
            FenceSlot.SOURCE_FAMILIES_FROZEN,
        ),
        start=1,
    ):
        entry = build_fence_entry(
            slot=slot,
            rendered=rendered,
            version_id=f"bootstrap-template-version-{index}",
            render_input_identity_sha256=f"{index:x}" * 64,
            stack_id=stack_id,
            migration_service_role_arn=migration_role,
            fence_service_role_arn=fence_role,
            bridge_seed_policy_sha256=seed_sha256,
            expected_prestate_policy_sha256=seed_sha256,
            publisher_deny_policy_sha256="e" * 64,
            batch_projection_contract=None,
            successor_contract_sha256=(
                None if slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE else "f" * 64
            ),
        )
        entries.append(entry.to_dict())
    return entries


def _support_build_inputs():
    from glm52_enforcement.task13_support_input_materialization import (
        SupportBuildInputs,
    )

    value = _support_inputs()
    for field in (
        "nat_gateway_id",
        "support_stack_id",
        "support_deletion_inventory",
    ):
        value.pop(field)
    value.pop("fence_manifest_coordinate")
    value["activation_id"] = "glm52-v2-amber-quartz"
    value["fence_stack_id"] = (
        "arn:aws:cloudformation:us-west-2:246813579024:"
        "stack/keep-glm52-h1g-fence/"
        "00000000-0000-4000-8000-000000000001"
    )
    manifest_coordinate = _bootstrap_manifest_coordinate_v2()
    retained_entries = _bootstrap_fence_entries_v2()
    prepare = retained_entries[0]
    value["organizations_id"] = "o-08ddnqdzd3"
    (
        value["attestation_port"],
        value["launch_admission_port"],
        value["numeric_binding_port"],
        value["retained_cancellation_port"],
    ) = (9443, 9444, 9445, 9446)
    value["model_bucket_name"] = "keep-glm52-models-246813579024-us-west-2"
    value["model_bucket_arn"] = "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2"
    value["retained_export_names"] = sorted(value["retained_export_names"])
    value.update(
        {
            "schema_version": 2,
            "record_type": "glm52_h1g_support_build_inputs_v2",
            "bootstrap_manifest_coordinate": manifest_coordinate,
            "prepare_entry_identity_sha256": prepare["entry_identity_sha256"],
            "prepare_template_body_sha256": prepare["template_body_sha256"],
            "prepare_policy_sha256": prepare["policy_sha256"],
            "fence_template_inventory": tuple(retained_entries),
        }
    )
    return SupportBuildInputs(**value)


def _support_build_input_mapping() -> dict[str, object]:
    from glm52_enforcement.support_plane import (
        support_build_inputs_projection,
    )

    return dict(support_build_inputs_projection(_support_build_inputs()))


def _pre_support_runtime_input_mapping() -> dict[str, object]:
    value = _support_inputs()
    absence_evidence: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_future_support_version_absence_v1",
        "account_id": value["account_id"],
        "region": value["region"],
        "support_stack_name": "keep-glm52-h1g-support",
        "observed_at": "2026-07-28T00:00:00Z",
        "lambda_function_names": [
            "keep-glm52-h1g-numeric-binding",
            "keep-glm52-h1g-retained-cancellation",
        ],
        "state_machine_names": [
            "keep-glm52-h1g-retained-lifecycle",
        ],
        "lambda_get_function_absent": True,
        "states_describe_state_machine_absent": True,
        "lambda_readback_sha256": "5" * 64,
        "states_readback_sha256": "6" * 64,
    }
    absence_evidence["canonical_body_sha256"] = hashlib.sha256(
        json.dumps(
            absence_evidence,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": 1,
        "record_type": "glm52_h1g_pre_support_runtime_inputs_v1",
        "account_id": value["account_id"],
        "region": value["region"],
        "run_id": value["run_id"],
        "activation_id": value["activation_id"],
        "retained_stack_id": value["retained_stack_id"],
        "retained_foundation_readback_sha256": "4" * 64,
        "fence_manifest_coordinate": value["fence_manifest_coordinate"],
        "future_support_version_absence": absence_evidence,
        "retained_kms_key_arn": value["retained_kms_key_arn"],
        "ledger_table_arn": value["ledger_table_arn"],
        "model_bucket_name": value["model_bucket_name"],
        "model_bucket_arn": value["model_bucket_arn"],
        "lambda_code_bucket": value["lambda_code_bucket"],
        "lambda_code_key": value["lambda_code_key"],
        "lambda_code_version_id": value["lambda_code_version_id"],
        "lambda_code_sha256": value["lambda_code_sha256"],
        "retained_export_names": [
            name
            for name in value["retained_export_names"]
            if name
            not in {
                "KeepGlm52Task12TerminalV2VersionArn",
                "KeepGlm52Task12WorkerDrainVersionArn",
                "KeepGlm52Task9LiabilityWatcherVersionArn",
            }
        ],
    }


def test_pre_support_runtime_inputs_materialize_before_terminal_export() -> None:
    """Break caught: Task10/12 retained runtime depends on its own TerminalV2 export."""

    from glm52_enforcement.support_plane import (
        build_pre_support_retained_runtime_fragment,
        pre_support_runtime_inputs_from_mapping,
        pre_support_runtime_inputs_projection,
    )

    value = _pre_support_runtime_input_mapping()
    inputs = pre_support_runtime_inputs_from_mapping(value)
    fragment = build_pre_support_retained_runtime_fragment(inputs)

    assert pre_support_runtime_inputs_projection(inputs) == value
    assert "KeepGlm52Task12TerminalV2VersionArn" not in inputs.retained_export_names
    assert "KeepGlm52Task12WorkerDrainVersionArn" not in inputs.retained_export_names
    assert (
        "KeepGlm52Task9LiabilityWatcherVersionArn" not in inputs.retained_export_names
    )
    assert fragment["Outputs"]["Task12TerminalV2VersionArn"]["Export"] == {
        "Name": "KeepGlm52Task12TerminalV2VersionArn"
    }
    assert fragment["Outputs"]["Task12WorkerDrainVersionArn"]["Export"] == {
        "Name": "KeepGlm52Task12WorkerDrainVersionArn"
    }
    assert fragment["Outputs"]["Task9LiabilityWatcherVersionArn"]["Export"] == {
        "Name": "KeepGlm52Task9LiabilityWatcherVersionArn"
    }
    assert len(fragment["Resources"]) == 74
    known = set(fragment["Resources"]) | set(fragment.get("Parameters", {}))
    dangling: list[str] = []

    def collect_dangling(value: object) -> None:
        if isinstance(value, dict):
            if (
                set(value) == {"Ref"}
                and isinstance(value["Ref"], str)
                and not value["Ref"].startswith("AWS::")
                and value["Ref"] not in known
            ):
                dangling.append(value["Ref"])
            for item in value.values():
                collect_dangling(item)
        elif isinstance(value, list):
            for item in value:
                collect_dangling(item)

    collect_dangling(fragment)
    assert dangling == []

    missing = _pre_support_runtime_input_mapping()
    evidence = dict(missing["future_support_version_absence"])
    evidence["lambda_get_function_absent"] = False
    evidence.pop("canonical_body_sha256")
    evidence["canonical_body_sha256"] = hashlib.sha256(
        json.dumps(
            evidence,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    missing["future_support_version_absence"] = evidence
    with pytest.raises(ValueError, match="absence"):
        pre_support_runtime_inputs_from_mapping(missing)


def test_pre_support_retained_fragment_has_only_task10_task12_runtime() -> None:
    """Break caught: retained runtime cannot exist until support physical IDs do."""

    from glm52_enforcement.canonical import canonical_json_bytes
    from glm52_enforcement.support_plane import (
        _build_task10_retained_fragment,
        _build_task12_retained_fragment,
        build_pre_support_retained_runtime_fragment,
        pre_support_runtime_inputs_from_mapping,
    )

    inputs = pre_support_runtime_inputs_from_mapping(
        _pre_support_runtime_input_mapping()
    )
    fragment = build_pre_support_retained_runtime_fragment(inputs)
    task12 = _build_task12_retained_fragment(inputs)
    task10 = _build_task10_retained_fragment(inputs)

    assert set(fragment["Resources"]) == {
        *task12["Resources"],
        *task10["Resources"],
    }
    assert fragment["Outputs"] == {
        "Task12TerminalV2VersionArn": {
            "Value": {"Ref": "TerminalV2Version"},
            "Export": {"Name": "KeepGlm52Task12TerminalV2VersionArn"},
        },
        "Task12WorkerDrainVersionArn": {
            "Value": {"Ref": "WorkerDrainVersion"},
            "Export": {"Name": "KeepGlm52Task12WorkerDrainVersionArn"},
        },
        "Task9LiabilityWatcherVersionArn": {
            "Value": {"Ref": "Task9LiabilityWatcherFunctionVersion"},
            "Export": {"Name": "KeepGlm52Task9LiabilityWatcherVersionArn"},
        },
    }
    assert "Parameters" not in fragment
    assert "Parameters" not in task12
    retained_substitutions = fragment["Resources"]["RetainedLifecycleStateMachine"][
        "Properties"
    ]["DefinitionSubstitutions"]
    snapshot_substitutions = fragment["Resources"]["SnapshotCleanupStateMachine"][
        "Properties"
    ]["DefinitionSubstitutions"]
    for substitutions in (
        retained_substitutions,
        snapshot_substitutions,
    ):
        assert substitutions["Task12ActivationOrdinal"] == "1"
        assert substitutions["Task12Generation"] == "1"
        assert substitutions["Task12GenerationText"] == "00000001"
        assert re.fullmatch(
            r"[0-9a-f]{64}",
            substitutions["Task12DispatchIdentitySha256"],
        )
    raw = canonical_json_bytes(fragment)
    for forbidden in (
        b"SupportStackId",
        b"NatGatewayId",
        b"SupportDeletionInventorySha256",
        b"SupportResourceReadbackSha256",
        b"GLM52_SUPPORT_HOST_INSTANCE_ID",
        b"GLM52_SUPPORT_HOST_SECURITY_GROUP_ID",
        _support_inputs()["host_ami_id"].encode("ascii"),
        _support_inputs()["host_private_ip"].encode("ascii"),
    ):
        assert forbidden not in raw


def test_postcreate_retained_fragment_is_physical_support_only() -> None:
    """Break caught: postcreate duplicates the already-deployable Task10/12 graph."""

    from glm52_enforcement.canonical import canonical_json_bytes
    from glm52_enforcement.support_plane import (
        build_postcreate_retained_support_fragment,
        build_pre_support_retained_runtime_fragment,
        support_inputs_from_mapping,
    )

    inputs = support_inputs_from_mapping(_support_inputs())
    pre_support = build_pre_support_retained_runtime_fragment(_support_build_inputs())
    postcreate = build_postcreate_retained_support_fragment(inputs)

    assert set(pre_support["Resources"]).isdisjoint(postcreate["Resources"])
    assert "TerminalV2Version" not in postcreate["Resources"]
    assert "Task10ProductionStateMachine" not in postcreate["Resources"]
    assert "Parameters" not in postcreate
    assert "Outputs" not in postcreate
    assert "task12_retained_ownership" not in postcreate["Metadata"]
    assert "Task10ProductionWorkflow" not in postcreate["Metadata"]
    raw = canonical_json_bytes(postcreate)
    assert inputs.support_stack_id.encode("ascii") in raw
    assert inputs.nat_gateway_id.encode("ascii") in raw
    assert b"i-00000000000000001" in raw
    assert b"sg-0000000000000000b" in raw
    assert b"SupportDeletionInventorySha256" in raw


def test_split_fragments_compose_v2_fence_over_legacy_augmentation() -> None:
    """Break caught: retained fence composition drops authenticated legacy runtime."""

    from glm52_enforcement.canonical import canonical_json_bytes
    from glm52_enforcement.support_plane import (
        _build_retained_augmentation,
        build_postcreate_retained_support_fragment,
        build_pre_support_retained_runtime_fragment,
        compose_complete_retained_template,
        support_inputs_from_mapping,
    )
    from glm52_enforcement.task13_support_input_materialization import (
        support_build_inputs_projection,
    )

    build_inputs = _support_build_inputs()
    build_projection = dict(support_build_inputs_projection(build_inputs))
    legacy_value = _support_inputs()
    legacy_activation_id = legacy_value["activation_id"]
    build_activation_id = build_projection["activation_id"]
    legacy_bucket = legacy_value["model_bucket_name"]
    build_bucket = build_projection["model_bucket_name"]
    legacy_closure_role = (
        "keep-glm52-h1g-closure-session-"
        + hashlib.sha256(legacy_activation_id.encode("ascii")).hexdigest()[:16]
    )
    build_closure_role = (
        "keep-glm52-h1g-closure-session-"
        + hashlib.sha256(build_activation_id.encode("ascii")).hexdigest()[:16]
    )
    for field in (
        "fence_manifest_coordinate",
        "fence_template_inventory",
        "task11_writer_bindings",
        "support_deletion_inventory",
    ):
        legacy_value[field] = json.loads(
            json.dumps(legacy_value[field])
            .replace(legacy_activation_id, build_activation_id)
            .replace(legacy_bucket, build_bucket)
            .replace(legacy_closure_role, build_closure_role)
        )
    for item in legacy_value["fence_template_inventory"]:
        old_change_set_name = item["change_set_name"]
        digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "run_id": build_projection["run_id"],
                    "activation_id": build_activation_id,
                    "generation": 1,
                    "slot": item["slot"],
                }
            )
        ).hexdigest()[:16]
        change_set_name = (
            "glm52-h1g-00000001-"
            + item["slot"].lower().replace("_", "-")
            + "-"
            + digest
        )
        item["change_set_name"] = change_set_name
        item["change_set_arn"] = item["change_set_arn"].replace(
            f"changeSet/{old_change_set_name}/",
            f"changeSet/{change_set_name}/",
        )
    coordinate = legacy_value["fence_manifest_coordinate"]
    coordinate["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {
                field: coordinate[field]
                for field in (
                    "input_kind",
                    "bucket",
                    "key",
                    "version_id",
                    "file_sha256",
                    "body_sha256",
                )
            }
        )
    ).hexdigest()
    readback = legacy_value["support_deletion_inventory"]["stack_resource_readback"]
    readback["deletion_inventory_projection"] = {
        field + "_sha256": hashlib.sha256(
            canonical_json_bytes(legacy_value["support_deletion_inventory"][field])
        ).hexdigest()
        for field in (
            "resource_arns",
            "resource_ids",
            "resource_names",
            "action_resources",
        )
    }
    readback_body = dict(readback)
    readback_body.pop("canonical_body_sha256")
    readback["canonical_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(readback_body)
    ).hexdigest()
    readback_evidence = legacy_value["support_deletion_inventory"][
        "stack_resource_readback_evidence"
    ]
    readback_evidence["resource_readback_identity_sha256"] = readback[
        "canonical_body_sha256"
    ]
    readback_evidence_body = dict(readback_evidence)
    readback_evidence_body.pop("canonical_body_sha256")
    readback_evidence["canonical_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(readback_evidence_body)
    ).hexdigest()
    deletion_inventory = legacy_value["support_deletion_inventory"]
    deletion_body = dict(deletion_inventory)
    deletion_body.pop("canonical_body_sha256")
    deletion_inventory["canonical_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(deletion_body)
    ).hexdigest()
    for field in set(legacy_value) & set(build_projection) - {
        "schema_version",
        "record_type",
        "fence_template_inventory",
        "task11_writer_bindings",
        "organizations_id",
        "attestation_port",
        "launch_admission_port",
        "retained_kms_key_arn",
        "retained_kms_key_id",
        "numeric_binding_port",
        "support_deletion_inventory",
        "pinned_relay_ports",
        "retained_cancellation_port",
    }:
        legacy_value[field] = build_projection[field]
    inputs = support_inputs_from_mapping(legacy_value)
    base = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": ("GLM-5.2 retained support lifecycle augmentation fragment"),
        "Parameters": {},
        "Outputs": {},
        "Metadata": {},
        "Resources": {},
    }
    composed = compose_complete_retained_template(
        base,
        build_pre_support_retained_runtime_fragment(build_inputs),
        build_postcreate_retained_support_fragment(inputs),
    )

    legacy = _build_retained_augmentation(inputs)
    for logical_id, resource in legacy["Resources"].items():
        expected = copy.deepcopy(resource)
        if logical_id in {
            "RetainedLifecycleStateMachine",
            "SnapshotCleanupStateMachine",
        }:
            expected["Properties"]["DefinitionSubstitutions"][
                "Task12DispatchIdentitySha256"
            ] = composed["Resources"][logical_id]["Properties"][
                "DefinitionSubstitutions"
            ]["Task12DispatchIdentitySha256"]
        assert composed["Resources"][logical_id] == expected
    assert set(composed["Resources"]) - set(legacy["Resources"]) == {
        "PreSupportFenceExecutorRole",
        "PreSupportFenceExecutorFunction",
        "PreSupportFenceExecutorVersion",
        "FenceSourceSettlementMaterializerRole",
        "FenceSourceSettlementMaterializerFunction",
        "FenceSourceSettlementMaterializerVersion",
        "FenceSourceSettledArtifactPublisherRole",
    }
    assert composed["Metadata"]["FenceRuntimeRecordType"] == (
        "glm52_h1g_retained_fence_runtime_v2"
    )
    assert (
        "KeepGlm52Task12TerminalV2VersionArn"
        not in composed["Metadata"]["SupportImportsRetainedOnly"]
    )
    assert (
        "KeepGlm52Task12WorkerDrainVersionArn"
        not in composed["Metadata"]["SupportImportsRetainedOnly"]
    )
    assert (
        "KeepGlm52Task9LiabilityWatcherVersionArn"
        not in composed["Metadata"]["SupportImportsRetainedOnly"]
    )
    assert hashlib.sha256(canonical_json_bytes(composed)).hexdigest() == (
        "8d4b73701b1d91320250a633b8f6d59cbd288610dd739ef8185e18c4e5f3002d"
    )


def test_complete_composition_preserves_gpu_teacher_graph_and_exports() -> None:
    """Break caught: retained augmentation replaces the actual base stack graph."""

    from glm52_enforcement.support_plane import (
        build_postcreate_retained_support_fragment,
        build_pre_support_retained_runtime_fragment,
        compose_complete_retained_template,
        support_inputs_from_mapping,
    )

    base = _gpu_teacher_template()
    original = copy.deepcopy(base)
    inputs = support_inputs_from_mapping(_support_inputs())
    pre_support = build_pre_support_retained_runtime_fragment(_support_build_inputs())
    postcreate = build_postcreate_retained_support_fragment(inputs)

    complete = compose_complete_retained_template(
        base,
        pre_support,
        postcreate,
    )

    assert base == original
    for section in (
        "Parameters",
        "Rules",
        "Conditions",
        "Resources",
        "Outputs",
    ):
        for key, value in original[section].items():
            assert complete[section][key] == value
    assert (
        complete["Outputs"]["ModelBucketName"]["Export"]
        == (original["Outputs"]["ModelBucketName"]["Export"])
    )
    assert (
        complete["Outputs"]["LaunchTemplateId"]["Export"]
        == (original["Outputs"]["LaunchTemplateId"]["Export"])
    )
    assert complete["Outputs"]["Task12TerminalV2VersionArn"]["Export"] == {
        "Name": "KeepGlm52Task12TerminalV2VersionArn"
    }
    assert complete["Outputs"]["Task12WorkerDrainVersionArn"]["Export"] == {
        "Name": "KeepGlm52Task12WorkerDrainVersionArn"
    }
    assert complete["Outputs"]["Task9LiabilityWatcherVersionArn"]["Export"] == {
        "Name": "KeepGlm52Task9LiabilityWatcherVersionArn"
    }
    assert set(complete["Resources"]) == {
        *original["Resources"],
        *pre_support["Resources"],
        *postcreate["Resources"],
    }


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda fragment: fragment["Resources"].update(
                {"BaseResource": {"Type": "AWS::S3::Bucket"}}
            ),
            "resource collision",
        ),
        (
            lambda fragment: fragment["Parameters"].update(
                {"BaseParameter": {"Type": "Number"}}
            ),
            "parameter collision",
        ),
        (
            lambda fragment: fragment["Outputs"].update(
                {"LegacyOutput": {"Value": "replacement"}}
            ),
            "output collision",
        ),
        (
            lambda fragment: fragment["Outputs"]["AddedOutput"].update(
                {"Export": {"Name": "keep-glm52-gpu-Legacy"}}
            ),
            "output export collision",
        ),
        (
            lambda fragment: fragment.update(
                {"Mappings": {"BaseMap": {"Key": {"Value": "foreign"}}}}
            ),
            "mapping collision",
        ),
        (
            lambda fragment: fragment.update(
                {"Conditions": {"BaseCondition": {"Fn::Equals": ["a", "b"]}}}
            ),
            "condition collision",
        ),
        (
            lambda fragment: fragment["Metadata"]["Nested"].update(
                {"Owner": "foreign"}
            ),
            "metadata conflict",
        ),
        (
            lambda fragment: fragment.update({"AWSTemplateFormatVersion": "foreign"}),
            "format",
        ),
    ],
)
def test_complete_composition_rejects_collisions_and_replacement(
    mutate,
    error: str,
) -> None:
    """Break caught: a fragment can delete or replace existing stack authority."""

    from glm52_enforcement.support_plane import (
        compose_complete_retained_template,
    )

    base = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Parameters": {"BaseParameter": {"Type": "String"}},
        "Mappings": {"BaseMap": {"Key": {"Value": "original"}}},
        "Conditions": {"BaseCondition": {"Fn::Equals": ["a", "a"]}},
        "Resources": {"BaseResource": {"Type": "AWS::S3::Bucket"}},
        "Outputs": {
            "LegacyOutput": {
                "Value": {"Ref": "BaseResource"},
                "Export": {"Name": "keep-glm52-gpu-Legacy"},
            }
        },
        "Metadata": {"Nested": {"Owner": "base"}},
    }
    fragment = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Parameters": {"AddedParameter": {"Type": "String"}},
        "Resources": {"AddedResource": {"Type": "AWS::SNS::Topic"}},
        "Outputs": {"AddedOutput": {"Value": "added"}},
        "Metadata": {"Nested": {"Added": True}},
    }
    mutate(fragment)
    snapshot = copy.deepcopy(base)

    with pytest.raises(ValueError, match=error):
        compose_complete_retained_template(base, fragment)
    assert base == snapshot


def test_complete_composition_rejects_malformed_fragments() -> None:
    """Break caught: non-template data is accepted as an additive fragment."""

    from glm52_enforcement.support_plane import (
        compose_complete_retained_template,
    )

    base = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {},
    }
    with pytest.raises(ValueError, match="malformed"):
        compose_complete_retained_template(
            base,
            {
                "AWSTemplateFormatVersion": "2010-09-09",
                "Resources": [],
            },
        )


def test_exact_typed_support_inputs_have_one_canonical_identity() -> None:
    """Break caught: guessed/unknown fields or noncanonical input can bind a stack."""

    from glm52_enforcement.support_plane import (
        support_inputs_from_mapping,
        support_inputs_identity,
        support_inputs_projection,
    )

    value = _support_inputs()
    inputs = support_inputs_from_mapping(value)
    assert support_inputs_projection(inputs) == value
    assert {
        "organizations_target_arn",
        "baseline_scp_arn",
        "maintenance_seal_scp_arn",
    }.isdisjoint(value)
    first = support_inputs_identity(inputs)
    second = support_inputs_identity(support_inputs_from_mapping(dict(value)))
    assert first == second
    assert (
        first
        == hashlib.sha256(
            __import__(
                "glm52_enforcement.canonical",
                fromlist=["canonical_json_bytes"],
            ).canonical_json_bytes(value)
        ).hexdigest()
    )

    unknown = dict(value)
    unknown["guessed_ami_alias"] = "latest"
    with __import__("pytest").raises(ValueError):
        support_inputs_from_mapping(unknown)

    wrong_digest = dict(value)
    wrong_digest["host_user_data_sha256"] = "f" * 64
    with __import__("pytest").raises(ValueError):
        support_inputs_from_mapping(wrong_digest)

    foreign_nat = dict(value)
    foreign_nat["nat_gateway_id"] = "nat-fffffffffffffffff"
    with __import__("pytest").raises(ValueError):
        support_inputs_from_mapping(foreign_nat)


@pytest.mark.parametrize(
    "builder",
    [
        "support_build_inputs_from_mapping",
        "support_inputs_from_mapping",
    ],
)
def test_retained_exports_preserve_legacy_inventory_as_required_subset(
    builder: str,
) -> None:
    """Break caught: safe additive retained exports were rejected as drift."""

    from glm52_enforcement import support_plane

    value = (
        _support_build_input_mapping()
        if builder == "support_build_inputs_from_mapping"
        else _support_inputs()
    )
    parsed = getattr(support_plane, builder)(value)

    assert parsed.retained_export_names == tuple(value["retained_export_names"])


@pytest.mark.parametrize(
    "builder",
    [
        "support_build_inputs_from_mapping",
        "support_inputs_from_mapping",
    ],
)
def test_retained_exports_require_terminal_version_authority(
    builder: str,
) -> None:
    """Break caught: support imported a retained export not in its authority."""

    from glm52_enforcement import support_plane

    value = (
        _support_build_input_mapping()
        if builder == "support_build_inputs_from_mapping"
        else _support_inputs()
    )
    value["retained_export_names"].remove("KeepGlm52Task12TerminalV2VersionArn")
    with pytest.raises(
        ValueError,
        match="retained export inventory.*(required|drifted)",
    ):
        getattr(support_plane, builder)(value)


def test_precreate_build_inputs_exclude_all_postcreate_physical_truth() -> None:
    """Break caught: a logical template build depends on invented CFN outputs."""

    from glm52_enforcement.support_plane import (
        support_build_inputs_from_mapping,
        support_build_inputs_projection,
    )

    value = _support_build_input_mapping()

    inputs = support_build_inputs_from_mapping(value)

    assert support_build_inputs_projection(inputs) == value
    assert not hasattr(inputs, "nat_gateway_id")
    assert not hasattr(inputs, "support_stack_id")
    assert not hasattr(inputs, "support_deletion_inventory")

    invented = dict(value)
    invented["nat_gateway_id"] = "nat-00000000000000003"
    with pytest.raises(ValueError, match="unknown"):
        support_build_inputs_from_mapping(invented)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("version", "$LATEST"),
        (
            "version_arn",
            (
                "arn:aws:lambda:us-west-2:246813579024:function:"
                "keep-glm52-h1g-worker-drain-signal"
            ),
        ),
        ("code_sha256", "4" * 64),
        ("export_name", "KeepGlm52Task12TerminalV2VersionArn"),
    ],
)
def test_support_inputs_bind_exact_retained_numeric_function_versions(
    field: str,
    replacement: str,
) -> None:
    from glm52_enforcement.support_plane import (
        support_build_inputs_from_mapping,
        support_inputs_from_mapping,
    )

    for builder in (
        support_build_inputs_from_mapping,
        support_inputs_from_mapping,
    ):
        value = (
            _support_build_input_mapping()
            if builder is support_build_inputs_from_mapping
            else _support_inputs()
        )
        bindings = value["retained_function_version_bindings"]
        assert type(bindings) is list
        bindings[1][field] = replacement
        with pytest.raises(ValueError, match="retained function"):
            builder(value)


def test_precreate_template_uses_only_logical_stack_and_nat_coordinates() -> None:
    """Break caught: the first-phase template embeds future physical IDs."""

    from glm52_enforcement.canonical import canonical_json_bytes
    from glm52_enforcement.support_plane import (
        build_support_precreate_plane,
        support_build_inputs_from_mapping,
        support_price_card_from_mapping,
    )

    complete = _support_inputs()
    future_values = {
        field: complete[field]
        for field in (
            "nat_gateway_id",
            "support_stack_id",
            "support_deletion_inventory",
        )
    }
    value = _support_build_input_mapping()
    inputs = support_build_inputs_from_mapping(value)
    bundle = build_support_precreate_plane(
        inputs=inputs,
        price_card=support_price_card_from_mapping(
            _price_card(),
            inputs=inputs,
        ),
    )

    assert bundle.retained_runtime_fragment == (
        __import__(
            "glm52_enforcement.support_plane",
            fromlist=["build_pre_support_retained_runtime_fragment"],
        ).build_pre_support_retained_runtime_fragment(inputs)
    )
    assert (
        bundle.manifest["pre_support_retained_runtime_body_sha256"]
        == hashlib.sha256(
            canonical_json_bytes(bundle.retained_runtime_fragment)
        ).hexdigest()
    )
    raw = canonical_json_bytes(bundle.support_template)
    assert future_values["nat_gateway_id"].encode("ascii") not in raw
    assert future_values["support_stack_id"].encode("ascii") not in raw
    assert b"glm52_h1g_support_deletion_inventory_v1" not in raw
    variables = bundle.support_template["Resources"]["SupportDeadlineFunction"][
        "Properties"
    ]["Environment"]["Variables"]
    assert variables["GLM52_SUPPORT_STACK_ID"] == {"Ref": "AWS::StackId"}
    assert variables["GLM52_NAT_GATEWAY_ID"] == {"Ref": "NatGateway"}


def test_postcreate_materializer_reads_every_cfn_resource_and_derives_children() -> (
    None
):
    """Break caught: caller-supplied physical IDs stand in for live CFN truth."""

    from glm52_enforcement.canonical import canonical_json_bytes
    from glm52_enforcement.fence_executor import (
        reviewed_support_resources_from_postcreate_inventory,
    )
    from glm52_enforcement.support_plane import (
        SupportMaterializationServices,
        build_support_precreate_plane,
        coordinate_support_postcreate,
        support_build_inputs_from_mapping,
        support_price_card_from_mapping,
    )

    build_value = _support_build_input_mapping()
    build_inputs = support_build_inputs_from_mapping(build_value)
    template = build_support_precreate_plane(
        inputs=build_inputs,
        price_card=support_price_card_from_mapping(
            _price_card(),
            inputs=build_inputs,
        ),
    ).support_template
    stack_id = (
        "arn:aws:cloudformation:us-west-2:246813579024:"
        "stack/keep-glm52-h1g-support/"
        "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    )
    physical = {
        logical_id: (
            "nat-00000000000000003"
            if logical_id == "NatGateway"
            else "i-00000000000000001"
            if logical_id == "CombinedHost"
            else "sg-0000000000000000b"
            if logical_id == "CombinedHostSecurityGroup"
            else "vpce-00000000000000011"
            if logical_id == "SecretsManagerEndpoint"
            else f"physical-{logical_id}"
        )
        for logical_id in template["Resources"]
    }
    for logical_id, resource in template["Resources"].items():
        properties = resource.get("Properties", {})
        if resource["Type"] == "AWS::S3::Bucket":
            physical[logical_id] = properties["BucketName"]
        elif resource["Type"] == "AWS::IAM::Role" and isinstance(
            properties.get("RoleName"), str
        ):
            physical[logical_id] = properties["RoleName"]
        elif resource["Type"] == "AWS::CloudWatch::Alarm":
            physical[logical_id] = properties["AlarmName"]
        elif resource["Type"] == "AWS::SQS::Queue":
            physical[logical_id] = (
                "https://sqs.us-west-2.amazonaws.com/246813579024/"
                f"{properties['QueueName']}"
            )
        elif resource["Type"] == "AWS::SecretsManager::Secret":
            physical[logical_id] = (
                "arn:aws:secretsmanager:us-west-2:246813579024:secret:"
                f"{properties['Name']}-ABC123"
            )
        elif resource["Type"] == "AWS::Lambda::Function":
            physical[logical_id] = properties.get(
                "FunctionName",
                f"keep-glm52-h1g-support-{logical_id.lower()}-ABC123",
            )
        elif resource["Type"] == "AWS::StepFunctions::StateMachine":
            physical[logical_id] = (
                "arn:aws:states:us-west-2:246813579024:stateMachine:"
                f"{properties.get('StateMachineName', logical_id)}"
            )
    for logical_id, resource in template["Resources"].items():
        properties = resource.get("Properties", {})
        if resource["Type"] == "AWS::Lambda::Version":
            function_logical_id = properties["FunctionName"]["Ref"]
            version = (
                1
                if logical_id
                in {
                    "NumericBindingVersion",
                    "RetainedCancellationVersion",
                }
                else 7
            )
            physical[logical_id] = (
                "arn:aws:lambda:us-west-2:246813579024:function:"
                f"{physical[function_logical_id]}:{version}"
            )
        elif resource["Type"] == "AWS::StepFunctions::StateMachineVersion":
            machine_logical_id = properties["StateMachineArn"]["Ref"]
            physical[logical_id] = f"{physical[machine_logical_id]}:7"

    class CloudFormation:
        def __init__(self) -> None:
            self.details: list[str] = []
            self.list_calls: list[dict[str, object]] = []

        def describe_stacks(self, **request: object) -> dict[str, object]:
            assert request == {"StackName": stack_id}
            return {
                "Stacks": [
                    {
                        "StackId": stack_id,
                        "StackName": "keep-glm52-h1g-support",
                        "StackStatus": "CREATE_COMPLETE",
                        "Outputs": [
                            {
                                "OutputKey": output_name,
                                "OutputValue": (
                                    "iss-" + "9" * 64
                                    if output_name == "IssuanceId"
                                    else (
                                        "ACTIVATION#act-20260728-0001#"
                                        "CUSTOM_RESOURCE_GRANT#"
                                        "aaaaaaaa-bbbb-4ccc-8ddd-"
                                        "eeeeeeeeeeee"
                                    )
                                    if output_name == "DirectGrantEvidenceCoordinate"
                                    else "2026-07-29T12:00:00Z"
                                    if output_name == "TlsNotValidBefore"
                                    else "2026-08-01T12:00:00Z"
                                    if output_name == "TlsNotValidAfter"
                                    else (
                                        str((index % 9) + 1) * 64
                                        if output_name.endswith("VersionId")
                                        else str((index % 9) + 1) * 64
                                    )
                                ),
                            }
                            for index, output_name in enumerate(
                                template["Outputs"],
                                start=1,
                            )
                        ],
                    }
                ],
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "describe-stack",
                },
            }

        def list_stack_resources(self, **request: object) -> dict[str, object]:
            self.list_calls.append(dict(request))
            all_rows = [
                {
                    "LogicalResourceId": logical_id,
                    "PhysicalResourceId": physical[logical_id],
                    "ResourceType": resource["Type"],
                    "ResourceStatus": "CREATE_COMPLETE",
                }
                for logical_id, resource in template["Resources"].items()
            ]
            if request == {"StackName": stack_id}:
                page = all_rows[:57]
                next_token = "task7-page-2"
            elif request == {
                "StackName": stack_id,
                "NextToken": "task7-page-2",
            }:
                page = all_rows[57:]
                next_token = None
            else:
                raise AssertionError(f"foreign pagination request: {request}")
            response = {
                "StackResourceSummaries": [*page],
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": f"list-resources-{len(self.list_calls)}",
                },
            }
            if next_token is not None:
                response["NextToken"] = next_token
            return response

        def describe_stack_resource(self, **request: object) -> dict[str, object]:
            logical_id = request["LogicalResourceId"]
            assert request == {
                "StackName": stack_id,
                "LogicalResourceId": logical_id,
            }
            self.details.append(logical_id)
            return {
                "StackResourceDetail": {
                    "StackId": stack_id,
                    "StackName": "keep-glm52-h1g-support",
                    "LogicalResourceId": logical_id,
                    "PhysicalResourceId": physical[logical_id],
                    "ResourceType": template["Resources"][logical_id]["Type"],
                    "ResourceStatus": "CREATE_COMPLETE",
                },
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": f"detail-{logical_id}",
                },
            }

    class Ec2:
        def describe_instances(self, **request: object) -> dict[str, object]:
            assert request == {"InstanceIds": ["i-00000000000000001"]}
            return {
                "Reservations": [
                    {
                        "Instances": [
                            {
                                "InstanceId": "i-00000000000000001",
                                "ImageId": build_inputs.host_ami_id,
                                "PrivateIpAddress": (build_inputs.host_private_ip),
                                "State": {"Name": "running"},
                                "IamInstanceProfile": {
                                    "Arn": (
                                        "arn:aws:iam::246813579024:"
                                        "instance-profile/"
                                        + physical["CombinedHostProfile"]
                                    )
                                },
                                "RootDeviceName": "/dev/sda1",
                                "BlockDeviceMappings": [
                                    {
                                        "DeviceName": "/dev/sda1",
                                        "Ebs": {"VolumeId": ("vol-00000000000000015")},
                                    }
                                ],
                            }
                        ]
                    }
                ],
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "instance-root",
                },
            }

        def describe_vpc_endpoints(self, **request: object) -> dict[str, object]:
            assert request == {"VpcEndpointIds": ["vpce-00000000000000011"]}
            return {
                "VpcEndpoints": [
                    {
                        "VpcEndpointId": "vpce-00000000000000011",
                        "NetworkInterfaceIds": [
                            "eni-00000000000000021",
                            "eni-00000000000000022",
                        ],
                    }
                ],
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "endpoint-enis",
                },
            }

    cloudformation = CloudFormation()
    postcreate = coordinate_support_postcreate(
        inputs=build_inputs,
        template=template,
        support_stack_id=stack_id,
        services=SupportMaterializationServices(
            cloudformation=cloudformation,
            ec2=Ec2(),
        ),
    )

    inventory = postcreate.inputs.support_deletion_inventory
    assert len(inventory["stack_resources"]) == len(template["Resources"]) == 202
    assert len(cloudformation.details) == 202
    assert set(cloudformation.details) == set(template["Resources"])
    assert cloudformation.list_calls == [
        {"StackName": stack_id},
        {"StackName": stack_id, "NextToken": "task7-page-2"},
    ]
    assert postcreate.inputs.nat_gateway_id == "nat-00000000000000003"
    assert tuple(
        resource["resource_kind"] for resource in inventory["derived_resources"]
    ) == ("ROOT_VOLUME", "INTERFACE_ENDPOINT_ENI", "INTERFACE_ENDPOINT_ENI")
    assert inventory["snapshot_contract"]["state"] == "CAPTURE_AT_DELETION"
    assert all(
        row["resource_type"] != "AWS::EC2::Snapshot"
        for row in inventory["stack_resources"]
    )
    reviewed = reviewed_support_resources_from_postcreate_inventory(inventory)
    assert len(reviewed) == 202
    assert all(row.source_api == "DescribeStackResource" for row in reviewed)
    assert postcreate.inputs.support_stack_id == stack_id
    assert (
        postcreate.retained_augmentation["Metadata"]["SupportDeletionInventorySha256"]
        == inventory["canonical_body_sha256"]
    )
    assert set(postcreate.postcreate_retained_fragment["Resources"]) == {
        logical_id
        for logical_id in postcreate.retained_augmentation["Resources"]
        if logical_id.startswith("H1gRetained")
        or logical_id == "H1gSupportDeletionServiceRole"
    }
    assert (
        postcreate.manifest["postcreate_retained_fragment_body_sha256"]
        == hashlib.sha256(
            canonical_json_bytes(postcreate.postcreate_retained_fragment)
        ).hexdigest()
    )
    assert (
        postcreate.manifest["task12_deployment_authority"]["materialization_phase"]
        == "AFTER_RETAINED_STACK_VERSION_PUBLICATION"
    )
    assert postcreate.manifest["task12_deployment_authority"]["caller_edge_count"] == 10

    physical["NumericBindingVersion"] = (
        "arn:aws:lambda:us-west-2:246813579024:function:"
        "keep-glm52-h1g-numeric-binding:2"
    )
    with pytest.raises(ValueError, match="first-version"):
        coordinate_support_postcreate(
            inputs=build_inputs,
            template=template,
            support_stack_id=stack_id,
            services=SupportMaterializationServices(
                cloudformation=CloudFormation(),
                ec2=Ec2(),
            ),
        )
    physical["NumericBindingVersion"] = (
        "arn:aws:lambda:us-west-2:246813579024:function:"
        "keep-glm52-h1g-numeric-binding:1"
    )

    physical["RehearsalBucket"] = "foreign-caller-invented-bucket"
    with pytest.raises(ValueError, match="physical"):
        coordinate_support_postcreate(
            inputs=build_inputs,
            template=template,
            support_stack_id=stack_id,
            services=SupportMaterializationServices(
                cloudformation=CloudFormation(),
                ec2=Ec2(),
            ),
        )
    physical["RehearsalBucket"] = template["Resources"]["RehearsalBucket"][
        "Properties"
    ]["BucketName"]
    physical["TlsHandlerVersion"] = (
        "arn:aws:lambda:us-west-2:000000000000:function:"
        f"{physical['TlsHandlerFunction']}:7"
    )
    with pytest.raises(ValueError, match="foreign account|callback"):
        coordinate_support_postcreate(
            inputs=build_inputs,
            template=template,
            support_stack_id=stack_id,
            services=SupportMaterializationServices(
                cloudformation=CloudFormation(),
                ec2=Ec2(),
            ),
        )


def test_host_user_data_rejects_private_material_even_with_matching_digest() -> None:
    """Break caught: authenticated user data embeds a key, token, or credential."""

    from glm52_enforcement.support_plane import support_inputs_from_mapping

    private_materials = (
        (
            "#!/bin/sh\n"
            "-----BEGIN PRIVATE KEY-----\n"
            "Zm9yYmlkZGVuLWZpeHR1cmU=\n"
            "-----END PRIVATE KEY-----\n"
        ),
        "#!/bin/sh\nexport SKY_TOKEN=fixture-token-value\n",
        "#!/bin/sh\nexport AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP\n",
        (
            "#!/bin/sh\n"
            "export AUTH='eyJhbGciOiJIUzI1NiJ9."
            "eyJzdWIiOiJmaXh0dXJlIn0.signature'\n"
        ),
    )
    for user_data in private_materials:
        value = _support_inputs()
        value["host_user_data"] = user_data
        value["host_user_data_sha256"] = hashlib.sha256(
            user_data.encode("ascii")
        ).hexdigest()
        with pytest.raises(ValueError):
            support_inputs_from_mapping(value)


def _price_card() -> dict[str, object]:
    return _mechanical_price_card()


def _bundle() -> object:
    from glm52_enforcement.support_plane import (
        build_support_plane,
        support_inputs_from_mapping,
        support_price_card_from_mapping,
    )

    return build_support_plane(
        inputs=support_inputs_from_mapping(_support_inputs()),
        price_card=support_price_card_from_mapping(
            _price_card(),
            inputs=support_inputs_from_mapping(_support_inputs()),
        ),
    )


def test_support_template_has_exact_sole_owner_resource_cardinality() -> None:
    """Break caught: a ninth secret, second host, or retained-owned resource appears."""

    from glm52_enforcement.support_plane import (
        SECRET_LOGICAL_IDS,
        support_resource_cardinality,
        validate_support_template,
    )

    bundle = _bundle()
    template = bundle.support_template
    assert validate_support_template(template, bundle.inputs) is template
    cardinality = support_resource_cardinality(template)
    assert cardinality["AWS::EC2::Subnet"] == 3
    assert cardinality["AWS::EC2::RouteTable"] == 3
    assert cardinality["AWS::EC2::SubnetRouteTableAssociation"] == 3
    assert cardinality["AWS::EC2::Route"] == 1
    assert cardinality["AWS::EC2::NatGateway"] == 1
    assert cardinality["AWS::EC2::EIP"] == 1
    assert cardinality["AWS::EC2::VPCEndpoint"] == 3
    assert cardinality["AWS::EC2::Instance"] == 1
    assert cardinality["AWS::EC2::Volume"] == 1
    assert cardinality["AWS::SecretsManager::Secret"] == 8
    assert cardinality["AWS::S3::Bucket"] == 1
    assert cardinality["AWS::S3::BucketPolicy"] == 1
    assert cardinality["AWS::StepFunctions::StateMachine"] == 1
    assert cardinality["AWS::StepFunctions::StateMachineVersion"] == 1
    assert set(SECRET_LOGICAL_IDS).issubset(template["Resources"])
    assert not {
        "AWS::KMS::Key",
        "AWS::KMS::Alias",
        "AWS::DynamoDB::Table",
    } & set(cardinality)
    assert set(bundle.manifest["resource_ownership"]) == set(template["Resources"])
    assert set(bundle.manifest["resource_ownership"].values()) == {
        "keep-glm52-h1g-support"
    }

    ninth = copy.deepcopy(template)
    ninth["Resources"]["ForbiddenNinthSecret"] = copy.deepcopy(
        ninth["Resources"][SECRET_LOGICAL_IDS[0]]
    )
    with pytest.raises(ValueError):
        validate_support_template(ninth, bundle.inputs)

    retained_key = copy.deepcopy(template)
    retained_key["Resources"]["ForbiddenSupportKmsKey"] = {
        "Type": "AWS::KMS::Key",
        "Properties": {},
    }
    with pytest.raises(ValueError):
        validate_support_template(retained_key, bundle.inputs)


def test_three_support_cidrs_are_contained_disjoint_and_exactly_az_mapped() -> None:
    """Break caught: overlap or wrong AZ lets support traffic escape its frozen shape."""

    from glm52_enforcement.support_plane import (
        support_inputs_from_mapping,
        validate_support_network_inputs,
    )

    inputs = support_inputs_from_mapping(_support_inputs())
    assert validate_support_network_inputs(inputs) is True
    bundle = _bundle()
    resources = bundle.support_template["Resources"]
    assert resources["HostEgressSubnet"]["Properties"] == {
        "AvailabilityZone": "us-west-2a",
        "CidrBlock": "10.20.101.0/24",
        "MapPublicIpOnLaunch": False,
        "VpcId": "vpc-0123456789abcdef0",
    }
    assert (
        resources["PrimaryIsolatedSubnet"]["Properties"]["AvailabilityZone"]
        == "us-west-2a"
    )
    assert resources["PrimaryIsolatedSubnet"]["Properties"]["CidrBlock"] == (
        "10.20.102.0/24"
    )
    assert (
        resources["AlternateIsolatedSubnet"]["Properties"]["AvailabilityZone"]
        == "us-west-2b"
    )
    assert resources["AlternateIsolatedSubnet"]["Properties"]["CidrBlock"] == (
        "10.20.103.0/24"
    )

    overlap = _support_inputs()
    overlap["existing_subnet_cidrs"] = [
        "10.20.0.0/24",
        "10.20.101.128/25",
    ]
    with pytest.raises(ValueError):
        validate_support_network_inputs(support_inputs_from_mapping(overlap))


def test_only_host_has_nat_route_and_endpoints_are_exactly_scoped() -> None:
    """Break caught: isolated Lambda egress or a second/KMS endpoint is admitted."""

    from glm52_enforcement.support_plane import (
        validate_support_template,
    )

    bundle = _bundle()
    resources = bundle.support_template["Resources"]
    routes = {
        logical_id: resource
        for logical_id, resource in resources.items()
        if resource["Type"] == "AWS::EC2::Route"
    }
    assert routes == {
        "HostDefaultRoute": {
            "Type": "AWS::EC2::Route",
            "DependsOn": "NatGateway",
            "Properties": {
                "DestinationCidrBlock": "0.0.0.0/0",
                "NatGatewayId": {"Ref": "NatGateway"},
                "RouteTableId": {"Ref": "HostRouteTable"},
            },
        }
    }
    endpoints = [
        resource["Properties"]
        for resource in resources.values()
        if resource["Type"] == "AWS::EC2::VPCEndpoint"
    ]
    assert sorted(item["VpcEndpointType"] for item in endpoints) == [
        "Gateway",
        "Gateway",
        "Interface",
    ]
    interface = next(
        item for item in endpoints if item["VpcEndpointType"] == "Interface"
    )
    assert interface["PrivateDnsEnabled"] is True
    assert interface["SubnetIds"] == [
        {"Ref": "PrimaryIsolatedSubnet"},
        {"Ref": "AlternateIsolatedSubnet"},
    ]
    assert all("kms" not in item["ServiceName"] for item in endpoints)

    isolated_route = copy.deepcopy(bundle.support_template)
    isolated_route["Resources"]["ForbiddenIsolatedDefault"] = {
        "Type": "AWS::EC2::Route",
        "Properties": {
            "DestinationCidrBlock": "0.0.0.0/0",
            "NatGatewayId": {"Ref": "NatGateway"},
            "RouteTableId": {"Ref": "PrimaryIsolatedRouteTable"},
        },
    }
    with pytest.raises(ValueError):
        validate_support_template(isolated_route, bundle.inputs)

    second_interface = copy.deepcopy(bundle.support_template)
    second_interface["Resources"]["ForbiddenKmsEndpoint"] = copy.deepcopy(
        second_interface["Resources"]["SecretsManagerEndpoint"]
    )
    second_interface["Resources"]["ForbiddenKmsEndpoint"]["Properties"][
        "ServiceName"
    ] = "com.amazonaws.us-west-2.kms"
    with pytest.raises(ValueError):
        validate_support_template(second_interface, bundle.inputs)


def test_four_client_security_paths_are_mutually_unusable() -> None:
    """Break caught: one client identity can reach another relay listener."""

    from glm52_enforcement.support_plane import (
        CLIENT_PATHS,
        validate_support_template,
    )

    bundle = _bundle()
    resources = bundle.support_template["Resources"]
    observed = {}
    for name, path in CLIENT_PATHS.items():
        egress = resources[path["egress_logical_id"]]["Properties"]
        ingress = resources[path["ingress_logical_id"]]["Properties"]
        assert egress["DestinationSecurityGroupId"] == {
            "Ref": "CombinedHostSecurityGroup"
        }
        assert ingress["SourceSecurityGroupId"] == {
            "Ref": path["client_security_group"]
        }
        assert egress["FromPort"] == egress["ToPort"] == path["port"](bundle.inputs)
        assert ingress["FromPort"] == ingress["ToPort"] == path["port"](bundle.inputs)
        observed[name] = (
            path["client_security_group"],
            ingress["FromPort"],
        )
    assert len(set(observed.values())) == 4

    cross_path = copy.deepcopy(bundle.support_template)
    cross_path["Resources"]["AttestationHostIngress"]["Properties"]["FromPort"] = (
        bundle.inputs.launch_admission_port
    )
    cross_path["Resources"]["AttestationHostIngress"]["Properties"]["ToPort"] = (
        bundle.inputs.launch_admission_port
    )
    with pytest.raises(ValueError):
        validate_support_template(cross_path, bundle.inputs)


def test_each_isolated_reader_has_one_exact_443_secret_path_and_no_cross_run_authority() -> (
    None
):
    """Break caught: an isolated reader cannot reach PrivateLink or can cross-read."""

    bundle = _bundle()
    resources = bundle.support_template["Resources"]
    expected = {
        "Attestation": (
            "AttestationRole",
            "AttestationClientSecurityGroup",
            (
                (
                    "RawSkyTokenSecret",
                    "h1g-act-20260728-0001-raw-sky-token-v1",
                ),
                (
                    "AttestationClientTlsSecret",
                    "h1g-act-20260728-0001-attestation-client-tls-v1",
                ),
            ),
        ),
        "LaunchAdmission": (
            "LaunchAdmissionRole",
            "LaunchAdmissionClientSecurityGroup",
            (
                (
                    "RawSkyTokenSecret",
                    "h1g-act-20260728-0001-raw-sky-token-v1",
                ),
                (
                    "LaunchAdmissionClientTlsSecret",
                    "h1g-act-20260728-0001-launch-admission-client-tls-v1",
                ),
            ),
        ),
        "NumericBinding": (
            "NumericBindingRole",
            "NumericBindingClientSecurityGroup",
            (
                (
                    "RawSkyTokenSecret",
                    "h1g-act-20260728-0001-raw-sky-token-v1",
                ),
                (
                    "NumericBindingClientTlsSecret",
                    "h1g-act-20260728-0001-numeric-binding-client-tls-v1",
                ),
            ),
        ),
        "RetainedCancellation": (
            "RetainedCancellationRole",
            "RetainedCancellationClientSecurityGroup",
            (
                (
                    "RawSkyTokenSecret",
                    "h1g-act-20260728-0001-raw-sky-token-v1",
                ),
                (
                    "RetainedCancellationClientTlsSecret",
                    "h1g-act-20260728-0001-retained-cancellation-client-tls-v1",
                ),
            ),
        ),
    }
    endpoint_statements = resources["SecretsManagerEndpoint"]["Properties"][
        "PolicyDocument"
    ]["Statement"]
    endpoint_allows = [
        statement for statement in endpoint_statements if statement["Effect"] == "Allow"
    ]
    assert len(endpoint_allows) == 20
    for prefix, (role, client_sg, secret_bindings) in expected.items():
        assert resources[f"{prefix}SecretsEndpointEgress"]["Properties"] == {
            "DestinationSecurityGroupId": {"Ref": "SecretsEndpointSecurityGroup"},
            "FromPort": 443,
            "GroupId": {"Ref": client_sg},
            "IpProtocol": "tcp",
            "ToPort": 443,
        }
        assert resources[f"{prefix}SecretsEndpointIngress"]["Properties"] == {
            "FromPort": 443,
            "GroupId": {"Ref": "SecretsEndpointSecurityGroup"},
            "IpProtocol": "tcp",
            "SourceSecurityGroupId": {"Ref": client_sg},
            "ToPort": 443,
        }
        identity_statements = resources[role]["Properties"]["Policies"][0][
            "PolicyDocument"
        ]["Statement"]
        identity_allows = [
            statement
            for statement in identity_statements
            if statement["Effect"] == "Allow"
        ]
        assert {
            (
                statement["Resource"]["Ref"],
                statement["Condition"]["StringEquals"]["secretsmanager:VersionStage"],
            )
            for statement in identity_allows
            if statement["Action"] == "secretsmanager:GetSecretValue"
        } == set(secret_bindings)
        assert {
            statement["Resource"]["Ref"]
            for statement in identity_allows
            if statement["Action"] == "secretsmanager:ListSecretVersionIds"
            and "Condition" not in statement
        } == {secret for secret, _stage in secret_bindings}
        for secret, stage in secret_bindings:
            assert {
                "Sid": f"{prefix}{secret}GetExactStage",
                "Effect": "Allow",
                "Principal": {"AWS": {"Fn::GetAtt": [role, "Arn"]}},
                "Action": "secretsmanager:GetSecretValue",
                "Resource": {"Ref": secret},
                "Condition": {
                    "StringEquals": {
                        "secretsmanager:VersionStage": stage,
                    }
                },
            } in endpoint_allows
            assert {
                "Sid": f"{prefix}{secret}ListExactSecret",
                "Effect": "Allow",
                "Principal": {"AWS": {"Fn::GetAtt": [role, "Arn"]}},
                "Action": "secretsmanager:ListSecretVersionIds",
                "Resource": {"Ref": secret},
            } in endpoint_allows
    assert {
        "Sid": "DenyCrossRunSecretAuthority",
        "Effect": "Deny",
        "Principal": "*",
        "Action": "secretsmanager:*",
        "NotResource": [
            {"Ref": logical_id}
            for logical_id in (
                "RawSkyTokenSecret",
                "SkyBootstrapSecret",
                "AttestationClientTlsSecret",
                "LaunchAdmissionClientTlsSecret",
                "NumericBindingClientTlsSecret",
                "RetainedCancellationClientTlsSecret",
                "CombinedHostTlsSecret",
                "CaIssuanceSecret",
            )
        ],
    } in endpoint_statements


def test_endpoint_splits_staged_get_from_unconditioned_exact_list() -> None:
    """Break caught: VersionStage condition makes ListSecretVersionIds unusable."""

    from glm52_enforcement.support_plane import validate_support_security_graph

    bundle = _bundle()
    statements = bundle.support_template["Resources"]["SecretsManagerEndpoint"][
        "Properties"
    ]["PolicyDocument"]["Statement"]
    allows = [statement for statement in statements if statement["Effect"] == "Allow"]
    gets = [
        statement
        for statement in allows
        if statement["Action"] == "secretsmanager:GetSecretValue"
    ]
    lists = [
        statement
        for statement in allows
        if statement["Action"] == "secretsmanager:ListSecretVersionIds"
    ]
    assert len(gets) == len(lists) == 10
    assert all(
        set(statement)
        == {
            "Sid",
            "Effect",
            "Principal",
            "Action",
            "Resource",
            "Condition",
        }
        and statement["Condition"]["StringEquals"][
            "secretsmanager:VersionStage"
        ].startswith("h1g-act-20260728-0001-")
        for statement in gets
    )
    assert all(
        set(statement) == {"Sid", "Effect", "Principal", "Action", "Resource"}
        for statement in lists
    )
    for role in (
        "AttestationRole",
        "RehearsalProbeRole",
        "LaunchAdmissionRole",
        "NumericBindingRole",
        "RetainedCancellationRole",
        "CombinedHostRole",
    ):
        identity_allows = [
            statement
            for statement in bundle.support_template["Resources"][role]["Properties"][
                "Policies"
            ][0]["PolicyDocument"]["Statement"]
            if statement["Effect"] == "Allow"
        ]
        assert (
            sum(
                statement["Action"] == "secretsmanager:GetSecretValue"
                and "Condition" in statement
                for statement in identity_allows
            )
            == 2
        )
        assert (
            sum(
                statement["Action"] == "secretsmanager:ListSecretVersionIds"
                and "Condition" not in statement
                for statement in identity_allows
            )
            == 2
        )
    conditioned_list = copy.deepcopy(bundle.support_template)
    list_statement = next(
        statement
        for statement in conditioned_list["Resources"]["SecretsManagerEndpoint"][
            "Properties"
        ]["PolicyDocument"]["Statement"]
        if statement["Effect"] == "Allow"
        and statement["Action"] == "secretsmanager:ListSecretVersionIds"
    )
    list_statement["Condition"] = {
        "StringEquals": {
            "secretsmanager:VersionStage": ("h1g-act-20260728-0001-raw-sky-token-v1")
        }
    }
    with pytest.raises(ValueError):
        validate_support_security_graph(conditioned_list, bundle.inputs)


def test_host_user_data_rejects_certificate_and_public_trust_material() -> None:
    """Break caught: certificate/public trust bytes enter first-boot user data."""

    from glm52_enforcement.support_plane import support_inputs_from_mapping

    forbidden = (
        (
            "-----BEGIN CERTIFICATE-----\n"
            "MIIBfixturecertificatebytes\n"
            "-----END CERTIFICATE-----"
        ),
        (
            "-----BEGIN PUBLIC KEY-----\n"
            "MIIBfixturepublickeybytes\n"
            "-----END PUBLIC KEY-----"
        ),
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFixture host@example",
        "export TLS_CERTIFICATE='MIIBfixturecertificatebytes'",
        (
            "-----BEGIN PGP PUBLIC KEY BLOCK-----\n"
            "mQENfixture\n"
            "-----END PGP PUBLIC KEY BLOCK-----"
        ),
        ("-----BEGIN PKCS7-----\nMIIBfixturepkcs7bytes\n-----END PKCS7-----"),
        ("-----BEGIN CMS-----\nMIIBfixturecmsbytes\n-----END CMS-----"),
        (
            "-----BEGIN PKCS #7 SIGNED DATA-----\n"
            "MIIBfixturesigneddata\n"
            "-----END PKCS #7 SIGNED DATA-----"
        ),
        (
            "ssh-ed25519-cert-v01@openssh.com "
            "AAAAC3NzaC1lZDI1NTE5AAAAIFixture host@example"
        ),
        (
            "ssh-rsa-cert-v01@openssh.com "
            "AAAAB3NzaC1yc2EAAAADAQABAAABAQFixture host@example"
        ),
        (
            "ecdsa-sha2-nistp256-cert-v01@openssh.com "
            "AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYFixture host@example"
        ),
        (
            "sk-ssh-ed25519-cert-v01@openssh.com "
            "AAAAGnNrLXNzaC1lZDI1NTE5Fixture host@example"
        ),
    )
    for material in forbidden:
        candidate = _support_inputs()
        user_data = f"#!/bin/sh\nset -eu\n{material}\nexit 64\n"
        candidate["host_user_data"] = user_data
        candidate["host_user_data_sha256"] = hashlib.sha256(
            user_data.encode("utf-8")
        ).hexdigest()
        with pytest.raises(ValueError):
            support_inputs_from_mapping(candidate)


def test_support_price_gate_is_exact_and_retained_costs_stay_separate() -> None:
    """Break caught: >$25 or retained liability/snapshot cost is hidden in support."""

    from glm52_enforcement.support_plane import (
        build_support_spend_descriptor,
        support_inputs_from_mapping,
        support_price_card_from_mapping,
        validate_support_spend_descriptor,
    )

    inputs = support_inputs_from_mapping(_support_inputs())
    card = support_price_card_from_mapping(_price_card(), inputs=inputs)
    descriptor = build_support_spend_descriptor(inputs=inputs, price_card=card)
    assert validate_support_spend_descriptor(descriptor, inputs) is descriptor
    assert (
        descriptor["estimated_support_total_usd"]
        == (_price_card()["estimated_support_total_usd"])
    )
    assert descriptor["estimated_support_ceiling_usd"] == "25.00"
    assert set(descriptor["retained_costs"]) == {
        "retained_s3",
        "retained_ledger",
        "retained_kms",
        "liability_watcher",
        "snapshot_cleanup",
        "snapshot_retention",
    }
    assert set(descriptor["retained_costs"]).isdisjoint(descriptor["support_costs"])
    assert descriptor["envelope"]["accepted_support_lambda_invocations"] == 10000
    assert descriptor["envelope"]["support_work_stop_hours"] == 68
    assert descriptor["envelope"]["delete_request_deadline_hours"] == 71
    assert descriptor["envelope"]["absence_expected_hours"] == 72
    assert descriptor["envelope"]["nat_processed_gib_max"] == 10
    assert descriptor["envelope"]["workflow_history_events_max"] == 12000
    assert descriptor["envelope"]["transient_log_retention_days"] == 14
    assert descriptor["envelope"]["application_log_ingestion_mib_max"] == 512

    over = _price_card()
    over["support_terms"][0]["unit_price_usd"] = "1.000000"
    over["support_terms"][0]["estimated_usd"] = "72.00"
    over["estimated_support_total_usd"] = "82.50"
    with pytest.raises(ValueError):
        support_price_card_from_mapping(over, inputs=inputs)


def _mechanical_price_card() -> dict[str, object]:
    term_rows = (
        ("combined_host_hours", "instance-hour", "72.000000", "0.100000"),
        ("root_ebs", "gib-hour", "2160.000000", "0.000500"),
        ("data_ebs", "gib-hour", "3600.000000", "0.000300"),
        ("nat_gateway_hours", "gateway-hour", "72.000000", "0.050000"),
        ("nat_processed_gib", "gib", "10.000000", "0.050000"),
        ("eip_hours", "ipv4-hour", "72.000000", "0.005000"),
        ("secrets_eight", "secret-hour", "576.000000", "0.002000"),
        ("secrets_api_calls", "api-call", "20000.000000", "0.000010"),
        (
            "interface_endpoint_two_eni_hours",
            "eni-hour",
            "144.000000",
            "0.010000",
        ),
        ("cross_az_bytes", "gib", "10.000000", "0.010000"),
        ("lambda", "invocation", "10000.000000", "0.000010"),
        ("step_functions", "state-transition", "12000.000000", "0.000010"),
        ("logs_ingestion", "gib", "0.500000", "0.500000"),
        ("logs_storage", "gib-day", "7.000000", "0.010000"),
        ("rehearsal_s3", "gib-day", "3.000000", "0.010000"),
    )
    retained_rows = (
        ("retained_s3", "gib-day", "30.000000", "0.010000"),
        ("retained_ledger", "request", "129600.000000", "0.000001"),
        ("retained_kms", "key-month", "1.000000", "1.000000"),
        ("liability_watcher", "scan", "43200.000000", "0.000010"),
        ("snapshot_cleanup", "delete-call", "12.000000", "0.010000"),
        ("snapshot_retention", "gib-day", "350.000000", "0.001000"),
    )

    def rows(
        values: tuple[tuple[str, str, str, str], ...],
    ) -> list[dict[str, str]]:
        from decimal import Decimal

        return [
            {
                "term": term,
                "unit": unit,
                "quantity": quantity,
                "unit_price_usd": unit_price,
                "estimated_usd": str(
                    (Decimal(quantity) * Decimal(unit_price)).quantize(Decimal("0.01"))
                ),
            }
            for term, unit, quantity, unit_price in values
        ]

    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_price_card_v1",
        "region": "us-west-2",
        "effective_date": "2026-07-28",
        "currency": "USD",
        "support_terms": rows(term_rows),
        "retained_terms": rows(retained_rows),
    }
    from decimal import Decimal

    body["estimated_support_total_usd"] = str(
        sum(
            (Decimal(item["estimated_usd"]) for item in body["support_terms"]),
            Decimal("0.00"),
        )
    )
    unsigned = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    body["price_card_identity_sha256"] = hashlib.sha256(unsigned).hexdigest()
    return body


def test_price_card_identity_and_mechanical_shape_reject_zero_or_substitution() -> None:
    """Break caught: a self-labeled or zero/arbitrary card admits the envelope."""

    from glm52_enforcement.support_plane import (
        support_inputs_from_mapping,
        support_price_card_from_mapping,
    )

    card = _mechanical_price_card()
    input_value = _support_inputs()
    input_value["price_card_identity_sha256"] = card["price_card_identity_sha256"]
    inputs = support_inputs_from_mapping(input_value)
    parsed = support_price_card_from_mapping(card, inputs=inputs)
    assert parsed.price_card_identity_sha256 == card["price_card_identity_sha256"]
    assert parsed.support_terms[0].unit == "instance-hour"
    assert parsed.support_terms[0].quantity == "72.000000"

    for field, replacement in (
        ("quantity", "0.000000"),
        ("unit_price_usd", "0.000000"),
        ("unit", "caller-selected-unit"),
    ):
        mutant = copy.deepcopy(card)
        mutant["support_terms"][0][field] = replacement
        with pytest.raises(ValueError):
            support_price_card_from_mapping(mutant, inputs=inputs)

    substituted = copy.deepcopy(card)
    substituted["support_terms"][0]["unit_price_usd"] = "0.200000"
    substituted["support_terms"][0]["estimated_usd"] = "14.40"
    substituted["estimated_support_total_usd"] = str(
        sum(
            (
                __import__("decimal").Decimal(item["estimated_usd"])
                for item in substituted["support_terms"]
            ),
            __import__("decimal").Decimal("0.00"),
        )
    )
    unsigned = dict(substituted)
    unsigned.pop("price_card_identity_sha256")
    substituted["price_card_identity_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError):
        support_price_card_from_mapping(substituted, inputs=inputs)


def test_combined_host_and_volume_are_private_encrypted_and_snapshot_only() -> None:
    """Break caught: public/SSH host, wrong shape, or restorable volume authority."""

    from glm52_enforcement.support_plane import (
        validate_support_template,
    )

    bundle = _bundle()
    resources = bundle.support_template["Resources"]
    host = resources["CombinedHost"]["Properties"]
    assert host["InstanceType"] == "c6a.xlarge"
    assert host["ImageId"] == bundle.inputs.host_ami_id
    assert host["MetadataOptions"]["HttpTokens"] == "required"
    assert host["NetworkInterfaces"] == [
        {
            "AssociatePublicIpAddress": False,
            "DeviceIndex": "0",
            "GroupSet": [{"Ref": "CombinedHostSecurityGroup"}],
            "PrivateIpAddress": "10.20.101.10",
            "SubnetId": {"Ref": "HostEgressSubnet"},
        }
    ]
    root = host["BlockDeviceMappings"][0]["Ebs"]
    assert root["Encrypted"] is True
    assert root["KmsKeyId"] == bundle.inputs.retained_kms_key_arn
    data = resources["CombinedHostDataVolume"]
    assert data["DeletionPolicy"] == "Snapshot"
    assert data["UpdateReplacePolicy"] == "Snapshot"
    assert data["Properties"]["Encrypted"] is True
    assert data["Properties"]["KmsKeyId"] == bundle.inputs.retained_kms_key_arn
    assert data["Properties"]["Size"] == 50
    assert data["Properties"]["VolumeType"] == "gp3"
    assert "AWS::EC2::SecurityGroupIngress" in {
        resource["Type"] for resource in resources.values()
    }
    assert all(
        resource.get("Properties", {}).get("FromPort") != 22
        for resource in resources.values()
    )
    serialized = __import__("json").dumps(bundle.support_template)
    assert "CreateSnapshot" not in serialized
    assert "Restore" not in serialized
    assert "RunInstances" not in serialized

    public = copy.deepcopy(bundle.support_template)
    public["Resources"]["CombinedHost"]["Properties"]["NetworkInterfaces"][0][
        "AssociatePublicIpAddress"
    ] = True
    with pytest.raises(ValueError):
        validate_support_template(public, bundle.inputs)


def test_exact_eight_secrets_have_one_stage_and_closed_reader_partition() -> None:
    """Break caught: rotation, duplicate stage, ninth secret, or cross-reader access."""

    from glm52_enforcement.support_plane import (
        SECRET_LOGICAL_IDS,
        secret_inventory,
        validate_secret_reader_partition,
        validate_support_template,
    )

    bundle = _bundle()
    inventory = secret_inventory(
        bundle.support_template,
        bundle.inputs,
        version_observations=_secret_version_observations(),
    )
    assert tuple(inventory) == SECRET_LOGICAL_IDS
    assert len({item["version_stage"] for item in inventory.values()}) == 8
    assert all(item["authorized_version_count"] == 1 for item in inventory.values())
    assert (
        validate_secret_reader_partition(bundle.support_template, bundle.inputs) is True
    )
    resources = bundle.support_template["Resources"]
    decision_policies = resources["DecisionRole"]["Properties"]["Policies"]
    assert len(decision_policies) == 1
    assert decision_policies[0]["PolicyName"] == ("AssumeExactClosureSessionOnly")
    assert all("Rotation" not in resource["Type"] for resource in resources.values())
    host_statements = resources["CombinedHostRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    assert all(
        "secretsmanager:PutSecretValue"
        not in (
            [statement["Action"]]
            if isinstance(statement["Action"], str)
            else statement["Action"]
        )
        for statement in host_statements
        if statement["Effect"] == "Allow"
    )

    rotate = copy.deepcopy(bundle.support_template)
    rotate["Resources"]["ForbiddenRotation"] = {
        "Type": "AWS::SecretsManager::RotationSchedule",
        "Properties": {"SecretId": {"Ref": "RawSkyTokenSecret"}},
    }
    with pytest.raises(ValueError):
        validate_support_template(rotate, bundle.inputs)

    cross_reader = copy.deepcopy(bundle.support_template)
    statements = cross_reader["Resources"]["AttestationRole"]["Properties"]["Policies"][
        0
    ]["PolicyDocument"]["Statement"]
    statements.insert(
        0,
        {
            "Sid": "ForbiddenCrossReader",
            "Effect": "Allow",
            "Action": "secretsmanager:GetSecretValue",
            "Resource": {"Ref": "LaunchAdmissionClientTlsSecret"},
            "Condition": {
                "StringEquals": {
                    "secretsmanager:VersionStage": (
                        "h1g-act-20260728-0001-launch-admission-client-tls-v1"
                    )
                }
            },
        },
    )
    with pytest.raises(ValueError):
        validate_secret_reader_partition(cross_reader, bundle.inputs)


def test_task11_closure_session_role_is_exact_nonattached_authority() -> None:
    """Break caught: Decision keeps data access or closure trust/session drifts."""

    from glm52_enforcement.support_plane import (
        RUNTIME_FUNCTIONS,
        validate_support_template,
    )

    bundle = _bundle()
    resources = bundle.support_template["Resources"]
    cutoff = "2026-07-28T00:15:00Z"
    activation_sha = hashlib.sha256(
        bundle.inputs.activation_id.encode("ascii")
    ).hexdigest()[:16]
    role_name = "keep-glm52-h1g-closure-session-" + activation_sha
    role_arn = {"Fn::GetAtt": ["ClosureSessionRole", "Arn"]}
    decision_role = resources["DecisionRole"]["Properties"]
    closure_role = resources["ClosureSessionRole"]["Properties"]

    decision_statements = [
        statement
        for policy in decision_role["Policies"]
        for statement in policy["PolicyDocument"]["Statement"]
    ]
    decision_allows = {
        action
        for statement in decision_statements
        if statement["Effect"] == "Allow"
        for action in (
            [statement["Action"]]
            if isinstance(statement["Action"], str)
            else statement["Action"]
        )
    }
    assert decision_allows == {
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "sts:AssumeRole",
    }
    assume_allow = next(
        statement
        for statement in decision_statements
        if statement.get("Action") == "sts:AssumeRole"
        and statement["Effect"] == "Allow"
    )
    assert assume_allow["Resource"] == role_arn

    cutoff_denies = [
        statement for statement in decision_statements if statement["Effect"] == "Deny"
    ]
    assert cutoff_denies == [
        {
            "Sid": "DenyPreCutoffRuntimeSessionAfterCutoff",
            "Effect": "Deny",
            "Action": "*",
            "Resource": "*",
            "Condition": {
                "DateGreaterThanEquals": {"aws:CurrentTime": cutoff},
                "DateLessThan": {"aws:TokenIssueTime": cutoff},
            },
        },
        {
            "Sid": "DenyMissingRuntimeTokenIssueTimeAfterCutoff",
            "Effect": "Deny",
            "Action": "*",
            "Resource": "*",
            "Condition": {
                "DateGreaterThanEquals": {"aws:CurrentTime": cutoff},
                "Null": {"aws:TokenIssueTime": "true"},
            },
        },
    ]

    assert closure_role["RoleName"] == role_name
    assert closure_role["MaxSessionDuration"] == 3600
    assert closure_role["AssumeRolePolicyDocument"] == {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": {"Fn::GetAtt": ["DecisionRole", "Arn"]}},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "ArnEquals": {
                        "aws:PrincipalArn": {"Fn::GetAtt": ["DecisionRole", "Arn"]}
                    },
                    "StringEquals": {
                        "sts:ExternalId": bundle.inputs.lambda_code_sha256
                    },
                    "NumericEquals": {"sts:DurationSeconds": 900},
                    "StringLike": {
                        "sts:RoleSessionName": ("h1g-decision-" + activation_sha + "-*")
                    },
                },
            }
        ],
    }
    closure_statements = [
        statement
        for policy in closure_role["Policies"]
        for statement in policy["PolicyDocument"]["Statement"]
    ]
    closure_allow_actions = {
        action
        for statement in closure_statements
        if statement["Effect"] == "Allow"
        for action in (
            [statement["Action"]]
            if isinstance(statement["Action"], str)
            else statement["Action"]
        )
    }
    assert "cloudformation:ListChangeSets" in closure_allow_actions
    assert "sts:GetCallerIdentity" in closure_allow_actions
    assert "sts:AssumeRole" not in closure_allow_actions
    assert "iam:PassRole" not in closure_allow_actions
    assert [
        statement for statement in closure_statements if statement["Effect"] == "Deny"
    ] == cutoff_denies

    attached_roles = [
        resource["Properties"]["Role"]
        for logical_id, resource in resources.items()
        if resource["Type"] == "AWS::Lambda::Function"
        and logical_id != "TlsHandlerFunction"
    ]
    assert role_arn not in attached_roles
    assert all(
        role_name not in str(resource.get("Properties", {}).get("Roles", []))
        for resource in resources.values()
    )
    assert all(
        resource.get("Properties", {}).get("RoleArn") != role_arn
        for resource in resources.values()
    )

    exact_callees = {
        "SourceGpuSpend",
        "SourceSubmissionIntent",
        "SourceControllerBaseline",
        "SourceControlPlaneReadiness",
        "SourceSubmissionAcquisition",
        "FenceExecutor",
        "FenceSuccessor",
        "ClaimWriter",
        "DecisionWriter",
        "TerminalV1Writer",
        "ClosureHandoff",
        "Attestation",
        "LaunchAdmission",
        "NumericBinding",
    }
    assert exact_callees < set(RUNTIME_FUNCTIONS)
    for name in exact_callees:
        permission = resources[f"{name}ClosureInvokePermission"]
        assert permission == {
            "Type": "AWS::Lambda::Permission",
            "Properties": {
                "Action": "lambda:InvokeFunction",
                "FunctionName": {"Ref": f"{name}Version"},
                "Principal": role_arn,
                "SourceAccount": "246813579024",
            },
        }
        variables = resources[f"{name}Function"]["Properties"]["Environment"][
            "Variables"
        ]
        if name.startswith("Source") or name in {
            "FenceSuccessor",
            "ClaimWriter",
            "DecisionWriter",
            "TerminalV1Writer",
            "ClosureHandoff",
        }:
            assert variables["GLM52_CLOSURE_ROLE_ARN"] == role_arn

    assert (
        resources["DecisionFunction"]["Properties"]["Environment"]["Variables"][
            "GLM52_CLOSURE_ROLE_ARN"
        ]
        == role_arn
    )
    assert (
        validate_support_template(
            bundle.support_template,
            bundle.inputs,
        )
        is bundle.support_template
    )


def test_task11_runtime_roles_have_finite_network_key_and_fence_authority() -> None:
    """Break caught: production roles cannot execute or target the wrong fence."""

    from glm52_enforcement.support_plane import (
        support_inputs_from_mapping,
        validate_support_template,
    )

    bundle = _bundle()
    resources = bundle.support_template["Resources"]

    def statements(role_id: str) -> list[dict[str, object]]:
        return [
            statement
            for policy in resources[role_id]["Properties"]["Policies"]
            for statement in policy["PolicyDocument"]["Statement"]
        ]

    eni_actions = {
        "ec2:CreateNetworkInterface",
        "ec2:DescribeNetworkInterfaces",
        "ec2:DeleteNetworkInterface",
    }
    vpc_roles = {
        "AttestationRole",
        "LaunchAdmissionRole",
        "NumericBindingRole",
        "RetainedCancellationRole",
    }
    for role_id in vpc_roles:
        eni = next(
            item
            for item in statements(role_id)
            if set(
                [item["Action"]] if isinstance(item["Action"], str) else item["Action"]
            )
            == eni_actions
        )
        assert eni["Resource"] == "*"
        assert eni["Condition"] == {
            "StringEquals": {"aws:RequestedRegion": "us-west-2"}
        }
    for role_id in (
        "DecisionRole",
        "ClosureSessionRole",
        "FenceExecutorRole",
    ):
        assert not any(
            eni_actions
            & set(
                [item["Action"]] if isinstance(item["Action"], str) else item["Action"]
            )
            for item in statements(role_id)
        )

    for role_id in (
        "ClosureSessionRole",
        "FenceExecutorRole",
        "LaunchAdmissionRole",
    ):
        decrypt = [
            item for item in statements(role_id) if item.get("Action") == "kms:Decrypt"
        ]
        assert len(decrypt) == 1
        assert decrypt[0]["Resource"] == bundle.inputs.retained_kms_key_arn
        assert decrypt[0]["Condition"]["StringEquals"] == {
            "kms:ViaService": "s3.us-west-2.amazonaws.com"
        }
        contexts = decrypt[0]["Condition"]["StringLike"][
            "kms:EncryptionContext:aws:s3:arn"
        ]
        values = [contexts] if isinstance(contexts, str) else contexts
        assert values
        assert all(
            item.startswith(
                bundle.inputs.model_bucket_arn + "/campaigns/glm52-sky-20260724/"
            )
            for item in values
        )

    pass_role = next(
        item
        for item in statements("FenceExecutorRole")
        if item.get("Action") == "iam:PassRole"
    )
    assert pass_role == {
        "Sid": "PassExactFenceServiceRole",
        "Effect": "Allow",
        "Action": "iam:PassRole",
        "Resource": bundle.inputs.fence_service_role_arn,
        "Condition": {
            "StringEquals": {"iam:PassedToService": "cloudformation.amazonaws.com"}
        },
    }
    rendered = json.dumps(bundle.support_template, sort_keys=True)
    assert "keep-glm52-gpu-fence" not in rendered
    assert bundle.inputs.fence_stack_id in rendered
    assert bundle.inputs.fence_stack_name in rendered

    for field, mutant in (
        ("fence_stack_name", "keep-glm52-gpu-fence"),
        (
            "fence_stack_id",
            bundle.inputs.fence_stack_id.replace(
                "keep-glm52-h1g-fence", "keep-glm52-gpu-fence"
            ),
        ),
        (
            "fence_service_role_arn",
            "arn:aws:iam::246813579024:role/foreign-fence-service",
        ),
    ):
        value = _support_inputs()
        value[field] = mutant
        with pytest.raises(ValueError):
            support_inputs_from_mapping(value)

    mutations = []
    missing_eni = copy.deepcopy(bundle.support_template)
    missing_eni_statement = next(
        item
        for policy in missing_eni["Resources"]["AttestationRole"]["Properties"][
            "Policies"
        ]
        for item in policy["PolicyDocument"]["Statement"]
        if item.get("Sid") == "ManageLambdaVpcNetworkInterfacesInRegion"
    )
    missing_eni_statement["Action"].remove("ec2:CreateNetworkInterface")
    mutations.append(missing_eni)

    wildcard_kms = copy.deepcopy(bundle.support_template)
    wildcard_kms_statement = next(
        item
        for policy in wildcard_kms["Resources"]["ClosureSessionRole"]["Properties"][
            "Policies"
        ]
        for item in policy["PolicyDocument"]["Statement"]
        if item.get("Action") == "kms:Decrypt"
    )
    wildcard_kms_statement["Resource"] = "*"
    mutations.append(wildcard_kms)

    wildcard_pass_role = copy.deepcopy(bundle.support_template)
    wildcard_pass_role_statement = next(
        item
        for policy in wildcard_pass_role["Resources"]["FenceExecutorRole"][
            "Properties"
        ]["Policies"]
        for item in policy["PolicyDocument"]["Statement"]
        if item.get("Action") == "iam:PassRole"
    )
    wildcard_pass_role_statement["Resource"] = "*"
    mutations.append(wildcard_pass_role)

    stale_fence = copy.deepcopy(bundle.support_template)
    stale_fence["Resources"]["DecisionFunction"]["Properties"]["Environment"][
        "Variables"
    ]["GLM52_FENCE_STACK_ID"] = "keep-glm52-gpu-fence"
    mutations.append(stale_fence)

    for mutant in mutations:
        with pytest.raises(ValueError):
            validate_support_template(mutant, bundle.inputs)


def test_task11_fence_executor_authenticates_artifacts_without_static_cf_inventory() -> (
    None
):
    """Break caught: IAM embeds a stale template or change-set inventory."""

    from glm52_enforcement.support_plane import (
        support_inputs_from_mapping,
        validate_support_template,
    )

    bundle = _bundle()
    statements = {
        item["Sid"]: item
        for policy in bundle.support_template["Resources"]["FenceExecutorRole"][
            "Properties"
        ]["Policies"]
        for item in policy["PolicyDocument"]["Statement"]
    }
    manifest = bundle.inputs.fence_manifest_coordinate
    entries = bundle.inputs.fence_template_inventory
    object_arns = [
        bundle.inputs.model_bucket_arn + "/" + manifest["key"],
        *[
            bundle.inputs.model_bucket_arn + "/" + item["template_key"]
            for item in entries
        ],
    ]
    assert statements["ReadExactFenceInventory"]["Resource"] == object_arns
    create = statements["CreateOnlyAuthenticatedFenceChangeSets"]
    assert create["Resource"] == bundle.inputs.fence_stack_id
    assert create["Condition"] == {
        "ArnEquals": {"cloudformation:RoleArn": bundle.inputs.fence_service_role_arn},
        "ForAllValues:StringEquals": {
            "cloudformation:ResourceTypes": ["AWS::S3::BucketPolicy"]
        },
    }
    execute = statements["ExecuteOnlyAuthenticatedFenceChangeSets"]
    assert execute == {
        "Sid": "ExecuteOnlyAuthenticatedFenceChangeSets",
        "Effect": "Allow",
        "Action": [
            "cloudformation:DescribeChangeSet",
            "cloudformation:ExecuteChangeSet",
        ],
        "Resource": bundle.inputs.fence_stack_id,
    }
    assert "authorities/task11/*" not in str(statements)
    assert "authorities/fence/*" not in str(statements)
    assert all("*" not in arn for arn in object_arns)

    for mutate in (
        lambda value: value["fence_manifest_coordinate"].__setitem__(
            "version_id", "foreign-version"
        ),
        lambda value: value["fence_template_inventory"][0].__setitem__(
            "template_url", "https://foreign.example/template.json"
        ),
        lambda value: value["fence_template_inventory"][0].__setitem__(
            "version_id", "foreign-version"
        ),
        lambda value: value["fence_template_inventory"][0].__setitem__(
            "change_set_name", "foreign-change-set"
        ),
        lambda value: value["fence_template_inventory"][0].__setitem__(
            "change_set_arn",
            value["fence_template_inventory"][1]["change_set_arn"],
        ),
    ):
        value = _support_inputs()
        mutate(value)
        with pytest.raises(ValueError):
            support_inputs_from_mapping(value)

    mutant = copy.deepcopy(bundle.support_template)
    candidate = next(
        item
        for policy in mutant["Resources"]["FenceExecutorRole"]["Properties"]["Policies"]
        for item in policy["PolicyDocument"]["Statement"]
        if item.get("Sid") == "CreateOnlyAuthenticatedFenceChangeSets"
    )
    candidate["Condition"]["ArnEquals"]["cloudformation:RoleArn"] = (
        "arn:aws:iam::246813579024:role/foreign"
    )
    with pytest.raises(ValueError):
        validate_support_template(mutant, bundle.inputs)
    mutant = copy.deepcopy(bundle.support_template)
    create = next(
        item
        for policy in mutant["Resources"]["FenceExecutorRole"]["Properties"]["Policies"]
        for item in policy["PolicyDocument"]["Statement"]
        if item.get("Sid") == "CreateOnlyAuthenticatedFenceChangeSets"
    )
    create["Condition"]["ForAllValues:StringEquals"]["cloudformation:ResourceTypes"] = [
        "AWS::IAM::Role"
    ]
    with pytest.raises(ValueError):
        validate_support_template(mutant, bundle.inputs)


def test_task11_h1d_reads_scope_every_resource_capable_action() -> None:
    """Break caught: a resource-capable H1d read regains wildcard authority."""

    from glm52_enforcement.support_plane import validate_support_template

    bundle = _bundle()
    closure = bundle.support_template["Resources"]["ClosureSessionRole"]["Properties"]
    statements = [
        item
        for policy in closure["Policies"]
        for item in policy["PolicyDocument"]["Statement"]
    ]
    unscopable = {
        "lambda:ListEventSourceMappings",
        "lambda:ListFunctions",
        "events:ListRules",
        "scheduler:ListSchedules",
        "sqs:ListQueues",
        "iam:ListInstanceProfiles",
        "iam:ListRoles",
        "states:ListStateMachines",
        "ec2:DescribeIamInstanceProfileAssociations",
        "ec2:DescribeInstances",
        "ssm:DescribeInstanceInformation",
        "cloudwatch:DescribeAlarms",
        "cloudwatch:GetMetricData",
        "logs:DescribeLogGroups",
    }
    resource_capable = {
        "cloudformation:DescribeStacks",
        "cloudformation:ListStackResources",
        "cloudformation:GetTemplate",
        "lambda:GetFunction",
        "lambda:GetPolicy",
        "iam:GetRole",
        "iam:GetInstanceProfile",
        "iam:GetRolePolicy",
        "iam:ListAttachedRolePolicies",
        "iam:ListInstanceProfilesForRole",
        "iam:ListRolePolicies",
        "states:DescribeStateMachine",
        "states:ListStateMachineVersions",
        "events:ListTargetsByRule",
        "events:DescribeRule",
        "scheduler:GetSchedule",
        "sqs:GetQueueAttributes",
        "sns:ListSubscriptionsByTopic",
    }
    wildcard_actions = {
        action
        for item in statements
        if item.get("Resource") == "*"
        for action in (
            [item["Action"]] if isinstance(item["Action"], str) else item["Action"]
        )
    }
    assert wildcard_actions & resource_capable == set()
    assert unscopable <= wildcard_actions
    retired_organization_reads = {
        "organizations:DescribeAccount",
        "organizations:DescribePolicy",
        "organizations:ListPoliciesForTarget",
        "organizations:ListTargetsForPolicy",
    }
    all_actions = {
        action
        for item in statements
        for action in (
            [item["Action"]] if isinstance(item["Action"], str) else item["Action"]
        )
    }
    assert all_actions.isdisjoint(retired_organization_reads)
    for action in resource_capable:
        owners = [
            item
            for item in statements
            if action
            in ([item["Action"]] if isinstance(item["Action"], str) else item["Action"])
        ]
        assert owners
        assert all(item["Resource"] != "*" for item in owners)

    mutant = copy.deepcopy(bundle.support_template)
    unscoped = next(
        item
        for policy in mutant["Resources"]["ClosureSessionRole"]["Properties"][
            "Policies"
        ]
        for item in policy["PolicyDocument"]["Statement"]
        if item.get("Sid") == "ReadOnlyUnscopableH1dIndexes"
    )
    unscoped["Action"].append("lambda:GetFunction")
    with pytest.raises(ValueError):
        validate_support_template(mutant, bundle.inputs)


def test_fence_stabilization_timeout_trust_clients_and_iam_are_exact() -> None:
    """The 12-second fence can only inspect and probe exact stable publishers."""

    from glm52_enforcement.support_plane import (
        validate_support_runtime_inventory,
        validate_support_template,
    )

    bundle = _bundle()
    resources = bundle.support_template["Resources"]
    fence = resources["FenceExecutorFunction"]["Properties"]
    assert fence["Timeout"] == 12
    assert fence["Environment"]["Variables"]["GLM52_FENCE_EXECUTOR_ROLE_ARN"] == {
        "Fn::GetAtt": ["FenceExecutorRole", "Arn"]
    }
    assert "GLM52_FENCE_SUCCESSOR_ROLE_ARN" not in fence["Environment"]["Variables"]
    assert "GLM52_FENCE_SERVICE_ROLE_ARN" not in fence["Environment"]["Variables"]
    statements = resources["FenceExecutorRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    actions = {
        action
        for statement in statements
        for action in (
            [statement["Action"]]
            if isinstance(statement["Action"], str)
            else statement["Action"]
        )
    }
    assert {
        "iam:GetRole",
        "iam:GetRolePolicy",
        "iam:ListAttachedRolePolicies",
        "iam:ListRolePolicies",
        "iam:GetPolicy",
        "iam:GetPolicyVersion",
    } <= actions
    assert "sts:AssumeRole" not in actions
    assert "cloudformation:DetectStackDrift" not in actions
    for name in (
        "SourceGpuSpend",
        "SourceSubmissionIntent",
        "SourceControllerBaseline",
        "SourceControlPlaneReadiness",
        "SourceSubmissionAcquisition",
    ):
        role = resources[name + "Role"]["Properties"]
        assert role["RoleName"] == (
            "keep-glm52-h1g-"
            + re.sub(
                r"([a-z0-9])([A-Z])",
                r"\1-\2",
                name,
            ).lower()
        )
        trust = role["AssumeRolePolicyDocument"]["Statement"]
        assert trust == [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ]

    mutant = copy.deepcopy(bundle.support_template)
    mutant["Resources"]["FenceExecutorFunction"]["Properties"]["Timeout"] = 30
    with pytest.raises(ValueError):
        validate_support_runtime_inventory(mutant, bundle.inputs)

    iam_mutant = copy.deepcopy(bundle.support_template)
    iam_statement = next(
        statement
        for statement in iam_mutant["Resources"]["FenceExecutorRole"]["Properties"][
            "Policies"
        ][0]["PolicyDocument"]["Statement"]
        if statement.get("Sid") == "ReadExactFenceStabilizationRoles"
    )
    iam_statement["Action"].remove("iam:GetRolePolicy")
    with pytest.raises(ValueError, match="stabilization IAM"):
        validate_support_template(iam_mutant, bundle.inputs)

    trust_mutant = copy.deepcopy(bundle.support_template)
    trust_mutant["Resources"]["SourceGpuSpendRole"]["Properties"][
        "AssumeRolePolicyDocument"
    ]["Statement"].pop()
    with pytest.raises(ValueError, match="publisher trust"):
        validate_support_template(trust_mutant, bundle.inputs)


def _secret_version_observations() -> dict[str, object]:
    purposes = {
        "RawSkyTokenSecret": "raw-sky-token",
        "SkyBootstrapSecret": "sky-bootstrap-hash",
        "AttestationClientTlsSecret": "attestation-client-tls",
        "LaunchAdmissionClientTlsSecret": "launch-admission-client-tls",
        "NumericBindingClientTlsSecret": "numeric-binding-client-tls",
        "RetainedCancellationClientTlsSecret": "retained-cancellation-client-tls",
        "CombinedHostTlsSecret": "combined-host-server-tls",
        "CaIssuanceSecret": "ca-issuance",
    }
    return {
        logical_id: {
            "secret_arn": (
                "arn:aws:secretsmanager:us-west-2:246813579024:"
                f"secret:/keep/glm52/glm52-sky-20260724/"
                f"act-20260728-0001/{purpose}-fixture"
            ),
            "version_id": f"{index + 1:x}" * 32,
            "version_stage": f"h1g-act-20260728-0001-{purpose}-v1",
            "list_versions_identity_sha256": f"{index + 1:x}" * 64,
            "versions": (
                {
                    "version_id": f"{index + 1:x}" * 32,
                    "version_stages": (f"h1g-act-20260728-0001-{purpose}-v1",),
                },
            ),
        }
        for index, (logical_id, purpose) in enumerate(purposes.items())
    }


def test_secret_inventory_requires_real_one_version_readback_and_exact_runtime_result() -> (
    None
):
    """Break caught: tags claim a version or a caller accepts another VersionId."""

    from glm52_enforcement.support_plane import (
        SECRET_LOGICAL_IDS,
        secret_inventory,
        validate_secret_read_result,
    )

    bundle = _bundle()
    observations = _secret_version_observations()
    inventory = secret_inventory(
        bundle.support_template,
        bundle.inputs,
        version_observations=observations,
    )
    assert tuple(inventory) == SECRET_LOGICAL_IDS
    assert all(item["authorized_version_count"] == 1 for item in inventory.values())
    raw = inventory["RawSkyTokenSecret"]
    assert (
        validate_secret_read_result(
            inventory=inventory,
            caller="ATTESTATION",
            secret_logical_id="RawSkyTokenSecret",
            requested_version_id=raw["version_id"],
            requested_version_stage=raw["version_stage"],
            returned_secret_arn=raw["secret_arn"],
            returned_version_id=raw["version_id"],
            returned_version_stages=(raw["version_stage"],),
        )
        is True
    )
    with pytest.raises(ValueError):
        validate_secret_read_result(
            inventory=inventory,
            caller="ATTESTATION",
            secret_logical_id="RawSkyTokenSecret",
            requested_version_id="f" * 32,
            requested_version_stage=raw["version_stage"],
            returned_secret_arn=raw["secret_arn"],
            returned_version_id="f" * 32,
            returned_version_stages=(raw["version_stage"],),
        )
    with pytest.raises(ValueError):
        validate_secret_read_result(
            inventory=inventory,
            caller="DECISION",
            secret_logical_id="RawSkyTokenSecret",
            requested_version_id=raw["version_id"],
            requested_version_stage=raw["version_stage"],
            returned_secret_arn=raw["secret_arn"],
            returned_version_id=raw["version_id"],
            returned_version_stages=(raw["version_stage"],),
        )

    extra = copy.deepcopy(observations)
    extra["RawSkyTokenSecret"]["versions"] += (
        {
            "version_id": "f" * 32,
            "version_stages": ("h1g-act-20260728-0001-raw-sky-token-v1",),
        },
    )
    with pytest.raises(ValueError):
        secret_inventory(
            bundle.support_template,
            bundle.inputs,
            version_observations=extra,
        )

    resources = bundle.support_template["Resources"]
    variables = resources["AttestationFunction"]["Properties"]["Environment"][
        "Variables"
    ]
    assert variables["GLM52_RAW_SKY_TOKEN_VERSION_ID"] == {
        "Fn::GetAtt": [
            "SkyBootstrapCustomResource",
            "RawSkyTokenSecretVersionId",
        ]
    }
    assert variables["GLM52_RAW_SKY_TOKEN_VERSION_STAGE"] == (
        "h1g-act-20260728-0001-raw-sky-token-v1"
    )


def test_template_wires_exact_secret_targets_handler_writes_and_host_versions() -> None:
    """Break caught: the template cannot create raw/bootstrap or bind host reads."""

    bundle = _bundle()
    resources = bundle.support_template["Resources"]
    expected = {
        "SkyBootstrapCustomResource": (
            "RawSkyTokenSecret",
            "SkyBootstrapSecret",
        ),
        "TlsBundleCustomResource": (
            "AttestationClientTlsSecret",
            "LaunchAdmissionClientTlsSecret",
            "NumericBindingClientTlsSecret",
            "RetainedCancellationClientTlsSecret",
            "CombinedHostTlsSecret",
            "CaIssuanceSecret",
        ),
    }
    purposes = {
        item: _secret_version_observations()[item]["version_stage"]
        for item in _secret_version_observations()
    }
    for custom_resource, logical_ids in expected.items():
        targets = resources[custom_resource]["Properties"]["SecretTargets"]
        assert tuple(targets) == logical_ids
        for logical_id in logical_ids:
            assert targets[logical_id] == {
                "SecretId": {"Ref": logical_id},
                "VersionStage": purposes[logical_id],
            }

    statements = resources["TlsHandlerRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    put_statements = [
        statement
        for statement in statements
        if statement["Action"] == "secretsmanager:PutSecretValue"
    ]
    assert {
        (
            statement["Resource"]["Ref"],
            statement["Condition"]["StringEquals"]["secretsmanager:VersionStage"],
        )
        for statement in put_statements
    } == set(purposes.items())
    create_grant = next(
        statement
        for statement in statements
        if statement["Sid"] == "CreateExactTlsHandlerGrant"
    )
    assert create_grant == {
        "Sid": "CreateExactTlsHandlerGrant",
        "Effect": "Allow",
        "Action": "kms:CreateGrant",
        "Resource": _support_inputs()["retained_kms_key_arn"],
        "Condition": {
            "StringEquals": {
                "kms:GranteePrincipal": (
                    "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-tls-handler"
                ),
                "kms:RetiringPrincipal": (
                    "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-deletion"
                ),
                "kms:EncryptionContext:RunId": "glm52-sky-20260724",
                "kms:EncryptionContext:ActivationId": (
                    _support_inputs()["activation_id"]
                ),
            },
            "ForAllValues:StringEquals": {
                "kms:GrantOperations": [
                    "Decrypt",
                    "Encrypt",
                    "GenerateDataKey",
                ]
            },
        },
    }
    assert next(
        statement
        for statement in statements
        if statement["Sid"] == "ReconcileExactTlsHandlerGrant"
    ) == {
        "Sid": "ReconcileExactTlsHandlerGrant",
        "Effect": "Allow",
        "Action": [
            "kms:ListGrants",
            "kms:RetireGrant",
            "kms:RevokeGrant",
        ],
        "Resource": _support_inputs()["retained_kms_key_arn"],
    }
    assert next(
        statement
        for statement in statements
        if statement["Sid"] == "PersistExactDirectGrantEvidence"
    ) == {
        "Sid": "PersistExactDirectGrantEvidence",
        "Effect": "Allow",
        "Action": [
            "dynamodb:GetItem",
            "dynamodb:PutItem",
        ],
        "Resource": _support_inputs()["ledger_table_arn"],
    }
    assert (
        resources["TlsHandlerFunction"]["Properties"]["Environment"]["Variables"][
            "GLM52_LEDGER_TABLE_NAME"
        ]
        == _support_inputs()["ledger_table_name"]
    )
    assert (
        resources["TlsHandlerFunction"]["Properties"]["Environment"]["Variables"][
            "GLM52_RETAINED_KMS_KEY_ARN"
        ]
        == _support_inputs()["retained_kms_key_arn"]
    )

    host_user_data = resources["CombinedHost"]["Properties"]["UserData"]
    assert tuple(host_user_data) == ("Fn::Base64",)
    sub = host_user_data["Fn::Base64"]["Fn::Sub"]
    assert isinstance(sub, list) and len(sub) == 2
    substitutions = sub[1]
    assert substitutions["Glm52SkyBootstrapVersionId"] == {
        "Fn::GetAtt": [
            "SkyBootstrapCustomResource",
            "SkyBootstrapSecretVersionId",
        ]
    }
    assert substitutions["Glm52CombinedHostTlsVersionId"] == {
        "Fn::GetAtt": [
            "TlsBundleCustomResource",
            "CombinedHostTlsSecretVersionId",
        ]
    }
    assert "PRIVATE KEY" not in sub[0]
    assert "SKY_TOKEN=" not in sub[0]


def test_tls_create_grant_rejects_missing_or_foreign_activation_context() -> None:
    """Break caught: IAM permits a direct grant not bound to this activation."""

    from copy import deepcopy

    from glm52_enforcement.support_plane import (
        validate_secret_reader_partition,
    )

    bundle = _bundle()
    for replacement in (None, "act-foreign-0001"):
        mutant = deepcopy(bundle.support_template)
        statements = mutant["Resources"]["TlsHandlerRole"]["Properties"]["Policies"][0][
            "PolicyDocument"
        ]["Statement"]
        grant = next(
            statement
            for statement in statements
            if statement["Sid"] == "CreateExactTlsHandlerGrant"
        )
        context = grant["Condition"]["StringEquals"]
        if replacement is None:
            del context["kms:EncryptionContext:ActivationId"]
        else:
            context["kms:EncryptionContext:ActivationId"] = replacement
        with pytest.raises(
            ValueError, match="TLS handler secret authority is not exact"
        ):
            validate_secret_reader_partition(mutant, bundle.inputs)


def test_retained_kms_grant_provenance_restores_exact_baseline() -> None:
    """Break caught: unknown/overbroad/recovery grant survives support deletion."""

    from glm52_enforcement.support_plane import validate_kms_grant_inventory

    bundle = _bundle()
    assert (
        bundle.support_template["Resources"]["TlsHandlerRole"]["Properties"]["RoleName"]
        == "keep-glm52-h1g-support-tls-handler"
    )
    baseline = (
        {
            "grant_id": "baseline-grant-01",
            "name": "retained-ledger",
            "grantee": "dynamodb.us-west-2.amazonaws.com",
            "retiring_principal": (
                "arn:aws:iam::246813579024:role/keep-glm52-retained-kms-cleanup"
            ),
            "operations": ("Decrypt", "Encrypt"),
            "encryption_context": (
                ("RunId", "glm52-sky-20260724"),
                ("Purpose", "retained-ledger"),
            ),
            "revocable": True,
        },
    )
    created = (
        {
            "request_identity_sha256": "5" * 64,
            "response_identity_sha256": "6" * 64,
            "grant_id": "h1g-grant-01",
            "name": "h1g-act-20260728-0001-secrets",
            "grantee": (
                "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-tls-handler"
            ),
            "retiring_principal": (
                "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-deletion"
            ),
            "operations": ("Decrypt", "Encrypt", "GenerateDataKey"),
            "encryption_context": (
                ("RunId", "glm52-sky-20260724"),
                ("ActivationId", "act-20260728-0001"),
            ),
            "revocable": True,
        },
    )
    service_created = (
        {
            "baseline_identity_sha256": "7" * 64,
            "diff_identity_sha256": "8" * 64,
            "list_grants_identity_sha256": "9" * 64,
            "cloudtrail_event_identity_sha256": "a" * 64,
            "cloudtrail_request_identity_sha256": "b" * 64,
            "cloudtrail_response_identity_sha256": "c" * 64,
            "grant_id": "service-grant-01",
            "name": "h1g-act-20260728-0001-ebs-data",
            "grantee": "ec2.us-west-2.amazonaws.com",
            "retiring_principal": (
                "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-deletion"
            ),
            "operations": (
                "Decrypt",
                "Encrypt",
                "GenerateDataKeyWithoutPlaintext",
            ),
            "originating_resource": "CombinedHostDataVolume",
            "originating_resource_identity": "vol-00000000000000002",
            "originating_service": "ec2.us-west-2.amazonaws.com",
            "encryption_context": (
                ("RunId", "glm52-sky-20260724"),
                ("ActivationId", "act-20260728-0001"),
                ("aws:ebs:id", "vol-00000000000000002"),
            ),
            "revocable": True,
            "settling_window_seconds": 300,
        },
    )
    created_active = {
        key: created[0][key]
        for key in (
            "grant_id",
            "name",
            "grantee",
            "retiring_principal",
            "operations",
            "encryption_context",
            "revocable",
        )
    }
    service_active = {
        key: service_created[0][key]
        for key in (
            "grant_id",
            "name",
            "grantee",
            "retiring_principal",
            "operations",
            "encryption_context",
            "revocable",
        )
    }
    assert (
        validate_kms_grant_inventory(
            inputs=bundle.inputs,
            baseline=baseline,
            active=baseline + (created_active, service_active),
            h1g_created=created,
            service_created=service_created,
            restored=baseline,
        )
        is True
    )

    overbroad = copy.deepcopy(created)
    overbroad[0]["operations"] = created[0]["operations"] + ("CreateGrant",)
    with pytest.raises(ValueError):
        validate_kms_grant_inventory(
            inputs=bundle.inputs,
            baseline=baseline,
            active=baseline + (created_active, service_active),
            h1g_created=overbroad,
            service_created=service_created,
            restored=baseline,
        )

    with pytest.raises(ValueError):
        validate_kms_grant_inventory(
            inputs=bundle.inputs,
            baseline=baseline,
            active=baseline + (created_active, service_active),
            h1g_created=created,
            service_created=service_created,
            restored=baseline + (created_active,),
        )


def test_kms_active_inventory_is_complete_attributed_revocable_and_exact() -> None:
    """Break caught: service grants are omitted or unknown/overbroad grants survive."""

    from glm52_enforcement.support_plane import validate_kms_grant_inventory

    bundle = _bundle()
    baseline_grant = {
        "grant_id": "baseline-grant-01",
        "name": "retained-ledger",
        "grantee": "dynamodb.us-west-2.amazonaws.com",
        "retiring_principal": (
            "arn:aws:iam::246813579024:role/keep-glm52-retained-kms-cleanup"
        ),
        "operations": ("Decrypt", "Encrypt"),
        "encryption_context": (
            ("RunId", "glm52-sky-20260724"),
            ("Purpose", "retained-ledger"),
        ),
        "revocable": True,
    }
    direct_grant = {
        "request_identity_sha256": "1" * 64,
        "response_identity_sha256": "2" * 64,
        "grant_id": "h1g-grant-01",
        "name": "h1g-act-20260728-0001-secrets",
        "grantee": (
            "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-tls-handler"
        ),
        "retiring_principal": (
            "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-deletion"
        ),
        "operations": ("Decrypt", "Encrypt", "GenerateDataKey"),
        "encryption_context": (
            ("RunId", "glm52-sky-20260724"),
            ("ActivationId", "act-20260728-0001"),
        ),
        "revocable": True,
    }
    service_grant = {
        "baseline_identity_sha256": "3" * 64,
        "diff_identity_sha256": "4" * 64,
        "list_grants_identity_sha256": "5" * 64,
        "cloudtrail_event_identity_sha256": "6" * 64,
        "cloudtrail_request_identity_sha256": "7" * 64,
        "cloudtrail_response_identity_sha256": "8" * 64,
        "grant_id": "service-grant-01",
        "name": "h1g-act-20260728-0001-ebs-data",
        "grantee": "ec2.us-west-2.amazonaws.com",
        "retiring_principal": (
            "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-deletion"
        ),
        "operations": (
            "Decrypt",
            "Encrypt",
            "GenerateDataKeyWithoutPlaintext",
        ),
        "encryption_context": (
            ("RunId", "glm52-sky-20260724"),
            ("ActivationId", "act-20260728-0001"),
            ("aws:ebs:id", "vol-00000000000000002"),
        ),
        "revocable": True,
        "originating_resource": "CombinedHostDataVolume",
        "originating_resource_identity": "vol-00000000000000002",
        "originating_service": "ec2.us-west-2.amazonaws.com",
        "settling_window_seconds": 300,
    }
    direct_active = {
        key: direct_grant[key]
        for key in (
            "grant_id",
            "name",
            "grantee",
            "retiring_principal",
            "operations",
            "encryption_context",
            "revocable",
        )
    }
    service_active = {
        key: service_grant[key]
        for key in (
            "grant_id",
            "name",
            "grantee",
            "retiring_principal",
            "operations",
            "encryption_context",
            "revocable",
        )
    }
    baseline = (baseline_grant,)
    active = (baseline_grant, direct_active, service_active)
    assert (
        validate_kms_grant_inventory(
            inputs=bundle.inputs,
            baseline=baseline,
            active=active,
            h1g_created=(direct_grant,),
            service_created=(service_grant,),
            restored=baseline,
        )
        is True
    )

    mutants = (
        (baseline + (direct_active,), (service_grant,)),
        (active + (service_active,), (service_grant,)),
        (
            active[:-1]
            + (
                {
                    **service_active,
                    "operations": service_active["operations"] + ("CreateGrant",),
                },
            ),
            (
                {
                    **service_grant,
                    "operations": service_grant["operations"] + ("CreateGrant",),
                },
            ),
        ),
        (
            active[:-1] + ({**service_active, "revocable": False},),
            ({**service_grant, "revocable": False},),
        ),
    )
    for active_mutant, service_mutant in mutants:
        with pytest.raises(ValueError):
            validate_kms_grant_inventory(
                inputs=bundle.inputs,
                baseline=baseline,
                active=active_mutant,
                h1g_created=(direct_grant,),
                service_created=service_mutant,
                restored=baseline,
            )


def test_kms_rejects_reviewer_foreign_unbound_grants() -> None:
    """Break caught: foreign activation/arbitrary principal/unbound EBS grant passes."""

    from glm52_enforcement.support_plane import validate_kms_grant_inventory

    bundle = _bundle()
    direct = {
        "request_identity_sha256": "1" * 64,
        "response_identity_sha256": "2" * 64,
        "grant_id": "foreign-direct-01",
        "name": "arbitrary-grant-name",
        "grantee": ("arn:aws:iam::246813579024:role/arbitrary-h1g-principal"),
        "retiring_principal": (
            "arn:aws:iam::246813579024:role/arbitrary-retiring-principal"
        ),
        "operations": ("Decrypt", "Encrypt", "GenerateDataKey"),
        "encryption_context": (
            ("RunId", "glm52-sky-20260724"),
            ("ActivationId", "act-foreign"),
        ),
        "revocable": True,
    }
    service = {
        "baseline_identity_sha256": "3" * 64,
        "diff_identity_sha256": "4" * 64,
        "list_grants_identity_sha256": "5" * 64,
        "cloudtrail_event_identity_sha256": "6" * 64,
        "cloudtrail_request_identity_sha256": "7" * 64,
        "cloudtrail_response_identity_sha256": "8" * 64,
        "grant_id": "foreign-service-01",
        "name": "arbitrary-service-grant",
        "grantee": "ec2.us-west-2.amazonaws.com",
        "retiring_principal": (
            "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-deletion"
        ),
        "operations": (
            "Decrypt",
            "Encrypt",
            "GenerateDataKeyWithoutPlaintext",
        ),
        "encryption_context": (
            ("RunId", "glm52-sky-20260724"),
            ("ActivationId", "act-foreign"),
        ),
        "revocable": True,
        "originating_resource": "CombinedHostDataVolume",
        "originating_resource_identity": "vol-0123456789abcdef0",
        "originating_service": "ec2.us-west-2.amazonaws.com",
        "settling_window_seconds": 300,
    }
    projection_fields = (
        "grant_id",
        "name",
        "grantee",
        "retiring_principal",
        "operations",
        "encryption_context",
        "revocable",
    )
    active = (
        {field: direct[field] for field in projection_fields},
        {field: service[field] for field in projection_fields},
    )
    with pytest.raises(ValueError):
        validate_kms_grant_inventory(
            inputs=bundle.inputs,
            baseline=(),
            active=active,
            h1g_created=(direct,),
            service_created=(service,),
            restored=(),
        )


def test_kms_service_grant_rejects_well_formed_foreign_physical_id() -> None:
    """Break caught: a regex-valid foreign volume is accepted as this stack's."""

    from glm52_enforcement.support_plane import validate_kms_grant_inventory

    bundle = _bundle()
    direct = {
        "request_identity_sha256": "1" * 64,
        "response_identity_sha256": "2" * 64,
        "grant_id": "direct-grant-01",
        "name": "h1g-act-20260728-0001-secrets",
        "grantee": (
            "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-tls-handler"
        ),
        "retiring_principal": (
            "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-deletion"
        ),
        "operations": ("Decrypt", "Encrypt", "GenerateDataKey"),
        "encryption_context": (
            ("RunId", "glm52-sky-20260724"),
            ("ActivationId", "act-20260728-0001"),
        ),
        "revocable": True,
    }
    service = {
        "baseline_identity_sha256": "3" * 64,
        "diff_identity_sha256": "4" * 64,
        "list_grants_identity_sha256": "5" * 64,
        "cloudtrail_event_identity_sha256": "6" * 64,
        "cloudtrail_request_identity_sha256": "7" * 64,
        "cloudtrail_response_identity_sha256": "8" * 64,
        "grant_id": "service-grant-01",
        "name": "h1g-act-20260728-0001-ebs-data",
        "grantee": "ec2.us-west-2.amazonaws.com",
        "retiring_principal": (
            "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-deletion"
        ),
        "operations": (
            "Decrypt",
            "Encrypt",
            "GenerateDataKeyWithoutPlaintext",
        ),
        "encryption_context": (
            ("RunId", "glm52-sky-20260724"),
            ("ActivationId", "act-20260728-0001"),
            ("aws:ebs:id", "vol-fffffffffffffffff"),
        ),
        "revocable": True,
        "originating_resource": "CombinedHostDataVolume",
        "originating_resource_identity": "vol-fffffffffffffffff",
        "originating_service": "ec2.us-west-2.amazonaws.com",
        "settling_window_seconds": 300,
    }
    projection = (
        "grant_id",
        "name",
        "grantee",
        "retiring_principal",
        "operations",
        "encryption_context",
        "revocable",
    )
    active = (
        {field: direct[field] for field in projection},
        {field: service[field] for field in projection},
    )
    with pytest.raises(ValueError):
        validate_kms_grant_inventory(
            inputs=bundle.inputs,
            baseline=(),
            active=active,
            h1g_created=(direct,),
            service_created=(service,),
            restored=(),
        )


def test_kms_service_grants_bind_volume_snapshot_and_secret_readbacks() -> None:
    """Break caught: one KMS resource family is only grammar-bound."""

    from glm52_enforcement.support_plane import validate_kms_grant_inventory

    bundle = _bundle()
    direct = {
        "request_identity_sha256": "1" * 64,
        "response_identity_sha256": "2" * 64,
        "grant_id": "direct-grant-01",
        "name": "h1g-act-20260728-0001-secrets",
        "grantee": (
            "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-tls-handler"
        ),
        "retiring_principal": (
            "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-deletion"
        ),
        "operations": ("Decrypt", "Encrypt", "GenerateDataKey"),
        "encryption_context": (
            ("RunId", "glm52-sky-20260724"),
            ("ActivationId", "act-20260728-0001"),
        ),
        "revocable": True,
    }
    projection = (
        "grant_id",
        "name",
        "grantee",
        "retiring_principal",
        "operations",
        "encryption_context",
        "revocable",
    )
    direct_active = {field: direct[field] for field in projection}
    cases = (
        (
            "CombinedHostDataVolume",
            "vol-00000000000000002",
            "ebs-data",
            "ec2.us-west-2.amazonaws.com",
            "aws:ebs:id",
            ("Decrypt", "Encrypt", "GenerateDataKeyWithoutPlaintext"),
            "vol-fffffffffffffffff",
        ),
        (
            "CombinedHostDataVolumeDeletionSnapshot",
            "snap-00000000000000016",
            "ebs-data-deletion-snapshot",
            "ec2.us-west-2.amazonaws.com",
            "aws:ebs:id",
            ("Decrypt", "Encrypt", "GenerateDataKeyWithoutPlaintext"),
            "snap-fffffffffffffffff",
        ),
        (
            "RawSkyTokenSecret",
            (
                "arn:aws:secretsmanager:us-west-2:246813579024:secret:"
                "/keep/glm52/glm52-sky-20260724/act-20260728-0001/"
                "raw-sky-token-ABC123"
            ),
            "raw-sky-token",
            "secretsmanager.us-west-2.amazonaws.com",
            "SecretARN",
            ("Decrypt", "Encrypt", "GenerateDataKey"),
            (
                "arn:aws:secretsmanager:us-west-2:246813579024:secret:"
                "/keep/glm52/glm52-sky-20260724/act-20260728-0001/"
                "raw-sky-token-FFFFFF"
            ),
        ),
    )
    for (
        logical_id,
        identity,
        suffix,
        service_principal,
        context_key,
        operations,
        foreign_identity,
    ) in cases:
        grant = {
            "baseline_identity_sha256": "3" * 64,
            "diff_identity_sha256": "4" * 64,
            "list_grants_identity_sha256": "5" * 64,
            "cloudtrail_event_identity_sha256": "6" * 64,
            "cloudtrail_request_identity_sha256": "7" * 64,
            "cloudtrail_response_identity_sha256": "8" * 64,
            "grant_id": f"service-{suffix}",
            "name": f"h1g-act-20260728-0001-{suffix}",
            "grantee": service_principal,
            "retiring_principal": (
                "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-deletion"
            ),
            "operations": operations,
            "encryption_context": (
                ("RunId", "glm52-sky-20260724"),
                ("ActivationId", "act-20260728-0001"),
                (context_key, identity),
            ),
            "revocable": True,
            "originating_resource": logical_id,
            "originating_resource_identity": identity,
            "originating_service": service_principal,
            "settling_window_seconds": 300,
        }
        service_active = {field: grant[field] for field in projection}
        assert (
            validate_kms_grant_inventory(
                inputs=bundle.inputs,
                baseline=(),
                active=(direct_active, service_active),
                h1g_created=(direct,),
                service_created=(grant,),
                restored=(),
            )
            is True
        )
        foreign = copy.deepcopy(grant)
        foreign["originating_resource_identity"] = foreign_identity
        foreign["encryption_context"] = (
            ("RunId", "glm52-sky-20260724"),
            ("ActivationId", "act-20260728-0001"),
            (context_key, foreign_identity),
        )
        foreign_active = {field: foreign[field] for field in projection}
        with pytest.raises(ValueError):
            validate_kms_grant_inventory(
                inputs=bundle.inputs,
                baseline=(),
                active=(direct_active, foreign_active),
                h1g_created=(direct,),
                service_created=(foreign,),
                restored=(),
            )


def test_generated_manifest_carries_closed_kms_grant_provenance_contract() -> None:
    """Break caught: the manifest says only pending and omits proof obligations."""

    bundle = _bundle()
    contract = bundle.manifest["kms_grant_provenance_contract"]
    assert contract == {
        "state": "REQUIRES_AUTHENTICATED_POSTCREATE_BINDING",
        "active_inventory_equality": ("baseline_plus_h1g_created_plus_service_created"),
        "final_restoration_equality": "restored_equals_frozen_baseline",
        "direct_required_fields": [
            "request_identity_sha256",
            "response_identity_sha256",
            "grant_id",
            "name",
            "grantee",
            "retiring_principal",
            "operations",
            "encryption_context",
            "revocable",
        ],
        "service_required_fields": [
            "baseline_identity_sha256",
            "diff_identity_sha256",
            "list_grants_identity_sha256",
            "cloudtrail_event_identity_sha256",
            "cloudtrail_request_identity_sha256",
            "cloudtrail_response_identity_sha256",
            "grant_id",
            "name",
            "grantee",
            "retiring_principal",
            "operations",
            "encryption_context",
            "revocable",
            "originating_resource",
            "originating_resource_identity",
            "originating_service",
            "settling_window_seconds",
        ],
        "reject": [
            "unknown",
            "missing",
            "duplicate",
            "overbroad",
            "unattributed",
            "unrevocable",
            "snapshot-recovery",
        ],
    }
    assert "kms_grant_evidence_state" not in bundle.manifest


def test_runtime_functions_versions_vpc_partition_logs_alarms_and_async_dlq_are_exact() -> (
    None
):
    """Break caught: a sensitive function gets NAT/host access or unversioned invoke."""

    from glm52_enforcement.support_plane import (
        RUNTIME_FUNCTIONS,
        validate_support_runtime_inventory,
    )

    bundle = _bundle()
    resources = bundle.support_template["Resources"]
    assert (
        validate_support_runtime_inventory(bundle.support_template, bundle.inputs)
        is True
    )
    assert tuple(RUNTIME_FUNCTIONS) == (
        "BudgetGate",
        "RehearsalCollector",
        "RehearsalProbe",
        "Decision",
        "SourceGpuSpend",
        "SourceSubmissionIntent",
        "SourceControllerBaseline",
        "SourceControlPlaneReadiness",
        "SourceSubmissionAcquisition",
        "FenceExecutor",
        "FenceSuccessor",
        "ClaimWriter",
        "DecisionWriter",
        "TerminalV1Writer",
        "ClosureHandoff",
        "Attestation",
        "LaunchAdmission",
        "NumericBinding",
        "RetainedCancellation",
        "SupportDeadline",
    )
    for name, contract in RUNTIME_FUNCTIONS.items():
        function = resources[f"{name}Function"]["Properties"]
        assert function["ReservedConcurrentExecutions"] == (
            5 if name in {"RehearsalCollector", "RehearsalProbe"} else 1
        )
        assert resources[f"{name}Version"]["Type"] == "AWS::Lambda::Version"
        assert resources[f"{name}LogGroup"]["Properties"]["RetentionInDays"] == 14
        assert resources[f"{name}ErrorAlarm"]["Type"] == "AWS::CloudWatch::Alarm"
        if contract["vpc_attached"]:
            assert function["VpcConfig"]["SubnetIds"] == [
                {"Ref": "PrimaryIsolatedSubnet"},
                {"Ref": "AlternateIsolatedSubnet"},
            ]
            assert function["VpcConfig"]["SecurityGroupIds"] == [
                {"Ref": contract["client_security_group"]}
            ]
        else:
            assert "VpcConfig" not in function
    assert "VpcConfig" not in resources["DecisionFunction"]["Properties"]
    assert "VpcConfig" not in resources["BudgetGateFunction"]["Properties"]
    assert "VpcConfig" not in resources["RehearsalCollectorFunction"]["Properties"]
    assert "VpcConfig" not in resources["TlsHandlerFunction"]["Properties"]
    assert resources["SupportAsyncDlq"]["Type"] == "AWS::SQS::Queue"
    assert resources["SupportDeadlineEventInvokeConfig"]["Properties"][
        "DestinationConfig"
    ]["OnFailure"]["Destination"] == {"Fn::GetAtt": ["SupportAsyncDlq", "Arn"]}
    assert "DeadLetterConfig" not in resources["DecisionFunction"]["Properties"]


def test_rehearsal_collector_is_exact_isolated_and_mutation_closed() -> None:
    """Break caught: deployed measurements can reach or be read by production."""

    from glm52_enforcement.support_plane import (
        validate_support_runtime_inventory,
        validate_task11_runtime_role_authority,
    )

    bundle = _bundle()
    template = bundle.support_template
    resources = template["Resources"]
    collector = resources["RehearsalCollectorFunction"]["Properties"]
    assert collector["Handler"] == ("support_rehearsal_collector_handler.main")
    assert collector["ReservedConcurrentExecutions"] == 5
    assert "VpcConfig" not in collector
    decision_environment = resources["DecisionFunction"]["Properties"]["Environment"][
        "Variables"
    ]
    assert collector["Environment"]["Variables"] == {
        **decision_environment,
        "GLM52_DECISION_VERSION_ARN": {"Ref": "DecisionVersion"},
        "GLM52_REHEARSAL_CONTROLLER_ROLE_ARN": {
            "Fn::GetAtt": ["FenceExecutorRole", "Arn"]
        },
        "GLM52_REHEARSAL_EXECUTOR_ROLE_ARN": {
            "Fn::GetAtt": ["RehearsalExecutorRole", "Arn"]
        },
        "GLM52_REHEARSAL_BUCKET": {"Ref": "RehearsalBucket"},
        "GLM52_EXPECTED_BUCKET_OWNER": "246813579024",
        "GLM52_REHEARSAL_PROBE_VERSION_ARN": {"Ref": "RehearsalProbeVersion"},
    }
    role = resources["RehearsalCollectorRole"]["Properties"]
    statements = [
        statement
        for policy in role["Policies"]
        for statement in policy["PolicyDocument"]["Statement"]
    ]
    actions = {
        action
        for statement in statements
        if statement["Effect"] == "Allow"
        for action in (
            [statement["Action"]]
            if isinstance(statement["Action"], str)
            else statement["Action"]
        )
    }
    assert {
        "cloudformation:DescribeStacks",
        "cloudformation:ListChangeSets",
        "ec2:DescribeInstances",
        "iam:GetRole",
        "lambda:GetFunction",
        "s3:GetObjectVersion",
        "s3:ListBucketVersions",
        "sts:GetCallerIdentity",
    }.issubset(actions)
    assert {
        "cloudformation:CreateChangeSet",
        "cloudformation:DeleteChangeSet",
        "cloudformation:ExecuteChangeSet",
        "dynamodb:PutItem",
        "dynamodb:TransactWriteItems",
        "ec2:StartInstances",
        "ec2:TerminateInstances",
        "iam:PassRole",
        "sts:AssumeRole",
    }.isdisjoint(actions)
    invoke = next(
        statement
        for statement in statements
        if statement.get("Sid") == "InvokeExactReadOnlyProbeVersion"
    )
    assert invoke["Resource"] == {"Ref": "RehearsalProbeVersion"}
    assert resources["RehearsalCollectorExecutorInvokePermission"]["Properties"][
        "FunctionName"
    ] == {"Ref": "RehearsalCollectorVersion"}
    assert resources["RehearsalCollectorExecutorInvokePermission"]["Properties"][
        "Principal"
    ] == {"Fn::GetAtt": ["RehearsalExecutorRole", "Arn"]}
    assert "RehearsalCollectorClosureInvokePermission" not in resources
    assert resources["RehearsalProbeCollectorInvokePermission"]["Properties"] == {
        "Action": "lambda:InvokeFunction",
        "FunctionName": {"Ref": "RehearsalProbeVersion"},
        "Principal": {"Fn::GetAtt": ["RehearsalCollectorRole", "Arn"]},
        "SourceAccount": "246813579024",
    }
    probe = resources["RehearsalProbeFunction"]["Properties"]
    assert probe["Handler"] == "support_rehearsal_probe_handler.main"
    assert probe["ReservedConcurrentExecutions"] == 5
    assert probe["VpcConfig"] == {
        "SecurityGroupIds": [{"Ref": "AttestationClientSecurityGroup"}],
        "SubnetIds": [
            {"Ref": "PrimaryIsolatedSubnet"},
            {"Ref": "AlternateIsolatedSubnet"},
        ],
    }
    closure_invoke = next(
        statement
        for statement in resources["ClosureSessionRole"]["Properties"]["Policies"][0][
            "PolicyDocument"
        ]["Statement"]
        if statement.get("Sid") == "InvokeExactSupportCallees"
    )
    assert {"Ref": "RehearsalCollectorVersion"} not in closure_invoke["Resource"]
    assert {"Ref": "RehearsalProbeVersion"} not in closure_invoke["Resource"]
    rehearsal_session_name = (
        "h1g-rehearsal-"
        + hashlib.sha256(bundle.inputs.activation_id.encode("ascii")).hexdigest()[:16]
    )
    executor = resources["RehearsalExecutorRole"]["Properties"]
    assert executor["AssumeRolePolicyDocument"]["Statement"] == [
        {
            "Effect": "Allow",
            "Principal": {"AWS": {"Fn::GetAtt": ["FenceExecutorRole", "Arn"]}},
            "Action": "sts:AssumeRole",
            "Condition": {
                "ArnEquals": {
                    "aws:PrincipalArn": {"Fn::GetAtt": ["FenceExecutorRole", "Arn"]}
                },
                "StringEquals": {"sts:RoleSessionName": rehearsal_session_name},
            },
        }
    ]
    fence_statements = resources["FenceExecutorRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    assert all(
        statement.get("Action") != "sts:AssumeRole" for statement in fence_statements
    )
    assert {"Ref": "RehearsalCollectorVersion"} not in resources["SupportWorkflowRole"][
        "Properties"
    ]["Policies"][0]["PolicyDocument"]["Statement"][0]["Resource"]
    bucket_policy = resources["RehearsalBucketPolicy"]["Properties"]["PolicyDocument"][
        "Statement"
    ]
    deny = next(
        statement
        for statement in bucket_policy
        if statement["Sid"] == "DenyNamedProductionPrincipalRehearsalRead"
    )
    denied = deny["Condition"]["ArnEquals"]["aws:PrincipalArn"]
    for logical_id in (
        "ClosureSessionRole",
        "DecisionRole",
        "LaunchAdmissionRole",
        "CombinedHostRole",
    ):
        assert {"Fn::GetAtt": [logical_id, "Arn"]} in denied
    assert "arn:aws:iam::246813579024:role/keep-glm52-gpu-worker" in denied
    assert validate_support_runtime_inventory(template, bundle.inputs) is True
    assert validate_task11_runtime_role_authority(template, bundle.inputs) is True

    widened = copy.deepcopy(template)
    widened["Resources"]["RehearsalCollectorFunction"]["Properties"][
        "ReservedConcurrentExecutions"
    ] = 6
    with pytest.raises(ValueError):
        validate_support_runtime_inventory(widened, bundle.inputs)

    forbidden_action = copy.deepcopy(template)
    forbidden_action["Resources"]["RehearsalCollectorRole"]["Properties"]["Policies"][
        0
    ]["PolicyDocument"]["Statement"].append(
        {
            "Sid": "ForeignLedgerRead",
            "Effect": "Allow",
            "Action": "dynamodb:GetItem",
            "Resource": bundle.inputs.ledger_table_arn,
        }
    )
    with pytest.raises(ValueError):
        validate_task11_runtime_role_authority(forbidden_action, bundle.inputs)

    forbidden_resource = copy.deepcopy(template)
    collector_statements = forbidden_resource["Resources"]["RehearsalCollectorRole"][
        "Properties"
    ]["Policies"][0]["PolicyDocument"]["Statement"]
    next(
        statement
        for statement in collector_statements
        if statement["Sid"] == "ReadWriteExactRehearsalObjects"
    )["Resource"] = bundle.inputs.model_bucket_arn + "/*"
    with pytest.raises(ValueError):
        validate_task11_runtime_role_authority(forbidden_resource, bundle.inputs)

    unqualified_probe = copy.deepcopy(template)
    collector_statements = unqualified_probe["Resources"]["RehearsalCollectorRole"][
        "Properties"
    ]["Policies"][0]["PolicyDocument"]["Statement"]
    next(
        statement
        for statement in collector_statements
        if statement["Sid"] == "InvokeExactReadOnlyProbeVersion"
    )["Resource"] = {"Fn::GetAtt": ["RehearsalProbeFunction", "Arn"]}
    with pytest.raises(ValueError):
        validate_task11_runtime_role_authority(unqualified_probe, bundle.inputs)

    closure_can_invoke = copy.deepcopy(template)
    closure_statements = closure_can_invoke["Resources"]["ClosureSessionRole"][
        "Properties"
    ]["Policies"][0]["PolicyDocument"]["Statement"]
    next(
        statement
        for statement in closure_statements
        if statement["Sid"] == "InvokeExactSupportCallees"
    )["Resource"].append({"Ref": "RehearsalCollectorVersion"})
    with pytest.raises(ValueError, match="invocation partition"):
        validate_task11_runtime_role_authority(
            closure_can_invoke,
            bundle.inputs,
        )

    executor_widened = copy.deepcopy(template)
    executor_widened["Resources"]["RehearsalExecutorRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"][0]["Resource"] = {
        "Fn::GetAtt": ["RehearsalCollectorFunction", "Arn"]
    }
    with pytest.raises(ValueError, match="executor authority"):
        validate_task11_runtime_role_authority(
            executor_widened,
            bundle.inputs,
        )

    broad_executor_principal = copy.deepcopy(template)
    broad_executor_principal["Resources"]["RehearsalExecutorRole"]["Properties"][
        "AssumeRolePolicyDocument"
    ]["Statement"][0]["Principal"] = {"AWS": "arn:aws:iam::246813579024:root"}
    with pytest.raises(ValueError, match="executor authority"):
        validate_task11_runtime_role_authority(
            broad_executor_principal,
            bundle.inputs,
        )

    wildcard_executor_session = copy.deepcopy(template)
    wildcard_executor_session["Resources"]["RehearsalExecutorRole"]["Properties"][
        "AssumeRolePolicyDocument"
    ]["Statement"][0]["Condition"]["StringEquals"][
        "sts:RoleSessionName"
    ] = "h1g-rehearsal-*"
    with pytest.raises(ValueError, match="executor authority"):
        validate_task11_runtime_role_authority(
            wildcard_executor_session,
            bundle.inputs,
        )

    public_external_id = copy.deepcopy(template)
    public_external_id["Resources"]["RehearsalExecutorRole"]["Properties"][
        "AssumeRolePolicyDocument"
    ]["Statement"][0]["Condition"]["StringEquals"][
        "sts:ExternalId"
    ] = bundle.inputs.lambda_code_sha256
    with pytest.raises(ValueError, match="executor authority"):
        validate_task11_runtime_role_authority(
            public_external_id,
            bundle.inputs,
        )

    wildcard_executor_target = copy.deepcopy(template)
    fence_statements = wildcard_executor_target["Resources"]["FenceExecutorRole"][
        "Properties"
    ]["Policies"][0]["PolicyDocument"]["Statement"]
    fence_statements.append(
        {
            "Sid": "ForeignAssumeRoleAuthority",
            "Effect": "Allow",
            "Action": "sts:AssumeRole",
            "Resource": "arn:aws:iam::246813579024:role/*",
        }
    )
    with pytest.raises(ValueError, match="fence stabilization"):
        validate_task11_runtime_role_authority(
            wildcard_executor_target,
            bundle.inputs,
        )

    closure_can_read = copy.deepcopy(template)
    named_deny = next(
        statement
        for statement in closure_can_read["Resources"]["RehearsalBucketPolicy"][
            "Properties"
        ]["PolicyDocument"]["Statement"]
        if statement["Sid"] == "DenyNamedProductionPrincipalRehearsalRead"
    )
    named_deny["Condition"]["ArnEquals"]["aws:PrincipalArn"].remove(
        {"Fn::GetAtt": ["ClosureSessionRole", "Arn"]}
    )
    with pytest.raises(ValueError, match="readability"):
        validate_task11_runtime_role_authority(
            closure_can_read,
            bundle.inputs,
        )

    readable = copy.deepcopy(template)
    readable["Resources"]["RehearsalBucketPolicy"]["Properties"]["PolicyDocument"][
        "Statement"
    ] = []
    with pytest.raises(ValueError):
        validate_task11_runtime_role_authority(readable, bundle.inputs)


def test_support_lifecycle_enforces_68_71_72_hours_without_child_deletion() -> None:
    """Break caught: support work/delete drifts or deadline handler deletes children."""

    from glm52_enforcement.support_plane import (
        SupportLifecycleState,
        evaluate_support_lifecycle,
    )

    start = "2026-07-28T00:00:00Z"
    active = evaluate_support_lifecycle(
        activation_started_at=start,
        observed_at="2026-07-30T19:59:59Z",
        stack_exists=True,
    )
    assert active.state is SupportLifecycleState.ACTIVE
    stopped = evaluate_support_lifecycle(
        activation_started_at=start,
        observed_at="2026-07-30T20:00:00Z",
        stack_exists=True,
    )
    assert stopped.state is SupportLifecycleState.WORK_STOPPED
    assert stopped.host_stopped is True
    delete = evaluate_support_lifecycle(
        activation_started_at=start,
        observed_at="2026-07-30T23:00:00Z",
        stack_exists=True,
    )
    assert delete.state is SupportLifecycleState.DELETE_REQUESTED
    assert delete.delete_stack_requested is True
    missed = evaluate_support_lifecycle(
        activation_started_at=start,
        observed_at="2026-07-31T00:00:00Z",
        stack_exists=True,
    )
    assert missed.state is SupportLifecycleState.DELETE_DEADLINE_MISSED
    assert missed.incident == "SUPPORT_DELETE_DEADLINE_MISSED"
    assert missed.page_operator is True
    assert missed.host_stopped is True
    assert missed.egress_disabled is True
    assert missed.read_only_stack_reconciliations == 1
    assert missed.direct_child_deletions == ()
    absent = evaluate_support_lifecycle(
        activation_started_at=start,
        observed_at="2026-07-31T00:00:00Z",
        stack_exists=False,
    )
    assert absent.state is SupportLifecycleState.ABSENT
    assert absent.incident is None


def test_deadline_and_egress_resources_have_executable_closed_authority() -> None:
    """Break caught: schedules/alarms exist but no role can enforce their outcome."""

    from glm52_enforcement.support_plane import validate_retained_augmentation

    bundle = _bundle()
    resources = bundle.support_template["Resources"]
    role_statements = [
        statement
        for policy in resources["SupportDeadlineRole"]["Properties"]["Policies"]
        for statement in policy["PolicyDocument"]["Statement"]
    ]
    allow_actions = {
        action
        for statement in role_statements
        if statement["Effect"] == "Allow"
        for action in (
            (statement["Action"],)
            if isinstance(statement["Action"], str)
            else tuple(statement["Action"])
        )
    }
    deny_actions = {
        action
        for statement in role_statements
        if statement["Effect"] == "Deny"
        for action in (
            (statement["Action"],)
            if isinstance(statement["Action"], str)
            else tuple(statement["Action"])
        )
    }
    actions = {
        action
        for statement in role_statements
        for action in (
            (statement["Action"],)
            if isinstance(statement["Action"], str)
            else tuple(statement["Action"])
        )
    }
    assert actions == {
        "ec2:StopInstances",
        "ec2:RevokeSecurityGroupEgress",
        "cloudformation:DescribeStacks",
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:UpdateItem",
        "dynamodb:Query",
        "s3:ListBucketVersions",
        "s3:GetObjectVersion",
        "s3:GetObjectVersionAttributes",
        "kms:Decrypt",
        "lambda:InvokeFunction",
        "events:PutEvents",
        "ec2:DescribeInstances",
        "ec2:TerminateInstances",
        "ssm:DescribeInstanceInformation",
        "states:GetExecutionHistory",
        "sns:Publish",
    }
    assert "ec2:TerminateInstances" not in allow_actions
    assert deny_actions == {"ec2:TerminateInstances"}
    continuation_invoke = next(
        statement
        for statement in role_statements
        if statement.get("Sid") == "InvokeExactContinuationCallees"
    )
    assert {
        "Fn::ImportValue": ("KeepGlm52Task12WorkerDrainVersionArn")
    } in continuation_invoke["Resource"]
    assert {
        "Fn::ImportValue": ("KeepGlm52Task9LiabilityWatcherVersionArn")
    } in continuation_invoke["Resource"]
    deadline_environment = resources["SupportDeadlineFunction"]["Properties"][
        "Environment"
    ]["Variables"]
    assert deadline_environment["GLM52_TASK9_LIABILITY_WATCHER_VERSION_ARN"] == {
        "Fn::ImportValue": ("KeepGlm52Task9LiabilityWatcherVersionArn")
    }
    history = next(
        statement
        for statement in role_statements
        if statement.get("Sid") == "ReadExactRetainedLifecycleExecutionHistory"
    )
    assert history == {
        "Sid": "ReadExactRetainedLifecycleExecutionHistory",
        "Effect": "Allow",
        "Action": "states:GetExecutionHistory",
        "Resource": (
            "arn:aws:states:us-west-2:246813579024:"
            "execution:keep-glm52-h1g-retainedlifecycle:*"
        ),
    }
    continuation_objects = next(
        statement
        for statement in role_statements
        if statement.get("Sid") == "ReadExactContinuationObjects"
    )
    assert (
        bundle.inputs.model_bucket_arn
        + "/task13/production/task10-worker-descriptor.json"
    ) in continuation_objects["Resource"]
    decrypt = next(
        statement
        for statement in role_statements
        if statement.get("Sid") == "DecryptExactContinuationObjectsViaS3Only"
    )
    assert decrypt["Condition"]["StringLike"]["kms:EncryptionContext:aws:s3:arn"][
        1
    ] == (
        bundle.inputs.model_bucket_arn
        + "/task13/production/task10-worker-descriptor.json"
    )
    assert {
        "cloudformation:UpdateTerminationProtection",
        "cloudformation:DeleteStack",
        "iam:PassRole",
    }.isdisjoint(actions)
    assert not any(
        action.startswith(
            (
                "ec2:Delete",
                "ec2:Terminate",
                "lambda:Delete",
                "secretsmanager:Delete",
                "kms:ScheduleKeyDeletion",
            )
        )
        and action not in deny_actions
        for action in actions
    )
    retained_role_statements = bundle.retained_augmentation["Resources"][
        "H1gRetainedSupportLifecycleRole"
    ]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    retained_actions = {
        action
        for statement in retained_role_statements
        for action in (
            (statement["Action"],)
            if isinstance(statement["Action"], str)
            else tuple(statement["Action"])
        )
    }
    assert retained_actions == {
        "cloudformation:DescribeStacks",
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "ec2:DescribeInstances",
        "ec2:DescribeSecurityGroups",
        "ec2:RevokeSecurityGroupEgress",
        "ec2:StopInstances",
        "cloudwatch:GetMetricData",
        "cloudwatch:PutMetricData",
        "scheduler:GetSchedule",
        "scheduler:UpdateSchedule",
        "scheduler:DeleteSchedule",
        "sns:Publish",
        "states:StartExecution",
    }
    assert {
        "cloudformation:UpdateTerminationProtection",
        "cloudformation:DeleteStack",
        "iam:PassRole",
    }.isdisjoint(retained_actions)
    workflow_statements = bundle.retained_augmentation["Resources"][
        "H1gRetainedLifecycleStateMachineRole"
    ]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    delete_statement = next(
        statement
        for statement in workflow_statements
        if "cloudformation:DeleteStack"
        in (
            (statement["Action"],)
            if isinstance(statement["Action"], str)
            else statement["Action"]
        )
    )
    assert delete_statement["Resource"] == bundle.inputs.support_stack_id
    pass_role = next(
        statement
        for statement in workflow_statements
        if statement["Action"] == "iam:PassRole"
    )
    assert pass_role["Resource"] == (
        "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-deletion"
    )
    assert pass_role["Condition"] == {
        "StringEquals": {"iam:PassedToService": "cloudformation.amazonaws.com"}
    }
    variables = resources["SupportDeadlineFunction"]["Properties"]["Environment"][
        "Variables"
    ]
    assert variables["GLM52_SUPPORT_STACK_ID"] == {"Ref": "AWS::StackId"}
    assert variables["GLM52_WORK_STOP_HOURS"] == "68"
    assert variables["GLM52_DELETE_REQUEST_HOURS"] == "71"
    assert variables["GLM52_ABSENCE_EXPECTED_HOURS"] == "72"
    assert variables["GLM52_HOST_EGRESS_WARNING_BYTES"] == str(5 * 1024**3)
    assert variables["GLM52_HOST_EGRESS_DRAIN_BYTES"] == str(6 * 1024**3)
    assert variables["GLM52_NAT_AUTHORITY_BYTES"] == str(10 * 1024**3)

    alarm_thresholds = {
        "HostEgressWarningAlarm": 5 * 1024**3,
        "HostEgressDrainAlarm": 6 * 1024**3,
        "NatEgressWarningAlarm": 5 * 1024**3,
        "NatEgressDrainAlarm": 6 * 1024**3,
        "NatEgressAuthorityAlarm": 10 * 1024**3,
    }
    for logical_id, threshold in alarm_thresholds.items():
        alarm = resources[logical_id]["Properties"]
        assert alarm["Threshold"] == threshold
        assert alarm["TreatMissingData"] == "notBreaching"
    rule = resources["SupportEgressAlarmRule"]["Properties"]
    assert rule["EventPattern"]["detail"]["state"]["value"] == ["ALARM"]
    assert rule["Targets"] == [
        {
            "Arn": {"Ref": "SupportDeadlineVersion"},
            "Id": "ExactSupportDeadlineVersion",
            "RetryPolicy": {
                "MaximumEventAgeInSeconds": 60,
                "MaximumRetryAttempts": 0,
            },
        }
    ]
    assert (
        validate_retained_augmentation(
            bundle.retained_augmentation,
            bundle.inputs,
        )
        is True
    )


def test_retained_delete_protocol_is_one_submit_and_readback_gated() -> None:
    """Break caught: protected DeleteStack skips readback or retries mutation."""

    from glm52_enforcement.support_plane import (
        evaluate_retained_delete_protocol,
    )

    inputs = _bundle().inputs
    disable = evaluate_retained_delete_protocol(
        inputs=inputs,
        finalization_state="SUPPORT_FINALIZED",
        stack_exists=True,
        termination_protection=True,
        update_submissions=0,
        delete_submissions=0,
        mutation_outcome="NONE",
    )
    assert disable.mutation == "UpdateTerminationProtection"
    assert disable.mutation_parameters == {
        "StackName": inputs.support_stack_id,
        "EnableTerminationProtection": False,
    }
    assert disable.reads == ()

    readback = evaluate_retained_delete_protocol(
        inputs=inputs,
        finalization_state="SUPPORT_FINALIZED",
        stack_exists=True,
        termination_protection=None,
        update_submissions=1,
        delete_submissions=0,
        mutation_outcome="SUCCESS",
    )
    assert readback.mutation is None
    assert readback.reads == ("DescribeStacks",)

    delete = evaluate_retained_delete_protocol(
        inputs=inputs,
        finalization_state="SUPPORT_FINALIZED",
        stack_exists=True,
        termination_protection=False,
        update_submissions=1,
        delete_submissions=0,
        mutation_outcome="SUCCESS",
    )
    assert delete.mutation == "DeleteStack"
    assert delete.mutation_parameters == {
        "StackName": inputs.support_stack_id,
        "RoleARN": ("arn:aws:iam::246813579024:role/keep-glm52-h1g-support-deletion"),
    }

    for outcome in ("AMBIGUOUS", "SUCCESS"):
        reconcile = evaluate_retained_delete_protocol(
            inputs=inputs,
            finalization_state="SUPPORT_FINALIZED",
            stack_exists=True,
            termination_protection=False,
            update_submissions=1,
            delete_submissions=1,
            mutation_outcome=outcome,
        )
        assert reconcile.mutation is None
        assert reconcile.reads == ("DescribeStacks",)
    absent = evaluate_retained_delete_protocol(
        inputs=inputs,
        finalization_state="SUPPORT_FINALIZED",
        stack_exists=False,
        termination_protection=None,
        update_submissions=1,
        delete_submissions=1,
        mutation_outcome="SUCCESS",
    )
    assert absent.complete is True
    assert absent.mutation is None
    blocked = evaluate_retained_delete_protocol(
        inputs=inputs,
        finalization_state="SUPPORT_FINALIZING",
        stack_exists=True,
        termination_protection=True,
        update_submissions=0,
        delete_submissions=0,
        mutation_outcome="NONE",
    )
    assert blocked.mutation is None


def test_deletion_service_role_has_complete_exact_resource_allowlists() -> None:
    """Break caught: CloudFormation deletion role is wildcarded or incomplete."""

    from glm52_enforcement.support_plane import (
        validate_support_deletion_role_policy,
    )

    bundle = _bundle()
    augmentation = bundle.retained_augmentation
    role = augmentation["Resources"]["H1gSupportDeletionServiceRole"]
    statements = role["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    actions = {
        action
        for statement in statements
        for action in (
            (statement["Action"],)
            if isinstance(statement["Action"], str)
            else tuple(statement["Action"])
        )
    }
    required = {
        "ec2:CreateSnapshot",
        "ec2:DeleteNatGateway",
        "ec2:DeleteNetworkInterface",
        "ec2:DeleteRoute",
        "ec2:DeleteRouteTable",
        "ec2:DisassociateRouteTable",
        "ec2:RevokeSecurityGroupEgress",
        "ec2:RevokeSecurityGroupIngress",
        "ec2:DeleteSecurityGroup",
        "ec2:DeleteSubnet",
        "ec2:DeleteVolume",
        "ec2:DetachVolume",
        "ec2:DeleteVpcEndpoints",
        "ec2:ReleaseAddress",
        "ec2:TerminateInstances",
        "lambda:InvokeFunction",
        "lambda:DeleteFunction",
        "lambda:DeleteFunctionEventInvokeConfig",
        "lambda:RemovePermission",
        "iam:DeleteRolePolicy",
        "iam:DeleteRole",
        "iam:RemoveRoleFromInstanceProfile",
        "iam:DeleteInstanceProfile",
        "s3:ListBucket",
        "s3:DeleteObject",
        "s3:DeleteObjectVersion",
        "s3:DeleteBucket",
        "logs:DeleteLogGroup",
        "cloudwatch:DeleteAlarms",
        "sqs:DeleteQueue",
        "secretsmanager:DeleteResourcePolicy",
        "secretsmanager:DeleteSecret",
        "states:DeleteStateMachine",
        "states:DeleteStateMachineVersion",
        "scheduler:DeleteSchedule",
        "events:RemoveTargets",
        "events:DeleteRule",
        "kms:CreateGrant",
        "kms:RetireGrant",
        "kms:RevokeGrant",
    }
    assert required.issubset(actions)
    unscoped_reads = {
        "cloudwatch:DescribeAlarms",
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
        "logs:DescribeLogGroups",
    }
    wildcard_statements = [
        statement
        for statement in statements
        if statement["Resource"] == "*"
        or (isinstance(statement["Resource"], list) and "*" in statement["Resource"])
    ]
    assert {statement["Action"] for statement in wildcard_statements} == unscoped_reads
    assert all(
        statement["Resource"] == "*"
        and statement["Condition"]
        == {"StringEquals": {"aws:RequestedRegion": "us-west-2"}}
        for statement in wildcard_statements
    )
    assert all(
        statement["Action"] in unscoped_reads
        or (
            statement["Resource"] != "*"
            and (
                not isinstance(statement["Resource"], list)
                or "*" not in statement["Resource"]
            )
        )
        for statement in statements
    )
    callbacks = augmentation["Metadata"]["CustomResourceDeleteCallbacks"]
    assert callbacks == [
        {
            "LogicalResourceId": "SkyBootstrapCustomResource",
            "HandlerVersionArn": (
                "arn:aws:lambda:us-west-2:246813579024:"
                "function:keep-glm52-h1g-support-tls-handler:7"
            ),
        },
        {
            "LogicalResourceId": "TlsBundleCustomResource",
            "HandlerVersionArn": (
                "arn:aws:lambda:us-west-2:246813579024:"
                "function:keep-glm52-h1g-support-tls-handler:7"
            ),
        },
    ]
    assert validate_support_deletion_role_policy(augmentation, bundle.inputs) is True

    missing_snapshot = copy.deepcopy(augmentation)
    snapshot_statements = missing_snapshot["Resources"][
        "H1gSupportDeletionServiceRole"
    ]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    snapshot_statements[:] = [
        statement
        for statement in snapshot_statements
        if "ec2:CreateSnapshot"
        not in (
            (statement["Action"],)
            if isinstance(statement["Action"], str)
            else statement["Action"]
        )
    ]
    with pytest.raises(ValueError):
        validate_support_deletion_role_policy(missing_snapshot, bundle.inputs)

    wildcard = copy.deepcopy(augmentation)
    wildcard_statements = wildcard["Resources"]["H1gSupportDeletionServiceRole"][
        "Properties"
    ]["Policies"][0]["PolicyDocument"]["Statement"]
    next(
        statement
        for statement in wildcard_statements
        if statement["Action"] == "ec2:DeleteVolume"
    )["Resource"] = "*"
    with pytest.raises(ValueError):
        validate_support_deletion_role_policy(wildcard, bundle.inputs)


def test_deletion_inventory_is_authenticated_stack_readback_not_arn_input() -> None:
    """Break caught: self-hashed foreign resources or callbacks gain deletion."""

    from glm52_enforcement.support_plane import (
        build_support_contract_artifacts,
        support_inputs_from_mapping,
    )

    value = _support_inputs()
    inputs = support_inputs_from_mapping(value)
    inventory = inputs.support_deletion_inventory
    readback = inventory["stack_resource_readback"]
    assert readback["support_stack_id"] == inputs.support_stack_id
    assert (
        inventory["stack_resource_readback_evidence"][
            "resource_readback_identity_sha256"
        ]
        == readback["canonical_body_sha256"]
    )
    projection = readback["deletion_inventory_projection"]
    assert (
        projection["action_resources_sha256"]
        == hashlib.sha256(
            json.dumps(
                inventory["action_resources"],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
    )
    metadata = _bundle().retained_augmentation["Metadata"]
    assert (
        metadata["SupportResourceReadbackSha256"] == readback["canonical_body_sha256"]
    )
    assert (
        metadata["SupportResourceReadbackEvidenceSha256"]
        == inventory["stack_resource_readback_evidence"]["canonical_body_sha256"]
    )
    authenticated_fields = build_support_contract_artifacts()[
        "support-input-contract-v1.json"
    ]["environment_specific_authenticated_fields"]
    assert "support_deletion_inventory" not in authenticated_fields
    assert "nat_gateway_id" not in authenticated_fields
    postcreate = build_support_contract_artifacts()["support-input-contract-v1.json"][
        "postcreate_materialization"
    ]
    assert postcreate["caller_physical_ids_accepted"] is False
    assert postcreate["snapshot_binding"] == "CAPTURE_AT_DELETION"

    retained_role = copy.deepcopy(value)
    retained_inventory = retained_role["support_deletion_inventory"]
    assert isinstance(retained_inventory, dict)
    foreign_arn = (
        "arn:aws:iam::246813579024:role/keep-glm52-h1g-retained-support-lifecycle"
    )
    retained_inventory["resource_arns"].append(foreign_arn)
    retained_inventory["resource_names"].append(
        "keep-glm52-h1g-retained-support-lifecycle"
    )
    retained_inventory["resource_arns"].sort()
    retained_inventory["resource_names"].sort()
    retained_inventory["action_resources"]["iam:DeleteRole"].append(foreign_arn)
    retained_inventory["stack_resource_readback"]["resources"].append(
        {
            "LogicalResourceId": "H1gRetainedSupportLifecycleRole",
            "ResourceType": "AWS::IAM::Role",
            "PhysicalResourceId": ("keep-glm52-h1g-retained-support-lifecycle"),
            "ResourceArn": foreign_arn,
            "SourceApi": "ListStackResources",
        }
    )
    _rehash_support_deletion_inventory(retained_inventory)
    with pytest.raises(ValueError):
        support_inputs_from_mapping(retained_role)

    foreign_callback = copy.deepcopy(value)
    callback_inventory = foreign_callback["support_deletion_inventory"]
    assert isinstance(callback_inventory, dict)
    foreign_version = (
        "arn:aws:lambda:us-west-2:246813579024:"
        "function:keep-glm52-h1g-support-tls-handler:99"
    )
    for callback in callback_inventory["custom_resource_delete_callbacks"]:
        callback["HandlerVersionArn"] = foreign_version
    callback_inventory["action_resources"]["lambda:InvokeFunction"] = [foreign_version]
    callback_inventory["resource_arns"].append(foreign_version)
    callback_inventory["resource_arns"].sort()
    _rehash_support_deletion_inventory(callback_inventory)
    with pytest.raises(ValueError):
        support_inputs_from_mapping(foreign_callback)


def test_independent_graph_validators_reject_reviewer_mutants() -> None:
    """Break caught: graph checks collapse to equality with the same builder."""

    from glm52_enforcement.support_plane import (
        validate_retained_augmentation,
        validate_support_lifecycle_resources,
        validate_support_security_graph,
    )

    bundle = _bundle()
    template = bundle.support_template
    assert validate_support_security_graph(template, bundle.inputs) is True
    assert validate_support_lifecycle_resources(template, bundle.inputs) is True
    assert (
        validate_retained_augmentation(
            bundle.retained_augmentation,
            bundle.inputs,
        )
        is True
    )

    mutants = []
    no_endpoint_ingress = copy.deepcopy(template)
    del no_endpoint_ingress["Resources"]["AttestationSecretsEndpointIngress"]
    mutants.append((validate_support_security_graph, no_endpoint_ingress))

    wildcard_endpoint = copy.deepcopy(template)
    wildcard_endpoint["Resources"]["SecretsManagerEndpoint"]["Properties"][
        "PolicyDocument"
    ]["Statement"][0]["Principal"] = "*"
    mutants.append((validate_support_security_graph, wildcard_endpoint))

    missing_secret_target = copy.deepcopy(template)
    del missing_secret_target["Resources"]["SkyBootstrapCustomResource"]["Properties"][
        "SecretTargets"
    ]["RawSkyTokenSecret"]
    mutants.append((validate_support_security_graph, missing_secret_target))

    wrong_metric = copy.deepcopy(template)
    wrong_metric["Resources"]["NatEgressDrainAlarm"]["Properties"]["Threshold"] = (
        10 * 1024**3
    )
    mutants.append((validate_support_lifecycle_resources, wrong_metric))

    fail_open_metrics = copy.deepcopy(template)
    fail_open_metrics["Resources"]["HostEgressDrainAlarm"]["Properties"][
        "TreatMissingData"
    ] = "breaching"
    mutants.append((validate_support_lifecycle_resources, fail_open_metrics))

    for validator, mutant in mutants:
        with pytest.raises(ValueError):
            validator(mutant, bundle.inputs)

    direct_child_delete = copy.deepcopy(bundle.retained_augmentation)
    statements = direct_child_delete["Resources"]["H1gRetainedSupportLifecycleRole"][
        "Properties"
    ]["Policies"][0]["PolicyDocument"]["Statement"]
    statements.append(
        {
            "Effect": "Allow",
            "Action": "ec2:DeleteSubnet",
            "Resource": "*",
        }
    )
    with pytest.raises(ValueError):
        validate_retained_augmentation(direct_child_delete, bundle.inputs)

    retained_mutation = copy.deepcopy(bundle.retained_augmentation)
    statements = retained_mutation["Resources"]["H1gRetainedSupportLifecycleRole"][
        "Properties"
    ]["Policies"][0]["PolicyDocument"]["Statement"]
    statements.append(
        {
            "Effect": "Allow",
            "Action": "cloudformation:UpdateStack",
            "Resource": bundle.inputs.retained_stack_id,
        }
    )
    with pytest.raises(ValueError):
        validate_retained_augmentation(retained_mutation, bundle.inputs)

    foreign_nat_dimension = copy.deepcopy(bundle.retained_augmentation)
    target = foreign_nat_dimension["Resources"]["H1gRetainedNatAccumulatorSchedule"][
        "Properties"
    ]["Target"]
    target["Input"] = target["Input"].replace(
        bundle.inputs.nat_gateway_id,
        "nat-fffffffffffffffff",
    )
    with pytest.raises(ValueError):
        validate_retained_augmentation(foreign_nat_dimension, bundle.inputs)

    unbounded_accumulator = copy.deepcopy(bundle.retained_augmentation)
    del unbounded_accumulator["Resources"]["H1gRetainedNatAccumulatorSchedule"][
        "Properties"
    ]["EndDate"]
    with pytest.raises(ValueError):
        validate_retained_augmentation(unbounded_accumulator, bundle.inputs)

    fail_open_delay = copy.deepcopy(bundle.retained_augmentation)
    fail_open_delay["Resources"]["H1gRetainedSupportLifecycleFunction"]["Properties"][
        "Environment"
    ]["Variables"]["GLM52_MISSING_METRIC_AFTER_GRACE"] = "AUTHORIZE"
    with pytest.raises(ValueError):
        validate_retained_augmentation(fail_open_delay, bundle.inputs)


def test_runtime_and_reader_validators_do_not_regenerate_the_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: validation trusts a second rendering of the same builder."""

    from glm52_enforcement import support_plane

    bundle = _bundle()

    def forbidden_builder(_inputs: object) -> object:
        raise AssertionError("validator regenerated the support template")

    monkeypatch.setattr(
        support_plane,
        "_build_support_template",
        forbidden_builder,
    )
    assert (
        support_plane.validate_support_runtime_inventory(
            bundle.support_template,
            bundle.inputs,
        )
        is True
    )
    assert (
        support_plane.validate_secret_reader_partition(
            bundle.support_template,
            bundle.inputs,
        )
        is True
    )


def test_task11_red_support_workflow_is_budget_gated_and_version_pinned() -> None:
    """Break caught: Task 11 runs through a disabled, alias, or short Lambda."""

    from glm52_enforcement.decision_closure import (
        build_task11_workflow_definition,
    )
    from glm52_enforcement.support_plane import (
        validate_support_runtime_inventory,
        validate_support_template,
    )

    bundle = _bundle()
    template = bundle.support_template
    resources = template["Resources"]
    decision = resources["DecisionFunction"]["Properties"]
    assert decision["Timeout"] == 840
    assert "VpcConfig" not in decision
    assert resources["BudgetGateFunction"]["Properties"]["Handler"] == (
        "support_budget_gate_handler.main"
    )
    assert resources["RehearsalBucket"]["Properties"]["VersioningConfiguration"] == {
        "Status": "Enabled"
    }
    machine = resources["SupportStateMachine"]["Properties"]
    assert machine["StateMachineType"] == "STANDARD"
    assert machine["Definition"] == build_task11_workflow_definition()
    assert "Retry" not in str(machine["Definition"])
    assert resources["SupportStateMachineVersion"] == {
        "Type": "AWS::StepFunctions::StateMachineVersion",
        "Properties": {
            "Description": "Exact Task11 private decision-to-POST version",
            "StateMachineArn": {"Ref": "SupportStateMachine"},
        },
    }
    invoke_resources = resources["SupportWorkflowRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"][0]["Resource"]
    assert invoke_resources == [
        {"Ref": "BudgetGateVersion"},
        {"Ref": "DecisionVersion"},
        {"Ref": "SupportDeadlineVersion"},
    ]
    assert validate_support_template(template, bundle.inputs) is template
    assert validate_support_runtime_inventory(template, bundle.inputs) is True

    alias = copy.deepcopy(template)
    alias["Resources"]["SupportStateMachine"]["Properties"]["Definition"]["States"][
        "DECISION_CLOSURE"
    ]["Resource"] = {"Fn::GetAtt": ["DecisionFunction", "Arn"]}
    with pytest.raises(ValueError):
        validate_support_runtime_inventory(alias, bundle.inputs)

    short = copy.deepcopy(template)
    short["Resources"]["DecisionFunction"]["Properties"]["Timeout"] = 720
    with pytest.raises(ValueError):
        validate_support_runtime_inventory(short, bundle.inputs)


def test_support_archive_imports_every_generated_lambda_handler(
    tmp_path: Path,
) -> None:
    """Break caught: the template names a handler absent from the Lambda zip."""

    bundle = _bundle()
    templates = (bundle.support_template, bundle.retained_augmentation)
    handlers = sorted(
        {
            value["Properties"]["Handler"]
            for template in templates
            for value in template["Resources"].values()
            if value.get("Type") == "AWS::Lambda::Function"
        }
    )
    archive = tmp_path / "support.zip"
    built = subprocess.run(
        [
            sys.executable,
            str(ROOT / "aws/glm52-gpu/scripts/package_h1g_support_lambdas.py"),
            str(archive),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    code = r"""
import importlib
import json
import sys

sys.path.insert(0, sys.argv[1])
for handler in json.loads(sys.argv[2]):
    module_name, attribute = handler.rsplit(".", 1)
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attribute))
"""
    imported = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(archive),
            json.dumps(handlers),
        ],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr


def test_task11_writer_roles_cannot_read_list_or_write_cross_family() -> None:
    """Break caught: one writer role could inspect or mutate another family."""

    from glm52_enforcement.support_plane import (
        support_inputs_from_mapping,
        validate_support_template,
    )

    bundle = _bundle()
    resources = bundle.support_template["Resources"]
    bindings = {item["role_id"]: item for item in bundle.inputs.task11_writer_bindings}
    for role_name, binding in bindings.items():
        policy = resources[role_name + "Role"]["Properties"]["Policies"][0]
        statements = {
            statement["Sid"]: statement
            for statement in policy["PolicyDocument"]["Statement"]
        }
        own = bundle.inputs.model_bucket_arn + "/" + binding["output_key"]
        reads = [
            bundle.inputs.model_bucket_arn + "/" + key for key in binding["read_keys"]
        ]
        assert statements["WriteOwnExactOutput"]["Resource"] == own
        assert statements["ReadOnlyBoundTask11Inputs"]["Resource"] == reads
        assert statements["ReadWriteExactActivationLedger"]["Condition"] == {
            "ForAllValues:StringEquals": {
                "dynamodb:LeadingKeys": binding["ledger_leading_keys"]
            }
        }
        assert "*" not in str(policy)
        for foreign_name, foreign in bindings.items():
            if foreign_name != role_name:
                foreign_output = (
                    bundle.inputs.model_bucket_arn + "/" + foreign["output_key"]
                )
                assert foreign_output not in str(policy)

    swapped = _support_inputs()
    first_output = swapped["task11_writer_bindings"][0]["output_key"]
    swapped["task11_writer_bindings"][0]["output_key"] = swapped[
        "task11_writer_bindings"
    ][1]["output_key"]
    swapped["task11_writer_bindings"][1]["output_key"] = first_output
    with pytest.raises(ValueError):
        support_inputs_from_mapping(swapped)

    missing_leading_key = _support_inputs()
    missing_leading_key["task11_writer_bindings"][0]["ledger_leading_keys"] = []
    with pytest.raises(ValueError):
        support_inputs_from_mapping(missing_leading_key)

    for mutation in ("swap-output", "drop-leading-key"):
        mutant = copy.deepcopy(bundle.support_template)
        policies = mutant["Resources"]["SourceGpuSpendRole"]["Properties"]["Policies"]
        statements = {
            item["Sid"]: item
            for policy in policies
            for item in policy["PolicyDocument"]["Statement"]
        }
        if mutation == "swap-output":
            statements["WriteOwnExactOutput"]["Resource"] = (
                bundle.inputs.model_bucket_arn
                + "/"
                + bindings["SourceSubmissionIntent"]["output_key"]
            )
        else:
            statements["ReadWriteExactActivationLedger"].pop("Condition")
        with pytest.raises(ValueError):
            validate_support_template(mutant, bundle.inputs)


def test_egress_5_6_10_gib_and_missing_metric_edges_fail_closed() -> None:
    """Break caught: lag/missing metrics authorize host egress past drain."""

    from glm52_enforcement.support_plane import (
        evaluate_support_egress,
        support_inputs_from_mapping,
    )

    gib = 1024**3
    inputs = support_inputs_from_mapping(_support_inputs())
    below = evaluate_support_egress(
        inputs=inputs,
        observed_at="2026-07-28T00:15:00Z",
        host_non_endpoint_bytes=5 * gib - 1,
        nat_cumulative_bytes=4 * gib,
        nat_gateway_id=inputs.nat_gateway_id,
        metric_state="OK",
    )
    assert below.warning is False
    warning = evaluate_support_egress(
        inputs=inputs,
        observed_at="2026-07-28T00:15:00Z",
        host_non_endpoint_bytes=5 * gib,
        nat_cumulative_bytes=3 * gib,
        nat_gateway_id=inputs.nat_gateway_id,
        metric_state="OK",
    )
    assert warning.warning is True
    assert warning.drain is False
    drain = evaluate_support_egress(
        inputs=inputs,
        observed_at="2026-07-28T00:15:00Z",
        host_non_endpoint_bytes=6 * gib,
        nat_cumulative_bytes=6 * gib,
        nat_gateway_id=inputs.nat_gateway_id,
        metric_state="OK",
    )
    assert drain.drain is True
    assert drain.egress_disabled is True
    authority = evaluate_support_egress(
        inputs=inputs,
        observed_at="2026-07-28T00:15:00Z",
        host_non_endpoint_bytes=6 * gib,
        nat_cumulative_bytes=11 * gib,
        nat_gateway_id=inputs.nat_gateway_id,
        metric_state="OK",
    )
    assert authority.authority_exceeded is True
    for metric_state in ("MISSING", "DELAYED", "INSUFFICIENT_DATA"):
        fail_closed = evaluate_support_egress(
            inputs=inputs,
            observed_at="2026-07-28T00:15:00Z",
            host_non_endpoint_bytes=0,
            nat_cumulative_bytes=0,
            nat_gateway_id=inputs.nat_gateway_id,
            metric_state=metric_state,
        )
        assert fail_closed.drain is True
        assert fail_closed.egress_disabled is True
    with pytest.raises(ValueError):
        evaluate_support_egress(
            inputs=inputs,
            observed_at="2026-07-28T00:15:00Z",
            host_non_endpoint_bytes=0,
            nat_cumulative_bytes=0,
            nat_gateway_id="nat-fffffffffffffffff",
            metric_state="OK",
        )


def test_nat_egress_accumulator_is_cumulative_and_replay_safe() -> None:
    """Break caught: minute-local NAT sums reset instead of accumulating."""

    from glm52_enforcement.support_plane import (
        accumulate_nat_window,
        initial_nat_accumulator_state,
        nat_metric_data_query_contract,
        support_counter_contract,
        support_inputs_from_mapping,
    )

    inputs = support_inputs_from_mapping(_support_inputs())
    metric_query = nat_metric_data_query_contract(inputs)
    assert [
        query["MetricStat"]["Metric"]["MetricName"] for query in metric_query["queries"]
    ] == [
        "BytesInFromSource",
        "BytesOutToDestination",
        "BytesInFromDestination",
        "BytesOutToSource",
    ]
    assert all(
        query["MetricStat"]["Metric"]["Dimensions"]
        == [
            {
                "Name": "NatGatewayId",
                "Value": inputs.nat_gateway_id,
            }
        ]
        for query in metric_query["queries"]
    )
    contract = support_counter_contract(inputs)
    assert contract["counter_table_name"] == (
        "keep-glm52-h1g-task7-counter-act-20260728-0001"
    )
    assert contract["counter_partition_key"] == "CounterId"
    assert contract["counter_id"] == (
        "ACTIVATION#act-20260728-0001#NAT_PROCESSED_BYTES"
    )
    assert contract["nat_gateway_id"] == inputs.nat_gateway_id
    assert (
        contract["nat_metric_query_contract_sha256"]
        == metric_query["canonical_body_sha256"]
    )
    initial = initial_nat_accumulator_state(inputs)
    assert initial.counter_table_name == contract["counter_table_name"]
    assert initial.counter_partition_key == contract["counter_partition_key"]
    assert initial.counter_id == contract["counter_id"]
    assert initial.support_counter_contract_sha256 == contract["canonical_body_sha256"]
    assert initial.nat_gateway_id == inputs.nat_gateway_id
    first = accumulate_nat_window(
        inputs=inputs,
        state=initial,
        window_started_at="2026-07-28T00:00:00Z",
        window_ended_at="2026-07-28T00:01:00Z",
        nat_gateway_id=inputs.nat_gateway_id,
        nat_counter_bytes=(1, 2, 3, 4),
    )
    assert first.cumulative_nat_processed_bytes == 10
    assert first.accepted_windows == 1
    assert (
        accumulate_nat_window(
            inputs=inputs,
            state=first,
            window_started_at="2026-07-28T00:00:00Z",
            window_ended_at="2026-07-28T00:01:00Z",
            nat_gateway_id=inputs.nat_gateway_id,
            nat_counter_bytes=(1, 2, 3, 4),
        )
        == first
    )
    second = accumulate_nat_window(
        inputs=inputs,
        state=first,
        window_started_at="2026-07-28T00:01:00Z",
        window_ended_at="2026-07-28T00:02:00Z",
        nat_gateway_id=inputs.nat_gateway_id,
        nat_counter_bytes=(5, 6, 7, 8),
    )
    assert second.cumulative_nat_processed_bytes == 36
    assert second.accepted_windows == 2
    with pytest.raises(ValueError):
        accumulate_nat_window(
            inputs=inputs,
            state=first,
            window_started_at="2026-07-28T00:00:00Z",
            window_ended_at="2026-07-28T00:01:00Z",
            nat_gateway_id=inputs.nat_gateway_id,
            nat_counter_bytes=(1, 2, 3, 5),
        )
    with pytest.raises(ValueError):
        accumulate_nat_window(
            inputs=inputs,
            state=second,
            window_started_at="2026-07-28T00:03:00Z",
            window_ended_at="2026-07-28T00:04:00Z",
            nat_gateway_id=inputs.nat_gateway_id,
            nat_counter_bytes=(1, 1, 1, 1),
        )
    with pytest.raises(ValueError):
        accumulate_nat_window(
            inputs=inputs,
            state=initial,
            window_started_at="2026-07-28T00:00:00Z",
            window_ended_at="2026-07-28T00:01:00Z",
            nat_gateway_id="nat-fffffffffffffffff",
            nat_counter_bytes=(1, 2, 3, 4),
        )


def test_retained_nat_observation_reuses_the_frozen_lifecycle_family() -> None:
    """Break caught: Task 7 adds a second table/role/function/version family."""

    bundle = _bundle()
    resources = bundle.retained_augmentation["Resources"]
    unauthorized = {
        "H1gTask7SupportCounterTable",
        "H1gRetainedNatAccumulatorRole",
        "H1gRetainedNatAccumulatorFunction",
        "H1gRetainedNatAccumulatorVersion",
    }
    assert unauthorized.isdisjoint(resources)
    schedule = resources["H1gRetainedNatAccumulatorSchedule"]["Properties"]
    assert schedule["Target"]["Arn"] == {"Ref": "H1gRetainedSupportLifecycleVersion"}
    variables = resources["H1gRetainedSupportLifecycleFunction"]["Properties"][
        "Environment"
    ]["Variables"]
    assert variables["GLM52_NAT_ACCUMULATOR_MODE"] == (
        "ACTIVATION_CUMULATIVE_FOUR_COUNTERS_60S"
    )
    assert "GLM52_COUNTER_TABLE_NAME" not in variables
    missing_alarm = resources["H1gRetainedNatObservationMissingAlarm"]["Properties"]
    assert missing_alarm["EvaluationPeriods"] == 16
    assert missing_alarm["DatapointsToAlarm"] == 16
    assert missing_alarm["TreatMissingData"] == "breaching"
    assert missing_alarm["MetricName"] == ("ActivationCumulativeNatProcessedBytes")
    rendered = json.dumps(
        {
            key: value
            for key, value in resources.items()
            if key.startswith("H1gRetained")
        },
        sort_keys=True,
    )
    assert "dynamodb:UpdateItem" not in rendered
    assert "retained_nat_accumulator_handler" not in rendered


def test_reused_lifecycle_handler_reads_four_60s_counters_without_state() -> None:
    """Break caught: NAT accumulation needs a table or enters workflow history."""

    from glm52_enforcement.retained_support_lifecycle_handler import (
        RetainedLifecycleServices,
        activation_cumulative_nat_request,
        main,
    )

    request = activation_cumulative_nat_request(
        activation_started_at="2026-07-28T00:00:00Z",
        observed_at="2026-07-31T00:00:00Z",
        nat_gateway_id="nat-00000000000000003",
    )
    assert request["MaxDatapoints"] == 4 * 72 * 60
    assert request["ScanBy"] == "TimestampAscending"
    assert len(request["MetricDataQueries"]) == 4
    assert {
        query["MetricStat"]["Metric"]["MetricName"]
        for query in request["MetricDataQueries"]
    } == {
        "BytesInFromSource",
        "BytesOutToDestination",
        "BytesInFromDestination",
        "BytesOutToSource",
    }
    assert all(
        query["MetricStat"]["Period"] == 60 for query in request["MetricDataQueries"]
    )
    with pytest.raises(ValueError, match="outside exact authority"):
        activation_cumulative_nat_request(
            activation_started_at="2026-07-28T00:00:00Z",
            observed_at="2026-07-31T00:00:01Z",
            nat_gateway_id="nat-00000000000000003",
        )

    class CloudWatch:
        def __init__(self) -> None:
            self.get_requests: list[dict[str, object]] = []
            self.put_requests: list[dict[str, object]] = []

        def get_metric_data(self, **kwargs: object) -> dict[str, object]:
            self.get_requests.append(dict(kwargs))
            timestamps = [
                "2026-07-28T00:00:00Z",
                "2026-07-28T00:01:00Z",
            ]
            return {
                "MetricDataResults": [
                    {
                        "Id": query_id,
                        "StatusCode": "Complete",
                        "Timestamps": timestamps,
                        "Values": [value, value],
                    }
                    for query_id, value in zip(
                        ("nat0", "nat1", "nat2", "nat3"),
                        (1, 2, 3, 4),
                    )
                ],
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "nat-read",
                },
            }

        def put_metric_data(self, **kwargs: object) -> dict[str, object]:
            self.put_requests.append(dict(kwargs))
            return {
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "nat-publish",
                }
            }

    cloudwatch = CloudWatch()
    environment = {
        "GLM52_ACTIVATION_ID": "act-20260728-0001",
        "GLM52_RUN_ID": "glm52-sky-20260724",
        "GLM52_NAT_GATEWAY_ID": "nat-00000000000000003",
        "GLM52_SUPPORT_HOST_INSTANCE_ID": "i-00000000000000001",
        "GLM52_SUPPORT_HOST_SECURITY_GROUP_ID": ("sg-0000000000000000b"),
        "GLM52_ACTIVATION_STARTED_AT": "2026-07-28T00:00:00Z",
        "GLM52_BOOTSTRAP_GRACE_DEADLINE": "2026-07-28T00:15:00Z",
        "GLM52_MISSING_METRIC_AFTER_GRACE": "DRAIN",
        "GLM52_NAT_ACCUMULATOR_MODE": ("ACTIVATION_CUMULATIVE_FOUR_COUNTERS_60S"),
    }
    event = {
        "action": "ACCUMULATE_NAT_60S",
        "activation_id": "act-20260728-0001",
        "activation_started_at": "2026-07-28T00:00:00Z",
        "nat_gateway_id": "nat-00000000000000003",
    }
    observed = main(
        event,
        None,
        services=RetainedLifecycleServices(
            cloudwatch=cloudwatch,
            ec2=object(),
        ),
        environment=environment,
        observed_at="2026-07-28T00:02:00Z",
    )
    assert observed == {
        "activation_id": "act-20260728-0001",
        "nat_gateway_id": "nat-00000000000000003",
        "observed_at": "2026-07-28T00:02:00Z",
        "window_count": 2,
        "cumulative_bytes": 20,
        "metric_state": "OK",
        "host_stopped": False,
        "egress_disabled": False,
    }
    assert len(cloudwatch.get_requests) == 1
    assert len(cloudwatch.put_requests) == 1
    assert cloudwatch.put_requests[0]["MetricData"][0]["Value"] == 20
    assert not any(
        "dynamodb" in repr(request).lower()
        for request in cloudwatch.get_requests + cloudwatch.put_requests
    )


def test_nat_observation_rejects_pagination_instead_of_hiding_a_retry() -> None:
    """Break caught: a partial CloudWatch page is treated as cumulative truth."""

    from glm52_enforcement.retained_support_lifecycle_handler import (
        activation_cumulative_nat_observation,
    )

    with pytest.raises(ValueError, match="foreign or paginated"):
        activation_cumulative_nat_observation(
            response={
                "NextToken": "unconsumed-page",
                "MetricDataResults": [],
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "partial",
                },
            },
            activation_id="act-20260728-0001",
            activation_started_at="2026-07-28T00:00:00Z",
            nat_gateway_id="nat-00000000000000003",
            observed_at="2026-07-28T00:01:00Z",
        )


@pytest.mark.parametrize(
    ("timestamps", "expected_state", "observed_at"),
    [
        (
            ["2026-07-28T00:59:00Z"],
            "DELAYED",
            "2026-07-28T01:00:00Z",
        ),
        ([], "MISSING", "2026-07-28T01:00:00Z"),
        ([], "MISSING", "2026-07-28T01:00:05Z"),
    ],
)
def test_post_grace_partial_or_missing_nat_observation_drains_exact_host(
    timestamps: list[str],
    expected_state: str,
    observed_at: str,
) -> None:
    """Break caught: partial/missing NAT data keeps work authorized."""

    from glm52_enforcement.retained_support_lifecycle_handler import (
        RetainedLifecycleServices,
        main,
    )

    class CloudWatch:
        def __init__(self) -> None:
            self.put_calls = 0
            self.get_calls: list[dict[str, object]] = []

        def get_metric_data(self, **kwargs: object) -> dict[str, object]:
            self.get_calls.append(dict(kwargs))
            return {
                "MetricDataResults": [
                    {
                        "Id": query_id,
                        "StatusCode": "Complete",
                        "Timestamps": timestamps,
                        "Values": [1 for _ in timestamps],
                    }
                    for query_id in ("nat0", "nat1", "nat2", "nat3")
                ],
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "nat-partial",
                },
            }

        def put_metric_data(self, **_kwargs: object) -> dict[str, object]:
            self.put_calls += 1
            return {
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "unexpected-publish",
                }
            }

    class Ec2:
        def __init__(self) -> None:
            self.stop_calls: list[dict[str, object]] = []
            self.revoke_calls: list[dict[str, object]] = []

        def describe_instances(self, **_kwargs: object) -> dict[str, object]:
            return {
                "Reservations": [
                    {
                        "Instances": [
                            {
                                "InstanceId": "i-00000000000000001",
                                "State": {"Name": "running"},
                            }
                        ]
                    }
                ]
            }

        def stop_instances(self, **kwargs: object) -> dict[str, object]:
            self.stop_calls.append(dict(kwargs))
            return {
                "StoppingInstances": [
                    {
                        "InstanceId": "i-00000000000000001",
                        "CurrentState": {"Name": "stopping"},
                    }
                ]
            }

        def describe_security_groups(self, **_kwargs: object) -> dict[str, object]:
            return {
                "SecurityGroups": [
                    {
                        "GroupId": "sg-0000000000000000b",
                        "IpPermissionsEgress": [
                            {
                                "IpProtocol": "-1",
                                "IpRanges": [
                                    {
                                        "CidrIp": "0.0.0.0/0",
                                        "Description": ("host-only bounded NAT egress"),
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }

        def revoke_security_group_egress(self, **kwargs: object) -> dict[str, object]:
            self.revoke_calls.append(dict(kwargs))
            return {"Return": True}

    cloudwatch = CloudWatch()
    ec2 = Ec2()
    environment = {
        "GLM52_ACTIVATION_ID": "act-20260728-0001",
        "GLM52_RUN_ID": "glm52-sky-20260724",
        "GLM52_NAT_GATEWAY_ID": "nat-00000000000000003",
        "GLM52_SUPPORT_HOST_INSTANCE_ID": "i-00000000000000001",
        "GLM52_SUPPORT_HOST_SECURITY_GROUP_ID": ("sg-0000000000000000b"),
        "GLM52_ACTIVATION_STARTED_AT": "2026-07-28T00:00:00Z",
        "GLM52_BOOTSTRAP_GRACE_DEADLINE": "2026-07-28T00:15:00Z",
        "GLM52_MISSING_METRIC_AFTER_GRACE": "DRAIN",
        "GLM52_NAT_ACCUMULATOR_MODE": ("ACTIVATION_CUMULATIVE_FOUR_COUNTERS_60S"),
    }
    result = main(
        {
            "action": "ACCUMULATE_NAT_60S",
            "activation_id": "act-20260728-0001",
            "activation_started_at": "2026-07-28T00:00:00Z",
            "nat_gateway_id": "nat-00000000000000003",
        },
        None,
        services=RetainedLifecycleServices(
            cloudwatch=cloudwatch,
            ec2=ec2,
        ),
        environment=environment,
        observed_at=observed_at,
    )
    assert result["metric_state"] == expected_state
    assert result["cumulative_bytes"] == 0
    assert result["host_stopped"] is True
    assert result["egress_disabled"] is True
    assert cloudwatch.put_calls == 0
    assert cloudwatch.get_calls[0]["EndTime"].isoformat() == (
        "2026-07-28T01:00:00+00:00"
    )
    assert result["observed_at"] == "2026-07-28T01:00:00Z"
    assert ec2.stop_calls == [{"InstanceIds": ["i-00000000000000001"]}]
    assert ec2.revoke_calls == [
        {
            "GroupId": "sg-0000000000000000b",
            "IpPermissions": [
                {
                    "IpProtocol": "-1",
                    "IpRanges": [
                        {
                            "CidrIp": "0.0.0.0/0",
                            "Description": ("host-only bounded NAT egress"),
                        }
                    ],
                }
            ],
        }
    ]


def test_retained_graph_binds_one_exact_standard_deletion_version() -> None:
    """Break caught: the durable deletion definition is not the deployed graph."""

    from glm52_enforcement.fence_executor import (
        build_support_deletion_workflow_definition,
        support_deletion_workflow_history_event_ceiling,
    )

    augmentation = _bundle().retained_augmentation
    resources = augmentation["Resources"]
    machine = resources["H1gRetainedLifecycleStateMachine"]["Properties"]
    assert machine["StateMachineName"] == ("keep-glm52-h1g-retained-lifecycle")
    assert machine["StateMachineType"] == "STANDARD"
    assert machine["Definition"] == (build_support_deletion_workflow_definition())
    version = resources["H1gRetainedLifecycleStateMachineVersion"]["Properties"]
    assert version["StateMachineArn"] == {"Ref": "H1gRetainedLifecycleStateMachine"}
    runtime_actions = json.dumps(
        resources["H1gRetainedSupportLifecycleRole"],
        sort_keys=True,
    )
    assert "UpdateTerminationProtection" not in runtime_actions
    assert "DeleteStack" not in runtime_actions
    workflow_actions = json.dumps(
        resources["H1gRetainedLifecycleStateMachineRole"],
        sort_keys=True,
    )
    assert "cloudformation:UpdateTerminationProtection" in workflow_actions
    assert "cloudformation:DeleteStack" in workflow_actions
    history_ceiling = support_deletion_workflow_history_event_ceiling()
    assert history_ceiling == 7_984
    assert history_ceiling < 12_000
    assert (
        augmentation["Metadata"]["RetainedLifecycleHistoryEventCeiling"]
        == history_ceiling
    )


def test_postcreate_authority_has_no_public_forgeable_materialization() -> None:
    """Break caught: a caller recomputes a hash around foreign coordinates."""

    from glm52_enforcement import support_plane

    assert not hasattr(support_plane, "SupportPostcreateMaterialization")
    assert not hasattr(support_plane, "materialize_support_postcreate")
    assert not hasattr(support_plane, "_SupportPostcreateMaterialization")
    assert not hasattr(support_plane, "_POSTCREATE_MINT_CAPABILITY")
    assert not hasattr(support_plane, "_support_inputs_from_materialization")
    assert (
        "materialization"
        not in __import__("inspect")
        .signature(support_plane._build_support_postcreate_plane)
        .parameters
    )
    assert hasattr(support_plane, "coordinate_support_postcreate")


def test_bootstrap_grace_is_activation_bound_and_never_authorizes_missing() -> None:
    """Break caught: a caller-controlled grace silently authorizes blind work."""

    from glm52_enforcement.canonical import canonical_json_bytes
    from glm52_enforcement.support_plane import (
        bootstrap_grace_contract,
        evaluate_support_egress,
        support_inputs_from_mapping,
        validate_bootstrap_grace_contract,
    )

    inputs = support_inputs_from_mapping(_support_inputs())
    contract = bootstrap_grace_contract(inputs)
    assert contract["grace_started_at"] == "2026-07-28T00:00:00Z"
    assert contract["grace_deadline"] == "2026-07-28T00:15:00Z"
    assert contract["grace_duration_seconds"] == 900
    assert validate_bootstrap_grace_contract(contract, inputs) is True

    pending = evaluate_support_egress(
        inputs=inputs,
        observed_at="2026-07-28T00:14:59Z",
        host_non_endpoint_bytes=0,
        nat_cumulative_bytes=0,
        nat_gateway_id=inputs.nat_gateway_id,
        metric_state="MISSING",
    )
    assert pending.bootstrap_grace_complete is False
    assert pending.work_authorized is False
    assert pending.drain is False
    ready = evaluate_support_egress(
        inputs=inputs,
        observed_at="2026-07-28T00:00:05Z",
        host_non_endpoint_bytes=0,
        nat_cumulative_bytes=0,
        nat_gateway_id=inputs.nat_gateway_id,
        metric_state="OK",
    )
    assert ready.work_authorized is True
    expired = evaluate_support_egress(
        inputs=inputs,
        observed_at="2026-07-28T00:15:00Z",
        host_non_endpoint_bytes=0,
        nat_cumulative_bytes=0,
        nat_gateway_id=inputs.nat_gateway_id,
        metric_state="MISSING",
    )
    assert expired.bootstrap_grace_complete is True
    assert expired.work_authorized is False
    assert expired.drain is True

    extended = copy.deepcopy(contract)
    extended["grace_deadline"] = "2026-07-28T00:30:00Z"
    body = dict(extended)
    del body["canonical_body_sha256"]
    extended["canonical_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    with pytest.raises(ValueError):
        validate_bootstrap_grace_contract(extended, inputs)


def test_nat_accumulator_schedule_retirement_is_one_submit_and_reconcile_only() -> None:
    """Break caught: minute schedule survives its exact 72-hour authority."""

    from glm52_enforcement.support_plane import (
        evaluate_nat_accumulator_schedule_retirement,
        support_inputs_from_mapping,
    )

    inputs = support_inputs_from_mapping(_support_inputs())
    before = evaluate_nat_accumulator_schedule_retirement(
        inputs=inputs,
        observed_at="2026-07-30T23:59:59Z",
        schedule_exists=True,
        schedule_state="ENABLED",
        disable_submissions=0,
        delete_submissions=0,
    )
    assert before.mutation is None
    disable = evaluate_nat_accumulator_schedule_retirement(
        inputs=inputs,
        observed_at="2026-07-31T00:00:00Z",
        schedule_exists=True,
        schedule_state="ENABLED",
        disable_submissions=0,
        delete_submissions=0,
    )
    assert disable.mutation == "UpdateSchedule"
    assert disable.mutation_parameters["Name"] == (
        "keep-glm52-h1g-act-20260728-0001-nat-accumulator"
    )
    assert disable.mutation_parameters["State"] == "DISABLED"
    readback = evaluate_nat_accumulator_schedule_retirement(
        inputs=inputs,
        observed_at="2026-07-31T00:00:01Z",
        schedule_exists=True,
        schedule_state=None,
        disable_submissions=1,
        delete_submissions=0,
    )
    assert readback.mutation is None
    assert readback.reads == ("GetSchedule",)
    delete = evaluate_nat_accumulator_schedule_retirement(
        inputs=inputs,
        observed_at="2026-07-31T00:00:02Z",
        schedule_exists=True,
        schedule_state="DISABLED",
        disable_submissions=1,
        delete_submissions=0,
    )
    assert delete.mutation == "DeleteSchedule"
    reconcile = evaluate_nat_accumulator_schedule_retirement(
        inputs=inputs,
        observed_at="2026-07-31T00:00:03Z",
        schedule_exists=True,
        schedule_state="DISABLED",
        disable_submissions=1,
        delete_submissions=1,
    )
    assert reconcile.mutation is None
    assert reconcile.reads == ("GetSchedule",)
    absent = evaluate_nat_accumulator_schedule_retirement(
        inputs=inputs,
        observed_at="2026-07-31T00:00:04Z",
        schedule_exists=False,
        schedule_state=None,
        disable_submissions=1,
        delete_submissions=1,
    )
    assert absent.complete is True


def test_every_support_usage_counter_and_retained_ceiling_is_closed() -> None:
    """Break caught: one accepted counter, log, event, cancel, or snapshot limit widens."""

    from glm52_enforcement.support_plane import (
        SUPPORT_USAGE_CEILINGS,
        validate_support_usage,
    )

    exact = dict(SUPPORT_USAGE_CEILINGS)
    assert validate_support_usage(exact) is True
    for field, limit in SUPPORT_USAGE_CEILINGS.items():
        mutant = dict(exact)
        if field in {
            "lambda_reserved_concurrency_per_function",
            "support_execution_count",
            "active_finalization_owners",
            "snapshot_schedule_count",
            "support_host_count",
            "support_nat_count",
            "support_eip_count",
            "support_data_volume_count",
            "support_interface_endpoint_az_set_count",
            "support_workflow_count",
            "transient_log_retention_days",
            "liability_scan_interval_seconds",
            "liability_lambda_memory_mib",
            "liability_lambda_timeout_seconds",
            "forensic_snapshot_retention_days",
        }:
            mutant[field] = limit + 1
        else:
            mutant[field] = limit + 1
        with pytest.raises(ValueError, match=field):
            validate_support_usage(mutant)


def test_builder_cli_is_deterministic_canonical_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    """Break caught: builder uses AWS/defaults, emits drift, or overwrites evidence."""

    from glm52_enforcement.canonical import canonical_json_bytes

    input_path = tmp_path / "inputs.json"
    price_path = tmp_path / "price-card.json"
    build_inputs = _support_build_input_mapping()
    input_path.write_bytes(canonical_json_bytes(build_inputs) + b"\n")
    price_path.write_bytes(canonical_json_bytes(_price_card()) + b"\n")
    first = tmp_path / "first"
    second = tmp_path / "second"
    script = ROOT / "aws/glm52-gpu/scripts/build_h1g_support_plane.py"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        str(script),
        "--inputs",
        str(input_path),
        "--price-card",
        str(price_path),
    ]
    one = subprocess.run(
        command + ["--output-dir", str(first)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    two = subprocess.run(
        command + ["--output-dir", str(second)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert one.returncode == 0, one.stderr
    assert two.returncode == 0, two.stderr
    expected_names = {
        "support-plane-v1.json",
        "support-spend-descriptor-v1.json",
        "support-manifest-v1.json",
    }
    assert {path.name for path in first.iterdir()} == expected_names
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    for path in first.iterdir():
        raw = path.read_bytes()
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
        parsed = json.loads(raw.decode("ascii"))
        assert canonical_json_bytes(parsed) + b"\n" == raw
    manifest = json.loads((first / "support-manifest-v1.json").read_text())
    for artifact in manifest["artifacts"]:
        raw = (first / artifact["file_name"]).read_bytes()
        assert artifact["sha256"] == hashlib.sha256(raw).hexdigest()
        assert artifact["bytes"] == len(raw)
    overwrite = subprocess.run(
        command + ["--output-dir", str(first)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert overwrite.returncode != 0
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


def test_builder_cli_serializes_pre_support_runtime_before_terminal_export(
    tmp_path: Path,
) -> None:
    """Break caught: Phase 4 cannot render retained runtime until Phase 7 exists."""

    from glm52_enforcement.canonical import canonical_json_bytes

    input_path = tmp_path / "pre-support-runtime-inputs.json"
    input_path.write_bytes(
        canonical_json_bytes(_pre_support_runtime_input_mapping()) + b"\n"
    )
    first = tmp_path / "runtime-first"
    second = tmp_path / "runtime-second"
    script = ROOT / "aws/glm52-gpu/scripts/build_h1g_support_plane.py"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        str(script),
        "--inputs",
        str(input_path),
        "--retained-runtime-output-dir",
    ]

    one = subprocess.run(
        command + [str(first)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    two = subprocess.run(
        command + [str(second)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert one.returncode == 0, one.stderr
    assert two.returncode == 0, two.stderr
    assert {path.name for path in first.iterdir()} == {
        "support-retained-runtime-v1.json"
    }
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    raw = (first / "support-retained-runtime-v1.json").read_bytes()
    fragment = json.loads(raw.decode("ascii"))
    assert raw == canonical_json_bytes(fragment) + b"\n"
    assert len(fragment["Resources"]) == 74
    assert fragment["Outputs"]["Task12TerminalV2VersionArn"]["Export"] == {
        "Name": "KeepGlm52Task12TerminalV2VersionArn"
    }


def test_postcreate_cli_guards_identity_and_uses_only_injected_read_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: the real postcreate route uses defaults, writes, or mutates."""

    from glm52_enforcement.canonical import canonical_json_bytes
    from glm52_enforcement.support_plane import (
        build_support_precreate_plane,
        support_build_inputs_from_mapping,
        support_price_card_from_mapping,
    )

    script_path = ROOT / "aws/glm52-gpu/scripts/materialize_h1g_support_plane.py"
    spec = importlib.util.spec_from_file_location(
        "task7_postcreate_cli",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    input_value = _support_build_input_mapping()
    inputs = support_build_inputs_from_mapping(input_value)
    template = build_support_precreate_plane(
        inputs=inputs,
        price_card=support_price_card_from_mapping(
            _price_card(),
            inputs=inputs,
        ),
    ).support_template
    input_path = tmp_path / "inputs.json"
    template_path = tmp_path / "support.json"
    input_path.write_bytes(canonical_json_bytes(input_value) + b"\n")
    template_path.write_bytes(canonical_json_bytes(template) + b"\n")
    stack_id = (
        "arn:aws:cloudformation:us-west-2:246813579024:"
        "stack/keep-glm52-h1g-support/"
        "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    )
    events: list[tuple[str, object]] = []

    class ReadClient:
        def __getattr__(self, name: str) -> object:
            if name in {
                "create_stack",
                "update_stack",
                "delete_stack",
                "update_termination_protection",
                "run_instances",
                "terminate_instances",
            }:
                raise AssertionError(f"mutation boundary exposed: {name}")
            raise AttributeError(name)

    class Runner:
        def __init__(self, *, profile: str, region: str) -> None:
            events.append(("runner", (profile, region)))
            self.cloudformation = ReadClient()
            self.ec2 = ReadClient()

        def get_caller_identity(self) -> dict[str, object]:
            events.append(("sts", None))
            return {
                "Account": "246813579024",
                "Arn": ("arn:aws:sts::246813579024:assumed-role/fixture/session"),
                "UserId": "fixture:session",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "sts-fixture",
                },
            }

    def coordinate(**kwargs: object) -> object:
        events.append(("coordinate", kwargs["support_stack_id"]))
        services = kwargs["services"]
        assert type(services.cloudformation) is ReadClient
        assert type(services.ec2) is ReadClient
        return SimpleNamespace(
            inputs=SimpleNamespace(
                support_deletion_inventory={"record_type": "fixture-deletion-authority"}
            ),
            manifest={"record_type": "fixture-postcreate-manifest"},
            postcreate_retained_fragment={
                "record_type": "fixture-postcreate-retained-fragment"
            },
            retained_augmentation={
                "record_type": "fixture-legacy-retained-augmentation"
            },
            task9_deployed_identity={"record_type": "glm52_task9_deployed_identity_v1"},
        )

    monkeypatch.setattr(module, "coordinate_support_postcreate", coordinate)
    monkeypatch.setattr(
        module,
        "support_inputs_projection",
        lambda inputs: {
            "record_type": "glm52_h1g_support_inputs_v1",
        },
    )
    monkeypatch.setenv("AWS_PAGER", "")
    output = tmp_path / "postcreate"
    argv = [
        "--profile",
        "keep-gpu",
        "--region",
        "us-west-2",
        "--support-stack-id",
        stack_id,
        "--inputs",
        str(input_path),
        "--support-template",
        str(template_path),
        "--output-dir",
        str(output),
    ]
    assert module.run(argv, runner_factory=Runner) == 0
    assert events == [
        ("runner", ("keep-gpu", "us-west-2")),
        ("sts", None),
        ("coordinate", stack_id),
    ]
    assert {path.name for path in output.iterdir()} == {
        "support-deletion-authority-v1.json",
        "support-materialized-inputs-v1.json",
        "support-postcreate-manifest-v1.json",
        "support-retained-augmentation-v1.json",
        "TASK9_DEPLOYED_IDENTITY.json",
    }
    for path in output.iterdir():
        raw = path.read_bytes()
        assert raw == canonical_json_bytes(json.loads(raw.decode("ascii"))) + b"\n"
    retained = json.loads(
        (output / "support-retained-augmentation-v1.json").read_bytes()
    )
    assert retained == {"record_type": "fixture-postcreate-retained-fragment"}
    with pytest.raises(FileExistsError):
        module.run(argv, runner_factory=Runner)
    calls_before_guard = list(events)
    foreign = list(argv)
    foreign[1] = "default"
    foreign[-1] = str(tmp_path / "foreign")
    with pytest.raises(ValueError, match="profile/region"):
        module.run(foreign, runner_factory=Runner)
    assert events == calls_before_guard

    class ForeignAccountRunner(Runner):
        def get_caller_identity(self) -> dict[str, object]:
            events.append(("sts-foreign", None))
            return {
                "Account": "000000000000",
                "Arn": ("arn:aws:sts::000000000000:assumed-role/fixture/session"),
                "UserId": "fixture:session",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "sts-foreign",
                },
            }

    foreign_account = list(argv)
    foreign_account[-1] = str(tmp_path / "foreign-account")
    with pytest.raises(ValueError, match="campaign account"):
        module.run(foreign_account, runner_factory=ForeignAccountRunner)
    assert not (tmp_path / "foreign-account").exists()
    assert not any(
        event == ("coordinate", stack_id) for event in events[len(calls_before_guard) :]
    )


def test_support_modules_are_python39_import_light_and_block_forbidden_imports() -> (
    None
):
    """Break caught: Task 7 imports MLX/NumPy/boto3/model packages."""

    import ast

    modules = (
        ROOT / "src/glm52_enforcement/support_plane.py",
        ROOT / "src/glm52_enforcement/support_custom_resources.py",
    )
    blocked = {"mlx", "numpy", "mlx_vq", "boto3", "botocore"}
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        assert imported.isdisjoint(blocked)
    code = """
import builtins
import sys
blocked = {"mlx", "numpy", "mlx_vq", "boto3", "botocore"}
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError("blocked import: " + name)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
import glm52_enforcement.support_plane
import glm52_enforcement.support_custom_resources
assert blocked.isdisjoint(sys.modules)
print("task7-import-light-ok")
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "task7-import-light-ok"


def test_checked_in_support_contract_artifacts_are_generated_canonical_and_nonlive() -> (
    None
):
    """Break caught: checked-in H.1g contract assets drift or embed invented live IDs."""

    from glm52_enforcement.canonical import canonical_json_bytes
    from glm52_enforcement.support_plane import build_support_contract_artifacts

    expected = build_support_contract_artifacts()
    artifact_dir = ROOT / "aws/glm52-gpu/cfn/h1g"
    assert set(expected) == {
        "support-input-contract-v1.json",
        "support-template-contract-v1.json",
        "support-spend-envelope-v1.json",
        "support-task11-workflow-v1.json",
        "support-task12-runtime-v1.json",
        "support-task10-production-v1.json",
        "support-contract-manifest-v1.json",
    }
    for file_name, projection in expected.items():
        path = artifact_dir / file_name
        raw = path.read_bytes()
        assert raw == canonical_json_bytes(projection) + b"\n"
        assert json.loads(raw.decode("ascii")) == projection
        if file_name != "support-contract-manifest-v1.json":
            assert b"ami-" not in raw
            assert b"vpc-" not in raw
            assert b"subnet-" not in raw
            assert b"arn:aws:kms:" not in raw
    manifest = expected["support-contract-manifest-v1.json"]
    for artifact in manifest["artifacts"]:
        raw = (artifact_dir / artifact["file_name"]).read_bytes()
        assert artifact["sha256"] == hashlib.sha256(raw).hexdigest()


def test_task12_retained_runtime_is_merged_without_replacing_support_delete() -> None:
    """Break caught: Task 12 remains a standalone fixture outside deployment."""

    augmentation = _bundle().retained_augmentation
    resources = augmentation["Resources"]

    assert len(resources) == 88
    assert "H1gRetainedLifecycleStateMachine" in resources
    assert (
        resources["H1gRetainedLifecycleStateMachine"]["Properties"]["Definition"][
            "StartAt"
        ]
        == "DescribeProtectedSupportStack"
    )
    assert "RetainedLifecycleStateMachine" in resources
    assert "SnapshotCleanupStateMachine" in resources
    task12 = augmentation["Metadata"]["task12_retained_ownership"]
    assert task12["support_owned_resources"] == []
    assert task12["support_stack_may_delete_retained_resources"] is False
    assert set(task12["retained_owned_resources"]).issubset(resources)


def test_postcreate_manifest_requires_exact_task12_version_authority_custody() -> None:
    """Break caught: self-hashed inline config guesses unpublished versions."""

    bundle = _bundle()
    manifest = bundle.manifest
    authority = manifest["task12_deployment_authority"]
    assert authority["materialization_phase"] == (
        "AFTER_RETAINED_STACK_VERSION_PUBLICATION"
    )
    assert authority["lookup_consistency"] == "STRONGLY_CONSISTENT"
    assert authority["lookup_key_source"] == "context.invoked_function_arn"
    assert authority["record_type"] == "glm52_task12_lambda_deployment_v1"
    assert authority["record_count"] == 10
    assert authority["caller_state_machine_version_required"] is True
    assert authority["function_version_required"] is True
    assert authority["inline_self_hashed_config_forbidden"] is True
    assert authority["static_worker_or_snapshot_ids_forbidden"] is True


def test_task12_runtime_contract_is_checked_in_with_retained_cardinality() -> None:
    """Break caught: generated contracts omit Task 12 ownership and cost shape."""

    from glm52_enforcement.support_plane import build_support_contract_artifacts

    artifacts = build_support_contract_artifacts()
    task12 = artifacts["support-task12-runtime-v1.json"]
    assert task12["retained_function_versions"] == 8
    assert task12["retained_standard_workflow_versions"] == 2
    assert task12["support_owned_resources"] == []
    assert task12["support_stack_may_delete_retained_resources"] is False
    assert task12["deployment_authority_records"] == 10
    assert task12["retained_cost_terms"] == [
        "lambda",
        "step_functions",
        "logs",
        "eventbridge",
        "scheduler",
    ]


def test_task10_contract_closes_six_az_sender_and_terminal_writer() -> None:
    """Break caught: checked-in contracts still describe the placeholder."""

    from glm52_enforcement.support_plane import build_support_contract_artifacts

    task10 = build_support_contract_artifacts()["support-task10-production-v1.json"]
    assert task10 == {
        "schema_version": 1,
        "record_type": "glm52_h1g_task10_production_contract_v1",
        "handler": "task10_sole_sender_handler.main",
        "runtime_adapter": "task10_sole_sender_runtime.py",
        "authority_record_type": ("glm52_task10_retained_sole_sender_authority_v1"),
        "terminal_record_type": ("glm52_task10_capacity_reconciliation_v1"),
        "retained_resources": 9,
        "merged_retained_resources": 83,
        "workflow_type": "STANDARD",
        "workflow_invocation_count": 1,
        "availability_zones": [
            "us-west-2a",
            "us-west-2b",
            "us-west-2c",
            "us-west-2d",
            "us-west-2e",
            "us-west-2f",
        ],
        "maximum_ec2_calls": 6,
        "ec2_calls_per_az": 1,
        "stop_on_first_worker_success": True,
        "launch_shape": {
            "market": "on-demand",
            "instance_type": "p5.48xlarge",
            "imds_v2_required": True,
            "root_volume": {
                "delete_on_termination": True,
                "encrypted": True,
                "size_gib": 300,
                "type": "gp3",
                "iops": 3000,
                "throughput_mibps": 125,
            },
            "data_disk_count": 0,
            "exact_tag_count": 15,
            "spot_allowed": False,
            "capacity_block_allowed": False,
        },
        "direct_and_ambiguous_describe_readback_required": True,
        "termination_only_liability_preserved": True,
        "terminal_writer": {
            "bucket": ("keep-glm52-models-246813579024-us-west-2"),
            "key_pattern": (
                "campaigns/glm52-sky-20260724/submissions/production/"
                "generations/<generation_text>/workflow/"
                "LAUNCH_OUTCOME.json"
            ),
            "exact_version_required": True,
            "lost_response_readback_required": True,
        },
    }


def _support_runtime_expected_contract() -> dict[str, object]:
    from glm52_enforcement.canonical import canonical_sha256

    value: dict[str, object] = {
        "authority_class": "SUPPORT_RUNTIME",
        "function_logical_id": "FenceExecutorFunction",
        "role_logical_id": "FenceExecutorRole",
        "function_code_sha256": "1" * 64,
        "execution_role_arn": (
            "arn:aws:iam::246813579024:role/keep-glm52-h1g-fence-executor"
        ),
        "role_trust_policy_sha256": "2" * 64,
        "role_permission_policy_sha256": "3" * 64,
        "expected_attachment_identity_sha256": "4" * 64,
        "disabled_support_profile_sha256": "5" * 64,
    }
    value["canonical_identity_sha256"] = canonical_sha256(value)
    return value


def _support_runtime_live_readback() -> dict[str, object]:
    expected = _support_runtime_expected_contract()
    return {
        "function_version_arn": (
            "arn:aws:lambda:us-west-2:246813579024:"
            "function:keep-glm52-h1g-fence-executor:17"
        ),
        "function_code_sha256": expected["function_code_sha256"],
        "execution_role_arn": expected["execution_role_arn"],
        "execution_role_id": "AROASUPPORTRUNTIME123",
        "role_trust_policy_sha256": expected["role_trust_policy_sha256"],
        "role_permission_policy_sha256": expected["role_permission_policy_sha256"],
        "stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "stack/keep-glm52-h1g-support/"
            "11111111-2222-4333-8444-555555555555"
        ),
        "stack_template_sha256": "6" * 64,
        "resource_policy_sha256": "7" * 64,
        "event_source_state_sha256": "8" * 64,
        "function_url_state_sha256": "9" * 64,
        "expected_attachment_identity_sha256": expected[
            "expected_attachment_identity_sha256"
        ],
        "observed_attachment_identity_sha256": expected[
            "expected_attachment_identity_sha256"
        ],
        "disabled_support_profile_sha256": expected["disabled_support_profile_sha256"],
    }


def _bootstrap_manifest_coordinate_v2() -> dict[str, str]:
    return {
        "bucket": "keep-glm52-models-246813579024-us-west-2",
        "key": (
            "campaigns/glm52-sky-20260724/authorities/fence/manifests/"
            "glm52-v2-amber-quartz/00000001/FENCE_BOOTSTRAP_MANIFEST.json"
        ),
        "version_id": "bootstrap-version/+?",
        "file_sha256": "a" * 64,
        "canonical_identity_sha256": "b" * 64,
    }


def test_support_runtime_identity_is_singular_and_adopts_only_equal_body() -> None:
    from glm52_enforcement.dynamodb import decode_item
    from glm52_enforcement.support_plane import (
        build_support_runtime_identity,
        commit_support_runtime_identity,
    )

    identity = build_support_runtime_identity(
        activation_id="glm52-v2-amber-quartz",
        generation=1,
        bootstrap_manifest_coordinate=_bootstrap_manifest_coordinate_v2(),
        expected_contract=_support_runtime_expected_contract(),
        live_readback=_support_runtime_live_readback(),
    )
    coordinate = {
        "bucket": "keep-glm52-models-246813579024-us-west-2",
        "key": (
            "campaigns/glm52-sky-20260724/authorities/fence/runtime/"
            "glm52-v2-amber-quartz/00000001/"
            "SUPPORT_RUNTIME_IDENTITY.json"
        ),
        "version_id": "runtime-version/+?",
        "file_sha256": "c" * 64,
        "canonical_identity_sha256": identity.canonical_identity_sha256,
    }

    class LostResponseDynamoDB:
        def __init__(self) -> None:
            self.item: dict[str, object] | None = None
            self.puts = 0

        def put_item(self, **request: object) -> object:
            if self.item is not None:
                raise RuntimeError("conditional write failed")
            self.puts += 1
            assert request["ConditionExpression"] == (
                "attribute_not_exists(PK) AND attribute_not_exists(SK)"
            )
            self.item = request["Item"]  # type: ignore[assignment]
            raise RuntimeError("lost response")

        def get_item(self, **request: object) -> object:
            assert request["ConsistentRead"] is True
            assert self.item is not None
            return {
                "Item": self.item,
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "get-1",
                    "RetryAttempts": 0,
                },
            }

    dynamodb = LostResponseDynamoDB()
    result = commit_support_runtime_identity(
        identity,
        coordinate=coordinate,
        dynamodb=dynamodb,
        table_name="keep-glm52-h1g-ledger-v1",
    )
    assert result["outcome"] == "RECONCILED"
    assert dynamodb.puts == 1
    assert dynamodb.item is not None
    decoded = decode_item(dynamodb.item)
    assert decoded["record_type"] == "glm52_h1g_support_runtime_identity_v1"
    assert decoded["canonical_identity_sha256"] == (identity.canonical_identity_sha256)

    foreign = dict(decoded)
    foreign["live_identity"] = {
        **foreign["live_identity"],  # type: ignore[arg-type]
        "function_code_sha256": "f" * 64,
    }
    from glm52_enforcement.dynamodb import encode_item

    dynamodb.item = encode_item(foreign)
    with pytest.raises(ValueError, match="different|foreign|drift"):
        commit_support_runtime_identity(
            identity,
            coordinate=coordinate,
            dynamodb=dynamodb,
            table_name="keep-glm52-h1g-ledger-v1",
        )


def test_support_runtime_identity_requires_fresh_expected_live_equality() -> None:
    from glm52_enforcement.support_plane import (
        build_support_runtime_identity,
        require_live_support_runtime_identity,
    )

    expected = _support_runtime_expected_contract()
    live = _support_runtime_live_readback()
    identity = build_support_runtime_identity(
        activation_id="glm52-v2-amber-quartz",
        generation=1,
        bootstrap_manifest_coordinate=_bootstrap_manifest_coordinate_v2(),
        expected_contract=expected,
        live_readback=live,
    )
    assert (
        require_live_support_runtime_identity(
            identity,
            expected_contract=expected,
            live_readback=live,
        )
        is identity
    )
    for field, replacement in (
        ("function_code_sha256", "d" * 64),
        ("execution_role_id", "AROARECREATEDSUPPORT99"),
        ("stack_template_sha256", "e" * 64),
        ("event_source_state_sha256", "f" * 64),
    ):
        drifted = dict(live)
        drifted[field] = replacement
        with pytest.raises(ValueError, match="live|drift|equality"):
            require_live_support_runtime_identity(
                identity,
                expected_contract=expected,
                live_readback=drifted,
            )

    deleted = dict(live)
    deleted["function_version_arn"] = None
    with pytest.raises(ValueError, match="live|absent|drift"):
        require_live_support_runtime_identity(
            identity,
            expected_contract=expected,
            live_readback=deleted,
        )


def test_retained_fence_runtime_is_least_privilege_and_survives_support() -> None:
    from glm52_enforcement.canonical import canonical_json_bytes
    from glm52_enforcement.fence_source_settlement import (
        SOURCE_SETTLEMENT_EVIDENCE_READER_NAMES,
    )
    from glm52_enforcement.support_plane import (
        build_pre_support_retained_runtime_fragment,
    )
    from glm52_enforcement.task13_fixed_artifacts import (
        ACCOUNT_ID,
        REGION,
        RUN_ID,
    )

    inputs = _support_build_inputs()
    fragment = build_pre_support_retained_runtime_fragment(inputs)
    resources = fragment["Resources"]
    retained = {
        "PreSupportFenceExecutorRole",
        "PreSupportFenceExecutorFunction",
        "PreSupportFenceExecutorVersion",
        "FenceSourceSettlementMaterializerRole",
        "FenceSourceSettlementMaterializerFunction",
        "FenceSourceSettlementMaterializerVersion",
        "FenceSourceSettledArtifactPublisherRole",
    }
    assert retained <= set(resources)
    assert fragment["Metadata"]["SupportDeletionSurvivors"] == sorted(retained)
    source_function = resources["FenceSourceSettlementMaterializerFunction"][
        "Properties"
    ]
    assert source_function["Handler"] == (
        "support_fence_handler.source_settlement_main"
    )
    assert source_function["Timeout"] == 60
    assert source_function["ReservedConcurrentExecutions"] == 1
    assert source_function["Environment"]["Variables"] == {
        "GLM52_ACCOUNT_ID": ACCOUNT_ID,
        "GLM52_REGION": REGION,
        "GLM52_RUN_ID": RUN_ID,
        "GLM52_ACTIVATION_ID": inputs.activation_id,
        "GLM52_MODEL_BUCKET_ARN": inputs.model_bucket_arn,
        "GLM52_RETAINED_KMS_KEY_ARN": inputs.retained_kms_key_arn,
        "GLM52_BOOTSTRAP_MANIFEST_COORDINATE": canonical_json_bytes(
            dict(inputs.bootstrap_manifest_coordinate)
        ).decode("ascii"),
        "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": inputs.lambda_code_sha256,
        "GLM52_SOURCE_PUBLISHER_ROLE_ARN": (
            f"arn:aws:iam::{ACCOUNT_ID}:role/"
            "keep-glm52-h1g-fence-source-settled-artifact-publisher"
        ),
        "GLM52_AUTHORITY_CLASS": "SOURCE_SETTLEMENT_MATERIALIZER",
    }
    rendered = json.dumps(
        {name: resources[name] for name in retained},
        sort_keys=True,
    )
    assert "s3:PutBucketPolicy" not in rendered
    assert "s3:DeleteBucketPolicy" not in rendered
    assert "ec2:RunInstances" not in rendered
    assert "iam:PassRole" not in json.dumps(
        resources["FenceSourceSettlementMaterializerRole"],
        sort_keys=True,
    )
    assert "cloudformation:CreateChangeSet" not in json.dumps(
        resources["FenceSourceSettlementMaterializerRole"],
        sort_keys=True,
    )
    assert "cloudformation:ExecuteChangeSet" not in json.dumps(
        resources["FenceSourceSettlementMaterializerRole"],
        sort_keys=True,
    )

    source_policy = resources["FenceSourceSettledArtifactPublisherRole"]["Properties"][
        "Policies"
    ][0]["PolicyDocument"]["Statement"]
    source_writes = {
        resource
        for statement in source_policy
        if statement["Action"] == "s3:PutObject"
        for resource in statement["Resource"]
    }
    assert len(source_writes) == 3
    materializer_statements = resources["FenceSourceSettlementMaterializerRole"][
        "Properties"
    ]["Policies"][0]["PolicyDocument"]["Statement"]
    evidence_reads = {
        resource
        for statement in materializer_statements
        if statement["Sid"] == "ReadOnlyPinnedBootstrapAndSourceEvidence"
        for resource in statement["Resource"]
        if "/authorities/fence/evidence/" in resource
    }
    assert evidence_reads == {
        (
            inputs.model_bucket_arn
            + f"/campaigns/{RUN_ID}/authorities/fence/evidence/"
            + f"{inputs.activation_id}/00000001/{reader_name}.json"
        )
        for reader_name in SOURCE_SETTLEMENT_EVIDENCE_READER_NAMES
    }
    read_statement = next(
        statement
        for statement in materializer_statements
        if statement["Sid"] == "ReadOnlyPinnedBootstrapAndSourceEvidence"
    )
    decrypt_statement = next(
        statement
        for statement in materializer_statements
        if statement["Sid"] == "DecryptOnlyPinnedBootstrapAndSourceEvidence"
    )
    assert decrypt_statement["Resource"] == inputs.retained_kms_key_arn
    assert set(
        decrypt_statement["Condition"]["ForAnyValue:StringLike"][
            "kms:EncryptionContext:aws:s3:arn"
        ]
    ) == set(read_statement["Resource"])


def test_support_source_roles_no_longer_trust_current_fence_executor() -> None:
    bundle = _bundle()
    resources = bundle.support_template["Resources"]
    executor_arn = {"Fn::GetAtt": ["FenceExecutorRole", "Arn"]}
    for name in (
        "SourceGpuSpend",
        "SourceSubmissionIntent",
        "SourceControllerBaseline",
        "SourceControlPlaneReadiness",
        "SourceSubmissionAcquisition",
    ):
        trust = resources[name + "Role"]["Properties"]["AssumeRolePolicyDocument"][
            "Statement"
        ]
        assert trust == [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ]
        assert executor_arn not in trust
    executor_statements = resources["FenceExecutorRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    assert all(
        statement.get("Sid") != "AssumeExactStablePublisherRoles"
        for statement in executor_statements
    )


def test_support_executor_has_no_static_inventory_or_change_set_arn() -> None:
    bundle = _bundle()
    resources = bundle.support_template["Resources"]
    environment = resources["FenceExecutorFunction"]["Properties"]["Environment"][
        "Variables"
    ]
    assert set(environment) == {
        "GLM52_ACCOUNT_ID",
        "GLM52_REGION",
        "GLM52_RUN_ID",
        "GLM52_ACTIVATION_ID",
        "GLM52_LEDGER_TABLE_NAME",
        "GLM52_CAMPAIGN_BUCKET",
        "GLM52_AUTHORITY_CLASS",
        "GLM52_BOOTSTRAP_MANIFEST_COORDINATE",
        "GLM52_FENCE_EXECUTOR_ROLE_ARN",
        "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256",
    }
    rendered = json.dumps(
        {
            "role": resources["FenceExecutorRole"],
            "function": resources["FenceExecutorFunction"],
        },
        sort_keys=True,
    )
    assert "GLM52_FENCE_TEMPLATE_INVENTORY" not in rendered
    assert "changeSet/" not in rendered
    assert "change_set_arn" not in rendered
    assert "s3:PutBucketPolicy" not in rendered
