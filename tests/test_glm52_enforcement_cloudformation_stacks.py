from __future__ import annotations

import json
from pathlib import Path
import copy

from glm52_enforcement.cloudformation_stacks import (
    ACCOUNT_ID,
    REGION,
    RUN_ID,
    STACK_OWNERSHIP,
    STACK_TAGS,
    StackIdentity,
    StackKind,
    build_fence_import_template,
    build_fence_transfer_template_v2,
    build_post_retain_template,
    build_retention_only_template,
    build_support_disabled_transition,
    canonical_json_bytes,
    stack_name,
    validate_bootstrap_template,
    validate_one_way_stack_graph,
    validate_stack_ownership,
)

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "aws/glm52-gpu/cfn/h1g/container-bootstrap-v1.json"
FENCE_BOOTSTRAP = ROOT / "aws/glm52-gpu/cfn/h1g/fence-bootstrap-v1.json"
SUPPORT_BOOTSTRAP = ROOT / "aws/glm52-gpu/cfn/h1g/support-bootstrap-v1.json"


def _archived_template() -> dict[str, object]:
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Parameters": {"Enabled": {"Type": "String"}},
        "Resources": {
            "ModelBucket": {
                "Type": "AWS::S3::Bucket",
                "Properties": {"BucketName": "exact-bucket"},
            },
            "CurrentPolicyOwner": {
                "Type": "AWS::S3::BucketPolicy",
                "Condition": "Enabled",
                "Properties": {
                    "Bucket": {"Ref": "ModelBucket"},
                    "PolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [],
                    },
                },
            },
            "Role": {"Type": "AWS::IAM::Role", "Properties": {}},
        },
        "Outputs": {"Bucket": {"Value": {"Ref": "ModelBucket"}}},
    }


def test_exact_three_stack_names_tags_and_nonoverlapping_ownership() -> None:
    assert {
        kind: stack_name(kind)
        for kind in StackKind
    } == {
        StackKind.RETAINED: "keep-glm52-gpu",
        StackKind.FENCE: "keep-glm52-h1g-fence",
        StackKind.SUPPORT: "keep-glm52-h1g-support",
    }
    assert ACCOUNT_ID == "246813579024"
    assert REGION == "us-west-2"
    assert RUN_ID == "glm52-sky-20260724"
    assert dict(STACK_TAGS) == {
        "Authority": "H1g",
        "Campaign": "GLM-5.2",
        "Environment": "production",
        "ManagedBy": "CloudFormation",
        "Project": "KEEP",
        "RunId": "glm52-sky-20260724",
    }
    assert set(STACK_OWNERSHIP) == set(StackKind)
    assert all(STACK_OWNERSHIP[kind] for kind in StackKind)
    assert validate_stack_ownership(STACK_OWNERSHIP) == STACK_OWNERSHIP
    all_families = [
        family
        for families in STACK_OWNERSHIP.values()
        for family in families
    ]
    assert len(all_families) == len(set(all_families))
    with __import__("pytest").raises(ValueError):
        StackIdentity(
            kind=StackKind.FENCE,
            name="keep-glm52-h1g-fence",
            stack_id=None,
            account_id="246813579024",
            region="us-west-2",
            termination_protection=True,
            tags=STACK_TAGS,
        )


def test_bootstrap_template_is_parameterless_single_wait_handle() -> None:
    template = json.loads(BOOTSTRAP.read_text(encoding="ascii"))
    assert validate_bootstrap_template(template) == template
    assert template == {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "ContainerAnchor": {
                "Type": "AWS::CloudFormation::WaitConditionHandle"
            }
        },
    }
    forbidden = {
        "Parameters",
        "Conditions",
        "Outputs",
        "Transform",
        "Rules",
        "Mappings",
    }
    assert forbidden.isdisjoint(template)
    assert "WaitCondition" not in json.dumps(template).replace(
        "WaitConditionHandle", ""
    )
    assert "{{resolve:" not in json.dumps(template)


def test_fence_and_support_bootstrap_assets_are_distinct_immutable_coordinates() -> None:
    generic = BOOTSTRAP.read_bytes()
    assert FENCE_BOOTSTRAP != SUPPORT_BOOTSTRAP
    assert FENCE_BOOTSTRAP.read_bytes() == generic
    assert SUPPORT_BOOTSTRAP.read_bytes() == generic
    assert validate_bootstrap_template(
        json.loads(FENCE_BOOTSTRAP.read_text(encoding="ascii"))
    )
    assert validate_bootstrap_template(
        json.loads(SUPPORT_BOOTSTRAP.read_text(encoding="ascii"))
    )


def test_retention_template_changes_only_bucket_policy_retain_attributes() -> None:
    before = _archived_template()
    unchanged = copy.deepcopy(before)
    result = build_retention_only_template(
        archived_template=before,
        current_policy_logical_id="CurrentPolicyOwner",
    )
    assert before == unchanged
    expected = copy.deepcopy(before)
    expected["Resources"]["CurrentPolicyOwner"]["DeletionPolicy"] = "Retain"
    expected["Resources"]["CurrentPolicyOwner"]["UpdateReplacePolicy"] = "Retain"
    assert result == expected
    result_without_attributes = copy.deepcopy(result)
    policy = result_without_attributes["Resources"]["CurrentPolicyOwner"]
    policy.pop("DeletionPolicy")
    policy.pop("UpdateReplacePolicy")
    assert result_without_attributes == before


def test_post_retain_template_removes_only_the_owned_bucket_policy() -> None:
    retained = build_retention_only_template(
        archived_template=_archived_template(),
        current_policy_logical_id="CurrentPolicyOwner",
    )
    result = build_post_retain_template(
        retention_template=retained,
        current_policy_logical_id="CurrentPolicyOwner",
    )
    expected = copy.deepcopy(retained)
    expected["Resources"].pop("CurrentPolicyOwner")
    assert result == expected
    assert "ModelBucket" in result["Resources"]
    assert "Role" in result["Resources"]


def _live_policy() -> dict[str, object]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "DenyInsecureTransport",
                "Effect": "Deny",
                "Principal": "*",
                "Action": "s3:*",
                "Resource": [
                    "arn:aws:s3:::exact-bucket",
                    "arn:aws:s3:::exact-bucket/*",
                ],
                "Condition": {"Bool": {"aws:SecureTransport": "false"}},
            }
        ],
    }


def test_fence_import_keeps_anchor_and_imports_exact_canonical_policy() -> None:
    bootstrap = json.loads(BOOTSTRAP.read_text(encoding="ascii"))
    policy = _live_policy()
    result = build_fence_import_template(
        bootstrap_template=bootstrap,
        bucket_name="exact-bucket",
        canonical_live_policy=policy,
    )
    assert result["Resources"]["ContainerAnchor"] == {
        "Type": "AWS::CloudFormation::WaitConditionHandle"
    }
    assert result["Resources"]["H1gProductionFenceBucketPolicy"] == {
        "Type": "AWS::S3::BucketPolicy",
        "DeletionPolicy": "Retain",
        "UpdateReplacePolicy": "Retain",
        "Properties": {
            "Bucket": "exact-bucket",
            "PolicyDocument": policy,
        },
    }
    assert "Condition" not in result["Resources"]["H1gProductionFenceBucketPolicy"]
    assert result["Resources"]["H1gProductionFenceBucketPolicy"][
        "Properties"
    ]["PolicyDocument"] is not policy


def test_final_fence_template_removes_only_anchor_and_has_one_unconditional_policy() -> None:
    imported = build_fence_import_template(
        bootstrap_template=json.loads(BOOTSTRAP.read_text(encoding="ascii")),
        bucket_name="exact-bucket",
        canonical_live_policy=_live_policy(),
    )
    result = build_fence_transfer_template_v2(fence_import_template=imported)
    expected = copy.deepcopy(imported)
    expected["Resources"].pop("ContainerAnchor")
    assert result == expected
    assert set(result) == {"AWSTemplateFormatVersion", "Resources"}
    assert set(result["Resources"]) == {"H1gProductionFenceBucketPolicy"}
    assert "Condition" not in result["Resources"]["H1gProductionFenceBucketPolicy"]


def test_support_disabled_transition_replaces_only_anchor_with_reviewed_inventory() -> None:
    bootstrap = json.loads(BOOTSTRAP.read_text(encoding="ascii"))
    inventory = {
        "SupportFunction": {
            "Type": "AWS::Lambda::Function",
            "Properties": {
                "Handler": "handler.main",
                "Runtime": "python3.13",
                "Role": "arn:aws:iam::246813579024:role/exact-support-role",
                "Code": {"S3Bucket": "exact-bucket", "S3Key": "exact.zip"},
            },
        },
        "SupportLogGroup": {
            "Type": "AWS::Logs::LogGroup",
            "Properties": {"RetentionInDays": 30},
        },
        "RehearsalBucket": {
            "Type": "AWS::S3::Bucket",
            "Properties": {
                "BucketName": "keep-glm52-h1g-rehearsal-246813579024-us-west-2"
            },
        },
    }
    inventory_sha256 = __import__("hashlib").sha256(
        canonical_json_bytes(inventory)
    ).hexdigest()
    result = build_support_disabled_transition(
        bootstrap_template=bootstrap,
        reviewed_inventory=inventory,
        reviewed_inventory_sha256=inventory_sha256,
        retained_resource_physical_ids=("exact-bucket",),
        rehearsal_bucket_name=(
            "keep-glm52-h1g-rehearsal-246813579024-us-west-2"
        ),
    )
    assert result["Resources"] == inventory
    expected = copy.deepcopy(bootstrap)
    expected["Resources"] = inventory
    assert result == expected
    assert result["Resources"] is not inventory


def test_support_inventory_rejects_retained_bucket_alias_and_arbitrary_bucket() -> None:
    bootstrap = json.loads(BOOTSTRAP.read_text(encoding="ascii"))
    for bucket_name in (
        "exact-bucket",
        "keep-glm52-unreviewed-support-bucket",
    ):
        inventory = {
            "AliasResource": {
                "Type": "AWS::S3::Bucket",
                "Properties": {"BucketName": bucket_name},
            }
        }
        with __import__("pytest").raises(ValueError):
            build_support_disabled_transition(
                bootstrap_template=bootstrap,
                reviewed_inventory=inventory,
                reviewed_inventory_sha256=__import__("hashlib").sha256(
                    canonical_json_bytes(inventory)
                ).hexdigest(),
                retained_resource_physical_ids=("exact-bucket",),
                rehearsal_bucket_name=(
                    "keep-glm52-h1g-rehearsal-246813579024-us-west-2"
                ),
            )


def test_exact_export_owner_graph_rejects_support_alias_without_name_substring() -> None:
    retained = _archived_template()
    retained["Resources"]["Role"]["Properties"]["Alias"] = {
        "Fn::ImportValue": "SupportHostId"
    }
    fence = build_fence_transfer_template_v2(
        fence_import_template=build_fence_import_template(
            bootstrap_template=json.loads(BOOTSTRAP.read_text(encoding="ascii")),
            bucket_name="exact-bucket",
            canonical_live_policy=_live_policy(),
        )
    )
    support = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "SupportLogGroup": {"Type": "AWS::Logs::LogGroup"}
        },
    }
    with __import__("pytest").raises(ValueError):
        validate_one_way_stack_graph(
            retained_template=retained,
            fence_template=fence,
            support_template=support,
            retained_export_names=("RetainedRoleArn",),
            support_export_names=("SupportHostId",),
        )


def test_retained_stacks_never_import_support_outputs() -> None:
    retained = _archived_template()
    fence = build_fence_transfer_template_v2(
        fence_import_template=build_fence_import_template(
            bootstrap_template=json.loads(BOOTSTRAP.read_text(encoding="ascii")),
            bucket_name="exact-bucket",
            canonical_live_policy=_live_policy(),
        )
    )
    support = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "SupportFunction": {
                "Type": "AWS::Lambda::Function",
                "Properties": {
                    "Role": {
                        "Fn::ImportValue": "keep-glm52-gpu-RetainedRoleArn"
                    }
                },
            }
        },
    }
    assert validate_one_way_stack_graph(
        retained_template=retained,
        fence_template=fence,
        support_template=support,
        retained_export_names=("keep-glm52-gpu-RetainedRoleArn",),
        support_export_names=("SupportHostId",),
    )
    poisoned = copy.deepcopy(retained)
    poisoned["Resources"]["Role"]["Properties"]["SupportValue"] = {
        "Fn::ImportValue": "keep-glm52-h1g-support-HostId"
    }
    try:
        validate_one_way_stack_graph(
            retained_template=poisoned,
            fence_template=fence,
            support_template=support,
            retained_export_names=("keep-glm52-gpu-RetainedRoleArn",),
            support_export_names=("keep-glm52-h1g-support-HostId",),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("retained support import was accepted")
