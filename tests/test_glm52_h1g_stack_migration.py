from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from glm52_enforcement.cloudformation_stacks import (
    FENCE_LOGICAL_ID,
    FENCE_STACK_POLICY_SHA256_V2,
    MigrationMutationStatus,
    MigrationRoleEvidenceV2,
    STACK_MIGRATION_TRANSFER_CHECKPOINT_RECORD_TYPE_V2,
    STACK_MIGRATION_TRANSFER_COMPLETED_KEYS_V2,
    StackIdentity,
    StackKind,
    build_bridge_seed_ownership_plan_v2,
    build_fence_import_template,
    build_fence_transfer_template_v2,
    build_stack_migration_transfer_checkpoint_v2,
    canonical_json_bytes,
    parse_stack_migration_transfer_checkpoint_v2,
)
from glm52_enforcement.fence_artifacts import (
    FENCE_ARTIFACT_KEYS,
    parse_bridge_seed_artifact,
)
from glm52_enforcement.task13_fixed_artifacts import CAMPAIGN_BUCKET


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads(
    (
        ROOT
        / ".superpowers/sdd/goal-objective/"
        "task-3ph1g-h1g-fence-policy-renderer-golden-v1.json"
    ).read_text(encoding="ascii")
)
BOOTSTRAP = {
    "AWSTemplateFormatVersion": "2010-09-09",
    "Resources": {
        "ContainerAnchor": {
            "Type": "AWS::CloudFormation::WaitConditionHandle"
        }
    },
}


def _seed():
    raw = canonical_json_bytes(GOLDEN["policy"])
    digest = hashlib.sha256(raw).hexdigest()
    return parse_bridge_seed_artifact(
        {
            "artifact_kind": "BRIDGE_SEED_POLICY",
            "record_type": "glm52_h1g_bridge_seed_policy_v1",
            "bucket": CAMPAIGN_BUCKET,
            "key": FENCE_ARTIFACT_KEYS[0],
            "version_id": "seed-version-1",
            "expected_live_preseed_policy_sha256": "a" * 64,
            "file_sha256": digest,
            "body_sha256": digest,
            "policy_sha256": digest,
            "rendered_policy_bytes": len(raw),
            "statement_ledger": GOLDEN["statement_ledger"],
            "publisher_deny_policy_sha256": None,
        },
        raw_bytes=raw,
    )


def _tags() -> tuple[tuple[str, str], ...]:
    return (
        ("Authority", "H1g"),
        ("Campaign", "GLM-5.2"),
        ("Environment", "production"),
        ("ManagedBy", "CloudFormation"),
        ("Project", "KEEP"),
        ("RunId", "glm52-sky-20260724"),
    )


def _identity(kind: StackKind) -> StackIdentity:
    names = {
        StackKind.RETAINED: "keep-glm52-gpu",
        StackKind.FENCE: "keep-glm52-h1g-fence",
        StackKind.SUPPORT: "keep-glm52-h1g-support",
    }
    name = names[kind]
    suffix = {
        StackKind.RETAINED: "11111111-1111-4111-8111-111111111111",
        StackKind.FENCE: "22222222-2222-4222-8222-222222222222",
        StackKind.SUPPORT: "33333333-3333-4333-8333-333333333333",
    }[kind]
    return StackIdentity(
        kind=kind,
        name=name,
        stack_id=(
            "arn:aws:cloudformation:us-west-2:246813579024:"
            f"stack/{name}/{suffix}"
        ),
        account_id="246813579024",
        region="us-west-2",
        termination_protection=True,
        tags=_tags(),
    )


def _role(name: str, role_id: str, marker: str) -> MigrationRoleEvidenceV2:
    return MigrationRoleEvidenceV2(
        role_arn=f"arn:aws:iam::246813579024:role/{name}",
        role_id=role_id,
        trust_policy_sha256=marker * 64,
        permission_policy_sha256=marker * 64,
    )


def _roles():
    return (
        _role("ExactMigrationApiCaller", "AROAAPICALLER12345678", "1"),
        _role(
            "keep-glm52-h1g-cloudformation-deployment",
            "AROAMIGRATIONSVC123456",
            "2",
        ),
        _role(
            "keep-glm52-h1g-fence-service",
            "AROAFENCESERVICE1234567",
            "3",
        ),
    )


def _retained_without_policy() -> dict[str, object]:
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "ModelBucket": {
                "Type": "AWS::S3::Bucket",
                "Properties": {"BucketName": "exact-bucket"},
            },
            "RetainedRole": {"Type": "AWS::IAM::Role", "Properties": {}},
        },
    }


def _checkpoint(plan=None):
    seed = _seed()
    if plan is None:
        plan = build_bridge_seed_ownership_plan_v2(
            retained_template=_retained_without_policy(),
            bucket_name="exact-bucket",
            bridge_seed=seed,
            current_policy_logical_id=None,
        )
    api, migration, fence = _roles()
    transfer_hash = "5" * 64
    return build_stack_migration_transfer_checkpoint_v2(
        bridge_seed_plan=plan,
        retained=_identity(StackKind.RETAINED),
        fence=_identity(StackKind.FENCE),
        support=_identity(StackKind.SUPPORT),
        bucket_name="exact-bucket",
        fence_import_template_sha256="4" * 64,
        fence_transfer_template_sha256=transfer_hash,
        active_original_template_sha256=transfer_hash,
        active_processed_template_sha256=transfer_hash,
        direct_policy_sha256=(
            plan.seed_policy_sha256,
            plan.seed_policy_sha256,
        ),
        api_caller=api,
        migration_service_role=migration,
        fence_service_role=fence,
        associated_stack_role_arn=migration.role_arn,
        associated_stack_role_id=migration.role_id,
        import_change_set_id=(
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "changeSet/h1g-import-production-fence/"
            "cccccccc-1111-4222-8333-444444444444"
        ),
        support_prestate_template_sha256="6" * 64,
    )


def test_bridge_seed_absent_branch_adds_one_exact_retained_owner() -> None:
    seed = _seed()
    plan = build_bridge_seed_ownership_plan_v2(
        retained_template=_retained_without_policy(),
        bucket_name="exact-bucket",
        bridge_seed=seed,
        current_policy_logical_id=None,
    )

    assert plan.owner_branch == "ADD"
    assert plan.owner_logical_id == FENCE_LOGICAL_ID
    policy_owner = plan.template["Resources"][FENCE_LOGICAL_ID]
    assert policy_owner == {
        "Type": "AWS::S3::BucketPolicy",
        "Properties": {
            "Bucket": "exact-bucket",
            "PolicyDocument": json.loads(seed.raw_bytes),
        },
    }
    assert canonical_json_bytes(policy_owner["Properties"]["PolicyDocument"]) == seed.raw_bytes


def test_bridge_seed_present_branch_modifies_only_the_authenticated_owner() -> None:
    seed = _seed()
    retained = _retained_without_policy()
    retained["Resources"]["LegacyOwner"] = {
        "Type": "AWS::S3::BucketPolicy",
        "Condition": "LegacyPolicyEnabled",
        "Properties": {"Bucket": "exact-bucket", "PolicyDocument": {}},
    }
    before = copy.deepcopy(retained["Resources"]["RetainedRole"])

    plan = build_bridge_seed_ownership_plan_v2(
        retained_template=retained,
        bucket_name="exact-bucket",
        bridge_seed=seed,
        current_policy_logical_id="LegacyOwner",
    )

    assert plan.owner_branch == "MODIFY"
    assert set(plan.template["Resources"]) == {
        "ModelBucket",
        "RetainedRole",
        "LegacyOwner",
    }
    assert plan.template["Resources"]["RetainedRole"] == before
    assert "Condition" not in plan.template["Resources"]["LegacyOwner"]
    assert canonical_json_bytes(
        plan.template["Resources"]["LegacyOwner"]["Properties"]["PolicyDocument"]
    ) == seed.raw_bytes


@pytest.mark.parametrize(
    ("current_owner", "extra_owner"),
    ((None, True), ("WrongOwner", False), ("LegacyOwner", True)),
)
def test_bridge_seed_branches_fail_closed_on_owner_cardinality(
    current_owner: str | None, extra_owner: bool
) -> None:
    retained = _retained_without_policy()
    retained["Resources"]["LegacyOwner"] = {
        "Type": "AWS::S3::BucketPolicy",
        "Properties": {"Bucket": "exact-bucket", "PolicyDocument": {}},
    }
    if extra_owner:
        retained["Resources"]["SecondOwner"] = {
            "Type": "AWS::S3::BucketPolicy",
            "Properties": {"Bucket": "exact-bucket", "PolicyDocument": {}},
        }
    with pytest.raises(ValueError, match="owner"):
        build_bridge_seed_ownership_plan_v2(
            retained_template=retained,
            bucket_name="exact-bucket",
            bridge_seed=_seed(),
            current_policy_logical_id=current_owner,
        )


def test_fence_transfer_has_no_legacy_final_fence_and_preserves_seed_bytes() -> None:
    seed = _seed()
    imported = build_fence_import_template(
        bootstrap_template=BOOTSTRAP,
        bucket_name="exact-bucket",
        canonical_live_policy=json.loads(seed.raw_bytes),
    )
    transferred = build_fence_transfer_template_v2(
        fence_import_template=imported
    )

    assert set(transferred["Resources"]) == {FENCE_LOGICAL_ID}
    owner = transferred["Resources"][FENCE_LOGICAL_ID]
    assert owner["DeletionPolicy"] == "Retain"
    assert owner["UpdateReplacePolicy"] == "Retain"
    assert canonical_json_bytes(owner["Properties"]["PolicyDocument"]) == seed.raw_bytes


def test_operation_6_checkpoint_is_exact_immutable_transfer_boundary() -> None:
    checkpoint = _checkpoint()

    assert checkpoint.record_type == STACK_MIGRATION_TRANSFER_CHECKPOINT_RECORD_TYPE_V2
    assert checkpoint.completed_mutation_keys == STACK_MIGRATION_TRANSFER_COMPLETED_KEYS_V2
    assert checkpoint.operation_7_status == MigrationMutationStatus.NOT_SUBMITTED.value
    assert checkpoint.support_state == "INERT_ANCHOR"
    assert checkpoint.direct_policy_sha256 == (
        checkpoint.bridge_seed_policy_sha256,
        checkpoint.bridge_seed_policy_sha256,
    )
    assert checkpoint.active_original_template_sha256 == checkpoint.fence_transfer_template_sha256
    assert checkpoint.active_processed_template_sha256 == checkpoint.fence_transfer_template_sha256
    assert checkpoint.associated_stack_role_arn == checkpoint.migration_service_role.role_arn
    assert checkpoint.rollback_stack_role_arn == checkpoint.migration_service_role.role_arn
    assert checkpoint.stack_policy_sha256 == FENCE_STACK_POLICY_SHA256_V2
    assert checkpoint.deletion_policy == "Retain"
    assert checkpoint.update_replace_policy == "Retain"
    assert parse_stack_migration_transfer_checkpoint_v2(checkpoint.to_dict()) == checkpoint


def test_checkpoint_rejects_role_id_drift_and_early_operation_7() -> None:
    checkpoint = _checkpoint()
    projection = dict(checkpoint.to_dict())
    projection["associated_stack_role_id"] = checkpoint.fence_service_role.role_id
    with pytest.raises(ValueError, match="not exact"):
        parse_stack_migration_transfer_checkpoint_v2(projection)

    projection = dict(checkpoint.to_dict())
    projection["operation_7_status"] = "COMPLETE"
    with pytest.raises(ValueError, match="not exact"):
        parse_stack_migration_transfer_checkpoint_v2(projection)


def test_role_surfaces_are_distinct_arn_role_id_trust_and_permission_evidence() -> None:
    checkpoint = _checkpoint()
    roles = {
        checkpoint.api_caller,
        checkpoint.migration_service_role,
        checkpoint.fence_service_role,
    }
    assert len({role.role_arn for role in roles}) == 3
    assert len({role.role_id for role in roles}) == 3
    assert all(role.trust_policy_sha256 for role in roles)
    assert all(role.permission_policy_sha256 for role in roles)
