from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.cloudformation_stacks import (
    MigrationRoleEvidenceV2,
    StackIdentity,
    StackKind,
    build_bridge_seed_ownership_plan_v2,
)
from glm52_enforcement.fence_artifacts import (
    ArtifactCoordinate,
    FENCE_ARTIFACT_KEYS,
    FenceSlot,
    parse_bridge_seed_artifact,
)
from glm52_enforcement.fence_executor import parse_fence_execution_result
from glm52_enforcement.task13_fixed_artifacts import CAMPAIGN_BUCKET
from glm52_enforcement.task13_migration_adapter import (
    MIGRATION_TEMPLATE_KEYS,
    STACK_MIGRATION_OPERATION_7_RECORD_TYPE_V2,
    SealedMigrationRuntime,
    Task13MigrationAdapterError,
    build_stack_migration_operation_7_evidence_v2,
    build_stack_migration_transfer_checkpoint_v2,
    execute_sealed_migration,
    parse_stack_migration_operation_7_evidence_v2,
    parse_stack_migration_transfer_checkpoint_v2,
)
from glm52_enforcement.task13_staged_deployment import (
    parse_disabled_support_deployment_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads(
    (
        ROOT
        / ".superpowers/sdd/goal-objective/"
        "task-3ph1g-h1g-fence-policy-renderer-golden-v1.json"
    ).read_text(encoding="ascii")
)


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
    suffixes = {
        StackKind.RETAINED: "11111111-1111-4111-8111-111111111111",
        StackKind.FENCE: "22222222-2222-4222-8222-222222222222",
        StackKind.SUPPORT: "33333333-3333-4333-8333-333333333333",
    }
    name = names[kind]
    return StackIdentity(
        kind=kind,
        name=name,
        stack_id=(
            "arn:aws:cloudformation:us-west-2:246813579024:"
            f"stack/{name}/{suffixes[kind]}"
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


def _checkpoint():
    plan = build_bridge_seed_ownership_plan_v2(
        retained_template={
            "AWSTemplateFormatVersion": "2010-09-09",
            "Resources": {"ModelBucket": {"Type": "AWS::S3::Bucket"}},
        },
        bucket_name="exact-bucket",
        bridge_seed=_seed(),
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
        direct_policy_sha256=(plan.seed_policy_sha256, plan.seed_policy_sha256),
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


def _prepare_result(checkpoint, *, slot=FenceSlot.PREPARE_GENESIS_LIVE_STATE):
    body = {
        "schema_version": 2,
        "record_type": "glm52_fence_execution_result_v2",
        "request_identity_sha256": "7" * 64,
        "manifest_identity_sha256": "6" * 64,
        "entry_identity_sha256": "9" * 64,
        "prepared_identity_sha256": "8" * 64,
        "change_set_inventory_identity_sha256": "5" * 64,
        "change_set_arn": (
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "changeSet/keep-glm52-h1g-00000001-prepare-genesis-live-state-"
            "7777777777777777/cccccccc-1111-4222-8333-444444444444"
        ),
        "stack_id": checkpoint.fence_stack_id,
        "slot": slot.value,
        "execute_authority_identity_sha256": "4" * 64,
        "execute_authority_class": "RETAINED_PRE_SUPPORT",
        "execute_api_caller_role_arn": (
            "arn:aws:iam::246813579024:role/ExactMigrationApiCaller"
        ),
        "execute_api_caller_role_id": "AROAAPICALLER12345678",
        "expected_poststate_stack_role_arn": (
            checkpoint.fence_service_role.role_arn
        ),
        "observed_poststate_stack_role_arn": (
            checkpoint.fence_service_role.role_arn
        ),
        "observed_poststate_stack_role_id": (
            checkpoint.fence_service_role.role_id
        ),
        "original_template_body_sha256": "a" * 64,
        "processed_template_body_sha256": "a" * 64,
        "poststate_policy_sha256": "b" * 64,
        "first_stable_snapshot_identity_sha256": "c" * 64,
        "second_stable_snapshot_identity_sha256": "c" * 64,
        "stabilization_first_evidence_sha256": "d" * 64,
        "stabilization_second_evidence_sha256": "e" * 64,
        "stabilization_identity_sha256": "f" * 64,
        "freeze_delta_classification": None,
        "batch_execution_eligible": False,
        "completed_at": "2026-07-31T00:00:00Z",
    }
    return parse_fence_execution_result(
        {**body, "canonical_identity_sha256": canonical_sha256(body)}
    )


def _disabled_support(prepare):
    manifest_coordinate = ArtifactCoordinate(
        bucket=CAMPAIGN_BUCKET,
        key=(
            "campaigns/glm52-sky-20260724/authorities/fence/"
            "manifests/glm52-v2-amber-quartz/00000001/"
            "FENCE_BOOTSTRAP_MANIFEST.json"
        ),
        version_id="bootstrap-manifest-version-1",
        file_sha256="2" * 64,
        canonical_identity_sha256="3" * 64,
    )
    body = {
        "schema_version": 1,
        "record_type": "glm52_h1g_disabled_support_deployment_v1",
        "profile_record_type": "disabled_support_profile_v1",
        "bootstrap_manifest_coordinate": manifest_coordinate.to_dict(),
        "prepare_entry_identity_sha256": prepare.entry_identity_sha256,
        "support_build_inputs_identity_sha256": "4" * 64,
        "disabled_support_profile_sha256": "5" * 64,
        "support_template_coordinate": {
            "artifact_kind": "DISABLED_SUPPORT_TEMPLATE",
            "bucket": "keep-glm52-models-246813579024-us-west-2",
            "key": "task13/migration/disabled-support-v2.json",
            "version_id": "disabled-support-version-1",
            "file_sha256": "6" * 64,
            "body_sha256": "8" * 64,
        },
        "support_template_sha256": "8" * 64,
        "no_launch_evidence_sha256": "9" * 64,
        "worker_activation_allowed": False,
        "source_action_allowed": False,
    }
    return parse_disabled_support_deployment_evidence(
        {**body, "canonical_identity_sha256": canonical_sha256(body)}
    )


def _operation_7_evidence():
    checkpoint = _checkpoint()
    prepare = _prepare_result(checkpoint)
    disabled = _disabled_support(prepare)
    return build_stack_migration_operation_7_evidence_v2(
        checkpoint=checkpoint,
        prepare_result=prepare,
        disabled_support=disabled,
        observed_api_caller=checkpoint.api_caller,
        observed_migration_service_role=checkpoint.migration_service_role,
        observed_fence_service_role=checkpoint.fence_service_role,
        support_stack_id=checkpoint.support_stack_id,
        support_prestate_template_sha256=checkpoint.support_prestate_template_sha256,
        support_poststate_template_sha256=disabled.support_template_sha256,
    )


def test_adapter_exports_only_split_v2_migration_template_keys() -> None:
    assert tuple(MIGRATION_TEMPLATE_KEYS) == (
        "bridge-seed-owner",
        "retention-only",
        "post-retain",
        "fence-import",
        "fence-transfer",
        "disabled-support",
    )
    assert "final-fence" not in MIGRATION_TEMPLATE_KEYS


def test_checkpoint_parser_round_trips_restart_safe_typed_result() -> None:
    checkpoint = _checkpoint()
    assert parse_stack_migration_transfer_checkpoint_v2(checkpoint.to_dict()) == checkpoint


def test_v1_sealed_migration_is_audit_only_and_execution_ineligible() -> None:
    runtime = SealedMigrationRuntime(
        coordinator=None,
        authority=None,
        evidence=None,
        bundle=None,
        bootstrap_result=None,
    )
    with pytest.raises(Task13MigrationAdapterError, match="audit-only"):
        execute_sealed_migration(runtime)


def test_operation_7_is_closed_before_exact_prepare() -> None:
    checkpoint = _checkpoint()
    wrong_slot = _prepare_result(checkpoint, slot=FenceSlot.RESERVATION_ONLY)
    disabled = _disabled_support(wrong_slot)
    with pytest.raises(Task13MigrationAdapterError, match="PREPARE"):
        build_stack_migration_operation_7_evidence_v2(
            checkpoint=checkpoint,
            prepare_result=wrong_slot,
            disabled_support=disabled,
            observed_api_caller=checkpoint.api_caller,
            observed_migration_service_role=checkpoint.migration_service_role,
            observed_fence_service_role=checkpoint.fence_service_role,
            support_stack_id=checkpoint.support_stack_id,
            support_prestate_template_sha256=checkpoint.support_prestate_template_sha256,
            support_poststate_template_sha256=disabled.support_template_sha256,
        )


def test_operation_7_accepts_only_disabled_support_and_preserves_sole_role_cutover() -> None:
    evidence = _operation_7_evidence()
    checkpoint = _checkpoint()

    assert evidence.record_type == STACK_MIGRATION_OPERATION_7_RECORD_TYPE_V2
    assert evidence.operation_7_status == "COMPLETE"
    assert evidence.mutation_scope == "SUPPORT_REPLACEMENT_ONLY"
    assert evidence.fence_stack_id == checkpoint.fence_stack_id
    assert evidence.fence_associated_role_arn == checkpoint.fence_service_role.role_arn
    assert evidence.fence_associated_role_id == checkpoint.fence_service_role.role_id
    assert parse_stack_migration_operation_7_evidence_v2(evidence.to_dict()) == evidence

    prepare = _prepare_result(checkpoint)
    with pytest.raises(TypeError, match="typed v2"):
        build_stack_migration_operation_7_evidence_v2(
            checkpoint=checkpoint,
            prepare_result=prepare,
            disabled_support=object(),
            observed_api_caller=checkpoint.api_caller,
            observed_migration_service_role=checkpoint.migration_service_role,
            observed_fence_service_role=checkpoint.fence_service_role,
            support_stack_id=checkpoint.support_stack_id,
            support_prestate_template_sha256=checkpoint.support_prestate_template_sha256,
            support_poststate_template_sha256="8" * 64,
        )


def test_operation_7_rejects_any_role_id_or_two_read_drift() -> None:
    checkpoint = _checkpoint()
    prepare = _prepare_result(checkpoint)
    disabled = _disabled_support(prepare)
    drifted = _role("FenceServiceRole", "AROAFENCESERVICEDRIFT123", "3")
    with pytest.raises(Task13MigrationAdapterError, match="RoleId"):
        build_stack_migration_operation_7_evidence_v2(
            checkpoint=checkpoint,
            prepare_result=prepare,
            disabled_support=disabled,
            observed_api_caller=checkpoint.api_caller,
            observed_migration_service_role=checkpoint.migration_service_role,
            observed_fence_service_role=drifted,
            support_stack_id=checkpoint.support_stack_id,
            support_prestate_template_sha256=checkpoint.support_prestate_template_sha256,
            support_poststate_template_sha256=disabled.support_template_sha256,
        )

    object.__setattr__(prepare, "second_stable_snapshot_identity_sha256", "0" * 64)
    with pytest.raises(Task13MigrationAdapterError, match="PREPARE"):
        build_stack_migration_operation_7_evidence_v2(
            checkpoint=checkpoint,
            prepare_result=prepare,
            disabled_support=disabled,
            observed_api_caller=checkpoint.api_caller,
            observed_migration_service_role=checkpoint.migration_service_role,
            observed_fence_service_role=checkpoint.fence_service_role,
            support_stack_id=checkpoint.support_stack_id,
            support_prestate_template_sha256=checkpoint.support_prestate_template_sha256,
            support_poststate_template_sha256=disabled.support_template_sha256,
        )


def test_operation_7_evidence_has_no_direct_bucket_policy_mutation_surface() -> None:
    evidence = _operation_7_evidence()
    projection = evidence.to_dict()
    assert projection["mutation_scope"] == "SUPPORT_REPLACEMENT_ONLY"
    assert "PutBucketPolicy" not in canonical_json_bytes(projection).decode("ascii")
    assert "policy_document" not in projection
