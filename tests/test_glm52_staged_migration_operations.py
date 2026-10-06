from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from test_glm52_task13_migration_adapter import (
    _checkpoint,
    _disabled_support,
    _identity,
    _prepare_result,
    _roles,
    _seed,
)

import glm52_enforcement.cloudformation_stacks as stacks
import glm52_enforcement.staged_migration_operations as staged
from glm52_enforcement.cloudformation_stacks import (
    AmbiguousMigrationTransportError,
    BootstrapCoordinate,
    MigrationBootstrapResult,
    MigrationExecutionAuthority,
    MigrationMutationRecord,
    MigrationMutationStatus,
    StackKind,
    StackMigrationAuthorityV2,
    StackMigrationEvidenceV2,
    StackMigrationTransferBundleV2,
    TemplateArtifact,
    build_bridge_seed_established_v2,
    build_bridge_seed_ownership_plan_v2,
    canonical_json_bytes,
    initial_migration_execution_state,
    migration_execution_state_projection,
    parse_bridge_seed_established_v2,
)
from glm52_enforcement.staged_migration_operations import (
    StagedMigrationOperationError,
    materialize_bridge_seed_establishment_request_v2,
    parse_bridge_seed_establishment_request_v2,
    parse_stack_ownership_transfer_request_v2,
)
from glm52_enforcement.task13_migration_adapter import (
    MIGRATION_TEMPLATE_KEYS,
    SeededMigrationRuntime,
    execute_stack_migration_operation_7_v2,
    execution_authority_projection,
    read_stack_migration_operation_7_effect_v2,
)
from glm52_enforcement.task13_production_operations import ProductionServices


def _plan():
    return build_bridge_seed_ownership_plan_v2(
        retained_template={
            "AWSTemplateFormatVersion": "2010-09-09",
            "Resources": {"ModelBucket": {"Type": "AWS::S3::Bucket"}},
        },
        bucket_name="exact-bucket",
        bridge_seed=_seed(),
        current_policy_logical_id=None,
    )


def _authority() -> StackMigrationAuthorityV2:
    api, migration, fence = _roles()
    bootstrap_template_url = (
        "https://keep-glm52-models-246813579024-us-west-2.s3.us-west-2."
        "amazonaws.com/bootstrap.json?versionId=bootstrap-version-1"
    )
    return StackMigrationAuthorityV2(
        api_caller=api,
        migration_service_role=migration,
        fence_service_role=fence,
        fence_bootstrap=BootstrapCoordinate(
            kind=StackKind.FENCE,
            template_url=bootstrap_template_url,
            version_id="bootstrap-version-1",
        ),
        support_bootstrap=BootstrapCoordinate(
            kind=StackKind.SUPPORT,
            template_url=bootstrap_template_url,
            version_id="bootstrap-version-1",
        ),
        import_change_set_name="h1g-import-production-fence",
        action_identity_sha256="a" * 64,
    )


def _bundle(plan) -> StackMigrationTransferBundleV2:
    stages = (
        "bridge-seed-owner",
        "retention-only",
        "post-retain",
        "fence-import",
        "fence-transfer",
    )
    identities = {
        "bridge-seed-owner": _identity(StackKind.RETAINED),
        "retention-only": _identity(StackKind.RETAINED),
        "post-retain": _identity(StackKind.RETAINED),
        "fence-import": _identity(StackKind.FENCE),
        "fence-transfer": _identity(StackKind.FENCE),
    }
    artifacts = tuple(
        TemplateArtifact(
            stage=stage,
            stack_id=identities[stage].stack_id,
            template_url=(
                "https://keep-glm52-models-246813579024-us-west-2."
                f"s3.us-west-2.amazonaws.com/{stage}.json?versionId={stage}-v1"
            ),
            version_id=stage + "-v1",
            body=b"{}\n",
            sha256=str(index) * 64,
            template_body_sha256=(
                plan.template_body_sha256
                if stage == "bridge-seed-owner"
                else format(index + 5, "x") * 64
            ),
            policy_sha256=None,
            template={},
        )
        for index, stage in enumerate(stages, start=1)
    )
    manifest = {
        "record_type": "glm52_h1g_stack_migration_transfer_manifest_v2",
        "operation_7_status": "NOT_SUBMITTED",
        "execution_eligible": True,
        "direct_put_bucket_policy_fallback": False,
        "bootstrap_template_body_sha256": "f" * 64,
    }
    projection = {
        "bridge_seed_plan_identity_sha256": plan.canonical_identity_sha256,
        "artifact_sha256s": [artifact.sha256 for artifact in artifacts],
        "manifest": manifest,
    }
    return StackMigrationTransferBundleV2(
        bridge_seed_plan=plan,
        artifacts=artifacts,
        manifest=manifest,
        canonical_identity_sha256=stacks.hashlib.sha256(
            canonical_json_bytes(projection)
        ).hexdigest(),
    )


def _bridge_inputs():
    plan = _plan()
    authority = _authority()
    retained = _identity(StackKind.RETAINED)
    fence = _identity(StackKind.FENCE)
    support = _identity(StackKind.SUPPORT)
    bootstrap = MigrationBootstrapResult(
        fence=fence,
        support=support,
        action_identity_sha256=authority.action_identity_sha256,
        state_revision=2,
        reconciled_creates=(),
    )
    evidence = StackMigrationEvidenceV2(
        retained=retained,
        fence=fence,
        support=support,
        current_policy_logical_id=None,
        current_policy_physical_id="exact-bucket",
        current_policy_stack_id=retained.stack_id,
        bucket_name="exact-bucket",
        import_identifier=(("Bucket", "exact-bucket"),),
        preseed_policy={"Statement": [{"Effect": "Deny"}]},
        direct_preseed_policy_readbacks=(
            {"Statement": [{"Effect": "Deny"}]},
            {"Statement": [{"Effect": "Deny"}]},
        ),
        retained_resource_physical_ids=(),
        retained_export_names=(),
        support_export_names=(),
    )
    return plan, _bundle(plan), evidence, authority, bootstrap


class _BridgeClient:
    def __init__(self, *, current: str, target: str, ambiguous: bool = False) -> None:
        self.current = current
        self.target = target
        self.ambiguous = ambiguous
        self.update_calls = 0

    def update_stack(self, **_request):
        self.update_calls += 1
        self.current = self.target
        if self.ambiguous:
            raise AmbiguousMigrationTransportError("lost response")
        return {"ResponseMetadata": {"HTTPStatusCode": 200, "RequestId": "request-1"}}


@pytest.mark.parametrize(
    ("initial", "ambiguous", "expected_updates"),
    (("preseed", False, 1), ("preseed", True, 1), ("bridge", False, 0)),
)
def test_bridge_nominal_lost_response_and_exact_adoption(
    monkeypatch, initial: str, ambiguous: bool, expected_updates: int
) -> None:
    plan, bundle, evidence, authority, bootstrap = _bridge_inputs()
    current = (
        plan.preseed_template_body_sha256
        if initial == "preseed"
        else plan.template_body_sha256
    )
    client = _BridgeClient(
        current=current,
        target=plan.template_body_sha256,
        ambiguous=ambiguous,
    )
    status = {
        "value": MigrationMutationStatus.NOT_SUBMITTED,
        "request_sha256": None,
    }
    coordinator = stacks.MigrationCoordinator(client=client, state_store=object())

    monkeypatch.setattr(
        coordinator, "_validate_bridge_inputs_v2", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        stacks, "_stack_template_sha256", lambda *_args, **_kwargs: client.current
    )
    monkeypatch.setattr(
        stacks, "_get_template_sha256", lambda *_args, **_kwargs: client.current
    )
    monkeypatch.setattr(
        stacks,
        "_direct_policy_sha256",
        lambda *_args, **_kwargs: plan.seed_policy_sha256,
    )
    monkeypatch.setattr(
        stacks,
        "_stabilize_stack",
        lambda *_args, **_kwargs: ("UPDATE_COMPLETE", client.current),
    )
    monkeypatch.setattr(
        stacks,
        "_state_record",
        lambda *_args, **_kwargs: MigrationMutationRecord(
            key="EstablishRetainedBridgeSeed",
            status=status["value"],
            request_sha256=status["request_sha256"],
            effect_identity=None,
        ),
    )

    def begin(**_kwargs):
        submit = status["value"] is MigrationMutationStatus.NOT_SUBMITTED
        status["request_sha256"] = stacks._request_sha256(_kwargs["request"])
        status["value"] = MigrationMutationStatus.SUBMITTED
        return object(), submit, "1" * 64

    monkeypatch.setattr(stacks, "_begin_mutation", begin)
    monkeypatch.setattr(stacks, "_complete_mutation", lambda **_kwargs: object())

    result = coordinator.establish_bridge_seed_v2(
        bundle=bundle,
        evidence=evidence,
        authority=authority,
        bootstrap_result=bootstrap,
    )

    assert result.policy_committed is True
    assert result.execution_eligible is True
    assert result.processed_template_body_sha256 == plan.template_body_sha256
    assert result.direct_policy_sha256 == plan.seed_policy_sha256
    assert client.update_calls == expected_updates
    assert parse_bridge_seed_established_v2(result.to_dict()) == result


def test_public_bridge_seed_v2_builds_two_readbacks_and_returns_committed_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preseed_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "Preseed",
                "Effect": "Deny",
                "Principal": "*",
                "Action": "s3:*",
                "Resource": "arn:aws:s3:::exact-bucket/*",
            }
        ],
    }
    base_seed = _seed()
    seed_projection = base_seed.to_dict()
    seed_projection["expected_live_preseed_policy_sha256"] = stacks.hashlib.sha256(
        canonical_json_bytes(preseed_policy)
    ).hexdigest()
    seed = staged.parse_bridge_seed_artifact(
        seed_projection,
        raw_bytes=base_seed.raw_bytes,
    )
    retained_template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "ModelBucket": {
                "Type": "AWS::S3::Bucket",
                "Properties": {"BucketName": "exact-bucket"},
            },
            "LegacyOwner": {
                "Type": "AWS::S3::BucketPolicy",
                "Properties": {
                    "Bucket": "exact-bucket",
                    "PolicyDocument": preseed_policy,
                },
            },
        },
    }
    plan = build_bridge_seed_ownership_plan_v2(
        retained_template=retained_template,
        bucket_name="exact-bucket",
        bridge_seed=seed,
        current_policy_logical_id="LegacyOwner",
    )
    authority = _authority()
    retained = _identity(StackKind.RETAINED)
    bootstrap = MigrationBootstrapResult(
        fence=_identity(StackKind.FENCE),
        support=_identity(StackKind.SUPPORT),
        action_identity_sha256=authority.action_identity_sha256,
        state_revision=2,
        reconciled_creates=(),
    )
    stages = (
        "bridge-seed-owner",
        "retention-only",
        "post-retain",
        "fence-import",
        "fence-transfer",
    )
    parsed = staged.BridgeSeedEstablishmentRequestV2(
        record_type="glm52_h1g_bridge_seed_establishment_request_v2",
        activation_id="h1g-two-readbacks",
        activation_identity_sha256=authority.action_identity_sha256,
        migration_seed={},
        authority=authority,
        evidence=staged.MigrationEvidenceInputV2(
            retained_stack_id=retained.stack_id,
            bucket_name="exact-bucket",
            current_policy_logical_id="LegacyOwner",
            import_identifier=(("Bucket", "exact-bucket"),),
            preseed_policy=preseed_policy,
            preseed_template_body_sha256=plan.preseed_template_body_sha256,
            retained_resource_physical_ids=(),
            retained_export_names=(),
            support_export_names=(),
        ),
        bridge_seed=seed_projection,
        template_coordinates=tuple(
            {
                "stage": stage,
                "template_url": (
                    "https://keep-glm52-models-246813579024-us-west-2."
                    f"s3.us-west-2.amazonaws.com/{stage}.json"
                    f"?versionId={stage}-v2"
                ),
                "version_id": stage + "-v2",
            }
            for stage in stages
        ),
    )
    committed = build_bridge_seed_established_v2(
        action_identity_sha256=authority.action_identity_sha256,
        retained=retained,
        fence=bootstrap.fence,
        support=bootstrap.support,
        bucket_name="exact-bucket",
        plan=plan,
        original_template_body_sha256=plan.template_body_sha256,
        processed_template_body_sha256=plan.template_body_sha256,
        direct_policy_sha256=plan.seed_policy_sha256,
        api_caller=authority.api_caller,
        migration_service_role=authority.migration_service_role,
        fence_service_role=authority.fence_service_role,
    )
    coordinator_calls: list[dict[str, object]] = []

    def establish(**kwargs: object):
        coordinator_calls.append(dict(kwargs))
        return committed

    runtime = SimpleNamespace(
        coordinator=SimpleNamespace(establish_bridge_seed_v2=establish)
    )
    unused = object()
    services = ProductionServices(
        sts=unused,
        cloudformation=unused,
        iam=unused,
        s3=unused,
        organizations=unused,
        ec2=unused,
        ssm=unused,
        kms=unused,
        dynamodb=unused,
        lambda_client=unused,
        states=unused,
        cloudtrail=unused,
        total_max_attempts=1,
    )
    bootstrap_template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "ContainerAnchor": {"Type": "AWS::CloudFormation::WaitConditionHandle"}
        },
    }
    built_evidence: list[StackMigrationEvidenceV2] = []
    real_build = staged.build_stack_migration_v2

    def build(**kwargs: object) -> StackMigrationTransferBundleV2:
        evidence = kwargs["evidence"]
        assert type(evidence) is StackMigrationEvidenceV2
        built_evidence.append(evidence)
        return real_build(**kwargs)

    monkeypatch.setattr(
        staged,
        "parse_bridge_seed_establishment_request_v2",
        lambda _request: parsed,
    )
    monkeypatch.setattr(staged, "_runtime", lambda **_kwargs: runtime)
    monkeypatch.setattr(
        staged,
        "bootstrap_seeded_migration",
        lambda _runtime: bootstrap,
    )
    monkeypatch.setattr(staged, "_load_bridge_seed", lambda *_args: seed)
    monkeypatch.setattr(
        staged,
        "_bridge_plan_from_live",
        lambda *_args: staged._LiveBridgePlanResult(
            plan=plan,
            direct_preseed_policy_readbacks=(
                dict(preseed_policy),
                dict(preseed_policy),
            ),
        ),
    )
    monkeypatch.setattr(
        staged,
        "_template",
        lambda *_args, **_kwargs: bootstrap_template,
    )
    monkeypatch.setattr(staged, "build_stack_migration_v2", build)

    result = staged.establish_bridge_seed_v2(
        request={"record_type": "public-bridge-seed"},
        services=services,
    )

    assert len(built_evidence) == 1
    evidence = built_evidence[0]
    assert type(evidence.direct_preseed_policy_readbacks) is tuple
    assert len(evidence.direct_preseed_policy_readbacks) == 2
    assert (
        tuple(
            canonical_json_bytes(readback)
            for readback in evidence.direct_preseed_policy_readbacks
        )
        == (canonical_json_bytes(preseed_policy),) * 2
    )
    assert len(coordinator_calls) == 1
    assert coordinator_calls[0]["evidence"] is evidence
    bundle = coordinator_calls[0]["bundle"]
    assert type(bundle) is StackMigrationTransferBundleV2
    assert tuple(artifact.stage for artifact in bundle.artifacts) == stages
    assert result is committed
    assert result.policy_committed is True
    assert result.execution_eligible is True


def test_operations_1_to_6_consume_bridge_without_bridge_resend(monkeypatch) -> None:
    plan, bundle, evidence, authority, bootstrap = _bridge_inputs()
    bridge = build_bridge_seed_established_v2(
        action_identity_sha256=authority.action_identity_sha256,
        retained=evidence.retained,
        fence=evidence.fence,
        support=evidence.support,
        bucket_name=evidence.bucket_name,
        plan=plan,
        original_template_body_sha256=plan.template_body_sha256,
        processed_template_body_sha256=plan.template_body_sha256,
        direct_policy_sha256=plan.seed_policy_sha256,
        api_caller=authority.api_caller,
        migration_service_role=authority.migration_service_role,
        fence_service_role=authority.fence_service_role,
    )
    client = SimpleNamespace()
    coordinator = stacks.MigrationCoordinator(client=client, state_store=object())
    mutation_keys = stacks.STACK_MIGRATION_TRANSFER_COMPLETED_KEYS_V2
    records = tuple(
        MigrationMutationRecord(
            key=key,
            status=MigrationMutationStatus.COMPLETE,
            request_sha256="1" * 64,
            effect_identity=(
                "arn:aws:cloudformation:us-west-2:246813579024:changeSet/"
                "h1g-import-production-fence/cccccccc-1111-4222-8333-444444444444"
                if key == "ImportPolicyIntoFenceStack.CreateChangeSet"
                else (
                    evidence.fence.stack_id
                    if key == "CreateBootstrapFenceStack"
                    else evidence.support.stack_id
                    if key == "CreateBootstrapSupportStack"
                    else None
                )
            ),
        )
        for key in mutation_keys
    ) + (
        MigrationMutationRecord(
            key="EstablishRetainedBridgeSeed",
            status=MigrationMutationStatus.COMPLETE,
            request_sha256="1" * 64,
            effect_identity=None,
        ),
        MigrationMutationRecord(
            key="UpdateSupportReplaceAnchor",
            status=MigrationMutationStatus.NOT_SUBMITTED,
            request_sha256=None,
            effect_identity=None,
        ),
    )
    state = SimpleNamespace(mutations=records)
    monkeypatch.setattr(coordinator, "_load_state", lambda *_args, **_kwargs: state)
    monkeypatch.setattr(
        stacks,
        "migration_bootstrap_result_from_state",
        lambda *_args, **_kwargs: bootstrap,
    )
    monkeypatch.setattr(coordinator, "_update", lambda **kwargs: kwargs["state"])
    monkeypatch.setattr(
        coordinator,
        "_create_import_change_set",
        lambda **kwargs: (kwargs["state"], object()),
    )
    change_set_id = next(
        record.effect_identity
        for record in records
        if record.key == "ImportPolicyIntoFenceStack.CreateChangeSet"
    )
    monkeypatch.setattr(
        coordinator,
        "_execute_import",
        lambda **kwargs: (kwargs["state"], change_set_id),
    )
    artifact_by_stage = {artifact.stage: artifact for artifact in bundle.artifacts}

    def stack_hash(_client, *, stack_id):
        if stack_id == evidence.retained.stack_id:
            return artifact_by_stage["post-retain"].template_body_sha256
        if stack_id == evidence.fence.stack_id:
            return artifact_by_stage["fence-transfer"].template_body_sha256
        return bundle.manifest["bootstrap_template_body_sha256"]

    monkeypatch.setattr(stacks, "_stack_template_sha256", stack_hash)
    monkeypatch.setattr(
        stacks,
        "_get_template_sha256",
        lambda *_args, **_kwargs: (
            artifact_by_stage["fence-transfer"].template_body_sha256
        ),
    )
    monkeypatch.setattr(
        stacks,
        "_stack_policy_sha256_v2",
        lambda *_args, **_kwargs: stacks.FENCE_STACK_POLICY_SHA256_V2,
    )
    monkeypatch.setattr(
        stacks,
        "_direct_policy_sha256",
        lambda *_args, **_kwargs: plan.seed_policy_sha256,
    )

    checkpoint = coordinator.execute_operations_1_to_6_v2(
        bundle=bundle,
        evidence=evidence,
        authority=authority,
        bootstrap_result=bootstrap,
        bridge_seed=bridge,
    )

    assert checkpoint.operation_7_status == "NOT_SUBMITTED"
    assert checkpoint.completed_mutation_keys == mutation_keys
    assert not hasattr(client, "update_stack")


class _BodyClient:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def get_object(self, **_request):
        return {
            "Body": BytesIO(self.raw),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "request-1",
                "RetryAttempts": 0,
            },
        }


class _Operation7Coordinator:
    def __init__(self, raw: bytes, *, lost: bool = False) -> None:
        self._client = _BodyClient(raw)
        self.lost = lost
        self.execute_calls = 0
        self.read_calls = 0

    def execute_support_anchor_replacement_v2(self, **kwargs):
        self.execute_calls += 1
        if self.lost:
            self.lost = False
        return (
            kwargs["checkpoint"].support_prestate_template_sha256,
            kwargs["artifact"].template_body_sha256,
        )

    def read_support_anchor_replacement_effect_v2(self, **kwargs):
        self.read_calls += 1
        return (
            kwargs["checkpoint"].support_prestate_template_sha256,
            kwargs["artifact"].template_body_sha256,
        )


def _operation7_fixture(*, lost: bool = False):
    checkpoint = _checkpoint()
    prepare = _prepare_result(checkpoint)
    disabled = _disabled_support(prepare)
    template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {"Disabled": {"Type": "AWS::Logs::LogGroup"}},
    }
    raw = canonical_json_bytes(template) + b"\n"
    coordinate = dict(disabled.support_template_coordinate)
    coordinate["file_sha256"] = stacks.hashlib.sha256(raw).hexdigest()
    coordinate["body_sha256"] = stacks.hashlib.sha256(raw[:-1]).hexdigest()
    object.__setattr__(disabled, "support_template_coordinate", coordinate)
    object.__setattr__(disabled, "support_template_sha256", coordinate["body_sha256"])
    coordinator = _Operation7Coordinator(raw, lost=lost)
    authority = SimpleNamespace(
        deployment_role_arn=checkpoint.migration_service_role.role_arn,
        deployment_role_id=checkpoint.migration_service_role.role_id,
    )
    runtime = SeededMigrationRuntime(
        coordinator=coordinator,
        backend=object(),
        seed=SimpleNamespace(authority=authority),
    )
    return runtime, checkpoint, prepare, disabled, coordinator


@pytest.mark.parametrize("lost", (False, True))
def test_operation_7_nominal_and_lost_response_converge(lost: bool) -> None:
    runtime, checkpoint, prepare, disabled, coordinator = _operation7_fixture(lost=lost)
    result = execute_stack_migration_operation_7_v2(
        runtime=runtime,
        checkpoint=checkpoint,
        prepare_result=prepare,
        disabled_support=disabled,
        observed_api_caller=checkpoint.api_caller,
        observed_migration_service_role=checkpoint.migration_service_role,
        observed_fence_service_role=checkpoint.fence_service_role,
    )
    assert result.operation_7_status == "COMPLETE"
    assert result.support_poststate_template_sha256 == disabled.support_template_sha256
    assert coordinator.execute_calls == 1


def test_operation_7_read_only_adoption_and_drift_rejection() -> None:
    runtime, checkpoint, prepare, disabled, coordinator = _operation7_fixture()
    adopted = read_stack_migration_operation_7_effect_v2(
        runtime=runtime,
        checkpoint=checkpoint,
        prepare_result=prepare,
        disabled_support=disabled,
        observed_api_caller=checkpoint.api_caller,
        observed_migration_service_role=checkpoint.migration_service_role,
        observed_fence_service_role=checkpoint.fence_service_role,
    )
    assert adopted.operation_7_status == "COMPLETE"
    assert coordinator.execute_calls == 0
    assert coordinator.read_calls == 1

    drifted = _roles()[2]
    object.__setattr__(drifted, "role_id", "AROAFENCESERVICEDRIFT123")
    with pytest.raises(Exception, match="role|RoleId|exact"):
        execute_stack_migration_operation_7_v2(
            runtime=runtime,
            checkpoint=checkpoint,
            prepare_result=prepare,
            disabled_support=disabled,
            observed_api_caller=checkpoint.api_caller,
            observed_migration_service_role=checkpoint.migration_service_role,
            observed_fence_service_role=drifted,
        )


def test_raw_request_parsers_reject_missing_and_unknown_fields() -> None:
    incomplete = {"schema_version": 2}
    with pytest.raises(StagedMigrationOperationError, match="fields"):
        parse_bridge_seed_establishment_request_v2(incomplete)
    with pytest.raises(StagedMigrationOperationError, match="fields"):
        parse_stack_ownership_transfer_request_v2({**incomplete, "unknown": True})


def _valid_nested_staged_request(record_type: str) -> dict[str, object]:
    authority = _authority()
    execution_authority = MigrationExecutionAuthority(
        deployment_role_arn=authority.migration_service_role.role_arn,
        deployment_role_id=authority.migration_service_role.role_id,
        fence_bootstrap=authority.fence_bootstrap,
        support_bootstrap=authority.support_bootstrap,
        import_change_set_name=authority.import_change_set_name,
        action_identity_sha256=authority.action_identity_sha256,
    )
    migration_seed_body = {
        "schema_version": 1,
        "record_type": "glm52_task13_stack_migration_seed_v1",
        "account_id": stacks.ACCOUNT_ID,
        "region": stacks.REGION,
        "run_id": stacks.RUN_ID,
        "authority": execution_authority_projection(execution_authority),
        "initial_state": migration_execution_state_projection(
            initial_migration_execution_state(execution_authority)
        ),
        "publication_plan": [
            {
                "stage": stage,
                "bucket": "keep-glm52-models-246813579024-us-west-2",
                "key": key,
            }
            for stage, key in MIGRATION_TEMPLATE_KEYS.items()
        ],
    }
    migration_seed = {
        **migration_seed_body,
        "canonical_identity_sha256": stacks.hashlib.sha256(
            canonical_json_bytes(migration_seed_body)
        ).hexdigest(),
    }
    preseed_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Deny",
                "Action": ["s3:DeleteBucket", "s3:PutBucketPolicy"],
                "Resource": "arn:aws:s3:::exact-bucket",
            }
        ],
    }
    bridge_seed = _seed().to_dict()
    bridge_seed["expected_live_preseed_policy_sha256"] = stacks.hashlib.sha256(
        canonical_json_bytes(preseed_policy)
    ).hexdigest()
    retained = _identity(StackKind.RETAINED)
    stages = (
        "bridge-seed-owner",
        "retention-only",
        "post-retain",
        "fence-import",
        "fence-transfer",
    )
    request = {
        "schema_version": 2,
        "record_type": record_type,
        "activation_id": "h1g-nested-detachment",
        "activation_identity_sha256": authority.action_identity_sha256,
        "migration_seed": migration_seed,
        "api_caller": authority.api_caller.to_dict(),
        "migration_service_role": authority.migration_service_role.to_dict(),
        "fence_service_role": authority.fence_service_role.to_dict(),
        "evidence": {
            "retained_stack_id": retained.stack_id,
            "bucket_name": "exact-bucket",
            "current_policy_logical_id": None,
            "import_identifier": [["Bucket", "exact-bucket"]],
            "preseed_policy": preseed_policy,
            "preseed_template_body_sha256": "e" * 64,
            "retained_resource_physical_ids": [],
            "retained_export_names": [],
            "support_export_names": [],
        },
        "template_coordinates": [
            {
                "stage": stage,
                "template_url": (
                    "https://keep-glm52-models-246813579024-us-west-2."
                    f"s3.us-west-2.amazonaws.com/{stage}.json"
                    f"?versionId={stage}-v2"
                ),
                "version_id": stage + "-v2",
            }
            for stage in stages
        ],
    }
    if record_type == "glm52_h1g_bridge_seed_establishment_request_v2":
        request["bridge_seed"] = bridge_seed
    return request


@pytest.mark.parametrize(
    ("parser", "record_type", "has_bridge_seed"),
    (
        (
            parse_bridge_seed_establishment_request_v2,
            "glm52_h1g_bridge_seed_establishment_request_v2",
            True,
        ),
        (
            parse_stack_ownership_transfer_request_v2,
            "glm52_h1g_stack_ownership_transfer_request_v2",
            False,
        ),
    ),
)
def test_raw_request_parsers_deep_detach_nested_caller_state(
    parser,
    record_type: str,
    has_bridge_seed: bool,
) -> None:
    request = _valid_nested_staged_request(record_type)
    expected = parser(json.loads(canonical_json_bytes(request)))
    parsed = parser(request)
    projection = {
        "migration_seed": parsed.migration_seed,
        "preseed_policy": parsed.evidence.preseed_policy,
    }
    if has_bridge_seed:
        projection["bridge_seed"] = parsed.bridge_seed
    canonical_before = canonical_json_bytes(projection)
    validated_seed_before = staged.validate_stack_migration_seed_projection(
        parsed.migration_seed
    )
    bridge_seed_before = (
        staged.parse_bridge_seed_artifact(parsed.bridge_seed)
        if has_bridge_seed
        else None
    )

    request["migration_seed"]["publication_plan"][0]["key"] = "caller-mutated"
    if has_bridge_seed:
        request["bridge_seed"]["statement_ledger"].clear()
    request["evidence"]["preseed_policy"]["Statement"][0]["Effect"] = "Allow"

    assert parsed == expected
    parsed_projection = {
        "migration_seed": parsed.migration_seed,
        "preseed_policy": parsed.evidence.preseed_policy,
    }
    if has_bridge_seed:
        parsed_projection["bridge_seed"] = parsed.bridge_seed
    assert canonical_json_bytes(parsed_projection) == canonical_before
    assert (
        staged.validate_stack_migration_seed_projection(parsed.migration_seed)
        == validated_seed_before
    )
    if has_bridge_seed:
        assert (
            staged.parse_bridge_seed_artifact(parsed.bridge_seed) == bridge_seed_before
        )


def test_bridge_seed_establishment_request_is_bound_only_after_publication() -> None:
    full_request = _valid_nested_staged_request(
        "glm52_h1g_bridge_seed_establishment_request_v2"
    )
    artifact = full_request.pop("bridge_seed")

    materialized = materialize_bridge_seed_establishment_request_v2(
        prototype=full_request,
        bridge_seed=artifact,
    )

    assert materialized["bridge_seed"] == artifact
    assert parse_bridge_seed_establishment_request_v2(materialized)
    conflicting = dict(full_request)
    conflicting["bridge_seed"] = artifact
    with pytest.raises(StagedMigrationOperationError, match="prototype|bridge seed"):
        materialize_bridge_seed_establishment_request_v2(
            prototype=conflicting,
            bridge_seed=artifact,
        )


def test_transfer_request_rejects_obsolete_embedded_bridge_seed() -> None:
    request = _valid_nested_staged_request(
        "glm52_h1g_stack_ownership_transfer_request_v2"
    )
    request["bridge_seed"] = _seed().to_dict()

    with pytest.raises(StagedMigrationOperationError, match="fields"):
        parse_stack_ownership_transfer_request_v2(request)


def test_raw_request_parser_maps_noncanonical_nested_policy_to_staged_error() -> None:
    request = _valid_nested_staged_request(
        "glm52_h1g_bridge_seed_establishment_request_v2"
    )
    request["evidence"]["preseed_policy"]["Statement"][0]["Action"] = {
        "s3:DeleteBucket"
    }

    with pytest.raises(StagedMigrationOperationError, match="canonical JSON"):
        parse_bridge_seed_establishment_request_v2(request)
