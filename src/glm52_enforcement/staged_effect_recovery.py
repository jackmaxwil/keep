"""Read-only recovery and adoption for the staged production effects."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .canonical import canonical_json_bytes, canonical_sha256
from .cloudformation_stacks import STACK_TAGS
from .task13_production_operations import (
    ACCOUNT_ID,
    ProductionOperationError,
    ProductionServices,
    _call,
    _guard_production_request,
    _template_body,
)
from .task13_staged_deployment import (
    FENCE_STACK_NAME,
    MODEL_BUCKET_NAME,
    BootstrapFencePublication,
    DeploymentStep,
    _validate_cross_step_evidence,
    _validate_evidence,
    parse_bootstrap_fence_publication,
    parse_disabled_support_deployment_evidence,
)


class StagedEffectRecoveryError(ValueError):
    """A durable staged effect was absent, foreign, duplicated, or drifted."""


Parser = Callable[[object], object]
LiveReader = Callable[..., object]


@dataclass(frozen=True)
class RecoveryStepBinding:
    """The sole parser and bounded live reader admitted for one staged effect."""

    parser: Parser
    live_reader: LiveReader
    mutating: bool

    def __post_init__(self) -> None:
        if (
            not callable(self.parser)
            or not callable(self.live_reader)
            or type(self.mutating) is not bool
        ):
            raise TypeError("recovery step binding is not exact")


def _parse_retained_runtime_deployment(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise StagedEffectRecoveryError(
            "retained runtime deployment evidence is not exact"
        )
    return dict(value)


def _parse_bridge_seed_publication(value: object) -> object:
    from .fence_bootstrap_publication import (
        parse_bridge_seed_publication_v2,
    )

    return parse_bridge_seed_publication_v2(value)


def _parse_bridge_seed(value: object) -> object:
    from .cloudformation_stacks import parse_bridge_seed_established_v2

    return parse_bridge_seed_established_v2(value)


def _parse_migration_checkpoint(value: object) -> object:
    from .task13_migration_adapter import (
        parse_stack_migration_transfer_checkpoint_v2,
    )

    return parse_stack_migration_transfer_checkpoint_v2(value)


def _parse_prepare_result(value: object) -> object:
    from .fence_executor import parse_fence_execution_result

    return parse_fence_execution_result(value)


def _parse_support_inputs(value: object) -> object:
    from .task13_support_input_materialization import (
        support_build_inputs_from_mapping,
    )

    return support_build_inputs_from_mapping(value)


def _parse_operation_7(value: object) -> object:
    from .task13_migration_adapter import (
        parse_stack_migration_operation_7_evidence_v2,
    )

    return parse_stack_migration_operation_7_evidence_v2(value)


def _parse_runtime_identity(value: object) -> object:
    from .support_plane import parse_support_runtime_identity

    return parse_support_runtime_identity(value)


def _parse_no_launch(value: object) -> dict[str, object]:
    fields = {
        "schema_version",
        "record_type",
        "worker_count",
        "raw_ec2_launch_calls",
        "source_action_calls",
        "launch_authority_present",
        "complete",
    }
    if type(value) is not dict or set(value) != fields:
        raise StagedEffectRecoveryError("no-launch evidence schema is not exact")
    if (
        value["schema_version"] != 2
        or value["record_type"] != "glm52_h1g_no_launch_evidence_v2"
        or type(value["worker_count"]) is not int
        or type(value["raw_ec2_launch_calls"]) is not int
        or type(value["source_action_calls"]) is not int
        or value["worker_count"] != 0
        or value["raw_ec2_launch_calls"] != 0
        or value["source_action_calls"] != 0
        or value["launch_authority_present"] is not False
        or value["complete"] is not True
    ):
        raise StagedEffectRecoveryError("no-launch evidence is not exact zero proof")
    return dict(value)


def _read_retained_bootstrap_runtime(
    *,
    request: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
    services: ProductionServices,
) -> object:
    del committed
    from .task13_production_operations import (
        read_retained_bootstrap_runtime_deployment_v2,
    )

    return read_retained_bootstrap_runtime_deployment_v2(
        request=request["retained_bootstrap_runtime_deployment"],
        services=services,
    )


def _read_bridge_seed_publication(
    *,
    request: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
    services: ProductionServices,
) -> object:
    from .fence_bootstrap_publication import (
        materialize_bridge_seed_publication_request_v2,
        read_bridge_seed_publication_v2,
    )
    from .task13_fixed_artifacts import Task13FixedArtifactServices

    bootstrap_runtime = committed[
        DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED
    ]
    if type(bootstrap_runtime) is not dict:
        raise StagedEffectRecoveryError(
            "bridge seed publication runtime predecessor is not exact"
        )
    materialized_request = materialize_bridge_seed_publication_request_v2(
        authority=request["bridge_seed_publication"],
        materializer_function_version_arn=bootstrap_runtime.get(
            "materializer_function_version_arn"
        ),
    )
    return read_bridge_seed_publication_v2(
        request=materialized_request,
        services=Task13FixedArtifactServices(
            sts=services.sts,
            s3=services.s3,
            total_max_attempts=services.total_max_attempts,
        ),
    )


def _read_bridge_seed(
    *,
    request: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
    services: ProductionServices,
) -> object:
    from .fence_bootstrap_publication import BridgeSeedPublicationV2
    from .staged_migration_operations import (
        materialize_bridge_seed_establishment_request_v2,
        read_bridge_seed_v2,
    )

    publication = committed[DeploymentStep.BRIDGE_SEED_PUBLISHED]
    if type(publication) is not BridgeSeedPublicationV2:
        raise StagedEffectRecoveryError(
            "bridge seed publication predecessor is not exact"
        )
    materialized_request = materialize_bridge_seed_establishment_request_v2(
        prototype=request["bridge_seed"],
        bridge_seed=publication.artifact,
    )
    return read_bridge_seed_v2(
        request=materialized_request,
        services=services,
    )


def _read_migration_checkpoint(
    *,
    request: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
    services: ProductionServices,
) -> object:
    from .staged_migration_operations import read_operations_1_to_6_v2

    return read_operations_1_to_6_v2(
        request=request["migration_operations_1_to_6"],
        bridge_seed=committed[DeploymentStep.BRIDGE_SEED_ESTABLISHED],
        services=services,
    )


def _read_bootstrap_publication(
    *,
    request: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
    services: ProductionServices,
) -> object:
    from .fence_bootstrap_publication import (
        materialize_bootstrap_publication_request_v2,
        read_bootstrap_fence_publication_v2,
    )

    bootstrap_runtime = committed[
        DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED
    ]
    checkpoint = committed[DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6]
    if type(bootstrap_runtime) is not dict:
        raise StagedEffectRecoveryError(
            "bootstrap publication runtime predecessor is not exact"
        )
    materialized_request = materialize_bootstrap_publication_request_v2(
        authority=request["bootstrap_fence_publication"],
        materializer_function_version_arn=bootstrap_runtime.get(
            "materializer_function_version_arn"
        ),
        checkpoint=checkpoint,
    )

    return read_bootstrap_fence_publication_v2(
        request=materialized_request,
        checkpoint=checkpoint,
        services=services,
    )


def _read_retained_fence_runtime(
    *,
    request: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
    services: ProductionServices,
) -> object:
    from .support_plane import materialize_retained_fence_runtime_inputs
    from .task13_production_operations import (
        read_retained_fence_runtime_deployment_v2,
    )

    checkpoint = committed[DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6]
    publication = committed[DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED]
    try:
        fence_stack_id = checkpoint.fence_stack_id
        manifest_coordinate = publication.manifest_coordinate.to_dict()
    except AttributeError as exc:
        raise StagedEffectRecoveryError(
            "retained fence runtime predecessors are not exact"
        ) from exc
    materialized_request = materialize_retained_fence_runtime_inputs(
        authority=request["retained_fence_runtime_deployment"],
        fence_stack_id=fence_stack_id,
        bootstrap_manifest_coordinate=manifest_coordinate,
    )

    return read_retained_fence_runtime_deployment_v2(
        request=materialized_request,
        checkpoint=checkpoint,
        publication=publication,
        services=services,
    )


def _read_prepare_result(
    *,
    request: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
    services: ProductionServices,
) -> object:
    from .fence_artifacts import FenceSlot, parse_fence_transition_request
    from .fence_executor import (
        DynamoFenceRecordStore,
        direct_policy_sha256,
        load_pinned_fence_entry_template,
        load_pinned_fence_manifest,
        parse_fence_execution_result,
    )
    from .task13_migration_adapter import StackMigrationTransferCheckpointV2

    checkpoint = committed[DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6]
    publication = committed[DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED]
    if (
        type(checkpoint) is not StackMigrationTransferCheckpointV2
        or type(publication) is not BootstrapFencePublication
    ):
        raise StagedEffectRecoveryError("PREPARE predecessors are not exact")
    transition = parse_fence_transition_request(request["prepare_execution"])
    if (
        transition.to_dict().get("manifest_coordinate")
        != publication.manifest_coordinate.to_dict()
    ):
        raise StagedEffectRecoveryError("PREPARE request is not publication-bound")
    if transition.slot is not FenceSlot.PREPARE_GENESIS_LIVE_STATE:
        raise StagedEffectRecoveryError(
            "PREPARE recovery request selected another slot"
        )
    manifest = load_pinned_fence_manifest(
        s3=services.s3,
        coordinate=publication.manifest_coordinate,
    )
    entry = load_pinned_fence_entry_template(
        s3=services.s3,
        entry=manifest.entry(FenceSlot.PREPARE_GENESIS_LIVE_STATE),
    )
    if entry.entry_identity_sha256 != publication.prepare_entry_identity_sha256:
        raise StagedEffectRecoveryError("PREPARE publication entry drifted")
    durable = DynamoFenceRecordStore(
        client=services.dynamodb,
        table_name="keep-glm52-h1g-ledger-v1",
    ).get_immutable(
        record_type="glm52_fence_execution_result_v2",
        request_identity_sha256=transition.canonical_identity_sha256,
        slot=transition.slot,
    )
    result = parse_fence_execution_result(dict(durable))
    if (
        result.request_identity_sha256 != transition.canonical_identity_sha256
        or result.entry_identity_sha256 != entry.entry_identity_sha256
        or result.manifest_identity_sha256 != manifest.canonical_identity_sha256
        or result.stack_id != checkpoint.fence_stack_id
        or result.slot is not transition.slot
        or result.original_template_body_sha256
        != entry.to_dict().get("template_body_sha256")
        or result.processed_template_body_sha256
        != entry.to_dict().get("template_body_sha256")
    ):
        raise StagedEffectRecoveryError("PREPARE durable result is foreign")

    stack_response = _call(
        services.cloudformation,
        "describe_stacks",
        operation="DescribeStacks",
        StackName=result.stack_id,
    )
    stacks = stack_response.get("Stacks")
    if type(stacks) is not list or len(stacks) != 1 or type(stacks[0]) is not dict:
        raise StagedEffectRecoveryError("PREPARE live stack read is not singular")
    stack = stacks[0]
    expected_tags = sorted(
        ({"Key": key, "Value": value} for key, value in STACK_TAGS),
        key=lambda item: item["Key"],
    )
    observed_tags = stack.get("Tags")
    if (
        stack.get("StackId") != result.stack_id
        or stack.get("StackName") != FENCE_STACK_NAME
        or stack.get("StackStatus") != "UPDATE_COMPLETE"
        or stack.get("EnableTerminationProtection") is not True
        or stack.get("RoleARN") != result.observed_poststate_stack_role_arn
        or type(observed_tags) is not list
        or any(type(item) is not dict for item in observed_tags)
        or sorted(observed_tags, key=lambda item: str(item.get("Key"))) != expected_tags
    ):
        raise StagedEffectRecoveryError("PREPARE live stack drifted")

    role_name = result.observed_poststate_stack_role_arn.rsplit("/", 1)[-1]
    role_response = _call(
        services.iam,
        "get_role",
        operation="GetRole",
        RoleName=role_name,
    )
    role = role_response.get("Role")
    if (
        type(role) is not dict
        or role.get("Arn") != result.observed_poststate_stack_role_arn
        or role.get("RoleId") != result.observed_poststate_stack_role_id
    ):
        raise StagedEffectRecoveryError("PREPARE live stack role drifted")

    original = _call(
        services.cloudformation,
        "get_template",
        operation="GetTemplateOriginal",
        StackName=result.stack_id,
        TemplateStage="Original",
    )
    processed = _call(
        services.cloudformation,
        "get_template",
        operation="GetTemplateProcessed",
        StackName=result.stack_id,
        TemplateStage="Processed",
    )
    original_sha = canonical_sha256(
        _template_body(original.get("TemplateBody"), "PREPARE original template")
    )
    processed_sha = canonical_sha256(
        _template_body(processed.get("TemplateBody"), "PREPARE processed template")
    )
    if (
        original_sha != result.original_template_body_sha256
        or processed_sha != result.processed_template_body_sha256
        or original_sha != processed_sha
    ):
        raise StagedEffectRecoveryError("PREPARE live template drifted")

    policy = _call(
        services.s3,
        "get_bucket_policy",
        operation="GetBucketPolicy",
        Bucket=MODEL_BUCKET_NAME,
        ExpectedBucketOwner=ACCOUNT_ID,
    )
    if direct_policy_sha256(policy.get("Policy")) != result.poststate_policy_sha256:
        raise StagedEffectRecoveryError("PREPARE live direct policy drifted")
    return result


def _read_support_snapshot(
    *,
    request: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
    services: ProductionServices,
) -> object:
    del committed
    from .task13_support_input_materialization import (
        SupportInputServices,
        collect_support_build_inputs,
    )

    return collect_support_build_inputs(
        request=request["support_input_materialization_request"],
        services=SupportInputServices(
            sts=services.sts,
            organizations=services.organizations,
            cloudformation=services.cloudformation,
            ec2=services.ec2,
            kms=services.kms,
            dynamodb=services.dynamodb,
            s3=services.s3,
            lambda_client=services.lambda_client,
            total_max_attempts=services.total_max_attempts,
        ),
    )


def _read_disabled_support(
    *,
    request: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
    services: ProductionServices,
) -> object:
    from .support_runtime_transition import read_disabled_support_deployment_v2

    return read_disabled_support_deployment_v2(
        request=request["disabled_support_deployment"],
        checkpoint=committed[DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6],
        support_inputs=committed[DeploymentStep.SUPPORT_INPUT_SNAPSHOT_TWO],
        services=services,
    )


def _read_operation_7(
    *,
    request: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
    services: ProductionServices,
) -> object:
    from .support_runtime_transition import read_operation_7_v2

    return read_operation_7_v2(
        request=request["operation_7"],
        checkpoint=committed[DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6],
        prepare_result=committed[DeploymentStep.PREPARE_EXECUTED_STABILIZED],
        disabled_support=committed[DeploymentStep.DISABLED_SUPPORT_DEPLOYED],
        services=services,
    )


def _read_runtime_identity(
    *,
    request: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
    services: ProductionServices,
) -> object:
    from .support_runtime_transition import read_support_runtime_identity_v2

    return read_support_runtime_identity_v2(
        request=request["support_runtime_identity"],
        operation_7=committed[DeploymentStep.STACK_MIGRATION_OPERATION_7],
        services=services,
    )


def _read_no_launch(
    *,
    request: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
    services: ProductionServices,
) -> object:
    from .support_runtime_transition import read_no_launch_v2

    return read_no_launch_v2(
        request=request["no_launch_evidence"],
        runtime_identity=committed[DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED],
        services=services,
    )


_MUTATING_STEPS = frozenset(
    {
        DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED,
        DeploymentStep.BRIDGE_SEED_PUBLISHED,
        DeploymentStep.BRIDGE_SEED_ESTABLISHED,
        DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6,
        DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED,
        DeploymentStep.RETAINED_FENCE_RUNTIME_DEPLOYED,
        DeploymentStep.PREPARE_EXECUTED_STABILIZED,
        DeploymentStep.DISABLED_SUPPORT_DEPLOYED,
        DeploymentStep.STACK_MIGRATION_OPERATION_7,
        DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED,
    }
)

_RECOVERY_STEP_BINDINGS: Mapping[DeploymentStep, RecoveryStepBinding] = (
    MappingProxyType(
        {
            DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED: RecoveryStepBinding(
                parser=_parse_retained_runtime_deployment,
                live_reader=_read_retained_bootstrap_runtime,
                mutating=True,
            ),
            DeploymentStep.BRIDGE_SEED_PUBLISHED: RecoveryStepBinding(
                parser=_parse_bridge_seed_publication,
                live_reader=_read_bridge_seed_publication,
                mutating=True,
            ),
            DeploymentStep.BRIDGE_SEED_ESTABLISHED: RecoveryStepBinding(
                parser=_parse_bridge_seed,
                live_reader=_read_bridge_seed,
                mutating=True,
            ),
            DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6: RecoveryStepBinding(
                parser=_parse_migration_checkpoint,
                live_reader=_read_migration_checkpoint,
                mutating=True,
            ),
            DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED: RecoveryStepBinding(
                parser=parse_bootstrap_fence_publication,
                live_reader=_read_bootstrap_publication,
                mutating=True,
            ),
            DeploymentStep.RETAINED_FENCE_RUNTIME_DEPLOYED: RecoveryStepBinding(
                parser=_parse_retained_runtime_deployment,
                live_reader=_read_retained_fence_runtime,
                mutating=True,
            ),
            DeploymentStep.PREPARE_EXECUTED_STABILIZED: RecoveryStepBinding(
                parser=_parse_prepare_result,
                live_reader=_read_prepare_result,
                mutating=True,
            ),
            DeploymentStep.SUPPORT_INPUT_SNAPSHOT_ONE: RecoveryStepBinding(
                parser=_parse_support_inputs,
                live_reader=_read_support_snapshot,
                mutating=False,
            ),
            DeploymentStep.SUPPORT_INPUT_SNAPSHOT_TWO: RecoveryStepBinding(
                parser=_parse_support_inputs,
                live_reader=_read_support_snapshot,
                mutating=False,
            ),
            DeploymentStep.DISABLED_SUPPORT_DEPLOYED: RecoveryStepBinding(
                parser=parse_disabled_support_deployment_evidence,
                live_reader=_read_disabled_support,
                mutating=True,
            ),
            DeploymentStep.STACK_MIGRATION_OPERATION_7: RecoveryStepBinding(
                parser=_parse_operation_7,
                live_reader=_read_operation_7,
                mutating=True,
            ),
            DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED: RecoveryStepBinding(
                parser=_parse_runtime_identity,
                live_reader=_read_runtime_identity,
                mutating=True,
            ),
            DeploymentStep.NO_WORKER_ACTIVATION_PROVED: RecoveryStepBinding(
                parser=_parse_no_launch,
                live_reader=_read_no_launch,
                mutating=False,
            ),
        }
    )
)


def _guard_request(value: object) -> dict[str, object]:
    try:
        guarded = _guard_production_request(value)
    except (TypeError, ValueError, ProductionOperationError) as exc:
        raise StagedEffectRecoveryError(
            "staged recovery production request is not exact"
        ) from exc
    return guarded


def _require_step_and_prefix(
    *,
    step: object,
    committed: object,
    services: object,
) -> tuple[DeploymentStep, Mapping[DeploymentStep, object], ProductionServices]:
    if type(step) is not DeploymentStep:
        raise StagedEffectRecoveryError("recovery step is not exact")
    if type(committed) is not dict:
        raise StagedEffectRecoveryError("recovery predecessor mapping is not exact")
    expected = tuple(DeploymentStep)[: tuple(DeploymentStep).index(step)]
    if tuple(committed) != expected:
        raise StagedEffectRecoveryError(
            step.value + " recovery predecessors are incomplete or reordered"
        )
    if type(services) is not ProductionServices or services.total_max_attempts != 1:
        raise StagedEffectRecoveryError(
            "recovery services are not exact one-attempt clients"
        )
    return step, committed, services


def _validated_projection(
    step: DeploymentStep,
    evidence: object,
    committed: Mapping[DeploymentStep, object],
) -> Mapping[str, object]:
    try:
        projected = _validate_evidence(step, evidence)
        _validate_cross_step_evidence(step, projected, committed)
    except (TypeError, ValueError) as exc:
        raise StagedEffectRecoveryError(
            step.value + " effect is foreign or drifted"
        ) from exc
    return projected


def _expected_operation_identity_sha256(
    *,
    step: DeploymentStep,
    request: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
) -> str:
    expected = tuple(DeploymentStep)[: tuple(DeploymentStep).index(step)]
    if tuple(committed) != expected:
        raise StagedEffectRecoveryError("operation seal predecessors are not exact")
    projections: dict[str, object] = {}
    prefix: dict[DeploymentStep, object] = {}
    for predecessor, evidence in committed.items():
        projections[predecessor.value] = _validated_projection(
            predecessor,
            evidence,
            prefix,
        )
        prefix[predecessor] = evidence
    request_identity = canonical_sha256(
        {
            "schema_version": 2,
            "record_type": "glm52_task13_staged_deployment_request_v2",
            "activation_id": request.get("activation_id"),
            "production_request": request,
        }
    )
    return canonical_sha256(
        {
            "request_identity_sha256": request_identity,
            "step": step.value,
            "committed_evidence_sha256": canonical_sha256(projections),
        }
    )


def _parse_operation_seal(
    *,
    value: object,
    expected_identity_sha256: str,
) -> None:
    if (
        type(value) is not dict
        or set(value) != {"operation_identity_sha256"}
        or value["operation_identity_sha256"] != expected_identity_sha256
    ):
        raise StagedEffectRecoveryError("POSSIBLY_SENT operation seal is not exact")


def _read_and_reparse(
    *,
    step: DeploymentStep,
    binding: RecoveryStepBinding,
    request: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
    services: ProductionServices,
) -> tuple[object, Mapping[str, object]]:
    try:
        observed = binding.live_reader(
            request=request,
            committed=committed,
            services=services,
        )
        observed_projection = _validated_projection(step, observed, committed)
        reparsed = binding.parser(dict(observed_projection))
        reparsed_projection = _validated_projection(step, reparsed, committed)
    except StagedEffectRecoveryError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise StagedEffectRecoveryError(
            step.value + " bounded durable effect read failed"
        ) from exc
    if canonical_json_bytes(observed_projection) != canonical_json_bytes(
        reparsed_projection
    ):
        raise StagedEffectRecoveryError(
            step.value + " live effect did not reparse exactly"
        )
    return reparsed, reparsed_projection


def reconcile_staged_mutation_v2(
    *,
    step: DeploymentStep,
    request: object,
    committed: Mapping[DeploymentStep, object],
    possible_send_evidence: Mapping[str, object],
    services: ProductionServices,
) -> object:
    """Resolve one POSSIBLY_SENT mutation exclusively through bounded reads."""

    step, committed, services = _require_step_and_prefix(
        step=step,
        committed=committed,
        services=services,
    )
    guarded = _guard_request(request)
    binding = _RECOVERY_STEP_BINDINGS[step]
    if not binding.mutating or step not in _MUTATING_STEPS:
        raise StagedEffectRecoveryError(
            step.value + " is not a journaled mutating effect"
        )
    _parse_operation_seal(
        value=possible_send_evidence,
        expected_identity_sha256=_expected_operation_identity_sha256(
            step=step,
            request=guarded,
            committed=committed,
        ),
    )
    recovered, _projection = _read_and_reparse(
        step=step,
        binding=binding,
        request=guarded,
        committed=committed,
        services=services,
    )
    return recovered


def adopt_staged_evidence_v2(
    *,
    step: DeploymentStep,
    request: object,
    evidence: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
    services: ProductionServices,
) -> object:
    """Reparse one committed projection and require equal fresh live evidence."""

    step, committed, services = _require_step_and_prefix(
        step=step,
        committed=committed,
        services=services,
    )
    if type(evidence) is not dict:
        raise StagedEffectRecoveryError("stored committed projection is not exact")
    guarded = _guard_request(request)
    binding = _RECOVERY_STEP_BINDINGS[step]
    try:
        stored = binding.parser(dict(evidence))
        stored_projection = _validated_projection(step, stored, committed)
    except StagedEffectRecoveryError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise StagedEffectRecoveryError(
            step.value + " stored committed projection did not reparse"
        ) from exc
    observed, observed_projection = _read_and_reparse(
        step=step,
        binding=binding,
        request=guarded,
        committed=committed,
        services=services,
    )
    if canonical_json_bytes(stored_projection) != canonical_json_bytes(
        observed_projection
    ):
        raise StagedEffectRecoveryError(
            step.value + " adopted evidence is not byte-identical to live state"
        )
    return observed


__all__ = [
    "RecoveryStepBinding",
    "StagedEffectRecoveryError",
    "adopt_staged_evidence_v2",
    "reconcile_staged_mutation_v2",
]
