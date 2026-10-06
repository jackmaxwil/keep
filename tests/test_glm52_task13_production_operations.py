from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm52_enforcement import task13_production_operations as production_operations
from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.fence_bootstrap_publication import (
    BridgeSeedPublicationV2,
    fence_policy_input_projection,
)
from glm52_enforcement.fence_policy_renderer import (
    RUN_GUARD_RESOURCE,
    SOURCE_FAMILY_SPECS,
    BuildMode,
    FencePolicyInput,
    PolicyHead,
    PrincipalIdentity,
    ReservedFamily,
    WriterCohort,
)
from glm52_enforcement.task13_fixed_artifacts import (
    FixedKeyPublicationDisposition,
)
from glm52_enforcement.task13_production_operations import (
    ProductionOperationBindings,
    ProductionOperationError,
    ProductionServices,
    Task13ProductionOperations,
    _guard_production_request,
    _validate_retained_transaction_composition,
    build_staged_deployment_operations,
    read_retained_bootstrap_runtime_deployment_v2,
    read_retained_fence_runtime_deployment_v2,
)
from glm52_enforcement.task13_staged_deployment import (
    DeploymentStep,
    StagedDeploymentRequest,
)

SHA_A = "a" * 64


def _support_request() -> dict[str, object]:
    coordinate = {
        "bucket": "keep-glm52-models-246813579024-us-west-2",
        "key": (
            "campaigns/glm52-sky-20260724/authorities/fence/manifests/"
            "glm52-v2-amber-quartz/00000001/FENCE_BOOTSTRAP_MANIFEST.json"
        ),
        "version_id": "manifest-v1",
        "file_sha256": SHA_A,
    }
    coordinate["canonical_identity_sha256"] = canonical_sha256(coordinate)
    return {
        "schema_version": 2,
        "record_type": "glm52_h1g_support_input_materialization_request_v2",
        "activation_id": "glm52-v2-amber-quartz",
        "bootstrap_manifest_coordinate": coordinate,
        "prepare_entry_identity_sha256": SHA_A,
        "host_user_data": "#!/bin/sh\ntrue",
        "host_boot_identity_sha256": SHA_A,
        "cryptography_layer_arn": (
            "arn:aws:lambda:us-west-2:246813579024:layer/h1g-crypto:7"
        ),
        "cryptography_layer_sha256": SHA_A,
        "lambda_code_bucket": "keep-glm52-code-246813579024-us-west-2",
        "lambda_code_key": "task13/code/support.zip",
        "lambda_code_version_id": "code-v1",
        "lambda_code_sha256": SHA_A,
        "price_card_identity_sha256": SHA_A,
        "activation_started_at": "2026-07-31T00:00:00Z",
        "runtime_credential_cutoff_at": "2026-07-31T01:00:00Z",
    }


def _seed_renderer_input() -> dict[str, object]:
    members = (
        PrincipalIdentity(
            binding_id="artifact-writer-one",
            arn=("arn:aws:iam::246813579024:role/artifact-writer-one"),
            role_id="AROAARTIFACTONE00000",
        ),
        PrincipalIdentity(
            binding_id="artifact-writer-two",
            arn=("arn:aws:iam::246813579024:role/artifact-writer-two"),
            role_id="AROAARTIFACTTWO00000",
        ),
    )
    return fence_policy_input_projection(
        FencePolicyInput(
            build_mode=BuildMode.LEGACY_DISABLED,
            policy_head=PolicyHead.BRIDGE_SEED,
            bucket_name="keep-glm52-models-246813579024-us-west-2",
            account_id="246813579024",
            kms_key_arn=(
                "arn:aws:kms:us-west-2:246813579024:"
                "key/01234567-89ab-cdef-0123-456789abcdef"
            ),
            reserved_families=tuple(
                ReservedFamily(
                    family_id=family_id,
                    resources=(resource,),
                )
                for family_id, resource in SOURCE_FAMILY_SPECS
            ),
            writer_cohorts=(
                WriterCohort(
                    cohort_id="bridge-seed",
                    members=members,
                    guard_resources=(RUN_GUARD_RESOURCE,),
                    cross_member_denial_evidence_sha256="b" * 64,
                ),
            ),
        )
    )


def _bootstrap_publication_authority() -> dict[str, object]:
    from test_glm52_fence_bootstrap_publication import (
        _checkpoint,
        _renderer_inputs,
        _request,
    )

    renderer_inputs = _renderer_inputs()
    authority = dict(_request(renderer_inputs, _checkpoint(renderer_inputs)))
    authority["record_type"] = (
        "glm52_fence_bootstrap_publication_authority_v2"
    )
    authority.pop("materializer_function_version_arn")
    authority.pop("checkpoint_identity_sha256")
    return authority


def _request(output_directory: Path) -> dict[str, object]:
    activation_id = "glm52-v2-amber-quartz"
    retained_stack_id = (
        "arn:aws:cloudformation:us-west-2:246813579024:stack/"
        "keep-glm52-gpu/11111111-1111-1111-1111-111111111111"
    )
    kms_key_arn = (
        "arn:aws:kms:us-west-2:246813579024:key/01234567-89ab-cdef-0123-456789abcdef"
    )
    support_request = _support_request()
    return {
        "schema_version": 2,
        "record_type": "glm52_task13_production_operations_v2",
        "activation_id": activation_id,
        "output_directory": str(output_directory),
        "retained_stack_id": retained_stack_id,
        "retained_bootstrap_runtime_deployment": {
            "schema_version": 2,
            "record_type": ("glm52_h1g_retained_fence_bootstrap_inputs_v2"),
            "activation_id": activation_id,
            "retained_stack_id": retained_stack_id,
            "model_bucket_arn": (
                "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2"
            ),
            "retained_kms_key_arn": kms_key_arn,
            "lambda_code_bucket": ("keep-glm52-models-246813579024-us-west-2"),
            "lambda_code_key": ("task13/artifacts/support-lambda/exact.zip"),
            "lambda_code_version_id": "code-v1",
            "lambda_code_sha256": SHA_A,
            "stage_operator_role_arn": ("arn:aws:iam::246813579024:role/operator"),
        },
        "bridge_seed_publication": {
            "schema_version": 2,
            "record_type": ("glm52_h1g_bridge_seed_publication_authority_v2"),
            "activation_id": activation_id,
            "generation": 1,
            "expected_live_preseed_policy_sha256": "c" * 64,
            "renderer_input": _seed_renderer_input(),
        },
        "bridge_seed": {"request": "seed"},
        "migration_operations_1_to_6": {"request": "checkpoint"},
        "bootstrap_fence_publication": _bootstrap_publication_authority(),
        "retained_fence_runtime_deployment": {
            "schema_version": 2,
            "record_type": "glm52_h1g_retained_fence_runtime_authority_v2",
            "activation_id": activation_id,
            "retained_stack_id": retained_stack_id,
            "fence_service_role_arn": (
                "arn:aws:iam::246813579024:role/keep-glm52-h1g-fence-service"
            ),
            "model_bucket_arn": (
                "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2"
            ),
            "retained_kms_key_arn": kms_key_arn,
            "ledger_table_arn": (
                "arn:aws:dynamodb:us-west-2:246813579024:table/keep-glm52-h1g-ledger-v1"
            ),
            "lambda_code_bucket": ("keep-glm52-models-246813579024-us-west-2"),
            "lambda_code_key": ("task13/artifacts/support-lambda/exact.zip"),
            "lambda_code_version_id": "code-v1",
            "lambda_code_sha256": SHA_A,
        },
        "prepare_execution": {"request": "prepare"},
        "support_input_materialization_request": support_request,
        "disabled_support_deployment": {"request": "disabled"},
        "operation_7": {"request": "operation-7"},
        "support_runtime_identity": {"request": "runtime"},
        "no_launch_evidence": {"request": "no-launch"},
    }


def _staged_request(
    tmp_path: Path,
    production_request: dict[str, object],
) -> StagedDeploymentRequest:
    return StagedDeploymentRequest(
        schema_version=2,
        record_type="glm52_task13_staged_deployment_request_v2",
        activation_id=str(production_request["activation_id"]),
        journal_path=tmp_path / "staged-deployment-journal.jsonl",
        production_request=production_request,
    )


class Sts:
    def get_caller_identity(self):
        return {
            "Account": "246813579024",
            "Arn": "arn:aws:iam::246813579024:role/operator",
            "UserId": "AROAEXAMPLE:session",
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "request",
                "HTTPHeaders": {},
                "RetryAttempts": 0,
            },
        }


def _services() -> ProductionServices:
    return ProductionServices(
        sts=Sts(),
        cloudformation=object(),
        s3=object(),
        iam=object(),
        ec2=object(),
        ssm=object(),
        organizations=object(),
        kms=object(),
        dynamodb=object(),
        lambda_client=object(),
        states=object(),
        cloudtrail=object(),
        total_max_attempts=1,
        credential_expiration=datetime(2030, 7, 31, tzinfo=UTC),
    )


class _RetainedReadbackCloudFormation:
    def __init__(
        self,
        *,
        stack_id: str,
        template: dict[str, object],
        outputs: dict[str, str],
    ) -> None:
        self.stack_id = stack_id
        self.template = template
        self.outputs = outputs

    @staticmethod
    def _metadata() -> dict[str, object]:
        return {
            "HTTPStatusCode": 200,
            "RequestId": "request",
            "HTTPHeaders": {},
            "RetryAttempts": 0,
        }

    def describe_stacks(self, **request: object) -> dict[str, object]:
        assert request == {"StackName": self.stack_id}
        return {
            "Stacks": [
                {
                    "StackId": self.stack_id,
                    "StackStatus": "UPDATE_COMPLETE",
                    "Outputs": [
                        {"OutputKey": key, "OutputValue": value}
                        for key, value in self.outputs.items()
                    ],
                }
            ],
            "ResponseMetadata": self._metadata(),
        }

    def get_template(self, **request: object) -> dict[str, object]:
        assert request == {
            "StackName": self.stack_id,
            "TemplateStage": "Original",
        }
        return {
            "TemplateBody": self.template,
            "ResponseMetadata": self._metadata(),
        }


def test_retained_runtime_readers_reconstruct_stable_live_evidence(
    tmp_path: Path,
) -> None:
    from glm52_enforcement.support_plane import (
        build_retained_fence_bootstrap_fragment,
        build_retained_fence_runtime_fragment,
        retained_fence_bootstrap_inputs_from_mapping,
        retained_fence_runtime_inputs_from_mapping,
    )

    production_request = _request(tmp_path)
    bootstrap_request = production_request["retained_bootstrap_runtime_deployment"]
    assert type(bootstrap_request) is dict
    bootstrap_fragment = build_retained_fence_bootstrap_fragment(
        retained_fence_bootstrap_inputs_from_mapping(bootstrap_request)
    )
    bootstrap_outputs = {
        "FenceBootstrapMaterializerVersionArn": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-fence-bootstrap-materializer:1"
        ),
        "FenceBootstrapPublisherRoleArn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-fence-bootstrap-artifact-publisher"
        ),
        "FenceBootstrapInvokerRoleArn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-fence-bootstrap-materializer-invoker"
        ),
    }
    bootstrap_stack = _RetainedReadbackCloudFormation(
        stack_id=str(bootstrap_request["retained_stack_id"]),
        template=bootstrap_fragment,
        outputs=bootstrap_outputs,
    )
    bootstrap = read_retained_bootstrap_runtime_deployment_v2(
        request=bootstrap_request,
        services=replace(_services(), cloudformation=bootstrap_stack),
    )
    assert (
        bootstrap["materializer_function_version_arn"]
        == (bootstrap_outputs["FenceBootstrapMaterializerVersionArn"])
    )
    assert bootstrap["worker_activation_allowed"] is False
    assert bootstrap["source_action_allowed"] is False
    assert bootstrap == read_retained_bootstrap_runtime_deployment_v2(
        request=bootstrap_request,
        services=replace(_services(), cloudformation=bootstrap_stack),
    )

    runtime_authority = production_request["retained_fence_runtime_deployment"]
    assert type(runtime_authority) is dict
    from glm52_enforcement.support_plane import (
        materialize_retained_fence_runtime_inputs,
    )

    manifest_coordinate = _support_request()["bootstrap_manifest_coordinate"]
    runtime_request = materialize_retained_fence_runtime_inputs(
        authority=runtime_authority,
        fence_stack_id=(
            "arn:aws:cloudformation:us-west-2:246813579024:stack/"
            "keep-glm52-h1g-fence/11111111-1111-1111-1111-111111111111"
        ),
        bootstrap_manifest_coordinate=manifest_coordinate,
    )
    runtime_fragment = build_retained_fence_runtime_fragment(
        retained_fence_runtime_inputs_from_mapping(runtime_request)
    )
    runtime_outputs = {
        "PreSupportFenceExecutorVersionArn": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-pre-support-fence-executor:1"
        ),
        "FenceSourceSettlementMaterializerVersionArn": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-fence-source-settlement-materializer:1"
        ),
    }
    runtime_stack = _RetainedReadbackCloudFormation(
        stack_id=str(runtime_request["retained_stack_id"]),
        template=runtime_fragment,
        outputs=runtime_outputs,
    )
    runtime = read_retained_fence_runtime_deployment_v2(
        request=runtime_request,
        checkpoint=SimpleNamespace(fence_stack_id=runtime_request["fence_stack_id"]),
        publication=SimpleNamespace(
            manifest_coordinate=SimpleNamespace(to_dict=lambda: manifest_coordinate)
        ),
        services=replace(_services(), cloudformation=runtime_stack),
    )
    assert runtime["bootstrap_manifest_coordinate"] == manifest_coordinate
    assert (
        runtime["pre_support_executor_function_version_arn"]
        == (runtime_outputs["PreSupportFenceExecutorVersionArn"])
    )
    assert (
        runtime["source_settlement_function_version_arn"]
        == (runtime_outputs["FenceSourceSettlementMaterializerVersionArn"])
    )
    assert runtime["worker_activation_allowed"] is False
    assert runtime["source_action_allowed"] is False


def test_retained_runtime_fragment_composes_after_bootstrap_v9(
    tmp_path: Path,
) -> None:
    from glm52_enforcement.support_plane import (
        build_retained_fence_bootstrap_fragment,
        build_retained_fence_runtime_fragment,
        compose_complete_retained_template,
        materialize_retained_fence_runtime_inputs,
        retained_fence_bootstrap_inputs_from_mapping,
        retained_fence_runtime_inputs_from_mapping,
    )

    production_request = _request(tmp_path)
    bootstrap = build_retained_fence_bootstrap_fragment(
        retained_fence_bootstrap_inputs_from_mapping(
            production_request["retained_bootstrap_runtime_deployment"]
        )
    )
    runtime_request = materialize_retained_fence_runtime_inputs(
        authority=production_request["retained_fence_runtime_deployment"],
        fence_stack_id=(
            "arn:aws:cloudformation:us-west-2:246813579024:stack/"
            "keep-glm52-h1g-fence/11111111-1111-1111-1111-111111111111"
        ),
        bootstrap_manifest_coordinate=_support_request()[
            "bootstrap_manifest_coordinate"
        ],
    )
    runtime = build_retained_fence_runtime_fragment(
        retained_fence_runtime_inputs_from_mapping(runtime_request)
    )
    base = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Metadata": {},
        "Outputs": {},
        "Parameters": {},
        "Resources": {},
    }

    with_bootstrap = compose_complete_retained_template(base, bootstrap)
    complete = compose_complete_retained_template(with_bootstrap, runtime)

    assert "FenceBootstrapArtifactPublisherRole" in complete["Resources"]
    assert (
        "FenceBootstrapArtifactPublisherRole"
        not in runtime["Resources"]
    )


def test_bootstrap_retained_phase_uses_only_fresh_v9_identity() -> None:
    assert production_operations._RETAINED_PHASES["fence-bootstrap-v9"] == {
        "artifact_kind": "RETAINED_FENCE_BOOTSTRAP_TEMPLATE_V9",
        "key": "task13/templates/retained-fence-bootstrap-v9.json",
        "change_set_name": "glm52-h1g-retained-fence-bootstrap-v9",
    }
    assert "fence-bootstrap" not in production_operations._RETAINED_PHASES
    assert "fence-bootstrap-v3" not in production_operations._RETAINED_PHASES
    assert "fence-bootstrap-v4" not in production_operations._RETAINED_PHASES
    assert "fence-bootstrap-v6" not in production_operations._RETAINED_PHASES


def test_bootstrap_deployment_adopts_complete_predeployment_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_request = _request(tmp_path)
    staged_request = _staged_request(tmp_path, production_request)
    fragment = {
        "Resources": {
            "BootstrapRuntime": {
                "Type": "AWS::Lambda::Function",
            }
        }
    }
    expected_runtime = {
        "record_type": "glm52_h1g_retained_bootstrap_runtime_deployment_v2"
    }
    evidence_directory = tmp_path / "retained-fence-bootstrap-v9"
    evidence_directory.mkdir()
    for suffix in ("before-template", "after-template", "change-set", "readback"):
        (evidence_directory / f"retained-fence-bootstrap-v9-{suffix}.json").write_text(
            "{}\n",
            encoding="utf-8",
        )

    adopted: list[dict[str, object]] = []

    def adopt(
        self: Task13ProductionOperations,
        **kwargs: object,
    ) -> dict[str, object]:
        del self
        adopted.append(dict(kwargs))
        return {"phase": "fence-bootstrap-v9"}

    def reject_apply(
        self: Task13ProductionOperations,
        **kwargs: object,
    ) -> dict[str, object]:
        del self, kwargs
        pytest.fail("complete predeployment evidence must be adopted, not re-applied")

    monkeypatch.setattr(
        Task13ProductionOperations,
        "_retained_transaction_evidence",
        adopt,
    )
    monkeypatch.setattr(Task13ProductionOperations, "_apply_retained", reject_apply)
    monkeypatch.setattr(
        production_operations,
        "read_retained_bootstrap_runtime_deployment_v2",
        lambda **_kwargs: expected_runtime,
    )
    operations = Task13ProductionOperations(
        production_request=production_request,
        services=_services(),
        bindings=ProductionOperationBindings.fake_for_tests(
            parse_retained_bootstrap_inputs=lambda value: value,
            build_retained_bootstrap_fragment=lambda _inputs: fragment,
        ),
        sleep=lambda _seconds: None,
        now=lambda: datetime(2030, 7, 31, tzinfo=UTC),
        max_polls=2,
    )

    assert (
        operations.deploy_retained_bootstrap_runtime(staged_request) == expected_runtime
    )
    assert adopted == [
        {
            "phase": "fence-bootstrap-v9",
            "source_fragment": fragment,
        }
    ]


def test_retained_transaction_composition_rejects_foreign_additions() -> None:
    before = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "Existing": {
                "Type": "AWS::S3::Bucket",
            },
            "GpuLaunchTemplate": {
                "Type": "AWS::EC2::LaunchTemplate",
                "Properties": {"LaunchTemplateData": {"ImageId": {"Ref": "GpuAmiId"}}},
            },
        },
        "Outputs": {
            "ExistingName": {
                "Value": {"Ref": "Existing"},
            }
        },
    }
    fragment = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "BootstrapRuntime": {
                "Type": "AWS::Lambda::Function",
            }
        },
        "Outputs": {
            "BootstrapRuntimeArn": {
                "Value": {"Ref": "BootstrapRuntime"},
            }
        },
    }
    after = deepcopy(before)
    after["Resources"].update(fragment["Resources"])
    after["Outputs"].update(fragment["Outputs"])
    after["Resources"]["GpuLaunchTemplate"]["Properties"]["LaunchTemplateData"][
        "ImageId"
    ] = "ami-02b19745d2b303803"
    changes = [
        {
            "action": "Add",
            "logical_id": "BootstrapRuntime",
            "replacement": "False",
            "resource_type": "AWS::Lambda::Function",
        },
        {
            "action": "Modify",
            "logical_id": "GpuLaunchTemplate",
            "replacement": "False",
            "resource_type": "AWS::EC2::LaunchTemplate",
        },
        {
            "action": "Modify",
            "logical_id": "SkyMustStartCancelRole",
            "replacement": "False",
            "resource_type": "AWS::IAM::Role",
        },
        {
            "action": "Modify",
            "logical_id": "SkyPilotControllerRole",
            "replacement": "False",
            "resource_type": "AWS::IAM::Role",
        },
    ]

    _validate_retained_transaction_composition(
        phase="fence-bootstrap-v9",
        before=before,
        after=after,
        source_fragment=fragment,
        changes=changes,
    )

    with pytest.raises(ProductionOperationError, match="composition"):
        _validate_retained_transaction_composition(
            phase="fence-bootstrap-v9",
            before=before,
            after={
                **after,
                "Outputs": {
                    **after["Outputs"],
                    "ForeignOutput": {"Value": "foreign"},
                },
            },
            source_fragment=fragment,
            changes=changes,
        )
    with pytest.raises(ProductionOperationError, match="resource additions"):
        _validate_retained_transaction_composition(
            phase="fence-bootstrap-v9",
            before=before,
            after=after,
            source_fragment=fragment,
            changes=[
                *changes,
                {
                    "action": "Add",
                    "logical_id": "ForeignResource",
                    "replacement": "False",
                    "resource_type": "AWS::IAM::Role",
                },
            ],
        )


def test_bootstrap_reconciliation_rejects_partial_transaction_evidence(
    tmp_path: Path,
) -> None:
    production_request = _request(tmp_path)
    staged_request = _staged_request(tmp_path, production_request)
    evidence_directory = tmp_path / "retained-fence-bootstrap-v9"
    evidence_directory.mkdir()
    (
        evidence_directory / "retained-fence-bootstrap-v9-before-template.json"
    ).write_text("{}\n", encoding="utf-8")

    def reject_generic_recovery(**_kwargs: object) -> object:
        pytest.fail("partial bootstrap evidence reached generic live-only recovery")

    operations = Task13ProductionOperations(
        production_request=production_request,
        services=_services(),
        bindings=ProductionOperationBindings.fake_for_tests(
            parse_retained_bootstrap_inputs=lambda value: value,
            build_retained_bootstrap_fragment=lambda _inputs: {"Resources": {}},
            reconcile_staged_mutation_v2=reject_generic_recovery,
        ),
        sleep=lambda _seconds: None,
        now=lambda: datetime(2030, 7, 31, tzinfo=UTC),
        max_polls=2,
    )

    with pytest.raises(ProductionOperationError, match="incomplete"):
        operations.reconcile_mutation(
            DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED,
            staged_request,
            {},
            {"operation_identity_sha256": SHA_A},
        )


def test_bootstrap_deployment_rejects_partial_predeployment_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_request = _request(tmp_path)
    staged_request = _staged_request(tmp_path, production_request)
    fragment = {"Resources": {}}
    evidence_directory = tmp_path / "retained-fence-bootstrap-v9"
    evidence_directory.mkdir()
    (evidence_directory / "retained-fence-bootstrap-v9-after-template.json").write_text(
        "{}\n", encoding="utf-8"
    )
    applied: list[dict[str, object]] = []

    def record_apply(
        self: Task13ProductionOperations,
        **kwargs: object,
    ) -> dict[str, object]:
        del self
        applied.append(dict(kwargs))
        return {"phase": "fence-bootstrap-v9"}

    monkeypatch.setattr(Task13ProductionOperations, "_apply_retained", record_apply)
    operations = Task13ProductionOperations(
        production_request=production_request,
        services=_services(),
        bindings=ProductionOperationBindings.fake_for_tests(
            parse_retained_bootstrap_inputs=lambda value: value,
            build_retained_bootstrap_fragment=lambda _inputs: fragment,
        ),
        sleep=lambda _seconds: None,
        now=lambda: datetime(2030, 7, 31, tzinfo=UTC),
        max_polls=2,
    )

    with pytest.raises(ProductionOperationError, match="incomplete"):
        operations.deploy_retained_bootstrap_runtime(staged_request)
    assert applied == []


def test_production_request_accepts_exact_v2_and_rejects_v1_support(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    assert _guard_production_request(request) == request
    legacy = dict(request)
    legacy_support = dict(_support_request())
    legacy_support["schema_version"] = 1
    legacy_support["record_type"] = (
        "glm52_task13_support_input_materialization_request_v1"
    )
    legacy["support_input_materialization_request"] = legacy_support
    with pytest.raises(ProductionOperationError, match="v2"):
        _guard_production_request(legacy)


def test_production_request_rejects_caller_authored_fence_inventory(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    support = dict(_support_request())
    support["task11_writer_bindings"] = []
    request["support_input_materialization_request"] = support
    with pytest.raises(ProductionOperationError, match="v2"):
        _guard_production_request(request)


def test_two_snapshot_methods_invoke_fresh_collector_independently(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def collect(*, request, services):
        del request, services
        calls.append("snapshot")
        return {"record_type": "snapshot", "ordinal": len(calls)}

    production_request = _request(tmp_path)
    request = _staged_request(tmp_path, production_request)

    operations = Task13ProductionOperations(
        production_request=production_request,
        services=_services(),
        bindings=ProductionOperationBindings.fake_for_tests(
            collect_support_build_inputs=collect
        ),
        sleep=lambda _seconds: None,
        now=lambda: datetime(2030, 7, 31, tzinfo=UTC),
        max_polls=2,
    )
    first = operations.collect_support_input_snapshot(
        request, object(), object(), object()
    )
    second = operations.collect_support_input_snapshot(
        request, object(), object(), object()
    )
    assert calls == ["snapshot", "snapshot"]
    assert first != second


def test_temporal_methods_delegate_one_exact_public_contract_each(
    tmp_path: Path,
) -> None:
    seen: list[tuple[str, dict[str, object]]] = []

    def bind(name, result=None):
        def call(**kwargs):
            seen.append((name, kwargs))
            return (
                result
                if result is not None
                else {
                    "record_type": name,
                    "canonical_identity_sha256": SHA_A,
                }
            )

        return call

    seed_artifact = SimpleNamespace(to_dict=lambda: {"artifact": "seed"})
    seed_publication = BridgeSeedPublicationV2(
        artifact=seed_artifact,
        materializer_function_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-fence-bootstrap-materializer:1"
        ),
        disposition=FixedKeyPublicationDisposition.WRITTEN,
        canonical_identity_sha256=SHA_A,
    )
    bootstrap_publication = SimpleNamespace(to_dict=lambda: {"manifest": "v1"})

    def invoke_materializer(**kwargs):
        phase = kwargs["event"]["phase"]
        name = "seed-publication" if phase == "BRIDGE_SEED" else "publication"
        seen.append((name, kwargs))
        return seed_publication if phase == "BRIDGE_SEED" else bootstrap_publication

    production_request = _request(tmp_path)
    request = _staged_request(tmp_path, production_request)

    from test_glm52_fence_bootstrap_publication import _checkpoint, _renderer_inputs

    exact_checkpoint = _checkpoint(_renderer_inputs())
    bindings = ProductionOperationBindings.fake_for_tests(
        invoke_bootstrap_materializer_v2=invoke_materializer,
        materialize_bridge_seed_establishment_request_v2=bind(
            "bind-seed",
            {"bound": "seed"},
        ),
        establish_bridge_seed_v2=bind("seed"),
        complete_operations_1_to_6_v2=bind(
            "checkpoint",
            exact_checkpoint,
        ),
        execute_prepare_v2=bind("prepare"),
        deploy_disabled_support_v2=bind("disabled"),
        complete_operation_7_v2=bind("operation-7"),
        commit_support_runtime_identity_v2=bind("runtime"),
        prove_no_launch_v2=bind("no-launch"),
    )
    operations = Task13ProductionOperations(
        production_request=production_request,
        services=_services(),
        bindings=bindings,
        sleep=lambda _seconds: None,
        now=lambda: datetime(2030, 7, 31, tzinfo=UTC),
        max_polls=2,
    )
    bootstrap_runtime = {
        "materializer_function_version_arn": (
            seed_publication.materializer_function_version_arn
        ),
        "invoker_role_arn": (
            "arn:aws:iam::246813579024:role/keep-glm52-h1g-fence-bootstrap-invoker"
        ),
    }
    published_seed = operations.publish_bridge_seed(request, bootstrap_runtime)
    seed = operations.establish_bridge_seed(request, published_seed)
    checkpoint = operations.complete_stack_migration_operations_1_to_6(request, seed)
    publication = operations.publish_bootstrap_fence_artifacts(
        request, checkpoint, bootstrap_runtime
    )
    prepare = operations.execute_prepare(request, checkpoint, publication)
    disabled = operations.deploy_disabled_support(request, checkpoint, object())
    operation_7 = operations.complete_stack_migration_operation_7(
        request, checkpoint, prepare, disabled
    )
    runtime = operations.commit_support_runtime_identity(request, operation_7, disabled)
    operations.prove_no_worker_activation(request, runtime)
    assert [name for name, _kwargs in seen] == [
        "seed-publication",
        "bind-seed",
        "seed",
        "checkpoint",
        "publication",
        "prepare",
        "disabled",
        "operation-7",
        "runtime",
        "no-launch",
    ]


def test_default_adapter_binds_every_staged_transition(tmp_path: Path) -> None:
    operations = build_staged_deployment_operations(
        _request(tmp_path),
        services=_services(),
        sleep=lambda _seconds: None,
        now=lambda: datetime(2030, 7, 31, tzinfo=UTC),
        max_polls=2,
    )
    expected_modules = {
        "materialize_bridge_seed_establishment_request_v2": (
            "staged_migration_operations"
        ),
        "establish_bridge_seed_v2": "staged_migration_operations",
        "complete_operations_1_to_6_v2": "staged_migration_operations",
        "publish_bootstrap_fence_artifacts_v2": "fence_bootstrap_publication",
        "execute_prepare_v2": "task13_production_operations",
        "deploy_disabled_support_v2": "support_runtime_transition",
        "complete_operation_7_v2": "support_runtime_transition",
        "commit_support_runtime_identity_v2": "support_runtime_transition",
        "prove_no_launch_v2": "support_runtime_transition",
        "reconcile_staged_mutation_v2": "staged_effect_recovery",
        "adopt_staged_evidence_v2": "staged_effect_recovery",
    }
    assert {
        name: getattr(operations.bindings, name).__module__.rsplit(".", 1)[-1]
        for name in expected_modules
    } == expected_modules


def test_prepare_binds_full_fence_aws_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import (
        fence_artifacts,
        fence_executor,
        task13_migration_adapter,
        task13_staged_deployment,
    )
    from glm52_enforcement.task11_production import _Task11FenceAwsClient
    from glm52_enforcement.task13_production_operations import (
        _execute_prepare_v2_binding,
    )

    stack_id = (
        "arn:aws:cloudformation:us-west-2:246813579024:"
        "stack/keep-glm52-h1g-fence/"
        "12345678-1234-1234-1234-123456789abc"
    )
    manifest_coordinate = {
        "bucket": "keep-glm52-models-246813579024-us-west-2",
        "key": "task13/migration/fence-transfer.json",
        "version_id": "manifest-version",
        "file_sha256": SHA_A,
        "canonical_identity_sha256": SHA_A,
    }
    coordinate = SimpleNamespace(to_dict=lambda: dict(manifest_coordinate))

    class Checkpoint:
        fence_stack_id = stack_id

    class Publication:
        manifest_coordinate = coordinate
        prepare_entry_identity_sha256 = SHA_A

    transition = SimpleNamespace(
        slot=fence_artifacts.FenceSlot.PREPARE_GENESIS_LIVE_STATE,
        to_dict=lambda: {"manifest_coordinate": dict(manifest_coordinate)},
    )
    entry = SimpleNamespace(entry_identity_sha256=SHA_A)
    manifest = SimpleNamespace(entry=lambda _slot: entry)
    captured: dict[str, object] = {}

    class RecordStore:
        def __init__(self, *, client: object, table_name: str) -> None:
            captured["record_client"] = client
            captured["table_name"] = table_name

    class Executor:
        def __init__(self, *, client: object, record_store: object) -> None:
            captured["client"] = client
            captured["record_store"] = record_store

    class Events:
        def register(self, *_args: object, **_kwargs: object) -> None:
            return None

    cloudformation = SimpleNamespace(meta=SimpleNamespace(events=Events()))
    services = replace(
        _services(),
        cloudformation=cloudformation,
    )
    monkeypatch.setattr(
        task13_migration_adapter,
        "StackMigrationTransferCheckpointV2",
        Checkpoint,
    )
    monkeypatch.setattr(
        task13_staged_deployment,
        "BootstrapFencePublication",
        Publication,
    )
    monkeypatch.setattr(
        fence_artifacts,
        "parse_fence_transition_request",
        lambda _value: transition,
    )
    monkeypatch.setattr(fence_executor, "DynamoFenceRecordStore", RecordStore)
    monkeypatch.setattr(fence_executor, "FenceExecutor", Executor)
    monkeypatch.setattr(
        fence_executor,
        "load_pinned_fence_manifest",
        lambda **_kwargs: manifest,
    )
    monkeypatch.setattr(
        fence_executor,
        "load_pinned_fence_entry_template",
        lambda **_kwargs: entry,
    )
    monkeypatch.setattr(
        fence_executor,
        "execute_prepare_v2",
        lambda **_kwargs: SimpleNamespace(stack_id=stack_id),
    )

    result = _execute_prepare_v2_binding(
        request={"manifest_coordinate": dict(manifest_coordinate)},
        checkpoint=Checkpoint(),
        publication=Publication(),
        services=services,
    )

    assert result.stack_id == stack_id
    assert type(captured["client"]) is _Task11FenceAwsClient
    assert callable(captured["client"].execute_change_set)
    assert callable(captured["client"].observe_fence_poststate)
    assert captured["record_client"] is services.dynamodb
    assert captured["table_name"] == "keep-glm52-h1g-ledger-v1"
