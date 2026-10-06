from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
import importlib
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.support_plane import SupportRuntimeIdentity
from glm52_enforcement.task13_production_operations import ProductionServices
import glm52_enforcement.support_runtime_transition as sut

ROOT = Path(__file__).resolve().parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64
ACTIVATION = "activation-test"
STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:stack/"
    "keep-glm52-h1g-support/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
)
VERSION_ARN = (
    "arn:aws:lambda:us-west-2:246813579024:function:"
    "keep-glm52-h1g-fence-executor:7"
)
ROLE_ARN = (
    "arn:aws:iam::246813579024:role/keep-glm52-h1g-fence-executor"
)


def _meta(request_id: str = "request") -> dict[str, object]:
    return {
        "ResponseMetadata": {
            "HTTPStatusCode": 200,
            "RequestId": request_id,
            "RetryAttempts": 0,
        }
    }


def _services(**overrides: object) -> ProductionServices:
    clients = {
        "sts": object(),
        "cloudformation": object(),
        "iam": object(),
        "s3": object(),
        "organizations": object(),
        "ec2": object(),
        "ssm": object(),
        "kms": object(),
        "dynamodb": object(),
        "lambda_client": object(),
        "states": object(),
        "cloudtrail": object(),
        "total_max_attempts": 1,
    }
    clients.update(overrides)
    return ProductionServices(**clients)


def _manifest() -> dict[str, object]:
    return {
        "bucket": "keep-glm52-models-246813579024-us-west-2",
        "key": (
            "campaigns/glm52-sky-20260724/authorities/fence/manifests/"
            f"{ACTIVATION}/00000001/FENCE_BOOTSTRAP_MANIFEST.json"
        ),
        "version_id": "opaque-version",
        "file_sha256": SHA_A,
        "canonical_identity_sha256": SHA_B,
    }


def _disabled_request() -> dict[str, object]:
    from glm52_enforcement.task13_support_input_materialization import (
        support_build_inputs_projection,
    )

    fixture = _load_fixture(
        "tests/test_glm52_enforcement_support_plane.py",
        "_closed_support_request_fixture",
    )
    inputs = fixture._support_build_inputs()
    return {
        "schema_version": 1,
        "record_type": "glm52_disabled_support_deployment_request_v1",
        "activation_id": inputs.activation_id,
        "generation": 1,
        "bootstrap_manifest_coordinate": dict(
            inputs.bootstrap_manifest_coordinate
        ),
        "prepare_entry_identity_sha256": (
            inputs.prepare_entry_identity_sha256
        ),
        "support_build_inputs_identity_sha256": (
            sut.support_build_inputs_identity(inputs)
        ),
        "support_build_inputs": support_build_inputs_projection(inputs),
        "price_card": fixture._price_card(),
        "kms_key_arn": (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        ),
    }


def _operation_request() -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_support_replacement_request_v1",
        "activation_id": ACTIVATION,
        "generation": 1,
        "migration_seed": {
            "authority": {"action_identity_sha256": SHA_A}
        },
        "checkpoint_identity_sha256": SHA_A,
        "prepare_execution_identity_sha256": SHA_B,
        "disabled_support_identity_sha256": "c" * 64,
    }


def _expected_contract() -> dict[str, object]:
    contract = {
        "authority_class": "SUPPORT_RUNTIME",
        "function_logical_id": "FenceExecutorFunction",
        "role_logical_id": "FenceExecutorRole",
        "function_code_sha256": SHA_A,
        "execution_role_arn": ROLE_ARN,
        "role_trust_policy_sha256": SHA_A,
        "role_permission_policy_sha256": SHA_A,
        "expected_attachment_identity_sha256": canonical_sha256([]),
        "disabled_support_profile_sha256": sut._disabled_profile_identity(),
    }
    contract["canonical_identity_sha256"] = canonical_sha256(contract)
    return contract


def _runtime_request() -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_support_runtime_commit_request_v1",
        "activation_id": ACTIVATION,
        "generation": 1,
        "bootstrap_manifest_coordinate": _manifest(),
        "operation_7_identity_sha256": SHA_A,
        "disabled_support_identity_sha256": SHA_B,
        "expected_contract": _expected_contract(),
        "kms_key_arn": (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        ),
        "ledger_table_name": "keep-glm52-h1g-ledger-v1",
    }


def _no_launch_request() -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_no_launch_observation_request_v1",
        "activation_id": ACTIVATION,
        "generation": 1,
        "runtime_identity_sha256": SHA_A,
        "ledger_table_name": "keep-glm52-h1g-ledger-v1",
        "evidence_window_start": "2026-07-31T00:00:00Z",
        "evidence_window_end": "2026-07-31T00:05:00Z",
    }


@pytest.mark.parametrize(
    ("factory", "parser"),
    [
        (_disabled_request, sut.parse_disabled_support_deployment_request),
        (_operation_request, sut.parse_operation_seven_completion_request),
        (_runtime_request, sut.parse_support_runtime_commit_request),
        (_no_launch_request, sut.parse_no_launch_observation_request),
    ],
)
def test_every_request_schema_rejects_missing_and_unknown_fields(factory, parser) -> None:
    value = factory()
    missing = dict(value)
    missing.pop(next(iter(missing)))
    with pytest.raises(sut.SupportRuntimeTransitionError, match="missing"):
        parser(missing)

    unknown = dict(value)
    unknown["caller_rendered_template"] = "forbidden"
    with pytest.raises(sut.SupportRuntimeTransitionError, match="unknown"):
        parser(unknown)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(generation=True), "activation/generation"),
        (
            lambda value: value["bootstrap_manifest_coordinate"].update(
                key="campaigns/glm52-sky-20260724/foreign.json"
            ),
            "foreign",
        ),
        (lambda value: value.update(kms_key_arn="alias/guess"), "KMS"),
    ],
)
def test_disabled_request_binds_outer_coordinate_and_kms(mutation, message) -> None:
    value = _disabled_request()
    mutation(value)
    with pytest.raises(sut.SupportRuntimeTransitionError, match=message):
        sut.parse_disabled_support_deployment_request(value)


def test_no_launch_window_must_be_positive_canonical_and_bounded() -> None:
    value = _no_launch_request()
    value["evidence_window_end"] = "2026-07-31T02:00:00Z"
    with pytest.raises(sut.SupportRuntimeTransitionError, match="unbounded"):
        sut.parse_no_launch_observation_request(value)
    value = _no_launch_request()
    value["evidence_window_start"] = "2026-07-31T00:00:00+00:00"
    with pytest.raises(sut.SupportRuntimeTransitionError, match="canonical UTC"):
        sut.parse_no_launch_observation_request(value)


def _load_fixture(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disabled_support_rejects_staged_snapshot_drift_before_effect() -> None:
    migration = _load_fixture(
        "tests/test_glm52_task13_migration_adapter.py",
        "_snapshot_checkpoint_fixture",
    )
    request = _disabled_request()
    inputs = sut.parse_disabled_support_deployment_request(
        request
    ).support_build_inputs
    drifted = replace(
        inputs,
        host_boot_identity_sha256="0" * 64,
    )
    with pytest.raises(
        sut.SupportRuntimeTransitionError,
        match="staged snapshot",
    ):
        sut.deploy_disabled_support_v2(
            request=request,
            checkpoint=migration._checkpoint(),
            support_inputs=drifted,
            services=_services(),
        )


def test_disabled_support_renders_locally_and_calls_one_publish_or_adopt(
    monkeypatch,
) -> None:
    migration = _load_fixture(
        "tests/test_glm52_task13_migration_adapter.py",
        "_support_checkpoint_fixture",
    )
    request = _disabled_request()
    checkpoint = migration._checkpoint()
    calls: list[dict[str, object]] = []
    inputs = sut.parse_disabled_support_deployment_request(
        request
    ).support_build_inputs

    def publish(**kwargs):
        calls.append(kwargs)
        return "created-or-adopted-version"

    fixed = importlib.import_module(
        "glm52_enforcement.task13_fixed_artifacts"
    )
    monkeypatch.setattr(fixed, "publish_fixed_key_bytes", publish)
    monkeypatch.setattr(
        sut,
        "support_price_card_from_mapping",
        lambda value, *, inputs: SimpleNamespace(),
    )
    template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "FenceExecutorFunction": {
                "Type": "AWS::Lambda::Function",
                "Properties": {},
            }
        },
    }
    monkeypatch.setattr(
        sut,
        "build_support_precreate_plane",
        lambda *, inputs, price_card: SimpleNamespace(support_template=template),
    )
    anchor_reads: list[object] = []
    monkeypatch.setattr(
        sut,
        "_require_inert_support_anchor",
        lambda **kwargs: anchor_reads.append(kwargs["checkpoint"]),
    )
    result = sut.deploy_disabled_support_v2(
        support_inputs=inputs,
        request=request,
        checkpoint=checkpoint,
        services=_services(),
    )
    assert len(calls) == 1
    assert anchor_reads == [checkpoint]
    assert calls[0]["raw"].endswith(b"\n")
    assert b"caller_rendered_template" not in calls[0]["raw"]
    assert result.support_template_coordinate["version_id"] == (
        "created-or-adopted-version"
    )
    assert result.disabled_support_profile_sha256 == sut._disabled_profile_identity()


def test_disabled_support_lost_publication_response_never_mutates_anchor(
    monkeypatch,
) -> None:
    migration = _load_fixture(
        "tests/test_glm52_task13_migration_adapter.py",
        "_support_lost_checkpoint_fixture",
    )
    checkpoint = migration._checkpoint()
    request = _disabled_request()
    calls = 0
    inputs = sut.parse_disabled_support_deployment_request(
        request
    ).support_build_inputs

    def lost(**_kwargs):
        nonlocal calls
        calls += 1
        raise ValueError("ambiguous write has no exact durable version")

    fixed = importlib.import_module(
        "glm52_enforcement.task13_fixed_artifacts"
    )
    monkeypatch.setattr(fixed, "publish_fixed_key_bytes", lost)
    monkeypatch.setattr(
        sut,
        "support_price_card_from_mapping",
        lambda value, *, inputs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        sut,
        "build_support_precreate_plane",
        lambda **kwargs: SimpleNamespace(
            support_template={
                "Resources": {
                    "Only": {"Type": "AWS::Lambda::Function", "Properties": {}}
                }
            }
        ),
    )
    with pytest.raises(ValueError, match="no exact durable"):
        sut.deploy_disabled_support_v2(
            request=request,
            checkpoint=checkpoint,
            support_inputs=inputs,
            services=_services(),
        )
    assert calls == 1


class _AnchorCloudFormation:
    def __init__(self, checkpoint, template):
        self.checkpoint = checkpoint
        self.template = template
        self.calls: list[str] = []

    def describe_stacks(self, **request):
        self.calls.append("DescribeStacks")
        assert request == {"StackName": self.checkpoint.support_stack_id}
        return {
            "Stacks": [
                {
                    "StackId": self.checkpoint.support_stack_id,
                    "StackName": "keep-glm52-h1g-support",
                    "StackStatus": "CREATE_COMPLETE",
                    "RoleARN": self.checkpoint.migration_service_role.role_arn,
                    "EnableTerminationProtection": True,
                    "Tags": [dict(row) for row in sut._STACK_TAGS],
                }
            ],
            **_meta("anchor-stack"),
        }

    def get_template(self, **request):
        self.calls.append("GetTemplate" + request["TemplateStage"])
        return {
            "TemplateBody": self.template,
            **_meta("anchor-template-" + request["TemplateStage"]),
        }


def test_disabled_support_requires_exact_two_stage_inert_anchor_read() -> None:
    fixture = _load_fixture(
        "tests/test_glm52_task13_migration_adapter.py",
        "_anchor_checkpoint_fixture",
    )
    checkpoint = fixture._checkpoint()
    anchor = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "ContainerAnchor": {
                "Type": "AWS::CloudFormation::WaitConditionHandle"
            }
        },
    }
    object.__setattr__(
        checkpoint,
        "support_prestate_template_sha256",
        canonical_sha256(anchor),
    )
    cloudformation = _AnchorCloudFormation(checkpoint, anchor)
    sut._require_inert_support_anchor(
        checkpoint=checkpoint,
        services=_services(cloudformation=cloudformation),
    )
    assert cloudformation.calls == [
        "DescribeStacks",
        "GetTemplateOriginal",
        "GetTemplateProcessed",
    ]

    cloudformation.template = {
        "Resources": {
            "FenceExecutorFunction": {"Type": "AWS::Lambda::Function"}
        }
    }
    with pytest.raises(
        sut.SupportRuntimeTransitionError,
        match="anchor template",
    ):
        sut._require_inert_support_anchor(
            checkpoint=checkpoint,
            services=_services(cloudformation=cloudformation),
        )


def test_operation_seven_delegates_once_and_validates_complete_scope(
    monkeypatch,
) -> None:
    fixture = _load_fixture(
        "tests/test_glm52_task13_migration_adapter.py", "_migration_fixture"
    )
    checkpoint = fixture._checkpoint()
    prepare = fixture._prepare_result(checkpoint)
    disabled = fixture._disabled_support(prepare)
    expected = fixture._operation_7_evidence()
    request = _operation_request()
    request.update(
        activation_id=disabled.bootstrap_manifest_coordinate.key.split("/")[-3],
        generation=int(disabled.bootstrap_manifest_coordinate.key.split("/")[-2]),
        checkpoint_identity_sha256=checkpoint.canonical_identity_sha256,
        prepare_execution_identity_sha256=prepare.canonical_identity_sha256,
        disabled_support_identity_sha256=disabled.canonical_identity_sha256,
    )
    runtime = SimpleNamespace(
        seed=SimpleNamespace(
            authority=SimpleNamespace(
                action_identity_sha256=request["migration_seed"]["authority"][
                    "action_identity_sha256"
                ]
            )
        )
    )
    calls = 0
    monkeypatch.setattr(sut, "build_seeded_migration_runtime", lambda **kwargs: runtime)
    monkeypatch.setattr(sut, "_fresh_role_evidence", lambda services, expected: expected)
    import glm52_enforcement.task13_migration_adapter as migration

    def execute(**kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["runtime"] is runtime
        return expected

    monkeypatch.setattr(migration, "execute_stack_migration_operation_7_v2", execute)
    assert (
        sut.complete_operation_7_v2(
            request=request,
            checkpoint=checkpoint,
            prepare_result=prepare,
            disabled_support=disabled,
            services=_services(),
        )
        == expected
    )
    assert calls == 1

    request["prepare_execution_identity_sha256"] = "0" * 64
    with pytest.raises(sut.SupportRuntimeTransitionError, match="ordering"):
        sut.complete_operation_7_v2(
            request=request,
            checkpoint=checkpoint,
            prepare_result=prepare,
            disabled_support=disabled,
            services=_services(),
        )
    assert calls == 1


class _AwsNotFound(Exception):
    def __init__(self) -> None:
        self.response = {
            "Error": {"Code": "ResourceNotFoundException"},
            "ResponseMetadata": {
                "HTTPStatusCode": 404,
                "RequestId": "not-found",
                "RetryAttempts": 0,
            },
        }


class _RuntimeCloudFormation:
    def __init__(self, stack_id: str = STACK_ID) -> None:
        self.stack_id = stack_id

    def describe_stacks(self, **request):
        assert request == {"StackName": self.stack_id}
        return {
            "Stacks": [
                {
                    "StackId": self.stack_id,
                    "StackName": "keep-glm52-h1g-support",
                    "StackStatus": "UPDATE_COMPLETE",
                    "EnableTerminationProtection": True,
                }
            ],
            **_meta("describe"),
        }

    def get_template(self, **request):
        assert request["TemplateStage"] == "Original"
        return {
            "TemplateBody": {
                "Resources": {
                    "FenceExecutorFunction": {
                        "Type": "AWS::Lambda::Function",
                        "Properties": {},
                    },
                    "FenceExecutorVersion": {
                        "Type": "AWS::Lambda::Version",
                        "Properties": {},
                    },
                    "FenceExecutorRole": {
                        "Type": "AWS::IAM::Role",
                        "Properties": {},
                    },
                }
            },
            **_meta("template"),
        }

    def describe_stack_resource(self, **request):
        physical = {
            "FenceExecutorFunction": "keep-glm52-h1g-fence-executor",
            "FenceExecutorVersion": VERSION_ARN,
            "FenceExecutorRole": "keep-glm52-h1g-fence-executor",
        }[request["LogicalResourceId"]]
        return {
            "StackResourceDetail": {
                "StackId": self.stack_id,
                "LogicalResourceId": request["LogicalResourceId"],
                "PhysicalResourceId": physical,
            },
            **_meta("resource-" + request["LogicalResourceId"]),
        }


class _RuntimeIam:
    def get_role(self, **request):
        return {
            "Role": {
                "Arn": ROLE_ARN,
                "RoleId": "AROAABCDEFGHIJKLMNOP",
                "AssumeRolePolicyDocument": {"Version": "2012-10-17", "Statement": []},
            },
            **_meta("role"),
        }

    def list_role_policies(self, **request):
        assert request["MaxItems"] == 100
        return {"PolicyNames": [], "IsTruncated": False, **_meta("inline")}

    def list_attached_role_policies(self, **request):
        assert request["MaxItems"] == 100
        return {"AttachedPolicies": [], "IsTruncated": False, **_meta("attached")}


class _RuntimeLambda:
    def get_function_configuration(self, **request):
        return {
            "FunctionName": "keep-glm52-h1g-fence-executor",
            "FunctionArn": VERSION_ARN,
            "Role": ROLE_ARN,
            "State": "Active",
            "LastUpdateStatus": "Successful",
            "CodeSha256": "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqo=",
            "Environment": {"Variables": {"GLM52_AUTHORITY_CLASS": "SUPPORT_RUNTIME"}},
            **_meta("function"),
        }

    def get_policy(self, **request):
        raise _AwsNotFound()

    def list_event_source_mappings(self, **request):
        assert request["MaxItems"] == 16
        return {"EventSourceMappings": [], **_meta("mappings")}

    def get_function_url_config(self, **request):
        raise _AwsNotFound()


def test_runtime_live_fields_are_derived_from_fresh_aws_responses() -> None:
    services = _services(
        cloudformation=_RuntimeCloudFormation(),
        iam=_RuntimeIam(),
        lambda_client=_RuntimeLambda(),
    )
    live, launch = sut._fresh_runtime_readback(
        services=services, support_stack_id=STACK_ID
    )
    assert launch is False
    assert live["function_version_arn"] == VERSION_ARN
    assert live["execution_role_id"] == "AROAABCDEFGHIJKLMNOP"
    assert live["stack_id"] == STACK_ID
    assert live["event_source_state_sha256"] == canonical_sha256([])
    assert live["expected_attachment_identity_sha256"] == canonical_sha256([])
    assert live["disabled_support_profile_sha256"] == sut._disabled_profile_identity()


@pytest.mark.parametrize(
    "method_name", ("get_policy", "get_function_url_config")
)
def test_runtime_commit_propagates_unexpected_lambda_observation_defect(
    method_name,
) -> None:
    fixture = _load_fixture(
        "tests/test_glm52_task13_migration_adapter.py", "_runtime_fixture"
    )
    operation = fixture._operation_7_evidence()
    checkpoint = fixture._checkpoint()
    prepare = fixture._prepare_result(checkpoint)
    disabled = fixture._disabled_support(prepare)
    request = _runtime_request()
    request.update(
        activation_id=disabled.bootstrap_manifest_coordinate.key.split("/")[-3],
        bootstrap_manifest_coordinate=disabled.bootstrap_manifest_coordinate.to_dict(),
        operation_7_identity_sha256=operation.canonical_identity_sha256,
        disabled_support_identity_sha256=disabled.canonical_identity_sha256,
    )
    expected = dict(request["expected_contract"])
    expected["disabled_support_profile_sha256"] = (
        disabled.disabled_support_profile_sha256
    )
    unsigned = dict(expected)
    unsigned.pop("canonical_identity_sha256")
    expected["canonical_identity_sha256"] = canonical_sha256(unsigned)
    request["expected_contract"] = expected
    lambda_client = _RuntimeLambda()
    defect = AssertionError("unexpected Lambda observation defect")

    def fail(**_request):
        raise defect

    setattr(lambda_client, method_name, fail)

    with pytest.raises(AssertionError) as raised:
        sut.commit_support_runtime_identity_v2(
            request=request,
            operation_7=operation,
            disabled_support=disabled,
            services=_services(
                cloudformation=_RuntimeCloudFormation(
                    operation.support_stack_id
                ),
                iam=_RuntimeIam(),
                lambda_client=lambda_client,
            ),
        )

    assert raised.value is defect


def test_runtime_commit_requires_two_independent_equal_live_reads(monkeypatch) -> None:
    fixture = _load_fixture(
        "tests/test_glm52_task13_migration_adapter.py", "_runtime_fixture"
    )
    operation = fixture._operation_7_evidence()
    checkpoint = fixture._checkpoint()
    prepare = fixture._prepare_result(checkpoint)
    disabled = fixture._disabled_support(prepare)
    request = _runtime_request()
    request.update(
        activation_id=disabled.bootstrap_manifest_coordinate.key.split("/")[-3],
        bootstrap_manifest_coordinate=disabled.bootstrap_manifest_coordinate.to_dict(),
        operation_7_identity_sha256=operation.canonical_identity_sha256,
        disabled_support_identity_sha256=disabled.canonical_identity_sha256,
    )
    expected = dict(request["expected_contract"])
    expected["disabled_support_profile_sha256"] = disabled.disabled_support_profile_sha256
    unsigned = dict(expected)
    unsigned.pop("canonical_identity_sha256")
    expected["canonical_identity_sha256"] = canonical_sha256(unsigned)
    request["expected_contract"] = expected
    identity = SupportRuntimeIdentity(
        MappingProxyType(
            {
                "activation_id": request["activation_id"],
                "generation": 1,
                "canonical_identity_sha256": SHA_A,
                "expected_contract": MappingProxyType(expected),
                "live_identity": MappingProxyType({"stack_id": operation.support_stack_id}),
            }
        )
    )
    reads = [({"read": 1}, False), ({"read": 2}, False)]
    monkeypatch.setattr(sut, "_fresh_runtime_readback", lambda **kwargs: reads.pop(0))
    monkeypatch.setattr(sut, "build_support_runtime_identity", lambda **kwargs: identity)
    monkeypatch.setattr(sut, "publish_support_runtime_identity", lambda *args, **kwargs: object())
    monkeypatch.setattr(sut, "commit_support_runtime_identity", lambda *args, **kwargs: {})
    monkeypatch.setattr(sut, "load_support_runtime_identity", lambda **kwargs: (identity, object()))
    monkeypatch.setattr(sut, "_reread_runtime_artifact", lambda **kwargs: identity)

    def require(loaded, *, expected_contract, live_readback):
        assert live_readback == {"read": 2}
        return loaded

    monkeypatch.setattr(sut, "require_live_support_runtime_identity", require)
    assert (
        sut.commit_support_runtime_identity_v2(
            request=request,
            operation_7=operation,
            disabled_support=disabled,
            services=_services(),
        )
        is identity
    )
    assert reads == []


class _ZeroDynamo:
    def __init__(self, counts=None, unbounded=False):
        self.counts = list(counts or [0, 0, 0, 0])
        self.unbounded = unbounded

    def query(self, **request):
        assert request["ConsistentRead"] is True
        assert request["Limit"] == 1
        count = self.counts.pop(0)
        return {
            "Count": count,
            "ScannedCount": count,
            "LastEvaluatedKey": {"PK": {"S": "more"}} if self.unbounded else None,
            **_meta("query"),
        }


class _ZeroEc2:
    def __init__(self, workers=0, unbounded=False):
        self.workers = workers
        self.unbounded = unbounded

    def describe_instances(self, **request):
        return {
            "Reservations": [
                {
                    "Instances": [
                        {"InstanceId": f"i-{index:017d}", "InstanceType": "p5.48xlarge"}
                        for index in range(self.workers)
                    ]
                }
            ],
            "NextToken": "more" if self.unbounded else None,
            **_meta("ec2"),
        }


class _ZeroCloudTrail:
    def __init__(self, event_name=None, unbounded=False):
        self.event_name = event_name
        self.unbounded = unbounded

    def lookup_events(self, **request):
        requested = request["LookupAttributes"][0]["AttributeValue"]
        events = []
        if requested == self.event_name:
            event_id = "event-1"
            event_time = datetime(2026, 7, 31, 0, 1, tzinfo=timezone.utc)
            events = [
                {
                    "EventId": event_id,
                    "EventName": requested,
                    "EventTime": event_time,
                    "EventSource": "ec2.amazonaws.com",
                    "CloudTrailEvent": (
                        '{"awsRegion":"us-west-2","eventID":"event-1",'
                        f'"eventName":"{requested}","eventSource":"ec2.amazonaws.com"}}'
                    ),
                }
            ]
        return {
            "Events": events,
            "NextToken": "more" if self.unbounded else None,
            **_meta("trail-" + requested),
        }


def _runtime_identity() -> SupportRuntimeIdentity:
    return SupportRuntimeIdentity(
        MappingProxyType(
            {
                "activation_id": ACTIVATION,
                "generation": 1,
                "canonical_identity_sha256": SHA_A,
                "expected_contract": MappingProxyType({}),
                "live_identity": MappingProxyType({"stack_id": STACK_ID}),
            }
        )
    )


def _patch_runtime_equality(monkeypatch, *, launch_present=False) -> None:
    monkeypatch.setattr(
        sut,
        "_fresh_runtime_readback",
        lambda **kwargs: ({"fresh": True}, launch_present),
    )
    monkeypatch.setattr(
        sut,
        "require_live_support_runtime_identity",
        lambda identity, **kwargs: identity,
    )


class _IncompleteCloudTrailEventWithExtraField(_ZeroCloudTrail):
    def __init__(self):
        super().__init__(event_name="RunInstances")

    def lookup_events(self, **request):
        response = super().lookup_events(**request)
        events = response["Events"]
        if events:
            event = events[0]
            event["UnexpectedField"] = "must-not-mask-missing-field"
            del event["EventId"]
        return response


def test_no_launch_returns_only_the_exact_seven_field_zero_proof(monkeypatch) -> None:
    _patch_runtime_equality(monkeypatch)
    result = sut.prove_no_launch_v2(
        request=_no_launch_request(),
        runtime_identity=_runtime_identity(),
        services=_services(
            dynamodb=_ZeroDynamo(),
            ec2=_ZeroEc2(),
            cloudtrail=_ZeroCloudTrail(),
        ),
    )
    assert result == {
        "schema_version": 2,
        "record_type": "glm52_h1g_no_launch_evidence_v2",
        "worker_count": 0,
        "raw_ec2_launch_calls": 0,
        "source_action_calls": 0,
        "launch_authority_present": False,
        "complete": True,
    }


def test_no_launch_rejects_incomplete_cloudtrail_event_with_extra_field(
    monkeypatch,
) -> None:
    _patch_runtime_equality(monkeypatch)
    with pytest.raises(
        sut.SupportRuntimeTransitionError,
        match="CloudTrail launch event is incomplete",
    ):
        sut.prove_no_launch_v2(
            request=_no_launch_request(),
            runtime_identity=_runtime_identity(),
            services=_services(
                dynamodb=_ZeroDynamo(),
                ec2=_ZeroEc2(),
                cloudtrail=_IncompleteCloudTrailEventWithExtraField(),
            ),
        )


@pytest.mark.parametrize(
    ("counts", "workers", "trail_event", "launch_present"),
    [
        ([1, 0, 0, 0], 0, None, False),
        ([0, 1, 0, 0], 0, None, False),
        ([0, 0, 1, 0], 0, None, False),
        ([0, 0, 0, 1], 0, None, False),
        ([0, 0, 0, 0], 1, None, False),
        ([0, 0, 0, 0], 0, "RunInstances", False),
        ([0, 0, 0, 0], 0, None, True),
    ],
)
def test_no_launch_rejects_every_nonzero_category(
    monkeypatch, counts, workers, trail_event, launch_present
) -> None:
    _patch_runtime_equality(monkeypatch, launch_present=launch_present)
    with pytest.raises(sut.SupportRuntimeTransitionError, match="nonzero"):
        sut.prove_no_launch_v2(
            request=_no_launch_request(),
            runtime_identity=_runtime_identity(),
            services=_services(
                dynamodb=_ZeroDynamo(counts),
                ec2=_ZeroEc2(workers),
                cloudtrail=_ZeroCloudTrail(trail_event),
            ),
        )


@pytest.mark.parametrize("source", ["dynamodb", "ec2", "cloudtrail"])
def test_no_launch_rejects_every_unbounded_inventory(monkeypatch, source) -> None:
    _patch_runtime_equality(monkeypatch)
    services = _services(
        dynamodb=_ZeroDynamo(unbounded=source == "dynamodb"),
        ec2=_ZeroEc2(unbounded=source == "ec2"),
        cloudtrail=_ZeroCloudTrail(unbounded=source == "cloudtrail"),
    )
    with pytest.raises(sut.SupportRuntimeTransitionError, match="unbounded"):
        sut.prove_no_launch_v2(
            request=_no_launch_request(),
            runtime_identity=_runtime_identity(),
            services=services,
        )
