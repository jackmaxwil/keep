from __future__ import annotations

from io import BytesIO

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.dynamodb import decode_item, encode_item
from glm52_enforcement.task12_postpublication import (
    _EDGES,
    _OPERATIONS,
    ACCOUNT_ID,
    PARTITION_KEY,
    Task12OperationSourceCoordinate,
    Task12PostpublicationError,
    Task12PostpublicationInputs,
    Task12PostpublicationServices,
    build_task12_postpublication_iam_policy,
    materialize_task12_postpublication,
)


def _metadata(request_id: str) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RequestId": request_id,
        "RetryAttempts": 0,
    }


def _inputs() -> Task12PostpublicationInputs:
    activation = "h1g-act-20260729-0001"
    source_body = _static_source_body(activation_id=activation)
    source = {
        **source_body,
        "canonical_body_sha256": canonical_sha256(source_body),
    }
    from glm52_enforcement.canonical import canonical_json_bytes

    source_raw = canonical_json_bytes(source)
    import hashlib

    return Task12PostpublicationInputs(
        retained_stack_id=(
            "arn:aws:cloudformation:us-west-2:246813579024:stack/"
            "keep-glm52-h1g-retained/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        ),
        support_stack_id=(
            "arn:aws:cloudformation:us-west-2:246813579024:stack/"
            "keep-glm52-h1g-support/bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
        ),
        activation_id=activation,
        activation_ordinal=1,
        generation=1,
        dispatch_identity_sha256="a" * 64,
        ledger_table_name="keep-glm52-h1g-ledger-v1",
        authority_bucket="keep-glm52-models",
        campaign_bucket="keep-glm52-models",
        kms_key_id=(
            "arn:aws:kms:us-west-2:246813579024:key/"
            "12345678-1234-4234-8234-1234567890ab"
        ),
        worker_drain_document_name="KeepGlm52GracefulStopV1",
        worker_drain_document_version="7",
        operation_source=Task12OperationSourceCoordinate(
            bucket="keep-glm52-models",
            key=(
                "campaigns/glm52-sky-20260724/task12/sources/"
                + activation
                + "/00000001/operation-source.json"
            ),
            version_id="source-version-1",
            file_sha256=hashlib.sha256(source_raw).hexdigest(),
        ),
        campaign_descriptor=Task12OperationSourceCoordinate(
            bucket="keep-glm52-models",
            key=(
                "campaigns/glm52-sky-20260724/submissions/production/"
                "descriptor.json"
            ),
            version_id="descriptor-version-1",
            file_sha256="d" * 64,
        ),
        gpu_spend_approval=Task12OperationSourceCoordinate(
            bucket="keep-glm52-models",
            key=(
                "campaigns/glm52-sky-20260724/authorities/"
                "GPU_SPEND_APPROVAL.json"
            ),
            version_id="approval-version-1",
            file_sha256="e" * 64,
        ),
    )


def _static_source_body(
    *,
    activation_id: str,
    activation_ordinal: int = 1,
    generation: int = 1,
) -> dict[str, object]:
    from dataclasses import asdict

    from glm52_enforcement.task11_boundary import (
        build_task11_input_coordinate,
    )

    task9_coordinate = asdict(
        build_task11_input_coordinate(
            input_kind="TASK9_DEPLOYED_IDENTITY_COORDINATE",
            bucket="keep-glm52-models",
            key=(
                "campaigns/glm52-sky-20260724/authorities/task9/"
                + activation_id
                + "/TASK9_DEPLOYED_IDENTITY.json"
            ),
            version_id="task9-deployed-version-1",
            file_sha256="6" * 64,
            body_sha256="7" * 64,
        )
    )
    return {
        "schema_version": 1,
        "record_type": "glm52_task12_operation_static_authority_v1",
        "account_id": ACCOUNT_ID,
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "activation_id": activation_id,
        "activation_ordinal": activation_ordinal,
        "generation": generation,
        "generation_text": f"{generation:08d}",
        "dispatch_identity_sha256": "a" * 64,
        "ledger_table_name": "keep-glm52-h1g-ledger-v1",
        "ledger_partition_key": PARTITION_KEY,
        "authority_bucket": "keep-glm52-models",
        "campaign_bucket": "keep-glm52-models",
        "kms_key_id": (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "12345678-1234-4234-8234-1234567890ab"
        ),
        "worker_drain_document_name": "KeepGlm52GracefulStopV1",
        "worker_drain_document_version": "7",
        "task9_deployed_identity_coordinate": task9_coordinate,
        "task9_deployed_identity_sha256": "7" * 64,
        "campaign_descriptor_coordinate": {
            "bucket": "keep-glm52-models",
            "key": (
                "campaigns/glm52-sky-20260724/submissions/production/"
                "descriptor.json"
            ),
            "version_id": "descriptor-version-1",
            "file_sha256": "d" * 64,
        },
        "gpu_spend_approval_coordinate": {
            "bucket": "keep-glm52-models",
            "key": (
                "campaigns/glm52-sky-20260724/authorities/"
                "GPU_SPEND_APPROVAL.json"
            ),
            "version_id": "approval-version-1",
            "file_sha256": "e" * 64,
        },
        "task10_worker_descriptor_coordinate": {
            "bucket": "keep-glm52-models",
            "key": "task13/production/task10-worker-descriptor.json",
            "version_id": "task10-worker-version-1",
            "file_sha256": "8" * 64,
            "body_sha256": "9" * 64,
        },
        "worker_script_hashes": {
            "glm52_checkpoint_commit.py": "1" * 64,
            "glm52_drain_and_stop.py": "2" * 64,
            "glm52_deadline_guard.py": "3" * 64,
        },
        "terminal_evidence_prefix": (
            "campaigns/glm52-sky-20260724/submissions/production/"
            f"generations/{generation:08d}/terminal-evidence/"
        ),
        "spend_runtime_prefix": (
            "campaigns/glm52-sky-20260724/runtime"
        ),
        "descriptor_inventory": sorted(
            {
                operation
            for operation_names in _OPERATIONS.values()
            for operation in operation_names
            }
        ),
    }


def _source_raw(inputs: Task12PostpublicationInputs) -> bytes:
    from glm52_enforcement.canonical import canonical_json_bytes
    body = _static_source_body(
        activation_id=inputs.activation_id,
        activation_ordinal=inputs.activation_ordinal,
        generation=inputs.generation,
    )
    body.update(
        campaign_bucket=inputs.campaign_bucket,
        campaign_descriptor_coordinate={
            "bucket": inputs.campaign_descriptor.bucket,
            "key": inputs.campaign_descriptor.key,
            "version_id": inputs.campaign_descriptor.version_id,
            "file_sha256": inputs.campaign_descriptor.file_sha256,
        },
        gpu_spend_approval_coordinate={
            "bucket": inputs.gpu_spend_approval.bucket,
            "key": inputs.gpu_spend_approval.key,
            "version_id": inputs.gpu_spend_approval.version_id,
            "file_sha256": inputs.gpu_spend_approval.file_sha256,
        },
    )
    return canonical_json_bytes(
        {**body, "canonical_body_sha256": canonical_sha256(body)}
    )


class _CloudFormation:
    def __init__(self, inputs: Task12PostpublicationInputs) -> None:
        self.inputs = inputs

    def describe_stacks(self, **kwargs: object) -> dict[str, object]:
        stack_id = kwargs["StackName"]
        if stack_id == self.inputs.support_stack_id:
            return {
                "Stacks": [
                    {
                        "StackId": self.inputs.support_stack_id,
                        "StackStatus": "UPDATE_COMPLETE",
                    }
                ],
                "ResponseMetadata": _metadata("describe-support"),
            }
        assert stack_id == self.inputs.retained_stack_id
        return {
            "Stacks": [
                {
                    "StackId": self.inputs.retained_stack_id,
                    "StackStatus": "UPDATE_COMPLETE",
                    "Parameters": [
                        {"ParameterKey": "Task12ActivationOrdinal", "ParameterValue": "1"},
                        {"ParameterKey": "Task12Generation", "ParameterValue": "1"},
                        {"ParameterKey": "Task12GenerationText", "ParameterValue": "00000001"},
                        {
                            "ParameterKey": "Task12DispatchIdentitySha256",
                            "ParameterValue": "a" * 64,
                        },
                    ],
                }
            ],
            "ResponseMetadata": _metadata("describe"),
        }

    def list_stack_resources(self, **kwargs: object) -> dict[str, object]:
        stack_id = kwargs["StackName"]
        if stack_id == self.inputs.support_stack_id:
            return {
                "StackResourceSummaries": [
                    {
                        "LogicalResourceId": "SupportDeadlineVersion",
                        "PhysicalResourceId": (
                            "arn:aws:lambda:us-west-2:246813579024:function:"
                            "keep-glm52-h1g-support-deadline:23"
                        ),
                    },
                    {
                        "LogicalResourceId": "SupportStateMachineVersion",
                        "PhysicalResourceId": (
                            "arn:aws:states:us-west-2:246813579024:"
                            "stateMachine:keep-glm52-h1g-support:29"
                        ),
                    },
                ],
                "ResponseMetadata": _metadata("list-support"),
            }
        assert stack_id == self.inputs.retained_stack_id
        logicals = {
            edge[0].removesuffix("Retained").removesuffix("Snapshot")
            for edge in _EDGES
        }
        rows = [
            {
                "LogicalResourceId": logical + "Version",
                "PhysicalResourceId": (
                    "arn:aws:lambda:us-west-2:246813579024:function:"
                    "keep-glm52-h1g-"
                    + logical.lower()
                    + ":7"
                ),
            }
            for logical in sorted(logicals)
        ]
        rows.extend(
            [
                {
                    "LogicalResourceId": "NumericBindingVersion",
                    "PhysicalResourceId": (
                        "arn:aws:lambda:us-west-2:246813579024:function:"
                        "keep-glm52-h1g-numeric-binding:17"
                    ),
                },
                {
                    "LogicalResourceId": "RetainedCancellationVersion",
                    "PhysicalResourceId": (
                        "arn:aws:lambda:us-west-2:246813579024:function:"
                        "keep-glm52-h1g-retained-cancellation:19"
                    ),
                },
                {
                    "LogicalResourceId": "RetainedLifecycleStateMachineVersion",
                    "PhysicalResourceId": (
                        "arn:aws:states:us-west-2:246813579024:stateMachine:"
                        "keep-glm52-h1g-retainedlifecycle:11"
                    ),
                },
                {
                    "LogicalResourceId": "SnapshotCleanupStateMachineVersion",
                    "PhysicalResourceId": (
                        "arn:aws:states:us-west-2:246813579024:stateMachine:"
                        "keep-glm52-h1g-snapshotcleanup:13"
                    ),
                },
                {
                    "LogicalResourceId": "SnapshotCleanupScheduleInvokeRole",
                    "PhysicalResourceId": (
                        "keep-glm52-h1g-snapshot-cleanup-schedule-invoke"
                    ),
                },
            ]
        )
        return {
            "StackResourceSummaries": rows,
            "ResponseMetadata": _metadata("resources"),
        }


class _DynamoDB:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, object]] = {}
        self.lose_next_response = False

    def put_item(self, **kwargs: object) -> dict[str, object]:
        item = decode_item(kwargs["Item"])
        key = (item["PK"], item["SK"])
        if key in self.items:
            raise RuntimeError("ConditionalCheckFailedException")
        self.items[key] = item
        if self.lose_next_response:
            self.lose_next_response = False
            raise TimeoutError("lost success response")
        return {"ResponseMetadata": _metadata("put")}

    def get_item(self, **kwargs: object) -> dict[str, object]:
        key = decode_item(kwargs["Key"])
        item = self.items.get((key["PK"], key["SK"]))
        return {
            **({"Item": encode_item(item)} if item else {}),
            "ResponseMetadata": _metadata("get"),
        }


class _S3:
    def __init__(
        self, inputs: Task12PostpublicationInputs | None = None
    ) -> None:
        self.objects: dict[tuple[str, str], list[tuple[str, bytes]]] = {}
        self.lose_next_response = False
        typed = _inputs() if inputs is None else inputs
        source = typed.operation_source
        self.objects[(source.bucket, source.key)] = [
            (source.version_id, _source_raw(typed))
        ]

    def put_object(self, **kwargs: object) -> dict[str, object]:
        coordinate = (kwargs["Bucket"], kwargs["Key"])
        if coordinate in self.objects:
            raise RuntimeError("PreconditionFailed")
        version = "version-" + str(len(self.objects) + 1)
        self.objects[coordinate] = [(version, kwargs["Body"])]
        if self.lose_next_response:
            self.lose_next_response = False
            raise TimeoutError("lost success response")
        return {
            "VersionId": version,
            "ResponseMetadata": _metadata("s3-put"),
        }

    def list_object_versions(self, **kwargs: object) -> dict[str, object]:
        coordinate = (kwargs["Bucket"], kwargs["Prefix"])
        return {
            "Versions": [
                {"Key": coordinate[1], "VersionId": version}
                for version, _raw in self.objects.get(coordinate, [])
            ],
            "ResponseMetadata": _metadata("s3-list"),
        }

    def get_object(self, **kwargs: object) -> dict[str, object]:
        coordinate = (kwargs["Bucket"], kwargs["Key"])
        version, raw = next(
            row
            for row in self.objects[coordinate]
            if row[0] == kwargs["VersionId"]
        )
        import base64
        import hashlib

        return {
            "Body": BytesIO(raw),
            "VersionId": version,
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(raw).digest()
            ).decode("ascii"),
            "ResponseMetadata": _metadata("s3-get"),
        }


def test_materializer_writes_all_32_inputs_and_ten_exact_version_records() -> None:
    inputs = _inputs()
    dynamodb = _DynamoDB()
    s3 = _S3()
    evidence = materialize_task12_postpublication(
        inputs=inputs,
        services=Task12PostpublicationServices(
            cloudformation=_CloudFormation(inputs),
            dynamodb=dynamodb,
            s3=s3,
        ),
    )

    assert evidence["operation_record_count"] == 32
    assert evidence["deployment_record_count"] == 10
    assert evidence["authority_manifest_count"] == 10
    assert len(dynamodb.items) == 42
    assert len(s3.objects) == 11
    deployments = [
        item
        for item in dynamodb.items.values()
        if item.get("record_type") == "glm52_task12_lambda_deployment_v1"
    ]
    assert len(deployments) == 10
    assert all(item["PK"] == PARTITION_KEY for item in deployments)
    assert all(item["invoked_function_version_arn"].endswith(":7") for item in deployments)
    assert {
        item["caller_state_machine_version_arn"].rsplit(":", 1)[-1]
        for item in deployments
    } == {"11", "13", "29"}
    support_edge = next(
        item
        for item in deployments
        if item["mode"] == "SUPPORT_OBSERVER"
    )
    assert support_edge["invoked_function_version_arn"].endswith(
        "keep-glm52-h1g-terminalv2:7"
    )
    assert support_edge["role_coordinates"][
        "support_observer_version_arn"
    ].endswith("keep-glm52-h1g-support-deadline:23")
    descriptors = [
        item
        for item in dynamodb.items.values()
        if item.get("record_type", "").endswith("_descriptor_v1")
    ]
    assert len(descriptors) == 32
    assert all("builder_kind" in item for item in descriptors)
    assert all("allowed_read_apis" in item for item in descriptors)
    import json

    authenticated_static_identity = json.loads(_source_raw(inputs))[
        "canonical_body_sha256"
    ]
    assert {
        item["static_authority_sha256"] for item in descriptors
    } == {authenticated_static_identity}
    assert all(
        "source_sort_keys" not in item["source_coordinates"]
        and "expected_record_types" not in item["source_coordinates"]
        for item in descriptors
    )
    source_kinds = {
        source["coordinate_kind"]
        for item in descriptors
        for source in item["source_coordinates"]["sources"]
    }
    assert source_kinds == {
        "DDB_EXACT",
        "DDB_DERIVED",
        "DDB_KEY_FROM_CONTROL",
        "S3_VERSIONED_FROM_CONTROL",
    }
    for item in descriptors:
        operation_source_kinds = {
            source["coordinate_kind"]
            for source in item["source_coordinates"]["sources"]
        }
        allowed_reads = set(item["allowed_read_apis"])
        assert "dynamodb:GetItem" in allowed_reads
        if "S3_VERSIONED_FROM_CONTROL" in operation_source_kinds:
            assert "s3:GetObjectVersion" in allowed_reads
            for source in item["source_coordinates"]["sources"]:
                if source["coordinate_kind"] == "S3_VERSIONED_FROM_CONTROL":
                    assert source["control"]["record_type"] == (
                        "glm52_task12_versioned_writer_control_v1"
                    )
                    assert "#TASK12_VERSIONED_WRITER_CONTROL#" in source[
                        "control"
                    ]["sort_key"]
    assert all(
        not any(
            forbidden in item
            for forbidden in (
                "runtime_scan",
                "candidate",
                "record",
                "operation_payloads",
            )
        )
        for item in descriptors
    )


def test_materializer_reconciles_lost_responses_and_rejects_foreign_rows() -> None:
    inputs = _inputs()
    dynamodb = _DynamoDB()
    s3 = _S3()
    dynamodb.lose_next_response = True
    s3.lose_next_response = True
    services = Task12PostpublicationServices(
        cloudformation=_CloudFormation(inputs), dynamodb=dynamodb, s3=s3
    )
    first = materialize_task12_postpublication(inputs=inputs, services=services)
    assert "RECONCILED" in first["operation_outcomes"].values()
    assert "RECONCILED" in first["authority_outcomes"].values()
    second = materialize_task12_postpublication(inputs=inputs, services=services)
    assert set(second["operation_outcomes"].values()) == {"RECONCILED"}
    assert set(second["deployment_outcomes"].values()) == {"RECONCILED"}
    assert set(second["authority_outcomes"].values()) == {"RECONCILED"}

    deployment_key = next(
        key for key, value in dynamodb.items.items()
        if value.get("record_type") == "glm52_task12_lambda_deployment_v1"
    )
    dynamodb.items[deployment_key]["generation"] = 2
    with pytest.raises(RuntimeError, match="ConditionalCheckFailed"):
        materialize_task12_postpublication(inputs=inputs, services=services)


def test_all_writer_rows_roundtrip_through_actual_adapter_authority_and_state_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: writer rows are self-consistent but unusable by Lambda."""

    from glm52_enforcement import task12_lambda_adapters as adapters
    from glm52_enforcement.task12_operations import OPERATION_SPECS

    inputs = _inputs()
    dynamodb = _DynamoDB()
    s3 = _S3(inputs)
    materialize_task12_postpublication(
        inputs=inputs,
        services=Task12PostpublicationServices(
            cloudformation=_CloudFormation(inputs),
            dynamodb=dynamodb,
            s3=s3,
        ),
    )
    monkeypatch.setattr(
        adapters,
        "_zero_retry_client",
        lambda service, session=None: {
            "dynamodb": dynamodb,
            "s3": s3,
        }[service],
    )
    deployments = [
        dict(item)
        for item in dynamodb.items.values()
        if item.get("record_type") == "glm52_task12_lambda_deployment_v1"
    ]
    support_edges = [
        item for item in deployments if item["mode"] == "SUPPORT_OBSERVER"
    ]
    assert len(support_edges) == 1
    observed: set[str] = set()
    for stored in deployments:
        if stored["mode"] == "SUPPORT_OBSERVER":
            continue
        stored.pop("PK")
        stored.pop("SK")
        coordinates = dict(stored["role_coordinates"])
        authority_coordinate = coordinates.pop("authority")
        deployment = adapters.Task12LambdaDeployment(
            handler_kind=stored["handler_kind"],
            mode=stored["mode"],
            activation_id=stored["activation_id"],
            activation_ordinal=stored["activation_ordinal"],
            generation=stored["generation"],
            generation_text=stored["generation_text"],
            function_name=stored["function_name"],
            function_version=stored["function_version"],
            invoked_function_version_arn=stored[
                "invoked_function_version_arn"
            ],
            caller_state_machine_version_arn=stored[
                "caller_state_machine_version_arn"
            ],
            authority=adapters.VersionedAuthorityCoordinate(
                **authority_coordinate
            ),
            role_coordinates=coordinates,
            canonical_body_sha256=stored["canonical_body_sha256"],
        )
        authority_raw = s3.objects[
            (
                authority_coordinate["bucket"],
                authority_coordinate["key"],
            )
        ][0][1]
        import json

        authority_manifest = json.loads(authority_raw)
        ports = adapters._AwsPorts(deployment=deployment)
        for operation in authority_manifest["operations"]:
            invocation = adapters.Task12LambdaInvocation(
                handler_kind=deployment.handler_kind,
                mode=deployment.mode,
                activation_id=inputs.activation_id,
                activation_ordinal=inputs.activation_ordinal,
                generation=inputs.generation,
                generation_text=f"{inputs.generation:08d}",
                dispatch_identity_sha256=inputs.dispatch_identity_sha256,
                deployment_identity_sha256=deployment.canonical_body_sha256,
                invoked_function_version_arn=(
                    deployment.invoked_function_version_arn
                ),
                caller_state_machine_arn=(
                    deployment.caller_state_machine_version_arn.rsplit(
                        ":", 1
                    )[0]
                ),
                caller_state_machine_version_arn=(
                    deployment.caller_state_machine_version_arn
                ),
                state_machine_execution_arn=(
                    "arn:aws:states:us-west-2:246813579024:execution:"
                    "keep-glm52-h1g-task12:roundtrip"
                ),
                operation_kind=operation,
                operation_input={},
            )
            selected = ports._read_authority(invocation)
            row = ports._read_state(invocation, selected)
            assert row["operation_kind"] == operation
            observed.add(operation)
    assert observed == set(OPERATION_SPECS)


def test_support_postcreate_handler_and_contract_expose_the_real_write_boundary() -> None:
    from pathlib import Path

    from glm52_enforcement.support_plane import (
        build_support_contract_artifacts,
        coordinate_support_task12_postpublication,
    )

    source = Path(
        "aws/glm52-gpu/scripts/materialize_h1g_support_plane.py"
    ).read_text()
    assert "--publish-task12-postpublication" in source
    assert "coordinate_support_task12_postpublication(" in source
    assert callable(coordinate_support_task12_postpublication)

    contract = build_support_contract_artifacts()[
        "support-task12-runtime-v1.json"
    ]
    materializer = contract["postpublication_materializer"]
    assert materializer["operation_input_records"] == 32
    assert materializer["versioned_authority_manifests"] == 10
    assert materializer["deployment_records"] == 10
    assert materializer["operation_record_kind"] == (
        "IMMUTABLE_OPERATION_DESCRIPTOR"
    )
    assert materializer["runtime_observations_pre_materialized"] is False
    assert materializer["descriptor_source_graph"] == (
        "CANONICAL_RETAINED_DDB_AND_VERSIONED_S3_AUTHORITIES"
    )
    assert materializer["iam_actions"] == [
        "cloudformation:DescribeStacks",
        "cloudformation:ListStackResources",
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:ListBucketVersions",
        "s3:PutObject",
    ]

    policy = build_task12_postpublication_iam_policy(_inputs())
    assert [statement["Sid"] for statement in policy["Statement"]] == [
        "ReadExactPublishedVersions",
        "MaterializeExactTask12Rows",
        "ReadExactVersionedOperationSource",
        "ListOnlyTask12AuthorityVersions",
        "WriteAndReconcileOnlyTask12Authority",
    ]
    encoded = str(policy)
    assert "'Resource': '*'" not in encoded
    assert _inputs().retained_stack_id in encoded
    assert _inputs().support_stack_id in encoded
    assert "RUN#glm52-sky-20260724" in encoded
    assert (
        "campaigns/glm52-sky-20260724/task12/deployment/"
        "h1g-act-20260729-0001/00000001/"
    ) in encoded


def test_materializer_inventory_equals_every_executable_lambda_state() -> None:
    """Break caught: a named ASL state has no producible operation input."""

    import json

    from glm52_enforcement.task12_support_plane import (
        Task12SupportPlaneInputs,
        render_task12_support_plane_fragment,
    )

    fragment = render_task12_support_plane_fragment(
        inputs=Task12SupportPlaneInputs(
            activation_id=_inputs().activation_id,
            lambda_code_sha256="b" * 64,
            lambda_code_bucket="keep-glm52-runtime-artifacts",
            lambda_code_key="releases/task12-runtime.zip",
            lambda_code_version="version-1",
            ledger_table_arn=(
                "arn:aws:dynamodb:us-west-2:246813579024:table/"
                + _inputs().ledger_table_name
            ),
            retained_kms_key_arn=_inputs().kms_key_id,
            model_bucket_arn="arn:aws:s3:::" + _inputs().campaign_bucket,
            worker_drain_document_arn=(
                "arn:aws:ssm:us-west-2:246813579024:document/"
                + _inputs().worker_drain_document_name
            ),
        )
    )
    rendered_operations: set[str] = set()
    for workflow in ("RetainedLifecycle", "SnapshotCleanup"):
        definition = json.loads(
            fragment["Resources"][workflow + "StateMachine"]["Properties"][
                "DefinitionString"
            ]
        )
        rendered_operations.update(
            state["Parameters"]["operation_kind"]
            for state in definition["States"].values()
            if state.get("Type") == "Task"
            and isinstance(state.get("Resource"), str)
            and state["Resource"].startswith("${")
        )
    materialized_operations = {
        operation
        for operation_names in _OPERATIONS.values()
        for operation in operation_names
    }
    assert rendered_operations == materialized_operations
    assert len(rendered_operations) == 32


def test_production_request_uses_one_versioned_source_not_opaque_operation_rows() -> None:
    """Break caught: the CLI caller directly supplies 32 prehashed fixture rows."""

    from dataclasses import fields

    input_fields = {field.name for field in fields(Task12PostpublicationInputs)}
    assert "operation_inputs" not in input_fields
    assert "operation_source" in input_fields


def test_descriptor_graph_has_no_orphan_consumer_after_initial_trigger() -> None:
    """Break caught: a later operation consumes only preexisting authorities."""

    from glm52_enforcement.records import RECORD_FIELDS
    from glm52_enforcement.task12_operations import (
        _CANONICAL_SOURCE_FAMILIES,
        EXTERNAL_PRODUCER,
        OPERATION_PREDECESSORS,
        OPERATION_SOURCE_PRODUCERS,
        OPERATION_SPECS,
        RETAINED_OPERATION_SEQUENCE,
        SNAPSHOT_OPERATION_SEQUENCE,
        validate_operation_source_producers,
    )

    validate_operation_source_producers()
    assert set(OPERATION_SOURCE_PRODUCERS) == set(OPERATION_SPECS)
    sequence = RETAINED_OPERATION_SEQUENCE + SNAPSHOT_OPERATION_SEQUENCE
    assert all(
        any(
            producer != EXTERNAL_PRODUCER
            for _alias, producer in OPERATION_SOURCE_PRODUCERS[operation]
        )
        for operation in sequence[1:]
    )
    assert all(
        family.get("control_field")
        in RECORD_FIELDS.get(family.get("control_type"), ())
        for family in _CANONICAL_SOURCE_FAMILIES.values()
        if "control_field" in family
    )
    combined = RETAINED_OPERATION_SEQUENCE + SNAPSHOT_OPERATION_SEQUENCE
    assert len(RETAINED_OPERATION_SEQUENCE) == 22
    assert len(SNAPSHOT_OPERATION_SEQUENCE) == 10
    assert len(combined) == len(set(combined)) == 32
    assert all(
        OPERATION_PREDECESSORS[operation] != operation
        for operation in combined
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("ledger_table_name", "foreign-table"),
        ("ledger_partition_key", "RUN#foreign"),
        ("dispatch_identity_sha256", "b" * 64),
        ("descriptor_inventory", ["RECONCILE_RETAINED_LIFECYCLE_TRIGGER"]),
    ],
)
def test_versioned_source_rejects_foreign_static_coordinates(
    field: str,
    replacement: object,
) -> None:
    import hashlib
    import json
    from dataclasses import replace

    from glm52_enforcement.canonical import canonical_json_bytes

    inputs = _inputs()
    value = json.loads(_source_raw(inputs))
    value[field] = replacement
    body = dict(value)
    body.pop("canonical_body_sha256")
    raw = canonical_json_bytes(
        {**body, "canonical_body_sha256": canonical_sha256(body)}
    )
    updated = replace(
        inputs,
        operation_source=replace(
            inputs.operation_source,
            file_sha256=hashlib.sha256(raw).hexdigest(),
        ),
    )
    s3 = _S3(updated)
    coordinate = updated.operation_source
    s3.objects[(coordinate.bucket, coordinate.key)] = [
        (coordinate.version_id, raw)
    ]

    with pytest.raises(
        Task12PostpublicationError,
        match="foreign or incomplete",
    ):
        materialize_task12_postpublication(
            inputs=updated,
            services=Task12PostpublicationServices(
                cloudformation=_CloudFormation(updated),
                dynamodb=_DynamoDB(),
                s3=s3,
            ),
        )


def test_versioned_source_rejects_foreign_activation_even_when_rehashed() -> None:
    import hashlib
    import json
    from dataclasses import replace

    from glm52_enforcement.canonical import canonical_json_bytes

    inputs = _inputs()
    value = json.loads(_source_raw(inputs))
    value["activation_id"] = "foreign-activation"
    body = dict(value)
    body.pop("canonical_body_sha256")
    raw = canonical_json_bytes(
        {**body, "canonical_body_sha256": canonical_sha256(body)}
    )
    updated = replace(
        inputs,
        operation_source=replace(
            inputs.operation_source,
            file_sha256=hashlib.sha256(raw).hexdigest(),
        ),
    )
    s3 = _S3(updated)
    coordinate = updated.operation_source
    s3.objects[(coordinate.bucket, coordinate.key)] = [
        (coordinate.version_id, raw)
    ]
    with pytest.raises(
        Task12PostpublicationError,
        match="foreign or incomplete",
    ):
        materialize_task12_postpublication(
            inputs=updated,
            services=Task12PostpublicationServices(
                cloudformation=_CloudFormation(updated),
                dynamodb=_DynamoDB(),
                s3=s3,
            ),
        )


def test_versioned_source_rejects_future_runtime_payloads() -> None:
    """Break caught: postpublication accepts observations before they exist."""

    import hashlib
    import json
    from dataclasses import replace

    from glm52_enforcement.canonical import canonical_json_bytes

    inputs = _inputs()
    value = json.loads(_source_raw(inputs))
    value["operation_payloads"] = {
        "RECONCILE_RETAINED_LIFECYCLE_TRIGGER": {
            "runtime_scan": {"fixture": True}
        }
    }
    body = dict(value)
    body.pop("canonical_body_sha256")
    raw = canonical_json_bytes(
        {**body, "canonical_body_sha256": canonical_sha256(body)}
    )
    inputs = replace(
        inputs,
        operation_source=replace(
            inputs.operation_source,
            file_sha256=hashlib.sha256(raw).hexdigest(),
        ),
    )
    s3 = _S3(inputs)
    coordinate = inputs.operation_source
    s3.objects[(coordinate.bucket, coordinate.key)] = [
        (coordinate.version_id, raw)
    ]
    with pytest.raises(
        Task12PostpublicationError,
        match="foreign or incomplete",
    ):
        materialize_task12_postpublication(
            inputs=inputs,
            services=Task12PostpublicationServices(
                cloudformation=_CloudFormation(inputs),
                dynamodb=_DynamoDB(),
                s3=s3,
            ),
        )


def test_versioned_source_rejects_non_sha_static_identity() -> None:
    """Break caught: static authority accepts a non-content identity."""

    import hashlib
    import json
    from dataclasses import replace

    from glm52_enforcement.canonical import canonical_json_bytes

    inputs = _inputs()
    value = json.loads(_source_raw(inputs))
    value["dispatch_identity_sha256"] = "fixture"
    body = dict(value)
    body.pop("canonical_body_sha256")
    raw = canonical_json_bytes(
        {**body, "canonical_body_sha256": canonical_sha256(body)}
    )
    inputs = replace(
        inputs,
        operation_source=replace(
            inputs.operation_source,
            file_sha256=hashlib.sha256(raw).hexdigest(),
        ),
    )
    s3 = _S3(inputs)
    coordinate = inputs.operation_source
    s3.objects[(coordinate.bucket, coordinate.key)] = [
        (coordinate.version_id, raw)
    ]
    with pytest.raises(
        Task12PostpublicationError,
        match="foreign or incomplete",
    ):
        materialize_task12_postpublication(
            inputs=inputs,
            services=Task12PostpublicationServices(
                cloudformation=_CloudFormation(inputs),
                dynamodb=_DynamoDB(),
                s3=s3,
            ),
        )


def test_real_postcreate_cli_mode_calls_the_typed_postpublication_coordinator(
    tmp_path: object,
) -> None:
    import importlib.util
    import json
    from pathlib import Path

    from glm52_enforcement.canonical import canonical_json_bytes

    script = Path("aws/glm52-gpu/scripts/materialize_h1g_support_plane.py")
    spec = importlib.util.spec_from_file_location("task12_postpub_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    typed = _inputs()
    request_body = {
        "schema_version": 1,
        "record_type": "glm52_task12_postpublication_request_v1",
        "retained_stack_id": typed.retained_stack_id,
        "support_stack_id": typed.support_stack_id,
        "activation_id": typed.activation_id,
        "activation_ordinal": typed.activation_ordinal,
        "generation": typed.generation,
        "dispatch_identity_sha256": typed.dispatch_identity_sha256,
        "ledger_table_name": typed.ledger_table_name,
        "authority_bucket": typed.authority_bucket,
        "campaign_bucket": typed.campaign_bucket,
        "kms_key_id": typed.kms_key_id,
        "worker_drain_document_name": typed.worker_drain_document_name,
        "worker_drain_document_version": typed.worker_drain_document_version,
        "operation_source": {
            "bucket": typed.operation_source.bucket,
            "key": typed.operation_source.key,
            "version_id": typed.operation_source.version_id,
            "file_sha256": typed.operation_source.file_sha256,
        },
        "campaign_descriptor": {
            "bucket": typed.campaign_descriptor.bucket,
            "key": typed.campaign_descriptor.key,
            "version_id": typed.campaign_descriptor.version_id,
            "file_sha256": typed.campaign_descriptor.file_sha256,
        },
        "gpu_spend_approval": {
            "bucket": typed.gpu_spend_approval.bucket,
            "key": typed.gpu_spend_approval.key,
            "version_id": typed.gpu_spend_approval.version_id,
            "file_sha256": typed.gpu_spend_approval.file_sha256,
        },
    }
    request = {
        **request_body,
        "canonical_identity_sha256": canonical_sha256(request_body),
    }
    root = Path(str(tmp_path))
    request_path = root / "request.json"
    output_path = root / "evidence.json"
    request_path.write_bytes(canonical_json_bytes(request) + b"\n")
    dynamodb = _DynamoDB()
    s3 = _S3()

    class Runner:
        def __init__(self, *, profile: str, region: str) -> None:
            assert profile == "keep-gpu"
            assert region == "us-west-2"
            self.cloudformation = _CloudFormation(typed)
            self.dynamodb = dynamodb
            self.s3 = s3

        def get_caller_identity(self) -> dict[str, object]:
            return {
                "Account": ACCOUNT_ID,
                "Arn": (
                    f"arn:aws:sts::{ACCOUNT_ID}:"
                    "assumed-role/task12-postpublication/test"
                ),
                "UserId": "task12:test",
                "ResponseMetadata": _metadata("sts"),
            }

    assert module.run(
        [
            "--profile",
            "keep-gpu",
            "--region",
            "us-west-2",
            "--publish-task12-postpublication",
            "--task12-postpublication-request",
            str(request_path),
            "--task12-postpublication-output",
            str(output_path),
        ],
        runner_factory=Runner,
    ) == 0
    evidence = json.loads(output_path.read_text())
    assert evidence["deployment_record_count"] == 10
    assert evidence["operation_record_count"] == 32
    assert len(dynamodb.items) == 42
