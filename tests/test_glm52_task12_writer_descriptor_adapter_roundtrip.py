from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256

PARTITION_KEY = "RUN#glm52-sky-20260724"
ACTIVATION_ID = "activation-1"
GENERATION_TEXT = "00000001"


def _load_test_module(file_name: str) -> object:
    spec = importlib.util.spec_from_file_location(
        "_task12_writer_descriptor_" + file_name.removesuffix(".py"),
        Path(__file__).with_name(file_name),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


POSTPUBLICATION = _load_test_module("test_glm52_task12_postpublication.py")
FACTORIES = _load_test_module("test_glm52_task12_lambda_adapter_factories.py")
RETAINED_STATE = _load_test_module("test_glm52_task12_retained_state.py")


class _Body:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def read(self) -> bytes:
        return self._raw


class _ExactS3(POSTPUBLICATION._S3):
    """Return an authenticated mismatched version for a foreign version read."""

    def get_object(self, **request: object) -> dict[str, object]:
        try:
            return super().get_object(**request)
        except StopIteration:
            coordinate = (request["Bucket"], request["Key"])
            actual_version, raw = self.objects[coordinate][0]
            return {
                "Body": _Body(raw),
                "VersionId": actual_version,
                "ChecksumSHA256": base64.b64encode(
                    hashlib.sha256(raw).digest()
                ).decode("ascii"),
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "s3-version-mismatch",
                    "RetryAttempts": 0,
                },
            }


class _SharedSession:
    def __init__(
        self,
        *,
        dynamodb: object,
        s3: object,
        cloudformation: object,
    ) -> None:
        self._clients = {
            "cloudformation": cloudformation,
            "dynamodb": dynamodb,
            "s3": s3,
        }

    def client(
        self,
        service: str,
        *,
        region_name: str,
        config: object,
    ) -> object:
        assert region_name == "us-west-2"
        assert config.retries == {
            "mode": "standard",
            "total_max_attempts": 1,
        }
        return self._clients[service]


def _postpublication_inputs() -> tuple[object, bytes]:
    base = POSTPUBLICATION._inputs()
    source_body = POSTPUBLICATION._static_source_body(
        activation_id=ACTIVATION_ID
    )
    task9_coordinate_body = {
        "input_kind": "TASK9_DEPLOYED_IDENTITY_COORDINATE",
        "bucket": "keep-glm52-campaign",
        "key": (
            "campaigns/glm52-sky-20260724/authorities/task9/"
            "activation-1/TASK9_DEPLOYED_IDENTITY.json"
        ),
        "version_id": "task9-version-1",
        "file_sha256": "9" * 64,
        "body_sha256": "8" * 64,
    }
    source_body.update(
        authority_bucket="keep-glm52-retained",
        campaign_bucket="keep-glm52-campaign",
        ledger_table_name="keep-glm52-ledger",
        task9_deployed_identity_coordinate={
            **task9_coordinate_body,
            "canonical_identity_sha256": canonical_sha256(
                task9_coordinate_body
            ),
        },
        task9_deployed_identity_sha256="8" * 64,
        campaign_descriptor_coordinate={
            "bucket": "keep-glm52-campaign",
            "key": (
                "campaigns/glm52-sky-20260724/submissions/production/"
                "descriptor.json"
            ),
            "version_id": "descriptor-version-1",
            "file_sha256": "d" * 64,
        },
        gpu_spend_approval_coordinate={
            "bucket": "keep-glm52-campaign",
            "key": (
                "campaigns/glm52-sky-20260724/authorities/"
                "GPU_SPEND_APPROVAL.json"
            ),
            "version_id": "approval-version-1",
            "file_sha256": "e" * 64,
        },
        task10_worker_descriptor_coordinate={
            "bucket": "keep-glm52-campaign",
            "key": "task13/production/task10-worker-descriptor.json",
            "version_id": "task10-worker-version-1",
            "file_sha256": "8" * 64,
            "body_sha256": "9" * 64,
        },
    )
    source_raw = canonical_json_bytes(
        {
            **source_body,
            "canonical_body_sha256": canonical_sha256(source_body),
        }
    )
    source = replace(
        base.operation_source,
        bucket="keep-glm52-retained",
        key="task12/activation-1/operation-source.json",
        file_sha256=hashlib.sha256(source_raw).hexdigest(),
    )
    return (
        replace(
            base,
            activation_id=ACTIVATION_ID,
            ledger_table_name="keep-glm52-ledger",
            authority_bucket="keep-glm52-retained",
            campaign_bucket="keep-glm52-campaign",
            campaign_descriptor=POSTPUBLICATION.Task12OperationSourceCoordinate(
                bucket="keep-glm52-campaign",
                key=(
                    "campaigns/glm52-sky-20260724/submissions/production/"
                    "descriptor.json"
                ),
                version_id="descriptor-version-1",
                file_sha256="d" * 64,
            ),
            gpu_spend_approval=POSTPUBLICATION.Task12OperationSourceCoordinate(
                bucket="keep-glm52-campaign",
                key=(
                    "campaigns/glm52-sky-20260724/authorities/"
                    "GPU_SPEND_APPROVAL.json"
                ),
                version_id="approval-version-1",
                file_sha256="e" * 64,
            ),
            operation_source=source,
        ),
        source_raw,
    )


def _materialize_shared_state() -> tuple[object, object, object, object]:
    inputs, source_raw = _postpublication_inputs()
    dynamodb = POSTPUBLICATION._DynamoDB()
    s3 = _ExactS3(inputs)
    source = inputs.operation_source
    s3.objects[(source.bucket, source.key)] = [
        (source.version_id, source_raw)
    ]
    cloudformation = POSTPUBLICATION._CloudFormation(inputs)
    evidence = POSTPUBLICATION.materialize_task12_postpublication(
        inputs=inputs,
        services=POSTPUBLICATION.Task12PostpublicationServices(
            cloudformation=cloudformation,
            dynamodb=dynamodb,
            s3=s3,
        ),
    )
    assert evidence["operation_record_count"] == 32
    assert evidence["authority_manifest_count"] == 10
    assert evidence["deployment_record_count"] == 10
    return inputs, dynamodb, s3, cloudformation


def _writer_case(kind: str) -> tuple[object, ...]:
    return next(case for case in FACTORIES.CASES if case[0] == kind)


def _execute_writer(
    *,
    kind: str,
    dynamodb: object,
    s3: object,
    cloudformation: object,
) -> None:
    from glm52_enforcement import task12_lambda_adapters as adapters

    (
        _kind,
        mode,
        function_name,
        coordinate_names,
        factory_name,
        state_record_type,
    ) = _writer_case(kind)
    harness = FACTORIES._AwsHarness(
        kind=kind,
        state_record_type=state_record_type,
        state_payload=FACTORIES._INPUT_BUILDERS[kind](),
        mode=mode,
        function_name=function_name,
        coordinate_names=coordinate_names,
    )
    authority = harness.deployment.authority
    s3.objects[(authority.bucket, authority.key)] = [
        (authority.version_id, harness.authority_raw)
    ]
    selected = harness.authority["operations"][harness.operation_kind]
    dynamodb.items[(PARTITION_KEY, selected["state_sort_key"])] = {
        "PK": PARTITION_KEY,
        "SK": selected["state_sort_key"],
        **harness.state,
    }
    coordinator = getattr(adapters, factory_name)(
        harness.deployment,
        _session=_SharedSession(
            dynamodb=dynamodb,
            s3=s3,
            cloudformation=cloudformation,
        ),
    )
    assert coordinator(harness.invocation) is not None


def _control_key(writer_slug: str) -> tuple[str, str]:
    return (
        PARTITION_KEY,
        "ACTIVATION#activation-1#TASK12_VERSIONED_WRITER_CONTROL#"
        "00000001#"
        + writer_slug,
    )


def _assert_exact_writer_control(
    *,
    dynamodb: object,
    s3: object,
    writer_kind: str,
    writer_slug: str,
    object_key: str,
) -> Mapping[str, object]:
    stored = dynamodb.items[_control_key(writer_slug)]
    assert stored["PK"] == PARTITION_KEY
    assert stored["SK"] == _control_key(writer_slug)[1]
    control = {
        key: value
        for key, value in stored.items()
        if key not in {"PK", "SK"}
    }
    assert control["record_type"] == (
        "glm52_task12_versioned_writer_control_v1"
    )
    assert control["writer_kind"] == writer_kind
    assert control["campaign_bucket"] == "keep-glm52-campaign"
    assert control["object_key"] == object_key

    versions = s3.objects[
        (control["campaign_bucket"], control["object_key"])
    ]
    assert len(versions) == 1
    version_id, raw = versions[0]
    assert control["object_version_id"] == version_id
    assert control["file_sha256"] == hashlib.sha256(raw).hexdigest()
    body = json.loads(raw)
    assert control["body_sha256"] == body["canonical_body_sha256"]
    return control


def _deployment(
    *,
    dynamodb: object,
    handler_kind: str,
) -> object:
    from glm52_enforcement import task12_lambda_adapters as adapters

    stored = next(
        dict(item)
        for item in dynamodb.items.values()
        if item.get("record_type") == "glm52_task12_lambda_deployment_v1"
        and item.get("handler_kind") == handler_kind
    )
    stored.pop("PK")
    stored.pop("SK")
    coordinates = dict(stored["role_coordinates"])
    authority = coordinates.pop("authority")
    return adapters.Task12LambdaDeployment(
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
        authority=adapters.VersionedAuthorityCoordinate(**authority),
        role_coordinates=coordinates,
        canonical_body_sha256=stored["canonical_body_sha256"],
    )


def _invocation(*, deployment: object, operation_kind: str) -> object:
    from glm52_enforcement import task12_lambda_adapters as adapters
    from glm52_enforcement.task12_nonce_capsule import (
        generate_owner_nonce_capsule,
    )

    caller_version = deployment.caller_state_machine_version_arn
    operation_input = FACTORIES._operation_input(operation_kind)
    if operation_kind == "RETAINED_ENTER_RECOVERY_COMPLETE":
        class _Kms:
            def generate_data_key(self, **request: object) -> object:
                return {
                    "KeyId": request["KeyId"],
                    "Plaintext": b"r" * 32,
                    "CiphertextBlob": b"roundtrip-kms-ciphertext",
                    "ResponseMetadata": {
                        "HTTPStatusCode": 200,
                        "RequestId": "roundtrip-generate-data-key",
                        "RetryAttempts": 0,
                    },
                }

        capsule, _ = generate_owner_nonce_capsule(
            ports=SimpleNamespace(
                deployment=deployment,
                client=lambda service: _Kms(),
            ),
            authority={
                "account_id": "246813579024",
                "region": "us-west-2",
                "run_id": "glm52-sky-20260724",
                "activation_id": ACTIVATION_ID,
                "authority_domain": "RECOVERY",
                "owner_execution_arn": (
                    "arn:aws:states:us-west-2:246813579024:execution:"
                    "keep-glm52-h1g-retainedlifecycle:"
                    "writer-descriptor-roundtrip"
                ),
                "owner_state_machine_version_arn": caller_version,
                "owner_attempt": 1,
                "barrier_nonce_sha256": "b" * 64,
                "control_revision": 4,
                "owner_hard_expires_at": "2099-07-29T13:00:00Z",
            },
            now=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
        )
        operation_input = {
            **operation_input,
            "task12_last_result": {
                "result": {"owner_nonce_capsule": capsule}
            },
        }
    return adapters.Task12LambdaInvocation(
        handler_kind=deployment.handler_kind,
        mode=deployment.mode,
        activation_id=ACTIVATION_ID,
        activation_ordinal=1,
        generation=1,
        generation_text=GENERATION_TEXT,
        dispatch_identity_sha256="a" * 64,
        deployment_identity_sha256=deployment.canonical_body_sha256,
        invoked_function_version_arn=(
            deployment.invoked_function_version_arn
        ),
        caller_state_machine_arn=caller_version.rsplit(":", 1)[0],
        caller_state_machine_version_arn=caller_version,
        state_machine_execution_arn=(
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retainedlifecycle:writer-descriptor-roundtrip"
        ),
        operation_kind=operation_kind,
        operation_input=operation_input,
    )


def _descriptor_context(
    *,
    dynamodb: object,
    s3: object,
    cloudformation: object,
    handler_kind: str,
    operation_kind: str,
) -> tuple[object, object, object]:
    from glm52_enforcement import task12_lambda_adapters as adapters

    deployment = _deployment(
        dynamodb=dynamodb,
        handler_kind=handler_kind,
    )
    invocation = _invocation(
        deployment=deployment,
        operation_kind=operation_kind,
    )
    ports = adapters._AwsPorts(
        deployment=deployment,
        session=_SharedSession(
            dynamodb=dynamodb,
            s3=s3,
            cloudformation=cloudformation,
        ),
    )
    selected = ports._read_authority(invocation)
    descriptor = ports._read_state(invocation, selected)
    assert descriptor["record_type"] == (
        "glm52_task12_" + operation_kind.lower() + "_descriptor_v1"
    )
    return ports, invocation, descriptor


def _seed_recovery_control(
    dynamodb: object, *, terminal_identity_sha256: str
) -> None:
    nonce_sha256 = hashlib.sha256(b"r" * 32).hexdigest()
    terminal_deployment = _deployment(
        dynamodb=dynamodb,
        handler_kind="RETAINED_TERMINAL_V2",
    )
    index = RETAINED_STATE._activation_index()
    control = RETAINED_STATE._control(
        phase="RECOVERY_SEALING",
        revision=4,
    )
    recovery = RETAINED_STATE._closed_record(
        "glm52_production_recovery_control",
        activation_id=ACTIVATION_ID,
        state="TERMINAL_V2_PUBLISHED",
        revision=4,
        terminal_v2_identity_sha256=terminal_identity_sha256,
        owner_attempt=1,
        owner_execution_arn=(
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-retainedlifecycle:"
            "writer-descriptor-roundtrip"
        ),
        owner_state_machine_version_arn=(
            terminal_deployment.caller_state_machine_version_arn
        ),
        owner_dispatch_identity_sha256="d" * 64,
        owner_invocation_nonce_sha256=nonce_sha256,
        owner_hard_expires_at="2099-07-29T13:00:00Z",
        recovery_barrier_nonce_sha256="b" * 64,
        support_control_revision_at_seal=4,
    )
    finalization = RETAINED_STATE._closed_record(
        "glm52_production_finalization_control",
        activation_id=ACTIVATION_ID,
    )
    records = {
        "ACTIVATION_INDEX": index,
        "ACTIVATION#activation-1#CONTROL": control,
        "ACTIVATION#activation-1#RECOVERY_CONTROL": recovery,
        "ACTIVATION#activation-1#FINALIZATION_CONTROL": finalization,
    }
    for sort_key, record in records.items():
        dynamodb.items[(PARTITION_KEY, sort_key)] = {
            "PK": PARTITION_KEY,
            "SK": sort_key,
            **record,
        }


def _rehash_control(
    *,
    dynamodb: object,
    writer_slug: str,
    field: str,
    value: object,
) -> None:
    key = _control_key(writer_slug)
    stored = dict(dynamodb.items[key])
    body = {
        name: item
        for name, item in stored.items()
        if name not in {"PK", "SK", "canonical_body_sha256"}
    }
    body[field] = value
    dynamodb.items[key] = {
        "PK": key[0],
        "SK": key[1],
        **body,
        "canonical_body_sha256": canonical_sha256(body),
    }


def test_real_writers_feed_materialized_descriptor_consumers_through_exact_versions(
) -> None:
    """Break caught: a writer control exists but its descriptor cannot consume it."""

    from glm52_enforcement import task12_lambda_adapters as adapters

    inputs, dynamodb, s3, cloudformation = _materialize_shared_state()
    assert inputs.activation_id == ACTIVATION_ID

    for kind in (
        "RETAINED_TERMINAL_V2",
        "RETAINED_FINALIZER",
        "RETAINED_H1G_DRAINED",
    ):
        _execute_writer(
            kind=kind,
            dynamodb=dynamodb,
            s3=s3,
            cloudformation=cloudformation,
        )

    terminal_control = _assert_exact_writer_control(
        dynamodb=dynamodb,
        s3=s3,
        writer_kind="TerminalV2",
        writer_slug="TERMINALV2",
        object_key=(
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/terminal/PRODUCTION_TERMINAL_V2.json"
        ),
    )
    finalized_control = _assert_exact_writer_control(
        dynamodb=dynamodb,
        s3=s3,
        writer_kind="SupportPlaneFinalized",
        writer_slug="SUPPORTPLANEFINALIZED",
        object_key=(
            "campaigns/glm52-sky-20260724/submissions/production/"
            "activations/activation-1/finalization/"
            "SUPPORT_PLANE_FINALIZED.json"
        ),
    )
    _assert_exact_writer_control(
        dynamodb=dynamodb,
        s3=s3,
        writer_kind="H1GDrained",
        writer_slug="H1GDRAINED",
        object_key=(
            "campaigns/glm52-sky-20260724/submissions/production/"
            "activations/activation-1/finalization/H1G_DRAINED.json"
        ),
    )

    _seed_recovery_control(
        dynamodb,
        terminal_identity_sha256=terminal_control["body_sha256"],
    )
    terminal_ports, terminal_invocation, terminal_descriptor = (
        _descriptor_context(
            dynamodb=dynamodb,
            s3=s3,
            cloudformation=cloudformation,
            handler_kind="RETAINED_TERMINAL_V2",
            operation_kind="RETAINED_ENTER_RECOVERY_COMPLETE",
        )
    )
    terminal_row, terminal_sources = adapters._read_live_operation_source(
        ports=terminal_ports,
        invocation=terminal_invocation,
        descriptor=terminal_descriptor,
    )
    assert terminal_sources["terminal_v2"]["canonical_body_sha256"] == (
        terminal_control["body_sha256"]
    )
    assert terminal_row["operation_kind"] == (
        "RETAINED_ENTER_RECOVERY_COMPLETE"
    )

    finalized_ports, finalized_invocation, finalized_descriptor = (
        _descriptor_context(
            dynamodb=dynamodb,
            s3=s3,
            cloudformation=cloudformation,
            handler_kind="RETAINED_FINALIZER",
            operation_kind="RETAINED_RECONCILE_DELETE_UNTIL_ABSENT",
        )
    )
    finalized_row, finalized_sources = adapters._read_live_operation_source(
        ports=finalized_ports,
        invocation=finalized_invocation,
        descriptor=finalized_descriptor,
    )
    assert finalized_sources["support_finalized"][
        "canonical_body_sha256"
    ] == finalized_control["body_sha256"]
    assert finalized_row["operation_kind"] == (
        "RETAINED_RECONCILE_DELETE_UNTIL_ABSENT"
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "object_version_id",
            "foreign-writer-version",
            "versioned S3 source read is unauthenticated",
        ),
        (
            "file_sha256",
            "0" * 64,
            "versioned S3 source read is unauthenticated",
        ),
        (
            "body_sha256",
            "0" * 64,
            "versioned S3 source body identity drifted",
        ),
    ],
)
def test_materialized_descriptor_rejects_mutated_writer_control_binding(
    field: str,
    replacement: object,
    message: str,
) -> None:
    """Break caught: an authenticated control can redirect a descriptor."""

    from glm52_enforcement import task12_lambda_adapters as adapters

    _inputs, dynamodb, s3, cloudformation = _materialize_shared_state()
    _execute_writer(
        kind="RETAINED_FINALIZER",
        dynamodb=dynamodb,
        s3=s3,
        cloudformation=cloudformation,
    )
    _rehash_control(
        dynamodb=dynamodb,
        writer_slug="SUPPORTPLANEFINALIZED",
        field=field,
        value=replacement,
    )
    ports, invocation, descriptor = _descriptor_context(
        dynamodb=dynamodb,
        s3=s3,
        cloudformation=cloudformation,
        handler_kind="RETAINED_FINALIZER",
        operation_kind="RETAINED_RECONCILE_DELETE_UNTIL_ABSENT",
    )

    with pytest.raises(adapters.Task12LambdaAdapterError, match=message):
        adapters._read_live_operation_source(
            ports=ports,
            invocation=invocation,
            descriptor=descriptor,
        )


def test_materialized_descriptor_rejects_mutated_writer_control_key() -> None:
    """Break caught: a descriptor points at a noncanonical writer-control key."""

    from glm52_enforcement import task12_lambda_adapters as adapters

    _inputs, dynamodb, s3, cloudformation = _materialize_shared_state()
    _execute_writer(
        kind="RETAINED_FINALIZER",
        dynamodb=dynamodb,
        s3=s3,
        cloudformation=cloudformation,
    )
    ports, invocation, descriptor = _descriptor_context(
        dynamodb=dynamodb,
        s3=s3,
        cloudformation=cloudformation,
        handler_kind="RETAINED_FINALIZER",
        operation_kind="RETAINED_RECONCILE_DELETE_UNTIL_ABSENT",
    )
    mutated = deepcopy(descriptor)
    source = mutated["source_coordinates"]["sources"][0]
    source["control"]["sort_key"] += "#FOREIGN"
    body = {
        key: value
        for key, value in mutated.items()
        if key != "canonical_body_sha256"
    }
    mutated["canonical_body_sha256"] = canonical_sha256(body)

    with pytest.raises(
        adapters.Task12LambdaAdapterError,
        match="live operation descriptor is invalid",
    ):
        adapters._read_live_operation_source(
            ports=ports,
            invocation=invocation,
            descriptor=mutated,
        )
