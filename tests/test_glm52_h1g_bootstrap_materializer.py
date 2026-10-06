from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
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
from glm52_enforcement.support_fence_handler import (
    BootstrapPublicationHandlerServices,
    SourceSettlementHandlerServices,
    bootstrap_publication_main,
    source_settlement_main,
)
from glm52_enforcement.support_plane import (
    RetainedFenceBootstrapInputs,
    build_retained_fence_bootstrap_fragment,
)
from glm52_enforcement.task13_fixed_artifacts import (
    CAMPAIGN_BUCKET,
    Task13FixedArtifactServices,
)

ACCOUNT_ID = "246813579024"
ACTIVATION_ID = "glm52-v2-amber-quartz"
BUCKET_ARN = "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2"
KMS_ARN = "arn:aws:kms:us-west-2:246813579024:key/01234567-89ab-cdef-0123-456789abcdef"
RETAINED_STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:stack/"
    "keep-glm52-gpu/11111111-1111-4111-8111-111111111111"
)
OPERATOR_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/aws-reserved/sso.amazonaws.com/"
    "us-west-2/AWSReservedSSO_AdministratorAccess_0123456789abcdef"
)
ROOT = Path(__file__).resolve().parents[1]


def _inputs() -> RetainedFenceBootstrapInputs:
    return RetainedFenceBootstrapInputs(
        schema_version=2,
        record_type="glm52_h1g_retained_fence_bootstrap_inputs_v2",
        activation_id=ACTIVATION_ID,
        retained_stack_id=RETAINED_STACK_ID,
        model_bucket_arn=BUCKET_ARN,
        retained_kms_key_arn=KMS_ARN,
        lambda_code_bucket="keep-glm52-models-246813579024-us-west-2",
        lambda_code_key="task13/artifacts/support-lambda/exact.zip",
        lambda_code_version_id="exact-version/+",
        lambda_code_sha256="a" * 64,
        stage_operator_role_arn=OPERATOR_ROLE_ARN,
    )


def _role_statement_actions(resource: dict[str, object]) -> set[str]:
    actions: set[str] = set()
    policies = resource["Properties"]["Policies"]  # type: ignore[index]
    for policy in policies:  # type: ignore[union-attr]
        for statement in policy["PolicyDocument"]["Statement"]:
            action = statement["Action"]
            actions.update(action if type(action) is list else [action])
    return actions


def test_retained_bootstrap_fragment_is_manifest_independent_and_least_privilege() -> (
    None
):
    fragment = build_retained_fence_bootstrap_fragment(_inputs())
    resources = fragment["Resources"]

    assert set(resources) == {
        "FenceBootstrapInvokerRole",
        "FenceBootstrapMaterializerLogGroup",
        "FenceBootstrapMaterializerRole",
        "FenceBootstrapMaterializerFunction",
        "FenceBootstrapMaterializerVersion",
        "FenceBootstrapMaterializerInvokePermission",
        "FenceBootstrapArtifactPublisherRole",
        "FenceBootstrapDeploymentRoleBootstrapAuthorityPolicy",
    }
    deployment_policy = resources[
        "FenceBootstrapDeploymentRoleBootstrapAuthorityPolicy"
    ]
    assert deployment_policy["Properties"] == {
        "PolicyName": "keep-glm52-h1g-retained-bootstrap-authority",
        "Roles": ["keep-glm52-h1g-cloudformation-deployment"],
        "PolicyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "ResolveExactGpuAmiParameter",
                    "Effect": "Allow",
                    "Action": "ssm:GetParameters",
                    "Resource": (
                        "arn:aws:ssm:us-west-2::parameter/aws/service/"
                        "deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-"
                        "ubuntu-22.04/latest/ami-id"
                    ),
                },
                {
                    "Sid": "PassExactBootstrapMaterializerRole",
                    "Effect": "Allow",
                    "Action": "iam:PassRole",
                    "Resource": (
                        "arn:aws:iam::246813579024:role/"
                        "keep-glm52-h1g-fence-bootstrap-materializer"
                    ),
                    "Condition": {
                        "StringEquals": {"iam:PassedToService": "lambda.amazonaws.com"}
                    },
                },
            ],
        },
    }
    function = resources["FenceBootstrapMaterializerFunction"]
    assert function["Properties"]["Handler"] == (
        "support_fence_handler.bootstrap_publication_main"
    )
    assert function["DependsOn"] == [
        "FenceBootstrapMaterializerLogGroup",
        "FenceBootstrapArtifactPublisherRole",
        "FenceBootstrapDeploymentRoleBootstrapAuthorityPolicy",
    ]
    version = resources["FenceBootstrapMaterializerVersion"]
    assert version["Properties"]["CodeSha256"] == base64.b64encode(
        bytes.fromhex(_inputs().lambda_code_sha256)
    ).decode("ascii")
    permission = resources["FenceBootstrapMaterializerInvokePermission"]
    assert permission["DependsOn"] == "FenceBootstrapInvokerRole"
    environment = function["Properties"]["Environment"]["Variables"]
    assert "GLM52_BOOTSTRAP_MANIFEST_COORDINATE" not in environment
    assert environment["GLM52_ACTIVATION_ID"] == ACTIVATION_ID
    assert environment["GLM52_BOOTSTRAP_PUBLISHER_ROLE_ARN"].endswith(
        ":role/keep-glm52-h1g-fence-bootstrap-artifact-publisher"
    )

    materializer = resources["FenceBootstrapMaterializerRole"]
    publisher = resources["FenceBootstrapArtifactPublisherRole"]
    invoker = resources["FenceBootstrapInvokerRole"]
    assert "s3:PutObject" not in _role_statement_actions(materializer)
    assert "s3:PutObject" not in _role_statement_actions(invoker)
    assert "s3:PutObject" in _role_statement_actions(publisher)
    assert "sts:AssumeRole" in _role_statement_actions(materializer)
    assert "lambda:InvokeFunction" in _role_statement_actions(invoker)
    assert publisher["Properties"]["AssumeRolePolicyDocument"]["Statement"][0][
        "Principal"
    ]["AWS"].endswith(":role/keep-glm52-h1g-fence-bootstrap-materializer")
    assert publisher["DependsOn"] == "FenceBootstrapMaterializerRole"

    raw = canonical_json_bytes(fragment)
    for forbidden in (
        b"ec2:RunInstances",
        b"source_action_allowed",
        b"worker_activation_allowed",
    ):
        assert forbidden not in raw


def test_retained_bootstrap_inputs_reject_manifest_and_launch_fields() -> None:
    projection = dict(_inputs().__dict__)
    projection["bootstrap_manifest_coordinate"] = {"foreign": True}
    try:
        RetainedFenceBootstrapInputs(**projection)
    except TypeError:
        pass
    else:
        raise AssertionError("bootstrap inputs accepted a future manifest coordinate")

    fragment = build_retained_fence_bootstrap_fragment(_inputs())
    assert fragment["Metadata"] == {
        "RecordType": "glm52_h1g_retained_fence_bootstrap_runtime_v2",
        "ActivationId": ACTIVATION_ID,
        "RetainedStackId": RETAINED_STACK_ID,
    }


def test_retained_bootstrap_cli_builds_write_once_canonical_fragment(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "retained-bootstrap-inputs.json"
    input_path.write_bytes(canonical_json_bytes(_inputs().__dict__) + b"\n")
    first = tmp_path / "bootstrap-first"
    second = tmp_path / "bootstrap-second"
    script = ROOT / "aws/glm52-gpu/scripts/build_h1g_support_plane.py"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        str(script),
        "--inputs",
        str(input_path),
        "--retained-bootstrap-output-dir",
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
    expected_name = "retained-fence-bootstrap-v9.json"
    assert {path.name for path in first.iterdir()} == {expected_name}
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    raw = (first / expected_name).read_bytes()
    assert raw == canonical_json_bytes(json.loads(raw.decode("ascii"))) + b"\n"
    assert json.loads(raw) == build_retained_fence_bootstrap_fragment(_inputs())

    overwrite = subprocess.run(
        command + [str(first)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert overwrite.returncode != 0


def _response_metadata(operation: str) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RequestId": operation,
        "RetryAttempts": 0,
    }


class _Sts:
    def get_caller_identity(self) -> dict[str, object]:
        return {
            "Account": ACCOUNT_ID,
            "Arn": (
                "arn:aws:sts::246813579024:assumed-role/"
                "keep-glm52-h1g-fence-bootstrap-materializer/invocation"
            ),
            "UserId": "AROAMATERIALIZER0000:invocation",
            "ResponseMetadata": _response_metadata("sts"),
        }


class _BootstrapS3:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}
        self.put_keys: list[str] = []

    def get_bucket_versioning(self, **_request: object) -> dict[str, object]:
        return {
            "Status": "Enabled",
            "ResponseMetadata": _response_metadata("versioning"),
        }

    def list_object_versions(self, **request: object) -> dict[str, object]:
        key = str(request["Prefix"])
        row = self.rows.get(key)
        versions = (
            []
            if row is None
            else [
                {
                    "Key": key,
                    "VersionId": row["VersionId"],
                    "Size": len(row["Body"]),
                }
            ]
        )
        return {
            "Versions": versions,
            "DeleteMarkers": [],
            "IsTruncated": False,
            "ResponseMetadata": _response_metadata("list"),
        }

    def put_object(self, **request: object) -> dict[str, object]:
        key = str(request["Key"])
        version_id = f"version-{len(self.rows) + 1}"
        self.put_keys.append(key)
        self.rows[key] = {**request, "VersionId": version_id}
        return {
            "VersionId": version_id,
            "ChecksumSHA256": request["ChecksumSHA256"],
            "ServerSideEncryption": request["ServerSideEncryption"],
            "SSEKMSKeyId": request["SSEKMSKeyId"],
            "ResponseMetadata": _response_metadata("put"),
        }

    def get_object(self, **request: object) -> dict[str, object]:
        row = self.rows[str(request["Key"])]
        return {
            "Body": io.BytesIO(row["Body"]),
            "ContentLength": len(row["Body"]),
            "ContentType": row["ContentType"],
            "VersionId": row["VersionId"],
            "ChecksumSHA256": row["ChecksumSHA256"],
            "Metadata": row["Metadata"],
            "ServerSideEncryption": row["ServerSideEncryption"],
            "SSEKMSKeyId": row["SSEKMSKeyId"],
            "ObjectLockMode": None,
            "ObjectLockRetainUntilDate": None,
            "ObjectLockLegalHoldStatus": None,
            "ResponseMetadata": _response_metadata("get"),
        }

    def get_object_tagging(self, **_request: object) -> dict[str, object]:
        return {
            "TagSet": [],
            "ResponseMetadata": _response_metadata("tags"),
        }


def _seed_projection() -> dict[str, object]:
    writer = PrincipalIdentity(
        binding_id="bootstrap-publisher",
        arn=(
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-fence-bootstrap-artifact-publisher"
        ),
        role_id="AROABOOTSTRAPPUBLISH",
    )
    peer_writer = PrincipalIdentity(
        binding_id="bootstrap-publisher-peer",
        arn=(
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-fence-bootstrap-artifact-publisher-peer"
        ),
        role_id="AROAARTIFACTPUBLISH2",
    )
    value = FencePolicyInput(
        build_mode=BuildMode.LEGACY_DISABLED,
        policy_head=PolicyHead.BRIDGE_SEED,
        bucket_name=CAMPAIGN_BUCKET,
        account_id=ACCOUNT_ID,
        kms_key_arn=KMS_ARN,
        reserved_families=tuple(
            ReservedFamily(family_id=family_id, resources=(resource,))
            for family_id, resource in SOURCE_FAMILY_SPECS
        ),
        writer_cohorts=(
            WriterCohort(
                cohort_id="bootstrap",
                members=(writer, peer_writer),
                guard_resources=(RUN_GUARD_RESOURCE,),
                cross_member_denial_evidence_sha256="b" * 64,
            ),
        ),
    )
    principals = [
        {
            "binding_id": item.binding_id,
            "arn": item.arn,
            "role_id": item.role_id,
        }
        for item in (writer, peer_writer)
    ]
    return {
        "build_mode": value.build_mode.value,
        "policy_head": value.policy_head.value,
        "bucket_name": value.bucket_name,
        "account_id": value.account_id,
        "kms_key_arn": value.kms_key_arn,
        "predecessor_policy_sha256": None,
        "freeze_denial_evidence_sha256": None,
        "source_publication_sealed_sha256": None,
        "all_version_inventory_sha256": None,
        "source_settlement_sha256": None,
        "publisher_deny_policy_sha256": None,
        "terminal_prerequisite_sha256": None,
        "legacy_statements": [],
        "permanent_enrolled_resources": [],
        "retired_publishers": [],
        "retired_publisher_resources": [],
        "reserved_families": [
            {"family_id": item.family_id, "resources": list(item.resources)}
            for item in value.reserved_families
        ],
        "writer_cohorts": [
            {
                "cohort_id": "bootstrap",
                "members": principals,
                "guard_resources": [RUN_GUARD_RESOURCE],
                "cross_member_denial_evidence_sha256": "b" * 64,
            }
        ],
        "writer_owned_resources": [],
        "source_validation_readers": [],
        "source_inventory_readers": [],
        "terminal_audit_readers": [],
        "selected_source_keys": [],
        "nonselected_source_keys": [],
        "policy_limits": {
            "max_policy_bytes": value.policy_limits.max_policy_bytes,
            "unallocated_headroom_bytes": (
                value.policy_limits.unallocated_headroom_bytes
            ),
            "component_max_bytes": dict(value.policy_limits.component_max_bytes),
        },
    }


def test_bootstrap_handler_publishes_seed_only_through_split_clients() -> None:
    backing = _BootstrapS3()

    class Reader:
        get_bucket_versioning = backing.get_bucket_versioning
        get_object = backing.get_object
        get_object_tagging = backing.get_object_tagging
        list_object_versions = backing.list_object_versions

    class Publisher:
        put_object = backing.put_object

    function_arn = (
        "arn:aws:lambda:us-west-2:246813579024:function:"
        "keep-glm52-h1g-fence-bootstrap-materializer:1"
    )
    request = {
        "schema_version": 2,
        "record_type": "glm52_h1g_bridge_seed_publication_request_v2",
        "activation_id": ACTIVATION_ID,
        "generation": 1,
        "materializer_function_version_arn": function_arn,
        "expected_live_preseed_policy_sha256": "c" * 64,
        "renderer_input": _seed_projection(),
    }
    services = BootstrapPublicationHandlerServices(
        fixed_artifacts=Task13FixedArtifactServices(
            sts=_Sts(),
            s3=Reader(),
            publisher_s3=Publisher(),
            total_max_attempts=1,
        )
    )

    result = bootstrap_publication_main(
        {"phase": "BRIDGE_SEED", "request": request},
        SimpleNamespace(invoked_function_arn=function_arn),
        services=services,
    )

    assert result["record_type"] == "glm52_h1g_bridge_seed_publication_v2"
    assert result["artifact"]["key"].endswith("/BRIDGE_SEED_POLICY.json")
    assert backing.put_keys == [result["artifact"]["key"]]

    with pytest.raises(ValueError, match="phase|event"):
        bootstrap_publication_main(
            {"phase": "SOURCE", "request": request},
            SimpleNamespace(invoked_function_arn=function_arn),
            services=services,
        )


def test_source_settlement_handler_uses_injected_live_reader_and_returns_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC, datetime

    from glm52_enforcement import fence_source_settlement

    captured: dict[str, object] = {}
    function_arn = (
        "arn:aws:lambda:us-west-2:246813579024:function:"
        "keep-glm52-h1g-fence-source-settlement:3"
    )
    parsed_request = SimpleNamespace(materializer_function_version_arn=function_arn)
    live_reader = SimpleNamespace(
        read_selected_source_rows=lambda: (
            {"source_family": "one", "keys": ("a", "b")},
        )
    )

    monkeypatch.setattr(
        fence_source_settlement,
        "parse_source_settlement_request_v2",
        lambda value: (
            captured.update(request=value),
            parsed_request,
        )[1],
    )

    def settle_source_publication(*, request, services):
        captured["parsed_request"] = request
        captured["selected"] = services.evidence.read_selected_source_rows()
        return fence_source_settlement.SourceSettlementResult(
            disposition=(
                fence_source_settlement.SourceSettlementDisposition.CLOSED_SOURCE
            ),
            classification="SOURCE_INVENTORY_UNPROVED",
            selected_entry=SimpleNamespace(to_dict=lambda: {"slot": "CLOSED_SOURCE"}),
            source_settlement=None,
            source_settled_manifest=None,
            publication_coordinates=(),
            no_launch_evidence={"no_launch": True},
        )

    monkeypatch.setattr(
        fence_source_settlement,
        "settle_source_publication",
        settle_source_publication,
    )
    reader_s3 = object()
    result = source_settlement_main(
        {"request": {"record_type": "request"}},
        SimpleNamespace(invoked_function_arn=function_arn),
        services=SourceSettlementHandlerServices(
            evidence_reader_factory=lambda request: (
                captured.update(reader_request=request),
                live_reader,
            )[1],
            s3=reader_s3,
            artifact_services=Task13FixedArtifactServices(
                sts=_Sts(),
                s3=reader_s3,
                publisher_s3=object(),
                total_max_attempts=1,
            ),
            now_utc=lambda: datetime(2030, 1, 1, tzinfo=UTC),
            sleep=lambda seconds: None,
        ),
    )

    assert captured["parsed_request"] is parsed_request
    assert captured["reader_request"] is parsed_request
    assert captured["selected"] == ({"source_family": "one", "keys": ("a", "b")},)
    assert result["record_type"] == "glm52_h1g_source_settlement_receipt_v2"
    assert result["executed_function_version_arn"] == function_arn
    assert result["classification"] == "SOURCE_INVENTORY_UNPROVED"
    assert result["canonical_identity_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in result.items()
            if key != "canonical_identity_sha256"
        }
    )
