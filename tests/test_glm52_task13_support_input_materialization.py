from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.fence_artifacts import ArtifactCoordinate, FenceSlot
from glm52_enforcement.task13_support_input_materialization import (
    ACCOUNT_ID,
    SupportInputMaterializationError,
    SupportInputServices,
    _organization,
    _parse_bootstrap_manifest,
    _require_singular_history,
    collect_support_build_inputs,
    parse_support_input_materialization_request,
    support_build_inputs_from_mapping,
    support_build_inputs_projection,
)

ROOT = Path(__file__).resolve().parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64


def _artifact_fixture_module():
    path = ROOT / "tests/test_glm52_h1g_fence_artifacts.py"
    spec = importlib.util.spec_from_file_location("_fence_artifact_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_and_coordinate():
    manifest = _artifact_fixture_module()._build_bootstrap_manifest()
    raw = canonical_json_bytes(manifest.to_dict()) + b"\n"
    coordinate = ArtifactCoordinate(
        bucket=manifest.to_dict()["bucket_name"],
        key=(
            "campaigns/glm52-sky-20260724/authorities/fence/manifests/"
            "glm52-v2-amber-quartz/00000001/FENCE_BOOTSTRAP_MANIFEST.json"
        ),
        version_id="manifest-version-1",
        file_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_identity_sha256=manifest.canonical_identity_sha256,
    )
    return manifest, coordinate, raw


def _request_mapping() -> dict[str, object]:
    manifest, coordinate, _raw = _manifest_and_coordinate()
    return {
        "schema_version": 2,
        "record_type": "glm52_h1g_support_input_materialization_request_v2",
        "activation_id": manifest.to_dict()["activation_id"],
        "bootstrap_manifest_coordinate": coordinate.to_dict(),
        "prepare_entry_identity_sha256": manifest.entry(
            FenceSlot.PREPARE_GENESIS_LIVE_STATE
        ).entry_identity_sha256,
        "host_user_data": "#!/bin/sh\ntrue",
        "host_boot_identity_sha256": SHA_A,
        "cryptography_layer_arn": (
            "arn:aws:lambda:us-west-2:246813579024:layer/h1g-crypto:7"
        ),
        "cryptography_layer_sha256": SHA_A,
        "lambda_code_bucket": "keep-glm52-code-246813579024-us-west-2",
        "lambda_code_key": "task13/code/support.zip",
        "lambda_code_version_id": "code-version-1",
        "lambda_code_sha256": SHA_A,
        "price_card_identity_sha256": SHA_A,
        "activation_started_at": "2026-07-31T00:00:00Z",
        "runtime_credential_cutoff_at": "2026-07-31T01:00:00Z",
    }


def _build_inputs():
    manifest, coordinate, _raw = _manifest_and_coordinate()
    prepare = manifest.entry(FenceSlot.PREPARE_GENESIS_LIVE_STATE)
    function_specs = (
        (
            "Task12TerminalV2VersionArn",
            "KeepGlm52Task12TerminalV2VersionArn",
            "keep-glm52-h1g-terminal-v2-writer",
        ),
        (
            "Task12WorkerDrainVersionArn",
            "KeepGlm52Task12WorkerDrainVersionArn",
            "keep-glm52-h1g-worker-drain-signal",
        ),
        (
            "Task9LiabilityWatcherVersionArn",
            "KeepGlm52Task9LiabilityWatcherVersionArn",
            "keep-glm52-h1g-worker-launch-custody",
        ),
    )
    return support_build_inputs_from_mapping(
        {
            "schema_version": 2,
            "record_type": "glm52_h1g_support_build_inputs_v2",
            "account_id": ACCOUNT_ID,
            "region": "us-west-2",
            "run_id": "glm52-sky-20260724",
            "activation_id": "glm52-v2-amber-quartz",
            "fence_stack_name": "keep-glm52-h1g-fence",
            "fence_stack_id": manifest.to_dict()["stack_id"],
            "fence_service_role_arn": manifest.to_dict()["fence_service_role"]["arn"],
            "bootstrap_manifest_coordinate": coordinate.to_dict(),
            "prepare_entry_identity_sha256": prepare.entry_identity_sha256,
            "prepare_template_body_sha256": prepare.to_dict()["template_body_sha256"],
            "prepare_policy_sha256": prepare.policy_sha256,
            "fence_template_inventory": [entry.to_dict() for entry in manifest.entries],
            "task11_writer_bindings": manifest.to_dict()["writer_inventory"],
            "organizations_id": "o-08ddnqdzd3",
            "retained_stack_id": (
                "arn:aws:cloudformation:us-west-2:246813579024:stack/"
                "keep-glm52-gpu/11111111-1111-1111-1111-111111111111"
            ),
            "retained_vpc_id": "vpc-11111111111111111",
            "retained_vpc_cidr": "10.20.0.0/16",
            "existing_subnet_cidrs": ["10.20.1.0/24"],
            "existing_secondary_cidrs": [],
            "primary_az": "us-west-2a",
            "alternate_az": "us-west-2b",
            "primary_public_subnet_id": "subnet-11111111111111111",
            "retained_public_s3_endpoint_id": "vpce-11111111111111111",
            "retained_kms_key_arn": (
                "arn:aws:kms:us-west-2:246813579024:key/"
                "12345678-1234-1234-1234-1234567890ab"
            ),
            "retained_kms_key_id": "12345678-1234-1234-1234-1234567890ab",
            "ledger_table_name": "keep-glm52-h1g-ledger-v1",
            "ledger_table_arn": (
                "arn:aws:dynamodb:us-west-2:246813579024:"
                "table/keep-glm52-h1g-ledger-v1"
            ),
            "model_bucket_name": coordinate.bucket,
            "model_bucket_arn": "arn:aws:s3:::" + coordinate.bucket,
            "model_prefix": "campaigns/glm52-sky-20260724/",
            "host_ami_id": "ami-11111111111111111",
            "host_private_ip": "10.20.101.10",
            "host_user_data": "#!/bin/sh\ntrue",
            "host_user_data_sha256": hashlib.sha256(b"#!/bin/sh\ntrue").hexdigest(),
            "host_boot_identity_sha256": SHA_A,
            "root_volume_gib": 30,
            "root_volume_type": "gp3",
            "root_volume_iops": 3000,
            "root_volume_throughput_mibps": 125,
            "cryptography_layer_arn": (
                "arn:aws:lambda:us-west-2:246813579024:layer/h1g-crypto:7"
            ),
            "cryptography_layer_sha256": SHA_A,
            "lambda_code_bucket": "code-bucket",
            "lambda_code_key": "support.zip",
            "lambda_code_version_id": "v1",
            "lambda_code_sha256": SHA_A,
            "attestation_port": 9443,
            "launch_admission_port": 9444,
            "numeric_binding_port": 9445,
            "retained_cancellation_port": 9446,
            "activation_started_at": "2026-07-31T00:00:00Z",
            "runtime_credential_cutoff_at": "2026-07-31T01:00:00Z",
            "price_card_identity_sha256": SHA_A,
            "retained_function_version_bindings": [
                {
                    "output_key": output_key,
                    "export_name": export_name,
                    "version_arn": (
                        "arn:aws:lambda:us-west-2:246813579024:function:"
                        f"{function_name}:1"
                    ),
                    "function_name": function_name,
                    "version": "1",
                    "code_sha256": SHA_A,
                }
                for output_key, export_name, function_name in function_specs
            ],
            "retained_export_names": sorted(
                [
                    "KeepGlm52VpcId",
                    "KeepGlm52PrimaryPublicSubnetId",
                    "KeepGlm52CampaignKmsKeyArn",
                    "KeepGlm52H1gLedgerArn",
                    *(spec[1] for spec in function_specs),
                ]
            ),
        }
    )


def test_request_v2_rejects_v1_and_caller_copied_inventories() -> None:
    exact = _request_mapping()
    legacy = dict(exact)
    legacy["schema_version"] = 1
    legacy["record_type"] = "glm52_task13_support_input_materialization_request_v1"
    with pytest.raises(SupportInputMaterializationError, match="v2"):
        parse_support_input_materialization_request(legacy)

    for copied in ("fence_template_inventory", "task11_writer_bindings"):
        mutant = dict(exact)
        mutant[copied] = []
        with pytest.raises(SupportInputMaterializationError, match="unknown"):
            parse_support_input_materialization_request(mutant)


def test_build_input_parser_rejects_malformed_retained_function_binding() -> None:
    inputs = _build_inputs()
    value = dict(support_build_inputs_projection(inputs))
    value["retained_function_version_bindings"][0]["version_arn"] = (
        "arn:aws:lambda:us-west-2:246813579024:function:foreign:1"
    )
    with pytest.raises(ValueError, match="function-version"):
        support_build_inputs_from_mapping(value)


def test_bootstrap_manifest_is_unique_prepare_and_excludes_late_slots() -> None:
    manifest, coordinate, raw = _manifest_and_coordinate()
    request = parse_support_input_materialization_request(_request_mapping())
    parsed = _parse_bootstrap_manifest(
        raw,
        request=request,
        fence_stack_id=manifest.to_dict()["stack_id"],
        fence_role_arn=manifest.to_dict()["fence_service_role"]["arn"],
        model_bucket=coordinate.bucket,
    )
    assert tuple(entry.slot for entry in parsed.entries) == (
        FenceSlot.PREPARE_GENESIS_LIVE_STATE,
        FenceSlot.RESERVATION_ONLY,
        FenceSlot.CLOSED_SOURCE,
        FenceSlot.SOURCE_FAMILIES_FROZEN,
    )
    assert FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION not in {
        entry.slot for entry in parsed.entries
    }
    assert FenceSlot.TERMINAL not in {entry.slot for entry in parsed.entries}


def test_pre_prepare_or_wrong_prepare_identity_is_refused() -> None:
    manifest, coordinate, raw = _manifest_and_coordinate()
    request_value = _request_mapping()
    request_value["prepare_entry_identity_sha256"] = SHA_B
    request = parse_support_input_materialization_request(request_value)
    with pytest.raises(SupportInputMaterializationError, match="drifted"):
        _parse_bootstrap_manifest(
            raw,
            request=request,
            fence_stack_id=manifest.to_dict()["stack_id"],
            fence_role_arn=manifest.to_dict()["fence_service_role"]["arn"],
            model_bucket=coordinate.bucket,
        )


class OrganizationsClient:
    def describe_organization(self):
        return {
            "Organization": {
                "Id": "o-08ddnqdzd3",
                "Arn": (
                    "arn:aws:organizations::008316604477:"
                    "organization/o-08ddnqdzd3"
                ),
                "FeatureSet": "ALL",
                "MasterAccountArn": (
                    "arn:aws:organizations::008316604477:"
                    "account/o-08ddnqdzd3/008316604477"
                ),
                "MasterAccountId": "008316604477",
            },
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "request",
                "HTTPHeaders": {},
                "RetryAttempts": 0,
            },
        }


class HistoryClient:
    def __init__(self, versions, markers=None):
        self.versions = versions
        self.markers = [] if markers is None else markers

    def list_object_versions(self, **request):
        del request
        return {
            "Versions": self.versions,
            "DeleteMarkers": self.markers,
            "IsTruncated": False,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "request",
                "HTTPHeaders": {},
                "RetryAttempts": 0,
            },
        }


def _history_services(client):
    return SupportInputServices(
        sts=object(),
        organizations=object(),
        cloudformation=object(),
        ec2=object(),
        kms=object(),
        dynamodb=object(),
        s3=client,
        lambda_client=object(),
        total_max_attempts=1,
    )

def test_live_member_account_organization_is_accepted() -> None:
    services = SupportInputServices(
        sts=object(),
        organizations=OrganizationsClient(),
        cloudformation=object(),
        ec2=object(),
        kms=object(),
        dynamodb=object(),
        s3=object(),
        lambda_client=object(),
        total_max_attempts=1,
    )
    assert _organization(services) == "o-08ddnqdzd3"



def test_manifest_and_prepare_reads_require_singular_current_history() -> None:
    key = "campaigns/glm52-sky-20260724/authorities/fence/x.json"
    exact = HistoryClient([{"Key": key, "VersionId": "v1", "IsLatest": True}])
    _require_singular_history(
        _history_services(exact), bucket="bucket", key=key, version_id="v1", label="x"
    )
    multiple = HistoryClient(
        [
            {"Key": key, "VersionId": "v2", "IsLatest": True},
            {"Key": key, "VersionId": "v1", "IsLatest": False},
        ]
    )
    with pytest.raises(SupportInputMaterializationError, match="singular"):
        _require_singular_history(
            _history_services(multiple), bucket="bucket", key=key, version_id="v1", label="x"
        )
    deleted = HistoryClient(
        [{"Key": key, "VersionId": "v1", "IsLatest": False}],
        [{"Key": key, "VersionId": "d1", "IsLatest": True}],
    )
    with pytest.raises(SupportInputMaterializationError, match="delete marker"):
        _require_singular_history(
            _history_services(deleted), bucket="bucket", key=key, version_id="v1", label="x"
        )


def test_complete_second_snapshot_drift_is_refused(monkeypatch) -> None:
    first = _build_inputs()
    second = replace(first, host_private_ip="10.20.101.11")
    snapshots = iter((first, second))
    monkeypatch.setattr(
        "glm52_enforcement.task13_support_input_materialization._collect_snapshot",
        lambda **_kwargs: next(snapshots),
    )
    services = _history_services(object())
    with pytest.raises(SupportInputMaterializationError, match="mutated"):
        collect_support_build_inputs(request=_request_mapping(), services=services)
