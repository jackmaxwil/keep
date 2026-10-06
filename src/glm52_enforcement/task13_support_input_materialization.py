"""Authenticated Task 13 pre-create support-input materialization.

The request accepted here contains only immutable facts which AWS cannot
discover.  Every environment fact is read twice through injected, zero-retry
service clients and the resulting projection is validated by the existing
``SupportBuildInputs`` contract before it can be written.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .canonical import canonical_json_bytes
from .fence_artifacts import (
    ArtifactCoordinate,
    FenceArtifactError,
    FenceManifest,
    FenceSlot,
    ManifestStage,
    parse_artifact_coordinate,
    parse_fence_entry,
    parse_fence_manifest,
)

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
ORGANIZATION_ID = "o-08ddnqdzd3"
ORGANIZATIONS_MANAGEMENT_ACCOUNT_ID = "008316604477"
RETAINED_STACK_NAME = "keep-glm52-gpu"
FENCE_STACK_NAME = "keep-glm52-h1g-fence"
PINNED_RELAY_PORTS = (9443, 9444, 9445, 9446)

PROFILE = "keep-gpu"
_HOST_PRIVATE_IP = "10.20.101.10"
_ROOT_VOLUME_GIB = 30
_ROOT_VOLUME_TYPE = "gp3"
_ROOT_VOLUME_IOPS = 3000
_ROOT_VOLUME_THROUGHPUT_MIBPS = 125
_LEDGER_TABLE_NAME = "keep-glm52-h1g-ledger-v1"
_REQUIRED_RETAINED_EXPORT_BINDINGS = (
    ("VpcId", "KeepGlm52VpcId"),
    ("SubnetId", "KeepGlm52PrimaryPublicSubnetId"),
    ("CampaignKmsKeyArn", "KeepGlm52CampaignKmsKeyArn"),
    ("H1gLedgerArn", "KeepGlm52H1gLedgerArn"),
    (
        "Task12TerminalV2VersionArn",
        "KeepGlm52Task12TerminalV2VersionArn",
    ),
    (
        "Task12WorkerDrainVersionArn",
        "KeepGlm52Task12WorkerDrainVersionArn",
    ),
    (
        "Task9LiabilityWatcherVersionArn",
        "KeepGlm52Task9LiabilityWatcherVersionArn",
    ),
)
_RETAINED_FUNCTION_BINDINGS = (
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
_STABLE_STACK_STATUSES = frozenset(
    {"CREATE_COMPLETE", "UPDATE_COMPLETE", "IMPORT_COMPLETE"}
)
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "activation_id",
        "bootstrap_manifest_coordinate",
        "prepare_entry_identity_sha256",
        "host_user_data",
        "host_boot_identity_sha256",
        "cryptography_layer_arn",
        "cryptography_layer_sha256",
        "lambda_code_bucket",
        "lambda_code_key",
        "lambda_code_version_id",
        "lambda_code_sha256",
        "price_card_identity_sha256",
        "activation_started_at",
        "runtime_credential_cutoff_at",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ACTIVATION = re.compile(r"[a-z0-9][a-z0-9-]{2,63}\Z")
_VERSION = re.compile(r"(?!null\Z)(?!None\Z)[A-Za-z0-9._~+/=-]{1,1024}\Z")
_STACK_ID = re.compile(
    r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
    r"([A-Za-z0-9-]+)/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_LAYER_ARN = re.compile(
    r"arn:aws:lambda:us-west-2:246813579024:"
    r"layer/([A-Za-z0-9-_]+):([1-9][0-9]*)\Z"
)
_LAMBDA_VERSION_ARN = re.compile(
    r"arn:aws:lambda:us-west-2:246813579024:"
    r"function:([A-Za-z0-9_-]+):([1-9][0-9]*)\Z"
)


class SupportInputMaterializationError(ValueError):
    """A live fact or immutable caller coordinate failed closed."""


@dataclass(frozen=True)
class SupportInputMaterializationRequest:
    """The closed caller surface; fence inventories are deliberately absent."""

    schema_version: int
    record_type: str
    activation_id: str
    bootstrap_manifest_coordinate: ArtifactCoordinate
    prepare_entry_identity_sha256: str
    host_user_data: str
    host_boot_identity_sha256: str
    cryptography_layer_arn: str
    cryptography_layer_sha256: str
    lambda_code_bucket: str
    lambda_code_key: str
    lambda_code_version_id: str
    lambda_code_sha256: str
    price_card_identity_sha256: str
    activation_started_at: str
    runtime_credential_cutoff_at: str


@dataclass(frozen=True)
class SupportBuildInputs:
    """Authenticated post-PREPARE inputs used to build disabled support."""

    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    activation_id: str
    fence_stack_name: str
    fence_stack_id: str
    fence_service_role_arn: str
    bootstrap_manifest_coordinate: Mapping[str, object]
    prepare_entry_identity_sha256: str
    prepare_template_body_sha256: str
    prepare_policy_sha256: str
    fence_template_inventory: tuple[Mapping[str, object], ...]
    task11_writer_bindings: tuple[Mapping[str, object], ...]
    organizations_id: str
    retained_stack_id: str
    retained_vpc_id: str
    retained_vpc_cidr: str
    existing_subnet_cidrs: tuple[str, ...]
    existing_secondary_cidrs: tuple[str, ...]
    primary_az: str
    alternate_az: str
    primary_public_subnet_id: str
    retained_public_s3_endpoint_id: str
    retained_kms_key_arn: str
    retained_kms_key_id: str
    ledger_table_name: str
    ledger_table_arn: str
    model_bucket_name: str
    model_bucket_arn: str
    model_prefix: str
    host_ami_id: str
    host_private_ip: str
    host_user_data: str
    host_user_data_sha256: str
    host_boot_identity_sha256: str
    root_volume_gib: int
    root_volume_type: str
    root_volume_iops: int
    root_volume_throughput_mibps: int
    cryptography_layer_arn: str
    cryptography_layer_sha256: str
    lambda_code_bucket: str
    lambda_code_key: str
    lambda_code_version_id: str
    lambda_code_sha256: str
    attestation_port: int
    launch_admission_port: int
    numeric_binding_port: int
    retained_cancellation_port: int
    activation_started_at: str
    runtime_credential_cutoff_at: str
    price_card_identity_sha256: str
    retained_function_version_bindings: tuple[Mapping[str, object], ...]
    retained_export_names: tuple[str, ...]


_BUILD_INPUT_FIELDS = tuple(SupportBuildInputs.__dataclass_fields__)


@dataclass(frozen=True)
class SupportInputServices:
    """Typed service boundary; all clients must use one-attempt configuration."""

    sts: object
    organizations: object
    cloudformation: object
    ec2: object
    kms: object
    dynamodb: object
    s3: object
    lambda_client: object
    total_max_attempts: int
    remaining_time_in_millis: object = lambda: 300_000


def _fail(message: str) -> None:
    raise SupportInputMaterializationError(message)


def _copy_json(value: object, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SupportInputMaterializationError(
            label + " is not canonical JSON data"
        ) from exc


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        _fail(label + " must be one nonempty ASCII string")
    return value


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(label + " must be one lowercase SHA-256")
    return value


def parse_support_input_materialization_request(
    value: object,
) -> SupportInputMaterializationRequest:
    """Parse the sole caller-authored immutable request."""

    if type(value) is not dict or set(value) != _REQUEST_FIELDS:
        missing = sorted(_REQUEST_FIELDS - set(value)) if type(value) is dict else []
        unknown = sorted(set(value) - _REQUEST_FIELDS) if type(value) is dict else []
        _fail(
            "support input materialization request schema mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    assert type(value) is dict
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 2
        or value["record_type"]
        != "glm52_h1g_support_input_materialization_request_v2"
    ):
        _fail("support input materialization request identity is not exact v2")
    activation_id = _text(value["activation_id"], "activation ID")
    if _ACTIVATION.fullmatch(activation_id) is None:
        _fail("activation ID grammar is unsafe")
    try:
        manifest = parse_artifact_coordinate(
            _copy_json(
                value["bootstrap_manifest_coordinate"],
                "bootstrap manifest coordinate",
            )
        )
    except (TypeError, ValueError, FenceArtifactError) as exc:
        raise SupportInputMaterializationError(
            "bootstrap manifest coordinate is not exact"
        ) from exc
    expected_manifest_key = (
        f"campaigns/{RUN_ID}/authorities/fence/manifests/{activation_id}/"
        "00000001/FENCE_BOOTSTRAP_MANIFEST.json"
    )
    if manifest.key != expected_manifest_key:
        _fail("bootstrap manifest coordinate key is foreign")
    prepare_entry_identity = _sha(
        value["prepare_entry_identity_sha256"],
        "PREPARE entry identity",
    )
    host_user_data = _text(value["host_user_data"], "host user data")
    for field in (
        "host_boot_identity_sha256",
        "cryptography_layer_sha256",
        "lambda_code_sha256",
        "price_card_identity_sha256",
    ):
        _sha(value[field], field)
    layer_arn = _text(
        value["cryptography_layer_arn"],
        "cryptography layer ARN",
    )
    if _LAYER_ARN.fullmatch(layer_arn) is None:
        _fail("cryptography layer ARN is not an exact version")
    lambda_bucket = _text(value["lambda_code_bucket"], "Lambda code bucket")
    lambda_key = _text(value["lambda_code_key"], "Lambda code key")
    lambda_version = _text(
        value["lambda_code_version_id"],
        "Lambda code VersionId",
    )
    if (
        ".." in lambda_bucket
        or lambda_key.startswith("/")
        or ".." in lambda_key
        or _VERSION.fullmatch(lambda_version) is None
    ):
        _fail("Lambda artifact coordinate is not immutable and exact")
    if any(
        type(value[field]) is not str
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value[field]
        )
        is None
        for field in ("activation_started_at", "runtime_credential_cutoff_at")
    ):
        _fail("support input time boundary is not exact")
    if datetime.fromisoformat(value["activation_started_at"]) >= datetime.fromisoformat(
        value["runtime_credential_cutoff_at"]
    ):
        _fail("support input time boundary is not increasing")
    return SupportInputMaterializationRequest(
        schema_version=2,
        record_type="glm52_h1g_support_input_materialization_request_v2",
        activation_id=activation_id,
        bootstrap_manifest_coordinate=manifest,
        prepare_entry_identity_sha256=prepare_entry_identity,
        host_user_data=host_user_data,
        host_boot_identity_sha256=value["host_boot_identity_sha256"],
        cryptography_layer_arn=layer_arn,
        cryptography_layer_sha256=value["cryptography_layer_sha256"],
        lambda_code_bucket=lambda_bucket,
        lambda_code_key=lambda_key,
        lambda_code_version_id=lambda_version,
        lambda_code_sha256=value["lambda_code_sha256"],
        price_card_identity_sha256=value["price_card_identity_sha256"],
        activation_started_at=_text(
            value["activation_started_at"],
            "activation timestamp",
        ),
        runtime_credential_cutoff_at=_text(
            value["runtime_credential_cutoff_at"],
            "credential cutoff timestamp",
        ),
    )


def read_canonical_support_input_materialization_request(
    path: Path,
) -> SupportInputMaterializationRequest:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail("materialization request must end in exactly one LF")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SupportInputMaterializationError(
            "materialization request is not ASCII JSON"
        ) from exc
    if type(value) is not dict or canonical_json_bytes(value) + b"\n" != raw:
        _fail("materialization request bytes are not canonical")
    return parse_support_input_materialization_request(value)


def _validate_build_environment(value: Mapping[str, object]) -> None:
    """Reject malformed live facts before constructing the durable v2 record."""

    activation_id = value["activation_id"]
    text_fields = (
        "retained_stack_id",
        "fence_stack_id",
        "fence_service_role_arn",
        "retained_vpc_id",
        "primary_az",
        "alternate_az",
        "retained_vpc_cidr",
        "primary_public_subnet_id",
        "retained_public_s3_endpoint_id",
        "retained_kms_key_arn",
        "retained_kms_key_id",
        "ledger_table_arn",
        "host_ami_id",
        "host_private_ip",
        "ledger_table_name",
        "host_user_data",
        "lambda_code_bucket",
        "lambda_code_key",
        "lambda_code_version_id",
        "model_bucket_name",
        "model_bucket_arn",
        "model_prefix",
        "activation_started_at",
        "runtime_credential_cutoff_at",
    )
    if (
        type(activation_id) is not str
        or _ACTIVATION.fullmatch(activation_id) is None
        or any(
            type(value[field]) is not str or not value[field]
            for field in text_fields
        )
        or _STACK_ID.fullmatch(value["fence_stack_id"]) is None
        or _STACK_ID.fullmatch(value["retained_stack_id"]) is None
        or not value["retained_stack_id"].split("/")[1].startswith(
            RETAINED_STACK_NAME
        )
        or value["fence_service_role_arn"]
        != f"arn:aws:iam::{ACCOUNT_ID}:role/keep-glm52-h1g-fence-service"
        or value["organizations_id"] != ORGANIZATION_ID
        or value["ledger_table_name"] != _LEDGER_TABLE_NAME
        or value["ledger_table_arn"]
        != f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/{_LEDGER_TABLE_NAME}"
        or value["model_bucket_name"]
        != f"keep-glm52-models-{ACCOUNT_ID}-{REGION}"
        or value["model_bucket_arn"]
        != f"arn:aws:s3:::{value['model_bucket_name']}"
        or value["model_prefix"] != f"campaigns/{RUN_ID}/"
        or value["host_private_ip"] != _HOST_PRIVATE_IP
        or value["root_volume_gib"] != _ROOT_VOLUME_GIB
        or value["root_volume_type"] != _ROOT_VOLUME_TYPE
        or value["root_volume_iops"] != _ROOT_VOLUME_IOPS
        or value["root_volume_throughput_mibps"]
        != _ROOT_VOLUME_THROUGHPUT_MIBPS
        or (
            value["attestation_port"],
            value["launch_admission_port"],
            value["numeric_binding_port"],
            value["retained_cancellation_port"],
        )
        != PINNED_RELAY_PORTS
        or _LAYER_ARN.fullmatch(value["cryptography_layer_arn"]) is None
        or _VERSION.fullmatch(value["lambda_code_version_id"]) is None
        or value["lambda_code_key"].startswith("/")
        or ".." in value["lambda_code_key"]
        or ".." in value["lambda_code_bucket"]
    ):
        raise ValueError("support build environment identity drifted")
    for field in (
        "prepare_entry_identity_sha256",
        "prepare_template_body_sha256",
        "prepare_policy_sha256",
        "host_user_data_sha256",
        "host_boot_identity_sha256",
        "cryptography_layer_sha256",
        "lambda_code_sha256",
        "price_card_identity_sha256",
    ):
        _sha(value[field], "support build " + field)
    try:
        host_hash = hashlib.sha256(
            value["host_user_data"].encode("ascii")
        ).hexdigest()
    except UnicodeEncodeError as exc:
        raise ValueError("support build host user-data is not ASCII") from exc
    if host_hash != value["host_user_data_sha256"]:
        raise ValueError("support build host user-data identity drifted")
    try:
        vpc = ipaddress.ip_network(value["retained_vpc_cidr"], strict=True)
        subnet_cidrs = value["existing_subnet_cidrs"]
        secondary_cidrs = value["existing_secondary_cidrs"]
        if type(subnet_cidrs) is not list or type(secondary_cidrs) is not list:
            raise ValueError
        subnets = [ipaddress.ip_network(cidr, strict=True) for cidr in subnet_cidrs]
        secondaries = [
            ipaddress.ip_network(cidr, strict=True) for cidr in secondary_cidrs
        ]
        vpc_networks = [vpc, *secondaries]
        if (
            not subnets
            or len(set(subnet_cidrs)) != len(subnet_cidrs)
            or len(set(secondary_cidrs)) != len(secondary_cidrs)
            or any(
                any(network.overlaps(other) for other in vpc_networks[index + 1 :])
                for index, network in enumerate(vpc_networks)
            )
            or any(
                not any(subnet.subnet_of(network) for network in vpc_networks)
                for subnet in subnets
            )
        ):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("support build network inventory drifted") from None
    if (
        not re.fullmatch(r"vpc-[0-9a-f]{8,17}", value["retained_vpc_id"])
        or not re.fullmatch(
            r"subnet-[0-9a-f]{8,17}", value["primary_public_subnet_id"]
        )
        or not re.fullmatch(
            r"vpce-[0-9a-f]{8,17}", value["retained_public_s3_endpoint_id"]
        )
        or not re.fullmatch(r"ami-[0-9a-f]{8,17}", value["host_ami_id"])
        or value["primary_az"] == value["alternate_az"]
        or not value["primary_az"].startswith(REGION)
        or not value["alternate_az"].startswith(REGION)
        or not value["retained_kms_key_arn"].startswith(
            f"arn:aws:kms:{REGION}:{ACCOUNT_ID}:key/"
        )
        or value["retained_kms_key_arn"].rsplit("/", 1)[-1]
        != value["retained_kms_key_id"]
    ):
        raise ValueError("support build retained resource identity drifted")
    if any(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value[field])
        is None
        for field in ("activation_started_at", "runtime_credential_cutoff_at")
    ):
        raise ValueError("support build time boundary drifted")
    try:
        started = datetime.fromisoformat(value["activation_started_at"])
        cutoff = datetime.fromisoformat(value["runtime_credential_cutoff_at"])
    except ValueError as exc:
        raise ValueError("support build time boundary drifted") from exc
    if started.tzinfo is None or cutoff.tzinfo is None or started >= cutoff:
        raise ValueError("support build time boundary drifted")
    bindings = value["retained_function_version_bindings"]
    if type(bindings) is not list or len(bindings) != len(
        _RETAINED_FUNCTION_BINDINGS
    ):
        raise ValueError("retained function-version inventory drifted")
    for row, expected in zip(bindings, _RETAINED_FUNCTION_BINDINGS, strict=True):
        output_key, export_name, function_name = expected
        if type(row) is not dict:
            raise ValueError("retained function-version inventory drifted")
        version_arn = row.get("version_arn")
        version_match = (
            _LAMBDA_VERSION_ARN.fullmatch(version_arn)
            if type(version_arn) is str
            else None
        )
        if (
            set(row)
            != {
                "output_key",
                "export_name",
                "version_arn",
                "function_name",
                "version",
                "code_sha256",
            }
            or row["output_key"] != output_key
            or row["export_name"] != export_name
            or row["function_name"] != function_name
            or row["code_sha256"] != value["lambda_code_sha256"]
            or version_match is None
            or version_match.group(1) != function_name
            or version_match.group(2) != row["version"]
        ):
            raise ValueError("retained function-version inventory drifted")
    export_names = value["retained_export_names"]
    required_exports = {item[1] for item in _REQUIRED_RETAINED_EXPORT_BINDINGS}
    if (
        type(export_names) is not list
        or export_names != sorted(export_names)
        or len(set(export_names)) != len(export_names)
        or not required_exports.issubset(export_names)
        or any(type(name) is not str or not name for name in export_names)
    ):
        raise ValueError("retained export inventory drifted")


def support_build_inputs_from_mapping(value: object) -> SupportBuildInputs:
    """Parse the exact manifest-derived post-PREPARE build record."""

    if type(value) is not dict or set(value) != set(_BUILD_INPUT_FIELDS):
        missing = (
            sorted(set(_BUILD_INPUT_FIELDS) - set(value))
            if type(value) is dict
            else []
        )
        unknown = (
            sorted(set(value) - set(_BUILD_INPUT_FIELDS))
            if type(value) is dict
            else []
        )
        raise ValueError(
            "support build input v2 schema mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    assert type(value) is dict
    if (
        value["schema_version"] != 2
        or value["record_type"] != "glm52_h1g_support_build_inputs_v2"
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["run_id"] != RUN_ID
        or value["fence_stack_name"] != FENCE_STACK_NAME
    ):
        raise ValueError("support build input v2 fixed identity drifted")
    _validate_build_environment(value)
    coordinate = parse_artifact_coordinate(value["bootstrap_manifest_coordinate"])
    expected_manifest_key = (
        f"campaigns/{RUN_ID}/authorities/fence/manifests/{value['activation_id']}/"
        "00000001/FENCE_BOOTSTRAP_MANIFEST.json"
    )
    if coordinate.key != expected_manifest_key:
        raise ValueError("support build input bootstrap manifest is foreign")
    prepare_identity = _sha(
        value["prepare_entry_identity_sha256"],
        "support build PREPARE entry identity",
    )
    prepare_template_hash = _sha(
        value["prepare_template_body_sha256"],
        "support build PREPARE template hash",
    )
    prepare_policy_hash = _sha(
        value["prepare_policy_sha256"],
        "support build PREPARE policy hash",
    )
    raw_inventory = value["fence_template_inventory"]
    if type(raw_inventory) is not list or len(raw_inventory) != 4:
        raise ValueError(
            "support build fence inventory is not the four bootstrap entries"
        )
    try:
        entries = tuple(parse_fence_entry(item) for item in raw_inventory)
    except (TypeError, ValueError, FenceArtifactError) as exc:
        raise ValueError("support build fence inventory is not authenticated") from exc
    if (
        tuple(entry.slot for entry in entries)
        != (
            FenceSlot.PREPARE_GENESIS_LIVE_STATE,
            FenceSlot.RESERVATION_ONLY,
            FenceSlot.CLOSED_SOURCE,
            FenceSlot.SOURCE_FAMILIES_FROZEN,
        )
        or entries[0].entry_identity_sha256 != prepare_identity
        or entries[0].to_dict()["template_body_sha256"] != prepare_template_hash
        or entries[0].policy_sha256 != prepare_policy_hash
    ):
        raise ValueError("support build PREPARE projection drifted")
    writers = value["task11_writer_bindings"]
    if (
        type(writers) is not list
        or not writers
        or any(type(row) is not dict for row in writers)
        or len({canonical_json_bytes(row) for row in writers}) != len(writers)
    ):
        raise ValueError("support build writer inventory is not exact")
    tuple_fields = {
        "fence_template_inventory",
        "task11_writer_bindings",
        "existing_subnet_cidrs",
        "existing_secondary_cidrs",
        "retained_function_version_bindings",
        "retained_export_names",
    }
    copied = _copy_json(value, "support build input v2")
    arguments = {
        field: tuple(copied[field]) if field in tuple_fields else copied[field]
        for field in _BUILD_INPUT_FIELDS
    }
    arguments["bootstrap_manifest_coordinate"] = coordinate.to_dict()
    return SupportBuildInputs(**arguments)


def support_build_inputs_projection(inputs: SupportBuildInputs) -> Mapping[str, object]:
    """Return canonical detached JSON for the exact v2 build record."""

    if type(inputs) is not SupportBuildInputs:
        raise TypeError("support build input projection requires exact v2 inputs")
    return {
        field: (
            [deepcopy(item) for item in getattr(inputs, field)]
            if field in {
                "fence_template_inventory",
                "task11_writer_bindings",
                "existing_subnet_cidrs",
                "existing_secondary_cidrs",
                "retained_function_version_bindings",
                "retained_export_names",
            }
            else deepcopy(getattr(inputs, field))
        )
        for field in _BUILD_INPUT_FIELDS
    }


def support_build_inputs_identity(inputs: SupportBuildInputs) -> str:
    """Hash the complete authenticated v2 build-input projection."""

    return hashlib.sha256(
        canonical_json_bytes(support_build_inputs_projection(inputs))
    ).hexdigest()


def _metadata(response: object, operation: str) -> Mapping[str, object]:
    if type(response) is not dict:
        _fail(operation + " did not return one response object")
    metadata = response.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
    ):
        _fail(operation + " response metadata or zero-retry proof is invalid")
    return response


def _call(
    client: object,
    method: str,
    *,
    operation: str,
    request: Mapping[str, object],
) -> Mapping[str, object]:
    function = getattr(client, method, None)
    if not callable(function):
        _fail(operation + " typed service method is absent")
    response = function(**dict(request))
    return _metadata(response, operation)


def _pages(
    client: object,
    method: str,
    *,
    operation: str,
    request: Mapping[str, object],
    remaining_time_in_millis: object,
    row_key: str,
    max_records: int,
) -> tuple[Mapping[str, object], ...]:
    if (
        not callable(remaining_time_in_millis)
        or type(row_key) is not str
        or not row_key
        or type(max_records) is not int
        or max_records < 1
    ):
        _fail(operation + " pagination authority is incomplete")
    pages = []
    token: str | None = None
    seen: set[str] = set()
    record_count = 0
    for _index in range(32):
        remaining = remaining_time_in_millis()
        if (
            type(remaining) is not int
            or isinstance(remaining, bool)
            or remaining < 10_000
        ):
            _fail(operation + " pagination lacks bounded remaining time")
        current = dict(request)
        if token is not None:
            current["NextToken"] = token
        response = _call(
            client,
            method,
            operation=operation,
            request=current,
        )
        rows = response.get(row_key)
        if type(rows) is not list:
            _fail(operation + " pagination row inventory is malformed")
        record_count += len(rows)
        if record_count > max_records:
            _fail(operation + " pagination record bound exceeded")
        pages.append(response)
        next_token = response.get("NextToken")
        if next_token is None:
            return tuple(pages)
        if type(next_token) is not str or not next_token or next_token in seen:
            _fail(operation + " pagination token repeated or malformed")
        seen.add(next_token)
        token = next_token
    _fail(operation + " pagination exceeded the closed 32-page bound")


def _unique_rows(
    value: object,
    *,
    label: str,
    identity: str,
) -> tuple[Mapping[str, object], ...]:
    if type(value) is not list or any(type(item) is not dict for item in value):
        _fail(label + " inventory is malformed")
    rows = tuple(value)
    identities = [item.get(identity) for item in rows]
    if any(type(item) is not str or not item for item in identities) or len(
        set(identities)
    ) != len(identities):
        _fail(label + " identity inventory is duplicated or incomplete")
    return rows


def _stack(
    services: SupportInputServices,
    name: str,
) -> tuple[
    Mapping[str, object],
    Mapping[str, str],
    Mapping[str, Mapping[str, object]],
    Mapping[str, object],
]:
    response = _call(
        services.cloudformation,
        "describe_stacks",
        operation="cloudformation.describe_stacks." + name,
        request={"StackName": name},
    )
    stacks = response.get("Stacks")
    if type(stacks) is not list or len(stacks) != 1 or type(stacks[0]) is not dict:
        _fail(name + " stack read did not resolve exactly one stack")
    stack = stacks[0]
    stack_id = stack.get("StackId")
    match = _STACK_ID.fullmatch(stack_id) if type(stack_id) is str else None
    if (
        match is None
        or match.group(1) != name
        or stack.get("StackName") != name
        or stack.get("StackStatus") not in _STABLE_STACK_STATUSES
    ):
        _fail(name + " stack identity or stable status drifted")
    parameters = stack.get("Parameters", [])
    if type(parameters) is not list or any(
        type(item) is not dict for item in parameters
    ):
        _fail(name + " stack parameters are malformed")
    parameter_map: dict[str, str] = {}
    for row in parameters:
        key = row.get("ParameterKey")
        value = row.get("ParameterValue")
        if type(key) is not str or type(value) is not str or key in parameter_map:
            _fail(name + " stack parameter inventory is not unique")
        parameter_map[key] = value
    outputs_value = stack.get("Outputs", [])
    if type(outputs_value) is not list or any(
        type(item) is not dict for item in outputs_value
    ):
        _fail(name + " stack outputs are malformed")
    outputs: dict[str, Mapping[str, object]] = {}
    for row in outputs_value:
        key = row.get("OutputKey")
        if type(key) is not str or not key or key in outputs:
            _fail(name + " stack output inventory is not unique")
        outputs[key] = row
    template_response = _call(
        services.cloudformation,
        "get_template",
        operation="cloudformation.get_template." + name,
        request={"StackName": stack_id, "TemplateStage": "Original"},
    )
    template = template_response.get("TemplateBody")
    if type(template) is str:
        try:
            template = json.loads(template)
        except json.JSONDecodeError as exc:
            raise SupportInputMaterializationError(
                name + " stack template is not JSON"
            ) from exc
    if type(template) is not dict:
        _fail(name + " stack template is not one exact object")
    template = _copy_json(template, name + " stack template")
    return stack, parameter_map, outputs, template


def _output(
    outputs: Mapping[str, Mapping[str, object]],
    key: str,
) -> str:
    row = outputs.get(key)
    if type(row) is not dict:
        _fail("retained stack output is absent: " + key)
    return _text(row.get("OutputValue"), "retained stack output " + key)


def _retained_function_bindings(
    *,
    services: SupportInputServices,
    outputs: Mapping[str, Mapping[str, object]],
    lambda_code_sha256: str,
) -> list[dict[str, object]]:
    expected_code = base64.b64encode(
        bytes.fromhex(lambda_code_sha256)
    ).decode("ascii")
    result: list[dict[str, object]] = []
    for output_key, export_name, expected_name in (
        _RETAINED_FUNCTION_BINDINGS
    ):
        version_arn = _output(outputs, output_key)
        match = _LAMBDA_VERSION_ARN.fullmatch(version_arn)
        if match is None or match.group(1) != expected_name:
            _fail(
                output_key
                + " output is not the exact published function version"
            )
        response = _call(
            services.lambda_client,
            "get_function",
            operation="lambda.get_function." + output_key,
            request={"FunctionName": version_arn},
        )
        configuration = response.get("Configuration")
        base_arn = version_arn.rsplit(":", 1)[0]
        if (
            type(configuration) is not dict
            or configuration.get("FunctionName") != expected_name
            or configuration.get("FunctionArn") != base_arn
            or configuration.get("Version") != match.group(2)
            or configuration.get("CodeSha256") != expected_code
            or configuration.get("State") != "Active"
            or configuration.get("LastUpdateStatus") != "Successful"
        ):
            _fail(
                output_key
                + " function version or code identity drifted"
            )
        result.append(
            {
                "output_key": output_key,
                "export_name": export_name,
                "version_arn": version_arn,
                "function_name": expected_name,
                "version": match.group(2),
                "code_sha256": lambda_code_sha256,
            }
        )
    return result


def _validate_retained_template(template: Mapping[str, object]) -> None:
    resources = template.get("Resources")
    parameters = template.get("Parameters")
    expected_resources = {
        "Vpc": "AWS::EC2::VPC",
        "PublicSubnet": "AWS::EC2::Subnet",
        "S3GatewayEndpoint": "AWS::EC2::VPCEndpoint",
        "CampaignKmsKey": "AWS::KMS::Key",
        "H1gLedger": "AWS::DynamoDB::Table",
        "ModelBucket": "AWS::S3::Bucket",
    }
    if type(resources) is not dict or any(
        type(resources.get(logical_id)) is not dict
        or resources[logical_id].get("Type") != resource_type
        for logical_id, resource_type in expected_resources.items()
    ):
        _fail("retained stack template resource identity drifted")
    if type(parameters) is not dict or not {
        "GpuAmiId",
        "AvailabilityZone",
        "AvailabilityZoneAlt",
    }.issubset(parameters):
        _fail("retained stack template parameter identity drifted")


def _require_singular_history(
    services: SupportInputServices,
    *,
    bucket: str,
    key: str,
    version_id: str,
    label: str,
) -> None:
    versions: list[Mapping[str, object]] = []
    markers: set[tuple[str, str]] = set()
    request: dict[str, object] = {
        "Bucket": bucket,
        "Prefix": key,
        "MaxKeys": 1000,
        "ExpectedBucketOwner": ACCOUNT_ID,
    }
    for _page in range(32):
        response = _call(
            services.s3,
            "list_object_versions",
            operation="s3.list_object_versions." + label,
            request=request,
        )
        rows = response.get("Versions")
        markers_rows = response.get("DeleteMarkers")
        if (
            type(rows) is not list
            or type(markers_rows) is not list
            or any(type(row) is not dict for row in rows + markers_rows)
        ):
            _fail(label + " fixed-key history is malformed")
        if markers_rows:
            _fail(label + " fixed-key history contains a delete marker")
        for row in rows:
            if row.get("Key") != key:
                _fail(label + " fixed-key history contains a foreign key")
            versions.append(row)
        if response.get("IsTruncated") is False:
            break
        if response.get("IsTruncated") is not True:
            _fail(label + " fixed-key history truncation state is malformed")
        key_marker = response.get("NextKeyMarker")
        version_marker = response.get("NextVersionIdMarker")
        marker = (key_marker, version_marker)
        if (
            type(key_marker) is not str
            or not key_marker
            or type(version_marker) is not str
            or not version_marker
            or marker in markers
        ):
            _fail(label + " fixed-key history pagination is malformed")
        markers.add(marker)
        request["KeyMarker"] = key_marker
        request["VersionIdMarker"] = version_marker
    else:
        _fail(label + " fixed-key history exceeded its page bound")
    if (
        len(versions) != 1
        or versions[0].get("VersionId") != version_id
        or versions[0].get("IsLatest") is not True
    ):
        _fail(label + " fixed-key history is not singular and current")


def _read_exact_object(
    services: SupportInputServices,
    *,
    bucket: str,
    key: str,
    version_id: str,
    file_sha256: str,
    canonical_body_sha256: str | None,
    label: str,
) -> bytes:
    _require_singular_history(
        services,
        bucket=bucket,
        key=key,
        version_id=version_id,
        label=label,
    )
    response = _call(
        services.s3,
        "get_object",
        operation="s3.get_object." + label,
        request={
            "Bucket": bucket,
            "Key": key,
            "VersionId": version_id,
            "ChecksumMode": "ENABLED",
            "ExpectedBucketOwner": ACCOUNT_ID,
        },
    )
    body = response.get("Body")
    if not hasattr(body, "read") or not callable(body.read):
        _fail(label + " object body is not readable")
    raw = body.read()
    if type(raw) is not bytes:
        _fail(label + " object body is not bytes")
    expected_checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    if (
        response.get("VersionId") != version_id
        or response.get("ContentLength") != len(raw)
        or response.get("ChecksumSHA256") != expected_checksum
        or hashlib.sha256(raw).hexdigest() != file_sha256
    ):
        _fail(label + " VersionId, checksum, or size drifted")
    if canonical_body_sha256 is not None:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            _fail(label + " canonical JSON framing drifted")
        try:
            value = json.loads(raw.decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SupportInputMaterializationError(
                label + " is not canonical ASCII JSON"
            ) from exc
        if (
            canonical_json_bytes(value) + b"\n" != raw
            or hashlib.sha256(raw[:-1]).hexdigest() != canonical_body_sha256
        ):
            _fail(label + " canonical body identity drifted")
    return raw


def _parse_bootstrap_manifest(
    raw: bytes,
    *,
    request: SupportInputMaterializationRequest,
    fence_stack_id: str,
    fence_role_arn: str,
    model_bucket: str,
) -> FenceManifest:
    try:
        value = json.loads(raw)
        manifest = parse_fence_manifest(value)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SupportInputMaterializationError(
            "bootstrap manifest is not one authenticated v2 record"
        ) from exc
    coordinate = request.bootstrap_manifest_coordinate
    manifest_value = manifest.to_dict()
    if (
        manifest.manifest_stage is not ManifestStage.BOOTSTRAP
        or manifest.canonical_identity_sha256
        != coordinate.canonical_identity_sha256
        or manifest_value["activation_id"] != request.activation_id
        or manifest_value["generation"] != 1
        or manifest_value["stack_id"] != fence_stack_id
        or manifest_value["fence_service_role"].get("arn") != fence_role_arn
        or manifest_value["bucket_name"] != model_bucket
        or manifest.entry(
            FenceSlot.PREPARE_GENESIS_LIVE_STATE
        ).entry_identity_sha256
        != request.prepare_entry_identity_sha256
    ):
        _fail("bootstrap manifest content drifted from the exact request")
    return manifest


def _organization(
    services: SupportInputServices,
) -> str:
    response = _call(
        services.organizations,
        "describe_organization",
        operation="organizations.describe_organization",
        request={},
    )
    organization = response.get("Organization")
    organization_id = (
        organization.get("Id") if type(organization) is dict else None
    )
    if (
        type(organization) is not dict
        or organization_id != ORGANIZATION_ID
        or organization.get("MasterAccountId")
        != ORGANIZATIONS_MANAGEMENT_ACCOUNT_ID
        or organization.get("MasterAccountArn")
        != (
            "arn:aws:organizations::"
            + ORGANIZATIONS_MANAGEMENT_ACCOUNT_ID
            + ":account/"
            + str(organization_id)
            + "/"
            + ORGANIZATIONS_MANAGEMENT_ACCOUNT_ID
        )
        or organization.get("FeatureSet") != "ALL"
        or organization.get("Arn")
        != (
            "arn:aws:organizations::"
            + ORGANIZATIONS_MANAGEMENT_ACCOUNT_ID
            + ":organization/"
            + str(organization_id)
        )
    ):
        _fail("Organizations account or feature set drifted")
    organization_id = _text(
        organization_id,
        "Organizations ID",
    )
    return organization_id


def _caller_account(services: SupportInputServices) -> None:
    response = _call(
        services.sts,
        "get_caller_identity",
        operation="sts.get_caller_identity",
        request={},
    )
    if response.get("Account") != ACCOUNT_ID:
        _fail("authenticated caller account is foreign")
    arn = response.get("Arn")
    if type(arn) is not str or f":{ACCOUNT_ID}:" not in arn:
        _fail("authenticated caller ARN is foreign")


def _collect_snapshot(
    *,
    request: SupportInputMaterializationRequest,
    services: SupportInputServices,
) -> SupportBuildInputs:
    _caller_account(services)
    organization_id = _organization(services)
    (
        retained_stack,
        retained_parameters,
        retained_outputs,
        retained_template,
    ) = _stack(services, RETAINED_STACK_NAME)
    fence_stack, _fence_parameters, fence_outputs, fence_template = _stack(
        services,
        FENCE_STACK_NAME,
    )
    _validate_retained_template(retained_template)
    fence_stack_id = _text(fence_stack.get("StackId"), "fence stack ID")
    fence_role = _text(
        fence_stack.get("RoleARN"),
        "fence service role ARN",
    )
    if fence_role != (f"arn:aws:iam::{ACCOUNT_ID}:role/keep-glm52-h1g-fence-service"):
        _fail("fence service role identity drifted")
    activation_output = fence_outputs.get("ActivationId")
    if (
        activation_output is not None
        and activation_output.get("OutputValue") != request.activation_id
    ):
        _fail("fence stack activation output drifted")
    retained_stack_id = _text(
        retained_stack.get("StackId"),
        "retained stack ID",
    )
    vpc_id = _output(retained_outputs, "VpcId")
    primary_subnet_id = _output(retained_outputs, "SubnetId")
    kms_arn = _output(retained_outputs, "CampaignKmsKeyArn")
    ledger_arn = _output(retained_outputs, "H1gLedgerArn")
    model_bucket = _output(retained_outputs, "ModelBucketName")
    export_owners: dict[str, str] = {}
    for output_key, row in retained_outputs.items():
        export_name = row.get("ExportName")
        if export_name is None:
            continue
        export_name = _text(
            export_name,
            "retained stack export name",
        )
        if export_name in export_owners:
            _fail("retained stack export ownership is duplicated")
        export_owners[export_name] = output_key
    required_export_names = {
        export_name for _output_key, export_name in _REQUIRED_RETAINED_EXPORT_BINDINGS
    }
    if not required_export_names.issubset(export_owners):
        _fail("retained stack required export identity is absent")
    for output_key, export_name in _REQUIRED_RETAINED_EXPORT_BINDINGS:
        if export_owners.get(export_name) != output_key:
            _fail("retained stack export ownership conflicts with output")
    retained_function_bindings = _retained_function_bindings(
        services=services,
        outputs=retained_outputs,
        lambda_code_sha256=request.lambda_code_sha256,
    )
    vpc_response = _call(
        services.ec2,
        "describe_vpcs",
        operation="ec2.describe_vpcs",
        request={"VpcIds": [vpc_id]},
    )
    vpcs = _unique_rows(
        vpc_response.get("Vpcs"),
        label="VPC",
        identity="VpcId",
    )
    if len(vpcs) != 1:
        _fail("retained VPC read did not resolve exactly one VPC")
    vpc = vpcs[0]
    if (
        vpc.get("VpcId") != vpc_id
        or vpc.get("OwnerId") != ACCOUNT_ID
        or vpc.get("State") != "available"
    ):
        _fail("retained VPC identity is foreign or unavailable")
    primary_cidr = _text(vpc.get("CidrBlock"), "retained VPC CIDR")
    associations = vpc.get("CidrBlockAssociationSet")
    if type(associations) is not list or any(
        type(item) is not dict for item in associations
    ):
        _fail("retained VPC CIDR association inventory is malformed")
    associated_cidrs = []
    for association in associations:
        state = association.get("CidrBlockState")
        cidr = association.get("CidrBlock")
        if (
            type(state) is not dict
            or state.get("State") != "associated"
            or type(cidr) is not str
        ):
            _fail("retained VPC CIDR association is not active")
        associated_cidrs.append(cidr)
    if associated_cidrs.count(primary_cidr) != 1:
        _fail("retained primary VPC CIDR is not uniquely associated")
    secondary_cidrs = sorted(cidr for cidr in associated_cidrs if cidr != primary_cidr)
    subnets: list[Mapping[str, object]] = []
    for page in _pages(
        services.ec2,
        "describe_subnets",
        operation="ec2.describe_subnets",
        request={
            "Filters": [{"Name": "vpc-id", "Values": [vpc_id]}],
            "MaxResults": 1000,
        },
        remaining_time_in_millis=services.remaining_time_in_millis,
        row_key="Subnets",
        max_records=128,
    ):
        rows = page.get("Subnets")
        if type(rows) is not list or any(type(item) is not dict for item in rows):
            _fail("retained subnet page is malformed")
        subnets.extend(rows)
    subnet_rows = _unique_rows(
        subnets,
        label="retained subnet",
        identity="SubnetId",
    )
    if not subnet_rows:
        _fail("retained subnet inventory is empty")
    for subnet in subnet_rows:
        if (
            subnet.get("VpcId") != vpc_id
            or subnet.get("OwnerId") != ACCOUNT_ID
            or subnet.get("State") != "available"
        ):
            _fail("retained subnet identity is foreign or unavailable")
    primary_rows = [
        subnet for subnet in subnet_rows if subnet.get("SubnetId") == primary_subnet_id
    ]
    if len(primary_rows) != 1:
        _fail("primary retained subnet is absent")
    primary_az = _text(
        retained_parameters.get("AvailabilityZone"),
        "primary availability zone",
    )
    alternate_az = _text(
        retained_parameters.get("AvailabilityZoneAlt"),
        "alternate availability zone",
    )
    if primary_rows[0].get("AvailabilityZone") != primary_az or not any(
        subnet.get("AvailabilityZone") == alternate_az for subnet in subnet_rows
    ):
        _fail("retained subnet availability-zone binding drifted")
    endpoint_pages = _pages(
        services.ec2,
        "describe_vpc_endpoints",
        operation="ec2.describe_vpc_endpoints",
        request={
            "Filters": [
                {"Name": "vpc-id", "Values": [vpc_id]},
                {
                    "Name": "service-name",
                    "Values": [f"com.amazonaws.{REGION}.s3"],
                },
                {"Name": "vpc-endpoint-type", "Values": ["Gateway"]},
            ],
            "MaxResults": 1000,
        },
        remaining_time_in_millis=services.remaining_time_in_millis,
        row_key="VpcEndpoints",
        max_records=16,
    )
    endpoint_rows: list[Mapping[str, object]] = []
    for page in endpoint_pages:
        values = page.get("VpcEndpoints")
        if type(values) is not list or any(type(item) is not dict for item in values):
            _fail("S3 VPC endpoint page is malformed")
        endpoint_rows.extend(values)
    endpoints = _unique_rows(
        endpoint_rows,
        label="S3 VPC endpoint",
        identity="VpcEndpointId",
    )
    if (
        len(endpoints) != 1
        or endpoints[0].get("VpcId") != vpc_id
        or endpoints[0].get("OwnerId") != ACCOUNT_ID
        or endpoints[0].get("State") != "available"
        or endpoints[0].get("ServiceName") != f"com.amazonaws.{REGION}.s3"
        or endpoints[0].get("VpcEndpointType") != "Gateway"
    ):
        _fail("retained public S3 endpoint identity drifted")
    endpoint_id = _text(
        endpoints[0].get("VpcEndpointId"),
        "retained public S3 endpoint ID",
    )
    kms_response = _call(
        services.kms,
        "describe_key",
        operation="kms.describe_key",
        request={"KeyId": kms_arn},
    )
    key = kms_response.get("KeyMetadata")
    if (
        type(key) is not dict
        or key.get("AWSAccountId") != ACCOUNT_ID
        or key.get("Arn") != kms_arn
        or key.get("Enabled") is not True
        or key.get("KeyState") != "Enabled"
        or key.get("KeyUsage") != "ENCRYPT_DECRYPT"
    ):
        _fail("retained KMS key identity or state drifted")
    kms_id = _text(key.get("KeyId"), "retained KMS key ID")
    table_response = _call(
        services.dynamodb,
        "describe_table",
        operation="dynamodb.describe_table",
        request={"TableName": _LEDGER_TABLE_NAME},
    )
    table = table_response.get("Table")
    if (
        type(table) is not dict
        or table.get("TableName") != _LEDGER_TABLE_NAME
        or table.get("TableArn") != ledger_arn
        or table.get("TableStatus") != "ACTIVE"
        or type(table.get("BillingModeSummary")) is not dict
        or table["BillingModeSummary"].get("BillingMode") != "PAY_PER_REQUEST"
    ):
        _fail("retained DynamoDB ledger identity or state drifted")
    for bucket in dict.fromkeys((model_bucket, request.lambda_code_bucket)):
        versioning = _call(
            services.s3,
            "get_bucket_versioning",
            operation="s3.get_bucket_versioning." + bucket,
            request={
                "Bucket": bucket,
                "ExpectedBucketOwner": ACCOUNT_ID,
            },
        )
        if versioning.get("Status") != "Enabled":
            _fail(bucket + " versioning is not enabled")
    host_ami_id = _text(
        retained_parameters.get("GpuAmiId"),
        "host AMI ID",
    )
    image_response = _call(
        services.ec2,
        "describe_images",
        operation="ec2.describe_images",
        request={"ImageIds": [host_ami_id], "Owners": [ACCOUNT_ID]},
    )
    images = _unique_rows(
        image_response.get("Images"),
        label="host AMI",
        identity="ImageId",
    )
    if (
        len(images) != 1
        or images[0].get("ImageId") != host_ami_id
        or images[0].get("OwnerId") != ACCOUNT_ID
        or images[0].get("State") != "available"
        or images[0].get("Architecture") != "x86_64"
        or images[0].get("RootDeviceType") != "ebs"
    ):
        _fail("host AMI identity or state drifted")
    manifest_coordinate = request.bootstrap_manifest_coordinate
    manifest_raw = _read_exact_object(
        services,
        bucket=manifest_coordinate.bucket,
        key=manifest_coordinate.key,
        version_id=manifest_coordinate.version_id,
        file_sha256=manifest_coordinate.file_sha256,
        canonical_body_sha256=None,
        label="bootstrap-manifest",
    )
    manifest = _parse_bootstrap_manifest(
        manifest_raw,
        request=request,
        fence_stack_id=fence_stack_id,
        fence_role_arn=fence_role,
        model_bucket=model_bucket,
    )
    prepare = manifest.entry(FenceSlot.PREPARE_GENESIS_LIVE_STATE)
    prepare_value = prepare.to_dict()
    prepare_raw = _read_exact_object(
        services,
        bucket=model_bucket,
        key=_text(prepare_value["template_key"], "PREPARE template key"),
        version_id=_text(
            prepare_value["version_id"],
            "PREPARE template VersionId",
        ),
        file_sha256=_sha(
            prepare_value["template_sha256"],
            "PREPARE template file SHA-256",
        ),
        canonical_body_sha256=_sha(
            prepare_value["template_body_sha256"],
            "PREPARE template body SHA-256",
        ),
        label="fence-template-PREPARE_GENESIS_LIVE_STATE",
    )
    prepare_template_body_sha256 = hashlib.sha256(prepare_raw[:-1]).hexdigest()
    if (
        hashlib.sha256(canonical_json_bytes(fence_template)).hexdigest()
        != prepare_template_body_sha256
    ):
        _fail("support inputs require the active PREPARE stack template")
    policy_response = _call(
        services.s3,
        "get_bucket_policy",
        operation="s3.get_bucket_policy.PREPARE",
        request={
            "Bucket": model_bucket,
            "ExpectedBucketOwner": ACCOUNT_ID,
        },
    )
    policy_text = policy_response.get("Policy")
    if type(policy_text) is not str:
        _fail("direct PREPARE policy readback is absent")
    duplicate = False

    def reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal duplicate
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                duplicate = True
            result[key] = item
        return result

    try:
        policy = json.loads(policy_text, object_pairs_hook=reject_duplicate_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SupportInputMaterializationError(
            "direct PREPARE policy is not strict JSON"
        ) from exc
    prepare_policy_sha256 = hashlib.sha256(canonical_json_bytes(policy)).hexdigest()
    if duplicate or prepare_policy_sha256 != prepare.policy_sha256:
        _fail("direct PREPARE policy identity drifted")
    fence_template_inventory = tuple(entry.to_dict() for entry in manifest.entries)
    task11_writer_bindings = tuple(manifest.to_dict()["writer_inventory"])
    _read_exact_object(
        services,
        bucket=request.lambda_code_bucket,
        key=request.lambda_code_key,
        version_id=request.lambda_code_version_id,
        file_sha256=request.lambda_code_sha256,
        canonical_body_sha256=None,
        label="lambda-code-zip",
    )
    layer_match = _LAYER_ARN.fullmatch(request.cryptography_layer_arn)
    assert layer_match is not None
    layer_response = _call(
        services.lambda_client,
        "get_layer_version_by_arn",
        operation="lambda.get_layer_version_by_arn",
        request={"Arn": request.cryptography_layer_arn},
    )
    content = layer_response.get("Content")
    if (
        layer_response.get("LayerVersionArn") != request.cryptography_layer_arn
        or layer_response.get("Version") != int(layer_match.group(2))
        or type(content) is not dict
        or content.get("CodeSha256")
        != base64.b64encode(bytes.fromhex(request.cryptography_layer_sha256)).decode(
            "ascii"
        )
        or type(content.get("CodeSize")) is not int
        or content["CodeSize"] <= 0
    ):
        _fail("cryptography layer version or code identity drifted")
    host_user_data_sha256 = hashlib.sha256(
        request.host_user_data.encode("ascii")
    ).hexdigest()
    mapping = {
        "schema_version": 2,
        "record_type": "glm52_h1g_support_build_inputs_v2",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": request.activation_id,
        "fence_stack_name": FENCE_STACK_NAME,
        "fence_stack_id": fence_stack_id,
        "fence_service_role_arn": fence_role,
        "bootstrap_manifest_coordinate": manifest_coordinate.to_dict(),
        "prepare_entry_identity_sha256": prepare.entry_identity_sha256,
        "prepare_template_body_sha256": prepare_template_body_sha256,
        "prepare_policy_sha256": prepare_policy_sha256,
        "fence_template_inventory": [
            deepcopy(item) for item in fence_template_inventory
        ],
        "task11_writer_bindings": [
            deepcopy(item) for item in task11_writer_bindings
        ],
        "organizations_id": organization_id,
        "retained_stack_id": retained_stack_id,
        "retained_vpc_id": vpc_id,
        "retained_vpc_cidr": primary_cidr,
        "existing_subnet_cidrs": sorted(
            _text(subnet.get("CidrBlock"), "retained subnet CIDR")
            for subnet in subnet_rows
        ),
        "existing_secondary_cidrs": secondary_cidrs,
        "primary_az": primary_az,
        "alternate_az": alternate_az,
        "primary_public_subnet_id": primary_subnet_id,
        "retained_public_s3_endpoint_id": endpoint_id,
        "retained_kms_key_arn": kms_arn,
        "retained_kms_key_id": kms_id,
        "ledger_table_name": _LEDGER_TABLE_NAME,
        "ledger_table_arn": ledger_arn,
        "model_bucket_name": model_bucket,
        "model_bucket_arn": f"arn:aws:s3:::{model_bucket}",
        "model_prefix": f"campaigns/{RUN_ID}/",
        "host_ami_id": host_ami_id,
        "host_private_ip": _HOST_PRIVATE_IP,
        "host_user_data": request.host_user_data,
        "host_user_data_sha256": host_user_data_sha256,
        "host_boot_identity_sha256": request.host_boot_identity_sha256,
        "root_volume_gib": _ROOT_VOLUME_GIB,
        "root_volume_type": _ROOT_VOLUME_TYPE,
        "root_volume_iops": _ROOT_VOLUME_IOPS,
        "root_volume_throughput_mibps": (_ROOT_VOLUME_THROUGHPUT_MIBPS),
        "cryptography_layer_arn": request.cryptography_layer_arn,
        "cryptography_layer_sha256": (request.cryptography_layer_sha256),
        "lambda_code_bucket": request.lambda_code_bucket,
        "lambda_code_key": request.lambda_code_key,
        "lambda_code_version_id": request.lambda_code_version_id,
        "lambda_code_sha256": request.lambda_code_sha256,
        "attestation_port": PINNED_RELAY_PORTS[0],
        "launch_admission_port": PINNED_RELAY_PORTS[1],
        "numeric_binding_port": PINNED_RELAY_PORTS[2],
        "retained_cancellation_port": PINNED_RELAY_PORTS[3],
        "activation_started_at": request.activation_started_at,
        "runtime_credential_cutoff_at": (request.runtime_credential_cutoff_at),
        "price_card_identity_sha256": (request.price_card_identity_sha256),
        "retained_function_version_bindings": (
            retained_function_bindings
        ),
        "retained_export_names": sorted(export_owners),
    }
    try:
        inputs = support_build_inputs_from_mapping(mapping)
    except (TypeError, ValueError) as exc:
        raise SupportInputMaterializationError(
            "authenticated facts do not form exact SupportBuildInputs v2"
        ) from exc
    return inputs


def collect_support_build_input_snapshot(
    *,
    request: object,
    services: SupportInputServices,
) -> SupportBuildInputs:
    """Collect one complete, independently authenticated support-input snapshot."""

    parsed = (
        request
        if type(request) is SupportInputMaterializationRequest
        else parse_support_input_materialization_request(request)
    )
    if (
        type(services) is not SupportInputServices
        or type(services.total_max_attempts) is not int
        or services.total_max_attempts != 1
    ):
        _fail("typed services are not pinned to one total attempt")
    return _collect_snapshot(request=parsed, services=services)


def collect_support_build_inputs(
    *,
    request: object,
    services: SupportInputServices,
) -> SupportBuildInputs:
    """Double-read and revalidate one exact pre-create input record."""

    first = collect_support_build_input_snapshot(
        request=request,
        services=services,
    )
    try:
        second = collect_support_build_input_snapshot(
            request=request,
            services=services,
        )
    except SupportInputMaterializationError as exc:
        raise SupportInputMaterializationError(
            "authenticated support fact revalidation failed"
        ) from exc
    first_projection = support_build_inputs_projection(first)
    second_projection = support_build_inputs_projection(second)
    if canonical_json_bytes(first_projection) != canonical_json_bytes(
        second_projection
    ):
        _fail("authenticated support facts mutated during revalidation")
    try:
        return support_build_inputs_from_mapping(
            _copy_json(
                second_projection,
                "revalidated support build inputs",
            )
        )
    except ValueError as exc:
        raise SupportInputMaterializationError(
            "SupportBuildInputs failed final revalidation"
        ) from exc


def write_support_build_inputs_once(
    path: Path,
    inputs: SupportBuildInputs,
) -> None:
    """Write canonical bytes once with owner-only permissions."""

    if type(inputs) is not SupportBuildInputs:
        raise TypeError("output writer requires exact SupportBuildInputs")
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite support build inputs")
    projection = support_build_inputs_projection(inputs)
    try:
        revalidated = support_build_inputs_from_mapping(
            _copy_json(projection, "support build input output")
        )
    except ValueError as exc:
        raise SupportInputMaterializationError(
            "support build input output failed revalidation"
        ) from exc
    if revalidated != inputs:
        _fail("support build input output revalidation drifted")
    raw = canonical_json_bytes(projection) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("canonical output write made no progress")
            offset += written
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "PROFILE",
    "SupportBuildInputs",
    "SupportInputMaterializationError",
    "SupportInputMaterializationRequest",
    "SupportInputServices",
    "collect_support_build_input_snapshot",
    "collect_support_build_inputs",
    "support_build_inputs_from_mapping",
    "support_build_inputs_identity",
    "support_build_inputs_projection",
    "parse_support_input_materialization_request",
    "read_canonical_support_input_materialization_request",
    "write_support_build_inputs_once",
]
