"""Static Task 9 policy and deployment-independent launch contract."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import ipaddress
import json
from pathlib import Path
import re
from typing import Mapping, Optional

from .canonical import canonical_sha256
from .sky_admission import (
    EXPECTED_INTERPRETER_PATH,
    EXPECTED_JOBS_SERVER_SHA256,
    EXPECTED_ORIGINAL_PROVISIONER_SHA256,
    PinnedSkyIdentity,
    build_default_identity_contract,
    build_pinned_sky_identity,
)


WHEEL_METADATA_SHA256 = (
    "846f799ff4bd85f135bc23d4503ece7e5941e125a7fdf6e02523a78c37a270b7"
)
DIST_RECORD_SHA256 = (
    "3f55d3067a765631930733b241017d512e67b0b495ae93f9415503098ffb1062"
)
INTERPRETER_SHA256 = (
    "68100c5188b837802c7ae52398389d121b1c063ed244dec11649775b539c3a30"
)
_PATCH = Path("aws/glm52-gpu/skypilot/worker_launch_intent_patch.py")
_LOCK = Path("aws/glm52-gpu/skypilot/skypilot-0.13.0-lock.txt")
_PATCH_SHA256 = (
    "b9d633d4f4756bec51f431ca5e35c23a7db4c30387bc2da618bbc4a006e7f423"
)
_LOCK_SHA256 = (
    "0a6eeed4861ba30701f9d2419dc46a3708e105d04bfb70fcdca569a9a214233e"
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_ACTIVATION = re.compile(r"^[a-z0-9](?:[a-z0-9-]{1,62}[a-z0-9])?$")
_INSTANCE = re.compile(r"^i-[0-9a-f]{17}$")
_AMI = re.compile(r"^ami-[0-9a-f]{17}$")
_STACK = re.compile(
    r"^arn:aws:cloudformation:us-west-2:246813579024:"
    r"stack/keep-glm52-h1g-support/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
_SECRET = re.compile(
    r"^arn:aws:secretsmanager:us-west-2:246813579024:"
    r"secret:/keep/glm52/glm52-sky-20260724/"
    r"[a-z0-9-]+/[A-Za-z0-9/_-]+-[A-Za-z0-9]{6}$"
)
_RELAY_PURPOSES = (
    "ATTESTATION",
    "LAUNCH_ADMISSION",
    "NUMERIC_BINDING",
    "RETAINED_CANCELLATION",
)
_CLIENT_SECRET_PURPOSES = {
    "ATTESTATION": "attestation-client-tls",
    "LAUNCH_ADMISSION": "launch-admission-client-tls",
    "NUMERIC_BINDING": "numeric-binding-client-tls",
    "RETAINED_CANCELLATION": "retained-cancellation-client-tls",
}


class Task9ContractError(ValueError):
    """The generated Task 9 deployment contract drifted."""


def _file_sha256(path: Path, *, expected: str) -> str:
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise Task9ContractError("required pinned-Sky source drifted")
    return expected


def _body(value: Mapping[str, object]) -> Mapping[str, object]:
    return {
        key: item
        for key, item in value.items()
        if key != "canonical_identity_sha256"
    }


def _build_task9_contract_unvalidated(
    repo_root: Path,
) -> Mapping[str, object]:
    if not isinstance(repo_root, Path) or not repo_root.is_absolute():
        raise Task9ContractError("repository root must be absolute")
    server_config = {
        "api_server": {
            "host": "127.0.0.1",
            "port": 46580,
            "api_version": 56,
        },
        "jobs_launch_retryable": False,
        "service_account_role": "user",
    }
    pinned = build_pinned_sky_identity(
        skypilot_version="0.13.0",
        wheel_metadata_sha256=WHEEL_METADATA_SHA256,
        dist_record_sha256=DIST_RECORD_SHA256,
        interpreter_path=EXPECTED_INTERPRETER_PATH,
        interpreter_sha256=INTERPRETER_SHA256,
        dependency_lock_sha256=_file_sha256(
            repo_root / _LOCK,
            expected=_LOCK_SHA256,
        ),
        server_config_sha256=canonical_sha256(server_config),
        original_provisioner_sha256=EXPECTED_ORIGINAL_PROVISIONER_SHA256,
        patched_provisioner_sha256=_file_sha256(
            repo_root / _PATCH,
            expected=_PATCH_SHA256,
        ),
        jobs_server_sha256=EXPECTED_JOBS_SERVER_SHA256,
        jobs_launch_retryable=False,
    )
    relay_policy = {
        purpose: {
            "port": port,
            "principal_arn": (
                "arn:aws:iam::246813579024:role/"
                "keep-glm52-h1g-" + purpose.lower().replace("_", "-")
            ),
            "allowed_paths": list(paths),
            "allow_imds": False,
            "allow_shell": False,
            "allow_aws_credentials": False,
        }
        for purpose, port, paths in (
            ("ATTESTATION", 18443, ("/api/health", "/users/role")),
            ("LAUNCH_ADMISSION", 18444, ("/jobs/launch",)),
            ("NUMERIC_BINDING", 18445, ("/requests/exact", "/jobs/exact")),
            (
                "RETAINED_CANCELLATION",
                18446,
                ("/api/cancel", "/jobs/cancel"),
            ),
        )
    }
    body = {
        "schema_version": 1,
        "record_type": "glm52_task9_static_contract_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "pinned_sky": asdict(pinned),
        "sky_server_config": server_config,
        "relay_policy": relay_policy,
        "effect_boundary": {
            "allowed_operations": [
                "ec2:RunInstances",
                "ec2:TerminateInstances",
                "sts:GetCallerIdentity",
            ],
            "botocore_total_max_attempts": 1,
            "client_token_enforcement_layer": (
                "PUBLISHED_FUNCTION_VERSION_AND_LEDGER_NOT_IAM"
            ),
            "published_version_and_resource_policy_required": True,
            "same_token_sender_is_sole_run_instances_principal": True,
            "combined_host_run_terminate_start_denied": True,
            "generic_pass_role_forbidden": True,
        },
        "launch_shape": {
            "market": "on-demand",
            "instance_type": "p5.48xlarge",
            "min_count": 1,
            "max_count": 1,
            "imds_v2_required": True,
            "root_volume": {
                "delete_on_termination": True,
                "encrypted": True,
                "iops": 3000,
                "throughput_mib_per_second": 125,
                "volume_size_gib": 300,
                "volume_type": "gp3",
            },
            "data_disk_count": 0,
        },
        "liability": {
            "prepared_wal_before_possibly_sent": True,
            "task8_reserve_required_before_possibly_sent": True,
            "gpu_reserve_seconds": 900,
            "gpu_reserve_cost_usd": "13.76",
            "root_volume_tail_usd_max": "0.01",
            "aws_hard_post_acceptance_billing_cap": False,
            "completion_max_calls": 6,
            "completion_window_seconds": 360,
            "watch_interval_seconds": 60,
            "watch_action_deadline_seconds": 20,
            "termination_max_calls_per_window": 6,
            "termination_window_seconds": 360,
            "control_period_seconds": 2_592_000,
            "termination_only_continuation_until_settlement": True,
            "no_instance_scan_settles_liability": False,
        },
    }
    contract = {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }
    return contract


def build_task9_contract(repo_root: Path) -> Mapping[str, object]:
    contract = _build_task9_contract_unvalidated(repo_root)
    return validate_task9_contract(contract, repo_root=repo_root)


def validate_task9_contract(
    value: object,
    *,
    repo_root: Optional[Path] = None,
) -> Mapping[str, object]:
    if type(value) is not dict:
        raise Task9ContractError("Task 9 contract must be an exact mapping")
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    expected = _build_task9_contract_unvalidated(repo_root)
    if value != expected:
        raise Task9ContractError(
            "Task 9 contract differs from the full code-owned contract"
        )
    return value


def _exact_sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise Task9ContractError(label + " must be a lowercase SHA-256")
    return value


def _exact_utc(value: object, label: str) -> datetime:
    if type(value) is not str:
        raise Task9ContractError(label + " must be canonical UTC")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise Task9ContractError(label + " must be canonical UTC") from exc


def _build_task9_deployed_identity_unvalidated(
    *,
    static_contract: Mapping[str, object],
    activation_id: str,
    support_binding: Mapping[str, object],
    combined_host: Mapping[str, object],
    tls: Mapping[str, object],
) -> Mapping[str, object]:
    validate_task9_contract(static_contract)
    if type(activation_id) is not str or _ACTIVATION.fullmatch(activation_id) is None:
        raise Task9ContractError("deployed activation identity is invalid")
    if type(support_binding) is not dict or set(support_binding) != {
        "support_stack_id",
        "support_template_body_sha256",
        "postcreate_manifest_sha256",
        "describe_stacks_response_sha256",
    }:
        raise Task9ContractError("deployed support binding is not closed")
    if _STACK.fullmatch(str(support_binding["support_stack_id"])) is None:
        raise Task9ContractError("deployed support stack identity is foreign")
    for field in (
        "support_template_body_sha256",
        "postcreate_manifest_sha256",
        "describe_stacks_response_sha256",
    ):
        _exact_sha(support_binding[field], field)
    if type(combined_host) is not dict or set(combined_host) != {
        "instance_id",
        "private_ip",
        "ami_id",
        "instance_profile_arn",
        "boot_identity_sha256",
        "service_identity_sha256",
        "consolidation_signal_identity_sha256",
    }:
        raise Task9ContractError("deployed combined-host binding is not closed")
    instance_id = combined_host["instance_id"]
    if (
        type(instance_id) is not str
        or _INSTANCE.fullmatch(instance_id) is None
        or instance_id == "i-00000000000000000"
    ):
        raise Task9ContractError("deployed combined-host instance is not real")
    try:
        private_ip = ipaddress.ip_address(combined_host["private_ip"])
    except (TypeError, ValueError) as exc:
        raise Task9ContractError("deployed combined-host IP is invalid") from exc
    if private_ip.version != 4 or not private_ip.is_private:
        raise Task9ContractError("deployed combined-host IP is not private IPv4")
    if (
        type(combined_host["ami_id"]) is not str
        or _AMI.fullmatch(combined_host["ami_id"]) is None
        or type(combined_host["instance_profile_arn"]) is not str
        or re.fullmatch(
            r"arn:aws:iam::246813579024:instance-profile/"
            r"[A-Za-z0-9+=,.@_-]{1,128}",
            combined_host["instance_profile_arn"],
        )
        is None
    ):
        raise Task9ContractError("deployed host image or profile is foreign")
    for field in (
        "boot_identity_sha256",
        "service_identity_sha256",
        "consolidation_signal_identity_sha256",
    ):
        _exact_sha(combined_host[field], field)
    if type(tls) is not dict or set(tls) != {
        "issuance_id",
        "bundle_sha256",
        "ca_certificate_der_sha256",
        "not_valid_before",
        "not_valid_after",
        "relays",
    }:
        raise Task9ContractError("deployed TLS binding is not closed")
    if (
        type(tls["issuance_id"]) is not str
        or re.fullmatch(r"iss-[0-9a-f]{64}", tls["issuance_id"]) is None
    ):
        raise Task9ContractError("deployed TLS issuance is invalid")
    _exact_sha(tls["bundle_sha256"], "TLS bundle")
    _exact_sha(tls["ca_certificate_der_sha256"], "TLS CA certificate")
    not_before = _exact_utc(tls["not_valid_before"], "TLS not-valid-before")
    not_after = _exact_utc(tls["not_valid_after"], "TLS not-valid-after")
    if not_after <= not_before:
        raise Task9ContractError("deployed TLS validity interval is empty")
    relays = tls["relays"]
    if (
        type(relays) is not list
        or tuple(
            item.get("purpose") if type(item) is dict else None
            for item in relays
        )
        != _RELAY_PURPOSES
    ):
        raise Task9ContractError("four ordered deployed relay identities are required")
    relay_pins = {}
    client_pins = set()
    server_pins = set()
    client_coordinates = set()
    server_coordinates = set()
    normalized_relays = []
    relay_policy = static_contract["relay_policy"]
    for relay in relays:
        if type(relay) is not dict or set(relay) != {
            "purpose",
            "client_secret_arn",
            "client_secret_version_id",
            "client_secret_version_stage",
            "client_certificate_der_sha256",
            "server_secret_arn",
            "server_secret_version_id",
            "server_secret_version_stage",
            "server_certificate_der_sha256",
            "port",
            "principal_arn",
            "allowed_paths",
        }:
            raise Task9ContractError("deployed relay binding is not closed")
        purpose = relay["purpose"]
        policy = relay_policy[purpose]
        if (
            relay["port"] != policy["port"]
            or relay["principal_arn"] != policy["principal_arn"]
            or relay["allowed_paths"] != policy["allowed_paths"]
            or _SECRET.fullmatch(str(relay["client_secret_arn"])) is None
            or _SECRET.fullmatch(str(relay["server_secret_arn"])) is None
            or relay["client_secret_version_stage"]
            != (
                "h1g-"
                + activation_id
                + "-"
                + _CLIENT_SECRET_PURPOSES[purpose]
                + "-v1"
            )
            or relay["server_secret_version_stage"]
            != (
                "h1g-"
                + activation_id
                + "-combined-host-server-tls-v1"
            )
        ):
            raise Task9ContractError("deployed relay policy or secret drifted")
        for field in (
            "client_secret_version_id",
            "server_secret_version_id",
            "client_certificate_der_sha256",
            "server_certificate_der_sha256",
        ):
            _exact_sha(relay[field], purpose + " " + field)
        client_pin = relay["client_certificate_der_sha256"]
        server_pin = relay["server_certificate_der_sha256"]
        if (
            client_pin
            == hashlib.sha256(("client:" + purpose).encode("ascii")).hexdigest()
            or server_pin
            == hashlib.sha256(("server:" + purpose).encode("ascii")).hexdigest()
        ):
            raise Task9ContractError("label-derived certificate identity is forbidden")
        client_pins.add(client_pin)
        server_pins.add(server_pin)
        client_coordinates.add(
            (
                relay["client_secret_arn"],
                relay["client_secret_version_id"],
                relay["client_secret_version_stage"],
            )
        )
        server_coordinates.add(
            (
                relay["server_secret_arn"],
                relay["server_secret_version_id"],
                relay["server_secret_version_stage"],
            )
        )
        relay_pins[purpose] = (client_pin, server_pin)
        normalized_relays.append(dict(relay))
    if (
        len(client_pins) != 4
        or len(server_pins) != 4
        or len(client_coordinates) != 4
        or len(server_coordinates) != 1
    ):
        raise Task9ContractError("deployed relay certificate leaves are not unique")
    pinned = PinnedSkyIdentity(**static_contract["pinned_sky"])
    sky_identity = build_default_identity_contract(
        pinned=pinned,
        relay_certificate_pins=relay_pins,
        service_account_user_id="glm52-sky-service",
        combined_host_instance_id=instance_id,
        combined_host_boot_identity_sha256=combined_host[
            "boot_identity_sha256"
        ],
        combined_host_service_identity_sha256=combined_host[
            "service_identity_sha256"
        ],
        consolidation_signal_identity_sha256=combined_host[
            "consolidation_signal_identity_sha256"
        ],
    )
    sky_mapping = asdict(sky_identity)
    sky_mapping["identities"] = [
        {
            **item,
            "allowed_paths": list(item["allowed_paths"]),
        }
        for item in sky_mapping["identities"]
    ]
    body = {
        "schema_version": 1,
        "record_type": "glm52_task9_deployed_identity_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "activation_id": activation_id,
        "static_contract_identity_sha256": static_contract[
            "canonical_identity_sha256"
        ],
        "support_binding": dict(support_binding),
        "combined_host": dict(combined_host),
        "tls": {
            **{key: value for key, value in tls.items() if key != "relays"},
            "relays": normalized_relays,
        },
        "sky_identity": sky_mapping,
    }
    encoded = json.dumps(body, sort_keys=True)
    if "PRIVATE KEY" in encoded or "private_key" in encoded.lower():
        raise Task9ContractError("deployed identity contains private material")
    return {**body, "canonical_identity_sha256": canonical_sha256(body)}


def build_task9_deployed_identity(
    *,
    static_contract: Mapping[str, object],
    activation_id: str,
    support_binding: Mapping[str, object],
    combined_host: Mapping[str, object],
    tls: Mapping[str, object],
) -> Mapping[str, object]:
    result = _build_task9_deployed_identity_unvalidated(
        static_contract=static_contract,
        activation_id=activation_id,
        support_binding=support_binding,
        combined_host=combined_host,
        tls=tls,
    )
    return validate_task9_deployed_identity(
        result,
        static_contract=static_contract,
    )


def validate_task9_deployed_identity(
    value: object,
    *,
    static_contract: Mapping[str, object],
) -> Mapping[str, object]:
    if type(value) is not dict:
        raise Task9ContractError("deployed Task 9 identity must be one mapping")
    if set(value) != {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "static_contract_identity_sha256",
        "support_binding",
        "combined_host",
        "tls",
        "sky_identity",
        "canonical_identity_sha256",
    }:
        raise Task9ContractError("deployed Task 9 identity fields drifted")
    expected = _build_task9_deployed_identity_unvalidated(
        static_contract=static_contract,
        activation_id=value["activation_id"],
        support_binding=value["support_binding"],
        combined_host=value["combined_host"],
        tls=value["tls"],
    )
    if value != expected:
        raise Task9ContractError("deployed Task 9 identity is internally inconsistent")
    return value


__all__ = [
    "Task9ContractError",
    "build_task9_deployed_identity",
    "build_task9_contract",
    "validate_task9_deployed_identity",
    "validate_task9_contract",
]
