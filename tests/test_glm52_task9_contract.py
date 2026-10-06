from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.task9_contract import (
    Task9ContractError,
    build_task9_deployed_identity,
    build_task9_contract,
    validate_task9_deployed_identity,
    validate_task9_contract,
)


ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "aws/glm52-gpu/cfn/h1g/task9-contract-v1.json"


def test_task9_contract_is_static_and_contains_no_deployed_identity() -> None:
    """Break caught: the immutable runtime package embeds fixture/live identity."""

    contract = build_task9_contract(ROOT)
    encoded = json.dumps(contract, sort_keys=True)
    assert contract["record_type"] == "glm52_task9_static_contract_v1"
    assert set(contract["relay_policy"]) == {
        "ATTESTATION",
        "LAUNCH_ADMISSION",
        "NUMERIC_BINDING",
        "RETAINED_CANCELLATION",
    }
    assert "sky_identity" not in contract
    assert "combined_host_instance_id" not in encoded
    assert "client_certificate_sha256" not in encoded
    assert "server_certificate_sha256" not in encoded
    assert "i-00000000000000000" not in encoded
    for purpose in contract["relay_policy"]:
        assert hashlib.sha256(
            ("client:" + purpose).encode("ascii")
        ).hexdigest() not in encoded
        assert hashlib.sha256(
            ("server:" + purpose).encode("ascii")
        ).hexdigest() not in encoded


def _deployed_identity() -> dict[str, object]:
    static = build_task9_contract(ROOT)
    relays = []
    for index, (purpose, policy) in enumerate(
        static["relay_policy"].items(),
        start=1,
    ):
        relays.append(
            {
                "purpose": purpose,
                "client_secret_arn": (
                    "arn:aws:secretsmanager:us-west-2:246813579024:"
                    "secret:/keep/glm52/glm52-sky-20260724/"
                    "activation-0001/" + purpose.lower() + "-client-ABC123"
                ),
                "client_secret_version_id": str(index) * 64,
                "client_secret_version_stage": (
                    "h1g-activation-0001-"
                    + purpose.lower().replace("_", "-")
                    + "-client-tls-v1"
                ),
                "client_certificate_der_sha256": str(index) * 64,
                "server_secret_arn": (
                    "arn:aws:secretsmanager:us-west-2:246813579024:"
                    "secret:/keep/glm52/glm52-sky-20260724/"
                    "activation-0001/combined-host-server-tls-ABC123"
                ),
                "server_secret_version_id": "5" * 64,
                "server_secret_version_stage": (
                    "h1g-activation-0001-combined-host-server-tls-v1"
                ),
                "server_certificate_der_sha256": str(index + 4) * 64,
                "port": policy["port"],
                "principal_arn": policy["principal_arn"],
                "allowed_paths": policy["allowed_paths"],
            }
        )
    return build_task9_deployed_identity(
        static_contract=static,
        activation_id="activation-0001",
        support_binding={
            "support_stack_id": (
                "arn:aws:cloudformation:us-west-2:246813579024:"
                "stack/keep-glm52-h1g-support/"
                "11111111-2222-4333-8444-555555555555"
            ),
            "support_template_body_sha256": "a" * 64,
            "postcreate_manifest_sha256": "b" * 64,
            "describe_stacks_response_sha256": "c" * 64,
        },
        combined_host={
            "instance_id": "i-0123456789abcdef0",
            "private_ip": "10.20.101.10",
            "ami_id": "ami-0123456789abcdef0",
            "instance_profile_arn": (
                "arn:aws:iam::246813579024:instance-profile/"
                "keep-glm52-h1g-combined-host"
            ),
            "boot_identity_sha256": "d" * 64,
            "service_identity_sha256": "e" * 64,
            "consolidation_signal_identity_sha256": "f" * 64,
        },
        tls={
            "issuance_id": "iss-" + "9" * 64,
            "bundle_sha256": "8" * 64,
            "ca_certificate_der_sha256": "7" * 64,
            "not_valid_before": "2026-07-29T12:00:00Z",
            "not_valid_after": "2026-08-01T12:00:00Z",
            "relays": relays,
        },
    )


def test_task9_deployed_identity_closes_real_host_and_der_pins() -> None:
    deployed = _deployed_identity()
    assert validate_task9_deployed_identity(
        deployed,
        static_contract=build_task9_contract(ROOT),
    ) == deployed
    assert deployed["record_type"] == "glm52_task9_deployed_identity_v1"
    assert deployed["combined_host"]["instance_id"] == "i-0123456789abcdef0"
    assert [
        item["server_certificate_sha256"]
        for item in deployed["sky_identity"]["identities"]
    ] == ["5" * 64, "6" * 64, "7" * 64, "8" * 64]
    assert "PRIVATE KEY" not in json.dumps(deployed)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("combined_host", "instance_id"), "i-00000000000000000"),
        (
            ("tls", "relays", 0, "client_certificate_der_sha256"),
            hashlib.sha256(b"client:ATTESTATION").hexdigest(),
        ),
        (
            ("sky_identity", "identities", 0, "server_certificate_sha256"),
            "6" * 64,
        ),
    ],
)
def test_task9_deployed_identity_mutants_fail_closed(path, value) -> None:
    mutant = json.loads(json.dumps(_deployed_identity()))
    target = mutant
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    mutant["canonical_identity_sha256"] = canonical_sha256(
        {
            key: item
            for key, item in mutant.items()
            if key != "canonical_identity_sha256"
        }
    )
    with pytest.raises(Task9ContractError):
        validate_task9_deployed_identity(
            mutant,
            static_contract=build_task9_contract(ROOT),
        )


def test_task9_contract_artifact_regenerates_byte_exact() -> None:
    built = build_task9_contract(ROOT)
    assert json.loads(ARTIFACT.read_text()) == built
    assert ARTIFACT.read_bytes() == (
        json.dumps(
            built,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def test_task9_contract_binds_pinned_sky_rbac_and_four_relays() -> None:
    contract = build_task9_contract(ROOT)
    sky = contract["pinned_sky"]
    assert sky["skypilot_version"] == "0.13.0"
    assert sky["jobs_launch_retryable"] is False
    assert sky["original_provisioner_sha256"] == (
        "fc5d2e4b94f97c10babb583859da24a4fb19256807e763e064e1f294d6c442ef"
    )
    assert contract["sky_server_config"]["api_server"]["host"] == "127.0.0.1"
    assert contract["sky_server_config"]["service_account_role"] == "user"
    assert [
        item["port"] for item in contract["relay_policy"].values()
    ] == [
        18443,
        18444,
        18445,
        18446,
    ]


def test_task9_contract_distinguishes_iam_shape_from_client_token_boundary() -> None:
    contract = build_task9_contract(ROOT)
    iam = contract["effect_boundary"]
    assert iam["allowed_operations"] == [
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "sts:GetCallerIdentity",
    ]
    assert iam["client_token_enforcement_layer"] == (
        "PUBLISHED_FUNCTION_VERSION_AND_LEDGER_NOT_IAM"
    )
    assert "ec2:ClientToken" not in json.dumps(iam, sort_keys=True)
    assert contract["liability"]["aws_hard_post_acceptance_billing_cap"] is False
    assert contract["liability"]["completion_max_calls"] == 6
    assert contract["liability"]["termination_max_calls_per_window"] == 6


def test_task9_contract_mutants_fail_closed() -> None:
    contract = build_task9_contract(ROOT)
    for path, value in (
        (("pinned_sky", "jobs_launch_retryable"), True),
        (("sky_server_config", "api_server"), {}),
        (("effect_boundary", "client_token_enforcement_layer"), "IAM"),
        (("liability", "completion_max_calls"), 7),
    ):
        mutant = json.loads(json.dumps(contract))
        mutant[path[0]][path[1]] = value
        mutant["canonical_identity_sha256"] = canonical_sha256(
            {
                key: item
                for key, item in mutant.items()
                if key != "canonical_identity_sha256"
            }
        )
        with pytest.raises(Task9ContractError):
            validate_task9_contract(mutant, repo_root=ROOT)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("launch_shape", "root_volume", "volume_size_gib"), 1),
        (
            (
                "effect_boundary",
                "same_token_sender_is_sole_run_instances_principal",
            ),
            False,
        ),
        (
            ("liability", "task8_reserve_required_before_possibly_sent"),
            False,
        ),
    ],
)
def test_rehashed_nested_contract_mutants_fail_closed(path, value) -> None:
    mutant = json.loads(json.dumps(build_task9_contract(ROOT)))
    target = mutant
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    mutant["canonical_identity_sha256"] = canonical_sha256(
        {
            key: item
            for key, item in mutant.items()
            if key != "canonical_identity_sha256"
        }
    )
    with pytest.raises(Task9ContractError):
        validate_task9_contract(mutant, repo_root=ROOT)
