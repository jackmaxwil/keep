from __future__ import annotations

import builtins
from copy import deepcopy
from dataclasses import asdict, replace
import base64
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Mapping

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.live_authority import (
    ACCOUNT_ID,
    PROFILE,
    REGION,
    REQUIRED_LIVE_FAMILIES,
    RUN_ID,
    CallerIdentityObservation,
    ExpectedStateAuthentication,
    H1dExpectedState,
    H1dLiveAuthorityError,
    H1dLiveAuthorityRequest,
    H1dLiveServices,
    LiveReadPage,
    LiveReadSpec,
    SkyRelayProbeResult,
    build_h1d_expected_state,
    build_expected_state_authentication,
    build_sky_relay_probe_result,
    h1d_expected_state_to_mapping,
    inspect_h1d_live_authority,
    reject_serialized_authority_input,
    serialize_non_authoritative_evidence,
)
from glm52_enforcement.spend_authority import (
    SpendListPage,
    spend_authority_result_from_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "aws/glm52-gpu/scripts/inspect_h1d_live_authority.py"
NOW = datetime(2026, 7, 29, 1, 0, tzinfo=timezone.utc)
ZERO = "0" * 64


def _raw_activation_items() -> tuple[dict[str, object], ...]:
    return (
        {
            "PK": {"S": f"RUN#{RUN_ID}"},
            "SK": {"S": "ACTIVATION_INDEX"},
            "record_type": {
                "S": "glm52_production_activation_index"
            },
            "activation_id": {"S": "act-20260729-0001"},
        },
        {
            "PK": {"S": f"RUN#{RUN_ID}"},
            "SK": {
                "S": "ACTIVATION#act-20260729-0001#CONTROL"
            },
            "record_type": {"S": "glm52_production_control"},
            "state": {"S": "AUTHORIZED"},
        },
        {
            "PK": {"S": f"RUN#{RUN_ID}"},
            "SK": {
                "S": (
                    "ACTIVATION#act-20260729-0001#"
                    "ACTION#00000001#SKY#00000001"
                )
            },
            "record_type": {"S": "glm52_production_action"},
            "state": {"S": "CONSUMED"},
        },
    )


def _expected_items() -> dict[str, tuple[dict[str, object], ...]]:
    raw_activation_items = _raw_activation_items()
    return {
        "cloudformation": (
            {
                "stack_id": (
                    "arn:aws:cloudformation:us-west-2:246813579024:"
                    "stack/keep-glm52-h1g-retained/uuid"
                ),
                "status": "UPDATE_COMPLETE",
                "template_sha256": "1" * 64,
                "parameters_sha256": "2" * 64,
                "resources_sha256": "3" * 64,
                "service_role_arn": (
                    "arn:aws:iam::246813579024:"
                    "role/keep-glm52-h1g-retained-cfn"
                ),
                "termination_protection": True,
                "manifest_identity_sha256": "4" * 64,
            },
        ),
        "lambda": (
            {
                "function_arn": (
                    "arn:aws:lambda:us-west-2:246813579024:"
                    "function:keep-glm52-h1g-support-decision:7"
                ),
                "code_sha256": canonical_sha256(
                    {
                        "S3Bucket": "keep-glm52-code",
                        "S3Key": "h1g/decision.zip",
                        "S3ObjectVersion": "decision-version-1",
                    }
                ),
                "configuration_sha256": "6" * 64,
                "policy_sha256": "7" * 64,
                "reserved_concurrency": 1,
                "vpc_attachment_sha256": "8" * 64,
                "timeout_seconds": 840,
                "memory_mib": 1024,
                "log_group_arn": (
                    "arn:aws:logs:us-west-2:246813579024:"
                    "log-group:/aws/lambda/keep-glm52-h1g-support-decision"
                ),
                "alarm_arns": (
                    "arn:aws:cloudwatch:us-west-2:246813579024:"
                    "alarm:keep-glm52-h1g-decision",
                ),
                "dlq_arn": (
                    "arn:aws:sqs:us-west-2:246813579024:"
                    "keep-glm52-h1g-support-dlq"
                ),
            },
        ),
        "iam": (
            {
                "role_arn": (
                    "arn:aws:iam::246813579024:"
                    "role/keep-glm52-h1g-support-decision"
                ),
                "instance_profile_arns": (),
                "inline_policy_sha256": canonical_sha256(
                    [
                        {
                            "PolicyName": "DecisionInline",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [],
                            },
                        }
                    ]
                ),
                "managed_policy_versions_sha256": "a" * 64,
                "attachment_sha256": "b" * 64,
                "passrole_targets": (),
            },
        ),
        "eventbridge": (
            {
                "rule_arn": (
                    "arn:aws:events:us-west-2:246813579024:"
                    "rule/keep-glm52-h1g-support"
                ),
                "state": "ENABLED",
                "targets_sha256": "c" * 64,
            },
        ),
        "scheduler": (
            {
                "schedule_arn": (
                    "arn:aws:scheduler:us-west-2:246813579024:"
                    "schedule/default/keep-glm52-h1g-deadline"
                ),
                "state": "ENABLED",
                "target_sha256": "d" * 64,
            },
        ),
        "sqs": (
            {
                "queue_arn": (
                    "arn:aws:sqs:us-west-2:246813579024:"
                    "keep-glm52-h1g-support-dlq"
                ),
                "approximate_messages": 0,
                "redrive_policy_sha256": "e" * 64,
            },
        ),
        "sns": (
            {
                "subscription_arn": (
                    "arn:aws:sns:us-west-2:246813579024:"
                    "keep-glm52-h1g-alerts:subscription"
                ),
                "protocol": "email",
                "confirmed": True,
                "confirmation_authenticated": True,
            },
        ),
        "ec2": (
            {
                "instance_id": "i-00000000000000001",
                "role": "support-host",
                "instance_type": "c6a.xlarge",
                "state": "running",
                "ami_id": "ami-0123456789abcdef0",
                "user_data_sha256": "f" * 64,
                "profile_arn": (
                    "arn:aws:iam::246813579024:"
                    "instance-profile/keep-glm52-h1g-support-combined-host"
                ),
                "network_sha256": canonical_sha256(
                    {
                        "SubnetId": "subnet-00000000000000001",
                        "SecurityGroupIds": [
                            {"Ref": "CombinedHostSecurityGroup"}
                        ],
                        "PrivateIpAddress": "10.20.101.10",
                        "NetworkInterfaces": None,
                    }
                ),
                "root_volume_gib": 30,
                "root_volume_type": "gp3",
                "root_volume_iops": 3000,
                "root_volume_throughput_mibps": 125,
                "root_volume_encrypted": True,
                "data_volume_gib": 50,
                "data_volume_type": "gp3",
                "data_volume_encrypted": True,
                "volume_attachments_sha256": canonical_sha256(
                    [
                        {"device": "/dev/sdf", "role": "data"},
                        {"device": "/dev/xvda", "role": "root"},
                    ]
                ),
                "tags": {
                    "campaign-run-id": RUN_ID,
                    "glm52-role": "support-host",
                },
            },
        ),
        "s3": (
            {
                "bucket": (
                    "keep-glm52-models-246813579024-us-west-2"
                ),
                "bucket_class": "retained_model_evidence",
                "versioning": "Enabled",
                "policy_sha256": "1" * 64,
                "lifecycle": None,
                "replication": None,
            },
            {
                "bucket": "keep-glm52-h1g-support-rehearsal",
                "bucket_class": "support_rehearsal",
                "versioning": None,
                "policy_sha256": canonical_sha256(None),
                "lifecycle": None,
                "replication": None,
            },
        ),
        "dynamodb": (
            {
                "key": f"RUN#{RUN_ID}|ACTIVATION_INDEX",
                "record_type": "glm52_production_activation_index",
                "body_sha256": canonical_sha256(
                    raw_activation_items[0]
                ),
                "consistent_read": True,
            },
            {
                "key": (
                    f"RUN#{RUN_ID}|"
                    "ACTIVATION#act-20260729-0001#CONTROL"
                ),
                "record_type": "glm52_production_control",
                "body_sha256": canonical_sha256(
                    raw_activation_items[1]
                ),
                "consistent_read": True,
            },
            {
                "key": (
                    f"RUN#{RUN_ID}|ACTIVATION#act-20260729-0001#"
                    "ACTION#00000001#SKY#00000001"
                ),
                "record_type": "glm52_production_action",
                "body_sha256": canonical_sha256(
                    raw_activation_items[2]
                ),
                "consistent_read": True,
            },
        ),
        "ssm": (
            {
                "instance_id": "i-00000000000000001",
                "ping_status": "Online",
                "platform_type": "Linux",
            },
        ),
        "cloudwatch": (
            {
                "alarm_arn": (
                    "arn:aws:cloudwatch:us-west-2:246813579024:"
                    "alarm:keep-glm52-h1g-decision"
                ),
                "state": "OK",
                "metric_status": "COMPLETE",
                "metric_evidence": "ALIGNED_NONEMPTY",
                "treat_missing_data": "breaching",
            },
        ),
        "logs": (
            {
                "log_group_arn": (
                    "arn:aws:logs:us-west-2:246813579024:"
                    "log-group:/aws/lambda/keep-glm52-h1g-support-decision"
                ),
                "retention_days": 30,
                "kms_key_arn": (
                    "arn:aws:kms:us-west-2:246813579024:"
                    "key/12345678-1234-4234-8234-1234567890ab"
                ),
            },
        ),
    }


def _identity_field(family: str) -> str:
    return {
        "cloudformation": "stack_id",
        "lambda": "function_arn",
        "iam": "role_arn",
        "eventbridge": "rule_arn",
        "scheduler": "schedule_arn",
        "sqs": "queue_arn",
        "sns": "subscription_arn",
        "ec2": "instance_id",
        "s3": "bucket",
        "dynamodb": "key",
        "ssm": "instance_id",
        "cloudwatch": "alarm_arn",
        "logs": "log_group_arn",
    }[family]


def _safe_cli_plan(module: object, spec: LiveReadSpec) -> dict[str, object]:
    return dict(
        module.AwsExpectedStateAuthority._code_owned_cli_plan(
            spec.family,
            tuple(spec.expected_items[0]),
        )
    )


def _expected_state(
    *,
    manifest_identity_sha256: str = "6" * 64,
    items: dict[str, tuple[dict[str, object], ...]] | None = None,
) -> H1dExpectedState:
    if items is None:
        items = _expected_items()
    specs = tuple(
        LiveReadSpec(
            family=family,
            operation=f"{family}.inspect_complete",
            parameters={"run_id": RUN_ID},
            identity_field=_identity_field(family),
            expected_items=items[family],
        )
        for family in REQUIRED_LIVE_FAMILIES
    )
    return build_h1d_expected_state(
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        activation_id="act-20260729-0001",
        manifest_identity_sha256=manifest_identity_sha256,
        support_host_instance_id="i-00000000000000001",
        must_start_by="2026-07-29T01:01:00Z",
        execution_deadline="2026-07-30T00:00:00Z",
        specs=specs,
    )


def _spend_result() -> object:
    return spend_authority_result_from_mapping(
        {
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "run_id": RUN_ID,
            "approval_identity_sha256": "7" * 64,
            "descriptor_identity_sha256": "8" * 64,
            "snapshot_identity_sha256": "9" * 64,
            "ledger_genesis_identity_sha256": "a" * 64,
            "ledger_tip_identity_sha256": "b" * 64,
            "ledger_record_count": 2,
            "used_gpu_seconds": 3600,
            "open_gpu_seconds": 0,
            "reserved_gpu_seconds": 0,
            "remaining_gpu_seconds": 82800,
            "refundable_gpu_seconds": 0,
            "used_gpu_cost_usd": "55.04",
            "open_gpu_cost_usd": "0.00",
            "reserved_gpu_cost_usd": "0.00",
            "remaining_gpu_cost_usd": "1265.92",
            "refundable_gpu_cost_usd": "0.00",
            "allocation_intervals": (
                {
                    "instance_id": "i-00000000000000002",
                    "job_id": "qualification-1",
                    "started_at": "2026-07-28T23:00:00Z",
                    "ended_at": "2026-07-29T00:00:00Z",
                    "charged_seconds": 3600,
                    "charged_cost_usd": "55.04",
                    "state": "CLOSED",
                },
            ),
            "observed_at": "2026-07-29T01:00:00Z",
        }
    )


def _probe_result(
    *,
    request_identity_sha256: str | None = None,
    observed_at: str = "2026-07-29T01:00:00Z",
    token_expires_at: str = "2026-07-29T02:00:00Z",
) -> SkyRelayProbeResult:
    if request_identity_sha256 is None:
        request_identity_sha256 = canonical_sha256(
            {
                "account_id": ACCOUNT_ID,
                "region": REGION,
                "run_id": RUN_ID,
                "activation_id": "act-20260729-0001",
            }
        )
    return build_sky_relay_probe_result(
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        request_identity_sha256=request_identity_sha256,
        direct_response_request_id="relay-request-1",
        tls_peer_certificate_sha256="d" * 64,
        attestation_identity_sha256="e" * 64,
        admission_identity_sha256="f" * 64,
        sky_user_identity="service-user",
        sky_roles=("jobs_admin",),
        token_expires_at=token_expires_at,
        effective_controller_identity_sha256="1" * 64,
        observed_at=observed_at,
    )


class _Identity:
    def __init__(self) -> None:
        self.calls = 0

    def get_caller_identity(self) -> CallerIdentityObservation:
        self.calls += 1
        return CallerIdentityObservation(
            account_id=ACCOUNT_ID,
            arn=(
                "arn:aws:sts::246813579024:"
                "assumed-role/keep-glm52-h1g-decision/session"
            ),
            user_id="AROATEST:session",
            credential_expiration="2026-07-29T02:00:00Z",
            request_id="sts-1",
            observed_at="2026-07-29T01:00:00Z",
        )


class _ExpectedAuthority:
    def __init__(self, expected: H1dExpectedState) -> None:
        self.expected_identity = expected.canonical_identity_sha256
        self.postcreate_identity = expected.manifest_identity_sha256
        self.calls = 0

    def authenticate(
        self, expected: H1dExpectedState
    ) -> ExpectedStateAuthentication:
        self.calls += 1
        if expected.canonical_identity_sha256 != self.expected_identity:
            raise H1dLiveAuthorityError(
                "caller substituted expected resources"
            )
        return build_expected_state_authentication(
            expected_state_identity_sha256=self.expected_identity,
            task6_manifest_identity_sha256="2" * 64,
            task6_templates_identity_sha256="3" * 64,
            task7_postcreate_manifest_identity_sha256=self.postcreate_identity,
            task7_inventory_identity_sha256="4" * 64,
            direct_read_request_ids=("task6-1", "task6-2", "task7-1", "task7-2"),
            observed_at="2026-07-29T01:00:00Z",
        )


class _Reader:
    def __init__(self, expected: H1dExpectedState) -> None:
        self.items = {
            spec.family: [deepcopy(item) for item in spec.expected_items]
            for spec in expected.specs
        }
        self.calls: list[tuple[str, object]] = []
        self.tokens: dict[str, tuple[str | None, ...]] = {}

    def read_page(
        self, *, spec: LiveReadSpec, continuation_token: str | None
    ) -> LiveReadPage:
        self.calls.append((spec.family, continuation_token))
        tokens = self.tokens.get(spec.family, (None,))
        index = sum(1 for family, _token in self.calls if family == spec.family) - 1
        pages = len(tokens)
        chunks = [
            self.items[spec.family][start::pages]
            for start in range(pages)
        ]
        return LiveReadPage(
            family=spec.family,
            operation=spec.operation,
            request_token=continuation_token,
            page_index=index,
            items=tuple(chunks[index]),
            next_token=tokens[index],
            request_id=f"{spec.family}-{index}",
            observed_at="2026-07-29T01:00:00Z",
        )


class _Spend:
    def __init__(self) -> None:
        self.calls = 0
        self.result = _spend_result()

    def inspect(self, request: object) -> object:
        self.calls += 1
        return self.result


class _Probe:
    def __init__(self) -> None:
        self.calls = 0
        self.result = _probe_result()

    def inspect(self, request: object) -> SkyRelayProbeResult:
        self.calls += 1
        return self.result


class _Clock:
    def __init__(self, elapsed: float = 6.0) -> None:
        self.values = [NOW, NOW + timedelta(seconds=elapsed)]

    def __call__(self) -> datetime:
        return self.values.pop(0)


def _request(expected: H1dExpectedState) -> H1dLiveAuthorityRequest:
    return H1dLiveAuthorityRequest(
        profile=PROFILE,
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        activation_id=expected.activation_id,
        expected_state_identity_sha256=expected.canonical_identity_sha256,
        spend_request={"kind": "fresh-live-spend"},
        sky_probe_request={
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "run_id": RUN_ID,
            "activation_id": expected.activation_id,
        },
    )


def _services(
    expected: H1dExpectedState, *, elapsed: float = 6.0
) -> tuple[H1dLiveServices, _Reader, _Spend, _Probe]:
    reader = _Reader(expected)
    spend = _Spend()
    probe = _Probe()
    return (
        H1dLiveServices(
            identity=_Identity(),
            reader=reader,
            spend=spend,
            sky_relay_probe=probe,
            clock=_Clock(elapsed),
            expected_state_authority=_ExpectedAuthority(expected),
        ),
        reader,
        spend,
        probe,
    )


def test_closed_h1d_request_result_and_canonical_identity() -> None:
    expected = _expected_state()
    services, reader, spend, probe = _services(expected)
    result = inspect_h1d_live_authority(
        _request(expected), expected, services
    )
    assert tuple(family for family, _ in reader.calls) == REQUIRED_LIVE_FAMILIES
    assert spend.calls == 1
    assert probe.calls == 1
    assert result.account_id == ACCOUNT_ID
    assert result.region == REGION
    assert result.run_id == RUN_ID
    assert (
        result.expected_state_authentication_identity_sha256
        == build_expected_state_authentication(
            expected_state_identity_sha256=(
                expected.canonical_identity_sha256
            ),
            task6_manifest_identity_sha256="2" * 64,
            task6_templates_identity_sha256="3" * 64,
            task7_postcreate_manifest_identity_sha256=(
                expected.manifest_identity_sha256
            ),
            task7_inventory_identity_sha256="4" * 64,
            direct_read_request_ids=(
                "task6-1",
                "task6-2",
                "task7-1",
                "task7-2",
            ),
            observed_at="2026-07-29T01:00:00Z",
        ).canonical_identity_sha256
    )
    assert result.phase_elapsed_seconds == Decimal("6.000000")
    assert len(result.canonical_identity_sha256) == 64
    assert len(result.page_identities) == len(REQUIRED_LIVE_FAMILIES)
    assert set(result.family_identities) == set(REQUIRED_LIVE_FAMILIES)


def test_distinct_authenticated_source_lineages_have_distinct_result_identities(
) -> None:
    expected = _expected_state()

    class LineageAuthority:
        def __init__(self, lineage: str) -> None:
            self.lineage = lineage

        def authenticate(
            self, value: H1dExpectedState
        ) -> ExpectedStateAuthentication:
            return build_expected_state_authentication(
                expected_state_identity_sha256=(
                    value.canonical_identity_sha256
                ),
                task6_manifest_identity_sha256=(
                    "2" * 64 if self.lineage == "a" else "5" * 64
                ),
                task6_templates_identity_sha256="3" * 64,
                task7_postcreate_manifest_identity_sha256=(
                    value.manifest_identity_sha256
                ),
                task7_inventory_identity_sha256="4" * 64,
                direct_read_request_ids=tuple(
                    f"{self.lineage}-{name}"
                    for name in (
                        "task6-1",
                        "task6-2",
                        "task7-1",
                        "task7-2",
                    )
                ),
                observed_at="2026-07-29T01:00:00Z",
            )

    def inspect(lineage: str) -> object:
        services, _reader, _spend, _probe = _services(expected)
        return inspect_h1d_live_authority(
            _request(expected),
            expected,
            replace(
                services,
                expected_state_authority=LineageAuthority(lineage),
            ),
        )

    first = inspect("a")
    same = inspect("a")
    distinct = inspect("b")
    assert first.expected_state_identity_sha256 == (
        distinct.expected_state_identity_sha256
    )
    assert first.expected_state_authentication_identity_sha256 != (
        distinct.expected_state_authentication_identity_sha256
    )
    assert first.canonical_identity_sha256 != (
        distinct.canonical_identity_sha256
    )
    assert same == first


@pytest.mark.parametrize(
    "substitution",
    ("", "f" * 64),
)
def test_missing_or_wrong_expected_authentication_result_identity_rejects(
    substitution: str,
) -> None:
    expected = _expected_state()
    services, _reader, _spend, _probe = _services(expected)
    result = inspect_h1d_live_authority(
        _request(expected), expected, services
    )
    tampered = replace(
        result,
        expected_state_authentication_identity_sha256=substitution,
    )
    with pytest.raises(H1dLiveAuthorityError, match="identity|digest"):
        serialize_non_authoritative_evidence(tampered)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile", "default"),
        ("account_id", "000000000000"),
        ("region", "us-east-1"),
        ("run_id", "foreign"),
        ("activation_id", "foreign"),
        ("expected_state_identity_sha256", ZERO),
    ],
)
def test_wrong_profile_account_region_run_or_manifest_stops_before_inspection(
    field: str, value: str
) -> None:
    expected = _expected_state()
    services, reader, spend, probe = _services(expected)
    request = replace(_request(expected), **{field: value})
    with pytest.raises(H1dLiveAuthorityError):
        inspect_h1d_live_authority(request, expected, services)
    assert reader.calls == []
    assert spend.calls == 0
    assert probe.calls == 0


def test_rehashed_substituted_expected_items_fail_before_service_inspection() -> None:
    original = _expected_state()
    services, reader, spend, probe = _services(original)
    items = _expected_items()
    items["ec2"][0]["instance_type"] = "c6a.2xlarge"
    specs = tuple(
        LiveReadSpec(
            family=family,
            operation=f"{family}.inspect_complete",
            parameters={"run_id": RUN_ID},
            identity_field=_identity_field(family),
            expected_items=items[family],
        )
        for family in REQUIRED_LIVE_FAMILIES
    )
    substituted = build_h1d_expected_state(
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        activation_id=original.activation_id,
        manifest_identity_sha256=original.manifest_identity_sha256,
        support_host_instance_id=original.support_host_instance_id,
        must_start_by=original.must_start_by,
        execution_deadline=original.execution_deadline,
        specs=specs,
    )
    with pytest.raises(H1dLiveAuthorityError, match="substituted"):
        inspect_h1d_live_authority(
            _request(substituted), substituted, services
        )
    assert services.expected_state_authority.calls == 1
    assert services.identity.calls == 0
    assert reader.calls == []
    assert spend.calls == 0
    assert probe.calls == 0


def test_full_pagination_binds_every_page_once() -> None:
    expected = _expected_state()
    services, reader, _spend, _probe = _services(expected)
    reader.tokens["dynamodb"] = ("ddb-page-2", None)
    result = inspect_h1d_live_authority(
        _request(expected), expected, services
    )
    assert [call for call in reader.calls if call[0] == "dynamodb"] == [
        ("dynamodb", None),
        ("dynamodb", "ddb-page-2"),
    ]
    assert len(result.page_identities) == len(REQUIRED_LIVE_FAMILIES) + 1


@pytest.mark.parametrize(
    "mutation",
    ["repeated-token", "missing-request-token", "duplicate-identity", "page-index"],
)
def test_pagination_rejects_repeated_missing_cyclic_duplicate_pages(
    mutation: str,
) -> None:
    expected = _expected_state()
    services, reader, _spend, _probe = _services(expected)
    if mutation == "duplicate-identity":
        reader.items["dynamodb"].append(deepcopy(reader.items["dynamodb"][0]))
    else:
        reader.tokens["dynamodb"] = ("ddb-page-2", "ddb-page-2")
        original = reader.read_page

        def mutant(
            *, spec: LiveReadSpec, continuation_token: str | None
        ) -> LiveReadPage:
            page = original(spec=spec, continuation_token=continuation_token)
            if spec.family == "dynamodb" and continuation_token is not None:
                if mutation == "missing-request-token":
                    return replace(page, request_token=None)
                if mutation == "page-index":
                    return replace(page, page_index=7)
            return page

        reader.read_page = mutant
    with pytest.raises(H1dLiveAuthorityError):
        inspect_h1d_live_authority(_request(expected), expected, services)


@pytest.mark.parametrize("family", REQUIRED_LIVE_FAMILIES)
def test_every_required_resource_family_rejects_exact_drift(family: str) -> None:
    expected = _expected_state()
    services, reader, _spend, _probe = _services(expected)
    reader.items[family][0]["unmodeled"] = True
    with pytest.raises(H1dLiveAuthorityError):
        inspect_h1d_live_authority(_request(expected), expected, services)


@pytest.mark.parametrize(
    ("family", "field", "value"),
    [
        ("cloudwatch", "state", "ALARM"),
        ("cloudwatch", "metric_status", "INSUFFICIENT_DATA"),
        ("sqs", "approximate_messages", 1),
        ("sns", "confirmed", False),
        ("sns", "confirmation_authenticated", False),
        ("ssm", "ping_status", "ConnectionLost"),
        ("ec2", "instance_type", "p5.48xlarge"),
        ("ec2", "root_volume_gib", 299),
        ("s3", "versioning", "Suspended"),
        ("s3", "lifecycle", {"Rules": []}),
        ("s3", "replication", {"Role": "foreign"}),
    ],
)
def test_necessary_observation_failures_block_without_becoming_sufficient(
    family: str, field: str, value: object
) -> None:
    expected = _expected_state()
    services, reader, _spend, _probe = _services(expected)
    reader.items[family][0][field] = value
    with pytest.raises(H1dLiveAuthorityError):
        inspect_h1d_live_authority(_request(expected), expected, services)


def test_exact_tagged_p5_or_unmodeled_instance_blocks() -> None:
    expected = _expected_state()
    services, reader, _spend, _probe = _services(expected)
    reader.items["ec2"].append(
        {
            "instance_id": "i-00000000000000099",
            "role": "production-worker",
            "instance_type": "p5.48xlarge",
            "state": "pending",
            "tags": {"campaign-run-id": RUN_ID},
        }
    )
    with pytest.raises(H1dLiveAuthorityError):
        inspect_h1d_live_authority(_request(expected), expected, services)


def test_sky_relay_probe_is_called_and_caller_asserted_mapping_is_rejected() -> None:
    expected = _expected_state()
    services, _reader, _spend, probe = _services(expected)
    probe.result = {
        "authenticated": True,
        "healthy": True,
        "effective_controller_closed": True,
    }
    with pytest.raises(H1dLiveAuthorityError):
        inspect_h1d_live_authority(_request(expected), expected, services)
    assert probe.calls == 1


@pytest.mark.parametrize(
    ("request_mutation", "probe_identity", "probe_observed_at"),
    (
        ({}, "0" * 64, "2026-07-29T01:00:00Z"),
        ({}, None, "2026-07-29T00:59:59Z"),
        (
            {"activation_id": "act-caller-substitution"},
            None,
            "2026-07-29T01:00:00Z",
        ),
    ),
)
def test_sky_probe_is_bound_to_exact_request_activation_and_current_phase(
    request_mutation: dict[str, object],
    probe_identity: str | None,
    probe_observed_at: str,
) -> None:
    expected = _expected_state()
    services, _reader, _spend, probe = _services(expected)
    request = _request(expected)
    request = replace(
        request,
        sky_probe_request={
            **request.sky_probe_request,
            **request_mutation,
        },
    )
    identity = (
        canonical_sha256(request.sky_probe_request)
        if probe_identity is None
        else probe_identity
    )
    probe.result = _probe_result(
        request_identity_sha256=identity,
        observed_at=probe_observed_at,
    )
    with pytest.raises(H1dLiveAuthorityError):
        inspect_h1d_live_authority(request, expected, services)


@pytest.mark.parametrize("elapsed", [7.000001, 8.0])
def test_live_reinspection_rejects_phase_above_seven_seconds(
    elapsed: float,
) -> None:
    expected = _expected_state()
    services, _reader, _spend, _probe = _services(expected, elapsed=elapsed)
    with pytest.raises(H1dLiveAuthorityError):
        inspect_h1d_live_authority(_request(expected), expected, services)


def test_execution_deadline_is_rechecked_at_phase_completion() -> None:
    original = _expected_state()
    expected = build_h1d_expected_state(
        account_id=original.account_id,
        region=original.region,
        run_id=original.run_id,
        activation_id=original.activation_id,
        manifest_identity_sha256=original.manifest_identity_sha256,
        support_host_instance_id=original.support_host_instance_id,
        must_start_by="2026-07-29T01:00:01Z",
        execution_deadline="2026-07-29T01:00:05Z",
        specs=original.specs,
    )
    services, _reader, _spend, _probe = _services(expected, elapsed=6.0)
    with pytest.raises(H1dLiveAuthorityError, match="execution deadline"):
        inspect_h1d_live_authority(_request(expected), expected, services)


def test_sky_token_must_remain_valid_through_phase_completion() -> None:
    expected = _expected_state()
    services, _reader, _spend, probe = _services(expected, elapsed=6.0)
    probe.result = _probe_result(
        token_expires_at="2026-07-29T01:00:03Z"
    )
    with pytest.raises(H1dLiveAuthorityError, match="token"):
        inspect_h1d_live_authority(_request(expected), expected, services)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("must_start_by", "2026-07-29T00:59:59Z"),
        ("execution_deadline", "2026-07-29T00:59:59Z"),
    ],
)
def test_stale_must_start_or_execution_deadline_blocks(
    field: str, value: str
) -> None:
    expected = replace(_expected_state(), **{field: value})
    if field == "execution_deadline":
        expected = replace(expected, must_start_by="2026-07-29T00:59:58Z")
    expected = build_h1d_expected_state(
        account_id=expected.account_id,
        region=expected.region,
        run_id=expected.run_id,
        activation_id=expected.activation_id,
        manifest_identity_sha256=expected.manifest_identity_sha256,
        support_host_instance_id=expected.support_host_instance_id,
        must_start_by=expected.must_start_by,
        execution_deadline=expected.execution_deadline,
        specs=expected.specs,
    )
    services, _reader, _spend, _probe = _services(expected)
    with pytest.raises(H1dLiveAuthorityError):
        inspect_h1d_live_authority(_request(expected), expected, services)


def test_open_foreign_spend_interval_blocks_live_authority() -> None:
    expected = _expected_state()
    services, _reader, spend, _probe = _services(expected)
    spend.result = replace(
        spend.result,
        open_gpu_seconds=60,
        open_gpu_cost_usd=Decimal("0.92"),
        remaining_gpu_seconds=82740,
        remaining_gpu_cost_usd=Decimal("1265.00"),
    )
    with pytest.raises(H1dLiveAuthorityError):
        inspect_h1d_live_authority(_request(expected), expected, services)


def test_spend_observation_must_be_inside_current_h1d_phase() -> None:
    expected = _expected_state()
    services, _reader, spend, _probe = _services(expected)
    mapping = asdict(spend.result)
    mapping.pop("canonical_identity_sha256")
    mapping["observed_at"] = "2025-07-29T01:00:00Z"
    spend.result = spend_authority_result_from_mapping(
        mapping
    )
    with pytest.raises(H1dLiveAuthorityError, match="spend.*phase"):
        inspect_h1d_live_authority(_request(expected), expected, services)


def test_serialized_evidence_is_non_authoritative_and_cannot_be_reused() -> None:
    expected = _expected_state()
    services, _reader, _spend, _probe = _services(expected)
    result = inspect_h1d_live_authority(
        _request(expected), expected, services
    )
    evidence = serialize_non_authoritative_evidence(result)
    assert evidence["authority_classification"] == "NON_AUTHORITATIVE_EVIDENCE"
    with pytest.raises(H1dLiveAuthorityError):
        reject_serialized_authority_input(evidence)
    assert "evidence" not in inspect.signature(inspect_h1d_live_authority).parameters


def _load_script():
    specification = importlib.util.spec_from_file_location(
        "_inspect_h1d_live_authority", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _without_no_paginate(
    operations: list[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    assert all("--no-paginate" in operation for operation in operations)
    return [
        tuple(part for part in operation if part != "--no-paginate")
        for operation in operations
    ]


def _production_expected_authority_fixture(module: object):
    expected_items = _expected_items()
    support_template = {
        "Resources": {
            "CombinedHost": {
                "Type": "AWS::EC2::Instance",
                "Properties": {
                    "InstanceType": expected_items["ec2"][0]["instance_type"],
                    "ImageId": expected_items["ec2"][0]["ami_id"],
                    "SubnetId": "subnet-00000000000000001",
                    "SecurityGroupIds": [
                        {"Ref": "CombinedHostSecurityGroup"}
                    ],
                    "PrivateIpAddress": "10.20.101.10",
                    "BlockDeviceMappings": [
                        {
                            "DeviceName": "/dev/xvda",
                            "Ebs": {
                                "VolumeSize": 30,
                                "VolumeType": "gp3",
                                "Iops": 3000,
                                "Throughput": 125,
                                "Encrypted": True,
                            },
                        }
                    ],
                    "IamInstanceProfile": {"Ref": "CombinedHostProfile"},
                    "Tags": [
                        {"Key": "campaign-run-id", "Value": RUN_ID},
                        {"Key": "glm52-role", "Value": "support-host"},
                    ],
                    "UserData": "fixture-user-data",
                },
            },
            "CombinedHostDataVolume": {
                "Type": "AWS::EC2::Volume",
                "Properties": {
                    "Size": 50,
                    "VolumeType": "gp3",
                    "Encrypted": True,
                },
            },
            "CombinedHostDataVolumeAttachment": {
                "Type": "AWS::EC2::VolumeAttachment",
                "Properties": {
                    "Device": "/dev/sdf",
                    "InstanceId": {"Ref": "CombinedHost"},
                    "VolumeId": {"Ref": "CombinedHostDataVolume"},
                },
            },
            "DecisionRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": "lambda.amazonaws.com"
                                },
                                "Action": "sts:AssumeRole",
                            }
                        ],
                    },
                    "Path": "/",
                    "PermissionsBoundary": (
                        f"arn:aws:iam::{ACCOUNT_ID}:policy/"
                        "keep-glm52-h1g-boundary"
                    ),
                    "Policies": [
                        {
                            "PolicyName": "DecisionInline",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [],
                            },
                        }
                    ],
                    "Tags": [
                        {"Key": "campaign-run-id", "Value": RUN_ID},
                        {"Key": "glm52-role", "Value": "decision"},
                    ],
                },
            },
            "CombinedHostProfile": {
                "Type": "AWS::IAM::InstanceProfile",
                "Properties": {"Roles": [{"Ref": "DecisionRole"}]},
            },
            "CombinedHostSecurityGroup": {
                "Type": "AWS::EC2::SecurityGroup",
                "Properties": {"GroupDescription": "fixture"},
            },
            "DecisionFunction": {
                "Type": "AWS::Lambda::Function",
                "Properties": {
                    "Code": {
                        "S3Bucket": "keep-glm52-code",
                        "S3Key": "h1g/decision.zip",
                        "S3ObjectVersion": "decision-version-1",
                    },
                    "DeadLetterConfig": {
                        "TargetArn": (
                            f"arn:aws:sqs:{REGION}:{ACCOUNT_ID}:"
                            "keep-glm52-h1g-support-dlq"
                        )
                    },
                    "Environment": {
                        "Variables": {
                            "GLM52_ACTIVATION_ID": "act-20260729-0001",
                        }
                    },
                    "Handler": "support_decision_handler.main",
                    "Layers": [
                        (
                            f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:"
                            "layer:keep-glm52-cryptography:3"
                        )
                    ],
                    "MemorySize": 1024,
                    "ReservedConcurrentExecutions": 1,
                    "Role": {"Fn::GetAtt": ["DecisionRole", "Arn"]},
                    "Runtime": "python3.12",
                    "Timeout": 840,
                    "VpcConfig": {
                        "SecurityGroupIds": ["sg-00000000000000001"],
                        "SubnetIds": ["subnet-00000000000000001"],
                    },
                },
            },
            "DecisionVersion": {
                "Type": "AWS::Lambda::Version",
                "Properties": {
                    "Description": (
                        "code="
                        + canonical_sha256(
                            {
                                "S3Bucket": "keep-glm52-code",
                                "S3Key": "h1g/decision.zip",
                                "S3ObjectVersion": "decision-version-1",
                            }
                        )
                    ),
                    "FunctionName": {"Ref": "DecisionFunction"},
                },
            },
            "DecisionInvokePermission": {
                "Type": "AWS::Lambda::Permission",
                "Properties": {
                    "Action": "lambda:InvokeFunction",
                    "FunctionName": {"Ref": "DecisionVersion"},
                    "Principal": "events.amazonaws.com",
                    "SourceAccount": ACCOUNT_ID,
                    "SourceArn": {
                        "Fn::GetAtt": ["DecisionRule", "Arn"]
                    },
                },
            },
            "DecisionEventSourceMapping": {
                "Type": "AWS::Lambda::EventSourceMapping",
                "Properties": {
                    "BatchSize": 10,
                    "Enabled": True,
                    "EventSourceArn": (
                        f"arn:aws:sqs:{REGION}:{ACCOUNT_ID}:"
                        "keep-glm52-h1g-support-source"
                    ),
                    "FunctionName": {"Ref": "DecisionVersion"},
                },
            },
            "RehearsalBucket": {
                "Type": "AWS::S3::Bucket",
                "Properties": {},
            },
            "DecisionRule": {
                "Type": "AWS::Events::Rule",
                "Properties": {"State": "ENABLED", "Targets": []},
            },
            "DeadlineSchedule": {
                "Type": "AWS::Scheduler::Schedule",
                "Properties": {
                    "State": "ENABLED",
                    "Target": {
                        "Arn": {"Ref": "DecisionVersion"},
                        "RoleArn": {"Fn::GetAtt": ["DecisionRole", "Arn"]},
                    },
                },
            },
            "SupportDlq": {
                "Type": "AWS::SQS::Queue",
                "Properties": {"RedrivePolicy": None},
            },
            "AlertSubscription": {
                "Type": "AWS::SNS::Subscription",
                "Properties": {
                    "Protocol": "email",
                    "TopicArn": (
                        f"arn:aws:sns:{REGION}:{ACCOUNT_ID}:"
                        "keep-glm52-h1g-alerts"
                    ),
                },
            },
            "DecisionAlarm": {
                "Type": "AWS::CloudWatch::Alarm",
                "Properties": {"TreatMissingData": "breaching"},
            },
            "DecisionLogs": {
                "Type": "AWS::Logs::LogGroup",
                "Properties": {
                    "RetentionInDays": 30,
                    "KmsKeyId": (
                        f"arn:aws:kms:{REGION}:{ACCOUNT_ID}:"
                        "key/12345678-1234-4234-8234-1234567890ab"
                    ),
                },
            },
        }
    }
    retained_template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "RetainedBucket": {
                "Type": "AWS::S3::Bucket",
                "Properties": {},
            }
        },
    }
    fence_template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "FencePolicy": {
                "Type": "AWS::S3::BucketPolicy",
                "Properties": {},
            }
        },
    }
    retained_stack_id = expected_items["cloudformation"][0]["stack_id"]
    fence_stack_id = (
        f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:"
        "stack/keep-glm52-h1g-fence/fence-uuid"
    )
    support_stack_id = (
        f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:"
        "stack/keep-glm52-h1g-support/support-uuid"
    )
    deployment_role_id = "AROAEXACTRETAINEDROLEID"
    stack_tags = {
        "Authority": "H1g",
        "Campaign": "GLM-5.2",
        "Environment": "production",
        "ManagedBy": "CloudFormation",
        "Project": "KEEP",
        "RunId": RUN_ID,
    }
    task6 = {
        "schema_version": 1,
        "record_type": "glm52_h1g_stack_migration_manifest_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "bucket_name": expected_items["s3"][0]["bucket"],
        "retained_deployment_role_id": deployment_role_id,
        "stacks": [
            {
                "kind": "retained",
                "name": "keep-glm52-h1g-retained",
                "stack_id": retained_stack_id,
                "termination_protection": True,
                "tags": deepcopy(stack_tags),
            },
            {
                "kind": "fence",
                "name": "keep-glm52-h1g-fence",
                "stack_id": fence_stack_id,
                "termination_protection": True,
                "tags": deepcopy(stack_tags),
            },
            {
                "kind": "support",
                "name": "keep-glm52-h1g-support",
                "stack_id": support_stack_id,
                "termination_protection": True,
                "tags": deepcopy(stack_tags),
            },
        ],
        "artifacts": [
            {
                "stage": "post-retain",
                "stack_id": retained_stack_id,
                "template_body_sha256": canonical_sha256(
                    retained_template
                ),
            },
            {
                "stage": "final-fence",
                "stack_id": fence_stack_id,
                "template_body_sha256": canonical_sha256(
                    fence_template
                ),
            },
        ],
    }
    source_base = _expected_state()
    source_specs = tuple(
        replace(
            spec,
            parameters={
                **dict(spec.parameters),
                "cli_plan": _safe_cli_plan(module, spec),
            },
        )
        for spec in source_base.specs
    )
    source_expected = build_h1d_expected_state(
        account_id=source_base.account_id,
        region=source_base.region,
        run_id=source_base.run_id,
        activation_id=source_base.activation_id,
        manifest_identity_sha256=source_base.manifest_identity_sha256,
        support_host_instance_id=source_base.support_host_instance_id,
        must_start_by=source_base.must_start_by,
        execution_deadline=source_base.execution_deadline,
        specs=source_specs,
    )
    source_specs_identity = module.canonical_sha256(
        tuple(
            {
                "family": spec.family,
                "operation": spec.operation,
                "parameters": dict(spec.parameters),
                "identity_field": spec.identity_field,
                "expected_items": tuple(
                    dict(item) for item in spec.expected_items
                ),
            }
            for spec in source_expected.specs
        )
    )
    templates_body = {
        "schema_version": 1,
        "record_type": "glm52_h1d_task6_task7_template_bundle_v1",
        "task6_templates": {
            "post-retain": retained_template,
            "final-fence": fence_template,
        },
        "task7_support_template": support_template,
        "h1d_specs_identity_sha256": source_specs_identity,
    }
    templates = {
        **templates_body,
        "canonical_body_sha256": module.canonical_sha256(templates_body),
    }
    resource_rows = [
        {
            "logical_id": logical_id,
            "resource_type": resource_type,
            "physical_id": expected_items[family][0][identity_field],
        }
        for family, logical_id, resource_type, identity_field in (
            (
                "lambda",
                "DecisionVersion",
                "AWS::Lambda::Version",
                "function_arn",
            ),
            ("iam", "DecisionRole", "AWS::IAM::Role", "role_arn"),
            (
                "eventbridge",
                "DecisionRule",
                "AWS::Events::Rule",
                "rule_arn",
            ),
            (
                "scheduler",
                "DeadlineSchedule",
                "AWS::Scheduler::Schedule",
                "schedule_arn",
            ),
            ("sqs", "SupportDlq", "AWS::SQS::Queue", "queue_arn"),
            (
                "sns",
                "AlertSubscription",
                "AWS::SNS::Subscription",
                "subscription_arn",
            ),
            ("ec2", "CombinedHost", "AWS::EC2::Instance", "instance_id"),
            ("s3", "ModelBucket", "AWS::S3::Bucket", "bucket"),
            (
                "cloudwatch",
                "DecisionAlarm",
                "AWS::CloudWatch::Alarm",
                "alarm_arn",
            ),
            ("logs", "DecisionLogs", "AWS::Logs::LogGroup", "log_group_arn"),
        )
    ]
    resource_rows.extend(
        (
            {
                "logical_id": "DecisionFunction",
                "resource_type": "AWS::Lambda::Function",
                "physical_id": "keep-glm52-h1g-support-decision",
            },
            {
                "logical_id": "DecisionInvokePermission",
                "resource_type": "AWS::Lambda::Permission",
                "physical_id": "DecisionInvokePermission-a1b2c3d4",
            },
            {
                "logical_id": "DecisionEventSourceMapping",
                "resource_type": "AWS::Lambda::EventSourceMapping",
                "physical_id": "12345678-1234-4234-8234-1234567890ab",
            },
            {
                "logical_id": "CombinedHostProfile",
                "resource_type": "AWS::IAM::InstanceProfile",
                "physical_id": "keep-glm52-h1g-support-combined-host",
            },
            {
                "logical_id": "CombinedHostSecurityGroup",
                "resource_type": "AWS::EC2::SecurityGroup",
                "physical_id": "sg-00000000000000001",
            },
        )
    )
    next(
        row
        for row in resource_rows
        if row["logical_id"] == "DecisionRole"
    )["physical_id"] = "keep-glm52-h1g-support-decision"
    next(
        row
        for row in resource_rows
        if row["resource_type"] == "AWS::SQS::Queue"
    )["physical_id"] = (
        "https://sqs.us-west-2.amazonaws.com/246813579024/"
        "keep-glm52-h1g-support-dlq"
    )
    rehearsal_row = next(
        row
        for row in resource_rows
        if row["resource_type"] == "AWS::S3::Bucket"
    )
    rehearsal_row["logical_id"] = "RehearsalBucket"
    rehearsal_row["physical_id"] = expected_items["s3"][1]["bucket"]
    inventory_body = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_postcreate_inventory_v1",
        "stack_resources": resource_rows,
        "support_template_body_sha256": module.canonical_sha256(
            support_template
        ),
    }
    inventory = {
        **inventory_body,
        "canonical_body_sha256": module.canonical_sha256(inventory_body),
    }
    postcreate_body = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_postcreate_manifest_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": "act-20260729-0001",
        "support_template_body_sha256": module.canonical_sha256(
            support_template
        ),
        "support_postcreate_inventory_sha256": inventory[
            "canonical_body_sha256"
        ],
    }
    postcreate = {
        **postcreate_body,
        "canonical_body_sha256": module.canonical_sha256(postcreate_body),
    }
    payloads = {
        "task6_manifest": task6,
        "task6_templates": templates,
        "task7_postcreate_manifest": postcreate,
        "task7_inventory": inventory,
    }

    class Store:
        def __init__(self) -> None:
            self.exact_versions: dict[str, str] = {}
            self.raw_by_key = {
                f"authority/{name}.json": (
                    module.canonical_json_bytes(payload) + b"\n"
                )
                for name, payload in payloads.items()
            }

        def get_object(self, *, key: str):
            raw = self.raw_by_key[key]
            return module.SpendObject(
                key=key,
                raw=raw,
                version_id=self.exact_versions[key],
                etag=f'"{hashlib.md5(raw).hexdigest()}"',
                checksum_sha256=hashlib.sha256(raw).hexdigest(),
                request_id=f"get-{key}",
                observed_at="2026-07-29T01:00:00Z",
            )

    store = Store()
    sources = {
        name: {
            "key": f"authority/{name}.json",
            "version_id": f"{name}-version",
            "file_sha256": hashlib.sha256(
                module.canonical_json_bytes(payload) + b"\n"
            ).hexdigest(),
        }
        for name, payload in payloads.items()
    }
    expected = build_h1d_expected_state(
        account_id=source_expected.account_id,
        region=source_expected.region,
        run_id=source_expected.run_id,
        activation_id=source_expected.activation_id,
        manifest_identity_sha256=postcreate["canonical_body_sha256"],
        support_host_instance_id=source_expected.support_host_instance_id,
        must_start_by=source_expected.must_start_by,
        execution_deadline=source_expected.execution_deadline,
        specs=source_expected.specs,
    )
    source_contract_sha256 = module._trusted_source_contract_sha256(
        task6=task6,
        templates=templates,
        postcreate=postcreate,
        inventory=inventory,
    )
    sealed_ddb_items = tuple(
        dict(item)
        for item in next(
            spec
            for spec in expected.specs
            if spec.family == "dynamodb"
        ).expected_items
    )
    task6_file_sha256 = hashlib.sha256(
        store.raw_by_key["authority/task6_manifest.json"]
    ).hexdigest()
    deployment_role_arn = (
        f"arn:aws:iam::{ACCOUNT_ID}:role/"
        "keep-glm52-h1g-cloudformation-deployment"
    )
    support_stack_rows = sorted(
        (
            {
                "logical_id": row["logical_id"],
                "resource_type": row["resource_type"],
                "physical_id": row["physical_id"],
                "resource_status": "CREATE_COMPLETE",
            }
            for row in resource_rows
        ),
        key=lambda row: row["logical_id"],
    )
    sealed_cloudformation_items = tuple(
        sorted(
            (
                {
                    "stack_id": retained_stack_id,
                    "status": "UPDATE_COMPLETE",
                    "template_sha256": canonical_sha256(
                        retained_template
                    ),
                    "parameters_sha256": canonical_sha256([]),
                    "resources_sha256": canonical_sha256(
                        [
                            {
                                "logical_id": "RetainedBucket",
                                "resource_type": "AWS::S3::Bucket",
                                "physical_id": (
                                    "keep-glm52-models-"
                                    "246813579024-us-west-2"
                                ),
                                "resource_status": "UPDATE_COMPLETE",
                            }
                        ]
                    ),
                    "stack_tags": [
                        {"key": key, "value": value}
                        for key, value in sorted(stack_tags.items())
                    ],
                    "service_role_arn": deployment_role_arn,
                    "termination_protection": True,
                    "manifest_identity_sha256": task6_file_sha256,
                },
                {
                    "stack_id": fence_stack_id,
                    "status": "UPDATE_COMPLETE",
                    "template_sha256": canonical_sha256(fence_template),
                    "parameters_sha256": canonical_sha256([]),
                    "resources_sha256": canonical_sha256(
                        [
                            {
                                "logical_id": "FencePolicy",
                                "resource_type": "AWS::S3::BucketPolicy",
                                "physical_id": (
                                    "keep-glm52-models-"
                                    "246813579024-us-west-2"
                                ),
                                "resource_status": "UPDATE_COMPLETE",
                            }
                        ]
                    ),
                    "stack_tags": [
                        {"key": key, "value": value}
                        for key, value in sorted(stack_tags.items())
                    ],
                    "service_role_arn": deployment_role_arn,
                    "termination_protection": True,
                    "manifest_identity_sha256": task6_file_sha256,
                },
                {
                    "stack_id": support_stack_id,
                    "status": "UPDATE_COMPLETE",
                    "template_sha256": canonical_sha256(
                        support_template
                    ),
                    "parameters_sha256": canonical_sha256([]),
                    "resources_sha256": canonical_sha256(
                        support_stack_rows
                    ),
                    "stack_tags": [
                        {"key": key, "value": value}
                        for key, value in sorted(stack_tags.items())
                    ],
                    "service_role_arn": deployment_role_arn,
                    "termination_protection": True,
                    "manifest_identity_sha256": task6_file_sha256,
                },
            ),
            key=lambda item: item["stack_id"],
        )
    )

    class TrustedContract:
        def read_contract(self) -> dict[str, object]:
            return {
                "authority_sources": deepcopy(sources),
                "source_contract_sha256": source_contract_sha256,
                "dynamodb_expected_items": sealed_ddb_items,
                "cloudformation_expected_items": (
                    sealed_cloudformation_items
                ),
                "cloudformation_deployment_role": {
                    "role_arn": deployment_role_arn,
                    "role_id": deployment_role_id,
                    "request_id": "iam-get-role-request",
                },
                "request_id": "ddb-trusted-source-fixture",
            }

    derived_items = module.AwsExpectedStateAuthority._derive_expected_items(
        task6=task6,
        task6_file_sha256=task6_file_sha256,
        support_template=support_template,
        inventory=inventory,
        activation_id=expected.activation_id,
        sealed_dynamodb_expected_items=sealed_ddb_items,
        sealed_cloudformation_expected_items=(
            sealed_cloudformation_items
        ),
        sealed_cloudformation_deployment_role={
            "role_arn": deployment_role_arn,
            "role_id": deployment_role_id,
            "request_id": "iam-get-role-request",
        },
    )
    derived_specs = tuple(
        replace(
            spec,
            expected_items=derived_items[spec.family],
        )
        for spec in expected.specs
    )
    expected = build_h1d_expected_state(
        account_id=expected.account_id,
        region=expected.region,
        run_id=expected.run_id,
        activation_id=expected.activation_id,
        manifest_identity_sha256=expected.manifest_identity_sha256,
        support_host_instance_id=expected.support_host_instance_id,
        must_start_by=expected.must_start_by,
        execution_deadline=expected.execution_deadline,
        specs=derived_specs,
    )
    return module.AwsExpectedStateAuthority(
        store=store,
        sources=sources,
        trusted_source_reader=TrustedContract(),
        clock=lambda: NOW,
    ), expected


def _substitute_expected_item(
    expected: H1dExpectedState,
    *,
    family: str,
    field: str,
    substitution: object,
) -> H1dExpectedState:
    specs = []
    for spec in expected.specs:
        items = tuple(deepcopy(item) for item in spec.expected_items)
        if spec.family == family:
            items[0][field] = substitution
        specs.append(replace(spec, expected_items=items))
    return build_h1d_expected_state(
        account_id=expected.account_id,
        region=expected.region,
        run_id=expected.run_id,
        activation_id=expected.activation_id,
        manifest_identity_sha256=expected.manifest_identity_sha256,
        support_host_instance_id=expected.support_host_instance_id,
        must_start_by=expected.must_start_by,
        execution_deadline=expected.execution_deadline,
        specs=tuple(specs),
    )


def _trusted_source_reader_after_coordinate_update(
    module: object,
    authority: object,
    sources: Mapping[str, Mapping[str, object]],
) -> object:
    contract = deepcopy(
        authority.trusted_source_reader.read_contract()
    )
    documents = {
        label: json.loads(
            authority.store.raw_by_key[coordinate["key"]]
        )
        for label, coordinate in sources.items()
    }
    contract.update(
        {
            "authority_sources": deepcopy(sources),
            "source_contract_sha256": (
                module._trusted_source_contract_sha256(
                    task6=documents["task6_manifest"],
                    templates=documents["task6_templates"],
                    postcreate=documents[
                        "task7_postcreate_manifest"
                    ],
                    inventory=documents["task7_inventory"],
                )
            ),
        }
    )

    class TrustedContract:
        def read_contract(self) -> dict[str, object]:
            return deepcopy(contract)

    return TrustedContract()


def _aws_normalizer_sources(
    module: object,
    authority: object,
) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    templates = json.loads(
        authority.store.raw_by_key[
            authority.sources["task6_templates"]["key"]
        ]
    )
    inventory = json.loads(
        authority.store.raw_by_key[
            authority.sources["task7_inventory"]["key"]
        ]
    )
    support = templates["task7_support_template"]
    physical = {
        row["logical_id"]: row["physical_id"]
        for row in inventory["stack_resources"]
    }
    return support, inventory, physical


def _aws_normalizer_case(
    module: object,
    authority: object,
    spec: LiveReadSpec,
    item_index: int = 0,
) -> tuple[str, dict[str, object], tuple[dict[str, object], ...]]:
    expected = spec.expected_items[item_index]
    identity = str(expected[spec.identity_field])
    support, inventory, physical = _aws_normalizer_sources(
        module, authority
    )
    resources = support["Resources"]

    def detail(
        response: dict[str, object],
        *,
        items: list[object] | None = None,
    ) -> dict[str, object]:
        value: dict[str, object] = {"responses": [response]}
        if items is not None:
            value["items"] = items
        return value

    if spec.family == "cloudformation":
        templates = json.loads(
            authority.store.raw_by_key[
                authority.sources["task6_templates"]["key"]
            ]
        )
        if "fence" in identity:
            template = templates["task6_templates"]["final-fence"]
            rows = [
                {
                    "LogicalResourceId": "FencePolicy",
                    "ResourceType": "AWS::S3::BucketPolicy",
                    "PhysicalResourceId": (
                        "keep-glm52-models-246813579024-us-west-2"
                    ),
                    "ResourceStatus": "UPDATE_COMPLETE",
                }
            ]
        elif "support" in identity:
            template = templates["task7_support_template"]
            rows = [
                {
                    "LogicalResourceId": row["logical_id"],
                    "ResourceType": row["resource_type"],
                    "PhysicalResourceId": row["physical_id"],
                    "ResourceStatus": "CREATE_COMPLETE",
                }
                for row in inventory["stack_resources"]
            ]
        else:
            template = templates["task6_templates"]["post-retain"]
            rows = [
                {
                    "LogicalResourceId": "RetainedBucket",
                    "ResourceType": "AWS::S3::Bucket",
                    "PhysicalResourceId": (
                        "keep-glm52-models-246813579024-us-west-2"
                    ),
                    "ResourceStatus": "UPDATE_COMPLETE",
                }
            ]
        return identity, {
            "StackId": identity,
            "StackStatus": "UPDATE_COMPLETE",
            "Parameters": [],
            "Tags": [
                {"Key": tag["key"], "Value": tag["value"]}
                for tag in expected["stack_tags"]
            ],
            "RoleARN": expected["service_role_arn"],
            "EnableTerminationProtection": True,
        }, (
            detail({"TemplateBody": template}),
            detail(
                {"StackResourceSummaries": rows},
                items=rows,
            ),
        )
    if spec.family == "lambda":
        props = resources["DecisionFunction"]["Properties"]
        configuration = module.AwsExpectedStateAuthority._resolve_ref(
            {
                key: value
                for key, value in props.items()
                if key not in {"Code", "ReservedConcurrentExecutions"}
            },
            physical,
        )
        configuration["Role"] = (
            f"arn:aws:iam::{ACCOUNT_ID}:"
            "role/keep-glm52-h1g-support-decision"
        )
        configuration["Layers"] = [
            {
                "Arn": props["Layers"][0],
                "CodeSize": 123456,
                "SigningProfileVersionArn": (
                    f"arn:aws:signer:{REGION}:{ACCOUNT_ID}:"
                    "signing-profiles/fixture/abc123"
                ),
            }
        ]
        configuration["Environment"] = {
            **configuration["Environment"],
            "Error": {
                "ErrorCode": "KMSKeyAccessDenied",
                "Message": "historical service response noise",
            },
        }
        configuration["VpcConfig"] = {
            **configuration["VpcConfig"],
            "VpcId": "vpc-00000000000000001",
            "Ipv6AllowedForDualStack": False,
        }
        configuration.update(
            {
                "Architectures": ["x86_64"],
                "EphemeralStorage": {"Size": 512},
                "LoggingConfig": {
                    "LogFormat": "Text",
                    "LogGroup": (
                        "/aws/lambda/"
                        "keep-glm52-h1g-support-decision"
                    ),
                },
                "PackageType": "Zip",
                "SnapStart": {
                    "ApplyOn": "None",
                    "OptimizationStatus": "Off",
                },
            }
        )
        configuration["CodeSha256"] = base64.b64encode(
            bytes.fromhex(str(expected["code_sha256"]))
        ).decode("ascii")
        policy = {
            "Version": "2012-10-17",
            "Id": "default",
            "Statement": [
                {
                    "Sid": "DecisionInvokePermission-a1b2c3d4",
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "events.amazonaws.com",
                    },
                    "Action": "lambda:InvokeFunction",
                    "Resource": identity,
                    "Condition": {
                        "ArnLike": {
                            "AWS:SourceArn": (
                                f"arn:aws:events:{REGION}:{ACCOUNT_ID}:"
                                "rule/keep-glm52-h1g-support"
                            )
                        },
                        "StringEquals": {
                            "AWS:SourceAccount": ACCOUNT_ID,
                        },
                    },
                }
            ],
        }
        return identity, {
            "FunctionArn": identity,
            "FunctionName": "keep-glm52-h1g-support-decision",
            "Version": "7",
        }, (
            detail(
                {
                    "Configuration": configuration,
                    "Code": {
                        "RepositoryType": "S3",
                        "Location": "https://lambda-code.example/object",
                    },
                }
            ),
            detail({"Policy": json.dumps(policy)}),
            detail({"ReservedConcurrentExecutions": 1}),
            detail({"Versions": []}, items=[]),
            detail(
                {
                    "EventSourceMappings": [
                        {
                            "UUID": "12345678-1234-4234-8234-1234567890ab",
                            "BatchSize": 10,
                            "EventSourceArn": (
                                f"arn:aws:sqs:{REGION}:{ACCOUNT_ID}:"
                                "keep-glm52-h1g-support-source"
                            ),
                            "FunctionArn": identity,
                            "State": "Enabled",
                            "LastModified": "2026-07-29T01:00:00Z",
                            "MaximumBatchingWindowInSeconds": 0,
                        }
                    ]
                },
                items=[
                    {
                        "UUID": "12345678-1234-4234-8234-1234567890ab",
                        "BatchSize": 10,
                        "EventSourceArn": (
                            f"arn:aws:sqs:{REGION}:{ACCOUNT_ID}:"
                            "keep-glm52-h1g-support-source"
                        ),
                        "FunctionArn": identity,
                        "State": "Enabled",
                        "LastModified": "2026-07-29T01:00:00Z",
                        "MaximumBatchingWindowInSeconds": 0,
                    }
                ],
            ),
        )
    if spec.family == "iam":
        role_props = resources["DecisionRole"]["Properties"]
        policy = role_props["Policies"][0]
        inline = {
            "PolicyName": policy["PolicyName"],
            "PolicyDocument": policy["PolicyDocument"],
        }
        profiles = [
            {"Arn": arn}
            for arn in expected["instance_profile_arns"]
        ]
        return identity, {
            "Arn": identity,
            "RoleName": "keep-glm52-h1g-support-decision",
        }, (
            detail(
                {
                    "Role": {
                        "Arn": identity,
                        "AssumeRolePolicyDocument": role_props[
                            "AssumeRolePolicyDocument"
                        ],
                        "Path": role_props["Path"],
                        "PermissionsBoundary": {
                            "PermissionsBoundaryType": "Policy",
                            "PermissionsBoundaryArn": role_props[
                                "PermissionsBoundary"
                            ],
                        },
                        "Tags": deepcopy(role_props["Tags"]),
                    }
                }
            ),
            detail({"InstanceProfiles": profiles}, items=profiles),
            detail({"PolicyNames": [policy["PolicyName"]]}, items=[
                policy["PolicyName"]
            ]),
            {"responses": [inline]},
            detail({"AttachedPolicies": []}, items=[]),
            {"responses": []},
            {"responses": []},
        )
    if spec.family == "eventbridge":
        return identity, {
            "Arn": identity,
            "Name": "keep-glm52-h1g-support",
        }, (
            detail({"Arn": identity, "State": "ENABLED"}),
            detail({"Targets": []}, items=[]),
        )
    if spec.family == "scheduler":
        target = module.AwsExpectedStateAuthority._resolve_ref(
            resources["DeadlineSchedule"]["Properties"]["Target"],
            physical,
        )
        target["RoleArn"] = (
            f"arn:aws:iam::{ACCOUNT_ID}:role/"
            "keep-glm52-h1g-support-decision"
        )
        return identity, {
            "Arn": identity,
            "Name": "keep-glm52-h1g-deadline",
            "GroupName": "default",
        }, (detail({"State": "ENABLED", "Target": target}),)
    if spec.family == "sqs":
        return identity, {
            "QueueUrl": (
                f"https://sqs.{REGION}.amazonaws.com/{ACCOUNT_ID}/"
                "keep-glm52-h1g-support-dlq"
            )
        }, (
            detail(
                {
                    "Attributes": {
                        "QueueArn": identity,
                        "ApproximateNumberOfMessages": "0",
                    }
                }
            ),
        )
    if spec.family == "sns":
        return identity, {"SubscriptionArn": identity}, (
            detail(
                {
                    "Attributes": {
                        "Protocol": "email",
                        "PendingConfirmation": "false",
                    }
                }
            ),
        )
    if spec.family == "ec2":
        tags = [
            {"Key": key, "Value": value}
            for key, value in expected["tags"].items()
        ]
        inventory_item = {
            "InstanceId": identity,
            "InstanceType": expected["instance_type"],
            "State": {"Name": "running"},
            "ImageId": expected["ami_id"],
            "SubnetId": "subnet-00000000000000001",
            "SecurityGroups": [
                {"GroupId": "sg-00000000000000001"}
            ],
            "PrivateIpAddress": "10.20.101.10",
            "NetworkInterfaces": [],
            "IamInstanceProfile": {"Arn": expected["profile_arn"]},
            "Tags": tags,
            "BlockDeviceMappings": [
                {"Ebs": {"VolumeId": "vol-00000000000000001"}},
                {"Ebs": {"VolumeId": "vol-00000000000000002"}},
            ],
        }
        root = {
            "VolumeId": "vol-00000000000000001",
            "Size": 30,
            "VolumeType": "gp3",
            "Iops": 3000,
            "Throughput": 125,
            "Encrypted": True,
            "Attachments": [
                {
                    "Device": "/dev/xvda",
                    "InstanceId": identity,
                    "State": "attached",
                }
            ],
        }
        data = {
            "VolumeId": "vol-00000000000000002",
            "Size": 50,
            "VolumeType": "gp3",
            "Iops": 3000,
            "Throughput": 125,
            "Encrypted": True,
            "Attachments": [
                {
                    "Device": "/dev/sdf",
                    "InstanceId": identity,
                    "State": "attached",
                }
            ],
        }
        encoded = base64.b64encode(b"fixture-user-data").decode("ascii")
        return identity, inventory_item, (
            {"responses": [{"Volumes": [root]}, {"Volumes": [data]}]},
            {"responses": [{"Images": [{"ImageId": expected["ami_id"]}]}]},
            detail({"UserData": {"Value": encoded}}),
        )
    if spec.family == "s3":
        versioning = (
            {"Status": "Enabled"}
            if expected["versioning"] == "Enabled"
            else {}
        )
        return identity, {"Name": identity}, (
            detail(versioning),
            detail({"Policy": None}),
            detail({"Rules": None}),
            detail({"ReplicationConfiguration": None}),
        )
    if spec.family == "ssm":
        return identity, {
            "InstanceId": identity,
            "PingStatus": "Online",
            "PlatformType": "Linux",
        }, ()
    if spec.family == "cloudwatch":
        return identity, {
            "AlarmArn": identity,
            "AlarmName": "keep-glm52-h1g-decision",
            "StateValue": "OK",
            "TreatMissingData": "breaching",
            "Namespace": "AWS/Lambda",
            "MetricName": "Errors",
            "Dimensions": [],
            "Period": 60,
            "Statistic": "Sum",
        }, (
            detail(
                {
                    "MetricDataResults": [
                        {
                            "Id": "h1d",
                            "StatusCode": "Complete",
                            "Timestamps": ["2026-07-29T01:00:00Z"],
                            "Values": [0.0],
                        }
                    ]
                },
                items=[
                    {
                        "Id": "h1d",
                        "StatusCode": "Complete",
                        "Timestamps": ["2026-07-29T01:00:00Z"],
                        "Values": [0.0],
                    }
                ],
            ),
        )
    if spec.family == "logs":
        return identity, {
            "arn": identity,
            "retentionInDays": 30,
            "kmsKeyId": expected["kms_key_arn"],
        }, ()
    raise AssertionError(spec.family)


def _substitute_expected_item_at(
    expected: H1dExpectedState,
    *,
    family: str,
    item_index: int,
    field: str,
    substitution: object,
) -> H1dExpectedState:
    specs = []
    for spec in expected.specs:
        items = tuple(deepcopy(item) for item in spec.expected_items)
        if spec.family == family:
            items[item_index][field] = substitution
        specs.append(replace(spec, expected_items=items))
    return build_h1d_expected_state(
        account_id=expected.account_id,
        region=expected.region,
        run_id=expected.run_id,
        activation_id=expected.activation_id,
        manifest_identity_sha256=expected.manifest_identity_sha256,
        support_host_instance_id=expected.support_host_instance_id,
        must_start_by=expected.must_start_by,
        execution_deadline=expected.execution_deadline,
        specs=tuple(specs),
    )


def _semantic_mutation(value: object) -> object:
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        if len(value) == 64 and set(value) <= set("0123456789abcdef"):
            return ("0" if value[0] != "0" else "1") + value[1:]
        return f"{value}-mutant"
    if value is None:
        return "mutant"
    if type(value) is tuple:
        return (*value, "mutant")
    if type(value) is dict:
        return {**value, "mutant": "present"}
    raise AssertionError(f"no mutation for {type(value)!r}")


_EVERY_EXPECTED_ITEM_FIELD = tuple(
    (family, item_index, field)
    for family, items in _expected_items().items()
    for item_index, item in enumerate(items)
    for field in item
)


@pytest.mark.parametrize(
    ("family", "field"),
    (
        ("iam", "managed_policy_versions_sha256"),
        ("lambda", "configuration_sha256"),
        ("cloudformation", "parameters_sha256"),
        ("sqs", "redrive_policy_sha256"),
        ("s3", "policy_sha256"),
        ("logs", "retention_days"),
    ),
)
def test_rehashed_expected_config_is_not_independent_authority(
    family: str,
    field: str,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    original = next(
        spec.expected_items[0][field]
        for spec in expected.specs
        if spec.family == family
    )
    forged = _substitute_expected_item(
        expected,
        family=family,
        field=field,
        substitution=_semantic_mutation(original),
    )
    with pytest.raises(ValueError):
        authority.authenticate(forged)


def test_production_specs_use_explicit_family_aws_normalizers() -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    del authority
    assert {
        spec.family: spec.parameters["cli_plan"]
        for spec in expected.specs
    } == {
        family: {
            "schema_version": 3,
            "normalizer": f"{family}.aws_response_v1",
        }
        for family in REQUIRED_LIVE_FAMILIES
    }


@pytest.mark.parametrize(
    "family",
    tuple(
        family
        for family in REQUIRED_LIVE_FAMILIES
        if family != "dynamodb"
    ),
)
def test_each_explicit_aws_normalizer_matches_source_derived_item(
    family: str,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    spec = next(spec for spec in expected.specs if spec.family == family)
    item_count = len(spec.expected_items)
    normalized = []
    for item_index in range(item_count):
        identity, inventory_item, details = _aws_normalizer_case(
            module, authority, spec, item_index
        )
        normalized.append(
            module.AwsCliLiveReader._normalize_aws_item(
                spec=spec,
                identity=identity,
                inventory_item=inventory_item,
                details=details,
            )
        )
    assert tuple(
        sorted(normalized, key=lambda item: str(item[spec.identity_field]))
    ) == tuple(
        sorted(
            spec.expected_items,
            key=lambda item: str(item[spec.identity_field]),
        )
    )


def test_lambda_aws_native_defaults_and_permission_match_source_semantics() -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    spec = next(spec for spec in expected.specs if spec.family == "lambda")
    identity, inventory_item, details = _aws_normalizer_case(
        module, authority, spec
    )

    configuration = details[0]["responses"][0]["Configuration"]
    assert configuration["PackageType"] == "Zip"
    assert configuration["Architectures"] == ["x86_64"]
    assert configuration["VpcConfig"]["VpcId"].startswith("vpc-")
    assert json.loads(details[1]["responses"][0]["Policy"])["Id"] == "default"
    assert module.AwsCliLiveReader._normalize_aws_item(
        spec=spec,
        identity=identity,
        inventory_item=inventory_item,
        details=details,
    ) == spec.expected_items[0]


def test_lambda_code_owned_role_arn_mutation_exposes_drift() -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    spec = next(spec for spec in expected.specs if spec.family == "lambda")
    identity, inventory_item, details = _aws_normalizer_case(
        module, authority, spec
    )
    details = tuple(deepcopy(detail) for detail in details)
    details[0]["responses"][0]["Configuration"]["Role"] = (
        f"arn:aws:iam::{ACCOUNT_ID}:role/foreign"
    )

    assert module.AwsCliLiveReader._normalize_aws_item(
        spec=spec,
        identity=identity,
        inventory_item=inventory_item,
        details=details,
    ) != spec.expected_items[0]


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("Sid",), "foreign-sid"),
        (("Action",), "lambda:GetFunction"),
        (("Principal", "Service"), "sns.amazonaws.com"),
        (
            ("Resource",),
            (
                f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:"
                "function:keep-glm52-h1g-support-decision:8"
            ),
        ),
        (
            ("Condition", "ArnLike", "AWS:SourceArn"),
            f"arn:aws:events:{REGION}:{ACCOUNT_ID}:rule/foreign",
        ),
        (("Condition", "StringEquals", "AWS:SourceAccount"), "000000000000"),
    ),
)
def test_lambda_permission_aws_native_mutations_expose_drift(
    path: tuple[str, ...],
    value: object,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    spec = next(spec for spec in expected.specs if spec.family == "lambda")
    identity, inventory_item, details = _aws_normalizer_case(
        module, authority, spec
    )
    details = tuple(deepcopy(detail) for detail in details)
    policy = json.loads(details[1]["responses"][0]["Policy"])
    target = policy["Statement"][0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    details[1]["responses"][0]["Policy"] = json.dumps(policy)

    assert module.AwsCliLiveReader._normalize_aws_item(
        spec=spec,
        identity=identity,
        inventory_item=inventory_item,
        details=details,
    ) != spec.expected_items[0]


def test_named_cfn_iam_resources_derive_full_aws_arns() -> None:
    module = _load_script()
    _authority, expected = _production_expected_authority_fixture(module)
    by_family = {
        spec.family: spec.expected_items for spec in expected.specs
    }
    profile_arn = (
        f"arn:aws:iam::{ACCOUNT_ID}:instance-profile/"
        "keep-glm52-h1g-support-combined-host"
    )
    assert by_family["iam"][0]["instance_profile_arns"] == (profile_arn,)
    assert by_family["ec2"][0]["profile_arn"] == profile_arn
    assert by_family["lambda"][0]["configuration_fields"] == (
        "DeadLetterConfig",
        "Environment",
        "Handler",
        "Layers",
        "MemorySize",
        "Role",
        "Runtime",
        "Timeout",
        "VpcConfig",
    )


def test_lambda_permission_cfn_maps_to_exact_qualified_aws_statement() -> None:
    module = _load_script()
    _authority, expected = _production_expected_authority_fixture(module)
    lambda_item = next(
        spec.expected_items[0]
        for spec in expected.specs
        if spec.family == "lambda"
    )
    function_arn = (
        f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:"
        "function:keep-glm52-h1g-support-decision:7"
    )
    assert lambda_item["policy_sha256"] == canonical_sha256(
        [
            {
                "Sid": "DecisionInvokePermission-a1b2c3d4",
                "Effect": "Allow",
                "Principal": {"Service": "events.amazonaws.com"},
                "Action": "lambda:InvokeFunction",
                "Resource": function_arn,
                "Condition": {
                    "ArnLike": {
                        "AWS:SourceArn": (
                            f"arn:aws:events:{REGION}:{ACCOUNT_ID}:"
                            "rule/keep-glm52-h1g-support"
                        )
                    },
                    "StringEquals": {
                        "AWS:SourceAccount": ACCOUNT_ID,
                    },
                },
            }
        ]
    )


def test_scheduler_cfn_role_getatt_matches_aws_role_arn() -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    spec = next(spec for spec in expected.specs if spec.family == "scheduler")
    identity, inventory_item, details = _aws_normalizer_case(
        module, authority, spec
    )
    assert module.AwsCliLiveReader._normalize_aws_item(
        spec=spec,
        identity=identity,
        inventory_item=inventory_item,
        details=details,
    ) == spec.expected_items[0]


def test_iam_foreign_assume_role_policy_is_live_drift() -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    spec = next(spec for spec in expected.specs if spec.family == "iam")
    identity, inventory_item, details = _aws_normalizer_case(
        module, authority, spec
    )
    details = tuple(deepcopy(detail) for detail in details)
    details[0]["responses"][0]["Role"]["AssumeRolePolicyDocument"][
        "Statement"
    ][0]["Principal"]["Service"] = "ec2.amazonaws.com"
    assert module.AwsCliLiveReader._normalize_aws_item(
        spec=spec,
        identity=identity,
        inventory_item=inventory_item,
        details=details,
    ) != spec.expected_items[0]


def test_lambda_source_event_mapping_is_bound_and_foreign_mapping_drifts() -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    spec = next(spec for spec in expected.specs if spec.family == "lambda")
    assert "event_source_mappings_sha256" in spec.expected_items[0]
    identity, inventory_item, details = _aws_normalizer_case(
        module, authority, spec
    )
    normalized = module.AwsCliLiveReader._normalize_aws_item(
        spec=spec,
        identity=identity,
        inventory_item=inventory_item,
        details=details,
    )
    assert normalized == spec.expected_items[0]

    details = tuple(deepcopy(detail) for detail in details)
    details[4]["items"].append(
        {
            "UUID": "87654321-4321-4321-8321-ba0987654321",
            "BatchSize": 10,
            "EventSourceArn": (
                f"arn:aws:sqs:{REGION}:{ACCOUNT_ID}:foreign"
            ),
            "FunctionArn": identity,
            "State": "Enabled",
        }
    )
    with pytest.raises(ValueError, match="event-source"):
        module.AwsCliLiveReader._normalize_aws_item(
            spec=spec,
            identity=identity,
            inventory_item=inventory_item,
            details=details,
        )


@pytest.mark.parametrize("mutation", ("stack-tag", "resource-status"))
def test_cloudformation_normalizer_binds_tags_and_resource_statuses(
    mutation: str,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    spec = next(
        spec for spec in expected.specs
        if spec.family == "cloudformation"
    )
    identity, inventory_item, details = _aws_normalizer_case(
        module, authority, spec
    )
    inventory_item = deepcopy(inventory_item)
    details = tuple(deepcopy(detail) for detail in details)
    if mutation == "stack-tag":
        inventory_item["Tags"] = [
            {"Key": "Campaign", "Value": "attacker"},
        ]
    else:
        details[1]["items"][0]["ResourceStatus"] = "UPDATE_FAILED"
    assert module.AwsCliLiveReader._normalize_aws_item(
        spec=spec,
        identity=identity,
        inventory_item=inventory_item,
        details=details,
    ) != spec.expected_items[0]


@pytest.mark.parametrize(
    "family",
    tuple(
        family
        for family in REQUIRED_LIVE_FAMILIES
        if family != "dynamodb"
    ),
)
def test_each_explicit_aws_normalizer_exposes_family_drift(
    family: str,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    spec = next(spec for spec in expected.specs if spec.family == family)
    identity, inventory_item, details = _aws_normalizer_case(
        module, authority, spec
    )
    inventory_item = deepcopy(inventory_item)
    details = tuple(deepcopy(detail) for detail in details)
    if family == "cloudformation":
        inventory_item["StackStatus"] = "ROLLBACK_COMPLETE"
    elif family == "lambda":
        details[0]["responses"][0]["Configuration"]["Timeout"] = 841
    elif family == "iam":
        details[3]["responses"][0]["PolicyDocument"]["Statement"].append(
            {"Effect": "Allow", "Action": "*", "Resource": "*"}
        )
    elif family == "eventbridge":
        details[0]["responses"][0]["State"] = "DISABLED"
    elif family == "scheduler":
        details[0]["responses"][0]["State"] = "DISABLED"
    elif family == "sqs":
        details[0]["responses"][0]["Attributes"][
            "ApproximateNumberOfMessages"
        ] = "1"
    elif family == "sns":
        details[0]["responses"][0]["Attributes"][
            "PendingConfirmation"
        ] = "true"
    elif family == "ec2":
        inventory_item["State"]["Name"] = "stopped"
    elif family == "s3":
        details[0]["responses"][0]["Status"] = "Suspended"
    elif family == "ssm":
        inventory_item["PingStatus"] = "ConnectionLost"
    elif family == "cloudwatch":
        inventory_item["StateValue"] = "ALARM"
    elif family == "logs":
        inventory_item["retentionInDays"] = 31
    else:
        raise AssertionError(family)
    normalized = module.AwsCliLiveReader._normalize_aws_item(
        spec=spec,
        identity=identity,
        inventory_item=inventory_item,
        details=details,
    )
    assert normalized != spec.expected_items[0]


def test_full_core_uses_realistic_aws_shapes_for_every_live_family() -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    specs = {spec.family: spec for spec in expected.specs}
    cases = {
        family: [
            _aws_normalizer_case(module, authority, specs[family], index)
            for index in range(len(specs[family].expected_items))
        ]
        for family in REQUIRED_LIVE_FAMILIES
        if family != "dynamodb"
    }
    _support, inventory_source, _physical = _aws_normalizer_sources(
        module, authority
    )

    class Runner:
        def __init__(self) -> None:
            self.calls = 0

        def _response(self, value: dict[str, object]) -> dict[str, object]:
            self.calls += 1
            return {
                **deepcopy(value),
                "ResponseMetadata": {
                    "RequestId": f"aws-realistic-{self.calls}"
                },
            }

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            command = operation[:2]
            if command == ("cloudformation", "describe-stacks"):
                return self._response(
                    {
                        "Stacks": [
                            case[1]
                            for case in cases["cloudformation"]
                        ]
                    }
                )
            if command == ("cloudformation", "get-template"):
                stack_id = operation[
                    operation.index("--stack-name") + 1
                ]
                case = next(
                    case
                    for case in cases["cloudformation"]
                    if case[0] == stack_id
                )
                return self._response(
                    case[2][0]["responses"][0]
                )
            if command == (
                "cloudformation",
                "list-stack-resources",
            ):
                stack_id = operation[
                    operation.index("--stack-name") + 1
                ]
                case = next(
                    case
                    for case in cases["cloudformation"]
                    if case[0] == stack_id
                )
                return self._response(
                    case[2][1]["responses"][0]
                )
            if command == ("lambda", "list-functions"):
                version = cases["lambda"][0][1]
                base_arn = version["FunctionArn"].rsplit(":", 1)[0]
                return self._response(
                    {
                        "Functions": [
                            {
                                "FunctionArn": base_arn,
                                "FunctionName": version["FunctionName"],
                            }
                        ]
                    }
                )
            if command == ("lambda", "list-versions-by-function"):
                version = cases["lambda"][0][1]
                base_arn = version["FunctionArn"].rsplit(":", 1)[0]
                return self._response(
                    {
                        "Versions": [
                            {
                                "FunctionArn": base_arn,
                                "FunctionName": version["FunctionName"],
                                "Version": "$LATEST",
                            },
                            version,
                        ]
                    }
                )
            if command == ("lambda", "get-function"):
                return self._response(
                    cases["lambda"][0][2][0]["responses"][0]
                )
            if command == ("lambda", "get-policy"):
                return self._response(
                    cases["lambda"][0][2][1]["responses"][0]
                )
            if command == ("lambda", "get-function-concurrency"):
                return self._response(
                    cases["lambda"][0][2][2]["responses"][0]
                )
            if command == ("lambda", "list-event-source-mappings"):
                return self._response(
                    cases["lambda"][0][2][4]["responses"][0]
                )
            if command == ("iam", "list-roles"):
                return self._response({"Roles": [cases["iam"][0][1]]})
            if command == ("iam", "get-role"):
                return self._response(
                    cases["iam"][0][2][0]["responses"][0]
                )
            if command == ("iam", "list-instance-profiles-for-role"):
                return self._response(
                    cases["iam"][0][2][1]["responses"][0]
                )
            if command == ("iam", "list-role-policies"):
                return self._response(
                    cases["iam"][0][2][2]["responses"][0]
                )
            if command == ("iam", "get-role-policy"):
                return self._response(
                    cases["iam"][0][2][3]["responses"][0]
                )
            if command == ("iam", "list-attached-role-policies"):
                return self._response({"AttachedPolicies": []})
            if command == ("events", "list-rules"):
                prefix = operation[operation.index("--name-prefix") + 1]
                return self._response(
                    {
                        "Rules": (
                            [cases["eventbridge"][0][1]]
                            if prefix == "keep-glm52-h1g"
                            else []
                        )
                    }
                )
            if command == ("events", "describe-rule"):
                return self._response(
                    cases["eventbridge"][0][2][0]["responses"][0]
                )
            if command == ("events", "list-targets-by-rule"):
                return self._response({"Targets": []})
            if command == ("scheduler", "list-schedules"):
                prefix = operation[operation.index("--name-prefix") + 1]
                return self._response(
                    {
                        "Schedules": (
                            [cases["scheduler"][0][1]]
                            if prefix == "keep-glm52-h1g"
                            else []
                        )
                    }
                )
            if command == ("scheduler", "get-schedule"):
                return self._response(
                    cases["scheduler"][0][2][0]["responses"][0]
                )
            if command == ("sqs", "list-queues"):
                prefix = operation[
                    operation.index("--queue-name-prefix") + 1
                ]
                return self._response(
                    {
                        "QueueUrls": (
                            [cases["sqs"][0][1]["QueueUrl"]]
                            if prefix == "keep-glm52-h1g"
                            else []
                        )
                    }
                )
            if command == ("sqs", "get-queue-attributes"):
                return self._response(
                    cases["sqs"][0][2][0]["responses"][0]
                )
            if command == ("sns", "list-subscriptions-by-topic"):
                return self._response(
                    {"Subscriptions": [cases["sns"][0][1]]}
                )
            if command == ("sns", "get-subscription-attributes"):
                return self._response(
                    cases["sns"][0][2][0]["responses"][0]
                )
            if command == ("ec2", "describe-instances"):
                instance = deepcopy(cases["ec2"][0][1])
                instance["BlockDeviceMappings"] = [
                    {"Ebs": {"VolumeId": "vol-00000000000000001"}},
                    {"Ebs": {"VolumeId": "vol-00000000000000002"}},
                ]
                return self._response({"Instances": [instance]})
            if command == ("ec2", "describe-volumes"):
                volume_id = operation[operation.index("--volume-ids") + 1]
                volumes = [
                    response["Volumes"][0]
                    for response in cases["ec2"][0][2][0]["responses"]
                    if response["Volumes"][0]["VolumeId"] == volume_id
                ]
                return self._response({"Volumes": volumes})
            if command == ("ec2", "describe-images"):
                return self._response(
                    cases["ec2"][0][2][1]["responses"][0]
                )
            if command == ("ec2", "describe-instance-attribute"):
                return self._response(
                    cases["ec2"][0][2][2]["responses"][0]
                )
            if command == ("s3api", "list-buckets"):
                return self._response(
                    {
                        "Buckets": [
                            case[1] for case in cases["s3"]
                        ]
                    }
                )
            if command[0] == "s3api" and command[1].startswith(
                "get-bucket-"
            ):
                bucket = operation[operation.index("--bucket") + 1]
                case = next(
                    case for case in cases["s3"]
                    if case[0] == bucket
                )
                index = {
                    "get-bucket-versioning": 0,
                    "get-bucket-policy": 1,
                    "get-bucket-lifecycle-configuration": 2,
                    "get-bucket-replication": 3,
                }[command[1]]
                return self._response(
                    case[2][index]["responses"][0]
                )
            if command == ("dynamodb", "query"):
                return self._response(
                    {"Items": list(_raw_activation_items())}
                )
            if command == ("ssm", "describe-instance-information"):
                return self._response(
                    {
                        "InstanceInformationList": [
                            cases["ssm"][0][1]
                        ]
                    }
                )
            if command == ("cloudwatch", "describe-alarms"):
                prefix = operation[
                    operation.index("--alarm-name-prefix") + 1
                ]
                return self._response(
                    {
                        "MetricAlarms": (
                            [cases["cloudwatch"][0][1]]
                            if prefix == "keep-glm52-h1g"
                            else []
                        )
                    }
                )
            if command == ("cloudwatch", "get-metric-data"):
                return self._response(
                    cases["cloudwatch"][0][2][0]["responses"][0]
                )
            if command == ("logs", "describe-log-groups"):
                prefix = operation[
                    operation.index("--log-group-name-prefix") + 1
                ]
                return self._response(
                    {
                        "logGroups": (
                            [cases["logs"][0][1]]
                            if prefix
                            == "/aws/lambda/keep-glm52-h1g"
                            else []
                        )
                    }
                )
            raise AssertionError(operation)

    base_services, _reader, spend, probe = _services(expected)
    services = H1dLiveServices(
        identity=base_services.identity,
        expected_state_authority=authority,
        reader=module.AwsSdkLiveReader(
            runner=Runner(),
            clock=lambda: NOW,
        ),
        spend=spend,
        sky_relay_probe=probe,
        clock=_Clock(),
    )
    result = inspect_h1d_live_authority(
        _request(expected), expected, services
    )
    assert result.run_id == RUN_ID
    assert set(result.family_identities) == set(REQUIRED_LIVE_FAMILIES)


def test_ec2_normalizer_rejects_third_attached_ebs_volume() -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    spec = next(spec for spec in expected.specs if spec.family == "ec2")
    identity, inventory_item, details = _aws_normalizer_case(
        module, authority, spec
    )
    inventory_item["BlockDeviceMappings"] = [
        {"Ebs": {"VolumeId": "vol-00000000000000001"}},
        {"Ebs": {"VolumeId": "vol-00000000000000002"}},
        {"Ebs": {"VolumeId": "vol-00000000000000003"}},
    ]
    details[0]["responses"].append(
        {
            "Volumes": [
                {
                    "VolumeId": "vol-00000000000000003",
                    "Size": 1,
                    "VolumeType": "gp3",
                    "Iops": 3000,
                    "Throughput": 125,
                    "Encrypted": True,
                    "Attachments": [
                        {
                            "Device": "/dev/sdg",
                            "InstanceId": identity,
                            "State": "attached",
                        }
                    ],
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="volume|cardinality|extra"):
        module.AwsCliLiveReader._normalize_aws_item(
            spec=spec,
            identity=identity,
            inventory_item=inventory_item,
            details=details,
        )


@pytest.mark.parametrize(
    ("timestamps", "values"),
    (
        ([], []),
        (["2026-07-29T01:00:00Z"], []),
    ),
)
def test_cloudwatch_complete_status_requires_aligned_nonempty_evidence(
    timestamps: list[str],
    values: list[float],
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    spec = next(
        spec for spec in expected.specs
        if spec.family == "cloudwatch"
    )
    identity, inventory_item, details = _aws_normalizer_case(
        module, authority, spec
    )
    metric = details[0]["items"][0]
    metric["Timestamps"] = timestamps
    metric["Values"] = values
    with pytest.raises(ValueError, match="metric.*evidence|aligned|coverage"):
        module.AwsCliLiveReader._normalize_aws_item(
            spec=spec,
            identity=identity,
            inventory_item=inventory_item,
            details=details,
        )


@pytest.mark.parametrize(
    ("family", "item_index", "field"),
    _EVERY_EXPECTED_ITEM_FIELD,
)
def test_every_expected_item_field_is_independently_derived(
    family: str,
    item_index: int,
    field: str,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    spec = next(spec for spec in expected.specs if spec.family == family)
    original = spec.expected_items[item_index][field]
    forged = _substitute_expected_item_at(
        expected,
        family=family,
        item_index=item_index,
        field=field,
        substitution=_semantic_mutation(original),
    )
    with pytest.raises(ValueError):
        authority.authenticate(forged)


@pytest.mark.parametrize(
    ("must_start_by", "execution_deadline"),
    (
        ("2099-01-01T00:00:00Z", "2099-01-02T00:00:00Z"),
        ("2026-07-29T02:00:00Z", "2099-01-02T00:00:00Z"),
    ),
)
def test_expected_authority_rejects_caller_extended_deadlines(
    must_start_by: str,
    execution_deadline: str,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    bounded = module.AwsExpectedStateAuthority(
        store=authority.store,
        sources=authority.sources,
        trusted_sources=authority.sources,
        clock=lambda: NOW,
    )
    forged = build_h1d_expected_state(
        account_id=expected.account_id,
        region=expected.region,
        run_id=expected.run_id,
        activation_id=expected.activation_id,
        manifest_identity_sha256=expected.manifest_identity_sha256,
        support_host_instance_id=expected.support_host_instance_id,
        must_start_by=must_start_by,
        execution_deadline=execution_deadline,
        specs=expected.specs,
    )
    with pytest.raises(ValueError, match="deadline"):
        bounded.authenticate(forged)


@pytest.mark.parametrize(
    ("family", "field", "substitution", "message"),
    (
        ("ec2", "instance_type", "c6a.2xlarge", "EC2 support shape"),
        (
            "iam",
            "role_arn",
            (
                "arn:aws:iam::246813579024:"
                "role/keep-glm52-h1g-attacker-substitution"
            ),
            "expected identities",
        ),
    ),
)
def test_real_expected_authority_rejects_rehashed_caller_substitution(
    family: str,
    field: str,
    substitution: str,
    message: str,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    authenticated = authority.authenticate(expected)
    assert (
        authenticated.expected_state_identity_sha256
        == expected.canonical_identity_sha256
    )

    substituted = _substitute_expected_item(
        expected,
        family=family,
        field=field,
        substitution=substitution,
    )
    assert substituted.canonical_identity_sha256 != expected.canonical_identity_sha256
    with pytest.raises(ValueError, match=message):
        authority.authenticate(substituted)


@pytest.mark.parametrize(
    ("family", "field", "substitution"),
    (
        ("iam", "inline_policy_sha256", "0" * 64),
        ("lambda", "code_sha256", "0" * 64),
        ("ec2", "network_sha256", "1" * 64),
    ),
)
def test_real_expected_authority_binds_every_immutable_config_field(
    family: str,
    field: str,
    substitution: str,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    substituted = _substitute_expected_item(
        expected,
        family=family,
        field=field,
        substitution=substitution,
    )
    with pytest.raises(ValueError, match="source-bound"):
        authority.authenticate(substituted)


def test_real_expected_authority_rejects_caller_controlled_ec2_query_plan() -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    specs = []
    for spec in expected.specs:
        parameters = dict(spec.parameters)
        if spec.family == "ec2":
            parameters["cli_plan"] = {
                "inventory_command_index": 0,
                "inventory_arguments": [
                    "--instance-ids",
                    expected.support_host_instance_id,
                ],
                "items_path": ["Reservations"],
                "next_token_path": ["NextToken"],
                "next_token_argument": "--next-token",
                "identity_path": ["InstanceId"],
                "detail_reads": [],
                "projections": {},
            }
        specs.append(replace(spec, parameters=parameters))
    substituted = build_h1d_expected_state(
        account_id=expected.account_id,
        region=expected.region,
        run_id=expected.run_id,
        activation_id=expected.activation_id,
        manifest_identity_sha256=expected.manifest_identity_sha256,
        support_host_instance_id=expected.support_host_instance_id,
        must_start_by=expected.must_start_by,
        execution_deadline=expected.execution_deadline,
        specs=tuple(specs),
    )
    with pytest.raises(ValueError, match="EC2.*plan"):
        authority.authenticate(substituted)


def _forge_iam_plan_and_bundle(
    module: object,
    authority: object,
    expected: H1dExpectedState,
    *,
    source_mutation: str,
    inventory_arguments: list[str] | None = None,
):
    specs = []
    for spec in expected.specs:
        parameters = dict(spec.parameters)
        if spec.family == "iam":
            parameters["cli_plan"] = {
                "inventory_command_index": 0,
                "inventory_arguments": (
                    [
                        "--path-prefix",
                        "/keep-glm52-h1g/",
                        "--max-items",
                        "1",
                    ]
                    if inventory_arguments is None
                    else inventory_arguments
                ),
                "items_path": ["Roles"],
                "next_token_path": ["Marker"],
                "next_token_argument": "--marker",
                "identity_path": ["Arn"],
                "detail_reads": [],
                "projections": {},
            }
        specs.append(replace(spec, parameters=parameters))
    forged_expected = build_h1d_expected_state(
        account_id=expected.account_id,
        region=expected.region,
        run_id=expected.run_id,
        activation_id=expected.activation_id,
        manifest_identity_sha256=expected.manifest_identity_sha256,
        support_host_instance_id=expected.support_host_instance_id,
        must_start_by=expected.must_start_by,
        execution_deadline=expected.execution_deadline,
        specs=tuple(specs),
    )
    original_sources = deepcopy(authority.sources)
    forged_sources = deepcopy(original_sources)
    original_coordinate = original_sources["task6_templates"]
    original_key = original_coordinate["key"]
    templates = json.loads(authority.store.raw_by_key[original_key])
    templates_body = dict(templates)
    templates_body.pop("canonical_body_sha256")
    templates_body["h1d_specs_identity_sha256"] = (
        authority._specs_identity(forged_expected)
    )
    forged_templates = {
        **templates_body,
        "canonical_body_sha256": module.canonical_sha256(templates_body),
    }
    forged_raw = module.canonical_json_bytes(forged_templates) + b"\n"
    forged_key = original_key
    if source_mutation == "coordinate":
        forged_key = "caller/forged-task6-templates.json"
        forged_sources["task6_templates"]["key"] = forged_key
    if source_mutation == "version":
        forged_sources["task6_templates"]["version_id"] = "caller-version"
    forged_sources["task6_templates"]["file_sha256"] = hashlib.sha256(
        forged_raw
    ).hexdigest()
    authority.store.raw_by_key[forged_key] = forged_raw
    return forged_expected, forged_sources, original_sources


@pytest.mark.parametrize(
    "source_mutation",
    ("bundle", "coordinate", "version", "digest"),
)
def test_template_bundle_source_is_independent_of_caller_envelope(
    source_mutation: str,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    forged_expected, forged_sources, trusted_sources = (
        _forge_iam_plan_and_bundle(
            module,
            authority,
            expected,
            source_mutation=source_mutation,
        )
    )
    forged_authority = module.AwsExpectedStateAuthority(
        store=authority.store,
        sources=forged_sources,
        trusted_sources=trusted_sources,
        clock=lambda: NOW,
    )
    with pytest.raises(ValueError, match="trusted campaign source"):
        forged_authority.authenticate(forged_expected)


def test_trusted_source_reader_uses_fixed_consistent_campaign_record() -> None:
    module = _load_script()
    sources = {
        name: {
            "key": f"authority/{name}.json",
            "version_id": f"{name}-version",
            "file_sha256": str(index) * 64,
        }
        for index, name in enumerate(
            (
                "task6_manifest",
                "task6_templates",
                "task7_postcreate_manifest",
                "task7_inventory",
            ),
            start=1,
        )
    }
    body = {
        "schema_version": 1,
        "record_type": "glm52_h1d_trusted_campaign_sources_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "PK": f"RUN#{RUN_ID}",
        "SK": "H1D_TRUSTED_SOURCE",
        "state": "SEALED",
        "authority_sources": sources,
        "source_contract_sha256": "a" * 64,
        "cloudformation_deployment_role": {
            "role_arn": (
                f"arn:aws:iam::{ACCOUNT_ID}:role/"
                "keep-glm52-h1g-cloudformation-deployment"
            ),
            "role_id": "AROAEXACTRETAINEDROLEID",
            "request_id": "iam-get-role-request",
        },
        "dynamodb_expected_items": sorted(
            [
            {
                "key": (
                    f"RUN#{RUN_ID}|ACTIVATION#act-20260729-0001#CONTROL"
                ),
                "record_type": "glm52_production_control",
                "body_sha256": "c" * 64,
                "consistent_read": True,
            },
            {
                "key": (
                    f"RUN#{RUN_ID}|ACTIVATION#act-20260729-0001#"
                    "ACTION#00000001#SKY#00000001"
                ),
                "record_type": "glm52_production_action",
                "body_sha256": "d" * 64,
                "consistent_read": True,
            },
            {
                "key": f"RUN#{RUN_ID}|ACTIVATION_INDEX",
                "record_type": "glm52_production_activation_index",
                "body_sha256": "b" * 64,
                "consistent_read": True,
            },
            ],
            key=lambda item: item["key"],
        ),
        "cloudformation_expected_items": [
            {
                "stack_id": (
                    f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:"
                    f"stack/keep-glm52-h1g-{kind}/{kind}-uuid"
                ),
                "status": "UPDATE_COMPLETE",
                "template_sha256": "1" * 64,
                "parameters_sha256": "2" * 64,
                "resources_sha256": "3" * 64,
                "stack_tags": [
                    {"key": "Campaign", "value": "GLM-5.2"},
                    {"key": "RunId", "value": RUN_ID},
                ],
                "service_role_arn": (
                    f"arn:aws:iam::{ACCOUNT_ID}:role/"
                    "keep-glm52-h1g-cloudformation-deployment"
                ),
                "termination_protection": True,
                "manifest_identity_sha256": "4" * 64,
            }
            for kind in ("fence", "retained", "support")
        ],
    }
    record = {
        **body,
        "canonical_body_sha256": module.canonical_sha256(body),
    }
    record_json = module.canonical_json_bytes(record).decode("utf-8")

    class Runner:
        def __init__(self) -> None:
            self.operations: list[tuple[str, ...]] = []

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            self.operations.append(operation)
            return {
                "Item": {
                    "PK": {"S": f"RUN#{RUN_ID}"},
                    "SK": {"S": "H1D_TRUSTED_SOURCE"},
                    "record_type": {
                        "S": "glm52_h1d_trusted_campaign_sources_v1"
                    },
                    "state": {"S": "SEALED"},
                    "canonical_body_json": {"S": record_json},
                    "canonical_body_sha256": {
                        "S": record["canonical_body_sha256"]
                    },
                },
                "ResponseMetadata": {"RequestId": "ddb-trusted-source-1"},
            }

    runner = Runner()
    reader = module.AwsTrustedCampaignSourceReader(runner=runner)
    contract = reader.read_contract()
    assert contract == {
        "authority_sources": sources,
        "source_contract_sha256": "a" * 64,
        "cloudformation_deployment_role": body[
            "cloudformation_deployment_role"
        ],
        "dynamodb_expected_items": tuple(
            body["dynamodb_expected_items"]
        ),
        "cloudformation_expected_items": tuple(
            body["cloudformation_expected_items"]
        ),
        "request_id": "ddb-trusted-source-1",
    }
    assert reader.read_sources() == sources
    assert runner.operations == [
        (
            "dynamodb",
            "get-item",
            "--table-name",
            "keep-glm52-h1g-ledger-v1",
            "--key",
            (
                '{"PK":{"S":"RUN#glm52-sky-20260724"},'
                '"SK":{"S":"H1D_TRUSTED_SOURCE"}}'
            ),
            "--consistent-read",
            "--return-consumed-capacity",
            "NONE",
        )
    ]


def test_dynamodb_pagination_uses_service_last_evaluated_key() -> None:
    module = _load_script()
    assert module._INVENTORY_PAGINATION["dynamodb"] == (
        ("Items",),
        ("LastEvaluatedKey",),
        "--exclusive-start-key",
    )


def test_dynamodb_reader_projects_exact_strong_query_rows_without_rescan() -> None:
    module = _load_script()
    raw_items = [
        {
            "PK": {"S": f"RUN#{RUN_ID}"},
            "SK": {"S": "ACTIVATION_INDEX"},
            "record_type": {"S": "glm52_production_activation_index"},
            "current_activation_id": {"S": "act-20260729-0001"},
        },
        {
            "PK": {"S": f"RUN#{RUN_ID}"},
            "SK": {"S": "ACTIVATION#act-20260729-0001#CONTROL"},
            "record_type": {"S": "glm52_production_control"},
        },
    ]
    token = {
        "PK": {"S": f"RUN#{RUN_ID}"},
        "SK": {"S": "ACTIVATION_INDEX"},
    }

    class Runner:
        def __init__(self) -> None:
            self.operations: list[tuple[str, ...]] = []

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            self.operations.append(operation)
            if "--exclusive-start-key" not in operation:
                return {
                    "Items": [raw_items[0]],
                    "LastEvaluatedKey": token,
                    "ResponseMetadata": {"RequestId": "ddb-query-1"},
                }
            return {
                "Items": [raw_items[1]],
                "ResponseMetadata": {"RequestId": "ddb-query-2"},
            }

    authority, expected = _production_expected_authority_fixture(module)
    del authority
    spec = next(
        spec for spec in expected.specs
        if spec.family == "dynamodb"
    )
    runner = Runner()
    reader = module.AwsCliLiveReader(runner=runner, clock=lambda: NOW)
    first = reader.read_page(spec=spec, continuation_token=None)
    second = reader.read_page(
        spec=spec,
        continuation_token=first.next_token,
    )
    assert first.items == (
        {
            "key": f"RUN#{RUN_ID}|ACTIVATION_INDEX",
            "record_type": "glm52_production_activation_index",
            "body_sha256": canonical_sha256(raw_items[0]),
            "consistent_read": True,
        },
    )
    assert second.items == (
        {
            "key": (
                f"RUN#{RUN_ID}|"
                "ACTIVATION#act-20260729-0001#CONTROL"
            ),
            "record_type": "glm52_production_control",
            "body_sha256": canonical_sha256(raw_items[1]),
            "consistent_read": True,
        },
    )
    assert all(operation[:2] == ("dynamodb", "query") for operation in runner.operations)
    assert json.loads(
        runner.operations[1][
            runner.operations[1].index("--exclusive-start-key") + 1
        ]
    ) == token


@pytest.mark.parametrize(
    "mutation",
    (
        "source_contract",
        "dynamodb",
        "cloudformation_role",
        "cloudformation_status",
        "cloudformation_tags",
    ),
)
def test_expected_authority_enforces_sealed_semantic_contract(
    mutation: str,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    documents = {
        name: json.loads(
            authority.store.raw_by_key[coordinate["key"]]
        )
        for name, coordinate in authority.sources.items()
    }
    source_contract = module._trusted_source_contract_sha256(
        task6=documents["task6_manifest"],
        templates=documents["task6_templates"],
        postcreate=documents["task7_postcreate_manifest"],
        inventory=documents["task7_inventory"],
    )
    ddb_items = tuple(
        dict(item)
        for item in next(
            spec
            for spec in expected.specs
            if spec.family == "dynamodb"
        ).expected_items
    )
    cloudformation_items = tuple(
        dict(item)
        for item in next(
            spec
            for spec in expected.specs
            if spec.family == "cloudformation"
        ).expected_items
    )
    cloudformation_deployment_role = (
        authority.trusted_source_reader.read_contract()[
            "cloudformation_deployment_role"
        ]
    )
    if mutation == "source_contract":
        source_contract = "f" * 64
    elif mutation == "dynamodb":
        ddb_items = (
            {
                **ddb_items[0],
                "body_sha256": "f" * 64,
            },
            *ddb_items[1:],
        )
    elif mutation == "cloudformation_role":
        cloudformation_deployment_role = {
            "role_arn": (
                f"arn:aws:iam::{ACCOUNT_ID}:role/"
                "attacker-deployment"
            ),
            "role_id": "AROAATTACKERROLEIDENTITY",
            "request_id": "attacker-request",
        }
        cloudformation_items = tuple(
            {
                **item,
                "service_role_arn": (
                    cloudformation_deployment_role["role_arn"]
                ),
            }
            for item in cloudformation_items
        )
    elif mutation == "cloudformation_status":
        cloudformation_items = (
            {
                **cloudformation_items[0],
                "status": "UPDATE_ROLLBACK_COMPLETE",
            },
            *cloudformation_items[1:],
        )
    else:
        cloudformation_items = (
            {
                **cloudformation_items[0],
                "stack_tags": [
                    {"key": "Campaign", "value": "attacker"},
                ],
            },
            *cloudformation_items[1:],
        )

    class TrustedContract:
        def read_contract(self) -> dict[str, object]:
            return {
                "authority_sources": deepcopy(authority.sources),
                "source_contract_sha256": source_contract,
                "dynamodb_expected_items": ddb_items,
                "cloudformation_expected_items": cloudformation_items,
                "cloudformation_deployment_role": deepcopy(
                    cloudformation_deployment_role
                ),
                "request_id": "ddb-trusted-source-1",
            }

    sealed_authority = module.AwsExpectedStateAuthority(
        store=authority.store,
        sources=authority.sources,
        trusted_source_reader=TrustedContract(),
        clock=lambda: NOW,
    )
    with pytest.raises(ValueError, match="semantic|sealed"):
        sealed_authority.authenticate(expected)


def test_spend_and_identity_boundaries_preserve_actual_service_request_ids() -> None:
    module = _load_script()

    class Runner:
        def get_s3_object(self, **kwargs: object):
            return {
                "VersionId": "version-exact",
                "ETag": '"etag"',
                "ResponseMetadata": {"RequestId": "s3-get-1"},
            }, b"{}\n"

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            if operation[:2] == ("s3api", "list-object-versions"):
                return {
                    "Versions": [],
                    "DeleteMarkers": [],
                    "IsTruncated": False,
                    "ResponseMetadata": {"RequestId": "s3-list-1"},
                }
            if operation[:2] == ("ec2", "describe-instances"):
                return {
                    "Reservations": [],
                    "ResponseMetadata": {"RequestId": "ec2-list-1"},
                }
            return {
                "Account": ACCOUNT_ID,
                "Arn": (
                    f"arn:aws:sts::{ACCOUNT_ID}:"
                    "assumed-role/keep-glm52-h1g-decision/session"
                ),
                "UserId": "AROATEST:session",
                "ResponseMetadata": {"RequestId": "sts-identity-1"},
            }

        def credential_expiration(self) -> str:
            return "2026-07-29T02:00:00Z"

    runner = Runner()
    store = module.AwsCliSpendObjectStore(
        runner=runner,
        bucket="keep-glm52-models",
        exact_versions={"authority.json": "version-exact"},
    )
    assert store.get_object(key="authority.json").request_id == "s3-get-1"
    assert (
        store.list_namespace(
            prefix="campaigns/run/",
            continuation_token=None,
        ).request_id
        == "s3-list-1"
    )
    assert module.AwsCliSpendEc2(
        runner=runner
    ).describe_allocation_history(
        run_id=RUN_ID,
        next_token=None,
    )["request_id"] == "ec2-list-1"
    assert module._IdentityAdapter(
        runner=runner,
        clock=lambda: NOW,
    ).get_caller_identity().request_id == "sts-identity-1"


def test_every_service_plan_is_closed_even_if_trusted_bundle_is_rehashed() -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    forged_expected, forged_sources, _trusted_sources = (
        _forge_iam_plan_and_bundle(
            module,
            authority,
            expected,
            source_mutation="bundle",
        )
    )
    forged_authority = module.AwsExpectedStateAuthority(
        store=authority.store,
        sources=forged_sources,
        trusted_sources=forged_sources,
        clock=lambda: NOW,
    )
    with pytest.raises(ValueError, match="IAM.*plan"):
        forged_authority.authenticate(forged_expected)


def test_caller_cannot_supply_or_omit_code_owned_detail_commands() -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    forged_expected, forged_sources, _trusted_sources = (
        _forge_iam_plan_and_bundle(
            module,
            authority,
            expected,
            source_mutation="bundle",
            inventory_arguments=[],
        )
    )
    forged_authority = module.AwsExpectedStateAuthority(
        store=authority.store,
        sources=forged_sources,
        trusted_sources=forged_sources,
        clock=lambda: NOW,
    )
    with pytest.raises(ValueError, match="code-owned version three"):
        forged_authority.authenticate(forged_expected)


def test_real_cli_command_inventory_is_read_only_complete_and_excludes_cost_explorer() -> None:
    module = _load_script()
    assert set(module.READ_COMMANDS) == set(REQUIRED_LIVE_FAMILIES)
    flattened = " ".join(
        " ".join(command) for commands in module.READ_COMMANDS.values()
        for command in commands
    ).lower()
    for service in (
        "cloudformation",
        "lambda",
        "iam",
        "events",
        "scheduler",
        "sqs",
        "sns",
        "ec2",
        "s3api",
        "dynamodb",
        "ssm",
        "cloudwatch",
        "logs",
    ):
        assert service in flattened
    assert "ce " not in f"{flattened} "
    for forbidden in (
        "create",
        "update",
        "delete",
        "put",
        "run-instances",
        "start-instances",
        "stop-instances",
        "terminate-instances",
    ):
        assert forbidden not in flattened


def test_real_cli_reader_paginates_with_monotonic_page_identity() -> None:
    module = _load_script()

    class Runner:
        def __init__(self) -> None:
            self.operations: list[tuple[str, ...]] = []

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            self.operations.append(operation)
            assert timeout_seconds == 5
            token = (
                operation[operation.index("--next-token") + 1]
                if "--next-token" in operation
                else None
            )
            index = {None: 0, "page-1": 1, "page-2": 2}[token]
            return {
                "logGroups": [
                    {
                        "arn": f"arn:aws:logs:us-west-2:{ACCOUNT_ID}:log-group:g{index}",
                        "retentionInDays": 30,
                        "kmsKeyId": f"arn:aws:kms:us-west-2:{ACCOUNT_ID}:key/k{index}",
                    }
                ],
                "nextToken": (f"page-{index + 1}" if index < 2 else None),
            }

    runner = Runner()
    reader = module.AwsCliLiveReader(
        runner=runner,
        clock=lambda: NOW,
    )
    spec = LiveReadSpec(
        family="logs",
        operation="logs.inspect_complete",
        parameters={
            "cli_plan": {
                "schema_version": 2,
                "projections": {
                    "log_group_arn": {
                        "source": 0,
                        "path": ["arn"],
                        "transform": "identity",
                    },
                    "retention_days": {
                        "source": 0,
                        "path": ["retentionInDays"],
                        "transform": "identity",
                    },
                    "kms_key_arn": {
                        "source": 0,
                        "path": ["kmsKeyId"],
                        "transform": "identity",
                    },
                },
            }
        },
        identity_field="log_group_arn",
        expected_items=(
            {
                "log_group_arn": (
                    f"arn:aws:logs:us-west-2:{ACCOUNT_ID}:log-group:g0"
                ),
                "retention_days": 30,
                "kms_key_arn": (
                    f"arn:aws:kms:us-west-2:{ACCOUNT_ID}:key/k0"
                ),
            },
        ),
    )
    pages = []
    token = None
    while not pages or token is not None:
        page = reader.read_page(spec=spec, continuation_token=token)
        pages.append(page)
        token = page.next_token
    assert [page.page_index for page in pages] == [0]
    assert [page.request_token for page in pages] == [None]
    assert len(pages[0].items) == 3
    assert _without_no_paginate(runner.operations) == [
        (
            "logs",
            "describe-log-groups",
            "--log-group-name-prefix",
            "/aws/lambda/keep-glm52-h1g",
        ),
        (
            "logs",
            "describe-log-groups",
            "--log-group-name-prefix",
            "/aws/lambda/keep-glm52-h1g",
            "--next-token",
            "page-1",
        ),
        (
            "logs",
            "describe-log-groups",
            "--log-group-name-prefix",
            "/aws/lambda/keep-glm52-h1g",
            "--next-token",
            "page-2",
        ),
        (
            "logs",
            "describe-log-groups",
            "--log-group-name-prefix",
            f"/aws/lambda/{RUN_ID}",
        ),
        (
            "logs",
            "describe-log-groups",
            "--log-group-name-prefix",
            f"/aws/lambda/{RUN_ID}",
            "--next-token",
            "page-1",
        ),
        (
            "logs",
            "describe-log-groups",
            "--log-group-name-prefix",
            f"/aws/lambda/{RUN_ID}",
            "--next-token",
            "page-2",
        ),
    ]


def test_real_cli_reader_completely_paginates_each_detail_operation() -> None:
    module = _load_script()
    rule_arn = (
        f"arn:aws:events:us-west-2:{ACCOUNT_ID}:rule/decision"
    )

    class Runner:
        def __init__(self) -> None:
            self.operations: list[tuple[str, ...]] = []

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            self.operations.append(operation)
            if operation[:2] == ("events", "list-rules"):
                return {
                    "Rules": [{"Arn": rule_arn, "Name": "decision"}],
                }
            if operation[:2] == ("events", "describe-rule"):
                return {"Arn": rule_arn, "Name": "decision"}
            if "--next-token" not in operation:
                return {"Targets": [{"Id": "A"}], "NextToken": "detail-2"}
            return {"Targets": [{"Id": "B"}]}

    runner = Runner()
    reader = module.AwsCliLiveReader(runner=runner, clock=lambda: NOW)
    spec = LiveReadSpec(
        family="eventbridge",
        operation="eventbridge.inspect_complete",
        parameters={
            "cli_plan": {
                "schema_version": 2,
                "projections": {
                    "rule_arn": {
                        "source": 0,
                        "path": ["Arn"],
                        "transform": "identity",
                    },
                    "targets": {
                        "source": 2,
                        "path": ["items"],
                        "transform": "identity",
                    },
                },
            }
        },
        identity_field="rule_arn",
        expected_items=(
            {
                "rule_arn": rule_arn,
                "targets": [{"Id": "A"}, {"Id": "B"}],
            },
        ),
    )
    page = reader.read_page(spec=spec, continuation_token=None)
    assert page.items == spec.expected_items
    assert _without_no_paginate(runner.operations) == [
        (
            "events",
            "list-rules",
            "--name-prefix",
            "keep-glm52-h1g",
        ),
        (
            "events",
            "list-rules",
            "--name-prefix",
            RUN_ID,
        ),
        ("events", "describe-rule", "--name", "decision"),
        ("events", "list-targets-by-rule", "--rule", "decision"),
        (
            "events",
            "list-targets-by-rule",
            "--rule",
            "decision",
            "--next-token",
            "detail-2",
        ),
    ]


def test_real_cli_reader_derives_canonical_hash_from_complete_detail_pages() -> None:
    module = _load_script()
    rule_arn = (
        f"arn:aws:events:us-west-2:{ACCOUNT_ID}:rule/decision"
    )

    class Runner:
        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            if operation[:2] == ("events", "list-rules"):
                return {"Rules": [{"Arn": rule_arn, "Name": "decision"}]}
            if operation[:2] == ("events", "describe-rule"):
                return {"Arn": rule_arn, "Name": "decision"}
            if "--next-token" not in operation:
                return {"Targets": [{"Id": "A"}], "NextToken": "next"}
            return {"Targets": [{"Id": "B"}]}

    expected_hash = canonical_sha256([{"Id": "A"}, {"Id": "B"}])
    spec = LiveReadSpec(
        family="eventbridge",
        operation="eventbridge.inspect_complete",
        parameters={
            "cli_plan": {
                "schema_version": 2,
                "projections": {
                    "rule_arn": {
                        "source": 0,
                        "path": ["Arn"],
                        "transform": "identity",
                    },
                    "targets_sha256": {
                        "source": 2,
                        "path": ["items"],
                        "transform": "canonical_sha256",
                    },
                },
            }
        },
        identity_field="rule_arn",
        expected_items=(
            {
                "rule_arn": rule_arn,
                "targets_sha256": expected_hash,
            },
        ),
    )
    page = module.AwsCliLiveReader(
        runner=Runner(),
        clock=lambda: NOW,
    ).read_page(spec=spec, continuation_token=None)
    assert page.items[0]["targets_sha256"] == expected_hash


def test_closed_iam_graph_accepts_absent_terminal_detail_tokens() -> None:
    module = _load_script()
    role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/decision"
    policy_a = f"arn:aws:iam::{ACCOUNT_ID}:policy/A"
    policy_b = f"arn:aws:iam::{ACCOUNT_ID}:policy/B"

    class Runner:
        def __init__(self) -> None:
            self.operations: list[tuple[str, ...]] = []

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            self.operations.append(operation)
            command = operation[:2]
            if command == ("iam", "list-roles"):
                return {"Roles": [{"Arn": role_arn, "RoleName": "decision"}]}
            if command == ("iam", "get-role"):
                return {"Role": {"Arn": role_arn, "RoleName": "decision"}}
            if command == ("iam", "list-instance-profiles-for-role"):
                return {"InstanceProfiles": []}
            if command == ("iam", "list-role-policies"):
                if "--marker" not in operation:
                    return {"PolicyNames": ["InlineA"], "Marker": "m2"}
                return {"PolicyNames": ["InlineB"]}
            if command == ("iam", "get-role-policy"):
                name = operation[operation.index("--policy-name") + 1]
                return {
                    "RoleName": "decision",
                    "PolicyName": name,
                    "PolicyDocument": {"Version": "2012-10-17"},
                }
            if command == ("iam", "list-attached-role-policies"):
                return {
                    "AttachedPolicies": [
                        {"PolicyName": "A", "PolicyArn": policy_a},
                        {"PolicyName": "B", "PolicyArn": policy_b},
                    ]
                }
            if command == ("iam", "get-policy"):
                arn = operation[operation.index("--policy-arn") + 1]
                return {
                    "Policy": {
                        "Arn": arn,
                        "DefaultVersionId": "v3" if arn == policy_a else "v1",
                    }
                }
            if command == ("iam", "get-policy-version"):
                return {
                    "PolicyVersion": {
                        "Document": {"Version": "2012-10-17"}
                    }
                }
            raise AssertionError(operation)

    spec = LiveReadSpec(
        family="iam",
        operation="iam.inspect_complete",
        parameters={
            "run_id": RUN_ID,
            "cli_plan": {
                "schema_version": 2,
                "projections": {
                    "role_arn": {
                        "source": 0,
                        "path": ["Arn"],
                        "transform": "identity",
                    },
                    "inline_policy_sha256": {
                        "source": 4,
                        "path": ["responses"],
                        "transform": "canonical_sha256",
                    },
                    "managed_policy_versions_sha256": {
                        "source": 7,
                        "path": ["responses"],
                        "transform": "canonical_sha256",
                    },
                },
            },
        },
        identity_field="role_arn",
        expected_items=(
            {
                "role_arn": role_arn,
                "inline_policy_sha256": ZERO,
                "managed_policy_versions_sha256": ZERO,
            },
        ),
    )
    runner = Runner()
    page = module.AwsCliLiveReader(
        runner=runner,
        clock=lambda: NOW,
    ).read_page(spec=spec, continuation_token=None)
    assert page.next_token is None
    assert _without_no_paginate(runner.operations) == [
        ("iam", "list-roles"),
        ("iam", "get-role", "--role-name", "decision"),
        (
            "iam",
            "list-instance-profiles-for-role",
            "--role-name",
            "decision",
        ),
        ("iam", "list-role-policies", "--role-name", "decision"),
        (
            "iam",
            "list-role-policies",
            "--role-name",
            "decision",
            "--marker",
            "m2",
        ),
        (
            "iam",
            "get-role-policy",
            "--role-name",
            "decision",
            "--policy-name",
            "InlineA",
        ),
        (
            "iam",
            "get-role-policy",
            "--role-name",
            "decision",
            "--policy-name",
            "InlineB",
        ),
        (
            "iam",
            "list-attached-role-policies",
            "--role-name",
            "decision",
        ),
        ("iam", "get-policy", "--policy-arn", policy_a),
        ("iam", "get-policy", "--policy-arn", policy_b),
        (
            "iam",
            "get-policy-version",
            "--policy-arn",
            policy_a,
            "--version-id",
            "v3",
        ),
        (
            "iam",
            "get-policy-version",
            "--policy-arn",
            policy_b,
            "--version-id",
            "v1",
        ),
    ]
    assert page.items[0]["role_arn"] == role_arn


def test_real_cli_reader_derives_ec2_dependent_ids_and_exact_attribute() -> None:
    module = _load_script()
    instance_id = "i-00000000000000001"

    class Runner:
        def __init__(self) -> None:
            self.operations: list[tuple[str, ...]] = []

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            self.operations.append(operation)
            command = operation[:2]
            if command == ("ec2", "describe-instances"):
                return {
                    "Instances": [
                        {
                            "InstanceId": instance_id,
                            "ImageId": "ami-0123456789abcdef0",
                            "BlockDeviceMappings": [
                                {"Ebs": {"VolumeId": "vol-00000000000000001"}},
                                {"Ebs": {"VolumeId": "vol-00000000000000002"}},
                            ],
                        }
                    ]
                }
            if command == ("ec2", "describe-volumes"):
                volume_id = operation[operation.index("--volume-ids") + 1]
                return {"Volumes": [{"VolumeId": volume_id}]}
            if command == ("ec2", "describe-images"):
                return {"Images": [{"ImageId": "ami-0123456789abcdef0"}]}
            if command == ("ec2", "describe-instance-attribute"):
                return {
                    "InstanceId": instance_id,
                    "UserData": {"Value": "encoded"},
                }
            raise AssertionError(operation)

    spec = LiveReadSpec(
        family="ec2",
        operation="ec2.inspect_complete",
        parameters={
            "run_id": RUN_ID,
            "cli_plan": {
                "schema_version": 2,
                "projections": {
                    "instance_id": {
                        "source": 0,
                        "path": ["InstanceId"],
                        "transform": "identity",
                    },
                    "user_data_sha256": {
                        "source": 3,
                        "path": ["responses", 0, "UserData", "Value"],
                        "transform": "canonical_sha256",
                    },
                },
            },
        },
        identity_field="instance_id",
        expected_items=(
            {
                "instance_id": instance_id,
                "user_data_sha256": ZERO,
            },
        ),
    )
    runner = Runner()
    page = module.AwsCliLiveReader(
        runner=runner,
        clock=lambda: NOW,
    ).read_page(spec=spec, continuation_token=None)
    assert page.next_token is None
    assert page.items[0]["user_data_sha256"] == canonical_sha256("encoded")
    assert _without_no_paginate(runner.operations) == [
        (
            "ec2",
            "describe-instances",
            "--filters",
            f"Name=tag:campaign-run-id,Values={RUN_ID}",
            "--max-results",
            "100",
            "--query",
            "{Instances: Reservations[].Instances[], NextToken: NextToken}",
        ),
        (
            "ec2",
            "describe-instances",
            "--filters",
            "Name=instance-type,Values=p5.48xlarge",
            "--max-results",
            "100",
            "--query",
            "{Instances: Reservations[].Instances[], NextToken: NextToken}",
        ),
        (
            "ec2",
            "describe-instances",
            "--filters",
            f"Name=tag:Name,Values={RUN_ID}",
            "--max-results",
            "100",
            "--query",
            "{Instances: Reservations[].Instances[], NextToken: NextToken}",
        ),
        (
            "ec2",
            "describe-volumes",
            "--volume-ids",
            "vol-00000000000000001",
        ),
        (
            "ec2",
            "describe-volumes",
            "--volume-ids",
            "vol-00000000000000002",
        ),
        (
            "ec2",
            "describe-images",
            "--image-ids",
            "ami-0123456789abcdef0",
        ),
        (
            "ec2",
            "describe-instance-attribute",
            "--instance-id",
            instance_id,
            "--attribute",
            "userData",
        ),
    ]


def test_real_cli_reader_accepts_absent_terminal_inventory_token() -> None:
    module = _load_script()

    class Runner:
        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            return {
                "logGroups": [
                    {
                        "arn": (
                            f"arn:aws:logs:us-west-2:{ACCOUNT_ID}:"
                            "log-group:/aws/lambda/decision"
                        )
                    }
                ]
            }

    spec = LiveReadSpec(
        family="logs",
        operation="logs.inspect_complete",
        parameters={
            "run_id": RUN_ID,
            "cli_plan": {
                "schema_version": 2,
                "projections": {
                    "log_group_arn": {
                        "source": 0,
                        "path": ["arn"],
                        "transform": "identity",
                    }
                },
            },
        },
        identity_field="log_group_arn",
        expected_items=(
            {
                "log_group_arn": (
                    f"arn:aws:logs:us-west-2:{ACCOUNT_ID}:"
                    "log-group:/aws/lambda/decision"
                )
            },
        ),
    )
    page = module.AwsCliLiveReader(
        runner=Runner(),
        clock=lambda: NOW,
    ).read_page(spec=spec, continuation_token=None)
    assert page.next_token is None


def test_cloudformation_termination_protection_uses_describe_stacks_field() -> None:
    module = _load_script()
    stack_id = (
        f"arn:aws:cloudformation:us-west-2:{ACCOUNT_ID}:"
        "stack/decision/00000000-0000-0000-0000-000000000000"
    )

    class Runner:
        def __init__(self) -> None:
            self.operations: list[tuple[str, ...]] = []

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            self.operations.append(operation)
            if operation[:2] == ("cloudformation", "describe-stacks"):
                return {
                    "Stacks": [
                        {
                            "StackId": stack_id,
                            "EnableTerminationProtection": True,
                        }
                    ]
                }
            if operation[:2] == ("cloudformation", "get-template"):
                return {"TemplateBody": {"Resources": {}}}
            if operation[:2] == (
                "cloudformation",
                "list-stack-resources",
            ):
                return {"StackResourceSummaries": []}
            raise AssertionError(operation)

    spec = LiveReadSpec(
        family="cloudformation",
        operation="cloudformation.inspect_complete",
        parameters={
            "run_id": RUN_ID,
            "cli_plan": {
                "schema_version": 2,
                "projections": {
                    "stack_id": {
                        "source": 0,
                        "path": ["StackId"],
                        "transform": "identity",
                    },
                    "termination_protection": {
                        "source": 0,
                        "path": ["EnableTerminationProtection"],
                        "transform": "identity",
                    },
                },
            },
        },
        identity_field="stack_id",
        expected_items=(
            {
                "stack_id": stack_id,
                "termination_protection": True,
            },
        ),
    )
    runner = Runner()
    page = module.AwsCliLiveReader(
        runner=runner,
        clock=lambda: NOW,
    ).read_page(spec=spec, continuation_token=None)
    assert page.items == spec.expected_items
    assert _without_no_paginate(runner.operations) == [
        ("cloudformation", "describe-stacks"),
        ("cloudformation", "get-template", "--stack-name", stack_id),
        (
            "cloudformation",
            "list-stack-resources",
            "--stack-name",
            stack_id,
        ),
    ]


def test_cli_runner_has_zero_hidden_retry_timeout_and_malformed_json_failure() -> None:
    module = _load_script()
    calls: list[dict[str, object]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, 0, "{malformed", "")

    runner = module.AwsCliCommandRunner(subprocess_run=fake_run)
    with pytest.raises(ValueError, match="malformed"):
        runner.run_json(("sts", "get-caller-identity"), timeout_seconds=3)
    assert len(calls) == 1
    assert calls[0]["timeout"] == 3
    env = calls[0]["env"]
    assert env["AWS_PAGER"] == ""
    assert env["AWS_MAX_ATTEMPTS"] == "1"
    assert env["AWS_RETRY_MODE"] == "standard"


def test_exact_s3_version_read_rejects_missing_response_version_id() -> None:
    module = _load_script()

    class Runner:
        def get_s3_object(self, **kwargs: object):
            return {}, b"{}\n"

    store = module.AwsCliSpendObjectStore(
        runner=Runner(),
        bucket="keep-glm52-models",
        exact_versions={"authority.json": "version-exact"},
    )
    with pytest.raises(ValueError, match="VersionId"):
        store.get_object(key="authority.json")


def test_h1d_identity_adapter_performs_fresh_sts_inside_measured_phase() -> None:
    module = _load_script()

    class Runner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            self.calls.append(operation)
            return {
                "Account": ACCOUNT_ID,
                "Arn": (
                    f"arn:aws:sts::{ACCOUNT_ID}:"
                    "assumed-role/keep-glm52-h1g-decision/session"
                ),
                "UserId": "AROATEST:session",
            }

        def credential_expiration(self) -> str:
            return "2026-07-29T02:00:00Z"

    runner = Runner()
    identity = module._IdentityAdapter(
        runner=runner,
        clock=lambda: NOW,
    ).get_caller_identity()
    assert identity.account_id == ACCOUNT_ID
    assert runner.calls == [("sts", "get-caller-identity")]


def test_h1d_identity_uses_authenticated_sdk_credential_expiration() -> None:
    module = _load_script()

    class Runner:
        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            return {
                "Account": ACCOUNT_ID,
                "Arn": (
                    f"arn:aws:sts::{ACCOUNT_ID}:"
                    "assumed-role/keep-glm52-h1g-decision/session"
                ),
                "UserId": "AROATEST:session",
                "ResponseMetadata": {"RequestId": "sts-request-1"},
            }

        def credential_expiration(self) -> str:
            return "2026-07-29T02:00:00Z"

    identity = module._IdentityAdapter(
        runner=Runner(),
        clock=lambda: NOW,
    ).get_caller_identity()
    assert identity.credential_expiration == "2026-07-29T02:00:00Z"


def test_production_spend_services_enumerate_held_reserves_fail_closed() -> None:
    module = _load_script()

    class Store:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def list_namespace(
            self, *, prefix: str, continuation_token: str | None
        ) -> SpendListPage:
            self.calls.append((prefix, continuation_token))
            return SpendListPage(
                keys=(),
                version_ids=(),
                next_token=None,
                request_id="reserve-list-1",
                observed_at="2026-07-29T01:00:00Z",
            )

    services = module.build_spend_services(
        runner=object(),
        store=Store(),
    )
    page = services.reserve_reader.list_reserves(next_token=None)
    assert isinstance(page, module.ReserveListPage)
    assert page.records == ()


def test_production_ec2_spend_adapter_normalizes_utc_and_does_not_invent_end() -> None:
    module = _load_script()

    class Runner:
        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            return {
                "Reservations": [
                    {
                        "Instances": [
                            {
                                "InstanceId": "i-00000000000000001",
                                "InstanceType": "p5.48xlarge",
                                "State": {"Name": "terminated"},
                                "LaunchTime": "2026-07-28T23:00:00+00:00",
                                "Placement": {
                                    "AvailabilityZone": "us-west-2a"
                                },
                                "Tags": [],
                            }
                        ]
                    }
                ],
                "NextToken": None,
            }

    page = module.AwsCliSpendEc2(
        runner=Runner()
    ).describe_allocation_history(run_id=RUN_ID, next_token=None)
    assert "TerminatedAt" not in page["instances"][0]
    assert page["instances"][0]["LaunchTime"] == "2026-07-28T23:00:00Z"


def test_executable_run_builds_exact_core_sky_probe_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    events: list[str] = []
    expected = _expected_state()
    spend_request = {
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "descriptor_key": "campaigns/run/descriptor.json",
        "descriptor_file_sha256": "1" * 64,
        "descriptor_version_id": "descriptor-version",
        "approval_key": "campaigns/run/approval.json",
        "approval_file_sha256": "2" * 64,
        "approval_version_id": "approval-version",
        "latest_key": "campaigns/run/latest.json",
        "latest_file_sha256": "3" * 64,
        "latest_version_id": "latest-version",
        "snapshot_key": "campaigns/run/snapshot.json",
        "snapshot_file_sha256": "4" * 64,
        "snapshot_version_id": "snapshot-version",
        "ledger_records_prefix": "campaigns/run/records/",
        "expected_snapshot_body_sha256": "5" * 64,
        "observed_at": "2026-07-29T01:00:00Z",
    }
    authority_sources = {
        name: {
            "key": f"authority/{name}.json",
            "version_id": f"{name}-version",
            "file_sha256": str(index) * 64,
        }
        for index, name in enumerate(
            (
                "task6_manifest",
                "task6_templates",
                "task7_postcreate_manifest",
                "task7_inventory",
            ),
            start=1,
        )
    }
    from glm52_enforcement.task11_boundary import (
        build_task11_input_coordinate,
    )

    task9_coordinate = asdict(
        build_task11_input_coordinate(
            input_kind="TASK9_DEPLOYED_IDENTITY_COORDINATE",
            bucket="keep-glm52-models-246813579024-us-west-2",
            key=(
                "campaigns/glm52-sky-20260724/authorities/task9/"
                + expected.activation_id
                + "/TASK9_DEPLOYED_IDENTITY.json"
            ),
            version_id="task9-deployed-version-1",
            file_sha256="6" * 64,
            body_sha256="7" * 64,
        )
    )
    envelope = {
        "schema_version": 1,
        "record_type": "glm52_h1d_live_inspection_input_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "campaign_bucket": (
            "keep-glm52-models-246813579024-us-west-2"
        ),
        "h1d_expected_state": h1d_expected_state_to_mapping(expected),
        "spend_request": spend_request,
        "authority_sources": authority_sources,
        "task9_deployed_identity_coordinate": task9_coordinate,
    }
    raw = module.canonical_json_bytes(envelope) + b"\n"
    input_path = tmp_path / "input.json"
    input_path.write_bytes(raw)

    class Runner:
        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            assert operation == ("sts", "get-caller-identity")
            events.append("initial-sts")
            return {
                "Account": ACCOUNT_ID,
                "Arn": (
                    f"arn:aws:sts::{ACCOUNT_ID}:"
                    "assumed-role/keep-glm52-h1g-decision/session"
                ),
                "UserId": "AROATEST:session",
            }

    class StopAfterRequest(Exception):
        pass

    def runner_factory(**kwargs: object) -> Runner:
        return Runner()

    class TrustedSources:
        def read_sources(self) -> dict[str, object]:
            events.append("measured-trusted-source-read")
            return deepcopy(authority_sources)

    def trusted_source_factory(**kwargs: object) -> TrustedSources:
        return TrustedSources()

    def capture_request(
        request: H1dLiveAuthorityRequest,
        expected_state: H1dExpectedState,
        services: H1dLiveServices,
    ) -> object:
        events.append("measured-core-entered")
        assert events == ["initial-sts", "measured-core-entered"]
        assert isinstance(services.reader, module.AwsSdkLiveReader)
        assert isinstance(
            services.sky_relay_probe,
            module.ProductionTask9Probe,
        )
        assert request.sky_probe_request == {
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "run_id": RUN_ID,
            "activation_id": expected.activation_id,
        }
        raise StopAfterRequest

    monkeypatch.setattr(
        module,
        "inspect_h1d_live_authority",
        capture_request,
    )
    with pytest.raises(StopAfterRequest):
        module.run(
            [
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--expected-state",
                str(input_path),
                "--expected-state-sha256",
                hashlib.sha256(raw).hexdigest(),
                "--output",
                str(tmp_path / "evidence.json"),
            ],
            runner_factory=runner_factory,
            trusted_source_factory=trusted_source_factory,
        )


def _production_task9_deployed_fixture():
    from glm52_enforcement.canonical import canonical_json_bytes
    from glm52_enforcement.task11_boundary import (
        build_task11_input_coordinate,
    )

    fixture_spec = importlib.util.spec_from_file_location(
        "task9_contract_fixture",
        ROOT / "tests/test_glm52_task9_contract.py",
    )
    assert fixture_spec is not None and fixture_spec.loader is not None
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)
    deployed = fixture_module._deployed_identity()
    raw = canonical_json_bytes(deployed) + b"\n"
    coordinate = build_task11_input_coordinate(
        input_kind="TASK9_DEPLOYED_IDENTITY_COORDINATE",
        bucket="keep-glm52-models-246813579024-us-west-2",
        key=(
            "campaigns/glm52-sky-20260724/authorities/task9/"
            "activation-0001/TASK9_DEPLOYED_IDENTITY.json"
        ),
        version_id="task9-deployed-version-1",
        file_sha256=hashlib.sha256(raw).hexdigest(),
        body_sha256=deployed["canonical_identity_sha256"],
    )
    return deployed, asdict(coordinate)


def test_production_task9_probe_invokes_exact_published_attestation_bridge() -> None:
    module = _load_script()
    deployed, coordinate = _production_task9_deployed_fixture()
    sky_identity = deployed["sky_identity"]
    attestation_identity = next(
        item
        for item in sky_identity["identities"]
        if item["purpose"] == "ATTESTATION"
    )
    function_arn = (
        f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:"
        "function:keep-glm52-h1g-attestation:7"
    )
    calls = []

    class Runner:
        def invoke_task9_attestation(
            self, *, function_arn, payload, timeout_seconds
        ):
            request = json.loads(payload)
            calls.append((function_arn, request, timeout_seconds))
            body = {
                "schema_version": 1,
                "record_type": "glm52_sky_attestation_v1",
                "account_id": ACCOUNT_ID,
                "region": REGION,
                "run_id": RUN_ID,
                "freshness_nonce": request["freshness_nonce"],
                "direct_response_request_id": "attestation-request-1",
                "tls_peer_certificate_sha256": (
                    attestation_identity["server_certificate_sha256"]
                ),
                "sky_user_identity": sky_identity[
                    "service_account_user_id"
                ],
                "sky_roles": [sky_identity["service_account_role"]],
                "token_expires_at": "2026-07-29T03:00:00Z",
                "effective_controller_identity_sha256": sky_identity[
                    "consolidation_signal_identity_sha256"
                ],
                "observed_at": "2026-07-29T01:00:01Z",
                "sky_identity_contract_sha256": sky_identity[
                    "canonical_identity_sha256"
                ],
            }
            return {
                **body,
                "canonical_identity_sha256": canonical_sha256(body),
            }

    expected = SimpleNamespace(
        activation_id="activation-0001",
        specs=(
            SimpleNamespace(
                family="lambda",
                expected_items=({"function_arn": function_arn},),
            ),
        )
    )
    probe = module.ProductionTask9Probe(
        runner=Runner(),
        expected=expected,
        deployed_identity_coordinate=coordinate,
        campaign_bucket=coordinate["bucket"],
        deployed_identity=deployed,
        nonce_source=lambda: "q" * 64,
    )
    request = {
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": "activation-0001",
    }
    result = probe.inspect(request)
    assert result.request_identity_sha256 == canonical_sha256(request)
    assert result.sky_roles == ("user",)
    assert calls == [
        (
            function_arn,
            {
                "schema_version": 1,
                "record_type": "glm52_task11_attestation_request_v1",
                "run_id": RUN_ID,
        "activation_id": "activation-0001",
                "request_identity_sha256": canonical_sha256(
                    {
                        "domain": "H1D_TASK9_ATTESTATION_V1",
                        "freshness_nonce": "q" * 64,
                    }
                ),
                "freshness_nonce": "q" * 64,
                "sequence": 1,
                "task9_deployed_identity_coordinate": coordinate,
                "task9_deployed_identity_sha256": deployed[
                    "canonical_identity_sha256"
                ],
            },
            5,
        )
    ]


@pytest.mark.parametrize(
    ("field", "foreign"),
    (
        ("tls_peer_certificate_sha256", "f" * 64),
        ("sky_user_identity", "foreign-admin"),
        ("sky_roles", ["admin"]),
        ("effective_controller_identity_sha256", "e" * 64),
        ("sky_identity_contract_sha256", "d" * 64),
    ),
)
def test_production_task9_probe_rejects_self_rehashed_foreign_identity(
    field: str,
    foreign: object,
) -> None:
    module = _load_script()
    deployed, coordinate = _production_task9_deployed_fixture()
    sky_identity = deployed["sky_identity"]
    attestation_identity = next(
        item
        for item in sky_identity["identities"]
        if item["purpose"] == "ATTESTATION"
    )
    function_arn = (
        f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:"
        "function:keep-glm52-h1g-attestation:7"
    )
    calls = []

    class Runner:
        def invoke_task9_attestation(
            self, *, function_arn, payload, timeout_seconds
        ):
            request = json.loads(payload)
            calls.append((function_arn, request, timeout_seconds))
            body = {
                "schema_version": 1,
                "record_type": "glm52_sky_attestation_v1",
                "account_id": ACCOUNT_ID,
                "region": REGION,
                "run_id": RUN_ID,
                "freshness_nonce": request["freshness_nonce"],
                "direct_response_request_id": "attestation-request-1",
                "tls_peer_certificate_sha256": (
                    attestation_identity["server_certificate_sha256"]
                ),
                "sky_user_identity": sky_identity[
                    "service_account_user_id"
                ],
                "sky_roles": [sky_identity["service_account_role"]],
                "token_expires_at": "2026-07-29T03:00:00Z",
                "effective_controller_identity_sha256": sky_identity[
                    "consolidation_signal_identity_sha256"
                ],
                "observed_at": "2026-07-29T01:00:01Z",
                "sky_identity_contract_sha256": sky_identity[
                    "canonical_identity_sha256"
                ],
            }
            body[field] = foreign
            return {
                **body,
                "canonical_identity_sha256": canonical_sha256(body),
            }

    expected = SimpleNamespace(
        activation_id="activation-0001",
        specs=(
            SimpleNamespace(
                family="lambda",
                expected_items=({"function_arn": function_arn},),
            ),
        )
    )
    probe = module.ProductionTask9Probe(
        runner=Runner(),
        expected=expected,
        deployed_identity_coordinate=coordinate,
        campaign_bucket=coordinate["bucket"],
        deployed_identity=deployed,
        nonce_source=lambda: "q" * 64,
    )
    request = {
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
                "activation_id": "activation-0001",
    }
    with pytest.raises(ValueError, match="identity"):
        probe.inspect(request)
    assert len(calls) == 1
    assert calls[0][0] == function_arn
    assert calls[0][1]["task9_deployed_identity_coordinate"] == coordinate
    assert calls[0][1]["task9_deployed_identity_sha256"] == deployed[
        "canonical_identity_sha256"
    ]
    assert calls[0][2] == 5


def test_cli_wrong_account_stops_before_manifest_or_service_inspection(
    tmp_path: Path,
) -> None:
    module = _load_script()

    class Runner:
        def __init__(self, **kwargs: object) -> None:
            self.calls: list[object] = []

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            self.calls.append(operation)
            return {
                "Account": "000000000000",
                "Arn": "arn:aws:iam::000000000000:user/foreign",
                "UserId": "foreign",
            }

    holder: dict[str, object] = {}

    def factory(**kwargs: object) -> Runner:
        holder["runner"] = Runner(**kwargs)
        return holder["runner"]

    missing = tmp_path / "must-not-be-read.json"
    with pytest.raises(ValueError, match="account"):
        module.run(
            [
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--expected-state",
                str(missing),
                "--expected-state-sha256",
                "1" * 64,
                "--output",
                str(tmp_path / "evidence.json"),
            ],
            runner_factory=factory,
        )
    assert holder["runner"].calls == [("sts", "get-caller-identity")]


def test_cli_rejects_wrong_profile_region_and_existing_output_before_runner(
    tmp_path: Path,
) -> None:
    module = _load_script()
    called = False

    def factory(**kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("runner must not be constructed")

    base = [
        "--profile",
        PROFILE,
        "--region",
        REGION,
        "--expected-state",
        str(tmp_path / "state.json"),
        "--expected-state-sha256",
        "1" * 64,
        "--output",
        str(tmp_path / "evidence.json"),
    ]
    with pytest.raises(ValueError):
        module.run([*base[:1], "default", *base[2:]], runner_factory=factory)
    with pytest.raises(ValueError):
        module.run(
            [*base[:3], "us-east-1", *base[4:]], runner_factory=factory
        )
    Path(base[-1]).write_text("existing")
    with pytest.raises(FileExistsError):
        module.run(base, runner_factory=factory)
    assert called is False


def test_dynamodb_inventory_excludes_only_fixed_trusted_source_metadata() -> None:
    module = _load_script()
    spec = next(
        spec for spec in _expected_state().specs
        if spec.family == "dynamodb"
    )
    arguments = module._closed_inventory_arguments(spec)
    values = json.loads(
        arguments[arguments.index("--expression-attribute-values") + 1]
    )
    assert values == {
        ":pk": {"S": f"RUN#{RUN_ID}"},
        ":trusted_source_sk": {"S": "H1D_TRUSTED_SOURCE"},
    }
    assert arguments[
        arguments.index("--filter-expression") + 1
    ] == "SK <> :trusted_source_sk"
    assert "--projection-expression" not in arguments


def test_lambda_inventory_fans_out_to_exact_published_version_arns() -> None:
    module = _load_script()
    base_arn = (
        f"arn:aws:lambda:us-west-2:{ACCOUNT_ID}:"
        "function:keep-glm52-h1g-decision"
    )
    version_arns = (f"{base_arn}:7", f"{base_arn}:8")

    class Runner:
        def __init__(self) -> None:
            self.operations: list[tuple[str, ...]] = []

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            self.operations.append(operation)
            command = operation[:2]
            if command == ("lambda", "list-functions"):
                return {
                    "Functions": [
                        {
                            "FunctionName": "keep-glm52-h1g-decision",
                            "FunctionArn": base_arn,
                        }
                    ]
                }
            if command == ("lambda", "list-versions-by-function"):
                return {
                    "Versions": [
                        {
                            "FunctionName": "keep-glm52-h1g-decision",
                            "FunctionArn": base_arn,
                            "Version": "$LATEST",
                        },
                        *[
                            {
                                "FunctionName":
                                    "keep-glm52-h1g-decision",
                                "FunctionArn": arn,
                                "Version": arn.rsplit(":", 1)[-1],
                            }
                            for arn in version_arns
                        ],
                    ]
                }
            function_name = operation[
                operation.index("--function-name") + 1
            ]
            if command == ("lambda", "get-function"):
                return {
                    "Configuration": {"FunctionArn": function_name},
                    "Code": {"RepositoryType": "S3"},
                }
            if command == ("lambda", "get-policy"):
                return {"Policy": "{}", "RevisionId": "policy"}
            if command == ("lambda", "get-function-concurrency"):
                assert function_name == "keep-glm52-h1g-decision"
                return {"ReservedConcurrentExecutions": 1}
            if command == ("lambda", "list-event-source-mappings"):
                return {"EventSourceMappings": []}
            raise AssertionError(operation)

    spec = LiveReadSpec(
        family="lambda",
        operation="lambda.inspect_complete",
        parameters={
            "run_id": RUN_ID,
            "cli_plan": {
                "schema_version": 2,
                "projections": {
                    "function_arn": {
                        "source": 0,
                        "path": ["FunctionArn"],
                        "transform": "identity",
                    },
                    "code_sha256": {
                        "source": 1,
                        "path": ["responses", 0, "Code"],
                        "transform": "canonical_sha256",
                    },
                    "policy_sha256": {
                        "source": 2,
                        "path": ["responses"],
                        "transform": "canonical_sha256",
                    },
                },
            },
        },
        identity_field="function_arn",
        expected_items=tuple(
            {
                "function_arn": arn,
                "code_sha256": ZERO,
                "policy_sha256": ZERO,
            }
            for arn in version_arns
        ),
    )
    runner = Runner()
    page = module.AwsCliLiveReader(
        runner=runner,
        clock=lambda: NOW,
    ).read_page(spec=spec, continuation_token=None)
    assert tuple(item["function_arn"] for item in page.items) == version_arns
    assert sum(
        operation[:2] == ("lambda", "list-versions-by-function")
        for operation in runner.operations
    ) == 1
    version_bound = {
        operation[operation.index("--function-name") + 1]
        for operation in runner.operations
        if operation[:2] in {
            ("lambda", "get-function"),
            ("lambda", "get-policy"),
            ("lambda", "list-event-source-mappings"),
        }
    }
    assert version_bound == set(version_arns)
    assert all("--no-paginate" in operation for operation in runner.operations)


def test_debug_cli_reader_disables_cli_pagination_on_every_call() -> None:
    module = _load_script()

    class Runner:
        def __init__(self) -> None:
            self.operations: list[tuple[str, ...]] = []

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            self.operations.append(operation)
            return {"logGroups": []}

    spec = LiveReadSpec(
        family="logs",
        operation="logs.inspect_complete",
        parameters={
            "cli_plan": {"schema_version": 2, "projections": {}},
        },
        identity_field="log_group_arn",
        expected_items=(),
    )
    runner = Runner()
    module.AwsCliLiveReader(
        runner=runner,
        clock=lambda: NOW,
    ).read_page(spec=spec, continuation_token=None)
    assert runner.operations == [
        (
            "logs",
            "describe-log-groups",
            "--log-group-name-prefix",
            "/aws/lambda/keep-glm52-h1g",
            "--no-paginate",
        ),
        (
            "logs",
            "describe-log-groups",
            "--log-group-name-prefix",
            f"/aws/lambda/{RUN_ID}",
            "--no-paginate",
        ),
    ]


@pytest.mark.parametrize(
    ("operation", "error_code", "expected"),
    (
        (
            ("s3api", "get-bucket-lifecycle-configuration", "--bucket", "b"),
            "NoSuchLifecycleConfiguration",
            {"Rules": None},
        ),
        (
            ("s3api", "get-bucket-replication", "--bucket", "b"),
            "ReplicationConfigurationNotFoundError",
            {"ReplicationConfiguration": None},
        ),
        (
            ("lambda", "get-policy", "--function-name", "function:version"),
            "ResourceNotFoundException",
            {"Policy": None},
        ),
    ),
)
def test_cli_runner_classifies_only_authenticated_expected_absence(
    operation: tuple[str, ...],
    error_code: str,
    expected: dict[str, object],
) -> None:
    module = _load_script()

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            command,
            254,
            "",
            (
                f"An error occurred ({error_code}) when calling the "
                "operation: absent"
            ),
        )

    runner = module.AwsCliCommandRunner(subprocess_run=fake_run)
    assert runner.run_json(operation, timeout_seconds=5) == expected
    with pytest.raises(ValueError):
        runner.run_json(
            ("s3api", "get-bucket-versioning", "--bucket", "b"),
            timeout_seconds=5,
        )


def test_sdk_runner_is_allowlisted_one_attempt_and_preserves_request_id() -> None:
    module = _load_script()
    configurations: list[dict[str, object]] = []
    calls: list[tuple[str, dict[str, object]]] = []

    def config_factory(**kwargs: object) -> object:
        configurations.append(dict(kwargs))
        return object()

    class Client:
        def describe_log_groups(self, **kwargs: object) -> dict[str, object]:
            calls.append(("describe_log_groups", dict(kwargs)))
            return {
                "logGroups": [],
                "ResponseMetadata": {"RequestId": "logs-request-1"},
            }

    class Session:
        def client(self, service: str, *, config: object) -> Client:
            assert service == "logs"
            return Client()

    runner = module.AwsSdkCommandRunner(
        session_factory=lambda **kwargs: Session(),
        config_factory=config_factory,
    )
    response = runner.run_json(
        ("logs", "describe-log-groups", "--no-paginate"),
        timeout_seconds=5,
    )
    assert response["ResponseMetadata"]["RequestId"] == "logs-request-1"
    assert configurations == [
        {
            "connect_timeout": 1,
            "read_timeout": 2,
            "max_pool_connections": module.SDK_MAX_WORKERS,
            "retries": {"total_max_attempts": 1, "mode": "standard"},
        }
    ]
    assert calls == [("describe_log_groups", {})]
    with pytest.raises(ValueError, match="allowlist"):
        runner.run_json(("ec2", "run-instances"), timeout_seconds=5)


def test_sdk_runner_fails_closed_with_clear_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    real_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name == "boto3" or name.startswith("botocore"):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(
        RuntimeError,
        match="boto3 and botocore are required for live H.1d",
    ):
        module.AwsSdkCommandRunner()


def test_sdk_runner_authenticates_exact_s3_expected_absence() -> None:
    module = _load_script()

    class ExpectedAbsence(Exception):
        def __init__(self) -> None:
            self.response = {
                "Error": {"Code": "NoSuchLifecycleConfiguration"},
                "ResponseMetadata": {"RequestId": "s3-absence-1"},
            }

    class Client:
        def get_bucket_lifecycle_configuration(
            self, **kwargs: object
        ) -> dict[str, object]:
            assert kwargs == {"Bucket": "retained-bucket"}
            raise ExpectedAbsence

    class Session:
        def client(self, service: str, *, config: object) -> Client:
            assert service == "s3"
            return Client()

    runner = module.AwsSdkCommandRunner(
        session_factory=lambda **kwargs: Session(),
        config_factory=lambda **kwargs: object(),
    )
    assert runner.run_json(
        (
            "s3api",
            "get-bucket-lifecycle-configuration",
            "--bucket",
            "retained-bucket",
            "--no-paginate",
        ),
        timeout_seconds=5,
    ) == {
        "Rules": None,
        "ResponseMetadata": {"RequestId": "s3-absence-1"},
    }


def test_sdk_runner_authenticates_absent_lambda_resource_policy() -> None:
    module = _load_script()

    class ExpectedAbsence(Exception):
        def __init__(self) -> None:
            self.response = {
                "Error": {"Code": "ResourceNotFoundException"},
                "ResponseMetadata": {"RequestId": "lambda-absence-1"},
            }

    class Client:
        def get_policy(self, **kwargs: object) -> dict[str, object]:
            assert kwargs == {"FunctionName": "function:version"}
            raise ExpectedAbsence

    class Session:
        def client(self, service: str, *, config: object) -> Client:
            assert service == "lambda"
            return Client()

    runner = module.AwsSdkCommandRunner(
        session_factory=lambda **kwargs: Session(),
        config_factory=lambda **kwargs: object(),
    )
    assert runner.run_json(
        (
            "lambda",
            "get-policy",
            "--function-name",
            "function:version",
            "--no-paginate",
        ),
        timeout_seconds=5,
    ) == {
        "Policy": None,
        "ResponseMetadata": {"RequestId": "lambda-absence-1"},
    }


def test_core_prepares_concurrent_reader_under_shared_seven_second_deadline() -> None:
    expected = _expected_state()
    services, reader, _spend, _probe = _services(expected)

    class PreparingReader(_Reader):
        def __init__(self, state: H1dExpectedState) -> None:
            super().__init__(state)
            self.prepared: list[object] = []

        def prepare(
            self,
            specs: tuple[LiveReadSpec, ...],
            *,
            deadline: datetime,
        ) -> None:
            self.prepared.append((specs, deadline))

    preparing = PreparingReader(expected)
    services = replace(services, reader=preparing)
    inspect_h1d_live_authority(_request(expected), expected, services)
    assert preparing.prepared == [
        (expected.specs, NOW + timedelta(seconds=7))
    ]


def test_production_s3_spend_pagination_uses_service_max_keys_not_cli_max_items() -> None:
    module = _load_script()

    class Runner:
        def __init__(self) -> None:
            self.operation: tuple[str, ...] | None = None

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            self.operation = operation
            return {
                "Versions": [],
                "DeleteMarkers": [],
                "IsTruncated": False,
            }

    runner = Runner()
    store = module.AwsCliSpendObjectStore(
        runner=runner,
        bucket="keep-glm52-models",
        exact_versions={},
    )
    store.list_namespace(prefix="campaigns/run/", continuation_token=None)
    assert "--max-items" not in runner.operation
    assert runner.operation == (
        "s3api",
        "list-object-versions",
        "--bucket",
        "keep-glm52-models",
        "--prefix",
        "campaigns/run/",
        "--max-keys",
        "1000",
        "--no-paginate",
    )


def test_live_page_preserves_each_actual_service_request_id() -> None:
    module = _load_script()
    rule_arn = (
        f"arn:aws:events:us-west-2:{ACCOUNT_ID}:rule/decision"
    )
    responses = {
        ("events", "list-rules"): {
            "Rules": [{"Arn": rule_arn, "Name": "decision"}],
            "ResponseMetadata": {"RequestId": "events-list-1"},
        },
        ("events", "describe-rule"): {
            "Arn": rule_arn,
            "Name": "decision",
            "ResponseMetadata": {"RequestId": "events-describe-1"},
        },
        ("events", "list-targets-by-rule"): {
            "Targets": [],
            "ResponseMetadata": {"RequestId": "events-targets-1"},
        },
    }

    class Runner:
        def __init__(self) -> None:
            self.list_calls = 0

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            response = deepcopy(responses[operation[:2]])
            if operation[:2] == ("events", "list-rules"):
                self.list_calls += 1
                response["ResponseMetadata"]["RequestId"] = (
                    f"events-list-{self.list_calls}"
                )
            return response

    spec = LiveReadSpec(
        family="eventbridge",
        operation="eventbridge.inspect_complete",
        parameters={
            "cli_plan": {
                "schema_version": 2,
                "projections": {
                    "rule_arn": {
                        "source": 0,
                        "path": ["Arn"],
                        "transform": "identity",
                    },
                    "targets_sha256": {
                        "source": 2,
                        "path": ["items"],
                        "transform": "canonical_sha256",
                    },
                },
            }
        },
        identity_field="rule_arn",
        expected_items=(
            {
                "rule_arn": rule_arn,
                "targets_sha256": canonical_sha256([]),
            },
        ),
    )
    page = module.AwsCliLiveReader(
        runner=Runner(),
        clock=lambda: NOW,
    ).read_page(spec=spec, continuation_token=None)
    assert page.request_id == "events-list-1"
    assert page.service_request_ids == (
        "events-list-1",
        "events-list-2",
        "events-describe-1",
        "events-targets-1",
    )


def test_sdk_reader_fans_out_families_concurrently_with_fixed_bound() -> None:
    module = _load_script()
    specs = _expected_state().specs[:3]
    barrier = threading.Barrier(len(specs))
    lock = threading.Lock()
    active = 0
    peak = 0

    class PageReader:
        def __init__(self, **kwargs: object) -> None:
            pass

        def read_page(
            self,
            *,
            spec: LiveReadSpec,
            continuation_token: str | None,
        ) -> LiveReadPage:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait(timeout=1)
            with lock:
                active -= 1
            return LiveReadPage(
                family=spec.family,
                operation=spec.operation,
                request_token=None,
                page_index=0,
                items=spec.expected_items,
                next_token=None,
                request_id=f"{spec.family}-request",
                observed_at="2026-07-29T01:00:00Z",
            )

    reader = module.AwsSdkLiveReader(
        runner=object(),
        clock=lambda: NOW,
        max_workers=3,
        page_reader_factory=PageReader,
    )
    reader.prepare(specs, deadline=NOW + timedelta(seconds=7))
    assert peak == 3
    for spec in specs:
        assert reader.read_page(
            spec=spec,
            continuation_token=None,
        ).items == spec.expected_items


def test_sdk_reader_cancels_pending_families_when_shared_deadline_elapses() -> None:
    module = _load_script()
    specs = _expected_state().specs[:2]
    release = threading.Event()
    calls: dict[str, int] = {}
    clock_values = iter((NOW, NOW + timedelta(seconds=8)))

    class PageReader:
        def __init__(self, **kwargs: object) -> None:
            pass

        def read_page(
            self,
            *,
            spec: LiveReadSpec,
            continuation_token: str | None,
        ) -> LiveReadPage:
            calls[spec.family] = calls.get(spec.family, 0) + 1
            release.wait(timeout=1)
            return LiveReadPage(
                family=spec.family,
                operation=spec.operation,
                request_token=None,
                page_index=0,
                items=spec.expected_items,
                next_token=None,
                request_id=f"{spec.family}-request",
                observed_at="2026-07-29T01:00:00Z",
            )

    reader = module.AwsSdkLiveReader(
        runner=object(),
        clock=lambda: next(clock_values),
        max_workers=1,
        page_reader_factory=PageReader,
    )
    try:
        with pytest.raises(ValueError, match="deadline"):
            reader.prepare(specs, deadline=NOW + timedelta(seconds=7))
    finally:
        release.set()
    assert all(count == 1 for count in calls.values())
    assert len(calls) <= 1


def test_sdk_reader_is_quiescent_before_timeout_returns() -> None:
    module = _load_script()
    spec = _expected_state().specs[0]
    calls: list[str | None] = []

    class PageReader:
        def __init__(self, **kwargs: object) -> None:
            pass

        def read_page(
            self,
            *,
            spec: LiveReadSpec,
            continuation_token: str | None,
        ) -> LiveReadPage:
            calls.append(continuation_token)
            time.sleep(0.04)
            return LiveReadPage(
                family=spec.family,
                operation=spec.operation,
                request_token=continuation_token,
                page_index=len(calls) - 1,
                items=spec.expected_items,
                next_token=(
                    "must-not-read-after-cancel"
                    if continuation_token is None
                    else None
                ),
                request_id=f"request-{len(calls)}",
                observed_at="2026-07-29T01:00:00Z",
            )

    reader = module.AwsSdkLiveReader(
        runner=object(),
        clock=lambda: datetime.now(timezone.utc),
        max_workers=1,
        page_reader_factory=PageReader,
    )
    started = time.monotonic()
    with pytest.raises(ValueError, match="deadline"):
        reader.prepare(
            (spec,),
            deadline=datetime.now(timezone.utc) + timedelta(seconds=0.01),
        )
    elapsed = time.monotonic() - started
    calls_at_return = tuple(calls)
    time.sleep(0.08)
    assert elapsed >= 0.035
    assert calls_at_return == (None,)
    assert tuple(calls) == calls_at_return


@pytest.mark.parametrize(
    ("family", "unrelated", "campaign_foreign"),
    (
        (
            "cloudformation",
            {
                "StackId": (
                    "arn:aws:cloudformation:us-west-2:246813579024:"
                    "stack/unrelated/uuid"
                ),
                "StackName": "unrelated",
            },
            {
                "StackId": (
                    "arn:aws:cloudformation:us-west-2:246813579024:"
                    "stack/keep-glm52-h1g-foreign/uuid"
                ),
                "StackName": "keep-glm52-h1g-foreign",
            },
        ),
        (
            "lambda",
            {
                "FunctionArn": (
                    f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:"
                    "function:unrelated"
                ),
                "FunctionName": "unrelated",
            },
            {
                "FunctionArn": (
                    f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:"
                    "function:keep-glm52-h1g-foreign"
                ),
                "FunctionName": "keep-glm52-h1g-foreign",
            },
        ),
        (
            "iam",
            {
                "Arn": f"arn:aws:iam::{ACCOUNT_ID}:role/unrelated",
                "RoleName": "unrelated",
            },
            {
                "Arn": (
                    f"arn:aws:iam::{ACCOUNT_ID}:"
                    "role/keep-glm52-h1g-foreign"
                ),
                "RoleName": "keep-glm52-h1g-foreign",
            },
        ),
        (
            "eventbridge",
            {
                "Arn": (
                    f"arn:aws:events:{REGION}:{ACCOUNT_ID}:rule/unrelated"
                ),
                "Name": "unrelated",
            },
            {
                "Arn": (
                    f"arn:aws:events:{REGION}:{ACCOUNT_ID}:"
                    "rule/keep-glm52-h1g-foreign"
                ),
                "Name": "keep-glm52-h1g-foreign",
            },
        ),
        (
            "scheduler",
            {
                "Arn": (
                    f"arn:aws:scheduler:{REGION}:{ACCOUNT_ID}:"
                    "schedule/default/unrelated"
                ),
                "Name": "unrelated",
            },
            {
                "Arn": (
                    f"arn:aws:scheduler:{REGION}:{ACCOUNT_ID}:"
                    "schedule/default/keep-glm52-h1g-foreign"
                ),
                "Name": "keep-glm52-h1g-foreign",
            },
        ),
        (
            "sqs",
            (
                f"https://sqs.{REGION}.amazonaws.com/"
                f"{ACCOUNT_ID}/unrelated"
            ),
            (
                f"https://sqs.{REGION}.amazonaws.com/{ACCOUNT_ID}/"
                "keep-glm52-h1g-foreign"
            ),
        ),
        (
            "s3",
            {"Name": "unrelated"},
            {"Name": "keep-glm52-h1g-foreign"},
        ),
    ),
)
def test_inventory_scope_ignores_unrelated_but_keeps_campaign_foreign(
    family: str,
    unrelated: object,
    campaign_foreign: object,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    del authority
    spec = next(spec for spec in expected.specs if spec.family == family)
    assert module._inventory_item_in_scope(
        spec,
        unrelated,
    ) is False
    assert module._inventory_item_in_scope(
        spec,
        campaign_foreign,
    ) is True


@pytest.mark.parametrize(
    ("family", "expected_argument"),
    (
        (
            "ssm",
            (
                "--filters",
                f"Key=tag:campaign-run-id,Values={RUN_ID}",
            ),
        ),
        (
            "cloudwatch",
            ("--alarm-name-prefix", "keep-glm52-h1g"),
        ),
        (
            "logs",
            ("--log-group-name-prefix", "/aws/lambda/keep-glm52-h1g"),
        ),
    ),
)
def test_inventory_scope_uses_service_side_campaign_filter(
    family: str,
    expected_argument: tuple[str, str],
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    del authority
    spec = next(spec for spec in expected.specs if spec.family == family)
    arguments = module._closed_inventory_arguments(spec)
    index = arguments.index(expected_argument[0])
    assert tuple(arguments[index:index + 2]) == expected_argument


@pytest.mark.parametrize(
    ("family", "flag", "support_prefix", "run_prefix"),
    (
        (
            "eventbridge",
            "--name-prefix",
            "keep-glm52-h1g",
            RUN_ID,
        ),
        (
            "scheduler",
            "--name-prefix",
            "keep-glm52-h1g",
            RUN_ID,
        ),
        (
            "sqs",
            "--queue-name-prefix",
            "keep-glm52-h1g",
            RUN_ID,
        ),
        (
            "cloudwatch",
            "--alarm-name-prefix",
            "keep-glm52-h1g",
            RUN_ID,
        ),
        (
            "logs",
            "--log-group-name-prefix",
            "/aws/lambda/keep-glm52-h1g",
            f"/aws/lambda/{RUN_ID}",
        ),
    ),
)
def test_inventory_scope_unions_support_and_run_id_prefixes(
    family: str,
    flag: str,
    support_prefix: str,
    run_prefix: str,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    del authority
    spec = next(spec for spec in expected.specs if spec.family == family)
    assert module._closed_inventory_argument_sets(spec) == [
        [flag, support_prefix],
        [flag, run_prefix],
    ]


@pytest.mark.parametrize(
    ("family", "item"),
    (
        (
            "eventbridge",
            {
                "Arn": (
                    f"arn:aws:events:{REGION}:{ACCOUNT_ID}:rule/"
                    f"{RUN_ID}-foreign"
                ),
                "Name": f"{RUN_ID}-foreign",
            },
        ),
        (
            "scheduler",
            {
                "Arn": (
                    f"arn:aws:scheduler:{REGION}:{ACCOUNT_ID}:"
                    f"schedule/default/{RUN_ID}-foreign"
                ),
                "Name": f"{RUN_ID}-foreign",
            },
        ),
        (
            "sqs",
            (
                f"https://sqs.{REGION}.amazonaws.com/{ACCOUNT_ID}/"
                f"{RUN_ID}-foreign"
            ),
        ),
        (
            "cloudwatch",
            {
                "AlarmArn": (
                    f"arn:aws:cloudwatch:{REGION}:{ACCOUNT_ID}:"
                    f"alarm:{RUN_ID}-foreign"
                )
            },
        ),
        (
            "logs",
            {
                "arn": (
                    f"arn:aws:logs:{REGION}:{ACCOUNT_ID}:log-group:"
                    f"/aws/lambda/{RUN_ID}-foreign"
                )
            },
        ),
    ),
)
def test_inventory_scope_keeps_run_id_only_foreign_resources(
    family: str,
    item: object,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    del authority
    spec = next(spec for spec in expected.specs if spec.family == family)
    assert module._inventory_item_in_scope(spec, item) is True


def test_ec2_inventory_unions_run_tag_global_p5_and_exact_sky_name() -> None:
    module = _load_script()
    spec = next(
        spec for spec in _expected_state().specs
        if spec.family == "ec2"
    )
    arguments = module._closed_inventory_argument_sets(spec)
    filters = {
        tuple(
            item for item in argument_set[
                argument_set.index("--filters") + 1:
                argument_set.index("--max-results")
            ]
        )
        for argument_set in arguments
    }
    assert filters == {
        (f"Name=tag:campaign-run-id,Values={RUN_ID}",),
        ("Name=instance-type,Values=p5.48xlarge",),
        (f"Name=tag:Name,Values={RUN_ID}",),
    }


@pytest.mark.parametrize(
    "tags",
    (
        [],
        [{"Key": "campaign-run-id", "Value": "foreign"}],
    ),
)
def test_ec2_union_exposes_p5_with_missing_or_altered_campaign_tag(
    tags: list[dict[str, str]],
) -> None:
    module = _load_script()
    expected = _expected_state()
    source_spec = next(
        item for item in expected.specs if item.family == "ec2"
    )
    support = deepcopy(source_spec.expected_items[0])
    spec = LiveReadSpec(
        family="ec2",
        operation="ec2.inspect_complete",
        parameters={
            "cli_plan": {
                "schema_version": 2,
                "projections": {
                    "instance_id": {
                        "source": 0,
                        "path": ["InstanceId"],
                        "transform": "identity",
                    }
                },
            }
        },
        identity_field="instance_id",
        expected_items=(
            {"instance_id": support["instance_id"]},
        ),
    )
    support_inventory = {
        "InstanceId": support["instance_id"],
        "ImageId": support["ami_id"],
        "InstanceType": support["instance_type"],
        "BlockDeviceMappings": [],
        "Tags": [
            {"Key": "campaign-run-id", "Value": RUN_ID},
            {"Key": "glm52-role", "Value": "support-host"},
        ],
    }
    p5 = {
        "InstanceId": "i-00000000000000099",
        "ImageId": "ami-0123456789abcdef0",
        "InstanceType": "p5.48xlarge",
        "BlockDeviceMappings": [],
        "Tags": tags,
    }

    class Runner:
        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            command = operation[:2]
            if command == ("ec2", "describe-instances"):
                if "Name=instance-type,Values=p5.48xlarge" in operation:
                    return {"Instances": [p5]}
                if f"Name=tag:Name,Values={RUN_ID}" in operation:
                    return {"Instances": []}
                return {"Instances": [support_inventory]}
            if command == ("ec2", "describe-volumes"):
                return {"Volumes": []}
            if command == ("ec2", "describe-images"):
                image_id = operation[operation.index("--image-ids") + 1]
                return {"Images": [{"ImageId": image_id}]}
            if command == ("ec2", "describe-instance-attribute"):
                instance_id = operation[operation.index("--instance-id") + 1]
                return {
                    "InstanceId": instance_id,
                    "UserData": {"Value": "encoded"},
                }
            raise AssertionError(operation)

    page = module.AwsCliLiveReader(
        runner=Runner(),
        clock=lambda: NOW,
    ).read_page(spec=spec, continuation_token=None)
    assert {item["instance_id"] for item in page.items} == {
        support["instance_id"],
        p5["InstanceId"],
    }


@pytest.mark.parametrize(
    ("family", "field", "substitution"),
    (
        ("cloudformation", "status", "ROLLBACK_COMPLETE"),
        ("eventbridge", "state", "DISABLED"),
        ("scheduler", "state", "DISABLED"),
        ("s3", "versioning", "Suspended"),
    ),
)
def test_rehashed_caller_spec_cannot_override_code_owned_live_semantics(
    family: str,
    field: str,
    substitution: object,
) -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    forged = _substitute_expected_item(
        expected,
        family=family,
        field=field,
        substitution=substitution,
    )
    coordinate = authority.sources["task6_templates"]
    key = coordinate["key"]
    templates = json.loads(authority.store.raw_by_key[key])
    body = dict(templates)
    body.pop("canonical_body_sha256")
    body["h1d_specs_identity_sha256"] = authority._specs_identity(forged)
    forged_templates = {
        **body,
        "canonical_body_sha256": module.canonical_sha256(body),
    }
    forged_raw = module.canonical_json_bytes(forged_templates) + b"\n"
    authority.store.raw_by_key[key] = forged_raw
    sources = deepcopy(authority.sources)
    sources["task6_templates"]["file_sha256"] = hashlib.sha256(
        forged_raw
    ).hexdigest()
    forged_authority = module.AwsExpectedStateAuthority(
        store=authority.store,
        sources=sources,
        trusted_source_reader=(
            _trusted_source_reader_after_coordinate_update(
                module,
                authority,
                sources,
            )
        ),
        clock=lambda: NOW,
    )
    with pytest.raises(ValueError, match="source-bound|sealed activation"):
        forged_authority.authenticate(forged)


def test_s3_health_distinguishes_retained_model_from_support_rehearsal() -> None:
    items = _expected_items()
    retained = {
        **items["s3"][0],
        "bucket_class": "retained_model_evidence",
    }
    rehearsal = {
        "bucket": "keep-glm52-h1g-support-rehearsal",
        "bucket_class": "support_rehearsal",
        "versioning": None,
        "policy_sha256": canonical_sha256(None),
        "lifecycle": None,
        "replication": None,
    }
    items["s3"] = (retained, rehearsal)
    expected = _expected_state(items=items)
    services, _reader, _spend, _probe = _services(expected)
    inspect_h1d_live_authority(_request(expected), expected, services)


def test_activation_scoped_dynamodb_keys_replace_invented_fixed_keys() -> None:
    module = _load_script()
    authority, expected = _production_expected_authority_fixture(module)
    activation = expected.activation_id
    ddb_items = (
        {
            "key": f"RUN#{RUN_ID}|ACTIVATION_INDEX",
            "record_type": "glm52_production_activation_index",
            "body_sha256": "2" * 64,
            "consistent_read": True,
        },
        {
            "key": f"RUN#{RUN_ID}|ACTIVATION#{activation}#CONTROL",
            "record_type": "glm52_production_control",
            "body_sha256": "3" * 64,
            "consistent_read": True,
        },
        {
            "key": (
                f"RUN#{RUN_ID}|ACTIVATION#{activation}#"
                "ACTION#00000001#SKY#00000001"
            ),
            "record_type": "glm52_production_action",
            "body_sha256": "4" * 64,
            "consistent_read": True,
        },
    )
    specs = tuple(
        replace(spec, expected_items=ddb_items)
        if spec.family == "dynamodb"
        else spec
        for spec in expected.specs
    )
    forged = build_h1d_expected_state(
        account_id=expected.account_id,
        region=expected.region,
        run_id=expected.run_id,
        activation_id=expected.activation_id,
        manifest_identity_sha256=expected.manifest_identity_sha256,
        support_host_instance_id=expected.support_host_instance_id,
        must_start_by=expected.must_start_by,
        execution_deadline=expected.execution_deadline,
        specs=specs,
    )
    coordinate = authority.sources["task6_templates"]
    key = coordinate["key"]
    templates = json.loads(authority.store.raw_by_key[key])
    body = dict(templates)
    body.pop("canonical_body_sha256")
    body["h1d_specs_identity_sha256"] = authority._specs_identity(forged)
    forged_templates = {
        **body,
        "canonical_body_sha256": module.canonical_sha256(body),
    }
    forged_raw = module.canonical_json_bytes(forged_templates) + b"\n"
    authority.store.raw_by_key[key] = forged_raw
    sources = deepcopy(authority.sources)
    sources["task6_templates"]["file_sha256"] = hashlib.sha256(
        forged_raw
    ).hexdigest()
    with pytest.raises(ValueError, match="sealed activation"):
        module.AwsExpectedStateAuthority(
            store=authority.store,
            sources=sources,
            trusted_source_reader=(
                _trusted_source_reader_after_coordinate_update(
                    module,
                    authority,
                    sources,
                )
            ),
            clock=lambda: NOW,
        ).authenticate(forged)
