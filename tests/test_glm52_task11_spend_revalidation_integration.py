"""Retained worker-launch authority reaches the Task 11 spend adapter."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.dynamodb import encode_item
from glm52_enforcement.task11_production import (
    Task11ReserveReader,
    _bind_spend_client_token,
    _finalize_spend_token_custody,
    _read_retained_terminal_v2_sources,
    _retained_liability_settlement_authority,
    _retained_known_launch_authority,
    _retained_launch_authority,
)


def _record() -> dict[str, object]:
    value: dict[str, object] = {
        "state": "ALLOCATION_OPEN", "activation_id": "approved-20260728",
        "activation_ordinal": 1, "generation": 1, "allocation_ordinal": 1,
        "campaign_identity_sha256": "1" * 64, "sky_action_key": "ACTION#00000001#SKY_POST#00000001",
        "sky_request_id": "sky-request-1", "sky_job_name": "glm52-sky-20260724",
        "task_yaml_sha256": "2" * 64, "request_body_sha256": "3" * 64,
        "ec2_client_token": "e" * 64,
    }
    return _seal_launch(value)


def _seal_launch(value: dict[str, object]) -> dict[str, object]:
    tags = {
        "Project": "KEEP", "Campaign": "GLM-5.2", "RunId": "glm52-sky-20260724", "Market": "on-demand",
        "campaign-identity-sha256": value["campaign_identity_sha256"], "activation-id": value["activation_id"],
        "activation-ordinal-text": f"{value['activation_ordinal']:08d}",
        "generation-text": f"{value['generation']:08d}",
        "allocation-ordinal-text": f"{value['allocation_ordinal']:08d}",
        "action-key": value["sky_action_key"], "sky-request-id": value["sky_request_id"],
        "sky-job-name": value["sky_job_name"], "sky-task-name": "glm52-production",
        "task-yaml-sha256": value["task_yaml_sha256"], "request-body-sha256": value["request_body_sha256"],
    }
    value["expected_worker_tags_sha256"] = canonical_sha256(tags)
    value["gpu_liability_reserve_ledger_identity_sha256"] = canonical_sha256(
        _reserve_body(value)
    )
    launch_body = dict(value)
    launch_body.pop("canonical_body_sha256", None)
    value["canonical_body_sha256"] = canonical_sha256(launch_body)
    return value


def test_retained_open_launch_is_the_sole_exact_dynamic_tag_and_token_authority() -> None:
    authority = _retained_launch_authority((_record(),), activation_id="approved-20260728")
    tags, token = authority[("approved-20260728", 1, 1)]
    assert tags["campaign-identity-sha256"] == "1" * 64
    assert tags["action-key"] == "ACTION#00000001#SKY_POST#00000001"
    assert token == "e" * 64


@pytest.mark.parametrize("mutation", ("foreign_dynamic_tag", "hash_drift", "missing_authority"))
def test_retained_authority_mutants_fail_closed(mutation: str) -> None:
    record = _record()
    if mutation == "foreign_dynamic_tag":
        record["sky_request_id"] = "foreign"
    elif mutation == "hash_drift":
        record["expected_worker_tags_sha256"] = "f" * 64
    else:
        record["state"] = "ALLOCATION_CLOSED"
    if mutation == "missing_authority":
        assert _retained_launch_authority((record,), activation_id="approved-20260728") == {}
    else:
        with pytest.raises(RuntimeError):
            _retained_launch_authority((record,), activation_id="approved-20260728")


@pytest.mark.parametrize("mutation", ("ec2_only", "reserve_only", "duplicate", "token_mismatch"))
def test_real_spend_custody_requires_exact_two_sided_bijection(mutation: str) -> None:
    expected = _retained_launch_authority((_record(),), activation_id="approved-20260728")
    identity = ("approved-20260728", 1, 1)
    custody: dict[tuple[str, int, int], object] = {}
    if mutation != "reserve_only":
        _bind_spend_client_token(custody, allocation_identity=identity, client_token="e" * 64, source="ec2")
    if mutation != "ec2_only":
        if mutation == "token_mismatch":
            with pytest.raises(RuntimeError):
                _bind_spend_client_token(custody, allocation_identity=identity, client_token="f" * 64, source="reserve")
            return
        _bind_spend_client_token(custody, allocation_identity=identity, client_token=("f" * 64 if mutation == "token_mismatch" else "e" * 64), source="reserve")
    if mutation == "duplicate":
        with pytest.raises(RuntimeError):
            _bind_spend_client_token(
                custody,
                allocation_identity=identity,
                client_token="f" * 64,
                source="ec2",
            )
        return
    with pytest.raises(RuntimeError):
        _finalize_spend_token_custody(custody, expected)


def _reserve_body(
    record: dict[str, object],
    *,
    request_identity_sha256: str = "a" * 64,
) -> dict[str, object]:
    activation_id = record["activation_id"]
    generation = record["generation"]
    allocation = record["allocation_ordinal"]
    assert isinstance(activation_id, str)
    assert isinstance(generation, int)
    assert isinstance(allocation, int)
    reserve_key = (
        f"GPU_LIABILITY_RESERVE#{activation_id}#"
        f"{generation:08d}#{allocation:08d}"
    )
    return {
        "schema_version": 1,
        "record_type": "glm52_gpu_liability_reserve_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "activation_id": activation_id,
        "generation": generation,
        "generation_text": f"{generation:08d}",
        "allocation_ordinal": allocation,
        "allocation_ordinal_text": f"{allocation:08d}",
        "ec2_client_token": record["ec2_client_token"],
        "request_identity_sha256": request_identity_sha256,
        "ledger_predecessor_identity_sha256": "b" * 64,
        "gpu_reserve_seconds": 900,
        "gpu_reserve_cost_usd": "13.76",
        "root_volume_gib": 300,
        "root_volume_tail_usd_max": "0.01",
        "residual_liability_approval_identity_sha256": "c" * 64,
        "epoch": 1,
        "revision": 1,
        "nonce_owner_identity_sha256": "d" * 64,
        "state": "HELD",
        "observed_at": "2026-07-29T16:00:00Z",
        "reserve_key": reserve_key,
    }


def _reserve_item(
    record: dict[str, object],
    *,
    request_identity_sha256: str = "a" * 64,
) -> dict[str, object]:
    body = _reserve_body(
        record,
        request_identity_sha256=request_identity_sha256,
    )
    reserve_key = body["reserve_key"]
    return encode_item(
        {
            "PK": "RUN#glm52-sky-20260724",
            "SK": reserve_key,
            **body,
            "canonical_body_sha256": canonical_sha256(body),
        }
    )


def _ddb_page(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "Items": items,
        "Count": len(items),
        "ScannedCount": len(items),
        "ResponseMetadata": {
            "HTTPStatusCode": 200,
            "RequestId": "ddb-current-activation",
            "RetryAttempts": 0,
        },
    }


def _settlement(
    launch: dict[str, object],
) -> dict[str, object]:
    no_instance = launch["state"] == "REJECTED_NO_INSTANCE"
    merged = (
        []
        if no_instance
        else [
            {
                "allocation_ordinal": launch["allocation_ordinal"],
                "instance_id": "i-0123456789abcdef0",
                "instance_terminal_identity_sha256": "8" * 64,
                "spend_allocation_close_identity_sha256": "9" * 64,
            }
        ]
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_production_worker_launch_liability_settlement",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "campaign_identity_sha256": launch["campaign_identity_sha256"],
        "activation_id": launch["activation_id"],
        "activation_ordinal": launch["activation_ordinal"],
        "generation": launch["generation"],
        "allocation_ordinal": launch["allocation_ordinal"],
        "worker_launch_identity_sha256": launch["canonical_body_sha256"],
        "worker_launch_liability_identity_sha256": "5" * 64,
        "settlement_kind": (
            "NO_INSTANCE_POSITIVE_REJECTION"
            if no_instance
            else "ALL_INSTANCES_TERMINAL_AND_SPEND_CLOSED"
        ),
        "terminal_v2_key": "terminal-v2.json",
        "terminal_v2_version_id": "terminal-version",
        "terminal_v2_body_sha256": "6" * 64,
        "terminal_v2_allocations_array_sha256": canonical_sha256(merged),
        "post_terminal_allocations": [],
        "post_terminal_allocations_array_sha256": canonical_sha256([]),
        "merged_final_allocations": merged,
        "merged_final_allocations_array_sha256": canonical_sha256(merged),
        "final_worker_cardinality": str(len(merged)),
        "service_rejection_evidence_sha256": (
            "7" * 64 if no_instance else None
        ),
        "instance_terminal_evidence_array_sha256": canonical_sha256(
            [] if no_instance else ["8" * 64]
        ),
        "spend_close_evidence_array_sha256": canonical_sha256(
            [] if no_instance else ["9" * 64]
        ),
        "gpu_liability_reserve_ledger_identity_sha256": launch[
            "gpu_liability_reserve_ledger_identity_sha256"
        ],
        "gpu_liability_reserve_release_identity_sha256": "a" * 64,
        "ebs_liability_reserve_cost_usd": "0.01",
        "residual_liability_approval_identity_sha256": "b" * 64,
        "final_spend_ledger_head_identity_sha256": "c" * 64,
        "prior_incident_identity_sha256": None,
        "owner_execution_arn": "arn:aws:states:us-west-2:246813579024:execution:test",
        "owner_state_machine_version_arn": "arn:aws:states:us-west-2:246813579024:stateMachine:test:1",
        "owner_dispatch_identity_sha256": "d" * 64,
        "owner_invocation_nonce_sha256": "e" * 64,
        "transaction_client_request_token_sha256": "f" * 64,
        "settled_at": "2026-07-29T16:01:00Z",
    }
    return _seal_settlement(body)


def _seal_settlement(body: dict[str, object]) -> dict[str, object]:
    merged = body["merged_final_allocations"]
    post = body["post_terminal_allocations"]
    assert isinstance(merged, list)
    assert isinstance(post, list)
    body["merged_final_allocations_array_sha256"] = canonical_sha256(
        merged
    )
    body["post_terminal_allocations_array_sha256"] = canonical_sha256(post)
    body["final_worker_cardinality"] = str(len(merged))
    body["instance_terminal_evidence_array_sha256"] = canonical_sha256(
        sorted(item.get("instance_terminal_identity_sha256") for item in merged)
    )
    body["spend_close_evidence_array_sha256"] = canonical_sha256(
        sorted(item.get("spend_allocation_close_identity_sha256") for item in merged)
    )
    body.pop("canonical_body_sha256", None)
    body["canonical_body_sha256"] = canonical_sha256(body)
    return body


def _terminal_v2(
    launch: dict[str, object],
    settlement: dict[str, object],
) -> dict[str, object]:
    allocations = list(settlement["merged_final_allocations"])
    worker_launch_evidence = [
        {
            "allocation_ordinal": launch["allocation_ordinal"],
            "ec2_client_token": launch["ec2_client_token"],
            "worker_launch_identity_sha256": launch[
                "canonical_body_sha256"
            ],
            "worker_launch_liability_identity_sha256": settlement[
                "worker_launch_liability_identity_sha256"
            ],
            "instance_terminal_identity_sha256": (
                None if not allocations else "8" * 64
            ),
            "spend_allocation_close_identity_sha256": (
                None if not allocations else "9" * 64
            ),
            "positive_service_rejection_evidence_sha256": (
                "7" * 64 if not allocations else None
            ),
        }
    ]
    worker_launch_liabilities = [
        {
            "activation_id": launch["activation_id"],
            "activation_ordinal": launch["activation_ordinal"],
            "generation": launch["generation"],
            "allocation_ordinal": launch["allocation_ordinal"],
            "ec2_client_token": launch["ec2_client_token"],
            "worker_launch_identity_sha256": launch[
                "canonical_body_sha256"
            ],
            "worker_launch_liability_identity_sha256": settlement[
                "worker_launch_liability_identity_sha256"
            ],
            "gpu_liability_reserve_ledger_identity_sha256": launch[
                "gpu_liability_reserve_ledger_identity_sha256"
            ],
        }
    ]
    terminal: dict[str, object] = {
        field: None
        for field in __import__(
            "glm52_enforcement.records",
            fromlist=["RECORD_FIELDS"],
        ).RECORD_FIELDS["glm52_production_terminal_v2"]
    }
    terminal.update(
        schema_version=2,
        record_type="glm52_production_terminal_v2",
        account_id="246813579024",
        region="us-west-2",
        run_id="glm52-sky-20260724",
        campaign_identity_sha256=launch["campaign_identity_sha256"],
        activation_id=launch["activation_id"],
        activation_ordinal=launch["activation_ordinal"],
        generation=launch["generation"],
        generation_text=f"{launch['generation']:08d}",
        action_key="terminal-v2.json",
        action_identity_sha256="1" * 64,
        handoff={"identity": "2" * 64},
        binding={"identity": "3" * 64},
        final_sky_state="FAILED",
        final_ec2_states=[] if not allocations else ["terminated"],
        request_cardinality="ONE",
        worker_cardinality="ZERO" if not allocations else "ONE",
        allocations=allocations,
        allocations_array_sha256=canonical_sha256(allocations),
        worker_launch_evidence=worker_launch_evidence,
        worker_launch_evidence_array_sha256=canonical_sha256(
            worker_launch_evidence
        ),
        worker_launch_liabilities=worker_launch_liabilities,
        worker_launch_liabilities_array_sha256=canonical_sha256(
            worker_launch_liabilities
        ),
        spend_ledger_head_identity={"identity": "4" * 64},
        remaining_approved_gpu_seconds=0,
        remaining_approved_gpu_usd="0.00",
        terminal_observation_window={"identity": "5" * 64},
        request_evidence=[],
        request_evidence_array_sha256=canonical_sha256([]),
        outcome=(
            "KNOWN_REJECTED_NO_JOB"
            if not allocations
            else "FAILED_TERMINAL_AFTER_ALLOCATION"
        ),
        operator_disposition_required=True,
        writer_function_version_arn="arn:aws:lambda:us-west-2:246813579024:function:terminal:1",
        writer_dispatch_identity_sha256="6" * 64,
        writer_invocation_nonce_sha256="7" * 64,
        created_at="2026-07-29T16:00:30Z",
    )
    terminal["canonical_body_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in terminal.items()
            if key != "canonical_body_sha256"
        }
    )
    return terminal


def _liability(
    launch: dict[str, object],
    settlement: dict[str, object],
) -> dict[str, object]:
    value: dict[str, object] = {
        "record_type": "glm52_production_worker_launch_liability",
        "activation_id": launch["activation_id"],
        "activation_ordinal": launch["activation_ordinal"],
        "generation": launch["generation"],
        "allocation_ordinal": launch["allocation_ordinal"],
        "worker_launch_identity_sha256": launch[
            "canonical_body_sha256"
        ],
        "ec2_client_token": launch["ec2_client_token"],
        "gpu_liability_reserve_ledger_identity_sha256": launch[
            "gpu_liability_reserve_ledger_identity_sha256"
        ],
        "gpu_liability_reserve_release_identity_sha256": settlement[
            "gpu_liability_reserve_release_identity_sha256"
        ],
        "settlement_identity_sha256": settlement[
            "canonical_body_sha256"
        ],
        "state": (
            "SETTLED_NO_INSTANCE_REJECTED"
            if settlement["settlement_kind"]
            == "NO_INSTANCE_POSITIVE_REJECTION"
            else "SETTLED_INSTANCE_CLOSED"
        ),
    }
    value["canonical_body_sha256"] = canonical_sha256(value)
    return value


class _TerminalS3:
    def __init__(
        self,
        terminal: dict[str, object],
        *,
        raw: bytes | None = None,
        response_version_id: str = "terminal-version",
    ) -> None:
        self.raw = (
            canonical_json_bytes(terminal) + b"\n"
            if raw is None
            else raw
        )
        self.response_version_id = response_version_id
        self.calls: list[dict[str, object]] = []

    def get_object(self, **request: object) -> object:
        self.calls.append(dict(request))
        return {
            "Body": BytesIO(self.raw),
            "VersionId": self.response_version_id,
            "ContentLength": len(self.raw),
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(self.raw).digest()
            ).decode("ascii"),
            "ETag": '"terminal-etag"',
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "terminal-get",
                "RetryAttempts": 0,
            },
        }


def _authenticated_settlement_inputs(
    launch: dict[str, object],
    settlement: dict[str, object],
    *,
    raw: bytes | None = None,
    response_version_id: str = "terminal-version",
) -> tuple[
    tuple[dict[str, object], ...],
    dict[tuple[str, str], dict[str, object]],
    _TerminalS3,
]:
    terminal = _terminal_v2(launch, settlement)
    settlement["terminal_v2_key"] = "terminal-v2.json"
    settlement["terminal_v2_version_id"] = "terminal-version"
    settlement["terminal_v2_body_sha256"] = terminal[
        "canonical_body_sha256"
    ]
    settlement["terminal_v2_allocations_array_sha256"] = terminal[
        "allocations_array_sha256"
    ]
    _seal_settlement(settlement)
    liability = _liability(launch, settlement)
    s3 = _TerminalS3(
        terminal,
        raw=raw,
        response_version_id=response_version_id,
    )
    sources = _read_retained_terminal_v2_sources(
        s3=s3,
        bucket="keep-glm52-campaign",
        settlements=(settlement,),
    )
    return (liability,), sources, s3


def _reader(
    records: tuple[dict[str, object], ...],
    reserve_records: list[dict[str, object]],
    custody: dict[tuple[str, int, int], object],
    settlements: tuple[dict[str, object], ...] = (),
) -> tuple[Task11ReserveReader, list[dict[str, object]]]:
    known = _retained_known_launch_authority(
        records, activation_id="approved-20260728"
    )
    active = _retained_launch_authority(
        records, activation_id="approved-20260728"
    )
    liabilities: tuple[dict[str, object], ...] = ()
    terminal_sources: dict[
        tuple[str, str], dict[str, object]
    ] = {}
    if settlements:
        liabilities, terminal_sources, _ = (
            _authenticated_settlement_inputs(records[-1], settlements[-1])
        )
    settled = _retained_liability_settlement_authority(
        records,
        liabilities,
        settlements,
        terminal_sources,
        activation_id="approved-20260728",
    )
    requests: list[dict[str, object]] = []

    class Dynamo:
        def query(self, **request: object) -> object:
            requests.append(dict(request))
            return _ddb_page(reserve_records)

    return (
        Task11ReserveReader(
            dynamodb=Dynamo(),
            table_name="keep-glm52-ledger",
            token_custody=custody,
            activation_id="approved-20260728",
            active_launches=active,
            known_launches=known,
            settled_launches=settled,
        ),
        requests,
    )


@pytest.mark.parametrize(
    "terminal_state",
    ("ALLOCATION_CLOSED", "INSTANCE_TERMINAL", "REJECTED_NO_INSTANCE"),
)
def test_terminal_launch_state_without_settlement_keeps_held_reserve_counted(
    terminal_state: str,
) -> None:
    active = _record()
    terminal = _record()
    terminal["allocation_ordinal"] = 2
    terminal["state"] = terminal_state
    terminal["ec2_client_token"] = "f" * 64
    _seal_launch(terminal)
    custody: dict[tuple[str, int, int], object] = {}
    reader, requests = _reader(
        (active, terminal),
        [_reserve_item(active), _reserve_item(terminal)],
        custody,
    )

    _bind_spend_client_token(
        custody,
        allocation_identity=("approved-20260728", 1, 1),
        client_token="e" * 64,
        source="ec2",
    )
    page = reader.list_reserves(next_token=None)

    assert [row["allocation_ordinal"] for row in page.records] == [1, 2]
    assert requests[0]["ExpressionAttributeValues"][":sk_prefix"] == {
        "S": "GPU_LIABILITY_RESERVE#approved-20260728#"
    }
    _finalize_spend_token_custody(
        custody,
        _retained_launch_authority(
            (active, terminal), activation_id="approved-20260728"
        ),
    )


@pytest.mark.parametrize(
    "terminal_state",
    ("ALLOCATION_CLOSED", "INSTANCE_TERMINAL", "REJECTED_NO_INSTANCE"),
)
def test_positive_exact_settlement_excludes_terminal_held_reserve(
    terminal_state: str,
) -> None:
    terminal = _record()
    terminal["state"] = terminal_state
    _seal_launch(terminal)
    reader, _ = _reader(
        (terminal,),
        [_reserve_item(terminal)],
        {},
        (_settlement(terminal),),
    )

    assert reader.list_reserves(next_token=None).records == ()


def test_terminal_source_read_is_exact_version_pinned_and_zero_retry() -> None:
    terminal = _record()
    terminal["state"] = "ALLOCATION_CLOSED"
    _seal_launch(terminal)
    settlement = _settlement(terminal)
    liabilities, terminal_sources, s3 = (
        _authenticated_settlement_inputs(terminal, settlement)
    )

    authority = _retained_liability_settlement_authority(
        (terminal,),
        liabilities,
        (settlement,),
        terminal_sources,
        activation_id="approved-20260728",
    )

    assert set(authority) == {("approved-20260728", 1, 1)}
    assert s3.calls == [
        {
            "Bucket": "keep-glm52-campaign",
            "Key": "terminal-v2.json",
            "VersionId": "terminal-version",
            "ExpectedBucketOwner": "246813579024",
            "ChecksumMode": "ENABLED",
        }
    ]


@pytest.mark.parametrize("mutation", ("bytes", "version_id"))
def test_terminal_source_transport_rejects_byte_and_version_mutants(
    mutation: str,
) -> None:
    terminal = _record()
    terminal["state"] = "ALLOCATION_CLOSED"
    _seal_launch(terminal)
    settlement = _settlement(terminal)

    with pytest.raises(RuntimeError, match="terminal|object|version"):
        _authenticated_settlement_inputs(
            terminal,
            settlement,
            raw=b'{"record_type":"glm52_production_terminal_v2"}\n'
            if mutation == "bytes"
            else None,
            response_version_id=(
                "substituted-version"
                if mutation == "version_id"
                else "terminal-version"
            ),
        )


def test_terminal_source_rejects_worker_launch_liability_identity_substitution() -> None:
    terminal = _record()
    terminal["state"] = "ALLOCATION_CLOSED"
    _seal_launch(terminal)
    settlement = _settlement(terminal)
    _, terminal_sources, _ = _authenticated_settlement_inputs(
        terminal, settlement
    )
    settlement["worker_launch_liability_identity_sha256"] = "0" * 64
    _seal_settlement(settlement)

    with pytest.raises(RuntimeError, match="liability|settlement"):
        _retained_liability_settlement_authority(
            (terminal,),
            (_liability(terminal, settlement),),
            (settlement,),
            terminal_sources,
            activation_id="approved-20260728",
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_terminal_evidence",
        "missing_spend_close_evidence",
        "foreign_allocation",
        "duplicate_instance",
    ),
)
def test_settlement_terminal_semantics_reject_non_closed_final_members(
    mutation: str,
) -> None:
    terminal = _record()
    terminal["state"] = "ALLOCATION_CLOSED"
    _seal_launch(terminal)
    settlement = _settlement(terminal)
    member = settlement["merged_final_allocations"][0]
    if mutation == "missing_terminal_evidence":
        member.pop("instance_terminal_identity_sha256")
    elif mutation == "missing_spend_close_evidence":
        member.pop("spend_allocation_close_identity_sha256")
    elif mutation == "foreign_allocation":
        member["allocation_ordinal"] = 2
    else:
        settlement["merged_final_allocations"].append(dict(member))
    _seal_settlement(settlement)
    liabilities, terminal_sources, _ = (
        _authenticated_settlement_inputs(terminal, settlement)
    )

    with pytest.raises(RuntimeError, match="settlement|terminal"):
        _retained_liability_settlement_authority(
            (terminal,),
            liabilities,
            (settlement,),
            terminal_sources,
            activation_id="approved-20260728",
        )


def test_positive_rejection_settlement_requires_zero_terminal_v2_allocations() -> None:
    rejected = _record()
    rejected["state"] = "REJECTED_NO_INSTANCE"
    _seal_launch(rejected)
    settlement = _settlement(rejected)
    liabilities, terminal_sources, _ = (
        _authenticated_settlement_inputs(rejected, settlement)
    )
    settlement["terminal_v2_allocations_array_sha256"] = canonical_sha256(
        [{"instance_id": "i-0123456789abcdef0"}]
    )
    _seal_settlement(settlement)
    liabilities = (_liability(rejected, settlement),)

    with pytest.raises(RuntimeError, match="settlement|terminal"):
        _retained_liability_settlement_authority(
            (rejected,),
            liabilities,
            (settlement,),
            terminal_sources,
            activation_id="approved-20260728",
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "activation",
        "allocation",
        "launch_identity",
        "client_token",
        "reserve_identity",
        "release_identity",
        "terminal_body",
        "settlement_kind",
    ),
)
def test_settlement_requires_exact_launch_token_allocation_and_reserve_binding(
    mutation: str,
) -> None:
    terminal = _record()
    terminal["state"] = "ALLOCATION_CLOSED"
    _seal_launch(terminal)
    settlement = _settlement(terminal)
    _, terminal_sources, _ = _authenticated_settlement_inputs(
        terminal, settlement
    )
    if mutation == "activation":
        settlement["activation_id"] = "foreign-activation"
    elif mutation == "allocation":
        settlement["allocation_ordinal"] = 2
    elif mutation == "launch_identity":
        settlement["worker_launch_identity_sha256"] = "0" * 64
    elif mutation == "client_token":
        terminal["ec2_client_token"] = "0" * 64
        _seal_launch(terminal)
    elif mutation == "reserve_identity":
        settlement[
            "gpu_liability_reserve_ledger_identity_sha256"
        ] = "0" * 64
    elif mutation == "release_identity":
        settlement[
            "gpu_liability_reserve_release_identity_sha256"
        ] = "not-a-sha"
    elif mutation == "terminal_body":
        settlement["terminal_v2_body_sha256"] = "0" * 64
    else:
        settlement["settlement_kind"] = "NO_INSTANCE_POSITIVE_REJECTION"
    _seal_settlement(settlement)
    liabilities = (_liability(terminal, settlement),)

    with pytest.raises(RuntimeError, match="settlement|terminal"):
        _retained_liability_settlement_authority(
            (terminal,),
            liabilities,
            (settlement,),
            terminal_sources,
            activation_id="approved-20260728",
        )


def test_settled_launch_cannot_exclude_a_different_held_reserve_body() -> None:
    terminal = _record()
    terminal["state"] = "ALLOCATION_CLOSED"
    _seal_launch(terminal)
    reader, _ = _reader(
        (terminal,),
        [
            _reserve_item(
                terminal,
                request_identity_sha256="0" * 64,
            )
        ],
        {},
        (_settlement(terminal),),
    )

    with pytest.raises(RuntimeError, match="reserve identity"):
        reader.list_reserves(next_token=None)


@pytest.mark.parametrize("mutation", ("unknown", "token_mismatch", "unsettled"))
def test_current_activation_reserve_lifecycle_rejects_non_authoritative_rows(
    mutation: str,
) -> None:
    active = _record()
    reserve = active
    if mutation == "unknown":
        reserve = _record()
        reserve["allocation_ordinal"] = 2
        reserve["ec2_client_token"] = "f" * 64
    elif mutation == "token_mismatch":
        reserve = _record()
        reserve["ec2_client_token"] = "f" * 64
    elif mutation == "unsettled":
        active["state"] = "INSTANCE_OBSERVED"
        _seal_launch(active)
    custody: dict[tuple[str, int, int], object] = {}
    reader, _ = _reader((active,), [_reserve_item(reserve)], custody)

    with pytest.raises(RuntimeError):
        reader.list_reserves(next_token=None)


def test_active_launch_without_reserve_fails_the_two_sided_spend_gate() -> None:
    active = _record()
    custody: dict[tuple[str, int, int], object] = {}
    reader, _ = _reader((active,), [], custody)
    _bind_spend_client_token(
        custody,
        allocation_identity=("approved-20260728", 1, 1),
        client_token="e" * 64,
        source="ec2",
    )
    assert reader.list_reserves(next_token=None).records == ()
    with pytest.raises(RuntimeError, match="incomplete|one-sided"):
        _finalize_spend_token_custody(
            custody,
            _retained_launch_authority((active,), activation_id="approved-20260728"),
        )
