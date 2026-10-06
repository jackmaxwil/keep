from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json

import pytest

from glm52_enforcement.approvals import build_gpu_residual_liability_approval
from glm52_enforcement.spend_authority import (
    ACCOUNT_ID,
    APPROVED_GPU_COST_USD,
    APPROVED_GPU_RUNTIME_SECONDS,
    APPROVED_HOURLY_COST_USD,
    REGION,
    RUN_ID,
    AmbiguousReserveTransport,
    GpuLiabilityReserveRequest,
    LiabilitySettlementEvidence,
    ReserveRejected,
    ReserveListPage,
    SpendAuthorityError,
    SpendAuthorityRequest,
    SpendAuthorityServices,
    SpendListPage,
    SpendObject,
    build_gpu_spend_approval,
    canonical_decimal_json_bytes,
    inspect_spend_authority,
    reserve_gpu_liability,
    spend_authority_result_from_mapping,
    validate_gpu_spend_approval,
    validate_liability_settlement,
)


SOURCE_BYTES = (
    b"Jack Mazac approved the exact residual liability terms in Codex chat.\n"
)
NOW = datetime(2026, 7, 29, 1, 0, tzinfo=timezone.utc)
ZERO = "0" * 64


def _raw(value: object) -> bytes:
    return canonical_decimal_json_bytes(value) + b"\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _launch_tags() -> dict[str, str]:
    return {
        "Project": "KEEP", "Campaign": "GLM-5.2", "RunId": RUN_ID,
        "Market": "on-demand", "campaign-identity-sha256": "1" * 64,
        "activation-id": "approved-20260728", "activation-ordinal-text": "00000001",
        "generation-text": "00000001", "allocation-ordinal-text": "00000001",
        "action-key": "ACTION#00000001#SKY_POST#00000001",
        "sky-request-id": "sky-request-0001", "sky-job-name": RUN_ID,
        "sky-task-name": "glm52-production", "task-yaml-sha256": "2" * 64,
        "request-body-sha256": "3" * 64,
    }


def _self_hash(body: dict[str, object], field: str) -> dict[str, object]:
    return {
        **body,
        field: hashlib.sha256(canonical_decimal_json_bytes(body)).hexdigest(),
    }


def _spend_fixture() -> tuple[
    SpendAuthorityRequest,
    SpendAuthorityServices,
    dict[str, object],
]:
    approval = build_gpu_spend_approval(
        ingested_at="2026-07-15T00:00:00Z",
        slack_permalink=None,
    )
    approval_raw = _raw(approval)
    descriptor_body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_sky_campaign_descriptor_v2",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "campaign_identity_sha256": "1" * 64,
        "approval_sha256": _sha(approval_raw),
        "approved_gpu_runtime_seconds": APPROVED_GPU_RUNTIME_SECONDS,
        "approved_gpu_cost_usd": APPROVED_GPU_COST_USD,
        "max_hourly_cost_usd": APPROVED_HOURLY_COST_USD,
    }
    descriptor = _self_hash(descriptor_body, "descriptor_body_sha256")
    descriptor_raw = _raw(descriptor)
    genesis = hashlib.sha256(
        canonical_decimal_json_bytes(
            {
                "record_type": "glm52_gpu_spend_ledger_genesis_v1",
                "run_id": RUN_ID,
                "approval_sha256": _sha(approval_raw),
                "approved_gpu_runtime_seconds": APPROVED_GPU_RUNTIME_SECONDS,
                "approved_gpu_cost_usd": APPROVED_GPU_COST_USD,
                "hourly_cost_usd": APPROVED_HOURLY_COST_USD,
            }
        )
    ).hexdigest()
    start_body: dict[str, object] = {
        "record_type": "glm52_gpu_spend_event_v1",
        "run_id": RUN_ID,
        "approval_sha256": _sha(approval_raw),
        "event": "allocation_started",
        "job_id": "qualification-1",
        "instance_id": "i-00000000000000001",
        "timestamp": "2026-07-28T23:00:00Z",
        "prior_record_sha256": genesis,
    }
    start = _self_hash(start_body, "record_sha256")
    end_body: dict[str, object] = {
        "record_type": "glm52_gpu_spend_event_v1",
        "run_id": RUN_ID,
        "approval_sha256": _sha(approval_raw),
        "event": "allocation_ended",
        "instance_id": "i-00000000000000001",
        "timestamp": "2026-07-29T00:00:00Z",
        "prior_record_sha256": start["record_sha256"],
    }
    end = _self_hash(end_body, "record_sha256")
    names = [
        f"000000-allocation_started-{start['record_sha256']}.json",
        f"000001-allocation_ended-{end['record_sha256']}.json",
    ]
    records = [_raw(start), _raw(end)]
    ledger_raw = b"".join(records)
    latest_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_ledger_latest_v1",
        "run_id": RUN_ID,
        "gpu_spend_authority_sha256": genesis,
        "record_count": 2,
        "record_keys": names,
        "latest_record_sha256": end["record_sha256"],
        "ledger_sha256": _sha(ledger_raw),
    }
    latest = _self_hash(latest_body, "latest_body_sha256")
    latest_raw = _raw(latest)
    consumed_seconds = 3600
    consumed_cost = Decimal("55.04")
    snapshot_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_snapshot_v1",
        "run_id": RUN_ID,
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "descriptor_sha256": _sha(descriptor_raw),
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "approval_sha256": _sha(approval_raw),
        "approval_body_sha256": approval["approval_body_sha256"],
        "gpu_spend_ledger_latest_sha256": _sha(latest_raw),
        "gpu_spend_ledger_latest_body_sha256": latest["latest_body_sha256"],
        "gpu_spend_ledger_genesis_sha256": genesis,
        "gpu_spend_ledger_record_count": 2,
        "gpu_spend_ledger_tip_record_sha256": end["record_sha256"],
        "gpu_spend_ledger_file_sha256": _sha(ledger_raw),
        "ec2_allocation_history_sha256": "2" * 64,
        "ec2_allocation_instance_ids": ["i-00000000000000001"],
        "observed_at": "2026-07-29T01:00:00Z",
        "approved_gpu_runtime_seconds": APPROVED_GPU_RUNTIME_SECONDS,
        "approved_gpu_cost_usd": APPROVED_GPU_COST_USD,
        "hourly_cost_usd": APPROVED_HOURLY_COST_USD,
        "consumed_gpu_seconds": consumed_seconds,
        "remaining_gpu_seconds": APPROVED_GPU_RUNTIME_SECONDS - consumed_seconds,
        "consumed_gpu_cost_usd": consumed_cost,
        "remaining_gpu_cost_usd": APPROVED_GPU_COST_USD - consumed_cost,
        "qualification_allowance_seconds": 14400,
        "qualification_allowance_cost_usd": Decimal("220.16"),
        "open_allocation_count": 0,
    }
    snapshot = _self_hash(snapshot_body, "snapshot_body_sha256")
    snapshot_raw = _raw(snapshot)
    prefix = f"campaigns/{RUN_ID}/spend-ledger/records/"
    keys = {
        "descriptor": f"campaigns/{RUN_ID}/descriptor.json",
        "approval": f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json",
        "latest": f"campaigns/{RUN_ID}/spend-ledger/latest.json",
        "snapshot": f"campaigns/{RUN_ID}/spend-snapshots/CURRENT.json",
        "records": tuple(prefix + name for name in names),
    }
    objects = {
        keys["descriptor"]: descriptor_raw,
        keys["approval"]: approval_raw,
        keys["latest"]: latest_raw,
        keys["snapshot"]: snapshot_raw,
        keys["records"][0]: records[0],
        keys["records"][1]: records[1],
    }

    class Store:
        def __init__(self) -> None:
            self.list_calls: list[object] = []
            self.get_calls: list[str] = []
            self.pages = [
                SpendListPage(
                    keys=(keys["records"][0],),
                    next_token="page-2",
                    request_id="list-1",
                    observed_at="2026-07-29T01:00:00Z",
                    version_ids=("version-5",),
                ),
                SpendListPage(
                    keys=(keys["records"][1],),
                    next_token=None,
                    request_id="list-2",
                    observed_at="2026-07-29T01:00:00Z",
                    version_ids=("version-6",),
                ),
            ]

        def list_namespace(
            self, *, prefix: str, continuation_token: str | None
        ) -> SpendListPage:
            self.list_calls.append(continuation_token)
            return self.pages[0 if continuation_token is None else 1]

        def get_object(self, *, key: str) -> SpendObject:
            self.get_calls.append(key)
            raw = objects[key]
            version_by_key = {
                keys["descriptor"]: "version-1",
                keys["approval"]: "version-2",
                keys["latest"]: "version-3",
                keys["snapshot"]: "version-4",
            }
            version_by_key.update(
                {
                    record_key: f"version-{index + 5}"
                    for index, record_key in enumerate(keys["records"])
                }
            )
            return SpendObject(
                key=key,
                raw=raw,
                version_id=version_by_key[key],
                etag=_sha(raw),
                checksum_sha256=_sha(raw),
                request_id=f"get-{len(self.get_calls)}",
                observed_at="2026-07-29T01:00:00Z",
            )

    class Ec2:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def describe_allocation_history(
            self, *, run_id: str, next_token: str | None
        ) -> dict[str, object]:
            self.calls.append(next_token)
            return {
                "instances": [
                    {
                        "InstanceId": "i-00000000000000001",
                        "InstanceType": "p5.48xlarge",
                        "InstanceLifecycle": None,
                        "State": "stopped",
                        "LaunchTime": "2026-07-28T23:00:00Z",
                        "TerminatedAt": "2026-07-29T00:00:00Z",
                        "AvailabilityZone": "us-west-2a",
                        "Tags": _launch_tags(),
                    }
                ],
                "next_token": None,
                "request_id": "ec2-1",
                "observed_at": "2026-07-29T01:00:00Z",
            }

    store = Store()
    ec2 = Ec2()

    class EmptyReserves:
        def list_reserves(
            self, *, next_token: str | None
        ) -> ReserveListPage:
            assert next_token is None
            return ReserveListPage(
                records=(),
                next_token=None,
                request_id="reserve-empty-1",
                observed_at="2026-07-29T01:00:00Z",
            )

    request = SpendAuthorityRequest(
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        descriptor_key=keys["descriptor"],
        descriptor_file_sha256=_sha(descriptor_raw),
        descriptor_version_id="version-1",
        approval_key=keys["approval"],
        approval_file_sha256=_sha(approval_raw),
        approval_version_id="version-2",
        latest_key=keys["latest"],
        latest_file_sha256=_sha(latest_raw),
        latest_version_id="version-3",
        snapshot_key=keys["snapshot"],
        snapshot_file_sha256=_sha(snapshot_raw),
        snapshot_version_id="version-4",
        ledger_records_prefix=prefix,
        expected_snapshot_body_sha256=snapshot["snapshot_body_sha256"],
        observed_at="2026-07-29T01:00:00Z",
    )
    return request, SpendAuthorityServices(
        object_store=store,
        ec2=ec2,
        reserve_reader=EmptyReserves(),
    ), {
        "objects": objects,
        "store": store,
        "ec2": ec2,
        "keys": keys,
        "latest": latest,
        "snapshot": snapshot,
    }


def _extend_allocations(
    request: SpendAuthorityRequest,
    services: SpendAuthorityServices,
    state: dict[str, object],
    *,
    events: list[dict[str, object]],
    instances: list[dict[str, object]],
) -> SpendAuthorityRequest:
    keys = state["keys"]
    objects = state["objects"]
    existing_keys = list(keys["records"])
    existing_raw = [objects[key] for key in existing_keys]
    prior = state["latest"]["latest_record_sha256"]
    record_names = [key.rsplit("/", 1)[1] for key in existing_keys]
    for event in events:
        body = {
            "record_type": "glm52_gpu_spend_event_v1",
            "run_id": RUN_ID,
            "approval_sha256": json.loads(
                existing_raw[0], parse_float=Decimal
            )["approval_sha256"],
            **event,
            "prior_record_sha256": prior,
        }
        record = _self_hash(body, "record_sha256")
        name = (
            f"{len(record_names):06d}-{event['event']}-"
            f"{record['record_sha256']}.json"
        )
        key = request.ledger_records_prefix + name
        raw = _raw(record)
        record_names.append(name)
        existing_keys.append(key)
        existing_raw.append(raw)
        objects[key] = raw
        prior = record["record_sha256"]
    keys["records"] = tuple(existing_keys)
    ledger_raw = b"".join(existing_raw)
    latest_body = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_ledger_latest_v1",
        "run_id": RUN_ID,
        "gpu_spend_authority_sha256": state["latest"][
            "gpu_spend_authority_sha256"
        ],
        "record_count": len(record_names),
        "record_keys": record_names,
        "latest_record_sha256": prior,
        "ledger_sha256": _sha(ledger_raw),
    }
    latest = _self_hash(latest_body, "latest_body_sha256")
    latest_raw = _raw(latest)
    objects[keys["latest"]] = latest_raw
    state["latest"] = latest
    store = state["store"]
    store.pages = [
        SpendListPage(
            keys=tuple(existing_keys),
            next_token=None,
            request_id="list-all",
            observed_at="2026-07-29T01:00:00Z",
            version_ids=tuple(
                f"version-{index + 5}" for index in range(len(existing_keys))
            ),
        )
    ]
    parsed_records = [
        json.loads(raw, parse_float=Decimal) for raw in existing_raw
    ]
    active: dict[str, object] | None = None
    closed_seconds = 0
    open_seconds = 0
    allocation_ids: list[str] = []
    for record in parsed_records:
        if record["event"] == "allocation_started":
            active = record
            allocation_ids.append(record["instance_id"])
        else:
            assert active is not None
            start = datetime.fromisoformat(
                str(active["timestamp"]).replace("Z", "+00:00")
            )
            end = datetime.fromisoformat(
                str(record["timestamp"]).replace("Z", "+00:00")
            )
            closed_seconds += int((end - start).total_seconds())
            active = None
    if active is not None:
        start = datetime.fromisoformat(
            str(active["timestamp"]).replace("Z", "+00:00")
        )
        open_seconds = int((NOW - start).total_seconds())
    charged_seconds = closed_seconds + open_seconds
    charged_cost = (
        Decimal(charged_seconds)
        * APPROVED_HOURLY_COST_USD
        / Decimal(3600)
    ).quantize(Decimal("0.01"))
    snapshot = json.loads(
        objects[keys["snapshot"]], parse_float=Decimal
    )
    snapshot.update(
        {
            "gpu_spend_ledger_latest_sha256": _sha(latest_raw),
            "gpu_spend_ledger_latest_body_sha256": latest[
                "latest_body_sha256"
            ],
            "gpu_spend_ledger_record_count": len(record_names),
            "gpu_spend_ledger_tip_record_sha256": prior,
            "gpu_spend_ledger_file_sha256": _sha(ledger_raw),
            "ec2_allocation_instance_ids": allocation_ids,
            "consumed_gpu_seconds": charged_seconds,
            "remaining_gpu_seconds": (
                APPROVED_GPU_RUNTIME_SECONDS - charged_seconds
            ),
            "consumed_gpu_cost_usd": charged_cost,
            "remaining_gpu_cost_usd": APPROVED_GPU_COST_USD - charged_cost,
            "open_allocation_count": 1 if active is not None else 0,
        }
    )
    snapshot_body = {
        key: value
        for key, value in snapshot.items()
        if key != "snapshot_body_sha256"
    }
    snapshot["snapshot_body_sha256"] = hashlib.sha256(
        canonical_decimal_json_bytes(snapshot_body)
    ).hexdigest()
    snapshot_raw = _raw(snapshot)
    objects[keys["snapshot"]] = snapshot_raw
    original_ec2 = services.ec2.describe_allocation_history

    def extended_ec2(**kwargs: object) -> dict[str, object]:
        page = original_ec2(**kwargs)
        page["instances"].extend(deepcopy(instances))
        return page

    services.ec2.describe_allocation_history = extended_ec2
    return replace(
        request,
        latest_file_sha256=_sha(latest_raw),
        snapshot_file_sha256=_sha(snapshot_raw),
        expected_snapshot_body_sha256=snapshot["snapshot_body_sha256"],
    )


def test_gpu_approval_reuses_exact_accepted_schema() -> None:
    approval = build_gpu_spend_approval(
        ingested_at="2026-07-15T00:00:00Z",
        slack_permalink=None,
    )
    assert validate_gpu_spend_approval(approval) == approval
    assert approval["approved_gpu_hours"] == 24
    assert Decimal(str(approval["approved_gpu_cost_usd"])) == Decimal("1320.96")
    mutant = deepcopy(approval)
    mutant["approved_gpu_cost_usd"] = Decimal("1320.95")
    mutant["approval_body_sha256"] = hashlib.sha256(
        canonical_decimal_json_bytes(
            {k: v for k, v in mutant.items() if k != "approval_body_sha256"}
        )
    ).hexdigest()
    with pytest.raises(SpendAuthorityError):
        validate_gpu_spend_approval(mutant)


def test_live_spend_walk_paginates_exact_namespace_and_reconstructs_decimal_balance() -> None:
    request, services, state = _spend_fixture()
    result = inspect_spend_authority(request, services)
    assert state["store"].list_calls == [None, "page-2"]
    assert len(state["store"].get_calls) == 6
    assert result.used_gpu_seconds == 3600
    assert result.open_gpu_seconds == 0
    assert result.reserved_gpu_seconds == 0
    assert result.remaining_gpu_seconds == 82800
    assert result.used_gpu_cost_usd == Decimal("55.04")
    assert result.remaining_gpu_cost_usd == Decimal("1265.92")
    assert result.refundable_gpu_seconds == 0
    assert result.ledger_tip_identity_sha256 == state["latest"][
        "latest_record_sha256"
    ]
    assert len(result.canonical_identity_sha256) == 64


def test_open_allocation_is_charged_nonrefundable_and_reconciles_running_ec2() -> None:
    request, services, state = _spend_fixture()
    request = _extend_allocations(
        request,
        services,
        state,
        events=[
            {
                "event": "allocation_started",
                "job_id": "possibly-sent-production",
                "instance_id": "i-00000000000000002",
                "timestamp": "2026-07-29T00:30:00Z",
            }
        ],
        instances=[
            {
                "InstanceId": "i-00000000000000002",
                "InstanceType": "p5.48xlarge",
                "InstanceLifecycle": None,
                "State": "running",
                "LaunchTime": "2026-07-29T00:30:00Z",
                "TerminatedAt": None,
                "AvailabilityZone": "us-west-2b",
                "Tags": _launch_tags(),
            }
        ],
    )
    result = inspect_spend_authority(request, services)
    assert result.used_gpu_seconds == 3600
    assert result.open_gpu_seconds == 1800
    assert result.refundable_gpu_seconds == 0
    assert result.used_gpu_cost_usd == Decimal("55.04")
    assert result.open_gpu_cost_usd == Decimal("27.52")
    assert result.remaining_gpu_seconds == 81000
    assert result.remaining_gpu_cost_usd == Decimal("1238.40")
    assert result.allocation_intervals[-1].state == "OPEN"


def test_multiple_short_allocations_round_once_from_total_seconds() -> None:
    request, services, state = _spend_fixture()
    request = _extend_allocations(
        request,
        services,
        state,
        events=[
            {
                "event": "allocation_started",
                "job_id": "short-1",
                "instance_id": "i-00000000000000002",
                "timestamp": "2026-07-29T00:10:00Z",
            },
            {
                "event": "allocation_ended",
                "instance_id": "i-00000000000000002",
                "timestamp": "2026-07-29T00:10:01Z",
            },
            {
                "event": "allocation_started",
                "job_id": "short-2",
                "instance_id": "i-00000000000000003",
                "timestamp": "2026-07-29T00:20:00Z",
            },
            {
                "event": "allocation_ended",
                "instance_id": "i-00000000000000003",
                "timestamp": "2026-07-29T00:20:01Z",
            },
        ],
        instances=[
            {
                "InstanceId": instance_id,
                "InstanceType": "p5.48xlarge",
                "InstanceLifecycle": None,
                "State": "stopped",
                "LaunchTime": started,
                "TerminatedAt": ended,
                "AvailabilityZone": "us-west-2a",
                "Tags": _launch_tags(),
            }
            for instance_id, started, ended in (
                (
                    "i-00000000000000002",
                    "2026-07-29T00:10:00Z",
                    "2026-07-29T00:10:01Z",
                ),
                (
                    "i-00000000000000003",
                    "2026-07-29T00:20:00Z",
                    "2026-07-29T00:20:01Z",
                ),
            )
        ],
    )
    result = inspect_spend_authority(request, services)
    assert result.used_gpu_seconds == 3602
    assert result.used_gpu_cost_usd == Decimal("55.07")
    assert sum(
        interval.charged_cost_usd for interval in result.allocation_intervals
    ) == Decimal("55.08")
    assert result.remaining_gpu_cost_usd == Decimal("1265.89")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-record",
        "duplicate-record",
        "repeated-token",
        "forked-chain",
        "self-hash",
        "future-record",
        "snapshot-drift",
        "ec2-drift",
        "foreign-run",
    ],
)
def test_spend_walk_rejects_chain_snapshot_namespace_and_allocation_mutants(
    mutation: str,
) -> None:
    request, services, state = _spend_fixture()
    store = state["store"]
    if mutation == "missing-record":
        store.pages[1] = replace(store.pages[1], keys=())
    elif mutation == "duplicate-record":
        store.pages[1] = replace(
            store.pages[1], keys=(state["keys"]["records"][0],)
        )
    elif mutation == "repeated-token":
        store.pages[1] = replace(store.pages[1], next_token="page-2")
    elif mutation in {"forked-chain", "self-hash", "future-record", "foreign-run"}:
        key = state["keys"]["records"][1]
        value = json.loads(state["objects"][key], parse_float=Decimal)
        if mutation == "forked-chain":
            value["prior_record_sha256"] = ZERO
        elif mutation == "self-hash":
            value["record_sha256"] = ZERO
        elif mutation == "future-record":
            value["timestamp"] = "2026-07-30T00:00:00Z"
        else:
            value["run_id"] = "foreign"
        state["objects"][key] = _raw(value)
    elif mutation == "snapshot-drift":
        key = state["keys"]["snapshot"]
        value = json.loads(state["objects"][key], parse_float=Decimal)
        value["remaining_gpu_seconds"] -= 1
        state["objects"][key] = _raw(value)
    else:
        original = services.ec2.describe_allocation_history

        def drift(**kwargs: object) -> dict[str, object]:
            value = original(**kwargs)
            value["instances"][0]["InstanceType"] = "p4d.24xlarge"
            return value

        services.ec2.describe_allocation_history = drift
    with pytest.raises(SpendAuthorityError):
        inspect_spend_authority(request, services)


def _reserve_request(current: object) -> GpuLiabilityReserveRequest:
    approval = build_gpu_residual_liability_approval(SOURCE_BYTES)
    return GpuLiabilityReserveRequest(
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        activation_id="act-20260729-0001",
        generation=1,
        allocation_ordinal=1,
        ec2_client_token="a" * 64,
        request_identity_sha256="3" * 64,
        ledger_predecessor_identity_sha256=(
            current.ledger_tip_identity_sha256
        ),
        gpu_reserve_seconds=900,
        gpu_reserve_cost_usd=Decimal("13.76"),
        root_volume_gib=300,
        root_volume_tail_usd_max=Decimal("0.01"),
        residual_liability_approval=approval,
        residual_liability_source_bytes=SOURCE_BYTES,
        residual_liability_approval_identity_sha256=approval[
            "canonical_body_sha256"
        ],
        epoch=1,
        revision=4,
        nonce_owner_identity_sha256="4" * 64,
        observed_at="2026-07-29T01:00:00Z",
    )


class _ReserveWriter:
    def __init__(self, outcome: str = "success") -> None:
        self.outcome = outcome
        self.existing: dict[str, object] | None = None
        self.read_calls = 0
        self.intent_calls = 0
        self.submit_calls = 0

    def exact_read(self, *, reserve_key: str) -> dict[str, object] | None:
        self.read_calls += 1
        return deepcopy(self.existing)

    def persist_intent(self, *, intent: dict[str, object]) -> None:
        self.intent_calls += 1

    def conditional_append(
        self,
        *,
        record: dict[str, object],
        expected_predecessor_identity_sha256: str,
    ) -> dict[str, object]:
        self.submit_calls += 1
        if self.outcome == "ambiguous":
            self.existing = deepcopy(record)
            raise AmbiguousReserveTransport("timeout")
        if self.outcome == "rejected":
            raise ReserveRejected("conditional rejection")
        self.existing = deepcopy(record)
        return deepcopy(record)


def test_reserve_is_one_submit_exact_900_seconds_and_separate_root_tail() -> None:
    request, services, _state = _spend_fixture()
    current = inspect_spend_authority(request, services)
    writer = _ReserveWriter()
    result = reserve_gpu_liability(
        _reserve_request(current), request, services, writer
    )
    assert writer.read_calls == 1
    assert writer.intent_calls == 1
    assert writer.submit_calls == 1
    assert result.disposition == "CREATED"
    assert result.gpu_reserve_seconds == 900
    assert result.gpu_reserve_cost_usd == Decimal("13.76")
    assert result.root_volume_gib == 300
    assert result.root_volume_tail_usd_max == Decimal("0.01")
    assert result.remaining_gpu_seconds == 81900
    assert result.remaining_gpu_cost_usd == Decimal("1252.16")


def test_existing_held_reserve_is_paginated_charged_and_not_double_counted() -> None:
    request, services, _state = _spend_fixture()
    current = inspect_spend_authority(request, services)
    created = reserve_gpu_liability(
        _reserve_request(current), request, services, _ReserveWriter()
    )

    class Reserves:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def list_reserves(self, *, next_token: str | None) -> ReserveListPage:
            self.calls.append(next_token)
            if next_token is None:
                return ReserveListPage(
                    records=(),
                    next_token="reserve-page-2",
                    request_id="reserve-1",
                    observed_at="2026-07-29T01:00:00Z",
                )
            return ReserveListPage(
                records=(created.record,),
                next_token=None,
                request_id="reserve-2",
                observed_at="2026-07-29T01:00:00Z",
            )

    reserves = Reserves()
    services.object_store.list_calls.clear()
    services.object_store.get_calls.clear()
    services.ec2.calls.clear()
    result = inspect_spend_authority(
        request, replace(services, reserve_reader=reserves)
    )
    assert reserves.calls == [None, "reserve-page-2"]
    assert result.used_gpu_seconds == 3600
    assert result.open_gpu_seconds == 0
    assert result.reserved_gpu_seconds == 900
    assert result.remaining_gpu_seconds == 81900
    assert result.used_gpu_cost_usd == Decimal("55.04")
    assert result.reserved_gpu_cost_usd == Decimal("13.76")
    assert result.remaining_gpu_cost_usd == Decimal("1252.16")
    assert result.refundable_gpu_seconds == 0


def test_held_reserve_predecessor_may_be_an_authenticated_historical_chain_node() -> None:
    request, services, state = _spend_fixture()
    current = inspect_spend_authority(request, services)
    created = reserve_gpu_liability(
        _reserve_request(current), request, services, _ReserveWriter()
    )
    historical = deepcopy(created.record)
    first_record = json.loads(
        state["objects"][state["keys"]["records"][0]],
        parse_float=Decimal,
    )
    historical["ledger_predecessor_identity_sha256"] = first_record[
        "record_sha256"
    ]
    historical_body = dict(historical)
    historical_body.pop("canonical_body_sha256")
    historical["canonical_body_sha256"] = hashlib.sha256(
        canonical_decimal_json_bytes(historical_body)
    ).hexdigest()

    class HistoricalReserve:
        def list_reserves(
            self, *, next_token: str | None
        ) -> ReserveListPage:
            return ReserveListPage(
                records=(historical,),
                next_token=None,
                request_id="historical-reserve-1",
                observed_at="2026-07-29T01:00:00Z",
            )

    result = inspect_spend_authority(
        request,
        replace(services, reserve_reader=HistoricalReserve()),
    )
    assert result.reserved_gpu_seconds == 900
    assert result.remaining_gpu_seconds == 81900


def test_closed_allocation_uses_authenticated_ledger_when_ec2_history_expires() -> None:
    request, services, _state = _spend_fixture()

    def expired_history(
        *, run_id: str, next_token: str | None
    ) -> dict[str, object]:
        return {
            "instances": [],
            "next_token": None,
            "request_id": "ec2-expired-history",
            "observed_at": "2026-07-29T01:00:00Z",
        }

    services.ec2.describe_allocation_history = expired_history
    result = inspect_spend_authority(request, services)
    assert result.used_gpu_seconds == 3600
    assert result.allocation_intervals[0].state == "CLOSED"
    assert result.allocation_intervals[0].ended_at == "2026-07-29T00:00:00Z"


def test_reserve_reconstructs_fresh_spend_and_rejects_forged_result() -> None:
    request, services, _state = _spend_fixture()
    current = inspect_spend_authority(request, services)
    forged_mapping = asdict(current)
    forged_mapping.pop("canonical_identity_sha256")
    forged_mapping["approval_identity_sha256"] = "a" * 64
    forged_mapping["snapshot_identity_sha256"] = "b" * 64
    forged_mapping["ledger_tip_identity_sha256"] = "c" * 64
    forged = spend_authority_result_from_mapping(forged_mapping)
    writer = _ReserveWriter()
    with pytest.raises(SpendAuthorityError):
        reserve_gpu_liability(
            _reserve_request(forged),
            forged,
            services,
            writer,
        )
    assert writer.submit_calls == 0


def test_ambiguous_reserve_uses_one_submit_and_exact_readback() -> None:
    request, services, _state = _spend_fixture()
    current = inspect_spend_authority(request, services)
    writer = _ReserveWriter("ambiguous")
    result = reserve_gpu_liability(
        _reserve_request(current), request, services, writer
    )
    assert result.disposition == "AMBIGUOUS_EXACT_READBACK"
    assert writer.submit_calls == 1
    assert writer.read_calls == 2


def test_exact_duplicate_held_reserve_is_read_only_and_not_double_debited() -> None:
    request, services, _state = _spend_fixture()
    current = inspect_spend_authority(request, services)
    reserve_request = _reserve_request(current)
    first_writer = _ReserveWriter()
    first = reserve_gpu_liability(
        reserve_request, request, services, first_writer
    )
    writer = _ReserveWriter()
    writer.existing = deepcopy(first.record)

    class HeldReserves:
        def list_reserves(
            self, *, next_token: str | None
        ) -> ReserveListPage:
            return ReserveListPage(
                records=(deepcopy(first.record),),
                next_token=None,
                request_id="held-exact-duplicate-1",
                observed_at="2026-07-29T01:00:00Z",
            )

    adopted = reserve_gpu_liability(
        reserve_request,
        request,
        replace(services, reserve_reader=HeldReserves()),
        writer,
    )
    assert adopted.disposition == "EXACT_DUPLICATE"
    assert writer.read_calls == 1
    assert writer.intent_calls == 0
    assert writer.submit_calls == 0
    assert adopted.remaining_gpu_seconds == 81900
    assert adopted.remaining_gpu_cost_usd == Decimal("1252.16")


@pytest.mark.parametrize(
    "mutation",
    ["seconds", "cost", "tail", "predecessor", "approval", "stale"],
)
def test_reserve_fails_closed_on_amount_approval_head_and_stale_mutants(
    mutation: str,
) -> None:
    request, services, _state = _spend_fixture()
    current = inspect_spend_authority(request, services)
    reserve_request = _reserve_request(current)
    if mutation == "seconds":
        reserve_request = replace(reserve_request, gpu_reserve_seconds=901)
    elif mutation == "cost":
        reserve_request = replace(
            reserve_request, gpu_reserve_cost_usd=Decimal("13.77")
        )
    elif mutation == "tail":
        reserve_request = replace(
            reserve_request, root_volume_tail_usd_max=Decimal("0.02")
        )
    elif mutation == "predecessor":
        reserve_request = replace(
            reserve_request, ledger_predecessor_identity_sha256=ZERO
        )
    elif mutation == "approval":
        approval = deepcopy(reserve_request.residual_liability_approval)
        approval["canonical_body_sha256"] = ZERO
        reserve_request = replace(
            reserve_request, residual_liability_approval=approval
        )
    else:
        reserve_request = replace(
            reserve_request, observed_at="2026-07-28T00:00:00Z"
        )
    writer = _ReserveWriter()
    with pytest.raises((SpendAuthorityError, ValueError)):
        reserve_gpu_liability(
            reserve_request,
            request,
            services,
            writer,
        )
    assert writer.submit_calls == 0


def test_definite_reserve_rejection_never_retries_or_adopts() -> None:
    request, services, _state = _spend_fixture()
    current = inspect_spend_authority(request, services)
    writer = _ReserveWriter("rejected")
    with pytest.raises(ReserveRejected):
        reserve_gpu_liability(
            _reserve_request(current), request, services, writer
        )
    assert writer.submit_calls == 1
    assert writer.read_calls == 1


def test_no_settlement_means_no_refund_authority() -> None:
    request, services, _state = _spend_fixture()
    current = inspect_spend_authority(request, services)
    reserve = reserve_gpu_liability(
        _reserve_request(current), request, services, _ReserveWriter()
    )
    with pytest.raises(SpendAuthorityError):
        validate_liability_settlement(reserve, None)
    settlement = LiabilitySettlementEvidence(
        schema_version=1,
        record_type="glm52_gpu_liability_settlement_v1",
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        reserve_identity_sha256=reserve.reserve_identity_sha256,
        liability_identity_sha256="5" * 64,
        allocation_close_identity_sha256="6" * 64,
        instance_terminal_identity_sha256="7" * 64,
        charged_gpu_seconds=900,
        charged_gpu_cost_usd=Decimal("13.76"),
        refundable_gpu_seconds=0,
        refundable_gpu_cost_usd=Decimal("0.00"),
        settled_at="2026-07-29T01:10:00Z",
        canonical_body_sha256=ZERO,
    )
    with pytest.raises(SpendAuthorityError):
        validate_liability_settlement(reserve, settlement)
