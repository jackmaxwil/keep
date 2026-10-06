"""Post-seed cumulative GPU spend snapshot and strict restore tests."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from mlx_vq.quality.glm52_sky_campaign import (
    GpuSpendLedger,
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "aws/glm52-gpu/scripts"
UTC = timezone.utc
RUN_ID = "glm52-sky-20260723"
OBSERVED_AT = datetime(2026, 7, 24, 13, 0, tzinfo=UTC)
FIRST_INSTANCE = "i-00000000000000001"
SECOND_INSTANCE = "i-00000000000000002"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_hashed(body: dict[str, object], field: str) -> dict[str, object]:
    return {**body, field: _sha(_canonical(body))}


def _module() -> Any:
    return importlib.import_module("mlx_vq.quality.glm52_gpu_spend_snapshot")


def _authorities() -> tuple[bytes, bytes]:
    approval = build_gpu_spend_approval(
        ingested_at=datetime(2026, 7, 23, 18, 5, tzinfo=UTC),
        slack_permalink=None,
    )
    approval_raw = _canonical(approval) + b"\n"
    approval_sha256 = _sha(approval_raw)
    descriptor = build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=datetime(2026, 7, 24, 6, 5, tzinfo=UTC),
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
        worker_identity=("arn:aws:iam::246813579024:role/keep-glm52-gpu-worker"),
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket="keep-glm52-us-west-2-246813579024",
        jobs_bucket="keep-glm52-us-west-2-246813579024",
        repo_tar_key=("campaigns/glm52-sky-20260723/repository/repo.tar.gz"),
        repo_tar_sha256="1" * 64,
        campaign_descriptor_key=(
            "campaigns/glm52-sky-20260723/submissions/post-seed/"
            "campaign-descriptor-v2.json"
        ),
        approval_key=(
            "campaigns/glm52-sky-20260723/authorities/GPU_SPEND_APPROVAL.json"
        ),
        approval_sha256=approval_sha256,
        artifacts={
            "source_snapshot_prefix": "source-snapshot/",
            "source_snapshot_sha256": "2" * 64,
            "non_vq_prefix": "non-vq-package/",
            "non_vq_package_sha256": "3" * 64,
            "teich_pack_key": "teich-pack/pack.json",
            "teich_pack_sha256": "4" * 64,
            "frozen_prompt_pack_key": "quality/frozen.json",
            "frozen_prompt_pack_sha256": "5" * 64,
            "training_baseline_prefix": "training-baseline/",
            "training_baseline_sha256": "6" * 64,
            "training_config_key": (
                "campaigns/glm52-sky-20260723/authorities/training.json"
            ),
            "training_config_sha256": "7" * 64,
            "artifact_inventory_key": (
                "campaigns/glm52-sky-20260723/inventories/"
                f"artifact-inventory-{'8' * 64}.json"
            ),
            "artifact_inventory_sha256": "8" * 64,
            "qualification_cache_prefix": (
                f"qualification-cache/seeds/glm52-sky-20260723/{'9' * 64}/"
            ),
            "qualification_cache_manifest_sha256": "9" * 64,
        },
    )
    return _canonical(descriptor) + b"\n", approval_raw


def _record_publication(
    tmp_path: Path,
    *,
    allocations: tuple[
        tuple[str, datetime, datetime],
        ...,
    ] = (
        (
            FIRST_INSTANCE,
            datetime(2026, 7, 24, 8, 0, tzinfo=UTC),
            datetime(2026, 7, 24, 10, 0, tzinfo=UTC),
        ),
        (
            SECOND_INSTANCE,
            datetime(2026, 7, 24, 11, 0, tzinfo=UTC),
            datetime(2026, 7, 24, 12, 0, tzinfo=UTC),
        ),
    ),
    leave_open: bool = False,
    duplicate_instance: bool = False,
) -> tuple[bytes, dict[str, bytes], str]:
    _descriptor_raw, approval_raw = _authorities()
    path = tmp_path / "fixture-ledger.jsonl"
    ledger = GpuSpendLedger(
        path,
        run_id=RUN_ID,
        approval_sha256=_sha(approval_raw),
        approved_gpu_runtime_seconds=86_400,
        approved_gpu_cost_usd=1_320.96,
        hourly_cost_usd=55.04,
    )
    for index, (instance_id, launched_at, ended_at) in enumerate(allocations):
        if duplicate_instance and index:
            instance_id = allocations[0][0]
        ledger.start_allocation(
            job_id="sky-job-3",
            instance_id=instance_id,
            launched_at=launched_at,
        )
        if not (leave_open and index == len(allocations) - 1):
            ledger.end_allocation(
                instance_id=instance_id,
                ended_at=ended_at,
            )
    records: dict[str, bytes] = {}
    record_keys: list[str] = []
    for index, record in enumerate(ledger.records):
        name = f"{index:06d}-{record['event']}-{record['record_sha256']}.json"
        record_keys.append(name)
        records[name] = _canonical(record) + b"\n"
    ledger_raw = b"".join(records[name] for name in record_keys)
    latest_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_gpu_spend_ledger_latest_v1",
        "run_id": RUN_ID,
        "gpu_spend_authority_sha256": ledger.genesis_sha256,
        "record_count": len(record_keys),
        "record_keys": record_keys,
        "latest_record_sha256": (
            ledger.records[-1]["record_sha256"] if ledger.records else None
        ),
        "ledger_sha256": _sha(ledger_raw),
    }
    latest = _self_hashed(latest_body, "latest_body_sha256")
    return _canonical(latest) + b"\n", records, ledger.genesis_sha256


def _history(
    *,
    allocations: tuple[
        tuple[str, datetime, datetime],
        ...,
    ] = (
        (
            FIRST_INSTANCE,
            datetime(2026, 7, 24, 8, 0, tzinfo=UTC),
            datetime(2026, 7, 24, 10, 0, tzinfo=UTC),
        ),
        (
            SECOND_INSTANCE,
            datetime(2026, 7, 24, 11, 0, tzinfo=UTC),
            datetime(2026, 7, 24, 12, 0, tzinfo=UTC),
        ),
    ),
    state: str = "terminated",
) -> bytes:
    instances: list[dict[str, object]] = []
    for instance_id, launched_at, ended_at in allocations:
        instances.append(
            {
                "InstanceId": instance_id,
                "InstanceType": "p5.48xlarge",
                "LaunchTime": launched_at.isoformat().replace("+00:00", "Z"),
                "State": {"Name": state},
                "StateTransitionReason": (
                    f"User initiated ({ended_at.strftime('%Y-%m-%d %H:%M:%S')} GMT)"
                ),
                "Placement": {"AvailabilityZone": "us-west-2a"},
                "Tags": [
                    {"Key": "project", "Value": "keep-glm52"},
                    {"Key": "owner", "Value": "jack.mazac"},
                    {"Key": "model", "Value": "glm-5.2"},
                    {
                        "Key": "cost-allocation",
                        "Value": "glm52-sky-campaign",
                    },
                    {"Key": "campaign-run-id", "Value": RUN_ID},
                ],
            }
        )
    return (
        _canonical(
            {
                "Reservations": [
                    {
                        "OwnerId": "246813579024",
                        "Instances": instances,
                    }
                ]
            }
        )
        + b"\n"
    )


def _inputs(tmp_path: Path) -> dict[str, object]:
    descriptor_raw, approval_raw = _authorities()
    latest_raw, records, _genesis = _record_publication(tmp_path)
    history_raw = _history()
    return {
        "descriptor_raw": descriptor_raw,
        "expected_descriptor_sha256": _sha(descriptor_raw),
        "approval_raw": approval_raw,
        "expected_approval_sha256": _sha(approval_raw),
        "latest_raw": latest_raw,
        "expected_latest_sha256": _sha(latest_raw),
        "immutable_record_raw_by_key": records,
        "ec2_allocation_history_raw": history_raw,
        "expected_ec2_allocation_history_sha256": _sha(history_raw),
        "observed_at": OBSERVED_AT,
    }


def test_snapshot_rebuilds_exact_closed_chain_and_is_deterministic(
    tmp_path: Path,
) -> None:
    module = _module()
    inputs = _inputs(tmp_path)

    first = module.build_gpu_spend_snapshot(**inputs)
    second = module.build_gpu_spend_snapshot(**inputs)

    assert _canonical(first) == _canonical(second)
    assert set(first) == {
        "schema_version",
        "record_type",
        "run_id",
        "campaign_identity_sha256",
        "descriptor_sha256",
        "descriptor_body_sha256",
        "approval_sha256",
        "approval_body_sha256",
        "gpu_spend_ledger_latest_sha256",
        "gpu_spend_ledger_latest_body_sha256",
        "gpu_spend_ledger_genesis_sha256",
        "gpu_spend_ledger_record_count",
        "gpu_spend_ledger_tip_record_sha256",
        "gpu_spend_ledger_file_sha256",
        "ec2_allocation_history_sha256",
        "ec2_allocation_instance_ids",
        "observed_at",
        "approved_gpu_runtime_seconds",
        "approved_gpu_cost_usd",
        "hourly_cost_usd",
        "consumed_gpu_seconds",
        "remaining_gpu_seconds",
        "consumed_gpu_cost_usd",
        "remaining_gpu_cost_usd",
        "qualification_allowance_seconds",
        "qualification_allowance_cost_usd",
        "open_allocation_count",
        "snapshot_body_sha256",
    }
    assert first["gpu_spend_ledger_record_count"] == 4
    assert first["ec2_allocation_instance_ids"] == [
        FIRST_INSTANCE,
        SECOND_INSTANCE,
    ]
    assert first["consumed_gpu_seconds"] == 10_800
    assert first["remaining_gpu_seconds"] == 75_600
    assert first["consumed_gpu_cost_usd"] == 165.12
    assert first["remaining_gpu_cost_usd"] == 1_155.84
    assert first["qualification_allowance_seconds"] == 14_400
    assert first["qualification_allowance_cost_usd"] == 220.16
    assert first["open_allocation_count"] == 0
    assert module.validate_gpu_spend_snapshot(first) == first


def test_snapshot_golden_byte_identities_are_fixed(tmp_path: Path) -> None:
    descriptor_raw, approval_raw = _authorities()
    latest_raw, records, _genesis = _record_publication(tmp_path)
    history_raw = _history()
    snapshot = _module().build_gpu_spend_snapshot(
        descriptor_raw=descriptor_raw,
        expected_descriptor_sha256=_sha(descriptor_raw),
        approval_raw=approval_raw,
        expected_approval_sha256=_sha(approval_raw),
        latest_raw=latest_raw,
        expected_latest_sha256=_sha(latest_raw),
        immutable_record_raw_by_key=records,
        ec2_allocation_history_raw=history_raw,
        expected_ec2_allocation_history_sha256=_sha(history_raw),
        observed_at=OBSERVED_AT,
    )

    assert {
        "descriptor": _sha(descriptor_raw),
        "approval": _sha(approval_raw),
        "latest": _sha(latest_raw),
        "records": {name: _sha(raw) for name, raw in records.items()},
        "history": _sha(history_raw),
        "snapshot_body": snapshot["snapshot_body_sha256"],
        "snapshot_file": _sha(_canonical(snapshot) + b"\n"),
    } == {
        "descriptor": (
            "f89f62dc6b92450d8358a4e5fd6200a89096f08c25c9704173b9865da6555ab3"
        ),
        "approval": (
            "9e49121eb3462c25ac652f68988c1510372151e43abc2bcb9c107d2713ea531e"
        ),
        "latest": ("eba902e3e9ecf61a55f9960843b20875c11dfc1af7b57f10d6d7adc08ce9ff32"),
        "records": {
            (
                "000000-allocation_started-"
                "f0343a373ac9e2e3293efdf102d88f274021374d9873dd1259d24c12f9fc9f9d"
                ".json"
            ): "c3a266cf72d3d3aa0ccf8e96433ad381ab5d0bb93151af612c6f2ca217e33ece",
            (
                "000001-allocation_ended-"
                "4853f293c44b80b061b4bcb52eb67efe4b10214b87010873ad1bf376341d79be"
                ".json"
            ): "d04456ccfe434b25e19cfcfab34710dc5f1c9d11ad2ba7062df28b8c2faefc8f",
            (
                "000002-allocation_started-"
                "bfdd9d1e368c9798c189b5cb84142a1df36c471ce55b7209c2d4ec85acc220b2"
                ".json"
            ): "7514a27d63bf459e3336c9067af853dce4761493c6dc85bd037451e946748765",
            (
                "000003-allocation_ended-"
                "885460cedc06dadc9a9913cabff2e75f568426c5ac7fe55426a7491e9b332bfc"
                ".json"
            ): "1c91a270a7003562ed3c2284faa57ca0313f33235c4f148108534c3baff86d9a",
        },
        "history": ("2dcccffdbab6f1b6d34f5cd5b88d33ffe1cafc74c776511d176163cbec846374"),
        "snapshot_body": (
            "838e196f69401fe7ec710641fc94bedd3c3c4d6f9528444ecc0b2270093dc1ac"
        ),
        "snapshot_file": (
            "7024c3f47578cb6dbdf09a409a9f28604dfb6525129678f21d11dea481080622"
        ),
    }


@pytest.mark.parametrize("failure", ("missing_tip", "rollback"))
def test_snapshot_rejects_missing_or_rollback_latest_tips(
    tmp_path: Path,
    failure: str,
) -> None:
    module = _module()
    inputs = _inputs(tmp_path)
    records = dict(inputs["immutable_record_raw_by_key"])
    if failure == "missing_tip":
        records.pop(next(reversed(records)))
    else:
        latest = json.loads(inputs["latest_raw"])
        retained_keys = list(latest["record_keys"][:2])
        retained_ledger = b"".join(records[name] for name in retained_keys)
        body = {
            **latest,
            "record_count": 2,
            "record_keys": retained_keys,
            "latest_record_sha256": json.loads(records[retained_keys[-1]])[
                "record_sha256"
            ],
            "ledger_sha256": _sha(retained_ledger),
        }
        body.pop("latest_body_sha256")
        rolled_back = _self_hashed(body, "latest_body_sha256")
        rolled_back_raw = _canonical(rolled_back) + b"\n"
        inputs["latest_raw"] = rolled_back_raw
        inputs["expected_latest_sha256"] = _sha(rolled_back_raw)
    inputs["immutable_record_raw_by_key"] = records

    with pytest.raises(
        module.GpuSpendSnapshotError,
        match="immutable record key set|missing",
    ):
        module.build_gpu_spend_snapshot(**inputs)


def test_snapshot_rejects_foreign_record_even_when_it_is_rehashed(
    tmp_path: Path,
) -> None:
    module = _module()
    inputs = _inputs(tmp_path)
    records = dict(inputs["immutable_record_raw_by_key"])
    first_name = next(iter(records))
    first = json.loads(records.pop(first_name))
    body = {**first, "run_id": "foreign-run"}
    body.pop("record_sha256")
    foreign = _self_hashed(body, "record_sha256")
    foreign_name = f"000000-allocation_started-{foreign['record_sha256']}.json"
    records = {foreign_name: _canonical(foreign) + b"\n", **records}
    latest = json.loads(inputs["latest_raw"])
    keys = [foreign_name, *list(latest["record_keys"])[1:]]
    latest_body = {
        **latest,
        "record_keys": keys,
        "ledger_sha256": _sha(b"".join(records[name] for name in keys)),
    }
    latest_body.pop("latest_body_sha256")
    latest_value = _self_hashed(latest_body, "latest_body_sha256")
    latest_raw = _canonical(latest_value) + b"\n"
    inputs.update(
        latest_raw=latest_raw,
        expected_latest_sha256=_sha(latest_raw),
        immutable_record_raw_by_key=records,
    )

    with pytest.raises(
        module.GpuSpendSnapshotError,
        match="foreign run",
    ):
        module.build_gpu_spend_snapshot(**inputs)


def test_snapshot_rejects_foreign_record_type_even_when_tip_is_rehashed(
    tmp_path: Path,
) -> None:
    module = _module()
    inputs = _inputs(tmp_path)
    records = dict(inputs["immutable_record_raw_by_key"])
    prior_latest = json.loads(inputs["latest_raw"])
    prior_name = list(prior_latest["record_keys"])[-1]
    prior_record = json.loads(records.pop(prior_name))
    body = {
        **prior_record,
        "record_type": "foreign_gpu_spend_event_v1",
    }
    body.pop("record_sha256")
    foreign = _self_hashed(body, "record_sha256")
    foreign_name = f"000003-allocation_ended-{foreign['record_sha256']}.json"
    records[foreign_name] = _canonical(foreign) + b"\n"
    keys = [*list(prior_latest["record_keys"])[:-1], foreign_name]
    latest_body = {
        **prior_latest,
        "record_keys": keys,
        "latest_record_sha256": foreign["record_sha256"],
        "ledger_sha256": _sha(b"".join(records[name] for name in keys)),
    }
    latest_body.pop("latest_body_sha256")
    latest_value = _self_hashed(latest_body, "latest_body_sha256")
    latest_raw = _canonical(latest_value) + b"\n"
    inputs.update(
        latest_raw=latest_raw,
        expected_latest_sha256=_sha(latest_raw),
        immutable_record_raw_by_key=records,
    )

    with pytest.raises(
        module.GpuSpendSnapshotError,
        match="schema mismatch",
    ):
        module.build_gpu_spend_snapshot(**inputs)


def test_snapshot_requires_no_open_allocation(tmp_path: Path) -> None:
    module = _module()
    inputs = _inputs(tmp_path)
    latest_raw, records, _genesis = _record_publication(
        tmp_path / "open",
        allocations=(
            (
                FIRST_INSTANCE,
                datetime(2026, 7, 24, 8, 0, tzinfo=UTC),
                datetime(2026, 7, 24, 10, 0, tzinfo=UTC),
            ),
        ),
        leave_open=True,
    )
    inputs.update(
        latest_raw=latest_raw,
        expected_latest_sha256=_sha(latest_raw),
        immutable_record_raw_by_key=records,
        ec2_allocation_history_raw=_history(
            allocations=(
                (
                    FIRST_INSTANCE,
                    datetime(2026, 7, 24, 8, 0, tzinfo=UTC),
                    datetime(2026, 7, 24, 10, 0, tzinfo=UTC),
                ),
            ),
            state="running",
        ),
    )
    inputs["expected_ec2_allocation_history_sha256"] = _sha(
        inputs["ec2_allocation_history_raw"]
    )

    with pytest.raises(
        module.GpuSpendSnapshotError,
        match="open allocation",
    ):
        module.build_gpu_spend_snapshot(**inputs)


def test_snapshot_rejects_ledger_closed_while_ec2_is_still_stopping(
    tmp_path: Path,
) -> None:
    module = _module()
    inputs = _inputs(tmp_path)
    history_raw = _history(state="stopping")
    inputs.update(
        ec2_allocation_history_raw=history_raw,
        expected_ec2_allocation_history_sha256=_sha(history_raw),
    )

    with pytest.raises(
        module.GpuSpendSnapshotError,
        match="active instance",
    ):
        module.build_gpu_spend_snapshot(**inputs)


@pytest.mark.parametrize("failure", ("duplicate", "over_budget", "time_skew"))
def test_snapshot_rejects_double_over_budget_or_time_skewed_records(
    tmp_path: Path,
    failure: str,
) -> None:
    module = _module()
    inputs = _inputs(tmp_path)
    if failure == "duplicate":
        allocations = (
            (
                FIRST_INSTANCE,
                datetime(2026, 7, 24, 8, 0, tzinfo=UTC),
                datetime(2026, 7, 24, 9, 0, tzinfo=UTC),
            ),
            (
                FIRST_INSTANCE,
                datetime(2026, 7, 24, 10, 0, tzinfo=UTC),
                datetime(2026, 7, 24, 11, 0, tzinfo=UTC),
            ),
        )
        latest_raw, records, _genesis = _record_publication(
            tmp_path / failure,
            allocations=allocations,
            duplicate_instance=True,
        )
        history_raw = _history(allocations=allocations[:1])
        expected_message = "duplicate allocation"
    elif failure == "over_budget":
        allocations = (
            (
                FIRST_INSTANCE,
                datetime(2026, 7, 23, 8, 0, tzinfo=UTC),
                datetime(2026, 7, 24, 9, 0, tzinfo=UTC),
            ),
        )
        latest_raw, records, _genesis = _record_publication(
            tmp_path / failure,
            allocations=allocations,
        )
        history_raw = _history(allocations=allocations)
        expected_message = "exceeds approved"
    else:
        inputs["observed_at"] = datetime(2026, 7, 24, 11, 59, 59, tzinfo=UTC)
        with pytest.raises(
            module.GpuSpendSnapshotError,
            match="observation time precedes",
        ):
            module.build_gpu_spend_snapshot(**inputs)
        return
    inputs.update(
        latest_raw=latest_raw,
        expected_latest_sha256=_sha(latest_raw),
        immutable_record_raw_by_key=records,
        ec2_allocation_history_raw=history_raw,
        expected_ec2_allocation_history_sha256=_sha(history_raw),
    )

    with pytest.raises(
        module.GpuSpendSnapshotError,
        match=expected_message,
    ):
        module.build_gpu_spend_snapshot(**inputs)


def test_snapshot_requires_exact_ec2_history_reconciliation(
    tmp_path: Path,
) -> None:
    module = _module()
    inputs = _inputs(tmp_path)
    history = json.loads(inputs["ec2_allocation_history_raw"])
    history["Reservations"][0]["Instances"][0]["Tags"][0]["Value"] = "foreign-project"
    history_raw = _canonical(history) + b"\n"
    inputs.update(
        ec2_allocation_history_raw=history_raw,
        expected_ec2_allocation_history_sha256=_sha(history_raw),
    )

    with pytest.raises(
        module.GpuSpendSnapshotError,
        match="foreign campaign instance",
    ):
        module.build_gpu_spend_snapshot(**inputs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_owner", "foreign account"),
        ("foreign_owner", "foreign account"),
        ("missing_placement", "foreign region"),
        ("foreign_placement", "foreign region"),
    ),
)
def test_snapshot_requires_explicit_ec2_account_and_region_provenance(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    module = _module()
    inputs = _inputs(tmp_path)
    history = json.loads(inputs["ec2_allocation_history_raw"])
    reservation = history["Reservations"][0]
    instance = reservation["Instances"][0]
    if mutation == "missing_owner":
        reservation.pop("OwnerId")
    elif mutation == "foreign_owner":
        reservation["OwnerId"] = "135792468013"
    elif mutation == "missing_placement":
        instance.pop("Placement")
    else:
        instance["Placement"]["AvailabilityZone"] = "us-west-20a"
    history_raw = _canonical(history) + b"\n"
    inputs.update(
        ec2_allocation_history_raw=history_raw,
        expected_ec2_allocation_history_sha256=_sha(history_raw),
    )

    with pytest.raises(
        module.GpuSpendSnapshotError,
        match=message,
    ):
        module.build_gpu_spend_snapshot(**inputs)


def test_snapshot_validator_rejects_unknown_fields_and_self_hash_drift(
    tmp_path: Path,
) -> None:
    module = _module()
    snapshot = module.build_gpu_spend_snapshot(**_inputs(tmp_path))

    with pytest.raises(
        module.GpuSpendSnapshotError,
        match="schema mismatch",
    ):
        module.validate_gpu_spend_snapshot({**snapshot, "mutable_latest": True})

    with pytest.raises(
        module.GpuSpendSnapshotError,
        match="body SHA-256",
    ):
        module.validate_gpu_spend_snapshot({**snapshot, "remaining_gpu_seconds": 1})


def _write_snapshot_inputs(
    tmp_path: Path,
) -> tuple[dict[str, Path | str], dict[str, object]]:
    values = _inputs(tmp_path)
    descriptor = tmp_path / "descriptor.json"
    approval = tmp_path / "approval.json"
    latest = tmp_path / "latest.json"
    records_dir = tmp_path / "records"
    history = tmp_path / "ec2-history.json"
    records_dir.mkdir()
    descriptor.write_bytes(values["descriptor_raw"])
    approval.write_bytes(values["approval_raw"])
    latest.write_bytes(values["latest_raw"])
    history.write_bytes(values["ec2_allocation_history_raw"])
    for name, raw in values["immutable_record_raw_by_key"].items():
        (records_dir / name).write_bytes(raw)
    return (
        {
            "descriptor": descriptor,
            "approval": approval,
            "latest": latest,
            "records_dir": records_dir,
            "history": history,
            "descriptor_sha256": values["expected_descriptor_sha256"],
            "approval_sha256": values["expected_approval_sha256"],
            "latest_sha256": values["expected_latest_sha256"],
            "history_sha256": values["expected_ec2_allocation_history_sha256"],
        },
        values,
    )


def _fake_aws(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    log = tmp_path / "aws-calls.jsonl"
    aws = fake_bin / "aws"
    aws.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

with Path(os.environ["AWS_CALL_LOG"]).open("a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
account = os.environ.get("FAKE_AWS_ACCOUNT", "246813579024")
print(json.dumps({
    "Account": account,
    "Arn": (
        f"arn:aws:sts::{account}:assumed-role/"
        "AWSReservedSSO_AdministratorAccess/user"
    ),
}))
"""
    )
    aws.chmod(0o755)
    return fake_bin, log


def _run_snapshot_cli(
    tmp_path: Path,
    *,
    profile: str = "keep-gpu",
    account: str = "246813579024",
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    paths, _values = _write_snapshot_inputs(tmp_path)
    fake_bin, log = _fake_aws(tmp_path)
    output = tmp_path / "GPU_SPEND_SNAPSHOT.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_gpu_spend_snapshot.py"),
            "--profile",
            profile,
            "--descriptor",
            str(paths["descriptor"]),
            "--descriptor-sha256",
            str(paths["descriptor_sha256"]),
            "--approval",
            str(paths["approval"]),
            "--approval-sha256",
            str(paths["approval_sha256"]),
            "--ledger-latest",
            str(paths["latest"]),
            "--ledger-latest-sha256",
            str(paths["latest_sha256"]),
            "--ledger-records-dir",
            str(paths["records_dir"]),
            "--ec2-allocation-history",
            str(paths["history"]),
            "--ec2-allocation-history-sha256",
            str(paths["history_sha256"]),
            "--observed-at",
            OBSERVED_AT.isoformat().replace("+00:00", "Z"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "AWS_CALL_LOG": str(log),
            "FAKE_AWS_ACCOUNT": account,
            "PYTHONPATH": str(ROOT / "src"),
        },
    )
    return result, output, log


def test_snapshot_cli_guards_account_then_writes_atomically(
    tmp_path: Path,
) -> None:
    module = _module()
    result, output, log = _run_snapshot_cli(tmp_path)

    assert result.returncode == 0, result.stderr
    value = module.validate_gpu_spend_snapshot(json.loads(output.read_bytes()))
    assert value["consumed_gpu_seconds"] == 10_800
    assert output.read_bytes() == _canonical(value) + b"\n"
    receipt = json.loads(result.stdout)
    assert receipt["gpu_spend_snapshot_sha256"] == _sha(output.read_bytes())
    assert receipt["snapshot_body_sha256"] == value["snapshot_body_sha256"]
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls == [
        [
            "sts",
            "get-caller-identity",
            "--profile",
            "keep-gpu",
            "--output",
            "json",
        ]
    ]
    assert not list(tmp_path.glob(".GPU_SPEND_SNAPSHOT.json.*"))


@pytest.mark.parametrize(
    ("profile", "account", "message"),
    (
        ("default", "246813579024", "keep-gpu"),
        ("keep-gpu", "135792468013", "not approved account"),
    ),
)
def test_snapshot_cli_rejects_unapproved_profile_or_account_before_output(
    tmp_path: Path,
    profile: str,
    account: str,
    message: str,
) -> None:
    result, output, _log = _run_snapshot_cli(
        tmp_path,
        profile=profile,
        account=account,
    )

    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()


def _fake_restore_aws(
    tmp_path: Path,
    *,
    mode: str,
    latest: Path | None = None,
    records_dir: Path | None = None,
) -> tuple[Path, Path]:
    fake_bin = tmp_path / f"restore-fake-bin-{mode}"
    fake_bin.mkdir()
    log = tmp_path / f"restore-aws-{mode}.jsonl"
    aws = fake_bin / "aws"
    aws.write_text(
        """#!/usr/bin/env python3
import json
import os
import shutil
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["AWS_CALL_LOG"]).open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
mode = os.environ["FAKE_RESTORE_MODE"]
if args[:2] == ["sts", "get-caller-identity"]:
    print(json.dumps({
        "Account": "246813579024",
        "Arn": (
            "arn:aws:sts::246813579024:assumed-role/"
            "AWSReservedSSO_AdministratorAccess/user"
        ),
    }))
    raise SystemExit(0)
if args[:2] == ["s3api", "head-object"]:
    if mode == "missing":
        print(
            "An error occurred (404) when calling the HeadObject operation: "
            "Not Found",
            file=sys.stderr,
        )
        raise SystemExit(255)
    if mode == "transport":
        print(
            "Could not connect to the endpoint URL: "
            "https://example.invalid",
            file=sys.stderr,
        )
        raise SystemExit(255)
    print("{}")
    raise SystemExit(0)
if args[:2] == ["s3api", "list-objects-v2"]:
    prefix = args[args.index("--prefix") + 1]
    latest = json.loads(Path(os.environ["RESTORE_LATEST"]).read_bytes())
    keys = [prefix + name for name in latest["record_keys"]]
    if mode == "rollback":
        keys.append(
            prefix
            + "999999-allocation_ended-"
            + ("f" * 64)
            + ".json"
        )
    value = {
        "IsTruncated": mode == "truncated",
        "Contents": [{"Key": key} for key in keys],
    }
    if mode == "truncated":
        value["NextContinuationToken"] = "opaque-next-page"
    print(json.dumps(value))
    raise SystemExit(0)
if args[:2] == ["s3", "cp"]:
    source, destination = args[2], Path(args[3])
    if source.endswith("GPU_SPEND_LEDGER_LATEST.json"):
        shutil.copyfile(os.environ["RESTORE_LATEST"], destination)
        raise SystemExit(0)
    name = source.rsplit("/", 1)[-1]
    shutil.copyfile(Path(os.environ["RESTORE_RECORDS"]) / name, destination)
    raise SystemExit(0)
print("unexpected fake AWS call: " + " ".join(args), file=sys.stderr)
raise SystemExit(97)
"""
    )
    aws.chmod(0o755)
    return fake_bin, log


def _run_required_restore(
    tmp_path: Path,
    *,
    mode: str,
    profile: str = "keep-gpu",
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    latest_raw, records, _genesis = _record_publication(
        tmp_path,
        allocations=()
        if mode == "empty"
        else (
            (
                FIRST_INSTANCE,
                datetime(2026, 7, 24, 8, 0, tzinfo=UTC),
                datetime(2026, 7, 24, 10, 0, tzinfo=UTC),
            ),
            (
                SECOND_INSTANCE,
                datetime(2026, 7, 24, 11, 0, tzinfo=UTC),
                datetime(2026, 7, 24, 12, 0, tzinfo=UTC),
            ),
        ),
    )
    fixture = tmp_path / f"restore-fixture-{mode}"
    records_dir = fixture / "records"
    records_dir.mkdir(parents=True)
    latest = fixture / "latest.json"
    latest.write_bytes(latest_raw)
    for name, raw in records.items():
        (records_dir / name).write_bytes(raw)
    fake_bin, log = _fake_restore_aws(
        tmp_path,
        mode=mode,
        latest=latest,
        records_dir=records_dir,
    )
    root = tmp_path / f"restored-{mode}"
    result = subprocess.run(
        [
            str(SCRIPTS / "restore_gpu_spend_ledger.sh"),
            "--require-latest",
            "--profile",
            profile,
            str(root),
            (f"s3://keep-glm52-us-west-2-246813579024/campaigns/{RUN_ID}/runtime"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "AWS_CALL_LOG": str(log),
            "FAKE_RESTORE_MODE": mode,
            "RESTORE_LATEST": str(latest),
            "RESTORE_RECORDS": str(records_dir),
        },
    )
    return result, root, log


def test_require_latest_restore_reconstructs_exact_immutable_chain(
    tmp_path: Path,
) -> None:
    result, root, log = _run_required_restore(tmp_path, mode="success")

    assert result.returncode == 0, result.stderr
    restored = root / "runtime/GPU_SPEND_LEDGER.jsonl"
    latest_raw, records, _genesis = _record_publication(tmp_path / "expected")
    latest = json.loads(latest_raw)
    assert restored.read_bytes() == b"".join(
        records[name] for name in latest["record_keys"]
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls
    assert all(call[-2:] == ["--profile", "keep-gpu"] for call in calls)


def test_require_latest_restore_requires_exact_named_profile_before_aws(
    tmp_path: Path,
) -> None:
    result, root, log = _run_required_restore(
        tmp_path,
        mode="success",
        profile="default",
    )

    assert result.returncode != 0
    assert "--profile must be exactly keep-gpu" in result.stderr
    assert not (root / "runtime/GPU_SPEND_LEDGER.jsonl").exists()
    assert not log.exists()


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("missing", "required GPU spend latest marker is missing"),
        ("transport", "Could not connect to the endpoint URL"),
        ("rollback", "stale or rollback"),
        ("empty", "contains no immutable records"),
        ("truncated", "inventory is incomplete"),
    ),
)
def test_require_latest_restore_fails_closed_without_legacy_fallback(
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    result, root, log = _run_required_restore(tmp_path, mode=mode)

    assert result.returncode != 0
    assert message in result.stderr
    assert not (root / "runtime/GPU_SPEND_LEDGER.jsonl").exists()
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert all(call[-2:] == ["--profile", "keep-gpu"] for call in calls)
    assert not any(
        any(
            isinstance(argument, str) and argument.endswith("GPU_SPEND_LEDGER.jsonl")
            for argument in call
        )
        for call in calls
    )
