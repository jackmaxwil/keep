#!/usr/bin/env python3
"""Bounded independent verification of durable BOOTSTRAP ledger completion."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Callable, NoReturn


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glm52_enforcement.canonical import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from glm52_enforcement.task10_durable_s3 import (  # noqa: E402
    exact_version_read,
)
from glm52_enforcement.task10_worker import (  # noqa: E402
    ACCOUNT_ID,
    CAMPAIGN_ROOT,
    DESCRIPTOR_PATH,
    REGION,
    RUN_ID,
    WORKER_BOOTSTRAP_DESCRIPTOR_PATH,
    build_bootstrap_ledger_receipt,
    worker_bootstrap_descriptor_from_mapping,
)
from mlx_vq.quality.glm52_sky_campaign import (  # noqa: E402
    SkyCampaignLedger,
    validate_sky_campaign_descriptor,
)


WAIT_SECONDS = 180
POLL_SECONDS = 1
MARKER = Path(CAMPAIGN_ROOT) / "BOOTSTRAP_READY.json"
LEDGER = Path(CAMPAIGN_ROOT) / "ledger/campaign-ledger.jsonl"
COMPLETION = (
    Path(CAMPAIGN_ROOT) / "runtime/BOOTSTRAP_LEDGER_DURABLE.json"
)
WRAPPER = Path(WORKER_BOOTSTRAP_DESCRIPTOR_PATH)
CAMPAIGN = Path(DESCRIPTOR_PATH)
_VERSION = re.compile(r"^[A-Za-z0-9._~+/=-]{1,1024}$")


def _fail(message: str) -> NoReturn:
    print("Task 10 bootstrap ledger wait: " + message, file=sys.stderr)
    raise SystemExit(70)


def _canonical(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(label + " is not canonical JSON")
    return value, raw


def _service_active() -> bool:
    result = subprocess.run(
        [
            "/usr/bin/systemctl",
            "is-active",
            "--quiet",
            "keep-glm52-campaign.service",
        ],
        check=False,
        timeout=10,
    )
    return result.returncode == 0


def _client() -> object:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=REGION,
        config=Config(
            retries={"mode": "standard", "total_max_attempts": 1},
            connect_timeout=2,
            read_timeout=15,
        ),
    )


def _expected_receipt() -> tuple[dict[str, object], bytes, str]:
    marker, marker_raw = _canonical(MARKER, "BOOTSTRAP_READY")
    wrapper_value, wrapper_raw = _canonical(
        WRAPPER,
        "worker bootstrap descriptor",
    )
    wrapper = worker_bootstrap_descriptor_from_mapping(wrapper_value)
    campaign_value, campaign_raw = _canonical(
        CAMPAIGN,
        "H.1c campaign descriptor",
    )
    campaign = validate_sky_campaign_descriptor(campaign_value)
    if (
        hashlib.sha256(wrapper_raw).hexdigest()
        != marker["deadline_state_descriptor_file_sha256"]
        or hashlib.sha256(campaign_raw).hexdigest()
        != wrapper.base_descriptor_file_sha256
        or campaign["descriptor_body_sha256"]
        != wrapper.base_descriptor_body_sha256
        or campaign["campaign_identity_sha256"]
        != wrapper.campaign_identity_sha256
    ):
        raise ValueError("bootstrap descriptor ancestry drifted")
    ledger_raw = LEDGER.read_bytes()
    lines = ledger_raw.splitlines()
    if not lines:
        raise ValueError("BOOTSTRAP ledger append is absent")
    ledger_record = json.loads(lines[0])
    if type(ledger_record) is not dict:
        raise ValueError("BOOTSTRAP ledger record is malformed")
    ledger = SkyCampaignLedger(
        LEDGER,
        run_id=RUN_ID,
        execution_deadline=wrapper.execution_deadline,
        gpu_spend_authority_sha256=(
            wrapper.task8_spend_authority_identity_sha256
        ),
    )
    if not ledger.records or ledger.records[0].phase.value != "BOOTSTRAP":
        raise ValueError("authenticated BOOTSTRAP ledger append is absent")
    completion, _ = _canonical(
        COMPLETION,
        "BOOTSTRAP ledger durable completion",
    )
    receipt = completion.get("receipt")
    if type(receipt) is not dict:
        raise ValueError("BOOTSTRAP ledger durable receipt is absent")
    expected = build_bootstrap_ledger_receipt(
        marker=marker,
        marker_file_sha256=hashlib.sha256(marker_raw).hexdigest(),
        ledger_record=ledger_record,
        ledger_file_sha256=hashlib.sha256(lines[0] + b"\n").hexdigest(),
        worker_descriptor=wrapper,
        instance_id=str(marker["instance_id"]),
        allocation_ordinal=int(marker["allocation_ordinal"]),
        ledger_record_s3_key=str(receipt.get("ledger_record_s3_key")),
        ledger_record_s3_version_id=str(
            receipt.get("ledger_record_s3_version_id")
        ),
    )
    if receipt != expected:
        raise ValueError("BOOTSTRAP ledger durable receipt drifted")
    return expected, canonical_json_bytes(expected) + b"\n", str(
        campaign["bucket"]
    )


def _validate_completion(
    client: object,
) -> None:
    completion, _ = _canonical(
        COMPLETION,
        "BOOTSTRAP ledger durable completion",
    )
    body = dict(completion)
    observed_body_sha = body.pop("completion_body_sha256", None)
    receipt, receipt_raw, bucket = _expected_receipt()
    receipt_file_sha = hashlib.sha256(receipt_raw).hexdigest()
    receipt_checksum = base64.b64encode(
        hashlib.sha256(receipt_raw).digest()
    ).decode("ascii")
    expected_key = (
        f"campaigns/{RUN_ID}/submissions/production/generations/"
        f"{int(receipt['generation']):08d}/allocations/"
        f"{int(receipt['allocation_ordinal']):08d}/"
        "BOOTSTRAP_LEDGER.json"
    )
    version_id = completion.get("version_id")
    if (
        set(completion)
        != {
            "schema_version",
            "record_type",
            "account_id",
            "region",
            "run_id",
            "generation",
            "allocation_ordinal",
            "bucket",
            "key",
            "version_id",
            "checksum_sha256_base64",
            "receipt_file_sha256",
            "receipt",
            "completion_body_sha256",
        }
        or observed_body_sha != canonical_sha256(body)
        or completion["schema_version"] != 1
        or completion["record_type"]
        != "glm52_task10_bootstrap_ledger_durable_completion_v1"
        or completion["account_id"] != ACCOUNT_ID
        or completion["region"] != REGION
        or completion["run_id"] != RUN_ID
        or completion["generation"] != receipt["generation"]
        or completion["allocation_ordinal"]
        != receipt["allocation_ordinal"]
        or completion["bucket"] != bucket
        or completion["key"] != expected_key
        or type(version_id) is not str
        or version_id == "null"
        or _VERSION.fullmatch(version_id) is None
        or completion["checksum_sha256_base64"] != receipt_checksum
        or completion["receipt_file_sha256"] != receipt_file_sha
    ):
        raise ValueError("BOOTSTRAP ledger durable completion drifted")
    exact_version_read(
        client,
        bucket=bucket,
        key=str(receipt["ledger_record_s3_key"]),
        version_id=str(receipt["ledger_record_s3_version_id"]),
        expected_raw=canonical_json_bytes(
            json.loads(LEDGER.read_bytes().splitlines()[0])
        )
        + b"\n",
    )
    exact_version_read(
        client,
        bucket=bucket,
        key=expected_key,
        version_id=version_id,
        expected_raw=receipt_raw,
    )


def main(
    argv: list[str],
    *,
    client: object | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    service_active: Callable[[], bool] = _service_active,
) -> int:
    if argv:
        _fail("this waiter accepts no arguments")
    deadline = monotonic() + WAIT_SECONDS
    try:
        while not COMPLETION.is_file():
            if not service_active():
                raise ValueError(
                    "campaign controller died before BOOTSTRAP durability"
                )
            if monotonic() >= deadline:
                raise ValueError("BOOTSTRAP ledger durability timed out")
            sleep(POLL_SECONDS)
        _validate_completion(_client() if client is None else client)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        _fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
