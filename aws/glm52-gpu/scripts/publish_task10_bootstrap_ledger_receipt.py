#!/usr/bin/env python3
"""Publish durable proof of the controller-owned BOOTSTRAP ledger append."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import NoReturn


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glm52_enforcement.canonical import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from glm52_enforcement.task10_durable_s3 import (  # noqa: E402
    publish_or_adopt_exact,
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


MARKER = Path(CAMPAIGN_ROOT) / "BOOTSTRAP_READY.json"
LEDGER = Path(CAMPAIGN_ROOT) / "ledger/campaign-ledger.jsonl"
COMPLETION = (
    Path(CAMPAIGN_ROOT) / "runtime/BOOTSTRAP_LEDGER_DURABLE.json"
)
WRAPPER = Path(WORKER_BOOTSTRAP_DESCRIPTOR_PATH)
CAMPAIGN = Path(DESCRIPTOR_PATH)


def _fail(message: str) -> NoReturn:
    print("Task 10 bootstrap ledger receipt: " + message, file=sys.stderr)
    raise SystemExit(70)


def _canonical(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(label + " is not canonical JSON")
    return value, raw


def _ledger_record() -> tuple[dict[str, object], bytes]:
    raw = LEDGER.read_bytes()
    lines = raw.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError("BOOTSTRAP must be the sole ledger record")
    value = json.loads(lines[0])
    if type(value) is not dict:
        raise ValueError("BOOTSTRAP ledger record is malformed")
    return value, raw


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


def _write_or_validate(path: Path, raw: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("bootstrap ledger completion drifted")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str], *, client: object | None = None) -> int:
    if argv:
        _fail("this publisher accepts no arguments")
    try:
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
        ledger_record, ledger_raw = _ledger_record()
        ledger = SkyCampaignLedger(
            LEDGER,
            run_id=RUN_ID,
            execution_deadline=wrapper.execution_deadline,
            gpu_spend_authority_sha256=(
                wrapper.task8_spend_authority_identity_sha256
            ),
        )
        if len(ledger.records) != 1 or ledger.current_phase.value != "BOOTSTRAP":
            raise ValueError("authenticated BOOTSTRAP ledger append is absent")
        bucket = str(campaign["bucket"])
        ledger_record_raw = canonical_json_bytes(ledger_record) + b"\n"
        ledger_record_key = (
            f"campaigns/{RUN_ID}/submissions/production/generations/"
            f"{wrapper.generation:08d}/allocations/"
            f"{int(marker['allocation_ordinal']):08d}/"
            "BOOTSTRAP_LEDGER_RECORD-"
            f"{ledger_record['record_sha256']}.json"
        )
        boundary = _client() if client is None else client
        ledger_record_version_id = publish_or_adopt_exact(
            boundary,
            bucket=bucket,
            key=ledger_record_key,
            raw=ledger_record_raw,
            metadata={},
        )
        receipt = build_bootstrap_ledger_receipt(
            marker=marker,
            marker_file_sha256=hashlib.sha256(marker_raw).hexdigest(),
            ledger_record=ledger_record,
            ledger_file_sha256=hashlib.sha256(ledger_raw).hexdigest(),
            worker_descriptor=wrapper,
            instance_id=str(marker["instance_id"]),
            allocation_ordinal=int(marker["allocation_ordinal"]),
            ledger_record_s3_key=ledger_record_key,
            ledger_record_s3_version_id=ledger_record_version_id,
        )
        receipt_raw = canonical_json_bytes(receipt) + b"\n"
        key = (
            f"campaigns/{RUN_ID}/submissions/production/generations/"
            f"{wrapper.generation:08d}/allocations/"
            f"{int(marker['allocation_ordinal']):08d}/"
            "BOOTSTRAP_LEDGER.json"
        )
        metadata = {
            "glm52-account-id": ACCOUNT_ID,
            "glm52-activation-id": wrapper.activation_id,
            "glm52-body-sha256": str(receipt["receipt_body_sha256"]),
            "glm52-file-sha256": hashlib.sha256(
                receipt_raw
            ).hexdigest(),
            "glm52-generation-text": wrapper.generation_text,
            "glm52-record-kind": "BOOTSTRAP_LEDGER",
            "glm52-region": REGION,
            "glm52-run-id": RUN_ID,
        }
        version_id = publish_or_adopt_exact(
            boundary,
            bucket=bucket,
            key=key,
            raw=receipt_raw,
            metadata=metadata,
        )
        completion_body = {
            "schema_version": 1,
            "record_type": (
                "glm52_task10_bootstrap_ledger_durable_completion_v1"
            ),
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "run_id": RUN_ID,
            "generation": wrapper.generation,
            "allocation_ordinal": int(marker["allocation_ordinal"]),
            "bucket": bucket,
            "key": key,
            "version_id": version_id,
            "checksum_sha256_base64": base64.b64encode(
                hashlib.sha256(receipt_raw).digest()
            ).decode("ascii"),
            "receipt_file_sha256": hashlib.sha256(
                receipt_raw
            ).hexdigest(),
            "receipt": receipt,
        }
        completion = {
            **completion_body,
            "completion_body_sha256": canonical_sha256(completion_body),
        }
        _write_or_validate(
            COMPLETION,
            canonical_json_bytes(completion) + b"\n",
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        _fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
