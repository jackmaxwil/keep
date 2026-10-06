#!/usr/bin/env python3
"""Retained exact-command observer for Task 10 SSM graceful-stop evidence."""

from __future__ import annotations

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
    exact_version_read,
    publish_or_adopt_exact,
)
from glm52_enforcement.task10_worker import (  # noqa: E402
    ACCOUNT_ID,
    GRACEFUL_STOP_DOCUMENT,
    REGION,
    RUN_ID,
    promote_graceful_stop_evidence_to_ssm,
    retained_ssm_graceful_stop_handoff_from_mapping,
    task12_worker_drain_authority_from_mapping,
    worker_bootstrap_descriptor_from_mapping,
)
from mlx_vq.quality.glm52_sky_campaign import (  # noqa: E402
    validate_sky_campaign_descriptor,
)


HANDOFF = Path(
    "/var/lib/keep-glm52/retained-ssm-graceful-stop-handoff.json"
)
COMPLETION = Path(
    "/var/lib/keep-glm52/retained-ssm-graceful-stop-published.json"
)


def _fail(message: str) -> NoReturn:
    print("Task 10 retained SSM observer: " + message, file=sys.stderr)
    raise SystemExit(70)


def _canonical(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(label + " is not canonical JSON")
    return value, raw


def _clients() -> tuple[object, object]:
    import boto3
    from botocore.config import Config

    config = Config(
        retries={"mode": "standard", "total_max_attempts": 1},
        connect_timeout=2,
        read_timeout=15,
    )
    return (
        boto3.client("ssm", region_name=REGION, config=config),
        boto3.client("s3", region_name=REGION, config=config),
    )


def _write_or_validate(raw: bytes) -> None:
    if COMPLETION.exists():
        if COMPLETION.is_symlink() or COMPLETION.read_bytes() != raw:
            raise ValueError("retained SSM completion drifted")
        return
    COMPLETION.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        COMPLETION,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            COMPLETION.unlink()
        except FileNotFoundError:
            pass
        raise


def main(
    argv: list[str],
    *,
    ssm_client: object | None = None,
    s3_client: object | None = None,
) -> int:
    if argv:
        _fail("this observer accepts no arguments")
    try:
        handoff_value, _ = _canonical(HANDOFF, "retained SSM handoff")
        handoff = retained_ssm_graceful_stop_handoff_from_mapping(
            handoff_value
        )
        wrapper = worker_bootstrap_descriptor_from_mapping(
            handoff.worker_descriptor
        )
        campaign = validate_sky_campaign_descriptor(
            handoff.campaign_descriptor
        )
        if (
            hashlib.sha256(
                canonical_json_bytes(campaign) + b"\n"
            ).hexdigest()
            != wrapper.base_descriptor_file_sha256
            or campaign["descriptor_body_sha256"]
            != wrapper.base_descriptor_body_sha256
            or campaign["campaign_identity_sha256"]
            != wrapper.campaign_identity_sha256
            or handoff.allocation_ordinal <= 0
        ):
            raise ValueError("retained SSM descriptor ancestry drifted")
        if ssm_client is None or s3_client is None:
            default_ssm, default_s3 = _clients()
            ssm_client = default_ssm if ssm_client is None else ssm_client
            s3_client = default_s3 if s3_client is None else s3_client
        task12_authority = task12_worker_drain_authority_from_mapping(
            handoff.task12_authority,
            worker_descriptor=wrapper,
        )
        task12_raw = (
            canonical_json_bytes(handoff.task12_authority) + b"\n"
        )
        task12_metadata = {
            "glm52-account-id": ACCOUNT_ID,
            "glm52-activation-id": wrapper.activation_id,
            "glm52-body-sha256": (
                task12_authority.authority_body_sha256
            ),
            "glm52-file-sha256": hashlib.sha256(
                task12_raw
            ).hexdigest(),
            "glm52-generation-text": wrapper.generation_text,
            "glm52-record-kind": "TASK12_WORKER_DRAIN_AUTHORITY",
            "glm52-region": REGION,
            "glm52-run-id": RUN_ID,
        }
        exact_version_read(
            s3_client,
            bucket=str(campaign["bucket"]),
            key=handoff.task12_authority_key,
            version_id=handoff.task12_authority_version_id,
            expected_raw=task12_raw,
            expected_metadata=task12_metadata,
        )
        get_invocation = getattr(
            ssm_client,
            "get_command_invocation",
            None,
        )
        if not callable(get_invocation):
            raise ValueError("retained SSM readback boundary is absent")
        response = get_invocation(
            CommandId=handoff.command_id,
            InstanceId=handoff.instance_id,
        )
        if type(response) is not dict:
            raise ValueError("retained SSM command readback drifted")
        metadata = response.get("ResponseMetadata")
        output = response.get("StandardOutputContent")
        if (
            type(metadata) is not dict
            or metadata.get("HTTPStatusCode") != 200
            or response.get("CommandId") != handoff.command_id
            or response.get("InstanceId") != handoff.instance_id
            or response.get("DocumentName") != GRACEFUL_STOP_DOCUMENT
            or response.get("DocumentVersion")
            != handoff.document_version
            or response.get("Status") != "Success"
            or response.get("StatusDetails") != "Success"
            or response.get("ResponseCode") != 0
            or response.get("StandardErrorContent") not in {"", None}
            or type(output) is not str
        ):
            raise ValueError("retained SSM command readback drifted")
        observation_raw = output.encode("utf-8")
        observation = json.loads(observation_raw)
        if (
            type(observation) is not dict
            or observation_raw
            != canonical_json_bytes(observation) + b"\n"
        ):
            raise ValueError("retained SSM observation is not canonical")
        evidence = promote_graceful_stop_evidence_to_ssm(
            observation,
            ssm_command_id=handoff.command_id,
            expected_unit_hashes=dict(
                task12_authority.expected_unit_hashes
            ),
            expected_script_hashes=dict(
                task12_authority.expected_script_hashes
            ),
        )
        raw = canonical_json_bytes(evidence) + b"\n"
        bucket = str(campaign["bucket"])
        key = (
            f"campaigns/{RUN_ID}/submissions/production/generations/"
            f"{wrapper.generation:08d}/allocations/"
            f"{handoff.allocation_ordinal:08d}/"
            "WORKER_GRACEFUL_STOP.json"
        )
        publication_metadata = {
            "glm52-account-id": ACCOUNT_ID,
            "glm52-activation-id": wrapper.activation_id,
            "glm52-body-sha256": str(
                evidence["graceful_stop_body_sha256"]
            ),
            "glm52-file-sha256": hashlib.sha256(raw).hexdigest(),
            "glm52-generation-text": wrapper.generation_text,
            "glm52-record-kind": "WORKER_GRACEFUL_STOP",
            "glm52-region": REGION,
            "glm52-run-id": RUN_ID,
        }
        version_id = publish_or_adopt_exact(
            s3_client,
            bucket=bucket,
            key=key,
            raw=raw,
            metadata=publication_metadata,
        )
        completion_body = {
            "schema_version": 1,
            "record_type": (
                "glm52_task10_retained_ssm_graceful_stop_published_v1"
            ),
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "run_id": RUN_ID,
            "command_id": handoff.command_id,
            "instance_id": handoff.instance_id,
            "generation": wrapper.generation,
            "allocation_ordinal": handoff.allocation_ordinal,
            "bucket": bucket,
            "key": key,
            "version_id": version_id,
            "evidence_file_sha256": hashlib.sha256(raw).hexdigest(),
            "evidence_body_sha256": evidence[
                "graceful_stop_body_sha256"
            ],
            "handoff_body_sha256": handoff.handoff_body_sha256,
        }
        completion = {
            **completion_body,
            "completion_body_sha256": canonical_sha256(completion_body),
        }
        _write_or_validate(canonical_json_bytes(completion) + b"\n")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        _fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
