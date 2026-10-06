#!/usr/bin/env python3
"""Install Task 10 wrapper and exact accepted H.1c campaign descriptor."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import NoReturn
from urllib.parse import urlsplit


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task10_worker import (  # noqa: E402
    ACCOUNT_ID,
    DESCRIPTOR_PATH,
    REGION,
    WORKER_BOOTSTRAP_DESCRIPTOR_PATH,
    worker_bootstrap_descriptor_from_mapping,
)
from mlx_vq.quality.glm52_sky_campaign import (  # noqa: E402
    validate_sky_campaign_descriptor,
)


DOWNLOAD = Path("/opt/keep-campaign/download/campaign.json")
WRAPPER = Path(WORKER_BOOTSTRAP_DESCRIPTOR_PATH)
CAMPAIGN = Path(DESCRIPTOR_PATH)


def _fail(message: str) -> NoReturn:
    print("Task 10 descriptor install: " + message, file=sys.stderr)
    raise SystemExit(70)


def _exclusive(path: Path, raw: bytes) -> None:
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
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str]) -> int:
    if argv:
        _fail("this installer accepts no arguments")
    if WRAPPER.exists() or CAMPAIGN.exists():
        _fail("worker descriptors are immutable")
    try:
        wrapper_raw = DOWNLOAD.read_bytes()
        wrapper_value = json.loads(wrapper_raw)
        if (
            type(wrapper_value) is not dict
            or wrapper_raw
            != canonical_json_bytes(wrapper_value) + b"\n"
            or hashlib.sha256(wrapper_raw).hexdigest()
            != os.environ.get("GLM52_DESCRIPTOR_FILE_SHA256")
        ):
            raise ValueError("authenticated wrapper descriptor drifted")
        wrapper = worker_bootstrap_descriptor_from_mapping(wrapper_value)
        if (
            wrapper.repository_archive_version_id
            != os.environ.get("GLM52_REPOSITORY_ARCHIVE_VERSION_ID")
            or wrapper.archive_identity_sha256
            != os.environ.get("GLM52_REPOSITORY_ARCHIVE_FILE_SHA256")
            or wrapper.approval_version_id
            != os.environ.get("GLM52_APPROVAL_VERSION_ID")
            or wrapper.approval_identity_sha256
            != os.environ.get("GLM52_APPROVAL_FILE_SHA256")
            or wrapper.intent_version_id
            != os.environ.get("GLM52_SUBMISSION_INTENT_VERSION_ID")
            or wrapper.intent_identity_sha256
            != os.environ.get("GLM52_SUBMISSION_INTENT_BODY_SHA256")
        ):
            raise ValueError("worker artifact authority drifted")
        parsed = urlsplit(wrapper.base_descriptor_s3_uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path[1:]:
            raise ValueError("accepted H.1c descriptor URI is invalid")
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            region_name=REGION,
            config=Config(
                retries={"mode": "standard", "total_max_attempts": 1},
                connect_timeout=2,
                read_timeout=15,
            ),
        )
        response = client.get_object(
            Bucket=parsed.netloc,
            Key=parsed.path[1:],
            VersionId=wrapper.base_descriptor_version_id,
            ExpectedBucketOwner=ACCOUNT_ID,
            ChecksumMode="ENABLED",
        )
        if response.get("VersionId") != wrapper.base_descriptor_version_id:
            raise ValueError("accepted descriptor VersionId drifted")
        body = response.get("Body")
        if getattr(body, "read", None) is None:
            raise ValueError("accepted descriptor response body is absent")
        campaign_raw = body.read()
        checksum = base64.b64encode(
            hashlib.sha256(campaign_raw).digest()
        ).decode("ascii")
        metadata = response.get("ResponseMetadata")
        if (
            response.get("ChecksumSHA256") != checksum
            or type(metadata) is not dict
            or metadata.get("HTTPStatusCode") != 200
            or hashlib.sha256(campaign_raw).hexdigest()
            != wrapper.base_descriptor_file_sha256
        ):
            raise ValueError("accepted descriptor file identity drifted")
        campaign_value = json.loads(campaign_raw)
        if (
            type(campaign_value) is not dict
            or campaign_raw
            != canonical_json_bytes(campaign_value) + b"\n"
        ):
            raise ValueError("accepted descriptor is not canonical JSON")
        campaign = validate_sky_campaign_descriptor(campaign_value)
        if (
            campaign["descriptor_body_sha256"]
            != wrapper.base_descriptor_body_sha256
            or campaign["campaign_identity_sha256"]
            != wrapper.campaign_identity_sha256
        ):
            raise ValueError("accepted descriptor ancestry drifted")
        _exclusive(WRAPPER, wrapper_raw)
        _exclusive(CAMPAIGN, campaign_raw)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        _fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
