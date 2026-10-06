#!/usr/bin/env python3
"""Materialize fixed self-instance/Task 9 tags for Task 10 bootstrap."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping, NoReturn, Optional
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glm52_enforcement.canonical import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from glm52_enforcement.task10_worker import (  # noqa: E402
    ACCOUNT_ID,
    MountFreeTaskInputs,
    REGION,
    RUN_ID,
    WORKER_BOOTSTRAP_DESCRIPTOR_PATH,
    WORKER_INSTANCE_OBSERVATION_PATH,
    build_jobs_launch_body,
    build_worker_instance_observation,
    render_mount_free_task,
    worker_bootstrap_descriptor_from_mapping,
)


DESCRIPTOR = Path(WORKER_BOOTSTRAP_DESCRIPTOR_PATH)
OUTPUT = Path(WORKER_INSTANCE_OBSERVATION_PATH)
IMDS = "http://169.254.169.254/latest"


def _fail(message: str) -> NoReturn:
    print("Task 10 worker observation: " + message, file=sys.stderr)
    raise SystemExit(70)


def _read_descriptor():
    raw = DESCRIPTOR.read_bytes()
    value = json.loads(raw)
    if (
        type(value) is not dict
        or raw != canonical_json_bytes(value) + b"\n"
        or hashlib.sha256(raw).hexdigest()
        != os.environ.get("GLM52_DESCRIPTOR_FILE_SHA256")
    ):
        raise ValueError("worker bootstrap descriptor is not canonical")
    return worker_bootstrap_descriptor_from_mapping(value)


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if type(value) is not str or not value:
        raise ValueError(name + " is absent")
    return value


def _closed_task_inputs(
    environment: Mapping[str, str],
) -> MountFreeTaskInputs:
    if _required(environment, "GLM52_MANAGED_MODE") != "production":
        raise ValueError("worker managed mode drifted")
    return MountFreeTaskInputs(
        job_name=_required(environment, "GLM52_EXPECTED_SKY_JOB_NAME"),
        descriptor_s3_uri=_required(
            environment,
            "GLM52_DESCRIPTOR_S3_URI",
        ),
        descriptor_version_id=_required(
            environment,
            "GLM52_DESCRIPTOR_VERSION_ID",
        ),
        descriptor_file_sha256=_required(
            environment,
            "GLM52_DESCRIPTOR_FILE_SHA256",
        ),
        approval_s3_uri=_required(environment, "GLM52_APPROVAL_S3_URI"),
        approval_version_id=_required(
            environment,
            "GLM52_APPROVAL_VERSION_ID",
        ),
        approval_file_sha256=_required(
            environment,
            "GLM52_APPROVAL_FILE_SHA256",
        ),
        intent_s3_uri=_required(
            environment,
            "GLM52_SUBMISSION_INTENT_S3_URI",
        ),
        intent_version_id=_required(
            environment,
            "GLM52_SUBMISSION_INTENT_VERSION_ID",
        ),
        intent_file_sha256=_required(
            environment,
            "GLM52_SUBMISSION_INTENT_FILE_SHA256",
        ),
        intent_body_sha256=_required(
            environment,
            "GLM52_SUBMISSION_INTENT_BODY_SHA256",
        ),
        repository_archive_s3_uri=_required(
            environment,
            "GLM52_REPOSITORY_ARCHIVE_S3_URI",
        ),
        repository_archive_version_id=_required(
            environment,
            "GLM52_REPOSITORY_ARCHIVE_VERSION_ID",
        ),
        repository_archive_file_sha256=_required(
            environment,
            "GLM52_REPOSITORY_ARCHIVE_FILE_SHA256",
        ),
    )


def _imds(
    method: str,
    path: str,
    *,
    token: Optional[str] = None,
) -> bytes:
    headers = {}
    if token is not None:
        headers["X-aws-ec2-metadata-token"] = token
    if method == "PUT":
        headers["X-aws-ec2-metadata-token-ttl-seconds"] = "60"
    request = urllib.request.Request(
        IMDS + path,
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.read()


def _self_identity() -> tuple[str, str, str]:
    token = _imds("PUT", "/api/token").decode("ascii")
    if not token:
        raise ValueError("IMDSv2 token is absent")
    raw = _imds(
        "GET",
        "/dynamic/instance-identity/document",
        token=token,
    )
    value = json.loads(raw)
    if (
        type(value) is not dict
        or value.get("accountId") != ACCOUNT_ID
        or value.get("region") != REGION
        or type(value.get("instanceId")) is not str
    ):
        raise ValueError("IMDS instance identity drifted")
    return (
        str(value["instanceId"]),
        str(value["accountId"]),
        str(value["region"]),
    )


def _instance(instance_id: str) -> dict[str, object]:
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "ec2",
        region_name=REGION,
        config=Config(
            retries={"mode": "standard", "total_max_attempts": 1},
            connect_timeout=2,
            read_timeout=15,
        ),
    )
    response = client.describe_instances(
        InstanceIds=[instance_id],
        DryRun=False,
    )
    reservations = response.get("Reservations")
    if type(reservations) is not list or len(reservations) != 1:
        raise ValueError("self instance readback is not singular")
    instances = reservations[0].get("Instances")
    if type(instances) is not list or len(instances) != 1:
        raise ValueError("self instance readback is not singular")
    instance = instances[0]
    if (
        type(instance) is not dict
        or instance.get("InstanceId") != instance_id
        or instance.get("InstanceType") != "p5.48xlarge"
        or instance.get("State", {}).get("Name") not in {"pending", "running"}
    ):
        raise ValueError("self instance shape/state drifted")
    return instance


def _tags(instance: dict[str, object]) -> dict[str, str]:
    raw = instance.get("Tags")
    if type(raw) is not list:
        raise ValueError("self instance tags are absent")
    tags: dict[str, str] = {}
    for item in raw:
        if (
            type(item) is not dict
            or type(item.get("Key")) is not str
            or type(item.get("Value")) is not str
            or item["Key"] in tags
        ):
            raise ValueError("self instance tags are malformed")
        tags[str(item["Key"])] = str(item["Value"])
    return tags


def _write(value: object) -> None:
    raw = canonical_json_bytes(value) + b"\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        OUTPUT,
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
            OUTPUT.unlink()
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str]) -> int:
    if argv:
        _fail("this materializer accepts no arguments")
    if OUTPUT.exists():
        _fail("worker instance observation is immutable")
    try:
        descriptor = _read_descriptor()
        inputs = _closed_task_inputs(os.environ)
        if (
            descriptor.sky_job_name != inputs.job_name
            or descriptor.archive_identity_sha256
            != inputs.repository_archive_file_sha256
            or descriptor.approval_identity_sha256
            != inputs.approval_file_sha256
            or descriptor.intent_identity_sha256
            != inputs.intent_body_sha256
        ):
            raise ValueError(
                "worker descriptor and closed task environment drifted"
            )
        task_yaml = render_mount_free_task(inputs)
        request_body = build_jobs_launch_body(
            task_yaml=task_yaml,
            job_name=inputs.job_name,
        )
        expected_task_sha256 = hashlib.sha256(
            task_yaml.encode("utf-8")
        ).hexdigest()
        expected_request_sha256 = canonical_sha256(request_body)
        instance_id, account_id, region = _self_identity()
        tags = _tags(_instance(instance_id))
        expected = {
            "RunId": RUN_ID,
            "campaign-identity-sha256": (
                descriptor.campaign_identity_sha256
            ),
            "activation-id": descriptor.activation_id,
            "activation-ordinal-text": (
                f"{descriptor.activation_ordinal:08d}"
            ),
            "generation-text": descriptor.generation_text,
            "action-key": descriptor.action_key,
            "sky-job-name": descriptor.sky_job_name,
            "task-yaml-sha256": expected_task_sha256,
            "request-body-sha256": expected_request_sha256,
        }
        if any(tags.get(key) != value for key, value in expected.items()):
            raise ValueError("Task 9 worker tags do not match descriptor")
        allocation_text = tags.get("allocation-ordinal-text")
        action_key = tags.get("action-key")
        if (
            type(allocation_text) is not str
            or len(allocation_text) != 8
            or not allocation_text.isdigit()
            or int(allocation_text) <= 0
            or action_key != descriptor.action_key
        ):
            raise ValueError("Task 9 allocation/action tags are invalid")
        observation = build_worker_instance_observation(
            schema_version=1,
            record_type="glm52_task10_worker_instance_observation_v1",
            account_id=account_id,
            region=region,
            run_id=RUN_ID,
            campaign_identity_sha256=descriptor.campaign_identity_sha256,
            activation_id=descriptor.activation_id,
            activation_ordinal=descriptor.activation_ordinal,
            activation_ordinal_text=(
                f"{descriptor.activation_ordinal:08d}"
            ),
            generation=descriptor.generation,
            generation_text=descriptor.generation_text,
            allocation_ordinal=int(allocation_text),
            allocation_ordinal_text=allocation_text,
            instance_id=instance_id,
            action_key=action_key,
            sky_job_name=descriptor.sky_job_name,
            task_yaml_sha256=expected_task_sha256,
            request_body_sha256=expected_request_sha256,
            task9_launch_identity_sha256=(
                descriptor.task9_launch_identity_sha256
            ),
            task9_custody_identity_sha256=(
                descriptor.task9_custody_identity_sha256
            ),
        )
        _write(asdict(observation))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        _fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
