"""Injected dynamic-v2 worker-start coordinator I/O tests."""

from __future__ import annotations

import base64
import builtins
import hashlib
import importlib.util
import json
import shlex
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py"
AUTHORITY_TEST_PATH = ROOT / "tests/test_glm52_sky_worker_must_start_v2.py"

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
RUN_ID = "glm52-sky-20260726"
INTENT_SHA = "1" * 64
LATCH_SHA = "2" * 64
INSTANCE_ID = "i-0123456789abcdef0"
DESCRIPTOR_KEY = (
    f"campaigns/{RUN_ID}/submissions/qualification/campaign-descriptor-v2.json"
)
INTENT_KEY = (
    f"campaigns/{RUN_ID}/submissions/qualification/intents/{INTENT_SHA}/"
    "SKYPILOT_SUBMISSION_INTENT.json"
)
LATCH_KEY = (
    f"campaigns/{RUN_ID}/monitor/must-start/qualification/{INTENT_SHA}/"
    f"worker-latches/{INSTANCE_ID}/{LATCH_SHA}.json"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_sky_worker_start_v2_coordinator",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_authority_fixtures() -> Any:
    name = "_task_3n_authority_fixtures"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, AUTHORITY_TEST_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.WORKER_ROLE = "arn:aws:iam::246813579024:role/keep-glm52-gpu-worker"
    return module


def _preload_flat_dependencies() -> None:
    for name in (
        "glm52_sky_campaign",
        "glm52_sky_must_start",
        "glm52_sky_must_start_dynamic",
        "glm52_sky_worker_must_start_v2",
    ):
        if name in sys.modules:
            continue
        path = ROOT / f"src/mlx_vq/quality/{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


class _NoCalls:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __getattr__(self, name: str) -> Any:
        def unexpected(**kwargs: object) -> object:
            self.calls.append((name, dict(kwargs)))
            raise AssertionError(f"unexpected external call: {name}")

        return unexpected


def _snapshot(value: object) -> object:
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return value


class FakeClientError(Exception):
    def __init__(self, code: str, message: str = "injected") -> None:
        super().__init__(message)
        self.response = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": 412},
        }


class ProcessDeath(BaseException):
    pass


class OneShotBody:
    def __init__(
        self,
        raw: bytes,
        *,
        read_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._stream = BytesIO(raw)
        self._read_error = read_error
        self._close_error = close_error
        self.read_count = 0
        self.close_count = 0

    def read(self) -> bytes:
        self.read_count += 1
        if self.read_count != 1:
            raise AssertionError("body stream read more than once")
        if self._read_error is not None:
            raise self._read_error
        return self._stream.read()

    def close(self) -> None:
        self.close_count += 1
        if self.close_count != 1:
            raise AssertionError("body stream closed more than once")
        if self._close_error is not None:
            raise self._close_error
        self._stream.close()


@dataclass
class StoredVersion:
    raw: bytes
    version_id: str
    etag: str
    last_modified: datetime
    content_length: int
    checksum: str
    checksum_type: str
    content_type: str
    metadata: dict[str, str]


class FakeS3:
    def __init__(
        self,
        ledger: list[tuple[str, str, object]],
    ) -> None:
        self.ledger = ledger
        self.calls: list[tuple[str, object]] = []
        self.versions: dict[tuple[str, str, str], StoredVersion] = {}
        self.current: dict[tuple[str, str], str] = {}
        self.body_streams: list[OneShotBody] = []
        self.list_page_size = 1000
        self.put_faults: dict[str, list[BaseException | Exception | str]] = {}
        self.put_fault_sequence: dict[int, BaseException | Exception | str] = {}
        self.pre_put_fault_sequence: dict[int, BaseException | Exception] = {}
        self.pre_put_hooks: dict[int, Any] = {}
        self.post_put_hooks: dict[int, Any] = {}
        self.body_faults: dict[str, tuple[Exception | None, Exception | None]] = {}
        self.after_list: Any = None
        self._put_sequence = 0

    def add(
        self,
        *,
        bucket: str = BUCKET,
        key: str,
        raw: bytes,
        run_id: str,
        body_sha256: str,
        version_id: str | None = None,
        last_modified: datetime | None = None,
        make_current: bool = True,
        **changes: object,
    ) -> str:
        version = version_id or f"v-{len(self.versions) + 1}"
        checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
        value = StoredVersion(
            raw=raw,
            version_id=version,
            etag=f'"{hashlib.md5(raw, usedforsecurity=False).hexdigest()}"',
            last_modified=last_modified or datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
            content_length=len(raw),
            checksum=checksum,
            checksum_type="FULL_OBJECT",
            content_type="application/json",
            metadata={
                "glm52-run-id": run_id,
                "glm52-body-sha256": body_sha256,
            },
        )
        for name, item in changes.items():
            setattr(value, name, item)
        self.versions[(bucket, key, version)] = value
        if make_current:
            self.current[(bucket, key)] = version
        return version

    def _record(self, operation: str, kwargs: dict[str, object]) -> None:
        snapshot = _snapshot(kwargs)
        self.calls.append((operation, snapshot))
        self.ledger.append(("s3", operation, snapshot))

    def _entry(
        self,
        key: str,
        version_id: str | None,
        *,
        bucket: str = BUCKET,
    ) -> StoredVersion:
        version = version_id or self.current.get((bucket, key))
        identity = (bucket, key, version)
        if version is None or identity not in self.versions:
            raise FakeClientError("NoSuchKey")
        return self.versions[identity]

    def keys(self, *, bucket: str = BUCKET) -> list[str]:
        return sorted(
            key for stored_bucket, key in self.current if stored_bucket == bucket
        )

    @staticmethod
    def _transport(entry: StoredVersion) -> dict[str, object]:
        return {
            "VersionId": entry.version_id,
            "ETag": entry.etag,
            "LastModified": entry.last_modified,
            "ContentLength": entry.content_length,
            "ChecksumSHA256": entry.checksum,
            "ChecksumType": entry.checksum_type,
            "ContentType": entry.content_type,
            "Metadata": dict(entry.metadata),
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    def head_object(self, **kwargs: object) -> dict[str, object]:
        self._record("head_object", dict(kwargs))
        entry = self._entry(
            str(kwargs["Key"]),
            str(kwargs["VersionId"]) if "VersionId" in kwargs else None,
            bucket=str(kwargs["Bucket"]),
        )
        return self._transport(entry)

    def get_object(self, **kwargs: object) -> dict[str, object]:
        self._record("get_object", dict(kwargs))
        bucket = str(kwargs["Bucket"])
        key = str(kwargs["Key"])
        entry = self._entry(
            key,
            str(kwargs["VersionId"]) if "VersionId" in kwargs else None,
            bucket=bucket,
        )
        read_error, close_error = self.body_faults.get(key, (None, None))
        stream = OneShotBody(
            entry.raw,
            read_error=read_error,
            close_error=close_error,
        )
        self.body_streams.append(stream)
        return {**self._transport(entry), "Body": stream}

    def list_objects_v2(self, **kwargs: object) -> dict[str, object]:
        self._record("list_objects_v2", dict(kwargs))
        bucket = str(kwargs["Bucket"])
        prefix = str(kwargs["Prefix"])
        keys = [key for key in self.keys(bucket=bucket) if key.startswith(prefix)]
        token = kwargs.get("ContinuationToken")
        offset = int(str(token).removeprefix("t-")) if token is not None else 0
        page = keys[offset : offset + self.list_page_size]
        next_offset = offset + len(page)
        truncated = next_offset < len(keys)
        result: dict[str, object] = {
            "IsTruncated": truncated,
            "KeyCount": len(page),
            "Contents": [
                {
                    "Key": key,
                    "Size": self._entry(
                        key,
                        None,
                        bucket=bucket,
                    ).content_length,
                }
                for key in page
            ],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }
        if truncated:
            result["NextContinuationToken"] = f"t-{next_offset}"
        if self.after_list is not None:
            callback = self.after_list
            self.after_list = None
            callback(prefix)
        return result

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self._record("put_object", dict(kwargs))
        bucket = str(kwargs["Bucket"])
        key = str(kwargs["Key"])
        self._put_sequence += 1
        hook = self.pre_put_hooks.get(self._put_sequence)
        if hook is not None:
            hook(self, key, dict(kwargs))
        pre_fault = self.pre_put_fault_sequence.get(self._put_sequence)
        if pre_fault is not None:
            raise pre_fault
        if kwargs.get("IfNoneMatch") == "*" and (bucket, key) in self.current:
            raise FakeClientError("PreconditionFailed")
        raw = bytes(kwargs["Body"])
        metadata = dict(kwargs["Metadata"])
        version = self.add(
            bucket=bucket,
            key=key,
            raw=raw,
            run_id=str(metadata["glm52-run-id"]),
            body_sha256=str(metadata["glm52-body-sha256"]),
            version_id=f"put-{self._put_sequence}",
            last_modified=datetime(2026, 7, 26, 12, 0, 10, tzinfo=UTC),
        )
        hook = self.post_put_hooks.get(self._put_sequence)
        if hook is not None:
            hook(self, key)
        fault_queue = self.put_faults.get(key, [])
        fault = self.put_fault_sequence.get(self._put_sequence)
        if fault is None and fault_queue:
            fault = fault_queue.pop(0)
        if fault is not None:
            if isinstance(fault, BaseException):
                raise fault
            if fault == "bad-response":
                return {"ResponseMetadata": {"HTTPStatusCode": 200}}
            if fault == "float-status":
                return {
                    "VersionId": version,
                    "ETag": self._entry(key, version, bucket=bucket).etag,
                    "ChecksumSHA256": self._entry(
                        key,
                        version,
                        bucket=bucket,
                    ).checksum,
                    "ResponseMetadata": {"HTTPStatusCode": 200.0},
                }
        return {
            "VersionId": version,
            "ETag": self._entry(key, version, bucket=bucket).etag,
            "ChecksumSHA256": self._entry(
                key,
                version,
                bucket=bucket,
            ).checksum,
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }


class FakeSts:
    def __init__(self, ledger: list[tuple[str, str, object]]) -> None:
        self.ledger = ledger
        self.calls: list[tuple[str, object]] = []

    def get_caller_identity(self) -> dict[str, object]:
        self.calls.append(("get_caller_identity", {}))
        self.ledger.append(("sts", "get_caller_identity", {}))
        return {
            "Account": ACCOUNT_ID,
            "Arn": (
                "arn:aws:sts::246813579024:assumed-role/"
                "AWSReservedSSO_AdministratorAccess_0123456789abcdef/"
                "operator@example.com"
            ),
            "UserId": "AROATEST:operator@example.com",
        }


class FakeEc2:
    def __init__(
        self,
        ledger: list[tuple[str, str, object]],
        *,
        controller: dict[str, object],
        worker: dict[str, object],
    ) -> None:
        self.ledger = ledger
        self.calls: list[tuple[str, object]] = []
        self.controller = controller
        self.worker = worker
        self.filter_pages: dict[str, list[dict[str, object]]] = {
            "cluster": [self._page([worker])],
            "campaign": [self._page([worker])],
        }
        tags = {
            str(item["Key"]): str(item["Value"])
            for item in worker["Tags"]
            if isinstance(item, dict)
        }
        campaign_tags = {
            name: tags[name]
            for name in (
                "project",
                "owner",
                "model",
                "campaign-run-id",
                "cost-allocation",
            )
        }
        base_filters = [
            {"Name": f"tag:{name}", "Values": [value]}
            for name, value in sorted(campaign_tags.items())
        ]
        cluster = tags["ray-cluster-name"]
        self.expected_filters = {
            "cluster": [
                *base_filters,
                {"Name": "tag:ray-cluster-name", "Values": [cluster]},
                {
                    "Name": "tag:skypilot-cluster-name",
                    "Values": [cluster],
                },
                {
                    "Name": "instance-state-name",
                    "Values": ["pending", "running", "stopping"],
                },
            ],
            "campaign": [
                *base_filters,
                {"Name": "instance-type", "Values": ["p5.48xlarge"]},
                {
                    "Name": "instance-state-name",
                    "Values": ["pending", "running", "stopping"],
                },
            ],
        }
        self.consumed_requests: set[tuple[str, str | None]] = set()
        self.exact_requests: set[str] = set()

    @staticmethod
    def _page(instances: list[dict[str, object]]) -> dict[str, object]:
        return {
            "Reservations": [
                {
                    "OwnerId": ACCOUNT_ID,
                    "Instances": deepcopy(instances),
                }
            ],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        snapshot = _snapshot(kwargs)
        self.calls.append(("describe_instances", snapshot))
        self.ledger.append(("ec2", "describe_instances", snapshot))
        if "InstanceIds" in kwargs:
            if set(kwargs) != {"InstanceIds"}:
                raise AssertionError("unexpected EC2 exact-ID request shape")
            instance_ids = kwargs["InstanceIds"]
            if type(instance_ids) is not list or len(instance_ids) != 1:
                raise AssertionError("unexpected EC2 exact-ID request")
            instance_id = str(instance_ids[0])
            if instance_id in self.exact_requests:
                raise AssertionError("repeated EC2 exact-ID request")
            self.exact_requests.add(instance_id)
            instances = {
                str(self.controller["InstanceId"]): self.controller,
                str(self.worker["InstanceId"]): self.worker,
            }
            if instance_id not in instances:
                raise AssertionError("unconfigured EC2 exact-ID request")
            return self._page([instances[instance_id]])
        filters = kwargs.get("Filters")
        if type(filters) is not list or set(kwargs) - {
            "Filters",
            "NextToken",
        }:
            raise AssertionError("unexpected EC2 filtered request shape")
        matching = [
            kind
            for kind, expected in self.expected_filters.items()
            if filters == expected
        ]
        if len(matching) != 1:
            raise AssertionError("unconfigured EC2 filtered request")
        kind = matching[0]
        token = kwargs.get("NextToken")
        if token is not None and not isinstance(token, str):
            raise AssertionError("EC2 token is not opaque text")
        request_identity = (kind, token)
        if request_identity in self.consumed_requests:
            raise AssertionError("repeated EC2 page request")
        self.consumed_requests.add(request_identity)
        pages = self.filter_pages[kind]
        if token is None:
            index = 0
        else:
            predecessors = [
                index
                for index, page in enumerate(pages)
                if page.get("NextToken") == token
            ]
            if len(predecessors) != 1:
                raise AssertionError("unconfigured EC2 opaque token")
            index = predecessors[0] + 1
        if index >= len(pages):
            raise AssertionError("EC2 token names a missing page")
        return deepcopy(pages[index])


class FakeSsm:
    def __init__(
        self,
        ledger: list[tuple[str, str, object]],
        *,
        controller_id: str,
        history: dict[str, object],
        history_command: str,
    ) -> None:
        self.ledger = ledger
        self.calls: list[tuple[str, object]] = []
        self.controller_id = controller_id
        self.history = history
        self.info_pages: list[dict[str, object]] = [
            {
                "InstanceInformationList": [
                    {
                        "InstanceId": controller_id,
                        "PingStatus": "Online",
                    }
                ],
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }
        ]
        self.info_consumed: set[str | None] = set()
        self.expected_info_filters = [
            {
                "Key": "InstanceIds",
                "Values": [controller_id],
            }
        ]
        self.expected_send = {
            "InstanceIds": [controller_id],
            "DocumentName": "AWS-RunShellScript",
            "Comment": "KEEP GLM52 observe exact dynamic-v2 worker authority",
            "Parameters": {"commands": [history_command]},
            "TimeoutSeconds": 45,
        }
        self.send_consumed = False
        self.invocations: list[dict[str, object] | Exception] = [
            {
                "CommandId": "command-1",
                "InstanceId": controller_id,
                "Status": "Success",
                "ResponseCode": 0,
                "StandardErrorContent": "",
                "StandardOutputContent": (
                    "GLM52_SUBMISSION_HISTORY="
                    + json.dumps(
                        history,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }
        ]

    def describe_instance_information(self, **kwargs: object) -> dict[str, object]:
        snapshot = _snapshot(kwargs)
        self.calls.append(("describe_instance_information", snapshot))
        self.ledger.append(("ssm", "describe_instance_information", snapshot))
        if (
            set(kwargs) - {"Filters", "NextToken"}
            or kwargs.get("Filters") != self.expected_info_filters
        ):
            raise AssertionError("unconfigured SSM information request")
        token = kwargs.get("NextToken")
        if token is not None and not isinstance(token, str):
            raise AssertionError("SSM token is not opaque text")
        if token in self.info_consumed:
            raise AssertionError("repeated SSM information page request")
        self.info_consumed.add(token)
        if token is None:
            index = 0
        else:
            predecessors = [
                index
                for index, page in enumerate(self.info_pages)
                if page.get("NextToken") == token
            ]
            if len(predecessors) != 1:
                raise AssertionError("unconfigured SSM opaque token")
            index = predecessors[0] + 1
        if index >= len(self.info_pages):
            raise AssertionError("SSM token names a missing page")
        return deepcopy(self.info_pages[index])

    def send_command(self, **kwargs: object) -> dict[str, object]:
        snapshot = _snapshot(kwargs)
        self.calls.append(("send_command", snapshot))
        self.ledger.append(("ssm", "send_command", snapshot))
        if kwargs != self.expected_send or self.send_consumed:
            raise AssertionError("unconfigured or repeated SSM command")
        self.send_consumed = True
        return {
            "Command": {
                "CommandId": "command-1",
            },
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    def get_command_invocation(self, **kwargs: object) -> dict[str, object]:
        snapshot = _snapshot(kwargs)
        self.calls.append(("get_command_invocation", snapshot))
        self.ledger.append(("ssm", "get_command_invocation", snapshot))
        if kwargs != {
            "CommandId": "command-1",
            "InstanceId": self.controller_id,
        }:
            raise AssertionError("unconfigured SSM invocation request")
        if not self.invocations:
            raise AssertionError("unexpected extra invocation poll")
        result = self.invocations.pop(0)
        if isinstance(result, Exception):
            raise result
        return deepcopy(result)


def _canonical(value: object, *, newline: bool) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return raw + (b"\n" if newline else b"")


def _body_sha(value: dict[str, object]) -> str:
    fields_by_type = {
        "glm52_sky_campaign_descriptor_v2": "descriptor_body_sha256",
        "glm52_sky_submission_intent_v2": "intent_body_sha256",
        "glm52_sky_submission_acquired_v1": "acquisition_body_sha256",
        "glm52_controller_baseline_v1": "baseline_body_sha256",
        "glm52_sky_submission_accepted_v2": "accepted_body_sha256",
        "glm52_sky_must_start_job_binding_v2": "job_binding_body_sha256",
        "glm52_sky_must_start_controller_observation_v1": ("observation_body_sha256"),
        "glm52_sky_worker_start_latch_v2": "worker_latch_body_sha256",
        "glm52_sky_worker_start_accepted_v2": ("worker_acceptance_body_sha256"),
    }
    field = fields_by_type.get(str(value.get("record_type")))
    if field is None or not isinstance(value.get(field), str):
        raise AssertionError("fixture has no body digest")
    return str(value[field])


def _request(module: Any, **changes: object) -> Any:
    values: dict[str, object] = {
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "bucket": BUCKET,
        "descriptor_key": DESCRIPTOR_KEY,
        "descriptor_file_sha256": "3" * 64,
        "intent_key": INTENT_KEY,
        "intent_file_sha256": "4" * 64,
        "intent_body_sha256": INTENT_SHA,
        "worker_latch_key": LATCH_KEY,
        "worker_latch_version_id": "v1-worker-latch",
    }
    values.update(changes)
    return module.WorkerStartCoordinatorRequest(**values)


def _services(module: Any) -> Any:
    return module.WorkerStartCoordinatorServices(
        sts=_NoCalls(),
        s3=_NoCalls(),
        ec2=_NoCalls(),
        ssm=_NoCalls(),
        clock=lambda: datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
        sleep=lambda _: None,
    )


@dataclass
class CampaignFixture:
    module: Any
    authority: Any
    upstream: dict[str, object]
    latch: dict[str, object]
    request: Any
    services: Any
    s3: FakeS3
    ec2: FakeEc2
    ssm: FakeSsm
    ledger: list[tuple[str, str, object]]
    clock_calls: list[datetime]
    sleeps: list[float]


def _advance_recovery(
    fixture: CampaignFixture,
    *,
    instance_id: str,
    cluster: str,
    recovery_count: int,
    pending_at: datetime,
    observed_at: datetime,
) -> tuple[Any, Any]:
    authority = fixture.authority
    pure = authority._module()
    latch = authority._latch(
        pure,
        fixture.upstream,
        instance_id=instance_id,
        pending_at=pending_at,
        entrypoint_at=pending_at + timedelta(seconds=1),
    )
    latch_key = pure.worker_start_latch_v2_s3_key(worker_latch=latch)
    latch_version = fixture.s3.add(
        key=latch_key,
        raw=_canonical(latch, newline=False),
        run_id=authority.RUN_ID,
        body_sha256=str(latch["worker_latch_body_sha256"]),
        version_id=f"v-{instance_id}-latch",
        last_modified=pending_at + timedelta(seconds=1),
    )
    request = replace(
        fixture.request,
        worker_latch_key=latch_key,
        worker_latch_version_id=latch_version,
    )
    worker = _instance(
        authority,
        instance_id=instance_id,
        instance_type="p5.48xlarge",
        image_id=authority.AMI_ID,
        profile_arn=(
            "arn:aws:iam::246813579024:instance-profile/keep-glm52-gpu-worker"
        ),
        state="running",
        launch_time=pending_at,
        cluster=cluster,
        controller=False,
    )
    ec2 = FakeEc2(
        fixture.ledger,
        controller=deepcopy(fixture.ec2.controller),
        worker=worker,
    )
    history = {
        "schema_version": 1,
        "record_type": "glm52_sky_controller_history_v1",
        "sky_job_name": authority.SKY_JOB_NAME,
        "workspace": "default",
        "rows": [
            {
                "sky_job_id": authority.SKY_JOB_ID,
                "sky_job_name": authority.SKY_JOB_NAME,
                "workspace": "default",
                "controller_submitted_at": authority._iso(authority.SUBMITTED_AT),
                "controller_status": "RECOVERING",
                "controller_identity": authority.CONTROLLER_ROLE,
                "schedule_state": "ALIVE",
                "start_at": authority._iso(pending_at + timedelta(seconds=2)),
                "worker_cluster_name": cluster,
                "recovery_count": recovery_count,
            }
        ],
        "observed_at": authority._iso(observed_at),
    }
    ssm = FakeSsm(
        fixture.ledger,
        controller_id=authority.CONTROLLER_INSTANCE_ID,
        history=history,
        history_command=fixture.module._history_command(
            job_id=authority.SKY_JOB_ID,
            job_name=authority.SKY_JOB_NAME,
            workspace="default",
            controller_identity=authority.CONTROLLER_ROLE,
        ),
    )
    calls: list[datetime] = []

    def clock() -> datetime:
        calls.append(observed_at)
        fixture.ledger.append(("clock", "now", {}))
        assert len(calls) == 1
        return observed_at

    services = fixture.module.WorkerStartCoordinatorServices(
        sts=FakeSts(fixture.ledger),
        s3=fixture.s3,
        ec2=ec2,
        ssm=ssm,
        clock=clock,
        sleep=lambda _: None,
    )
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=services,
        request=request,
    )
    return outcome, request


def _build_three_worker_chain() -> tuple[CampaignFixture, Any]:
    authority = _load_authority_fixtures()
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=authority.WORKER_CLUSTER,
    )
    initial = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert initial.status == "accepted-initial"
    first_pending = authority.MUST_START_BY + timedelta(minutes=1)
    first, _ = _advance_recovery(
        fixture,
        instance_id=authority.RECOVERY_INSTANCE_ID,
        cluster="sky-glm52-worker-b",
        recovery_count=1,
        pending_at=first_pending,
        observed_at=first_pending + timedelta(seconds=3),
    )
    assert first.status == "accepted-managed-recovery"
    second_pending = authority.MUST_START_BY + timedelta(minutes=2)
    second, second_request = _advance_recovery(
        fixture,
        instance_id=authority.SECOND_RECOVERY_INSTANCE_ID,
        cluster="sky-glm52-worker-c",
        recovery_count=3,
        pending_at=second_pending,
        observed_at=second_pending + timedelta(seconds=3),
    )
    assert second.status == "accepted-managed-recovery"
    return fixture, second_request


def _existing_replay_services(
    fixture: CampaignFixture,
    *,
    now: datetime,
) -> Any:
    calls: list[datetime] = []

    def clock() -> datetime:
        calls.append(now)
        assert len(calls) == 1
        return now

    return fixture.module.WorkerStartCoordinatorServices(
        sts=FakeSts(fixture.ledger),
        s3=fixture.s3,
        ec2=_NoCalls(),
        ssm=_NoCalls(),
        clock=clock,
        sleep=lambda _: pytest.fail("existing replay must not sleep"),
    )


def _stored_worker_acceptances(
    fixture: CampaignFixture,
) -> dict[str, dict[str, object]]:
    values: dict[str, dict[str, object]] = {}
    for key in fixture.s3.keys():
        if "/worker-acceptances/" not in key:
            continue
        value = json.loads(fixture.s3._entry(key, None).raw)
        assert isinstance(value, dict)
        values[key] = value
    return values


def _instance(
    authority: Any,
    *,
    instance_id: str,
    instance_type: str,
    image_id: str,
    profile_arn: str,
    state: str,
    launch_time: datetime,
    cluster: str,
    controller: bool,
) -> dict[str, object]:
    tags = {
        "project": "keep-glm52",
        "owner": "jack.mazac",
        "model": "glm-5.2",
        "campaign-run-id": authority.RUN_ID,
        "cost-allocation": "glm52-sky-campaign",
        "ray-cluster-name": cluster,
    }
    if not controller:
        tags["skypilot-cluster-name"] = cluster
    return {
        "InstanceId": instance_id,
        "InstanceType": instance_type,
        "ImageId": image_id,
        "State": {"Name": state},
        "LaunchTime": launch_time,
        "Placement": {"AvailabilityZone": "us-west-2a"},
        "IamInstanceProfile": {"Arn": profile_arn},
        "Tags": [{"Key": key, "Value": value} for key, value in sorted(tags.items())],
    }


def _campaign_fixture(
    *,
    controller_status: str = "STARTING",
    worker_cluster: str | None = None,
    start_at: datetime | None = None,
    recovery_count: int = 0,
    controller_observed_at: datetime | None = None,
    clock_at: datetime | None = None,
) -> CampaignFixture:
    module = _load_module()
    authority = _load_authority_fixtures()
    descriptor_seed = authority._descriptor()
    artifacts = dict(descriptor_seed["artifacts"])
    artifacts["non_vq_prefix"] = "non-vq-package/"
    artifacts["training_baseline_prefix"] = "training-baseline/"
    artifacts["qualification_cache_prefix"] = (
        f"qualification-cache/seeds/{authority.RUN_ID}/"
        f"{artifacts['qualification_cache_manifest_sha256']}/"
    )
    artifacts["teich_pack_key"] = "teich-pack/pack.json"
    artifacts["frozen_prompt_pack_key"] = "quality/frozen.json"
    artifacts["training_config_key"] = (
        f"campaigns/{authority.RUN_ID}/authorities/training.json"
    )
    artifacts["artifact_inventory_key"] = (
        f"campaigns/{authority.RUN_ID}/inventories/artifact-inventory-"
        f"{artifacts['artifact_inventory_sha256']}.json"
    )
    descriptor_seed = authority._rebuild_descriptor(
        descriptor_seed,
        artifacts=artifacts,
    )
    intent_seed = authority._intent(descriptor_seed)
    baseline_seed = authority._baseline(intent_seed)
    acquisition_seed = authority._acquisition(intent_seed, baseline_seed)
    binding_seed = authority._binding(
        descriptor_seed,
        intent_seed,
        baseline_seed,
        acquisition_seed,
    )
    submission_observation_seed = authority._observation(
        descriptor_seed,
        intent_seed,
        status="PENDING",
        start_at=None,
        worker_cluster_name=None,
        recovery_count=0,
        observed_at=authority.SUBMISSION_OBSERVED_AT,
    )
    upstream = {
        "descriptor": descriptor_seed,
        "intent": intent_seed,
        "baseline": baseline_seed,
        "acquisition": acquisition_seed,
        "binding": binding_seed,
        "submission_observation": submission_observation_seed,
        "submission_accepted": authority._submission_accepted(
            intent_seed,
            binding_seed,
            submission_observation_seed,
        ),
    }
    descriptor = upstream["descriptor"]
    intent = upstream["intent"]
    baseline = upstream["baseline"]
    acquisition = upstream["acquisition"]
    binding = upstream["binding"]
    submission_observation = upstream["submission_observation"]
    submission_accepted = upstream["submission_accepted"]
    assert isinstance(descriptor, dict)
    assert isinstance(intent, dict)
    assert isinstance(baseline, dict)
    assert isinstance(acquisition, dict)
    assert isinstance(binding, dict)
    assert isinstance(submission_observation, dict)
    assert isinstance(submission_accepted, dict)
    latch = authority._latch(authority._module(), upstream)
    cluster = worker_cluster or authority.WORKER_CLUSTER
    start = (
        authority.START_AT
        if worker_cluster is not None and start_at is None
        else start_at
    )
    observed_at = controller_observed_at or authority.WORKER_OBSERVED_AT
    history: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_sky_controller_history_v1",
        "sky_job_name": authority.SKY_JOB_NAME,
        "workspace": "default",
        "rows": [
            {
                "sky_job_id": authority.SKY_JOB_ID,
                "sky_job_name": authority.SKY_JOB_NAME,
                "workspace": "default",
                "controller_submitted_at": authority._iso(authority.SUBMITTED_AT),
                "controller_status": controller_status,
                "controller_identity": authority.CONTROLLER_ROLE,
                "schedule_state": "ALIVE",
                "start_at": (None if start is None else authority._iso(start)),
                "worker_cluster_name": worker_cluster,
                "recovery_count": recovery_count,
            }
        ],
        "observed_at": authority._iso(observed_at),
    }
    ledger: list[tuple[str, str, object]] = []
    s3 = FakeS3(ledger)

    def stage(
        key: str,
        value: dict[str, object],
        *,
        newline: bool,
        version_id: str | None = None,
        last_modified: datetime | None = None,
    ) -> str:
        return s3.add(
            key=key,
            raw=_canonical(value, newline=newline),
            run_id=authority.RUN_ID,
            body_sha256=_body_sha(value),
            version_id=version_id,
            last_modified=last_modified,
        )

    descriptor_version = stage(
        str(descriptor["campaign_descriptor_key"]),
        descriptor,
        newline=True,
    )
    del descriptor_version
    stage(authority._intent_key(intent), intent, newline=True)
    stage(authority._acquisition_key(intent), acquisition, newline=True)
    stage(authority._baseline_key(baseline), baseline, newline=True)
    stage(
        authority._submission_accepted_key(submission_accepted),
        submission_accepted,
        newline=True,
    )
    stage(
        str(submission_accepted["job_binding_key"]),
        binding,
        newline=False,
    )
    stage(
        str(submission_accepted["controller_observation_key"]),
        submission_observation,
        newline=False,
    )
    latch_key = authority._module().worker_start_latch_v2_s3_key(worker_latch=latch)
    latch_version = stage(
        latch_key,
        latch,
        newline=False,
        version_id="v1-worker-latch",
        last_modified=authority.ENTRYPOINT_AT,
    )

    controller = _instance(
        authority,
        instance_id=authority.CONTROLLER_INSTANCE_ID,
        instance_type=authority.CONTROLLER_INSTANCE_TYPE,
        image_id="ami-0c0ffee0000000000",
        profile_arn=authority.CONTROLLER_PROFILE,
        state="running",
        launch_time=authority.BASELINE_AT - timedelta(minutes=5),
        cluster=authority.CONTROLLER_CLUSTER,
        controller=True,
    )
    worker = _instance(
        authority,
        instance_id=authority.WORKER_INSTANCE_ID,
        instance_type="p5.48xlarge",
        image_id=authority.AMI_ID,
        profile_arn=(
            "arn:aws:iam::246813579024:instance-profile/keep-glm52-gpu-worker"
        ),
        state="running",
        launch_time=authority.PENDING_AT,
        cluster=cluster,
        controller=False,
    )
    ec2 = FakeEc2(ledger, controller=controller, worker=worker)
    ssm = FakeSsm(
        ledger,
        controller_id=authority.CONTROLLER_INSTANCE_ID,
        history=history,
        history_command=module._history_command(
            job_id=authority.SKY_JOB_ID,
            job_name=authority.SKY_JOB_NAME,
            workspace="default",
            controller_identity=authority.CONTROLLER_ROLE,
        ),
    )
    clock_value = clock_at or observed_at
    clock_calls: list[datetime] = []

    def clock() -> datetime:
        clock_calls.append(clock_value)
        if len(clock_calls) != 1:
            raise AssertionError("coordinator sampled its clock more than once")
        ledger.append(("clock", "now", {}))
        return clock_value

    sleeps: list[float] = []
    services = module.WorkerStartCoordinatorServices(
        sts=FakeSts(ledger),
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        clock=clock,
        sleep=sleeps.append,
    )
    request = module.WorkerStartCoordinatorRequest(
        account_id=ACCOUNT_ID,
        region=REGION,
        bucket=authority.BUCKET,
        descriptor_key=str(descriptor["campaign_descriptor_key"]),
        descriptor_file_sha256=hashlib.sha256(
            _canonical(descriptor, newline=True)
        ).hexdigest(),
        intent_key=authority._intent_key(intent),
        intent_file_sha256=hashlib.sha256(_canonical(intent, newline=True)).hexdigest(),
        intent_body_sha256=str(intent["intent_body_sha256"]),
        worker_latch_key=latch_key,
        worker_latch_version_id=latch_version,
    )
    return CampaignFixture(
        module=module,
        authority=authority,
        upstream=upstream,
        latch=latch,
        request=request,
        services=services,
        s3=s3,
        ec2=ec2,
        ssm=ssm,
        ledger=ledger,
        clock_calls=clock_calls,
        sleeps=sleeps,
    )


def test_flat_import_blocks_forbidden_roots_and_performs_no_io() -> None:
    original_import = builtins.__import__
    forbidden = {"boto3", "botocore", "sky", "mlx", "numpy"}
    attempted: list[str] = []
    flat_names = (
        "glm52_sky_campaign",
        "glm52_sky_must_start",
        "glm52_sky_must_start_dynamic",
        "glm52_sky_worker_must_start_v2",
    )
    saved = {name: sys.modules.pop(name, None) for name in flat_names}

    def guarded_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> Any:
        root = name.split(".", 1)[0]
        if root in forbidden:
            attempted.append(root)
            raise AssertionError(f"forbidden import attempted: {root}")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        _preload_flat_dependencies()
        module = _load_module()
    finally:
        builtins.__import__ = original_import
        for name in flat_names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]

    assert attempted == []
    assert module._PINNED_HISTORY_SHIM_SHA256 == (
        "7f94e25a29e917e3758a878198d97120aaaaf6583a549818167a2b819f21f1c9"
    )
    assert hashlib.sha256(module._PINNED_HISTORY_SHIM.encode()).hexdigest() == (
        module._PINNED_HISTORY_SHIM_SHA256
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", "135792468013"),
        ("region", "us-east-1"),
        ("bucket", ""),
        ("bucket", None),
        ("bucket", "a"),
        ("bucket", "aa..bb"),
        ("descriptor_key", "descriptor.json"),
        (
            "descriptor_key",
            f"campaigns/{RUN_ID}/submissions/production/descriptor.json",
        ),
        ("descriptor_file_sha256", "A" * 64),
        ("descriptor_file_sha256", None),
        (
            "intent_key",
            f"campaigns/{RUN_ID}/submissions/production/intents/{INTENT_SHA}/"
            "SKYPILOT_SUBMISSION_INTENT.json",
        ),
        ("intent_key", None),
        ("intent_body_sha256", "0" * 64),
        ("intent_body_sha256", None),
        ("worker_latch_key", f"campaigns/{RUN_ID}/JOB_BINDING.json"),
        ("worker_latch_key", None),
        ("worker_latch_version_id", ""),
        ("worker_latch_version_id", None),
        ("worker_latch_version_id", True),
    ],
)
def test_request_validation_precedes_every_external_call(
    field: str,
    value: object,
) -> None:
    module = _load_module()
    services = _services(module)
    with pytest.raises(module.WorkerStartCoordinatorError):
        module.coordinate_worker_start_acceptance_v2(
            services=services,
            request=_request(module, **{field: value}),
        )
    assert services.sts.calls == []
    assert services.s3.calls == []
    assert services.ec2.calls == []
    assert services.ssm.calls == []


def test_wrong_sts_account_stops_before_s3_ec2_or_ssm() -> None:
    module = _load_module()

    class Sts:
        def __init__(self) -> None:
            self.calls = 0

        def get_caller_identity(self) -> dict[str, object]:
            self.calls += 1
            return {
                "Account": "135792468013",
                "Arn": (
                    "arn:aws:sts::135792468013:assumed-role/AdministratorAccess/jack"
                ),
                "UserId": "AROATEST:jack",
            }

    sts = Sts()
    services = module.WorkerStartCoordinatorServices(
        sts=sts,
        s3=_NoCalls(),
        ec2=_NoCalls(),
        ssm=_NoCalls(),
        clock=lambda: datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
        sleep=lambda _: None,
    )
    with pytest.raises(module.WorkerStartCoordinatorError):
        module.coordinate_worker_start_acceptance_v2(
            services=services,
            request=_request(module),
        )
    assert sts.calls == 1
    assert services.s3.calls == []
    assert services.ec2.calls == []
    assert services.ssm.calls == []


def test_real_campaign_descriptor_key_passes_request_boundary() -> None:
    module = _load_module()
    services = _services(module)
    validated_services, request, run_id, instance_id = module._validate_request(
        services, _request(module)
    )
    assert validated_services is services
    assert request.descriptor_key == (
        f"campaigns/{RUN_ID}/submissions/qualification/campaign-descriptor-v2.json"
    )
    assert run_id == RUN_ID
    assert instance_id == INSTANCE_ID
    assert services.sts.calls == []


@pytest.mark.parametrize("field", ["clock", "sleep"])
def test_injected_callable_shape_is_validated_before_aws(field: str) -> None:
    module = _load_module()
    services = _services(module)
    invalid = replace(services, **{field: None})
    with pytest.raises(module.WorkerStartCoordinatorError):
        module.coordinate_worker_start_acceptance_v2(
            services=invalid,
            request=_request(module),
        )
    assert services.sts.calls == []
    assert services.s3.calls == []
    assert services.ec2.calls == []
    assert services.ssm.calls == []


def test_initial_worker_is_observed_then_accepted_with_exact_transports() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert outcome.status == "accepted-initial"
    assert outcome.decision_action == "accept-initial"
    assert outcome.instance_id == fixture.authority.WORKER_INSTANCE_ID
    assert outcome.prior_worker_acceptance_count == 0
    assert outcome.published_observation is True
    assert outcome.published_acceptance is True
    assert outcome.worker_controller_observation_key is not None
    assert (
        "/worker-controller-observations/"
        f"{fixture.authority.WORKER_INSTANCE_ID}/"
        in outcome.worker_controller_observation_key
    )
    assert outcome.worker_acceptance_key is not None
    assert outcome.worker_acceptance_body_sha256 is not None
    put_calls = [item for item in fixture.ledger if item[:2] == ("s3", "put_object")]
    assert len(put_calls) == 2
    assert put_calls[0][2]["Key"] == outcome.worker_controller_observation_key
    assert put_calls[1][2]["Key"] == outcome.worker_acceptance_key
    for _, _, kwargs in put_calls:
        assert kwargs["IfNoneMatch"] == "*"
        assert kwargs["ExpectedBucketOwner"] == ACCOUNT_ID
        assert kwargs["ChecksumAlgorithm"] == "SHA256"
        assert kwargs["ContentType"] == "application/json"
        assert kwargs["ChecksumSHA256"] == base64.b64encode(
            hashlib.sha256(bytes(kwargs["Body"])).digest()
        ).decode("ascii")
        assert not bytes(kwargs["Body"]).endswith(b"\n")
    assert len(fixture.clock_calls) == 1
    assert fixture.sleeps == []
    assert all(stream.close_count == 1 for stream in fixture.s3.body_streams)

    operations = [(service, operation) for service, operation, _ in fixture.ledger]
    assert operations[0] == ("sts", "get_caller_identity")
    assert operations.index(("ssm", "send_command")) < operations.index(
        ("s3", "put_object")
    )
    for service, client in (
        ("sts", fixture.services.sts),
        ("s3", fixture.s3),
        ("ec2", fixture.ec2),
        ("ssm", fixture.ssm),
    ):
        assert client.calls == [
            (operation, kwargs)
            for recorded_service, operation, kwargs in fixture.ledger
            if recorded_service == service
        ]


@pytest.mark.parametrize("status", ["PENDING", "STARTING", "RECOVERING"])
def test_valid_unrealized_worker_waits_without_writes(status: str) -> None:
    fixture = _campaign_fixture(
        controller_status=status,
        worker_cluster=None,
        start_at=None,
    )
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert outcome.status == "waiting-worker-authority"
    assert outcome.decision_action == "wait-for-authority"
    assert outcome.worker_controller_observation_key is None
    assert outcome.worker_acceptance_key is None
    assert outcome.worker_acceptance_body_sha256 is None
    assert outcome.published_observation is False
    assert outcome.published_acceptance is False
    assert not [item for item in fixture.ledger if item[:2] == ("s3", "put_object")]


def test_existing_winner_replays_immutable_evidence_without_ssm_ec2_or_put() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    first = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert first.status == "accepted-initial"
    boundary = len(fixture.ledger)
    later = fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
    replay_clock_calls: list[datetime] = []

    def replay_clock() -> datetime:
        replay_clock_calls.append(later)
        assert len(replay_clock_calls) == 1
        return later

    replay_services = fixture.module.WorkerStartCoordinatorServices(
        sts=FakeSts(fixture.ledger),
        s3=fixture.s3,
        ec2=_NoCalls(),
        ssm=_NoCalls(),
        clock=replay_clock,
        sleep=lambda _: pytest.fail("idempotent replay must not sleep"),
    )
    replay = fixture.module.coordinate_worker_start_acceptance_v2(
        services=replay_services,
        request=fixture.request,
    )
    assert replay.status == "idempotent-complete"
    assert replay.decision_action == "idempotent-complete"
    assert replay.worker_controller_observation_key == (
        first.worker_controller_observation_key
    )
    assert replay.worker_acceptance_key == first.worker_acceptance_key
    assert replay.worker_acceptance_body_sha256 == (first.worker_acceptance_body_sha256)
    assert replay.published_observation is False
    assert replay.published_acceptance is False
    assert len(replay_clock_calls) == 1
    replay_operations = fixture.ledger[boundary:]
    assert not [
        item
        for item in replay_operations
        if item[0] in {"ec2", "ssm"} or (item[0] == "s3" and item[1] == "put_object")
    ]


def test_existing_winner_rejects_additive_foreign_campaign_tag() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    first = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    accepted_key = str(first.worker_acceptance_key)
    accepted = json.loads(fixture.s3._entry(accepted_key, None).raw)
    assert isinstance(accepted["worker_campaign_tags"], dict)
    accepted["worker_campaign_tags"]["foreign"] = "must-not-replay"
    accepted = fixture.authority._rehash(
        accepted,
        "worker_acceptance_body_sha256",
    )
    fixture.s3.add(
        key=accepted_key,
        raw=_canonical(accepted, newline=False),
        run_id=fixture.authority.RUN_ID,
        body_sha256=str(accepted["worker_acceptance_body_sha256"]),
        version_id="foreign-campaign-tag",
    )
    services = _existing_replay_services(
        fixture,
        now=fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30),
    )
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=services,
            request=fixture.request,
        )


def test_existing_winner_rejects_foreign_worker_schedule_state() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    first = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    accepted_key = str(first.worker_acceptance_key)
    accepted = json.loads(fixture.s3._entry(accepted_key, None).raw)
    observation_key = str(accepted["worker_controller_observation_key"])
    observation = json.loads(fixture.s3._entry(observation_key, None).raw)
    observation["schedule_state"] = "ALIEN"
    observation = fixture.authority._rehash(
        observation,
        "observation_body_sha256",
    )
    observation_body_sha256 = str(observation["observation_body_sha256"])
    replacement_key = (
        observation_key.rsplit("/", 1)[0] + f"/{observation_body_sha256}.json"
    )
    observation_raw = _canonical(observation, newline=False)
    fixture.s3.add(
        key=replacement_key,
        raw=observation_raw,
        run_id=fixture.authority.RUN_ID,
        body_sha256=observation_body_sha256,
        version_id="foreign-worker-schedule-state",
    )
    accepted["worker_controller_observation_key"] = replacement_key
    accepted["worker_controller_observation_file_sha256"] = hashlib.sha256(
        observation_raw
    ).hexdigest()
    accepted["worker_controller_observation_body_sha256"] = observation_body_sha256
    accepted = fixture.authority._rehash(
        accepted,
        "worker_acceptance_body_sha256",
    )
    fixture.s3.add(
        key=accepted_key,
        raw=_canonical(accepted, newline=False),
        run_id=fixture.authority.RUN_ID,
        body_sha256=str(accepted["worker_acceptance_body_sha256"]),
        version_id="foreign-worker-observation-pointer",
    )
    services = _existing_replay_services(
        fixture,
        now=fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30),
    )
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=services,
            request=fixture.request,
        )


def test_new_candidate_samples_exact_clock_after_controller_observation() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
        clock_at=_load_authority_fixtures().WORKER_OBSERVED_AT + timedelta(seconds=1),
    )
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert outcome.status == "accepted-initial"
    operations = [(service, operation) for service, operation, _ in fixture.ledger]
    assert operations.index(("ssm", "get_command_invocation")) < operations.index(
        ("clock", "now")
    )


def test_real_ec2_owned_state_and_profile_fields_are_accepted() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )

    def add_aws_owned_fields(instance: dict[str, object]) -> None:
        state = instance["State"]
        profile = instance["IamInstanceProfile"]
        assert isinstance(state, dict)
        assert isinstance(profile, dict)
        state["Code"] = 16
        profile["Id"] = "AIPAEXAMPLEOWNEDFIELD"

    add_aws_owned_fields(fixture.ec2.controller)
    add_aws_owned_fields(fixture.ec2.worker)
    for pages in fixture.ec2.filter_pages.values():
        for page in pages:
            reservations = page["Reservations"]
            assert isinstance(reservations, list)
            for reservation in reservations:
                assert isinstance(reservation, dict)
                instances = reservation["Instances"]
                assert isinstance(instances, list)
                for instance in instances:
                    assert isinstance(instance, dict)
                    add_aws_owned_fields(instance)
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert outcome.status == "accepted-initial"


@pytest.mark.parametrize(
    ("put_number", "fault", "expected_flags"),
    [
        (1, FakeClientError("PreconditionFailed"), (False, True)),
        (2, FakeClientError("ConditionalRequestConflict"), (True, False)),
        (2, TimeoutError("response lost after persist"), (True, False)),
    ],
)
def test_marker_last_conflict_or_lost_response_converges_to_exact_winner(
    put_number: int,
    fault: BaseException,
    expected_flags: tuple[bool, bool],
) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    if isinstance(fault, FakeClientError):

        def install_exact_winner(
            store: FakeS3,
            key: str,
            kwargs: dict[str, object],
        ) -> None:
            metadata = dict(kwargs["Metadata"])
            store.add(
                key=key,
                raw=bytes(kwargs["Body"]),
                run_id=str(metadata["glm52-run-id"]),
                body_sha256=str(metadata["glm52-body-sha256"]),
                version_id=f"concurrent-{put_number}",
            )

        fixture.s3.pre_put_hooks[put_number] = install_exact_winner
    else:
        fixture.s3.put_fault_sequence[put_number] = fault
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert outcome.status == "accepted-initial"
    assert (
        outcome.published_observation,
        outcome.published_acceptance,
    ) == expected_flags
    put_calls = [item for item in fixture.ledger if item[:2] == ("s3", "put_object")]
    assert len(put_calls) == 2


def test_lost_response_without_durable_winner_fails_without_blind_retry() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    fixture.s3.pre_put_fault_sequence[2] = TimeoutError(
        "response lost before persistence"
    )
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    put_calls = [item for item in fixture.ledger if item[:2] == ("s3", "put_object")]
    assert len(put_calls) == 2


def test_process_death_after_observation_reuses_it_on_restart() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    fixture.s3.put_fault_sequence[1] = ProcessDeath()
    with pytest.raises(ProcessDeath):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    persisted = [
        key for key in fixture.s3.keys() if "/worker-controller-observations/" in key
    ]
    assert len(persisted) == 1
    assert not [key for key in fixture.s3.keys() if "/worker-acceptances/" in key]
    ledger_boundary = len(fixture.ledger)
    history = fixture.ssm.history
    replay_ec2 = FakeEc2(
        fixture.ledger,
        controller=deepcopy(fixture.ec2.controller),
        worker=deepcopy(fixture.ec2.worker),
    )
    replay_ssm = FakeSsm(
        fixture.ledger,
        controller_id=fixture.authority.CONTROLLER_INSTANCE_ID,
        history=deepcopy(history),
        history_command=fixture.module._history_command(
            job_id=fixture.authority.SKY_JOB_ID,
            job_name=fixture.authority.SKY_JOB_NAME,
            workspace="default",
            controller_identity=fixture.authority.CONTROLLER_ROLE,
        ),
    )
    later = fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
    replay_clock_calls: list[datetime] = []

    def replay_clock() -> datetime:
        replay_clock_calls.append(later)
        fixture.ledger.append(("clock", "now", {}))
        assert len(replay_clock_calls) == 1
        return later

    fixture.s3.put_fault_sequence.clear()
    replay_services = fixture.module.WorkerStartCoordinatorServices(
        sts=FakeSts(fixture.ledger),
        s3=fixture.s3,
        ec2=replay_ec2,
        ssm=replay_ssm,
        clock=replay_clock,
        sleep=lambda _: None,
    )
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=replay_services,
        request=fixture.request,
    )
    assert outcome.status == "accepted-initial"
    assert outcome.published_observation is False
    assert outcome.published_acceptance is True
    replay_puts = [
        item
        for item in fixture.ledger[ledger_boundary:]
        if item[:2] == ("s3", "put_object")
    ]
    assert len(replay_puts) == 1
    assert replay_puts[0][2]["Key"] == outcome.worker_acceptance_key


def test_acceptance_conflict_replays_a_different_valid_observation_winner() -> None:
    authority = _load_authority_fixtures()
    later = authority.WORKER_OBSERVED_AT + timedelta(seconds=1)
    concurrent = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=authority.WORKER_CLUSTER,
        controller_observed_at=later,
        clock_at=later,
    )
    concurrent_outcome = concurrent.module.coordinate_worker_start_acceptance_v2(
        services=concurrent.services,
        request=concurrent.request,
    )
    assert concurrent_outcome.status == "accepted-initial"
    assert concurrent_outcome.worker_controller_observation_key is not None
    assert concurrent_outcome.worker_acceptance_key is not None
    concurrent_observation = concurrent.s3._entry(
        concurrent_outcome.worker_controller_observation_key,
        None,
    )
    concurrent_acceptance = concurrent.s3._entry(
        concurrent_outcome.worker_acceptance_key,
        None,
    )

    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=authority.WORKER_CLUSTER,
        clock_at=later + timedelta(seconds=1),
    )

    def install_concurrent_winner(
        store: FakeS3,
        acceptance_key: str,
        _: dict[str, object],
    ) -> None:
        store.add(
            key=concurrent_outcome.worker_controller_observation_key,
            raw=concurrent_observation.raw,
            run_id=authority.RUN_ID,
            body_sha256=concurrent_observation.metadata["glm52-body-sha256"],
            version_id="race-observation",
            last_modified=concurrent_observation.last_modified,
        )
        store.add(
            key=acceptance_key,
            raw=concurrent_acceptance.raw,
            run_id=authority.RUN_ID,
            body_sha256=concurrent_acceptance.metadata["glm52-body-sha256"],
            version_id="race-acceptance",
            last_modified=concurrent_acceptance.last_modified,
        )

    fixture.s3.pre_put_hooks[2] = install_concurrent_winner
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert outcome.status == "idempotent-complete"
    assert outcome.worker_controller_observation_key == (
        concurrent_outcome.worker_controller_observation_key
    )
    assert outcome.worker_acceptance_key == (concurrent_outcome.worker_acceptance_key)
    assert outcome.worker_acceptance_body_sha256 == (
        concurrent_outcome.worker_acceptance_body_sha256
    )
    assert outcome.published_observation is False
    assert outcome.published_acceptance is False


def test_acceptance_conflict_preserves_successful_same_observation_flag() -> None:
    authority = _load_authority_fixtures()
    concurrent = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=authority.WORKER_CLUSTER,
    )
    concurrent.ec2.worker["State"] = {"Name": "pending"}
    for pages in concurrent.ec2.filter_pages.values():
        for page in pages:
            reservations = page["Reservations"]
            assert isinstance(reservations, list)
            for reservation in reservations:
                assert isinstance(reservation, dict)
                instances = reservation["Instances"]
                assert isinstance(instances, list)
                for instance in instances:
                    assert isinstance(instance, dict)
                    instance["State"] = {"Name": "pending"}
    concurrent_outcome = concurrent.module.coordinate_worker_start_acceptance_v2(
        services=concurrent.services,
        request=concurrent.request,
    )
    assert concurrent_outcome.status == "accepted-initial"
    assert concurrent_outcome.worker_acceptance_key is not None
    concurrent_acceptance = concurrent.s3._entry(
        concurrent_outcome.worker_acceptance_key,
        None,
    )

    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=authority.WORKER_CLUSTER,
    )

    def install_same_observation_winner(
        store: FakeS3,
        acceptance_key: str,
        _: dict[str, object],
    ) -> None:
        store.add(
            key=acceptance_key,
            raw=concurrent_acceptance.raw,
            run_id=authority.RUN_ID,
            body_sha256=concurrent_acceptance.metadata["glm52-body-sha256"],
            version_id="same-observation-race-acceptance",
            last_modified=concurrent_acceptance.last_modified,
        )

    fixture.s3.pre_put_hooks[2] = install_same_observation_winner
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert outcome.status == "idempotent-complete"
    assert outcome.worker_controller_observation_key == (
        concurrent_outcome.worker_controller_observation_key
    )
    assert outcome.worker_acceptance_body_sha256 == (
        concurrent_outcome.worker_acceptance_body_sha256
    )
    assert outcome.published_observation is True
    assert outcome.published_acceptance is False


@pytest.mark.parametrize("source", ["latch", "observation"])
def test_post_publication_exact_source_drift_fails_before_success(
    source: str,
) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )

    def corrupt_exact_source(store: FakeS3, _: str) -> None:
        if source == "latch":
            entry = store._entry(
                fixture.request.worker_latch_key,
                fixture.request.worker_latch_version_id,
            )
        else:
            observation_keys = [
                key for key in store.keys() if "/worker-controller-observations/" in key
            ]
            assert len(observation_keys) == 1
            entry = store._entry(observation_keys[0], None)
        entry.checksum = "corrupted-after-publication"

    fixture.s3.post_put_hooks[2] = corrupt_exact_source
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )


def test_two_managed_recoveries_reconstruct_complete_linear_chain() -> None:
    authority = _load_authority_fixtures()
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=authority.WORKER_CLUSTER,
    )
    initial = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert initial.status == "accepted-initial"

    first_pending = authority.MUST_START_BY + timedelta(minutes=1)
    first_observed = first_pending + timedelta(seconds=3)
    first, first_request = _advance_recovery(
        fixture,
        instance_id=authority.RECOVERY_INSTANCE_ID,
        cluster="sky-glm52-worker-b",
        recovery_count=1,
        pending_at=first_pending,
        observed_at=first_observed,
    )
    assert first.status == "accepted-managed-recovery"
    assert first.prior_worker_acceptance_count == 1
    assert first.worker_latch_key == first_request.worker_latch_key

    second_pending = authority.MUST_START_BY + timedelta(minutes=2)
    second_observed = second_pending + timedelta(seconds=3)
    second, second_request = _advance_recovery(
        fixture,
        instance_id=authority.SECOND_RECOVERY_INSTANCE_ID,
        cluster="sky-glm52-worker-c",
        recovery_count=3,
        pending_at=second_pending,
        observed_at=second_observed,
    )
    assert second.status == "accepted-managed-recovery"
    assert second.prior_worker_acceptance_count == 2
    assert second.worker_latch_key == second_request.worker_latch_key
    acceptance_keys = sorted(
        key for key in fixture.s3.keys() if "/worker-acceptances/" in key
    )
    assert len(acceptance_keys) == 3


@pytest.mark.parametrize(
    "mutation",
    [
        lambda entry: setattr(entry, "etag", '"abcd-2"'),
        lambda entry: setattr(entry, "checksum", "not-a-checksum"),
        lambda entry: setattr(entry, "checksum_type", "COMPOSITE"),
        lambda entry: setattr(entry, "content_type", "text/plain"),
        lambda entry: setattr(
            entry,
            "content_length",
            entry.content_length + 1,
        ),
        lambda entry: entry.metadata.__setitem__("foreign", "value"),
        lambda entry: setattr(entry, "raw", b"{}"),
    ],
)
def test_exact_s3_transport_corruption_fails_closed_and_closes_streams(
    mutation: Any,
) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    descriptor_version = fixture.s3.current[(BUCKET, fixture.request.descriptor_key)]
    entry = fixture.s3._entry(
        fixture.request.descriptor_key,
        descriptor_version,
    )
    mutation(entry)
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert all(stream.close_count == 1 for stream in fixture.s3.body_streams)


@pytest.mark.parametrize("fault_kind", ["read", "close"])
def test_s3_body_read_or_close_failure_is_translated_and_closed_once(
    fault_kind: str,
) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    errors: tuple[Exception | None, Exception | None] = (
        (OSError("read failed"), None)
        if fault_kind == "read"
        else (None, OSError("close failed"))
    )
    fixture.s3.body_faults[fixture.request.descriptor_key] = errors
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert len(fixture.s3.body_streams) == 1
    assert fixture.s3.body_streams[0].close_count == 1


def test_caller_pinned_latch_version_is_never_replaced_by_latest() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    fixture.s3.add(
        key=fixture.request.worker_latch_key,
        raw=b"{}\n",
        run_id=fixture.authority.RUN_ID,
        body_sha256="0" * 64,
        version_id="foreign-latest",
        last_modified=fixture.authority.ENTRYPOINT_AT + timedelta(seconds=1),
    )
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert outcome.status == "accepted-initial"
    latch_gets = [
        kwargs
        for service, operation, kwargs in fixture.ledger
        if service == "s3"
        and operation == "get_object"
        and kwargs["Key"] == fixture.request.worker_latch_key
    ]
    assert latch_gets
    assert {kwargs["VersionId"] for kwargs in latch_gets} == {
        fixture.request.worker_latch_version_id
    }


def test_empty_pinned_version_is_rejected_without_latest_substitution() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    boundary = len(fixture.ledger)
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module._read_artifact(
            fixture.s3,
            bucket=fixture.request.bucket,
            key=fixture.request.worker_latch_key,
            run_id=fixture.authority.RUN_ID,
            newline=False,
            version_id="",
        )
    assert fixture.ledger[boundary:] == []


def test_s3_delete_marker_transport_fails_closed() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    original = fixture.s3.head_object

    def delete_marker_head(**kwargs: object) -> dict[str, object]:
        response = original(**kwargs)
        if kwargs["Key"] == fixture.request.descriptor_key:
            response["DeleteMarker"] = True
        return response

    fixture.s3.head_object = delete_marker_head  # type: ignore[method-assign]
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )


def test_s3_explicit_false_delete_marker_is_allowed() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    original_head = fixture.s3.head_object
    original_get = fixture.s3.get_object

    def false_marker_head(**kwargs: object) -> dict[str, object]:
        response = original_head(**kwargs)
        response["DeleteMarker"] = False
        return response

    def false_marker_get(**kwargs: object) -> dict[str, object]:
        response = original_get(**kwargs)
        response["DeleteMarker"] = False
        return response

    fixture.s3.head_object = false_marker_head  # type: ignore[method-assign]
    fixture.s3.get_object = false_marker_get  # type: ignore[method-assign]
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert outcome.status == "accepted-initial"


def test_current_delete_marker_error_is_not_treated_as_absent() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    original = fixture.s3.head_object

    def current_delete_marker(**kwargs: object) -> dict[str, object]:
        key = str(kwargs["Key"])
        if "VersionId" not in kwargs and (
            "/worker-controller-observations/" in key or "/worker-acceptances/" in key
        ):
            error = FakeClientError("NotFound")
            error.response["ResponseMetadata"]["HTTPHeaders"] = {
                "x-amz-delete-marker": "true"
            }
            raise error
        return original(**kwargs)

    fixture.s3.head_object = current_delete_marker  # type: ignore[method-assign]
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert not [item for item in fixture.ledger if item[:2] == ("s3", "put_object")]


def test_rehashed_wrong_job_binding_fails_before_controller_ssm() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    binding = deepcopy(fixture.upstream["binding"])
    assert isinstance(binding, dict)
    binding["sky_job_id"] = fixture.authority.SKY_JOB_ID + 1
    binding = fixture.authority._rehash(
        binding,
        "job_binding_body_sha256",
    )
    binding_key = str(fixture.upstream["submission_accepted"]["job_binding_key"])
    fixture.s3.add(
        key=binding_key,
        raw=_canonical(binding, newline=False),
        run_id=fixture.authority.RUN_ID,
        body_sha256=str(binding["job_binding_body_sha256"]),
        version_id="rehashed-wrong-binding",
    )

    accepted = deepcopy(fixture.upstream["submission_accepted"])
    assert isinstance(accepted, dict)
    accepted["sky_job_id"] = fixture.authority.SKY_JOB_ID + 1
    accepted["job_binding"] = binding
    accepted["job_binding_file_sha256"] = hashlib.sha256(
        _canonical(binding, newline=False)
    ).hexdigest()
    accepted["job_binding_body_sha256"] = binding["job_binding_body_sha256"]
    accepted = fixture.authority._rehash(
        accepted,
        "accepted_body_sha256",
    )
    old_accepted_key = fixture.authority._submission_accepted_key(
        fixture.upstream["submission_accepted"]
    )
    fixture.s3.current.pop((BUCKET, old_accepted_key))
    fixture.s3.add(
        key=fixture.authority._submission_accepted_key(accepted),
        raw=_canonical(accepted, newline=True),
        run_id=fixture.authority.RUN_ID,
        body_sha256=str(accepted["accepted_body_sha256"]),
        version_id="rehashed-wrong-accepted",
    )

    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert not [
        item
        for item in fixture.ledger
        if item[0] in {"ec2", "ssm"} or (item[0] == "s3" and item[1] == "put_object")
    ]


def test_submission_observation_pointer_cannot_escape_dynamic_prefix() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    accepted = deepcopy(fixture.upstream["submission_accepted"])
    assert isinstance(accepted, dict)
    foreign_key = (
        f"campaigns/{fixture.authority.RUN_ID}/foreign/"
        f"{accepted['controller_observation_body_sha256']}.json"
    )
    accepted["controller_observation_key"] = foreign_key
    accepted = fixture.authority._rehash(
        accepted,
        "accepted_body_sha256",
    )
    old_accepted_key = fixture.authority._submission_accepted_key(
        fixture.upstream["submission_accepted"]
    )
    fixture.s3.current.pop((BUCKET, old_accepted_key))
    fixture.s3.add(
        key=fixture.authority._submission_accepted_key(accepted),
        raw=_canonical(accepted, newline=True),
        run_id=fixture.authority.RUN_ID,
        body_sha256=str(accepted["accepted_body_sha256"]),
        version_id="foreign-pointer-accepted",
    )

    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert not [
        kwargs
        for service, operation, kwargs in fixture.ledger
        if service == "s3"
        and operation in {"head_object", "get_object"}
        and kwargs["Key"] == foreign_key
    ]
    assert not [item for item in fixture.ledger if item[0] in {"ec2", "ssm"}]


def test_submission_accepted_unknown_field_fails_before_controller_ssm() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    accepted = deepcopy(fixture.upstream["submission_accepted"])
    assert isinstance(accepted, dict)
    accepted["foreign"] = "coherently rehashed"
    accepted = fixture.authority._rehash(
        accepted,
        "accepted_body_sha256",
    )
    old_accepted_key = fixture.authority._submission_accepted_key(
        fixture.upstream["submission_accepted"]
    )
    fixture.s3.current.pop((BUCKET, old_accepted_key))
    fixture.s3.add(
        key=fixture.authority._submission_accepted_key(accepted),
        raw=_canonical(accepted, newline=True),
        run_id=fixture.authority.RUN_ID,
        body_sha256=str(accepted["accepted_body_sha256"]),
        version_id="unknown-field-accepted",
    )
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert not [item for item in fixture.ledger if item[0] in {"ec2", "ssm"}]


def test_legacy_authority_fails_before_controller_or_worker_writes() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    key = (
        f"campaigns/{fixture.authority.RUN_ID}/monitor/must-start/"
        "qualification/"
        f"{fixture.request.intent_body_sha256}/JOB_BINDING.json"
    )
    fixture.s3.add(
        key=key,
        raw=b'{"legacy":true}',
        run_id=fixture.authority.RUN_ID,
        body_sha256="0" * 64,
    )
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert not [
        item
        for item in fixture.ledger
        if item[0] in {"ec2", "ssm"} or (item[0] == "s3" and item[1] == "put_object")
    ]


def test_foreign_bucket_cannot_shadow_campaign_authority() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    foreign_bucket = "keep-glm52-foreign-246813579024-us-west-2"
    legacy_key = (
        f"campaigns/{fixture.authority.RUN_ID}/monitor/must-start/"
        "qualification/"
        f"{fixture.request.intent_body_sha256}/JOB_BINDING.json"
    )
    fixture.s3.add(
        bucket=foreign_bucket,
        key=legacy_key,
        raw=b'{"legacy":true}',
        run_id=fixture.authority.RUN_ID,
        body_sha256="0" * 64,
    )
    assert legacy_key in fixture.s3.keys(bucket=foreign_bucket)
    assert legacy_key not in fixture.s3.keys(bucket=fixture.request.bucket)
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert outcome.status == "accepted-initial"


def test_s3_ec2_and_ssm_pagination_complete_before_acceptance() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    fixture.s3.list_page_size = 1
    worker = deepcopy(fixture.ec2.worker)
    fixture.ec2.filter_pages["cluster"] = [
        {
            "Reservations": [],
            "NextToken": "cluster-next",
            "ResponseMetadata": {"HTTPStatusCode": 200},
        },
        fixture.ec2._page([worker]),
    ]
    fixture.ec2.filter_pages["campaign"] = [
        {
            "Reservations": [],
            "NextToken": "campaign-next",
            "ResponseMetadata": {"HTTPStatusCode": 200},
        },
        fixture.ec2._page([worker]),
    ]
    fixture.ssm.info_pages = [
        {
            "InstanceInformationList": [],
            "NextToken": "ssm-next",
            "ResponseMetadata": {"HTTPStatusCode": 200},
        },
        {
            "InstanceInformationList": [
                {
                    "InstanceId": fixture.authority.CONTROLLER_INSTANCE_ID,
                    "PingStatus": "Online",
                }
            ],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        },
    ]
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert outcome.status == "accepted-initial"
    ec2_tokens = [
        kwargs["NextToken"]
        for service, operation, kwargs in fixture.ledger
        if service == "ec2"
        and operation == "describe_instances"
        and "NextToken" in kwargs
    ]
    assert ec2_tokens == ["cluster-next", "campaign-next"]
    ssm_tokens = [
        kwargs["NextToken"]
        for service, operation, kwargs in fixture.ledger
        if service == "ssm"
        and operation == "describe_instance_information"
        and "NextToken" in kwargs
    ]
    assert ssm_tokens == ["ssm-next"]


@pytest.mark.parametrize("service", ["ec2", "ssm"])
def test_opaque_pagination_token_cannot_name_a_missing_page(
    service: str,
) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    if service == "ec2":
        fixture.ec2.filter_pages["cluster"] = [
            {
                "Reservations": [],
                "NextToken": "missing-ec2-page",
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }
        ]
    else:
        fixture.ssm.info_pages = [
            {
                "InstanceInformationList": [],
                "NextToken": "missing-ssm-page",
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }
        ]
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert not [item for item in fixture.ledger if item[:2] == ("s3", "put_object")]


def test_repeated_s3_continuation_token_fails_closed() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    original = fixture.s3.list_objects_v2
    target_prefix = (
        f"campaigns/{fixture.authority.RUN_ID}/submissions/qualification/accepted/"
    )

    def token_loop(**kwargs: object) -> dict[str, object]:
        if kwargs["Prefix"] == target_prefix:
            fixture.s3._record("list_objects_v2", dict(kwargs))
            return {
                "IsTruncated": True,
                "KeyCount": 0,
                "Contents": [],
                "NextContinuationToken": "same-token",
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }
        return original(**kwargs)

    fixture.s3.list_objects_v2 = token_loop  # type: ignore[method-assign]
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )


def test_ssm_polling_and_pinned_command_are_exact() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    success = fixture.ssm.invocations[0]
    fixture.ssm.invocations = [
        FakeClientError("InvocationDoesNotExist"),
        {
            "CommandId": "command-1",
            "InstanceId": fixture.authority.CONTROLLER_INSTANCE_ID,
            "Status": "Pending",
            "ResponseCode": -1,
            "StandardErrorContent": "",
            "StandardOutputContent": "",
            "ResponseMetadata": {"HTTPStatusCode": 200},
        },
        success,
    ]
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert outcome.status == "accepted-initial"
    assert fixture.sleeps == [1, 1]
    send = next(
        kwargs
        for service, operation, kwargs in fixture.ledger
        if service == "ssm" and operation == "send_command"
    )
    assert send == {
        "InstanceIds": [fixture.authority.CONTROLLER_INSTANCE_ID],
        "DocumentName": "AWS-RunShellScript",
        "Comment": "KEEP GLM52 observe exact dynamic-v2 worker authority",
        "Parameters": {"commands": [send["Parameters"]["commands"][0]]},
        "TimeoutSeconds": 45,
    }
    command = send["Parameters"]["commands"][0]
    arguments = shlex.split(command)
    assert arguments[0] == "/home/ubuntu/skypilot-runtime/bin/python"
    assert arguments[1] == "-c"
    assert hashlib.sha256(arguments[2].encode()).hexdigest() == (
        fixture.module._PINNED_HISTORY_SHIM_SHA256
    )
    assert arguments[3:] == [
        str(fixture.authority.SKY_JOB_ID),
        fixture.authority.SKY_JOB_NAME,
        "default",
        fixture.authority.CONTROLLER_ROLE,
    ]


@pytest.mark.parametrize(
    "mutation",
    ["spot", "wrong-profile", "duplicate-tag", "wrong-az"],
)
def test_worker_identity_or_transport_drift_fails_closed(
    mutation: str,
) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    instances = [fixture.ec2.worker]
    for pages in fixture.ec2.filter_pages.values():
        for page in pages:
            reservations = page["Reservations"]
            assert isinstance(reservations, list)
            for reservation in reservations:
                assert isinstance(reservation, dict)
                values = reservation["Instances"]
                assert isinstance(values, list)
                instances.extend(item for item in values if isinstance(item, dict))
    for instance in instances:
        if mutation == "spot":
            instance["InstanceLifecycle"] = "spot"
        elif mutation == "wrong-profile":
            instance["IamInstanceProfile"] = {
                "Arn": ("arn:aws:iam::246813579024:instance-profile/foreign")
            }
        elif mutation == "duplicate-tag":
            tags = instance["Tags"]
            assert isinstance(tags, list)
            tags.append({"Key": "owner", "Value": "jack.mazac"})
        else:
            instance["Placement"] = {"AvailabilityZone": "us-east-1a"}
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert not [item for item in fixture.ledger if item[:2] == ("s3", "put_object")]


def test_two_active_campaign_p5_instances_fail_closed() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    foreign = deepcopy(fixture.ec2.worker)
    foreign["InstanceId"] = "i-1123456789abcdef0"
    fixture.ec2.filter_pages["campaign"] = [
        fixture.ec2._page([fixture.ec2.worker, foreign])
    ]
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ImageId", "ami-0badbadbadbadbad0"),
        (
            "LaunchTime",
            _load_authority_fixtures().PENDING_AT + timedelta(seconds=1),
        ),
    ],
)
def test_campaign_filtered_worker_revalidates_ami_and_launch_time(
    field: str,
    value: object,
) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    page = fixture.ec2.filter_pages["campaign"][0]
    reservations = page["Reservations"]
    assert isinstance(reservations, list)
    instance = reservations[0]["Instances"][0]
    assert isinstance(instance, dict)
    instance[field] = value
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert not [item for item in fixture.ledger if item[:2] == ("s3", "put_object")]


@pytest.mark.parametrize("view", ["cluster", "campaign"])
def test_worker_exact_and_filtered_availability_zones_must_agree(
    view: str,
) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    page = fixture.ec2.filter_pages[view][0]
    reservations = page["Reservations"]
    assert isinstance(reservations, list)
    instance = reservations[0]["Instances"][0]
    assert isinstance(instance, dict)
    instance["Placement"] = {"AvailabilityZone": "us-west-2b"}
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )


def test_controller_rejects_non_null_instance_lifecycle() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    fixture.ec2.controller["InstanceLifecycle"] = "spot"
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert not [item for item in fixture.ledger if item[:2] == ("ssm", "send_command")]


def test_campaign_stopping_view_counts_active_without_replacing_exact_state() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    page = fixture.ec2.filter_pages["campaign"][0]
    reservations = page["Reservations"]
    assert isinstance(reservations, list)
    instance = reservations[0]["Instances"][0]
    assert isinstance(instance, dict)
    instance["State"] = {"Name": "stopping", "Code": 64}
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert outcome.status == "accepted-initial"
    accepted_entry = fixture.s3._entry(
        str(outcome.worker_acceptance_key),
        None,
    )
    accepted = json.loads(accepted_entry.raw)
    assert accepted["worker_instance_state"] == "running"
    assert accepted["active_campaign_p5_instance_ids"] == [
        fixture.authority.WORKER_INSTANCE_ID
    ]


def test_ssm_inventory_rejects_a_hidden_malformed_second_member() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    fixture.ssm.info_pages[0]["InstanceInformationList"].append("malformed")
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert not [item for item in fixture.ledger if item[:2] == ("ssm", "send_command")]


@pytest.mark.parametrize("surface", ["inventory", "invocation"])
def test_ssm_read_responses_require_http_200(surface: str) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    if surface == "inventory":
        fixture.ssm.info_pages[0]["ResponseMetadata"] = {"HTTPStatusCode": 500}
    else:
        fixture.ssm.invocations[0]["ResponseMetadata"] = {"HTTPStatusCode": 500}
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert not [item for item in fixture.ledger if item[:2] == ("s3", "put_object")]


def test_ssm_non_ascii_stdout_is_translated_to_coordinator_error() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    fixture.ssm.invocations[0]["StandardOutputContent"] = (
        'GLM52_SUBMISSION_HISTORY={"foreign":"\u2603"}'
    )
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )


def test_generic_ambiguous_put_error_reconciles_persisted_exact_winner() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    fixture.s3.put_fault_sequence[2] = RuntimeError(
        "generic transport ambiguity after persist"
    )
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=fixture.services,
        request=fixture.request,
    )
    assert outcome.status == "accepted-initial"
    assert outcome.published_observation is True
    assert outcome.published_acceptance is False


@pytest.mark.parametrize(
    "mutation",
    [
        "failed",
        "cancelling",
        "timed-out",
        "nonzero",
        "stderr",
        "wrong-command",
        "wrong-instance",
        "extra-stdout",
        "malformed-json",
    ],
)
def test_ssm_terminal_status_or_output_drift_fails_closed(
    mutation: str,
) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    invocation = fixture.ssm.invocations[0]
    assert isinstance(invocation, dict)
    if mutation in {"failed", "cancelling", "timed-out"}:
        invocation["Status"] = {
            "failed": "Failed",
            "cancelling": "Cancelling",
            "timed-out": "TimedOut",
        }[mutation]
    elif mutation == "nonzero":
        invocation["ResponseCode"] = 1
    elif mutation == "stderr":
        invocation["StandardErrorContent"] = "unexpected stderr"
    elif mutation == "wrong-command":
        invocation["CommandId"] = "command-foreign"
    elif mutation == "wrong-instance":
        invocation["InstanceId"] = "i-1123456789abcdef0"
    elif mutation == "extra-stdout":
        invocation["StandardOutputContent"] = (
            str(invocation["StandardOutputContent"]) + "\nextra"
        )
    else:
        invocation["StandardOutputContent"] = "GLM52_SUBMISSION_HISTORY={malformed"
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert not [item for item in fixture.ledger if item[:2] == ("s3", "put_object")]


def test_ssm_polling_exhaustion_is_bounded_to_ten_attempts() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    pending = {
        "CommandId": "command-1",
        "InstanceId": fixture.authority.CONTROLLER_INSTANCE_ID,
        "Status": "InProgress",
        "ResponseCode": -1,
        "StandardErrorContent": "",
        "StandardOutputContent": "",
        "ResponseMetadata": {"HTTPStatusCode": 200},
    }
    fixture.ssm.invocations = [deepcopy(pending) for _ in range(10)]
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    polls = [
        item for item in fixture.ledger if item[:2] == ("ssm", "get_command_invocation")
    ]
    assert len(polls) == 10
    assert fixture.sleeps == [1] * 10


def test_ssm_pagination_cannot_hide_a_second_controller_identity() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    fixture.ssm.info_pages = [
        {
            "InstanceInformationList": [
                {
                    "InstanceId": fixture.authority.CONTROLLER_INSTANCE_ID,
                    "PingStatus": "Online",
                }
            ],
            "NextToken": "second-controller",
            "ResponseMetadata": {"HTTPStatusCode": 200},
        },
        {
            "InstanceInformationList": [
                {
                    "InstanceId": "i-1123456789abcdef0",
                    "PingStatus": "Online",
                }
            ],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        },
    ]
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert not [item for item in fixture.ledger if item[:2] == ("ssm", "send_command")]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-row",
        "duplicate-row",
        "unknown-status",
        "unknown-schedule",
        "changed-submitted-at",
        "negative-recovery",
        "unsafe-cluster",
    ],
)
def test_controller_history_row_drift_fails_closed(mutation: str) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    history = deepcopy(fixture.ssm.history)
    rows = history["rows"]
    assert isinstance(rows, list)
    row = rows[0]
    assert isinstance(row, dict)
    if mutation == "missing-row":
        history["rows"] = []
    elif mutation == "duplicate-row":
        history["rows"] = [deepcopy(row), deepcopy(row)]
    elif mutation == "unknown-status":
        row["controller_status"] = "FOREIGN"
    elif mutation == "unknown-schedule":
        row["schedule_state"] = "FOREIGN"
    elif mutation == "changed-submitted-at":
        row["controller_submitted_at"] = fixture.authority._iso(
            fixture.authority.SUBMITTED_AT + timedelta(microseconds=1)
        )
    elif mutation == "negative-recovery":
        row["recovery_count"] = -1
    else:
        row["worker_cluster_name"] = "unsafe cluster"
    invocation = fixture.ssm.invocations[0]
    assert isinstance(invocation, dict)
    invocation["StandardOutputContent"] = "GLM52_SUBMISSION_HISTORY=" + json.dumps(
        history, sort_keys=True, separators=(",", ":")
    )
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert not [item for item in fixture.ledger if item[:2] == ("s3", "put_object")]


def test_late_legacy_object_after_acceptance_publication_fails_closed() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )

    def add_late_legacy(store: FakeS3, _: str) -> None:
        key = (
            f"campaigns/{fixture.authority.RUN_ID}/monitor/must-start/"
            f"qualification/{fixture.request.intent_body_sha256}/"
            "TIMELY_START_ACCEPTED.json"
        )
        store.add(
            key=key,
            raw=b'{"legacy":true}',
            run_id=fixture.authority.RUN_ID,
            body_sha256="0" * 64,
            version_id="late-legacy",
        )

    fixture.s3.post_put_hooks[2] = add_late_legacy
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    assert (
        len([item for item in fixture.ledger if item[:2] == ("s3", "put_object")]) == 2
    )


def test_post_publication_extra_chain_root_fails_fresh_relist() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )

    def add_second_root(store: FakeS3, acceptance_key: str) -> None:
        entry = store._entry(acceptance_key, None)
        foreign_key = (
            f"campaigns/{fixture.authority.RUN_ID}/monitor/must-start/"
            f"qualification/{fixture.request.intent_body_sha256}/"
            f"worker-acceptances/i-1123456789abcdef0/{'f' * 64}/"
            "WORKER_START_ACCEPTED.json"
        )
        store.add(
            key=foreign_key,
            raw=entry.raw,
            run_id=fixture.authority.RUN_ID,
            body_sha256=entry.metadata["glm52-body-sha256"],
            version_id="post-publication-second-root",
        )

    fixture.s3.post_put_hooks[2] = add_second_root
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )


@pytest.mark.parametrize("put_number", [1, 2])
def test_malformed_successful_put_response_fails_that_invocation(
    put_number: int,
) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    fixture.s3.put_fault_sequence[put_number] = "bad-response"
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    persisted = [
        key
        for key in fixture.s3.keys()
        if ("/worker-controller-observations/" in key or "/worker-acceptances/" in key)
    ]
    assert len(persisted) == put_number


@pytest.mark.parametrize("put_number", [1, 2])
def test_float_http_200_put_response_is_not_conclusive(
    put_number: int,
) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    fixture.s3.put_fault_sequence[put_number] = "float-status"
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    persisted = [
        key
        for key in fixture.s3.keys()
        if ("/worker-controller-observations/" in key or "/worker-acceptances/" in key)
    ]
    assert len(persisted) == put_number


@pytest.mark.parametrize("put_number", [1, 2])
def test_definitive_access_denied_does_not_reconcile_concurrent_winner(
    put_number: int,
) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )

    def install_exact_winner(
        store: FakeS3,
        key: str,
        kwargs: dict[str, object],
    ) -> None:
        metadata = dict(kwargs["Metadata"])
        store.add(
            bucket=str(kwargs["Bucket"]),
            key=key,
            raw=bytes(kwargs["Body"]),
            run_id=str(metadata["glm52-run-id"]),
            body_sha256=str(metadata["glm52-body-sha256"]),
            version_id=f"access-denied-winner-{put_number}",
        )

    fixture.s3.pre_put_hooks[put_number] = install_exact_winner
    fixture.s3.pre_put_fault_sequence[put_number] = FakeClientError("AccessDenied")
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    put_index = max(
        index
        for index, item in enumerate(fixture.ledger)
        if item[:2] == ("s3", "put_object")
    )
    assert not [
        item
        for item in fixture.ledger[put_index + 1 :]
        if item[:2]
        in {
            ("s3", "head_object"),
            ("s3", "get_object"),
        }
    ]


def test_later_invocation_authenticates_acceptance_from_bad_put_response() -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )
    fixture.s3.put_fault_sequence[2] = "bad-response"
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )
    fixture.s3.put_fault_sequence.clear()
    later = fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
    calls: list[datetime] = []

    def replay_clock() -> datetime:
        calls.append(later)
        assert len(calls) == 1
        return later

    services = fixture.module.WorkerStartCoordinatorServices(
        sts=FakeSts(fixture.ledger),
        s3=fixture.s3,
        ec2=_NoCalls(),
        ssm=_NoCalls(),
        clock=replay_clock,
        sleep=lambda _: pytest.fail("replay must not sleep"),
    )
    outcome = fixture.module.coordinate_worker_start_acceptance_v2(
        services=services,
        request=fixture.request,
    )
    assert outcome.status == "idempotent-complete"
    assert outcome.published_observation is False
    assert outcome.published_acceptance is False


@pytest.mark.parametrize("put_number", [1, 2])
def test_successful_put_with_corrupt_exact_reread_fails(
    put_number: int,
) -> None:
    fixture = _campaign_fixture(
        controller_status="STARTING",
        worker_cluster=_load_authority_fixtures().WORKER_CLUSTER,
    )

    def corrupt_reread(store: FakeS3, key: str) -> None:
        store._entry(key, None).checksum = "corrupt-immediate-reread"

    fixture.s3.post_put_hooks[put_number] = corrupt_reread
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=fixture.services,
            request=fixture.request,
        )


@pytest.mark.parametrize(
    "mutation",
    ["fork", "cycle", "reorder", "truncate"],
)
def test_recovery_chain_fork_cycle_reorder_or_truncation_fails(
    mutation: str,
) -> None:
    fixture, current_request = _build_three_worker_chain()
    acceptances = _stored_worker_acceptances(fixture)
    assert len(acceptances) == 3
    by_recovery = {
        int(value["recovery_count"]): (key, value) for key, value in acceptances.items()
    }
    root_key, root = by_recovery[0]
    first_key, first = by_recovery[1]
    tip_key, _ = by_recovery[3]
    if mutation == "fork":
        tip_entry = fixture.s3._entry(tip_key, None)
        fork_key = (
            f"campaigns/{fixture.authority.RUN_ID}/monitor/must-start/"
            f"qualification/{current_request.intent_body_sha256}/"
            f"worker-acceptances/i-2123456789abcdef0/{'e' * 64}/"
            "WORKER_START_ACCEPTED.json"
        )
        fixture.s3.add(
            key=fork_key,
            raw=tip_entry.raw,
            run_id=fixture.authority.RUN_ID,
            body_sha256=tip_entry.metadata["glm52-body-sha256"],
            version_id="forked-successor",
        )
    elif mutation == "cycle":
        first_entry = fixture.s3._entry(first_key, None)
        cyclic_root = deepcopy(root)
        cyclic_root["prior_worker_acceptance_key"] = first_key
        cyclic_root["prior_worker_acceptance_file_sha256"] = hashlib.sha256(
            first_entry.raw
        ).hexdigest()
        cyclic_root["prior_worker_acceptance_body_sha256"] = first_entry.metadata[
            "glm52-body-sha256"
        ]
        cyclic_root = fixture.authority._rehash(
            cyclic_root,
            "worker_acceptance_body_sha256",
        )
        fixture.s3.add(
            key=root_key,
            raw=_canonical(cyclic_root, newline=False),
            run_id=fixture.authority.RUN_ID,
            body_sha256=str(cyclic_root["worker_acceptance_body_sha256"]),
            version_id="cyclic-root",
        )
    elif mutation == "truncate":
        fixture.s3.current.pop((BUCKET, first_key))
    else:
        original = fixture.s3.list_objects_v2
        acceptance_prefix = (
            f"campaigns/{fixture.authority.RUN_ID}/monitor/must-start/"
            f"qualification/{current_request.intent_body_sha256}/"
            "worker-acceptances/"
        )

        def reordered(**kwargs: object) -> dict[str, object]:
            response = original(**kwargs)
            if kwargs["Prefix"] == acceptance_prefix:
                contents = response["Contents"]
                assert isinstance(contents, list)
                response["Contents"] = list(reversed(contents))
            return response

        fixture.s3.list_objects_v2 = reordered  # type: ignore[method-assign]
    services = _existing_replay_services(
        fixture,
        now=fixture.authority.MUST_START_BY + timedelta(minutes=5),
    )
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=services,
            request=current_request,
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing-latch-version", "wrong-etag", "wrong-last-modified", "observation"],
)
def test_prior_recovery_source_transport_drift_fails(
    mutation: str,
) -> None:
    fixture, current_request = _build_three_worker_chain()
    acceptances = _stored_worker_acceptances(fixture)
    first = next(
        value for value in acceptances.values() if value["recovery_count"] == 1
    )
    if mutation == "observation":
        observation_key = str(first["worker_controller_observation_key"])
        fixture.s3._entry(observation_key, None).checksum = "replaced-prior-observation"
    else:
        latch_key = str(first["worker_latch_key"])
        latch_version = str(first["worker_latch_version_id"])
        if mutation == "missing-latch-version":
            fixture.s3.versions.pop((BUCKET, latch_key, latch_version))
        elif mutation == "wrong-etag":
            fixture.s3._entry(
                latch_key, latch_version
            ).etag = '"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"'
        else:
            fixture.s3._entry(
                latch_key,
                latch_version,
            ).last_modified += timedelta(seconds=1)
    services = _existing_replay_services(
        fixture,
        now=fixture.authority.MUST_START_BY + timedelta(minutes=5),
    )
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=services,
            request=current_request,
        )


@pytest.mark.parametrize(
    "mutation",
    ["additive-campaign-tag", "foreign-schedule-state"],
)
def test_prior_worker_authority_uses_exact_replay_validators(
    mutation: str,
) -> None:
    fixture, current_request = _build_three_worker_chain()
    by_recovery = {
        int(value["recovery_count"]): (key, deepcopy(value))
        for key, value in _stored_worker_acceptances(fixture).items()
    }
    root_key, root = by_recovery[0]
    if mutation == "additive-campaign-tag":
        tags = root["worker_campaign_tags"]
        assert isinstance(tags, dict)
        tags["foreign"] = "must-not-replay"
    else:
        observation_key = str(root["worker_controller_observation_key"])
        observation = json.loads(fixture.s3._entry(observation_key, None).raw)
        observation["schedule_state"] = "ALIEN"
        observation = fixture.authority._rehash(
            observation,
            "observation_body_sha256",
        )
        observation_body_sha256 = str(observation["observation_body_sha256"])
        replacement_key = (
            observation_key.rsplit("/", 1)[0] + f"/{observation_body_sha256}.json"
        )
        observation_raw = _canonical(observation, newline=False)
        fixture.s3.add(
            key=replacement_key,
            raw=observation_raw,
            run_id=fixture.authority.RUN_ID,
            body_sha256=observation_body_sha256,
            version_id="prior-foreign-schedule-state",
        )
        root["worker_controller_observation_key"] = replacement_key
        root["worker_controller_observation_file_sha256"] = hashlib.sha256(
            observation_raw
        ).hexdigest()
        root["worker_controller_observation_body_sha256"] = observation_body_sha256
    rewritten: tuple[str, bytes, str] | None = None
    for recovery_count in sorted(by_recovery):
        key, value = by_recovery[recovery_count]
        if recovery_count == 0:
            value = root
            assert key == root_key
        elif rewritten is not None:
            prior_key, prior_raw, prior_body_sha256 = rewritten
            value["prior_worker_acceptance_key"] = prior_key
            value["prior_worker_acceptance_file_sha256"] = hashlib.sha256(
                prior_raw
            ).hexdigest()
            value["prior_worker_acceptance_body_sha256"] = prior_body_sha256
        value = fixture.authority._rehash(
            value,
            "worker_acceptance_body_sha256",
        )
        raw = _canonical(value, newline=False)
        body_sha256 = str(value["worker_acceptance_body_sha256"])
        fixture.s3.add(
            key=key,
            raw=raw,
            run_id=fixture.authority.RUN_ID,
            body_sha256=body_sha256,
            version_id=f"strict-prior-{mutation}-{recovery_count}",
        )
        rewritten = (key, raw, body_sha256)
    services = _existing_replay_services(
        fixture,
        now=fixture.authority.MUST_START_BY + timedelta(minutes=5),
    )
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=services,
            request=current_request,
        )


def test_prior_source_pointer_is_rejected_before_foreign_s3_read() -> None:
    fixture, current_request = _build_three_worker_chain()
    acceptances = _stored_worker_acceptances(fixture)
    by_recovery = {
        int(value["recovery_count"]): (key, deepcopy(value))
        for key, value in acceptances.items()
    }
    root_key, root = by_recovery[0]
    first_key, first = by_recovery[1]
    tip_key, tip = by_recovery[3]
    foreign_key = (
        f"campaigns/{fixture.authority.RUN_ID}/foreign/worker-observation.json"
    )
    root["worker_controller_observation_key"] = foreign_key
    root = fixture.authority._rehash(
        root,
        "worker_acceptance_body_sha256",
    )
    root_raw = _canonical(root, newline=False)
    first["prior_worker_acceptance_file_sha256"] = hashlib.sha256(root_raw).hexdigest()
    first["prior_worker_acceptance_body_sha256"] = root["worker_acceptance_body_sha256"]
    first = fixture.authority._rehash(
        first,
        "worker_acceptance_body_sha256",
    )
    first_raw = _canonical(first, newline=False)
    tip["prior_worker_acceptance_file_sha256"] = hashlib.sha256(first_raw).hexdigest()
    tip["prior_worker_acceptance_body_sha256"] = first["worker_acceptance_body_sha256"]
    tip = fixture.authority._rehash(
        tip,
        "worker_acceptance_body_sha256",
    )
    for key, value, version in (
        (root_key, root, "foreign-pointer-root"),
        (first_key, first, "foreign-pointer-first"),
        (tip_key, tip, "foreign-pointer-tip"),
    ):
        fixture.s3.add(
            key=key,
            raw=_canonical(value, newline=False),
            run_id=fixture.authority.RUN_ID,
            body_sha256=str(value["worker_acceptance_body_sha256"]),
            version_id=version,
        )
    services = _existing_replay_services(
        fixture,
        now=fixture.authority.MUST_START_BY + timedelta(minutes=5),
    )
    with pytest.raises(fixture.module.WorkerStartCoordinatorError):
        fixture.module.coordinate_worker_start_acceptance_v2(
            services=services,
            request=current_request,
        )
    assert not [
        kwargs
        for service, operation, kwargs in fixture.ledger
        if service == "s3"
        and operation in {"head_object", "get_object"}
        and kwargs["Key"] == foreign_key
    ]
