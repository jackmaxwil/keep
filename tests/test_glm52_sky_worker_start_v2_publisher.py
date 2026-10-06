"""Qualification worker-v2 publisher transport tests."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from copy import deepcopy
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.request import Request

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "aws/glm52-gpu/skypilot/publish_worker_start_v2.py"
COORDINATOR_TEST_PATH = ROOT / "tests/test_glm52_sky_worker_start_v2_coordinator.py"
AWS_EXECUTABLE = Path(
    "/usr/local/aws-cli/v2/2.15.1/dist/aws",
)
SOURCE_PATHS = {
    "campaign": ROOT / "src/mlx_vq/quality/glm52_sky_campaign.py",
    "campaign_native": ROOT / "src/glm52_enforcement/glm52_sky_campaign.py",
    "must_start": ROOT / "src/mlx_vq/quality/glm52_sky_must_start.py",
    "must_start_native": ROOT / "src/glm52_enforcement/glm52_sky_must_start.py",
    "dynamic": ROOT / "src/mlx_vq/quality/glm52_sky_must_start_dynamic.py",
    "worker": ROOT / "src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py",
    "coordinator": (ROOT / "aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py"),
}
FROZEN_HASHES = {
    "campaign": ("77412115f6b8ce325177864418d838e0335d536329e8e7565d7811bcc806cd94"),
    "campaign_native": ("f023eaeddf8fac73fa6546e3c4a0b60dd32e5266f06e1519fe21e0444d445d03"),
    "must_start": ("478ac3a16ce8a77a4e844c1e0a3dca99454d50e7233fd7cd67dfa184b486860a"),
    "must_start_native": ("e910d3d03f7b30b7b9b668e214b61ecc7cbd641a53dedabd620b8e14573d33f3"),
    "dynamic": ("527019a41b54f810e9f98734c2ca99acc5a7b370344adf48c80f46f925396e18"),
    "worker": ("8dfad6682d8afc9774dc31385cfc9ecc8c8e2f80d9a535e3c8d4cf6cf0fa97df"),
    "coordinator": ("efa7c48ad259e10c6ce0e4cb1db9ff0a6e6a9fa203274329a7c2eb5203366406"),
}
FROZEN_MODULE_NAMES = (
    "glm52_sky_campaign_native",
    "glm52_sky_must_start_native",
    "glm52_sky_campaign",
    "glm52_sky_must_start",
    "glm52_sky_must_start_dynamic",
    "glm52_sky_worker_must_start_v2",
    "sky_worker_start_v2_coordinator",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object, *, newline: bool) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return raw + (b"\n" if newline else b"")


def _load_module() -> Any:
    name = "_task_3pa_worker_start_v2_publisher"
    sys.modules.pop(name, None)
    for frozen_name in FROZEN_MODULE_NAMES:
        sys.modules.pop(frozen_name, None)
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_coordinator_fixtures() -> Any:
    name = "_task_3pa_coordinator_fixtures"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, COORDINATOR_TEST_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source_bundle(module: Any, **changes: object) -> Any:
    values: dict[str, object] = {
        "publisher_file_sha256": _sha(MODULE_PATH.read_bytes()),
        "campaign_policy_path": SOURCE_PATHS["campaign"],
        "campaign_policy_file_sha256": FROZEN_HASHES["campaign"],
        "campaign_policy_native_path": SOURCE_PATHS["campaign_native"],
        "campaign_policy_native_file_sha256": FROZEN_HASHES["campaign_native"],
        "must_start_policy_path": SOURCE_PATHS["must_start"],
        "must_start_policy_file_sha256": FROZEN_HASHES["must_start"],
        "must_start_policy_native_path": SOURCE_PATHS["must_start_native"],
        "must_start_policy_native_file_sha256": FROZEN_HASHES[
            "must_start_native"
        ],
        "dynamic_policy_path": SOURCE_PATHS["dynamic"],
        "dynamic_policy_file_sha256": FROZEN_HASHES["dynamic"],
        "worker_policy_path": SOURCE_PATHS["worker"],
        "worker_policy_file_sha256": FROZEN_HASHES["worker"],
        "coordinator_path": SOURCE_PATHS["coordinator"],
        "coordinator_file_sha256": FROZEN_HASHES["coordinator"],
    }
    values.update(changes)
    return module.WorkerStartSourceBundle(**values)


class _Imds:
    def __init__(self, identity_document: bytes) -> None:
        self.identity_document = identity_document
        self.calls: list[tuple[Request, float]] = []

    def __call__(self, request: Request, timeout: float) -> bytes:
        self.calls.append((request, timeout))
        if request.full_url == "http://169.254.169.254/latest/api/token":
            assert request.get_method() == "PUT"
            assert request.headers == {"X-aws-ec2-metadata-token-ttl-seconds": "60"}
            return b"imds-token"
        assert request.full_url == (
            "http://169.254.169.254/latest/dynamic/instance-identity/document"
        )
        assert request.get_method() == "GET"
        assert request.headers == {"X-aws-ec2-metadata-token": "imds-token"}
        return self.identity_document


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


class _CliRunner:
    """Translate exact AWS CLI argv into the existing real fake-S3 service."""

    def __init__(self, module: Any, s3: Any, worker_role: str) -> None:
        self.module = module
        self.s3 = s3
        self.worker_role = worker_role
        self.calls: list[tuple[tuple[str, ...], dict[str, str], int]] = []
        self.put_last_modified = datetime(
            2026,
            7,
            26,
            12,
            0,
            10,
            tzinfo=UTC,
        )

    @staticmethod
    def _value(argv: tuple[str, ...], name: str) -> str:
        index = argv.index(name)
        return argv[index + 1]

    def _result(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> Any:
        return self.module.WorkerStartCommandResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    def _transport_json(value: dict[str, object]) -> dict[str, object]:
        result = dict(value)
        result.pop("ResponseMetadata", None)
        modified = result.get("LastModified")
        if isinstance(modified, datetime):
            result["LastModified"] = (
                modified.astimezone(UTC).isoformat().replace("+00:00", "Z")
            )
        return result

    def _client_error(self, operation: str, error: BaseException) -> Any:
        code = getattr(error, "response", {}).get("Error", {}).get("Code")
        assert isinstance(code, str)
        return self._result(
            returncode=255,
            stderr=(
                f"An error occurred ({code}) when calling the "
                f"{operation} operation: injected\n"
            ).encode("ascii"),
        )

    def __call__(
        self,
        argv: tuple[str, ...],
        environ: dict[str, str],
        timeout: int,
    ) -> Any:
        self.calls.append((argv, dict(environ), timeout))
        assert argv[0] == str(AWS_EXECUTABLE)
        assert "--no-sign-request" not in argv
        assert "--region" in argv and self._value(argv, "--region") == "us-west-2"
        assert "--no-cli-pager" in argv
        assert "--output" in argv and self._value(argv, "--output") == "json"
        assert self._value(argv, "--cli-connect-timeout") == "3"
        assert self._value(argv, "--cli-read-timeout") == "10"
        assert timeout == 30
        assert environ == {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/root",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "AWS_REGION": "us-west-2",
            "AWS_DEFAULT_REGION": "us-west-2",
            "AWS_PAGER": "",
            "AWS_EC2_METADATA_DISABLED": "false",
            "AWS_METADATA_SERVICE_NUM_ATTEMPTS": "1",
            "AWS_CONFIG_FILE": "/dev/null",
            "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
            "BOTO_CONFIG": "/dev/null",
            "AWS_MAX_ATTEMPTS": "1",
        }
        if argv[1:3] == ("sts", "get-caller-identity"):
            return self._result(
                stdout=_json_bytes(
                    {
                        "Account": "246813579024",
                        "Arn": (
                            "arn:aws:sts::246813579024:assumed-role/"
                            f"{self.worker_role.rsplit('/', 1)[-1]}/worker"
                        ),
                        "UserId": "AROATEST:worker",
                    }
                )
            )
        assert argv[1] == "s3api"
        operation = argv[2]
        bucket = self._value(argv, "--bucket")
        assert self._value(argv, "--expected-bucket-owner") == "246813579024"
        if operation == "put-object":
            key = self._value(argv, "--key")
            assert "--if-none-match" in argv
            assert self._value(argv, "--if-none-match") == "*"
            assert self._value(argv, "--checksum-algorithm") == "SHA256"
            raw = Path(self._value(argv, "--body")).read_bytes()
            checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
            assert self._value(argv, "--checksum-sha256") == checksum
            if (bucket, key) in self.s3.current:
                return self._client_error(
                    "PutObject",
                    self._coordinator_error("PreconditionFailed"),
                )
            metadata = dict(
                item.split("=", 1)
                for item in self._value(argv, "--metadata").split(",")
            )
            version = self.s3.add(
                bucket=bucket,
                key=key,
                raw=raw,
                run_id=metadata["glm52-run-id"],
                body_sha256=metadata["glm52-body-sha256"],
                version_id="cli-put-1",
                last_modified=self.put_last_modified,
            )
            stored = self.s3._entry(key, version, bucket=bucket)
            return self._result(
                stdout=_json_bytes(
                    {
                        "VersionId": version,
                        "ETag": stored.etag,
                        "ChecksumSHA256": stored.checksum,
                    }
                )
            )
        kwargs: dict[str, object] = {
            "Bucket": bucket,
            "ChecksumMode": "ENABLED",
            "ExpectedBucketOwner": "246813579024",
        }
        if operation != "list-objects-v2":
            kwargs["Key"] = self._value(argv, "--key")
        if "--version-id" in argv:
            kwargs["VersionId"] = self._value(argv, "--version-id")
        try:
            if operation == "head-object":
                value = self.s3.head_object(**kwargs)
                return self._result(stdout=_json_bytes(self._transport_json(value)))
            if operation == "get-object":
                response = self.s3.get_object(**kwargs)
                stream = response.pop("Body")
                raw = stream.read()
                stream.close()
                Path(argv[-1]).write_bytes(raw)
                return self._result(stdout=_json_bytes(self._transport_json(response)))
            if operation == "list-objects-v2":
                assert "--no-paginate" in argv
                list_kwargs: dict[str, object] = {
                    "Bucket": bucket,
                    "Prefix": self._value(argv, "--prefix"),
                    "ExpectedBucketOwner": "246813579024",
                }
                if "--continuation-token" in argv:
                    list_kwargs["ContinuationToken"] = self._value(
                        argv,
                        "--continuation-token",
                    )
                response = self.s3.list_objects_v2(**list_kwargs)
                return self._result(stdout=_json_bytes(self._transport_json(response)))
        except BaseException as error:
            if hasattr(error, "response"):
                return self._client_error(
                    {
                        "head-object": "HeadObject",
                        "get-object": "GetObject",
                        "list-objects-v2": "ListObjectsV2",
                    }[operation],
                    error,
                )
            raise
        raise AssertionError(f"unexpected AWS operation: {operation}")

    def _coordinator_error(self, code: str) -> BaseException:
        fixtures = _load_coordinator_fixtures()
        return fixtures.FakeClientError(code)


class _State:
    def __init__(self, module: Any, tmp_path: Path) -> None:
        coord = _load_coordinator_fixtures()
        coord._preload_flat_dependencies()
        authority_fixtures = coord._load_authority_fixtures()
        self.fixture = coord._campaign_fixture(
            worker_cluster=authority_fixtures.WORKER_CLUSTER,
        )
        self.module = module
        self.authority_module = self.fixture.authority._module()
        authority = self.fixture.authority
        identity_value = {
            "accountId": authority.ACCOUNT_ID,
            "architecture": "x86_64",
            "availabilityZone": "us-west-2a",
            "imageId": authority.AMI_ID,
            "instanceId": authority.WORKER_INSTANCE_ID,
            "instanceType": "p5.48xlarge",
            "pendingTime": authority._iso(authority.PENDING_AT),
            "privateIp": "10.0.1.9",
            "region": authority.REGION,
            "version": "2017-09-30",
        }
        self.identity_raw = _canonical(identity_value, newline=False)
        new_latch = self.authority_module.build_worker_start_latch_v2(
            descriptor=self.fixture.upstream["descriptor"],
            intent=self.fixture.upstream["intent"],
            controller_injected_sky_job_id=str(authority.SKY_JOB_ID),
            instance_id=authority.WORKER_INSTANCE_ID,
            instance_type="p5.48xlarge",
            image_id=authority.AMI_ID,
            worker_role_arn=authority.WORKER_ROLE,
            instance_identity_document_sha256=_sha(self.identity_raw),
            ec2_pending_time=authority.PENDING_AT,
            entrypoint_observed_at=authority.ENTRYPOINT_AT,
        )
        old_key = str(self.fixture.request.worker_latch_key)
        old_version = str(self.fixture.request.worker_latch_version_id)
        self.fixture.s3.current.pop((authority.BUCKET, old_key))
        self.fixture.s3.versions.pop((authority.BUCKET, old_key, old_version))
        new_key = self.authority_module.worker_start_latch_v2_s3_key(
            worker_latch=new_latch
        )
        new_version = self.fixture.s3.add(
            key=new_key,
            raw=_canonical(new_latch, newline=False),
            run_id=authority.RUN_ID,
            body_sha256=str(new_latch["worker_latch_body_sha256"]),
            version_id="v1-worker-latch",
            last_modified=authority.ENTRYPOINT_AT,
        )
        self.fixture.latch = new_latch
        self.fixture.request = replace(
            self.fixture.request,
            worker_latch_key=new_key,
            worker_latch_version_id=new_version,
        )
        self.descriptor_path = tmp_path / "descriptor.json"
        self.intent_path = tmp_path / "intent.json"
        self.latch_path = tmp_path / "worker-latch.json"
        self.transport_path = tmp_path / "worker-latch-transport.json"
        self.accepted_path = tmp_path / "worker-accepted.json"
        self.receipt_path = tmp_path / "worker-admission-receipt.json"
        descriptor_raw = _canonical(
            self.fixture.upstream["descriptor"],
            newline=True,
        )
        intent_raw = _canonical(self.fixture.upstream["intent"], newline=True)
        self.descriptor_path.write_bytes(descriptor_raw)
        self.intent_path.write_bytes(intent_raw)
        descriptor = self.fixture.upstream["descriptor"]
        intent = self.fixture.upstream["intent"]
        self.sources = _source_bundle(module)
        self.authority = module.WorkerStartAuthorityInputs(
            descriptor_path=self.descriptor_path,
            descriptor_s3_uri=(
                f"s3://{authority.BUCKET}/{descriptor['campaign_descriptor_key']}"
            ),
            descriptor_file_sha256=_sha(descriptor_raw),
            intent_path=self.intent_path,
            intent_s3_uri=(f"s3://{authority.BUCKET}/{authority._intent_key(intent)}"),
            intent_file_sha256=_sha(intent_raw),
            intent_body_sha256=str(intent["intent_body_sha256"]),
        )
        self.imds = _Imds(self.identity_raw)
        self.runner = _CliRunner(
            module,
            self.fixture.s3,
            str(descriptor["worker_identity"]),
        )
        self.services = module.WorkerStartPublisherServices(
            command_runner=self.runner,
            imds_opener=self.imds,
            utc_now=lambda: authority.ENTRYPOINT_AT,
            monotonic=lambda: 1.0,
            sleep=lambda _: pytest.fail("unexpected publisher sleep"),
            environ={"SKYPILOT_MANAGED_JOB_ID": str(authority.SKY_JOB_ID)},
        )

    def publish(self) -> Any:
        return self.module.publish_worker_start_latch_v2(
            sources=self.sources,
            authority=self.authority,
            outputs=self.module.WorkerStartLatchOutputs(
                latch_path=self.latch_path,
                latch_transport_path=self.transport_path,
            ),
            services=self.services,
        )

    def stage_acceptance(self) -> Any:
        return self.fixture.module.coordinate_worker_start_acceptance_v2(
            services=self.fixture.services,
            request=self.fixture.request,
        )

    def advance_recovery(
        self,
        *,
        instance_id: str,
        cluster: str,
        recovery_count: int,
        pending_at: datetime,
        observed_at: datetime,
    ) -> Any:
        coord = _load_coordinator_fixtures()
        authority = self.fixture.authority
        identity = json.loads(self.identity_raw)
        identity["instanceId"] = instance_id
        identity["pendingTime"] = authority._iso(pending_at)
        self.identity_raw = _canonical(identity, newline=False)
        self.imds.identity_document = self.identity_raw
        for path in (self.latch_path, self.transport_path):
            path.unlink(missing_ok=True)
        entrypoint_at = pending_at + timedelta(seconds=1)
        self.runner.put_last_modified = entrypoint_at
        self.services = replace(
            self.services,
            utc_now=lambda: entrypoint_at,
            monotonic=lambda: 1.0,
            sleep=lambda _: pytest.fail("unexpected publisher sleep"),
        )
        published = self.publish()
        assert published.status == "latch-published"
        latch = json.loads(self.latch_path.read_bytes())
        transport = json.loads(self.transport_path.read_bytes())
        request = replace(
            self.fixture.request,
            worker_latch_key=transport["worker_latch_key"],
            worker_latch_version_id=transport["worker_latch_version_id"],
        )
        worker = coord._instance(
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
        ec2 = coord.FakeEc2(
            self.fixture.ledger,
            controller=deepcopy(self.fixture.ec2.controller),
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
        ssm = coord.FakeSsm(
            self.fixture.ledger,
            controller_id=authority.CONTROLLER_INSTANCE_ID,
            history=history,
            history_command=self.fixture.module._history_command(
                job_id=authority.SKY_JOB_ID,
                job_name=authority.SKY_JOB_NAME,
                workspace="default",
                controller_identity=authority.CONTROLLER_ROLE,
            ),
        )
        clock_calls: list[datetime] = []

        def clock() -> datetime:
            clock_calls.append(observed_at)
            self.fixture.ledger.append(("clock", "now", {}))
            assert len(clock_calls) == 1
            return observed_at

        coordinator_services = self.fixture.module.WorkerStartCoordinatorServices(
            sts=coord.FakeSts(self.fixture.ledger),
            s3=self.fixture.s3,
            ec2=ec2,
            ssm=ssm,
            clock=clock,
            sleep=lambda _: pytest.fail("unexpected coordinator sleep"),
        )
        outcome = self.fixture.module.coordinate_worker_start_acceptance_v2(
            services=coordinator_services,
            request=request,
        )
        self.fixture.latch = latch
        self.fixture.request = request
        self.fixture.services = coordinator_services
        self.fixture.ec2 = ec2
        self.fixture.ssm = ssm
        self.services = replace(
            self.services,
            utc_now=lambda: observed_at + timedelta(seconds=30),
            monotonic=lambda: 10.0,
        )
        return outcome


@pytest.fixture
def module() -> Any:
    return _load_module()


@pytest.fixture
def state(module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _State:
    monkeypatch.setattr(
        module,
        "_attest_aws_cli",
        lambda: module._AwsCliAttestation(
            executable=AWS_EXECUTABLE,
            entrypoint_identity=None,
            target_identity=None,
        ),
    )
    monkeypatch.setattr(module, "_revalidate_aws_cli", lambda _: None)
    module._validate_source_bundle(_source_bundle(module))
    return _State(module, tmp_path)


def _materialize_initial_receipt(state: _State) -> None:
    state.publish()
    outcome = state.stage_acceptance()
    assert outcome.status == "accepted-initial"
    state.services = replace(
        state.services,
        utc_now=lambda: (
            state.fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
        ),
        monotonic=lambda: 10.0,
    )
    replayed = state.module.replay_worker_start_accepted_v2(
        sources=state.sources,
        authority=state.authority,
        latch_path=state.latch_path,
        latch_transport_path=state.transport_path,
        outputs=state.module.WorkerStartAdmissionOutputs(
            accepted_path=state.accepted_path,
            admission_receipt_path=state.receipt_path,
        ),
        services=state.services,
    )
    assert replayed.status == "accepted-initial"


def _stage_two_recovery_chain(state: _State) -> Any:
    state.publish()
    initial = state.stage_acceptance()
    assert initial.status == "accepted-initial"
    authority = state.fixture.authority
    first_pending = authority.MUST_START_BY + timedelta(minutes=1)
    first = state.advance_recovery(
        instance_id=authority.RECOVERY_INSTANCE_ID,
        cluster="sky-glm52-worker-b",
        recovery_count=1,
        pending_at=first_pending,
        observed_at=first_pending + timedelta(seconds=3),
    )
    assert first.status == "accepted-managed-recovery"
    second_pending = authority.MUST_START_BY + timedelta(minutes=2)
    second = state.advance_recovery(
        instance_id=authority.SECOND_RECOVERY_INSTANCE_ID,
        cluster="sky-glm52-worker-c",
        recovery_count=3,
        pending_at=second_pending,
        observed_at=second_pending + timedelta(seconds=3),
    )
    assert second.status == "accepted-managed-recovery"
    return second


def test_public_module_boundary_exists() -> None:
    """Removing the separate v2 entrypoint must fail this boundary test."""

    assert MODULE_PATH.is_file()


def test_public_contract_is_exact(module: Any) -> None:
    assert [field.name for field in fields(module.WorkerStartCommandResult)] == [
        "returncode",
        "stdout",
        "stderr",
    ]
    assert [field.name for field in fields(module.WorkerStartSourceBundle)] == [
        "publisher_file_sha256",
        "campaign_policy_path",
        "campaign_policy_file_sha256",
        "campaign_policy_native_path",
        "campaign_policy_native_file_sha256",
        "must_start_policy_path",
        "must_start_policy_file_sha256",
        "must_start_policy_native_path",
        "must_start_policy_native_file_sha256",
        "dynamic_policy_path",
        "dynamic_policy_file_sha256",
        "worker_policy_path",
        "worker_policy_file_sha256",
        "coordinator_path",
        "coordinator_file_sha256",
    ]
    assert [field.name for field in fields(module.WorkerStartAuthorityInputs)] == [
        "descriptor_path",
        "descriptor_s3_uri",
        "descriptor_file_sha256",
        "intent_path",
        "intent_s3_uri",
        "intent_file_sha256",
        "intent_body_sha256",
    ]
    assert [field.name for field in fields(module.WorkerStartPublisherOutcome)] == [
        "status",
        "run_id",
        "instance_id",
        "sky_job_id",
        "worker_latch_key",
        "worker_acceptance_key",
        "worker_acceptance_body_sha256",
    ]


@pytest.mark.parametrize(
    "uri",
    [
        "S3://bucket/key",
        "s3://bucket/key?",
        "s3://bucket/key#",
        "s3://bucket/key\n",
        "s3://bucket:invalid/key",
    ],
)
def test_s3_uri_parser_requires_exact_canonical_raw_form(
    module: Any,
    uri: str,
) -> None:
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="S3 URI is invalid",
    ):
        module._parse_s3_uri(uri, label="test S3 URI")


def test_source_hash_drift_fails_before_imds_or_command(
    state: _State,
) -> None:
    state.sources = replace(
        state.sources,
        worker_policy_file_sha256="0" * 64,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="source",
    ):
        state.publish()
    assert state.imds.calls == []
    assert state.runner.calls == []


@pytest.mark.parametrize(
    "digest",
    [
        "A" * 64,
        "g" * 64,
        "0" * 63,
        "0" * 65,
    ],
)
def test_noncanonical_source_digest_fails_before_imds_or_command(
    state: _State,
    digest: str,
) -> None:
    state.sources = replace(
        state.sources,
        worker_policy_file_sha256=digest,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="file SHA-256 is not a lowercase SHA-256",
    ):
        state.publish()
    assert state.imds.calls == []
    assert state.runner.calls == []


@pytest.mark.parametrize(
    "path_field",
    [
        "campaign_policy_path",
        "must_start_policy_path",
        "dynamic_policy_path",
        "worker_policy_path",
        "coordinator_path",
    ],
)
def test_each_missing_frozen_source_fails_before_imds_or_command(
    state: _State,
    tmp_path: Path,
    path_field: str,
) -> None:
    state.sources = replace(
        state.sources,
        **{path_field: tmp_path / f"missing-{path_field}.py"},
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="source",
    ):
        state.publish()
    assert state.imds.calls == []
    assert state.runner.calls == []


@pytest.mark.parametrize(
    "mutation",
    ["relative", "symlink", "policy-swap"],
)
def test_foreign_source_path_shapes_fail_before_imds_or_command(
    state: _State,
    tmp_path: Path,
    mutation: str,
) -> None:
    if mutation == "relative":
        campaign_path = Path("src/mlx_vq/quality/glm52_sky_campaign.py")
    elif mutation == "symlink":
        campaign_path = tmp_path / "campaign-policy.py"
        campaign_path.symlink_to(SOURCE_PATHS["campaign"])
    else:
        campaign_path = SOURCE_PATHS["must_start"]
    state.sources = replace(
        state.sources,
        campaign_policy_path=campaign_path,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="source",
    ):
        state.publish()
    assert state.imds.calls == []
    assert state.runner.calls == []


@pytest.mark.parametrize(
    ("environ", "case"),
    [
        ({"SKYPILOT_JOB_ID": "17"}, "alternate-only"),
        ({"SKYPILOT_MANAGED_JOB_ID": "9" * 5000}, "huge"),
    ],
)
def test_alternate_only_or_huge_job_id_fails_before_imds_or_command(
    state: _State,
    environ: dict[str, str],
    case: str,
) -> None:
    del case
    state.services = replace(state.services, environ=environ)
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="job ID",
    ):
        state.publish()
    assert state.imds.calls == []
    assert state.runner.calls == []


def test_literal_python39_frozen_import_and_valid_latch_roundtrip(
    state: _State,
) -> None:
    descriptor = base64.b64encode(
        _canonical(state.fixture.upstream["descriptor"], newline=False)
    ).decode("ascii")
    intent = base64.b64encode(
        _canonical(state.fixture.upstream["intent"], newline=False)
    ).decode("ascii")
    authority = state.fixture.authority
    script = f"""
import base64
import datetime as dt
import hashlib
import importlib.util
import json
import pathlib
import sys

assert sys.version_info[:2] == (3, 9)
assert not hasattr(dt, "UTC")

root = pathlib.Path({str(ROOT)!r})
module_path = root / "aws/glm52-gpu/skypilot/publish_worker_start_v2.py"
spec = importlib.util.spec_from_file_location("_task3pa_python39", module_path)
assert spec is not None and spec.loader is not None
publisher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = publisher
spec.loader.exec_module(publisher)

sources = publisher.WorkerStartSourceBundle(
    publisher_file_sha256={_sha(MODULE_PATH.read_bytes())!r},
    campaign_policy_path=root / "src/mlx_vq/quality/glm52_sky_campaign.py",
    campaign_policy_file_sha256={FROZEN_HASHES["campaign"]!r},
    campaign_policy_native_path=root / "src/glm52_enforcement/glm52_sky_campaign.py",
    campaign_policy_native_file_sha256={FROZEN_HASHES["campaign_native"]!r},
    must_start_policy_path=root / "src/mlx_vq/quality/glm52_sky_must_start.py",
    must_start_policy_file_sha256={FROZEN_HASHES["must_start"]!r},
    must_start_policy_native_path=root / "src/glm52_enforcement/glm52_sky_must_start.py",
    must_start_policy_native_file_sha256={FROZEN_HASHES["must_start_native"]!r},
    dynamic_policy_path=root / "src/mlx_vq/quality/glm52_sky_must_start_dynamic.py",
    dynamic_policy_file_sha256={FROZEN_HASHES["dynamic"]!r},
    worker_policy_path=root / "src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py",
    worker_policy_file_sha256={FROZEN_HASHES["worker"]!r},
    coordinator_path=root / "aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py",
    coordinator_file_sha256={FROZEN_HASHES["coordinator"]!r},
)
source_identities = (
    (module_path, sources.publisher_file_sha256),
    (sources.campaign_policy_path, sources.campaign_policy_file_sha256),
    (sources.campaign_policy_native_path, sources.campaign_policy_native_file_sha256),
    (sources.must_start_policy_path, sources.must_start_policy_file_sha256),
    (sources.must_start_policy_native_path, sources.must_start_policy_native_file_sha256),
    (sources.dynamic_policy_path, sources.dynamic_policy_file_sha256),
    (sources.worker_policy_path, sources.worker_policy_file_sha256),
    (sources.coordinator_path, sources.coordinator_file_sha256),
)
before = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in source_identities)
assert before == tuple(expected for _, expected in source_identities)
loaded = publisher._validate_source_bundle(sources)
assert dt.UTC == dt.timezone.utc
after = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in source_identities)
assert after == before
descriptor = json.loads(base64.b64decode({descriptor!r}))
intent = json.loads(base64.b64decode({intent!r}))
identity_raw = {state.identity_raw!r}
latch = loaded.worker.build_worker_start_latch_v2(
    descriptor=descriptor,
    intent=intent,
    controller_injected_sky_job_id={authority.SKY_JOB_ID},
    instance_id={authority.WORKER_INSTANCE_ID!r},
    instance_type="p5.48xlarge",
    image_id={authority.AMI_ID!r},
    worker_role_arn={authority.WORKER_ROLE!r},
    instance_identity_document_sha256=hashlib.sha256(identity_raw).hexdigest(),
    ec2_pending_time={authority._iso(authority.PENDING_AT)!r},
    entrypoint_observed_at={authority._iso(authority.ENTRYPOINT_AT)!r},
)
validated = loaded.worker.validate_worker_start_latch_v2(
    latch,
    descriptor=descriptor,
    intent=intent,
)
raw = loaded.worker.worker_v2_canonical_bytes(validated)
key = loaded.worker.worker_start_latch_v2_s3_key(worker_latch=validated)
assert not raw.endswith(b"\\n")
assert key.endswith("/" + validated["worker_latch_body_sha256"] + ".json")
print("python3.9-roundtrip-ok")
"""
    completed = subprocess.run(
        ["/usr/bin/python3", "-"],
        input=script.encode("ascii"),
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8",
        errors="replace",
    )
    assert completed.stdout == b"python3.9-roundtrip-ok\n"
    assert completed.stderr == b""


def test_preloaded_frozen_module_runtime_monkeypatch_is_rejected(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _source_bundle(module)
    module._validate_source_bundle(sources)
    worker = sys.modules["glm52_sky_worker_must_start_v2"]
    monkeypatch.setattr(
        worker,
        "build_worker_start_latch_v2",
        lambda **_: {},
    )
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="runtime identity is not trusted",
    ):
        module._validate_source_bundle(sources)


@pytest.mark.parametrize(
    "job_id",
    [None, "", "0", "-1", "+1", "01", "1.0", "1e3", " 1", "1 ", "x"],
)
def test_invalid_managed_job_id_fails_before_imds_or_command(
    state: _State,
    job_id: str | None,
) -> None:
    environ = {} if job_id is None else {"SKYPILOT_MANAGED_JOB_ID": job_id}
    state.services = replace(state.services, environ=environ)
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="job ID",
    ):
        state.publish()
    assert state.imds.calls == []
    assert state.runner.calls == []


@pytest.mark.parametrize(
    "name",
    [
        "AWS_ACCESS_KEY_ID",
        "AWS_PROFILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_EC2_METADATA_SERVICE_ENDPOINT",
        "AWS_CA_BUNDLE",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "BOTO_CONFIG",
    ],
)
def test_credential_or_endpoint_override_fails_before_imds(
    state: _State,
    name: str,
) -> None:
    state.services = replace(
        state.services,
        environ={
            **state.services.environ,
            name: "forbidden",
            "HTTPS_PROXY": "http://also-not-forwarded.invalid",
        },
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="environment",
    ):
        state.publish()
    assert state.imds.calls == []
    assert state.runner.calls == []


def test_conditional_conflict_reconciles_exact_latch_without_retry(
    state: _State,
) -> None:
    outcome = state.publish()
    assert outcome.status == "latch-idempotent"
    assert outcome.worker_latch_key == state.fixture.request.worker_latch_key
    assert state.latch_path.read_bytes() == _canonical(
        state.fixture.latch,
        newline=False,
    )
    assert stat.S_IMODE(state.latch_path.stat().st_mode) == 0o600
    transport = json.loads(state.transport_path.read_bytes())
    assert state.transport_path.read_bytes() == _canonical(
        transport,
        newline=True,
    )
    assert stat.S_IMODE(state.transport_path.stat().st_mode) == 0o600
    assert transport["record_type"] == ("glm52_sky_worker_start_latch_transport_v1")
    assert transport["worker_instance_type"] == "p5.48xlarge"
    assert transport["worker_image_id"] == state.fixture.authority.AMI_ID
    assert transport["worker_role_arn"] == state.fixture.authority.WORKER_ROLE
    assert transport["instance_identity_document_sha256"] == _sha(state.identity_raw)
    assert transport["worker_latch_version_id"] == "v1-worker-latch"
    put_calls = [
        call for call in state.runner.calls if call[0][1:3] == ("s3api", "put-object")
    ]
    assert len(put_calls) == 1
    assert all("HTTPS_PROXY" not in call[1] for call in state.runner.calls)


def test_fresh_conditional_put_is_verified_and_marker_last(
    state: _State,
) -> None:
    bucket = state.fixture.authority.BUCKET
    key = str(state.fixture.request.worker_latch_key)
    version = str(state.fixture.request.worker_latch_version_id)
    state.fixture.s3.current.pop((bucket, key))
    state.fixture.s3.versions.pop((bucket, key, version))
    outcome = state.publish()
    assert outcome.status == "latch-published"
    assert state.latch_path.is_file()
    assert state.transport_path.is_file()
    s3_calls = [call[0] for call in state.runner.calls if call[0][1] == "s3api"]
    put_index = next(
        index for index, argv in enumerate(s3_calls) if argv[2] == "put-object"
    )
    assert [argv[2] for argv in s3_calls[put_index:]] == [
        "put-object",
        "get-object",
        "head-object",
    ]
    common = (
        "--region",
        "us-west-2",
        "--cli-connect-timeout",
        "3",
        "--cli-read-timeout",
        "10",
        "--no-cli-pager",
        "--output",
        "json",
    )
    expected_read_base = (
        str(AWS_EXECUTABLE),
        "s3api",
        "get-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--expected-bucket-owner",
        "246813579024",
        "--version-id",
        "cli-put-1",
        "--checksum-mode",
        "ENABLED",
    )
    get_argv = s3_calls[put_index + 1]
    assert get_argv[:-1] == (*expected_read_base, *common)
    assert Path(get_argv[-1]).name.startswith("glm52-s3-get-")
    head_argv = s3_calls[put_index + 2]
    assert head_argv == (
        *expected_read_base[:2],
        "head-object",
        *expected_read_base[3:],
        *common,
    )


@pytest.mark.parametrize("publication", ["fresh", "conflict"])
@pytest.mark.parametrize(
    ("lost_operation", "expected_error"),
    [
        ("get-object", "GetObject transport result is ambiguous"),
        ("head-object", "HeadObject transport result is ambiguous"),
    ],
)
def test_lost_winner_get_or_versioned_head_never_becomes_success(
    state: _State,
    publication: str,
    lost_operation: str,
    expected_error: str,
) -> None:
    bucket = state.fixture.authority.BUCKET
    key = str(state.fixture.request.worker_latch_key)
    version = str(state.fixture.request.worker_latch_version_id)
    if publication == "fresh":
        state.fixture.s3.current.pop((bucket, key))
        state.fixture.s3.versions.pop((bucket, key, version))
    original = state.runner
    response_lost = False

    def lose_read_response(
        argv: tuple[str, ...],
        environ: dict[str, str],
        timeout: int,
    ) -> Any:
        nonlocal response_lost
        result = original(argv, environ, timeout)
        if (
            argv[1:3] == ("s3api", lost_operation)
            and "--version-id" in argv
            and not response_lost
        ):
            response_lost = True
            raise TimeoutError(f"simulated lost {lost_operation} response")
        return result

    state.services = replace(
        state.services,
        command_runner=lose_read_response,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match=expected_error,
    ):
        state.publish()
    assert response_lost is True
    puts = [call for call in original.calls if call[0][1:3] == ("s3api", "put-object")]
    assert len(puts) == 1
    assert not state.latch_path.exists()
    assert not state.transport_path.exists()


def test_wrong_imds_instance_fails_before_s3(
    state: _State,
) -> None:
    identity = json.loads(state.identity_raw)
    identity["imageId"] = "ami-11111111111111111"
    state.imds.identity_document = _canonical(identity, newline=False)
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="IMDS",
    ):
        state.publish()
    assert [call for call in state.runner.calls if call[0][1] == "s3api"] == []


def test_existing_foreign_local_output_is_never_replaced(
    state: _State,
) -> None:
    state.latch_path.write_bytes(b"foreign")
    os.chmod(state.latch_path, 0o600)
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="existing",
    ):
        state.publish()
    assert state.latch_path.read_bytes() == b"foreign"
    assert not state.transport_path.exists()


def test_full_initial_acceptance_replay_and_receipt_reverification(
    state: _State,
) -> None:
    state.publish()
    first = state.stage_acceptance()
    assert first.status == "accepted-initial"
    later = state.fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
    state.services = replace(
        state.services,
        utc_now=lambda: later,
        monotonic=lambda: 10.0,
    )
    outcome = state.module.replay_worker_start_accepted_v2(
        sources=state.sources,
        authority=state.authority,
        latch_path=state.latch_path,
        latch_transport_path=state.transport_path,
        outputs=state.module.WorkerStartAdmissionOutputs(
            accepted_path=state.accepted_path,
            admission_receipt_path=state.receipt_path,
        ),
        services=state.services,
    )
    assert outcome.status == "accepted-initial"
    assert outcome.worker_acceptance_key == first.worker_acceptance_key
    accepted = json.loads(state.accepted_path.read_bytes())
    receipt = json.loads(state.receipt_path.read_bytes())
    assert state.accepted_path.read_bytes() == _canonical(
        accepted,
        newline=False,
    )
    assert state.receipt_path.read_bytes() == _canonical(
        receipt,
        newline=True,
    )
    assert receipt["record_type"] == ("glm52_sky_worker_admission_receipt_v1")
    assert set(receipt) == {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "bucket",
        "run_id",
        "intent_body_sha256",
        "sky_job_id",
        "instance_id",
        "worker_instance_type",
        "worker_image_id",
        "worker_role_arn",
        "ec2_pending_time",
        "instance_identity_document_sha256",
        "worker_latch_key",
        "worker_latch_file_sha256",
        "worker_latch_body_sha256",
        "worker_latch_version_id",
        "worker_latch_etag",
        "worker_latch_last_modified",
        "worker_acceptance_key",
        "worker_acceptance_file_sha256",
        "worker_acceptance_body_sha256",
        "worker_acceptance_version_id",
        "worker_acceptance_etag",
        "worker_acceptance_last_modified",
        "source_transports",
        "legacy_forbidden_basenames",
        "admission_receipt_body_sha256",
    }
    receipt_body = dict(receipt)
    receipt_digest = receipt_body.pop("admission_receipt_body_sha256")
    assert receipt_digest == _sha(_canonical(receipt_body, newline=False))
    assert receipt["instance_id"] == state.fixture.authority.WORKER_INSTANCE_ID
    assert receipt["worker_instance_type"] == "p5.48xlarge"
    assert receipt["worker_image_id"] == state.fixture.authority.AMI_ID
    assert receipt["worker_role_arn"] == state.fixture.authority.WORKER_ROLE
    assert receipt["instance_identity_document_sha256"] == _sha(state.identity_raw)
    source_fields = {
        "key",
        "file_sha256",
        "body_sha256",
        "version_id",
        "etag",
        "last_modified",
        "content_length",
        "checksum_sha256",
        "checksum_type",
        "content_type",
        "metadata",
        "trailing_newline",
    }
    assert all(set(source) == source_fields for source in receipt["source_transports"])
    keys = [item["key"] for item in receipt["source_transports"]]
    assert keys == sorted(set(keys))
    expected_keys = {
        state.fixture.upstream["descriptor"]["campaign_descriptor_key"],
        state.fixture.authority._intent_key(state.fixture.upstream["intent"]),
        accepted["controller_baseline_key"],
        accepted["submission_acquisition_key"],
        accepted["submission_accepted_key"],
        accepted["job_binding_key"],
        accepted["worker_controller_observation_key"],
        accepted["worker_latch_key"],
        str(first.worker_acceptance_key),
    }
    submission_observation_key = next(
        key for key in state.fixture.s3.keys() if "/observations/" in key
    )
    expected_keys.add(submission_observation_key)
    assert set(keys) == expected_keys
    assert len(keys) == 10
    assert str(first.worker_acceptance_key) in keys
    assert receipt["legacy_forbidden_basenames"] == [
        "JOB_BINDING.json",
        "TIMELY_START_ACCEPTED.json",
        "TIMELY_START_LATCH.json",
    ]
    before_verify = len(state.runner.calls)
    verify = state.module.verify_worker_start_receipt_v2(
        sources=state.sources,
        authority=state.authority,
        latch_path=state.latch_path,
        latch_transport_path=state.transport_path,
        accepted_path=state.accepted_path,
        admission_receipt_path=state.receipt_path,
        services=state.services,
    )
    assert verify.status == "receipt-verified"
    assert verify.worker_acceptance_key == first.worker_acceptance_key
    verify_s3_calls = [
        call[0] for call in state.runner.calls[before_verify:] if call[0][1] == "s3api"
    ]
    assert not any(argv[2] == "put-object" for argv in verify_s3_calls)
    for source in receipt["source_transports"]:
        source_key = source["key"]
        source_version = source["version_id"]
        gets = [
            argv
            for argv in verify_s3_calls
            if argv[2] == "get-object"
            and state.runner._value(argv, "--key") == source_key
            and state.runner._value(argv, "--version-id") == source_version
        ]
        versioned_heads = [
            argv
            for argv in verify_s3_calls
            if argv[2] == "head-object"
            and "--version-id" in argv
            and state.runner._value(argv, "--key") == source_key
            and state.runner._value(argv, "--version-id") == source_version
        ]
        assert len(gets) >= 2
        assert len(versioned_heads) >= 2
        for argv in (*gets, *versioned_heads):
            assert state.runner._value(argv, "--checksum-mode") == "ENABLED"
            assert (
                state.runner._value(argv, "--expected-bucket-owner") == "246813579024"
            )


@pytest.mark.parametrize("recovery_generations", [1, 2])
def test_full_managed_recovery_replay_authenticates_complete_ancestry(
    state: _State,
    recovery_generations: int,
) -> None:
    state.publish()
    initial = state.stage_acceptance()
    assert initial.status == "accepted-initial"
    authority = state.fixture.authority
    recoveries = [
        (
            authority.RECOVERY_INSTANCE_ID,
            "sky-glm52-worker-b",
            1,
            authority.MUST_START_BY + timedelta(minutes=1),
        ),
        (
            authority.SECOND_RECOVERY_INSTANCE_ID,
            "sky-glm52-worker-c",
            3,
            authority.MUST_START_BY + timedelta(minutes=2),
        ),
    ]
    latest = None
    for instance_id, cluster, recovery_count, pending_at in recoveries[
        :recovery_generations
    ]:
        latest = state.advance_recovery(
            instance_id=instance_id,
            cluster=cluster,
            recovery_count=recovery_count,
            pending_at=pending_at,
            observed_at=pending_at + timedelta(seconds=3),
        )
        assert latest.status == "accepted-managed-recovery"
    assert latest is not None

    outcome = state.module.replay_worker_start_accepted_v2(
        sources=state.sources,
        authority=state.authority,
        latch_path=state.latch_path,
        latch_transport_path=state.transport_path,
        outputs=state.module.WorkerStartAdmissionOutputs(
            accepted_path=state.accepted_path,
            admission_receipt_path=state.receipt_path,
        ),
        services=state.services,
    )
    assert outcome.status == "accepted-managed-recovery"
    assert outcome.instance_id == recoveries[recovery_generations - 1][0]
    assert outcome.worker_acceptance_key == latest.worker_acceptance_key
    receipt = json.loads(state.receipt_path.read_bytes())
    traced_acceptances = {
        item["key"]
        for item in receipt["source_transports"]
        if "/worker-acceptances/" in item["key"]
    }
    stored_acceptances = {
        key for key in state.fixture.s3.keys() if "/worker-acceptances/" in key
    }
    assert traced_acceptances == stored_acceptances
    assert len(traced_acceptances) == recovery_generations + 1

    verified = state.module.verify_worker_start_receipt_v2(
        sources=state.sources,
        authority=state.authority,
        latch_path=state.latch_path,
        latch_transport_path=state.transport_path,
        accepted_path=state.accepted_path,
        admission_receipt_path=state.receipt_path,
        services=state.services,
    )
    assert verified.status == "receipt-verified"
    assert verified.worker_acceptance_key == latest.worker_acceptance_key


@pytest.mark.parametrize(
    ("mutation", "expected_cause"),
    [
        ("missing", "worker acceptance chain has multiple roots"),
        (
            "extra",
            "worker acceptance winner is not the exact fresh chain tip",
        ),
        ("fork", "worker acceptance chain forks"),
        ("cycle", "worker acceptance chain has multiple roots"),
        (
            "disconnected",
            "worker acceptance chain has multiple roots",
        ),
        (
            "repeated-instance",
            "worker acceptance chain repeats an instance",
        ),
        (
            "reordered",
            "worker acceptance recovery count does not increase",
        ),
        (
            "transport-drift",
            "prior latch transport identity drifted",
        ),
    ],
)
def test_corrupt_managed_recovery_ancestry_fails_before_local_markers(
    state: _State,
    tmp_path: Path,
    mutation: str,
    expected_cause: str,
) -> None:
    _stage_two_recovery_chain(state)
    authority = state.fixture.authority
    bucket = authority.BUCKET
    acceptances: dict[str, dict[str, object]] = {}
    for key in state.fixture.s3.keys():
        if "/worker-acceptances/" in key:
            acceptances[key] = json.loads(state.fixture.s3._entry(key, None).raw)
    by_recovery = {
        int(value["recovery_count"]): (key, value) for key, value in acceptances.items()
    }
    root_key, root = by_recovery[0]
    first_key, first = by_recovery[1]
    tip_key, tip = by_recovery[3]
    for key, accepted in acceptances.items():
        assert (
            state.authority_module.worker_start_accepted_v2_s3_key(
                worker_start_accepted=accepted
            )
            == key
        )

    def copy_unique_branch_objects(branch: _State) -> None:
        for key in branch.fixture.s3.keys():
            if (bucket, key) in state.fixture.s3.current:
                continue
            entry = branch.fixture.s3._entry(key, None)
            state.fixture.s3.add(
                bucket=bucket,
                key=key,
                raw=entry.raw,
                run_id=entry.metadata["glm52-run-id"],
                body_sha256=entry.metadata["glm52-body-sha256"],
                version_id=entry.version_id,
                last_modified=entry.last_modified,
                checksum_type=entry.checksum_type,
                content_type=entry.content_type,
            )

    def branch_state(name: str) -> _State:
        branch_path = tmp_path / name
        branch_path.mkdir()
        return _State(state.module, branch_path)

    def replace_acceptance(
        *,
        key: str,
        value: dict[str, object],
        version_id: str,
        last_modified: datetime,
    ) -> None:
        assert (
            state.authority_module.worker_start_accepted_v2_s3_key(
                worker_start_accepted=value
            )
            == key
        )
        state.fixture.s3.add(
            bucket=bucket,
            key=key,
            raw=_canonical(value, newline=False),
            run_id=authority.RUN_ID,
            body_sha256=str(value["worker_acceptance_body_sha256"]),
            version_id=version_id,
            last_modified=last_modified,
        )

    if mutation == "missing":
        state.fixture.s3.current.pop((bucket, root_key))
    elif mutation == "extra":
        branch = branch_state("extra-successor")
        _stage_two_recovery_chain(branch)
        third_pending = authority.MUST_START_BY + timedelta(minutes=3)
        extra = branch.advance_recovery(
            instance_id="i-3123456789abcdef0",
            cluster="sky-glm52-worker-d",
            recovery_count=4,
            pending_at=third_pending,
            observed_at=third_pending + timedelta(seconds=3),
        )
        assert extra.status == "accepted-managed-recovery"
        copy_unique_branch_objects(branch)
        extra_entry = branch.fixture.s3._entry(
            str(extra.worker_acceptance_key),
            None,
        )
        extra_value = json.loads(extra_entry.raw)
        assert (
            state.authority_module.worker_start_accepted_v2_s3_key(
                worker_start_accepted=extra_value
            )
            == extra.worker_acceptance_key
        )
    elif mutation == "fork":
        fork_keys: list[str] = []
        for branch_name, instance_id, cluster in (
            (
                "forked-successor-a",
                "i-3123456789abcdef0",
                "sky-glm52-worker-d",
            ),
            (
                "forked-successor-b",
                "i-4123456789abcdef0",
                "sky-glm52-worker-e",
            ),
        ):
            branch = branch_state(branch_name)
            branch.publish()
            branch.stage_acceptance()
            first_pending = authority.MUST_START_BY + timedelta(minutes=1)
            branch.advance_recovery(
                instance_id=authority.RECOVERY_INSTANCE_ID,
                cluster="sky-glm52-worker-b",
                recovery_count=1,
                pending_at=first_pending,
                observed_at=first_pending + timedelta(seconds=3),
            )
            second_pending = authority.MUST_START_BY + timedelta(minutes=2)
            fork = branch.advance_recovery(
                instance_id=instance_id,
                cluster=cluster,
                recovery_count=3,
                pending_at=second_pending,
                observed_at=second_pending + timedelta(seconds=3),
            )
            assert fork.status == "accepted-managed-recovery"
            copy_unique_branch_objects(branch)
            fork_key = str(fork.worker_acceptance_key)
            fork_entry = branch.fixture.s3._entry(fork_key, None)
            fork_value = json.loads(fork_entry.raw)
            assert (
                state.authority_module.worker_start_accepted_v2_s3_key(
                    worker_start_accepted=fork_value
                )
                == fork_key
            )
            assert fork_value["prior_worker_acceptance_key"] == first_key
            fork_keys.append(fork_key)
        assert len(set(fork_keys)) == 2
    elif mutation == "cycle":
        first_entry = state.fixture.s3._entry(first_key, None)
        cyclic_root = deepcopy(root)
        cyclic_root["prior_worker_acceptance_key"] = first_key
        cyclic_root["prior_worker_acceptance_file_sha256"] = _sha(first_entry.raw)
        cyclic_root["prior_worker_acceptance_body_sha256"] = first_entry.metadata[
            "glm52-body-sha256"
        ]
        cyclic_root = authority._rehash(
            cyclic_root,
            "worker_acceptance_body_sha256",
        )
        assert (
            state.authority_module.worker_start_accepted_v2_s3_key(
                worker_start_accepted=cyclic_root
            )
            == root_key
        )
        state.fixture.s3.add(
            key=root_key,
            raw=_canonical(cyclic_root, newline=False),
            run_id=authority.RUN_ID,
            body_sha256=str(cyclic_root["worker_acceptance_body_sha256"]),
            version_id="cyclic-root",
        )
    elif mutation == "disconnected":
        branch = branch_state("disconnected-root")
        disconnected = branch.advance_recovery(
            instance_id="i-3123456789abcdef0",
            cluster="sky-glm52-worker-d",
            recovery_count=0,
            pending_at=authority.PENDING_AT,
            observed_at=authority.WORKER_OBSERVED_AT,
        )
        assert disconnected.status == "accepted-initial"
        copy_unique_branch_objects(branch)
        disconnected_entry = branch.fixture.s3._entry(
            str(disconnected.worker_acceptance_key),
            None,
        )
        disconnected_value = json.loads(disconnected_entry.raw)
        assert (
            state.authority_module.worker_start_accepted_v2_s3_key(
                worker_start_accepted=disconnected_value
            )
            == disconnected.worker_acceptance_key
        )
    elif mutation == "repeated-instance":
        first_entry = state.fixture.s3._entry(first_key, None)
        repeated = deepcopy(first)
        repeated["instance_id"] = root["instance_id"]
        repeated = authority._rehash(
            repeated,
            "worker_acceptance_body_sha256",
        )
        repeated_key = state.authority_module.worker_start_accepted_v2_s3_key(
            worker_start_accepted=repeated
        )
        state.fixture.s3.current.pop((bucket, first_key))
        replace_acceptance(
            key=repeated_key,
            version_id="repeated-instance",
            value=repeated,
            last_modified=first_entry.last_modified,
        )
        repeated_raw = _canonical(repeated, newline=False)
        relinked_tip = deepcopy(tip)
        relinked_tip["prior_worker_acceptance_key"] = repeated_key
        relinked_tip["prior_worker_acceptance_file_sha256"] = _sha(repeated_raw)
        relinked_tip["prior_worker_acceptance_body_sha256"] = repeated[
            "worker_acceptance_body_sha256"
        ]
        relinked_tip = authority._rehash(
            relinked_tip,
            "worker_acceptance_body_sha256",
        )
        replace_acceptance(
            key=tip_key,
            value=relinked_tip,
            version_id="repeated-instance-tip",
            last_modified=state.fixture.s3._entry(tip_key, None).last_modified,
        )
    elif mutation == "reordered":
        reordered_root = deepcopy(root)
        reordered_root["recovery_count"] = 2
        reordered_root = authority._rehash(
            reordered_root,
            "worker_acceptance_body_sha256",
        )
        reordered_root_raw = _canonical(reordered_root, newline=False)
        reordered_first = deepcopy(first)
        reordered_first["prior_worker_acceptance_key"] = root_key
        reordered_first["prior_worker_acceptance_file_sha256"] = _sha(
            reordered_root_raw
        )
        reordered_first["prior_worker_acceptance_body_sha256"] = reordered_root[
            "worker_acceptance_body_sha256"
        ]
        reordered_first = authority._rehash(
            reordered_first,
            "worker_acceptance_body_sha256",
        )
        reordered_first_raw = _canonical(reordered_first, newline=False)
        reordered_tip = deepcopy(tip)
        reordered_tip["prior_worker_acceptance_key"] = first_key
        reordered_tip["prior_worker_acceptance_file_sha256"] = _sha(reordered_first_raw)
        reordered_tip["prior_worker_acceptance_body_sha256"] = reordered_first[
            "worker_acceptance_body_sha256"
        ]
        reordered_tip = authority._rehash(
            reordered_tip,
            "worker_acceptance_body_sha256",
        )
        for key, value, version_id in (
            (root_key, reordered_root, "reordered-root"),
            (first_key, reordered_first, "reordered-first"),
            (tip_key, reordered_tip, "reordered-tip"),
        ):
            replace_acceptance(
                key=key,
                value=value,
                version_id=version_id,
                last_modified=state.fixture.s3._entry(key, None).last_modified,
            )
        assert [
            reordered_root["recovery_count"],
            reordered_first["recovery_count"],
        ] == [2, 1]
    else:
        latch_key = str(first["worker_latch_key"])
        latch_version = str(first["worker_latch_version_id"])
        state.fixture.s3._entry(
            latch_key,
            latch_version,
        ).etag = '"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"'

    with pytest.raises(state.module.WorkerStartPublisherError) as captured:
        state.module.replay_worker_start_accepted_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            outputs=state.module.WorkerStartAdmissionOutputs(
                accepted_path=state.accepted_path,
                admission_receipt_path=state.receipt_path,
            ),
            services=state.services,
        )
    assert str(captured.value) == "frozen coordinator replay rejected authority"
    assert captured.value.__cause__ is not None
    assert str(captured.value.__cause__) == expected_cause
    assert not state.accepted_path.exists()
    assert not state.receipt_path.exists()


def test_post_replay_source_version_drift_fails_before_local_markers(
    state: _State,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state.publish()
    state.stage_acceptance()
    later = state.fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
    state.services = replace(state.services, utc_now=lambda: later)
    original_sweep = state.module._post_replay_sweep

    def drift_then_sweep(s3: Any) -> None:
        key = state.authority.descriptor_s3_uri.split("/", 3)[-1]
        current = state.fixture.s3._entry(key, None)
        state.fixture.s3.add(
            key=key,
            raw=current.raw,
            run_id=state.fixture.authority.RUN_ID,
            body_sha256=current.metadata["glm52-body-sha256"],
            version_id="post-replay-descriptor-drift",
            last_modified=later,
        )
        original_sweep(s3)

    monkeypatch.setattr(
        state.module,
        "_post_replay_sweep",
        drift_then_sweep,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="source current version drifted during replay",
    ):
        state.module.replay_worker_start_accepted_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            outputs=state.module.WorkerStartAdmissionOutputs(
                accepted_path=state.accepted_path,
                admission_receipt_path=state.receipt_path,
            ),
            services=state.services,
        )
    assert not state.accepted_path.exists()
    assert not state.receipt_path.exists()


@pytest.mark.parametrize(
    "basename",
    [
        "JOB_BINDING.json",
        "TIMELY_START_ACCEPTED.json",
        "TIMELY_START_LATCH.json",
    ],
)
def test_legacy_object_introduced_after_full_replay_fails_before_local_markers(
    state: _State,
    monkeypatch: pytest.MonkeyPatch,
    basename: str,
) -> None:
    state.publish()
    state.stage_acceptance()
    later = state.fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
    state.services = replace(state.services, utc_now=lambda: later)
    original_sweep = state.module._post_replay_sweep

    def sweep_then_introduce_legacy(s3: Any) -> None:
        original_sweep(s3)
        dynamic_prefix = (
            f"campaigns/{state.fixture.authority.RUN_ID}/monitor/must-start/"
            "qualification/"
            f"{state.fixture.upstream['intent']['intent_body_sha256']}/"
        )
        state.fixture.s3.add(
            key=f"{dynamic_prefix}{basename}",
            raw=b"{}",
            run_id=state.fixture.authority.RUN_ID,
            body_sha256="0" * 64,
            version_id=f"post-replay-legacy-{basename}",
            last_modified=later,
        )

    monkeypatch.setattr(
        state.module,
        "_post_replay_sweep",
        sweep_then_introduce_legacy,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="legacy",
    ):
        state.module.replay_worker_start_accepted_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            outputs=state.module.WorkerStartAdmissionOutputs(
                accepted_path=state.accepted_path,
                admission_receipt_path=state.receipt_path,
            ),
            services=state.services,
        )
    assert not state.accepted_path.exists()
    assert not state.receipt_path.exists()


@pytest.mark.parametrize(
    "basename",
    [
        "JOB_BINDING.json",
        "TIMELY_START_ACCEPTED.json",
        "TIMELY_START_LATCH.json",
    ],
)
def test_legacy_object_introduced_between_latch_and_acceptance_fails(
    state: _State,
    basename: str,
) -> None:
    state.publish()
    dynamic_prefix = (
        f"campaigns/{state.fixture.authority.RUN_ID}/monitor/must-start/"
        "qualification/"
        f"{state.fixture.upstream['intent']['intent_body_sha256']}/"
    )
    state.fixture.s3.add(
        key=f"{dynamic_prefix}{basename}",
        raw=b"{}",
        run_id=state.fixture.authority.RUN_ID,
        body_sha256="0" * 64,
        version_id=f"between-latch-and-acceptance-{basename}",
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="legacy",
    ):
        state.module.replay_worker_start_accepted_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            outputs=state.module.WorkerStartAdmissionOutputs(
                accepted_path=state.accepted_path,
                admission_receipt_path=state.receipt_path,
            ),
            services=state.services,
        )
    assert not state.accepted_path.exists()
    assert not state.receipt_path.exists()


def test_marker_only_coordinator_cannot_supply_a_fresh_replay_trace(
    state: _State,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state.publish()
    first = state.stage_acceptance()
    prepared = state.module._prepare(
        sources=state.sources,
        authority=state.authority,
        services=state.services,
    )
    latch, latch_transport = state.module._latch_and_transport(
        prepared,
        latch_path=state.latch_path,
        latch_transport_path=state.transport_path,
    )
    acceptance_key = state.module._acceptance_key(prepared, latch)

    def marker_only(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            status="idempotent-complete",
            decision_action="idempotent-complete",
            published_observation=False,
            published_acceptance=False,
            worker_latch_key=latch_transport["worker_latch_key"],
            worker_acceptance_key=acceptance_key,
            worker_acceptance_body_sha256=first.worker_acceptance_body_sha256,
        )

    monkeypatch.setattr(
        prepared.sources.coordinator,
        "coordinate_worker_start_acceptance_v2",
        marker_only,
    )
    fresh_s3 = state.module._AwsCliS3(
        services=state.services,
        cli=prepared.cli,
        bucket=prepared.authority.bucket,
        run_id=prepared.authority.run_id,
        required_acceptance_key=acceptance_key,
        acceptance_prefix=state.module._acceptance_prefix(prepared),
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="omitted core authority reads",
    ):
        state.module._coordinator_outcome(
            prepared=prepared,
            latch_transport=latch_transport,
            s3=fresh_s3,
            services=state.services,
            acceptance_key=acceptance_key,
        )
    assert fresh_s3.records == {}


def test_replay_times_out_without_acceptance_and_writes_no_marker(
    state: _State,
) -> None:
    state.publish()
    monotonic_values = iter([0.0, 0.0, 1.0])
    sleeps: list[float] = []
    state.services = replace(
        state.services,
        monotonic=lambda: next(monotonic_values),
        sleep=sleeps.append,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="worker-start acceptance wait timed out",
    ):
        state.module.replay_worker_start_accepted_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            outputs=state.module.WorkerStartAdmissionOutputs(
                accepted_path=state.accepted_path,
                admission_receipt_path=state.receipt_path,
            ),
            services=state.services,
            wait_timeout_seconds=1,
            poll_interval_seconds=1,
        )
    assert sleeps == [1.0]
    assert not state.accepted_path.exists()
    assert not state.receipt_path.exists()


@pytest.mark.parametrize("overshoot_seconds", [0.0, 0.25])
def test_replay_rejects_acceptance_first_visible_at_or_after_poll_deadline(
    state: _State,
    overshoot_seconds: float,
) -> None:
    state.publish()
    state.runner.calls.clear()
    now = 0.0
    staged: list[Any] = []

    def sleep_and_stage(seconds: float) -> None:
        nonlocal now
        assert seconds == 1.0
        now += seconds + overshoot_seconds
        staged.append(state.stage_acceptance())

    state.services = replace(
        state.services,
        utc_now=lambda: (
            state.fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
        ),
        monotonic=lambda: now,
        sleep=sleep_and_stage,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="worker-start acceptance wait timed out",
    ):
        state.module.replay_worker_start_accepted_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            outputs=state.module.WorkerStartAdmissionOutputs(
                accepted_path=state.accepted_path,
                admission_receipt_path=state.receipt_path,
            ),
            services=state.services,
            wait_timeout_seconds=1,
            poll_interval_seconds=1,
        )
    assert len(staged) == 1
    acceptance_key = staged[0].worker_acceptance_key
    acceptance_operations = [
        argv[2]
        for argv, _environ, _timeout in state.runner.calls
        if acceptance_key in argv
    ]
    assert acceptance_operations == ["head-object"]
    assert not state.accepted_path.exists()
    assert not state.receipt_path.exists()


@pytest.mark.parametrize("invalid", [True, "0", float("nan"), float("inf")])
def test_replay_rejects_invalid_monotonic_sample_after_successful_head(
    state: _State,
    invalid: object,
) -> None:
    state.publish()
    staged = state.stage_acceptance()
    state.runner.calls.clear()
    values = iter([0.0, invalid])
    state.services = replace(
        state.services,
        utc_now=lambda: (
            state.fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
        ),
        monotonic=lambda: next(values),
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="monotonic clock is invalid",
    ):
        state.module.replay_worker_start_accepted_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            outputs=state.module.WorkerStartAdmissionOutputs(
                accepted_path=state.accepted_path,
                admission_receipt_path=state.receipt_path,
            ),
            services=state.services,
            wait_timeout_seconds=1,
            poll_interval_seconds=1,
        )
    acceptance_operations = [
        argv[2]
        for argv, _environ, _timeout in state.runner.calls
        if staged.worker_acceptance_key in argv
    ]
    assert acceptance_operations == ["head-object"]
    assert not state.accepted_path.exists()
    assert not state.receipt_path.exists()


def test_replay_rejects_numeric_subclass_with_unstable_float_conversion(
    state: _State,
) -> None:
    class FlipFloat(float):
        conversions = 0

        def __float__(self) -> float:
            self.conversions += 1
            return 0.0 if self.conversions == 1 else float("nan")

    state.publish()
    state.stage_acceptance()
    unstable = FlipFloat(0.0)
    values = iter([unstable, 0.0, 0.0])
    state.services = replace(
        state.services,
        utc_now=lambda: (
            state.fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
        ),
        monotonic=lambda: next(values),
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="monotonic clock is invalid",
    ):
        state.module.replay_worker_start_accepted_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            outputs=state.module.WorkerStartAdmissionOutputs(
                accepted_path=state.accepted_path,
                admission_receipt_path=state.receipt_path,
            ),
            services=state.services,
            wait_timeout_seconds=1,
            poll_interval_seconds=1,
        )
    assert not state.accepted_path.exists()
    assert not state.receipt_path.exists()


def test_replay_rejects_monotonic_failure_after_successful_head(
    state: _State,
) -> None:
    state.publish()
    staged = state.stage_acceptance()
    state.runner.calls.clear()
    calls = 0

    def monotonic() -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic monotonic failure")
        return 0.0

    state.services = replace(
        state.services,
        utc_now=lambda: (
            state.fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
        ),
        monotonic=monotonic,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="monotonic clock failed",
    ):
        state.module.replay_worker_start_accepted_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            outputs=state.module.WorkerStartAdmissionOutputs(
                accepted_path=state.accepted_path,
                admission_receipt_path=state.receipt_path,
            ),
            services=state.services,
            wait_timeout_seconds=1,
            poll_interval_seconds=1,
        )
    acceptance_operations = [
        argv[2]
        for argv, _environ, _timeout in state.runner.calls
        if staged.worker_acceptance_key in argv
    ]
    assert acceptance_operations == ["head-object"]
    assert not state.accepted_path.exists()
    assert not state.receipt_path.exists()


def test_replay_rejects_acceptance_when_deadline_arrives_before_admission(
    state: _State,
) -> None:
    state.publish()
    staged = state.stage_acceptance()
    state.runner.calls.clear()
    values = iter([0.0, 0.0, 1.0])
    state.services = replace(
        state.services,
        utc_now=lambda: (
            state.fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
        ),
        monotonic=lambda: next(values),
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="worker-start acceptance wait timed out",
    ):
        state.module.replay_worker_start_accepted_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            outputs=state.module.WorkerStartAdmissionOutputs(
                accepted_path=state.accepted_path,
                admission_receipt_path=state.receipt_path,
            ),
            services=state.services,
            wait_timeout_seconds=1,
            poll_interval_seconds=1,
        )
    acceptance_operations = [
        argv[2]
        for argv, _environ, _timeout in state.runner.calls
        if staged.worker_acceptance_key in argv
    ]
    assert acceptance_operations == ["head-object", "get-object", "head-object"]
    assert not state.accepted_path.exists()
    assert not state.receipt_path.exists()


@pytest.mark.parametrize(
    ("wait", "poll"),
    [
        (True, 1),
        (0, 1),
        (-1, 1),
        (901, 1),
        (1, 0),
        (1, 31),
        (1, 2),
    ],
)
def test_invalid_wait_bounds_fail_before_imds(
    state: _State,
    wait: int,
    poll: int,
) -> None:
    state.publish()
    state.imds.calls.clear()
    state.runner.calls.clear()
    with pytest.raises(state.module.WorkerStartPublisherError):
        state.module.replay_worker_start_accepted_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            outputs=state.module.WorkerStartAdmissionOutputs(
                accepted_path=state.accepted_path,
                admission_receipt_path=state.receipt_path,
            ),
            services=state.services,
            wait_timeout_seconds=wait,
            poll_interval_seconds=poll,
        )
    assert state.imds.calls == []
    assert state.runner.calls == []


@pytest.mark.parametrize(
    "basename",
    [
        "JOB_BINDING.json",
        "TIMELY_START_LATCH.json",
        "TIMELY_START_ACCEPTED.json",
    ],
)
def test_forbidden_legacy_object_blocks_publication(
    state: _State,
    basename: str,
) -> None:
    dynamic_prefix = (
        f"campaigns/{state.fixture.authority.RUN_ID}/monitor/must-start/"
        f"qualification/{state.fixture.upstream['intent']['intent_body_sha256']}/"
    )
    state.fixture.s3.add(
        key=f"{dynamic_prefix}{basename}",
        raw=b"{}",
        run_id=state.fixture.authority.RUN_ID,
        body_sha256="0" * 64,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="legacy",
    ):
        state.publish()
    assert [
        call for call in state.runner.calls if call[0][1:3] == ("s3api", "put-object")
    ] == []


def test_current_worker_drift_rejects_predecessor_receipt(
    state: _State,
) -> None:
    state.publish()
    state.stage_acceptance()
    later = state.fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
    state.services = replace(state.services, utc_now=lambda: later)
    state.module.replay_worker_start_accepted_v2(
        sources=state.sources,
        authority=state.authority,
        latch_path=state.latch_path,
        latch_transport_path=state.transport_path,
        outputs=state.module.WorkerStartAdmissionOutputs(
            accepted_path=state.accepted_path,
            admission_receipt_path=state.receipt_path,
        ),
        services=state.services,
    )
    identity = json.loads(state.identity_raw)
    identity["instanceId"] = "i-11111111111111111"
    state.imds.identity_document = _canonical(identity, newline=False)
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="current worker",
    ):
        state.module.verify_worker_start_receipt_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            accepted_path=state.accepted_path,
            admission_receipt_path=state.receipt_path,
            services=state.services,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "instance-id",
        "instance-type",
        "image-id",
        "pending-time",
        "account",
        "region",
        "identity-document",
        "worker-role",
        "job-id",
    ],
)
def test_receipt_reverification_rejects_complete_worker_identity_drift_matrix(
    state: _State,
    mutation: str,
) -> None:
    _materialize_initial_receipt(state)
    receipt_before = state.receipt_path.read_bytes()
    identity = json.loads(state.identity_raw)
    if mutation == "instance-id":
        identity["instanceId"] = "i-11111111111111111"
    elif mutation == "instance-type":
        identity["instanceType"] = "p4d.24xlarge"
    elif mutation == "image-id":
        identity["imageId"] = "ami-11111111111111111"
    elif mutation == "pending-time":
        identity["pendingTime"] = "2026-07-26T12:00:07Z"
    elif mutation == "account":
        identity["accountId"] = "135792468013"
    elif mutation == "region":
        identity["region"] = "us-east-1"
    elif mutation == "identity-document":
        identity["privateIp"] = "10.0.1.10"
    elif mutation == "worker-role":
        state.runner.worker_role = "arn:aws:iam::246813579024:role/foreign-worker-role"
    else:
        state.services = replace(
            state.services,
            environ={"SKYPILOT_MANAGED_JOB_ID": "18"},
        )
    if mutation not in {"worker-role", "job-id"}:
        state.imds.identity_document = _canonical(identity, newline=False)
    with pytest.raises(state.module.WorkerStartPublisherError):
        state.module.verify_worker_start_receipt_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            accepted_path=state.accepted_path,
            admission_receipt_path=state.receipt_path,
            services=state.services,
        )
    assert state.receipt_path.read_bytes() == receipt_before


@pytest.mark.parametrize(
    "mutation",
    ["symlink", "mode", "partial", "unknown-field"],
)
def test_local_receipt_shape_failures_precede_external_authority(
    state: _State,
    tmp_path: Path,
    mutation: str,
) -> None:
    _materialize_initial_receipt(state)
    receipt_raw = state.receipt_path.read_bytes()
    if mutation == "symlink":
        target = tmp_path / "receipt-target.json"
        target.write_bytes(receipt_raw)
        os.chmod(target, 0o600)
        state.receipt_path.unlink()
        state.receipt_path.symlink_to(target)
    elif mutation == "mode":
        os.chmod(state.receipt_path, 0o644)
    elif mutation == "partial":
        state.receipt_path.write_bytes(receipt_raw[: len(receipt_raw) // 2])
    else:
        receipt = json.loads(receipt_raw)
        receipt["unknown"] = "forbidden"
        state.receipt_path.write_bytes(_canonical(receipt, newline=True))
    state.imds.calls.clear()
    state.runner.calls.clear()
    with pytest.raises(state.module.WorkerStartPublisherError):
        state.module.verify_worker_start_receipt_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            accepted_path=state.accepted_path,
            admission_receipt_path=state.receipt_path,
            services=state.services,
        )
    assert state.imds.calls == []
    assert state.runner.calls == []


def test_same_bytes_new_current_version_fails_receipt_reverification(
    state: _State,
) -> None:
    state.publish()
    state.stage_acceptance()
    later = state.fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
    state.services = replace(state.services, utc_now=lambda: later)
    state.module.replay_worker_start_accepted_v2(
        sources=state.sources,
        authority=state.authority,
        latch_path=state.latch_path,
        latch_transport_path=state.transport_path,
        outputs=state.module.WorkerStartAdmissionOutputs(
            accepted_path=state.accepted_path,
            admission_receipt_path=state.receipt_path,
        ),
        services=state.services,
    )
    receipt = json.loads(state.receipt_path.read_bytes())
    source = receipt["source_transports"][0]
    stored = state.fixture.s3._entry(source["key"], source["version_id"])
    state.fixture.s3.add(
        key=source["key"],
        raw=stored.raw,
        run_id=stored.metadata["glm52-run-id"],
        body_sha256=stored.metadata["glm52-body-sha256"],
        version_id="new-current-same-bytes",
        last_modified=stored.last_modified,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="current version",
    ):
        state.module.verify_worker_start_receipt_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            accepted_path=state.accepted_path,
            admission_receipt_path=state.receipt_path,
            services=state.services,
        )


def test_remote_receipt_source_transport_drift_fails_reverification(
    state: _State,
) -> None:
    _materialize_initial_receipt(state)
    receipt = json.loads(state.receipt_path.read_bytes())
    source = receipt["source_transports"][0]
    state.fixture.s3._entry(
        source["key"],
        source["version_id"],
    ).checksum = base64.b64encode(b"\0" * 32).decode("ascii")
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="transport|checksum",
    ):
        state.module.verify_worker_start_receipt_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            accepted_path=state.accepted_path,
            admission_receipt_path=state.receipt_path,
            services=state.services,
        )


def test_legacy_object_introduced_during_receipt_reverification_fails(
    state: _State,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _materialize_initial_receipt(state)
    original_coordinator = state.module._coordinator_outcome

    def replay_then_introduce_legacy(**kwargs: object) -> object:
        outcome = original_coordinator(**kwargs)
        dynamic_prefix = (
            f"campaigns/{state.fixture.authority.RUN_ID}/monitor/must-start/"
            "qualification/"
            f"{state.fixture.upstream['intent']['intent_body_sha256']}/"
        )
        state.fixture.s3.add(
            key=f"{dynamic_prefix}TIMELY_START_ACCEPTED.json",
            raw=b"{}",
            run_id=state.fixture.authority.RUN_ID,
            body_sha256="0" * 64,
            version_id="verify-legacy-race",
        )
        return outcome

    monkeypatch.setattr(
        state.module,
        "_coordinator_outcome",
        replay_then_introduce_legacy,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="legacy",
    ):
        state.module.verify_worker_start_receipt_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            accepted_path=state.accepted_path,
            admission_receipt_path=state.receipt_path,
            services=state.services,
        )


def test_interruption_before_receipt_persistence_leaves_no_authoritative_marker(
    state: _State,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state.publish()
    state.stage_acceptance()
    later = state.fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
    state.services = replace(state.services, utc_now=lambda: later)
    original_persist = state.module._persist_exact

    def interrupt_receipt(path: Path, raw: bytes, *, label: str) -> None:
        if label == "admission receipt output":
            raise state.module.WorkerStartPublisherError(
                "simulated receipt persistence interruption"
            )
        original_persist(path, raw, label=label)

    monkeypatch.setattr(
        state.module,
        "_persist_exact",
        interrupt_receipt,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="simulated receipt persistence interruption",
    ):
        state.module.replay_worker_start_accepted_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            outputs=state.module.WorkerStartAdmissionOutputs(
                accepted_path=state.accepted_path,
                admission_receipt_path=state.receipt_path,
            ),
            services=state.services,
        )
    assert state.accepted_path.exists()
    assert stat.S_IMODE(state.accepted_path.stat().st_mode) == 0o600
    assert not state.receipt_path.exists()


def test_lost_put_response_reconciles_once_without_retry(
    state: _State,
) -> None:
    bucket = state.fixture.authority.BUCKET
    key = str(state.fixture.request.worker_latch_key)
    version = str(state.fixture.request.worker_latch_version_id)
    state.fixture.s3.current.pop((bucket, key))
    state.fixture.s3.versions.pop((bucket, key, version))
    original = state.runner
    lost = False

    def lose_put_response(
        argv: tuple[str, ...],
        environ: dict[str, str],
        timeout: int,
    ) -> Any:
        nonlocal lost
        result = original(argv, environ, timeout)
        if argv[1:3] == ("s3api", "put-object") and not lost:
            lost = True
            raise ConnectionError("simulated response loss")
        return result

    state.services = replace(state.services, command_runner=lose_put_response)
    outcome = state.publish()
    assert outcome.status == "latch-idempotent"
    puts = [call for call in original.calls if call[0][1:3] == ("s3api", "put-object")]
    assert len(puts) == 1


@pytest.mark.parametrize(
    "code",
    [
        "409",
        "412",
        "ConditionalRequestConflict",
        "PreconditionFailed",
    ],
)
def test_every_narrow_latch_conflict_reconciles_without_a_second_put(
    state: _State,
    code: str,
) -> None:
    original = state.runner

    def conflict(
        argv: tuple[str, ...],
        environ: dict[str, str],
        timeout: int,
    ) -> Any:
        if argv[1:3] == ("s3api", "put-object"):
            original.calls.append((argv, dict(environ), timeout))
            return state.module.WorkerStartCommandResult(
                returncode=255,
                stdout=b"",
                stderr=(
                    f"An error occurred ({code}) when calling the "
                    "PutObject operation: conflict\n"
                ).encode("ascii"),
            )
        return original(argv, environ, timeout)

    state.services = replace(state.services, command_runner=conflict)
    outcome = state.publish()
    assert outcome.status == "latch-idempotent"
    puts = [call for call in original.calls if call[0][1:3] == ("s3api", "put-object")]
    assert len(puts) == 1


def test_ambiguous_put_without_a_winner_fails_without_retry(
    state: _State,
) -> None:
    bucket = state.fixture.authority.BUCKET
    key = str(state.fixture.request.worker_latch_key)
    version = str(state.fixture.request.worker_latch_version_id)
    state.fixture.s3.current.pop((bucket, key))
    state.fixture.s3.versions.pop((bucket, key, version))
    original = state.runner

    def lose_before_write(
        argv: tuple[str, ...],
        environ: dict[str, str],
        timeout: int,
    ) -> Any:
        if argv[1:3] == ("s3api", "put-object"):
            original.calls.append((argv, dict(environ), timeout))
            raise TimeoutError("simulated pre-write transport loss")
        return original(argv, environ, timeout)

    state.services = replace(
        state.services,
        command_runner=lose_before_write,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="winner is unavailable",
    ):
        state.publish()
    puts = [call for call in original.calls if call[0][1:3] == ("s3api", "put-object")]
    assert len(puts) == 1


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        (
            b"",
            (
                b"An error occurred (MethodNotAllowed) when calling the "
                b"PutObject operation: rejected\n"
            ),
        ),
        (b"", b"malformed service failure\n"),
        (
            b'{"Error":"PreconditionFailed"}',
            (
                b"An error occurred (PreconditionFailed) when calling the "
                b"PutObject operation: conflict\n"
            ),
        ),
    ],
)
def test_noncanonical_or_nonconflict_put_failure_never_reconciles(
    state: _State,
    stdout: bytes,
    stderr: bytes,
) -> None:
    original = state.runner

    def reject(
        argv: tuple[str, ...],
        environ: dict[str, str],
        timeout: int,
    ) -> Any:
        if argv[1:3] == ("s3api", "put-object"):
            original.calls.append((argv, dict(environ), timeout))
            return state.module.WorkerStartCommandResult(
                returncode=255,
                stdout=stdout,
                stderr=stderr,
            )
        return original(argv, environ, timeout)

    state.services = replace(state.services, command_runner=reject)
    with pytest.raises(state.module.WorkerStartPublisherError):
        state.publish()
    operations = [call[0][2] for call in original.calls if call[0][1] == "s3api"]
    assert operations[-1] == "put-object"
    assert "head-object" not in operations[operations.index("put-object") + 1 :]


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "etag",
        "last-modified",
        "content-length",
        "checksum",
        "checksum-type",
        "content-type",
        "metadata",
        "missing-field",
    ],
)
def test_latch_winner_transport_corruption_fails_before_local_output(
    state: _State,
    mutation: str,
) -> None:
    original = state.runner
    corrupted = False

    def corrupt_versioned_head(
        argv: tuple[str, ...],
        environ: dict[str, str],
        timeout: int,
    ) -> Any:
        nonlocal corrupted
        result = original(argv, environ, timeout)
        if (
            argv[1:3] == ("s3api", "head-object")
            and "--version-id" in argv
            and not corrupted
        ):
            corrupted = True
            response = json.loads(result.stdout)
            if mutation == "version":
                response["VersionId"] = "foreign-version"
            elif mutation == "etag":
                response["ETag"] = '"00000000000000000000000000000000"'
            elif mutation == "last-modified":
                response["LastModified"] = "2026-07-26T12:00:11Z"
            elif mutation == "content-length":
                response["ContentLength"] += 1
            elif mutation == "checksum":
                response["ChecksumSHA256"] = base64.b64encode(b"\0" * 32).decode(
                    "ascii"
                )
            elif mutation == "checksum-type":
                response["ChecksumType"] = "COMPOSITE"
            elif mutation == "content-type":
                response["ContentType"] = "text/plain"
            elif mutation == "metadata":
                response["Metadata"] = {
                    "glm52-run-id": state.fixture.authority.RUN_ID,
                    "glm52-body-sha256": "0" * 64,
                }
            else:
                response.pop("ChecksumType")
            return state.module.WorkerStartCommandResult(
                returncode=0,
                stdout=_json_bytes(response),
                stderr=b"",
            )
        return result

    state.services = replace(
        state.services,
        command_runner=corrupt_versioned_head,
    )
    with pytest.raises(state.module.WorkerStartPublisherError):
        state.publish()
    assert corrupted is True
    assert not state.latch_path.exists()
    assert not state.transport_path.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "foreign-bytes",
        "content-length",
        "checksum",
        "checksum-type",
        "content-type",
        "metadata",
    ],
)
def test_coherent_wrong_get_and_head_winner_fails_before_local_output(
    state: _State,
    mutation: str,
) -> None:
    key = str(state.fixture.request.worker_latch_key)
    version = str(state.fixture.request.worker_latch_version_id)
    entry = state.fixture.s3._entry(key, version)
    if mutation == "foreign-bytes":
        entry.raw = b"{}"
        entry.content_length = len(entry.raw)
        entry.checksum = base64.b64encode(hashlib.sha256(entry.raw).digest()).decode(
            "ascii"
        )
    elif mutation == "content-length":
        entry.content_length += 1
    elif mutation == "checksum":
        entry.checksum = base64.b64encode(b"\0" * 32).decode("ascii")
    elif mutation == "checksum-type":
        entry.checksum_type = "COMPOSITE"
    elif mutation == "content-type":
        entry.content_type = "text/plain"
    else:
        entry.metadata = {
            "glm52-run-id": state.fixture.authority.RUN_ID,
            "glm52-body-sha256": "0" * 64,
        }

    with pytest.raises(state.module.WorkerStartPublisherError):
        state.publish()
    assert not state.latch_path.exists()
    assert not state.transport_path.exists()


@pytest.mark.parametrize(
    "missing_field",
    [
        "VersionId",
        "ETag",
        "LastModified",
        "ContentLength",
        "ChecksumSHA256",
        "ChecksumType",
        "ContentType",
        "Metadata",
    ],
)
def test_missing_winner_get_transport_field_fails_before_local_output(
    state: _State,
    missing_field: str,
) -> None:
    original = state.runner
    mutated = False

    def omit_get_field(
        argv: tuple[str, ...],
        environ: dict[str, str],
        timeout: int,
    ) -> Any:
        nonlocal mutated
        result = original(argv, environ, timeout)
        if argv[1:3] == ("s3api", "get-object") and not mutated:
            mutated = True
            response = json.loads(result.stdout)
            response.pop(missing_field)
            return state.module.WorkerStartCommandResult(
                returncode=0,
                stdout=_json_bytes(response),
                stderr=b"",
            )
        return result

    state.services = replace(
        state.services,
        command_runner=omit_get_field,
    )
    with pytest.raises(state.module.WorkerStartPublisherError):
        state.publish()
    assert mutated is True
    assert not state.latch_path.exists()
    assert not state.transport_path.exists()


def test_definitive_put_error_never_reconciles(
    state: _State,
) -> None:
    original = state.runner

    def deny_put(
        argv: tuple[str, ...],
        environ: dict[str, str],
        timeout: int,
    ) -> Any:
        if argv[1:3] == ("s3api", "put-object"):
            original.calls.append((argv, dict(environ), timeout))
            return state.module.WorkerStartCommandResult(
                returncode=255,
                stdout=b"",
                stderr=(
                    b"An error occurred (AccessDenied) when calling the "
                    b"PutObject operation: denied\n"
                ),
            )
        return original(argv, environ, timeout)

    state.services = replace(state.services, command_runner=deny_put)
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="definitively",
    ):
        state.publish()
    s3_operations = [call[0][2] for call in original.calls if call[0][1] == "s3api"]
    assert "put-object" in s3_operations
    assert "head-object" not in s3_operations


def test_conflict_without_current_winner_fails_closed(
    state: _State,
) -> None:
    bucket = state.fixture.authority.BUCKET
    key = str(state.fixture.request.worker_latch_key)
    version = str(state.fixture.request.worker_latch_version_id)
    state.fixture.s3.current.pop((bucket, key))
    state.fixture.s3.versions.pop((bucket, key, version))
    original = state.runner

    def conflict_without_winner(
        argv: tuple[str, ...],
        environ: dict[str, str],
        timeout: int,
    ) -> Any:
        if argv[1:3] == ("s3api", "put-object"):
            original.calls.append((argv, dict(environ), timeout))
            return state.module.WorkerStartCommandResult(
                returncode=255,
                stdout=b"",
                stderr=(
                    b"An error occurred (PreconditionFailed) when calling the "
                    b"PutObject operation: conflict\n"
                ),
            )
        return original(argv, environ, timeout)

    state.services = replace(
        state.services,
        command_runner=conflict_without_winner,
    )
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="winner is unavailable",
    ):
        state.publish()
    puts = [call for call in original.calls if call[0][1:3] == ("s3api", "put-object")]
    assert len(puts) == 1


def test_replay_rejects_malformed_local_latch_before_external_calls(
    state: _State,
) -> None:
    state.publish()
    state.latch_path.write_bytes(b"{}")
    state.imds.calls.clear()
    state.runner.calls.clear()
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="local worker latch",
    ):
        state.module.replay_worker_start_accepted_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            outputs=state.module.WorkerStartAdmissionOutputs(
                accepted_path=state.accepted_path,
                admission_receipt_path=state.receipt_path,
            ),
            services=state.services,
        )
    assert state.imds.calls == []
    assert state.runner.calls == []


def test_verify_rejects_malformed_receipt_inventory_before_external_calls(
    state: _State,
) -> None:
    state.publish()
    state.stage_acceptance()
    state.services = replace(
        state.services,
        utc_now=lambda: (
            state.fixture.authority.WORKER_OBSERVED_AT + timedelta(seconds=30)
        ),
    )
    state.module.replay_worker_start_accepted_v2(
        sources=state.sources,
        authority=state.authority,
        latch_path=state.latch_path,
        latch_transport_path=state.transport_path,
        outputs=state.module.WorkerStartAdmissionOutputs(
            accepted_path=state.accepted_path,
            admission_receipt_path=state.receipt_path,
        ),
        services=state.services,
    )
    receipt = json.loads(state.receipt_path.read_bytes())
    receipt["source_transports"][0]["content_length"] = True
    receipt.pop("admission_receipt_body_sha256")
    receipt["admission_receipt_body_sha256"] = _sha(_canonical(receipt, newline=False))
    state.receipt_path.write_bytes(_canonical(receipt, newline=True))
    state.imds.calls.clear()
    state.runner.calls.clear()
    with pytest.raises(
        state.module.WorkerStartPublisherError,
        match="source transport",
    ):
        state.module.verify_worker_start_receipt_v2(
            sources=state.sources,
            authority=state.authority,
            latch_path=state.latch_path,
            latch_transport_path=state.transport_path,
            accepted_path=state.accepted_path,
            admission_receipt_path=state.receipt_path,
            services=state.services,
        )
    assert state.imds.calls == []
    assert state.runner.calls == []


def _stat_with_identity(
    details: os.stat_result,
    *,
    uid: int,
    mode: int | None = None,
) -> os.stat_result:
    return os.stat_result(
        (
            details.st_mode if mode is None else mode,
            details.st_ino,
            details.st_dev,
            details.st_nlink,
            uid,
            details.st_gid,
            details.st_size,
            details.st_atime,
            details.st_mtime,
            details.st_ctime,
        )
    )


def _root_owned_cli_closure(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    fixed_root = tmp_path / "usr" / "local"
    prefix = fixed_root / "aws-cli" / "v2"
    version_root = prefix / "2.15.1"
    executable = version_root / "dist" / "aws"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"aws")
    os.chmod(executable, 0o755)
    payload = version_root / "lib" / "payload"
    payload.parent.mkdir()
    payload.write_bytes(b"immutable")
    os.chmod(payload, 0o644)
    current = prefix / "current"
    current.symlink_to(version_root.name)
    entrypoint = fixed_root / "bin" / "aws"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.symlink_to(current / "dist" / "aws")
    return prefix, version_root, executable, payload, entrypoint


def _patch_cli_lstat_as_root(
    monkeypatch: pytest.MonkeyPatch,
    *,
    root: Path,
    overrides: dict[Path, tuple[int, int | None]],
) -> None:
    original_lstat = Path.lstat

    def fake_lstat(path: Path) -> os.stat_result:
        details = original_lstat(path)
        try:
            Path(path).relative_to(root)
        except ValueError:
            return details
        uid, mode = overrides.get(Path(path), (0, None))
        return _stat_with_identity(details, uid=uid, mode=mode)

    monkeypatch.setattr(Path, "lstat", fake_lstat)


@pytest.mark.parametrize(
    "mutation",
    [
        "nonroot-entrypoint",
        "group-writable-member",
        "world-writable-directory",
        "forbidden-type",
        "nonexecutable-target",
        "closure-drift",
    ],
)
def test_aws_cli_attestation_rejects_real_closure_authority_failures(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    prefix, version_root, executable, payload, entrypoint = _root_owned_cli_closure(
        tmp_path
    )
    overrides: dict[Path, tuple[int, int | None]] = {}
    if mutation == "nonroot-entrypoint":
        overrides[entrypoint] = (501, None)
    elif mutation == "group-writable-member":
        overrides[payload] = (
            0,
            (payload.lstat().st_mode & ~0o777) | 0o664,
        )
    elif mutation == "world-writable-directory":
        library = payload.parent
        overrides[library] = (
            0,
            (library.lstat().st_mode & ~0o777) | 0o777,
        )
    elif mutation == "forbidden-type":
        os.mkfifo(version_root / "forbidden")
    elif mutation == "nonexecutable-target":
        overrides[executable] = (
            0,
            (executable.lstat().st_mode & ~0o777) | 0o644,
        )
    monkeypatch.setattr(module, "AWS_ENTRYPOINT", entrypoint)
    monkeypatch.setattr(module, "AWS_INSTALL_PREFIX", prefix)
    _patch_cli_lstat_as_root(
        monkeypatch,
        root=tmp_path,
        overrides=overrides,
    )
    if mutation == "closure-drift":
        original_snapshot = module._snapshot_cli_closure
        snapshots = 0

        def mutate_after_snapshot(root: Path) -> Any:
            nonlocal snapshots
            snapshot = original_snapshot(root)
            snapshots += 1
            if snapshots == 1:
                payload.write_bytes(b"changed-after-first-snapshot")
            return snapshot

        monkeypatch.setattr(
            module,
            "_snapshot_cli_closure",
            mutate_after_snapshot,
        )
    with pytest.raises(module.WorkerStartPublisherError):
        module._attest_aws_cli()


def test_symlink_resolver_follows_symlinked_parent_components(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version_root = tmp_path / "aws-cli" / "v2" / "2.15.1"
    executable = version_root / "dist" / "aws"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"aws")
    current = tmp_path / "aws-cli" / "v2" / "current"
    current.symlink_to(version_root.name)
    entrypoint = tmp_path / "bin" / "aws"
    entrypoint.parent.mkdir()
    entrypoint.symlink_to(current / "dist" / "aws")
    monkeypatch.setattr(module, "_check_root_owned_immutable", lambda *_: None)
    resolved, links = module._resolve_symlinks(entrypoint)
    assert resolved == executable
    assert links == (entrypoint, current)


def test_fixed_aws_cli_attestation_and_executable_replacement_detection(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_root = tmp_path / "usr" / "local"
    prefix = fixed_root / "aws-cli" / "v2"
    version_root = prefix / "2.15.1"
    executable = version_root / "dist" / "aws"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"aws-v1")
    os.chmod(executable, 0o755)
    current = prefix / "current"
    current.symlink_to(version_root.name)
    entrypoint = fixed_root / "bin" / "aws"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.symlink_to(current / "dist" / "aws")
    monkeypatch.setattr(module, "AWS_ENTRYPOINT", entrypoint)
    monkeypatch.setattr(module, "AWS_INSTALL_PREFIX", prefix)
    monkeypatch.setattr(module, "_check_root_owned_immutable", lambda *_: None)
    attestation = module._attest_aws_cli()
    assert attestation.executable == executable
    module._revalidate_aws_cli(attestation)

    replacement_root = prefix / "2.15.2"
    replacement = replacement_root / "dist" / "aws"
    replacement.parent.mkdir(parents=True)
    replacement.write_bytes(b"aws-v2")
    os.chmod(replacement, 0o755)
    current.unlink()
    current.symlink_to(replacement_root.name)
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="changed before execution",
    ):
        module._revalidate_aws_cli(attestation)


def test_aws_cli_attestation_rejects_invalid_version_and_escaping_closure_link(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_root = tmp_path / "usr" / "local"
    prefix = fixed_root / "aws-cli" / "v2"
    invalid_root = prefix / "current-version"
    invalid_executable = invalid_root / "dist" / "aws"
    invalid_executable.parent.mkdir(parents=True)
    invalid_executable.write_bytes(b"aws")
    os.chmod(invalid_executable, 0o755)
    entrypoint = fixed_root / "bin" / "aws"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.symlink_to(invalid_executable)
    monkeypatch.setattr(module, "AWS_ENTRYPOINT", entrypoint)
    monkeypatch.setattr(module, "AWS_INSTALL_PREFIX", prefix)
    monkeypatch.setattr(module, "_check_root_owned_immutable", lambda *_: None)
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="version root is not canonical",
    ):
        module._attest_aws_cli()

    entrypoint.unlink()
    valid_root = prefix / "2.15.1"
    executable = valid_root / "dist" / "aws"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"aws")
    os.chmod(executable, 0o755)
    (valid_root / "escape").symlink_to(tmp_path / "outside")
    (tmp_path / "outside").write_bytes(b"outside")
    entrypoint.symlink_to(executable)
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="closure symlink escapes",
    ):
        module._attest_aws_cli()


def test_aws_cli_symlink_loop_is_rejected(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.symlink_to(second)
    second.symlink_to(first)
    monkeypatch.setattr(module, "_check_root_owned_immutable", lambda *_: None)
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="loops",
    ):
        module._resolve_symlinks(first)


@pytest.mark.parametrize(
    "forbidden",
    ["--aws-bin", "--mode"],
)
def test_cli_has_no_executable_or_mode_override(
    module: Any,
    forbidden: str,
) -> None:
    parser = module._parser()
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="usage is invalid",
    ):
        parser.parse_args(["publish-latch-v2", forbidden, "forbidden"])


def test_cli_has_no_legacy_command_alias(module: Any) -> None:
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="usage is invalid",
    ):
        module._parser().parse_args(["publish-must-start-latch"])


def test_cli_rejects_long_option_abbreviation(
    module: Any,
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    argv = [
        "publish-latch-v2",
        "--publisher-file-sha",
        digest,
        "--campaign-policy",
        str(SOURCE_PATHS["campaign"]),
        "--campaign-policy-file-sha256",
        digest,
        "--must-start-policy",
        str(SOURCE_PATHS["must_start"]),
        "--must-start-policy-file-sha256",
        digest,
        "--dynamic-policy",
        str(SOURCE_PATHS["dynamic"]),
        "--dynamic-policy-file-sha256",
        digest,
        "--worker-policy",
        str(SOURCE_PATHS["worker"]),
        "--worker-policy-file-sha256",
        digest,
        "--coordinator",
        str(SOURCE_PATHS["coordinator"]),
        "--coordinator-file-sha256",
        digest,
        "--descriptor",
        str(tmp_path / "descriptor.json"),
        "--descriptor-s3-uri",
        "s3://bucket/descriptor.json",
        "--descriptor-file-sha256",
        digest,
        "--intent",
        str(tmp_path / "intent.json"),
        "--intent-s3-uri",
        "s3://bucket/intent.json",
        "--intent-file-sha256",
        digest,
        "--intent-body-sha256",
        digest,
        "--latch-output",
        str(tmp_path / "latch.json"),
        "--latch-transport-output",
        str(tmp_path / "transport.json"),
    ]
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="usage is invalid",
    ):
        module._parser().parse_args(argv)


def test_cli_usage_error_is_redacted_at_entrypoint(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_like_value = "DO_NOT_ECHO_THIS_VALUE"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "publish-latch-v2",
            "--unknown",
            secret_like_value,
        ],
    )
    assert module._entrypoint() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "worker-start publisher usage is invalid\n"
    assert secret_like_value not in captured.err


def test_entrypoint_maps_only_wait_timeout_to_exit_75(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> int:
        raise module.WorkerStartPublisherError("worker-start acceptance wait timed out")

    monkeypatch.setattr(module, "main", fail)
    assert module._entrypoint() == 75
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "worker-start acceptance wait timed out\n"


def test_entrypoint_redacts_unexpected_failure_and_exits_2(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> int:
        raise RuntimeError("secret-like diagnostic")

    monkeypatch.setattr(module, "main", fail)
    assert module._entrypoint() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "worker-start publisher failed closed\n"


def test_local_persistence_allows_expected_directory_metadata_change(
    module: Any,
    tmp_path: Path,
) -> None:
    parent = tmp_path / "authority"
    parent.mkdir()
    output = parent / "marker.json"
    module._persist_exact(output, b"exact", label="test marker")
    assert output.read_bytes() == b"exact"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_local_persistence_closes_file_before_parent_directory_fsync(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority"
    parent.mkdir()
    output = parent / "marker.json"
    real_fsync = os.fsync
    real_close = os.close
    operations: list[tuple[str, str]] = []

    def descriptor_kind(descriptor: int) -> str:
        return "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"

    def recording_fsync(descriptor: int) -> None:
        operations.append(("fsync", descriptor_kind(descriptor)))
        real_fsync(descriptor)

    def recording_close(descriptor: int) -> None:
        operations.append(("close", descriptor_kind(descriptor)))
        real_close(descriptor)

    monkeypatch.setattr(module.os, "fsync", recording_fsync)
    monkeypatch.setattr(module.os, "close", recording_close)
    module._persist_exact(output, b"exact", label="test marker")

    assert operations == [
        ("fsync", "file"),
        ("close", "file"),
        ("fsync", "directory"),
        ("close", "file"),
        ("close", "directory"),
    ]
    assert output.read_bytes() == b"exact"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_local_persistence_rejects_same_inode_mutation_after_final_read(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority"
    parent.mkdir()
    output = parent / "marker.json"
    real_fstat = os.fstat
    file_fstat_calls = 0
    mutated = False

    def fstat_and_mutate(descriptor: int) -> os.stat_result:
        nonlocal file_fstat_calls, mutated
        details = real_fstat(descriptor)
        if stat.S_ISREG(details.st_mode):
            file_fstat_calls += 1
            if file_fstat_calls == 4:
                mutation_fd = os.open(output, os.O_WRONLY | os.O_NOFOLLOW)
                try:
                    assert os.write(mutation_fd, b"evil!") == 5
                finally:
                    os.close(mutation_fd)
                mutated = True
                details = real_fstat(descriptor)
        return details

    monkeypatch.setattr(module.os, "fstat", fstat_and_mutate)
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="final bytes changed during persistence",
    ):
        module._persist_exact(output, b"exact", label="test marker")
    assert mutated is True
    assert output.read_bytes() == b"evil!"


def test_local_persistence_closes_parent_when_initial_fstat_fails(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority"
    parent.mkdir()
    output = parent / "marker.json"
    real_fstat = os.fstat
    real_close = os.close
    failed_parent_fd: int | None = None
    closed: list[int] = []

    def failing_fstat(descriptor: int) -> os.stat_result:
        nonlocal failed_parent_fd
        details = real_fstat(descriptor)
        if stat.S_ISDIR(details.st_mode) and failed_parent_fd is None:
            failed_parent_fd = descriptor
            raise OSError("synthetic parent fstat failure")
        return details

    def recording_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(module.os, "fstat", failing_fstat)
    monkeypatch.setattr(module.os, "close", recording_close)
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="parent cannot be opened",
    ):
        module._persist_exact(output, b"exact", label="test marker")
    assert failed_parent_fd is not None
    closed_by_publisher = closed.count(failed_parent_fd) == 1
    if not closed_by_publisher:
        real_close(failed_parent_fd)
    assert closed_by_publisher is True
    assert not output.exists()


def test_local_persistence_does_not_double_close_after_close_reports_failure(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority"
    parent.mkdir()
    output = parent / "marker.json"
    real_fstat = os.fstat
    real_close = os.close
    close_attempts: list[int] = []
    failed_file_fd: int | None = None

    def close_then_report_failure(descriptor: int) -> None:
        nonlocal failed_file_fd
        close_attempts.append(descriptor)
        if failed_file_fd is None:
            details = real_fstat(descriptor)
            if stat.S_ISREG(details.st_mode):
                failed_file_fd = descriptor
                real_close(descriptor)
                raise OSError("synthetic close-after-success failure")
        real_close(descriptor)

    monkeypatch.setattr(module.os, "close", close_then_report_failure)
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="persistence failed",
    ):
        module._persist_exact(output, b"exact", label="test marker")
    assert failed_file_fd is not None
    assert close_attempts.count(failed_file_fd) == 1


def test_local_persistence_rejects_actual_parent_path_replacement(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority"
    moved_parent = tmp_path / "authority-original"
    parent.mkdir()
    output = parent / "marker.json"
    real_fsync = os.fsync
    swapped = False

    def fsync_and_swap(descriptor: int) -> None:
        nonlocal swapped
        details = os.fstat(descriptor)
        real_fsync(descriptor)
        if stat.S_ISDIR(details.st_mode) and not swapped:
            swapped = True
            parent.rename(moved_parent)
            parent.mkdir()

    monkeypatch.setattr(module.os, "fsync", fsync_and_swap)
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="parent changed during persistence",
    ):
        module._persist_exact(output, b"exact", label="test marker")
    assert swapped is True
    assert not output.exists()
    assert (moved_parent / output.name).read_bytes() == b"exact"


def test_local_persistence_rejects_same_directory_entry_replacement(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority"
    parent.mkdir()
    output = parent / "marker.json"
    real_fsync = os.fsync
    replaced = False

    def fsync_and_replace(descriptor: int) -> None:
        nonlocal replaced
        details = os.fstat(descriptor)
        real_fsync(descriptor)
        if stat.S_ISDIR(details.st_mode) and not replaced:
            replaced = True
            output.unlink()
            output.write_bytes(b"foreign")
            os.chmod(output, 0o600)

    monkeypatch.setattr(module.os, "fsync", fsync_and_replace)
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="final entry changed during persistence",
    ):
        module._persist_exact(output, b"exact", label="test marker")
    assert replaced is True
    assert output.read_bytes() == b"foreign"


def test_local_persistence_rejects_same_inode_rewrite_during_directory_fsync(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority"
    parent.mkdir()
    output = parent / "marker.json"
    real_fsync = os.fsync
    rewritten = False
    before_rewrite: os.stat_result | None = None
    after_rewrite: os.stat_result | None = None

    def fsync_and_rewrite(descriptor: int) -> None:
        nonlocal rewritten, before_rewrite, after_rewrite
        details = os.fstat(descriptor)
        real_fsync(descriptor)
        if stat.S_ISDIR(details.st_mode) and not rewritten:
            before_rewrite = output.stat()
            mutation_fd = os.open(output, os.O_WRONLY | os.O_NOFOLLOW)
            try:
                os.lseek(mutation_fd, 0, os.SEEK_SET)
                assert os.write(mutation_fd, b"evil!") == 5
                os.lseek(mutation_fd, 0, os.SEEK_SET)
                assert os.write(mutation_fd, b"exact") == 5
            finally:
                os.close(mutation_fd)
            changed = output.stat()
            os.utime(
                output,
                ns=(changed.st_atime_ns, changed.st_mtime_ns + 1_000_000_000),
            )
            after_rewrite = output.stat()
            rewritten = True

    monkeypatch.setattr(module.os, "fsync", fsync_and_rewrite)
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="final entry changed during persistence",
    ):
        module._persist_exact(output, b"exact", label="test marker")
    assert rewritten is True
    assert before_rewrite is not None
    assert after_rewrite is not None
    assert (before_rewrite.st_dev, before_rewrite.st_ino) == (
        after_rewrite.st_dev,
        after_rewrite.st_ino,
    )
    assert before_rewrite.st_size == after_rewrite.st_size
    assert (
        before_rewrite.st_mtime_ns,
        before_rewrite.st_ctime_ns,
    ) != (
        after_rewrite.st_mtime_ns,
        after_rewrite.st_ctime_ns,
    )
    assert output.read_bytes() == b"exact"


def test_local_persistence_retry_completes_file_and_directory_fsync(
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority"
    parent.mkdir()
    output = parent / "marker.json"
    real_fsync = os.fsync
    failed_directory_fsync = False

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal failed_directory_fsync
        details = os.fstat(descriptor)
        if stat.S_ISDIR(details.st_mode) and not failed_directory_fsync:
            failed_directory_fsync = True
            raise OSError("synthetic directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_first_directory_fsync)
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="persistence failed",
    ):
        module._persist_exact(output, b"exact", label="test marker")
    assert failed_directory_fsync is True
    assert output.read_bytes() == b"exact"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600

    fsync_kinds: list[str] = []

    def recording_fsync(descriptor: int) -> None:
        details = os.fstat(descriptor)
        fsync_kinds.append("directory" if stat.S_ISDIR(details.st_mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", recording_fsync)
    module._persist_exact(output, b"exact", label="test marker")
    assert fsync_kinds == ["file", "directory"]
    assert output.read_bytes() == b"exact"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_fd_reader_enforces_mode_on_the_open_file(
    module: Any,
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.json"
    path.write_bytes(b"{}")
    os.chmod(path, 0o644)
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="mode is invalid",
    ):
        module._read_stable_regular(
            path,
            label="test authority",
            required_mode=0o600,
        )


def test_cli_closure_accepts_normal_symlink_mode_but_requires_searchable_directory(
    module: Any,
) -> None:
    symlink_details = os.stat_result(
        (stat.S_IFLNK | 0o777, 1, 1, 1, 0, 0, 0, 0, 0, 0),
    )
    module._check_root_owned_immutable(Path("/synthetic-link"), symlink_details)
    unsearchable_directory = os.stat_result(
        (stat.S_IFDIR | 0o600, 2, 1, 1, 0, 0, 0, 0, 0, 0),
    )
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="not searchable",
    ):
        module._check_root_owned_immutable(
            Path("/synthetic-directory"),
            unsearchable_directory,
        )


def test_aws_cli_utc_offset_last_modified_is_canonicalized(
    module: Any,
) -> None:
    canonical, parsed = module._last_modified(
        "2026-07-26T12:00:10+00:00",
        label="test transport",
    )
    assert canonical == "2026-07-26T12:00:10Z"
    assert parsed == datetime(2026, 7, 26, 12, 0, 10, tzinfo=UTC)
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="not UTC",
    ):
        module._last_modified(
            "2026-07-26T05:00:10-07:00",
            label="test transport",
        )


def test_cli_list_response_translates_member_last_modified(
    module: Any,
) -> None:
    translated = module._cli_response(
        {
            "IsTruncated": False,
            "KeyCount": 1,
            "Contents": [
                {
                    "Key": "campaigns/run/example.json",
                    "LastModified": "2026-07-26T12:00:10+00:00",
                    "Size": 123,
                }
            ],
        }
    )
    assert translated == {
        "IsTruncated": False,
        "KeyCount": 1,
        "Contents": [
            {
                "Key": "campaigns/run/example.json",
                "LastModified": datetime(
                    2026,
                    7,
                    26,
                    12,
                    0,
                    10,
                    tzinfo=UTC,
                ),
                "Size": 123,
            }
        ],
        "ResponseMetadata": {"HTTPStatusCode": 200},
    }


def test_default_imds_opener_rejects_redirected_endpoint(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers: list[object] = []

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        @staticmethod
        def geturl() -> str:
            return "http://attacker.invalid/identity"

        @staticmethod
        def read() -> bytes:
            pytest.fail("redirected IMDS response must not be read")

    class Opener:
        @staticmethod
        def open(request: Request, *, timeout: float) -> Response:
            assert request.full_url.startswith("http://169.254.169.254/")
            assert timeout == 2.0
            return Response()

    def build_opener(*values: object) -> Opener:
        handlers.extend(values)
        return Opener()

    monkeypatch.setattr(module.urllib.request, "build_opener", build_opener)
    request = Request(
        "http://169.254.169.254/latest/dynamic/instance-identity/document"
    )
    with pytest.raises(
        module.WorkerStartPublisherError,
        match="endpoint drifted",
    ):
        module._default_imds_opener(request, 2.0)
    assert any(isinstance(value, module._NoRedirect) for value in handlers)


def test_mounted_native_policies_execute_before_repository_and_reject_substitution(
    tmp_path: Path,
) -> None:
    """The pre-repository publisher accepts only its mounted native siblings."""

    python = Path("/usr/bin/python3")
    if not python.is_file():
        pytest.skip("system Python is unavailable")
    mount = tmp_path / "glm52-worker-start-v2"
    mount.mkdir()
    mounted = {
        "publisher": "publish_worker_start_v2.py",
        "campaign": "glm52_sky_campaign.py",
        "campaign_native": "glm52_sky_campaign_native.py",
        "must_start": "glm52_sky_must_start.py",
        "must_start_native": "glm52_sky_must_start_native.py",
        "dynamic": "glm52_sky_must_start_dynamic.py",
        "worker": "glm52_sky_worker_must_start_v2.py",
        "coordinator": "sky_worker_start_v2_coordinator.py",
    }
    source_paths = {"publisher": MODULE_PATH, **SOURCE_PATHS}
    for name, filename in mounted.items():
        shutil.copyfile(source_paths[name], mount / filename)

    bootstrap = r'''
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
expected = json.loads(sys.argv[2])
spec = importlib.util.spec_from_file_location(
    "mounted_publisher", root / "publish_worker_start_v2.py"
)
assert spec is not None and spec.loader is not None
publisher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = publisher
spec.loader.exec_module(publisher)
sources = publisher.WorkerStartSourceBundle(
    publisher_file_sha256=hashlib.sha256(
        (root / "publish_worker_start_v2.py").read_bytes()
    ).hexdigest(),
    campaign_policy_path=root / "glm52_sky_campaign.py",
    campaign_policy_file_sha256=expected["campaign"],
    campaign_policy_native_path=root / "glm52_sky_campaign_native.py",
    campaign_policy_native_file_sha256=expected["campaign_native"],
    must_start_policy_path=root / "glm52_sky_must_start.py",
    must_start_policy_file_sha256=expected["must_start"],
    must_start_policy_native_path=root / "glm52_sky_must_start_native.py",
    must_start_policy_native_file_sha256=expected["must_start_native"],
    dynamic_policy_path=root / "glm52_sky_must_start_dynamic.py",
    dynamic_policy_file_sha256=expected["dynamic"],
    worker_policy_path=root / "glm52_sky_worker_must_start_v2.py",
    worker_policy_file_sha256=expected["worker"],
    coordinator_path=root / "sky_worker_start_v2_coordinator.py",
    coordinator_file_sha256=expected["coordinator"],
)
loaded = publisher._validate_source_bundle(sources)
now = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)
artifacts = {
    "source_snapshot_prefix": "source-snapshot/",
    "source_snapshot_sha256": "1" * 64,
    "non_vq_prefix": "non-vq-package/",
    "non_vq_package_sha256": "2" * 64,
    "teich_pack_key": "teich-pack/pack.json",
    "teich_pack_sha256": "3" * 64,
    "frozen_prompt_pack_key": "quality/frozen.json",
    "frozen_prompt_pack_sha256": "4" * 64,
    "training_baseline_prefix": "training-baseline/",
    "training_baseline_sha256": "5" * 64,
    "training_config_key": "campaigns/mounted/authorities/training.json",
    "training_config_sha256": "6" * 64,
    "artifact_inventory_key": (
        "campaigns/mounted/inventories/artifact-inventory-" + "7" * 64 + ".json"
    ),
    "artifact_inventory_sha256": "7" * 64,
    "qualification_cache_prefix": "qualification-cache/",
    "qualification_cache_manifest_sha256": "0" * 64,
}
descriptor = loaded.campaign.build_sky_campaign_descriptor(
    run_id="mounted",
    must_start_by=now,
    controller_identity="arn:aws:iam::246813579024:role/controller",
    worker_identity="arn:aws:iam::246813579024:role/worker",
    vpc_name="vpc-mounted",
    image_id="ami-0123456789abcdef0",
    bucket="mounted-bucket",
    jobs_bucket="mounted-bucket",
    repo_tar_key="campaigns/mounted/repository/repo.tar.gz",
    repo_tar_sha256="9" * 64,
    campaign_descriptor_key="campaigns/mounted/submissions/descriptor.json",
    approval_key="campaigns/mounted/authorities/approval.json",
    approval_sha256="a" * 64,
    artifacts=artifacts,
)
assert loaded.campaign.validate_sky_campaign_descriptor(descriptor) == descriptor
observation = loaded.must_start.build_must_start_controller_observation(
    run_id="mounted",
    managed_mode="qualification",
    account_id="246813579024",
    region="us-west-2",
    bucket="mounted-bucket",
    descriptor_body_sha256=descriptor["descriptor_body_sha256"],
    submission_body_sha256="b" * 64,
    sky_job_name="mounted-qualification",
    must_start_by=now,
    target_job_id=1,
    workspace="default",
    controller_instance_id="i-0123456789abcdef0",
    controller_instance_type="m5.large",
    controller_profile_arn=(
        "arn:aws:iam::246813579024:instance-profile/controller"
    ),
    controller_cluster_name="sky-jobs-controller-mounted",
    status="PENDING",
    schedule_state="PENDING",
    submitted_at=now,
    start_at=None,
    worker_cluster_name=None,
    recovery_count=0,
    observed_at=now,
)
assert loaded.must_start.validate_must_start_controller_observation(
    observation
) == observation
print("mounted-policy-ok")
'''
    result = subprocess.run(
        [str(python), "-I", "-c", bootstrap, str(mount), json.dumps(FROZEN_HASHES)],
        cwd=mount,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "mounted-policy-ok"

    (mount / mounted["campaign_native"]).write_bytes(b"# substituted native\n")
    substituted = subprocess.run(
        [str(python), "-I", "-c", bootstrap, str(mount), json.dumps(FROZEN_HASHES)],
        cwd=mount,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert substituted.returncode != 0
    assert "glm52_sky_campaign_native source hash drifted" in substituted.stderr
