from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256

ACCOUNT = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
RUN_ID = "glm52-sky-20260724"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
ACTIVATION = "approved-20260731"
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "aws/glm52-gpu/scripts/"
    "materialize_glm52_task13_fixed_artifacts.py"
)
STAGER_SCRIPT = (
    ROOT
    / "aws/glm52-gpu/scripts/"
    "stage_sky_campaign_bundle.py"
)


def _metadata(label: str) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RequestId": label,
        "RetryAttempts": 0,
    }


class _LostPut(RuntimeError):
    pass


class ExactS3:
    def __init__(
        self,
        *,
        lose_after_put_key: str | None = None,
        page_size: int = 1000,
    ) -> None:
        self.lose_after_put_key = lose_after_put_key
        self.page_size = page_size
        self.versions: dict[str, list[dict[str, object]]] = {}
        self.put_requests: list[dict[str, object]] = []
        self.get_requests: list[dict[str, object]] = []
        self.list_requests: list[dict[str, object]] = []
        self._version = 0

    def seed(
        self,
        key: str,
        raw: bytes,
        *,
        metadata: dict[str, str] | None = None,
    ) -> str:
        self._version += 1
        version_id = f"version-{self._version:04d}"
        self.versions.setdefault(key, []).append(
            {
                "VersionId": version_id,
                "raw": raw,
                "Metadata": dict(metadata or {}),
            }
        )
        return version_id

    def get_bucket_versioning(self, **request: object) -> object:
        assert request == {
            "Bucket": BUCKET,
            "ExpectedBucketOwner": ACCOUNT,
        }
        return {
            "Status": "Enabled",
            "ResponseMetadata": _metadata("versioning"),
        }

    def list_object_versions(self, **request: object) -> object:
        self.list_requests.append(dict(request))
        assert request["Bucket"] == BUCKET
        assert request["ExpectedBucketOwner"] == ACCOUNT
        prefix = str(request["Prefix"])
        flattened = [
            {
                "Key": key,
                "VersionId": row["VersionId"],
                "IsLatest": index == len(rows) - 1,
                "Size": len(row["raw"]),
            }
            for key, rows in sorted(self.versions.items())
            if key.startswith(prefix)
            for index, row in enumerate(rows)
        ]
        start = int(str(request.get("KeyMarker", "0")))
        page = flattened[start : start + self.page_size]
        next_index = start + len(page)
        truncated = next_index < len(flattened)
        return {
            "Versions": page,
            "DeleteMarkers": [],
            "IsTruncated": truncated,
            **(
                {
                    "NextKeyMarker": str(next_index),
                    "NextVersionIdMarker": "opaque-" + str(next_index),
                }
                if truncated
                else {}
            ),
            "ResponseMetadata": _metadata("list-" + str(start)),
        }

    def put_object(self, **request: object) -> object:
        self.put_requests.append(dict(request))
        assert request["Bucket"] == BUCKET
        assert request["ExpectedBucketOwner"] == ACCOUNT
        assert request["IfNoneMatch"] == "*"
        assert request["ChecksumAlgorithm"] == "SHA256"
        raw = request["Body"]
        assert type(raw) is bytes
        assert request["ChecksumSHA256"] == base64.b64encode(
            hashlib.sha256(raw).digest()
        ).decode("ascii")
        key = str(request["Key"])
        if self.versions.get(key):
            raise AssertionError("test boundary received an overwrite")
        version_id = self.seed(
            key,
            raw,
            metadata=dict(request["Metadata"]),
        )
        if key == self.lose_after_put_key:
            self.lose_after_put_key = None
            raise _LostPut("simulated lost PutObject response")
        return {
            "VersionId": version_id,
            "ChecksumSHA256": request["ChecksumSHA256"],
            "ResponseMetadata": _metadata("put-" + version_id),
        }

    def get_object(self, **request: object) -> object:
        self.get_requests.append(dict(request))
        assert request["Bucket"] == BUCKET
        assert request["ExpectedBucketOwner"] == ACCOUNT
        assert request["ChecksumMode"] == "ENABLED"
        key = str(request["Key"])
        version_id = str(request["VersionId"])
        row = next(
            item
            for item in self.versions[key]
            if item["VersionId"] == version_id
        )
        raw = row["raw"]
        assert type(raw) is bytes
        return {
            "Body": io.BytesIO(raw),
            "ContentLength": len(raw),
            "VersionId": version_id,
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(raw).digest()
            ).decode("ascii"),
            "Metadata": row["Metadata"],
            "ResponseMetadata": _metadata("get-" + version_id),
        }

    def current_raw(self, key: str) -> bytes:
        raw = self.versions[key][-1]["raw"]
        assert type(raw) is bytes
        return raw


class InteropS3(ExactS3):
    def head_object(self, key: str) -> dict[str, object] | None:
        rows = self.versions.get(key)
        if not rows:
            return None
        row = rows[-1]
        raw = row["raw"]
        assert type(raw) is bytes
        return {
            "ContentLength": len(raw),
            "VersionId": row["VersionId"],
        }

    def get_object(self, **request: object) -> object:
        destination = request.get("destination")
        if isinstance(destination, Path):
            key = str(request["key"])
            version_id = str(request["version_id"])
            row = next(
                item
                for item in self.versions[key]
                if item["VersionId"] == version_id
            )
            raw = row["raw"]
            assert type(raw) is bytes
            destination.write_bytes(raw)
            return {
                "VersionId": version_id,
                "Metadata": row["Metadata"],
            }
        return super().get_object(**request)

    def put_object(self, **request: object) -> object:
        path = request.get("path")
        if isinstance(path, Path):
            key = str(request["key"])
            if self.versions.get(key):
                return subprocess.CompletedProcess(
                    ["fake-aws"],
                    1,
                    "",
                    "412 PreconditionFailed",
                )
            version_id = self.seed(
                key,
                path.read_bytes(),
                metadata=dict(request["metadata"]),
            )
            return subprocess.CompletedProcess(
                ["fake-aws"],
                0,
                json.dumps({"VersionId": version_id}),
                "",
            )
        return super().put_object(**request)


def _load_bundle_stager():
    module_name = "_task13_bundle_stager_under_test"
    spec = importlib.util.spec_from_file_location(module_name, STAGER_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ExactSts:
    def get_caller_identity(self) -> object:
        return {
            "Account": ACCOUNT,
            "Arn": (
                "arn:aws:sts::246813579024:"
                "assumed-role/task13-fixed-artifacts/test"
            ),
            "UserId": "AROATASK13:test",
            "ResponseMetadata": _metadata("sts"),
        }


def _services(s3: ExactS3):
    from glm52_enforcement.task13_fixed_artifacts import (
        Task13FixedArtifactServices,
    )

    return Task13FixedArtifactServices(
        sts=ExactSts(),
        s3=s3,
        total_max_attempts=1,
    )


def _canonical_file(path: Path, value: object) -> dict[str, object]:
    raw = canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return {
        "path": str(path),
        "size_bytes": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _driver_request(
    *,
    operation_kind: str,
    argv: list[str],
    environment: dict[str, str],
    sources: list[dict[str, object]],
) -> dict[str, object]:
    body = {
        "schema_version": 2,
        "record_type": (
            "glm52_task13_driver_materialization_request_v2"
        ),
        "account_id": ACCOUNT,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "operation_kind": operation_kind,
        "argv": argv,
        "environment": environment,
        "source_coordinates": sources,
    }
    return {**body, "canonical_identity_sha256": canonical_sha256(body)}

def test_mutable_reviewed_artifact_keys_are_activation_scoped() -> None:
    from glm52_enforcement.task13_fixed_artifacts import (
        activation_artifact_key,
        validate_activation_artifact_key,
    )

    expected = {
        "QUALIFICATION_CACHE_SEED_INPUT": (
            f"task13/activations/{ACTIVATION}/"
            "qualification/cache-seed-input.json"
        ),
        "H100_QUALIFICATION_INPUT": (
            f"task13/activations/{ACTIVATION}/"
            "qualification/h100-input.json"
        ),
        "PRODUCTION_DESCRIPTOR": (
            f"task13/activations/{ACTIVATION}/"
            "inputs/campaign-descriptor-v2.json"
        ),
    }
    for kind, key in expected.items():
        assert activation_artifact_key(kind, ACTIVATION) == key
        assert validate_activation_artifact_key(
            key,
            artifact_kind=kind,
            expected_activation_id=ACTIVATION,
        ) == key
        with pytest.raises(ValueError, match="lineage"):
            validate_activation_artifact_key(
                key,
                artifact_kind=kind,
                expected_activation_id="approved-20260730",
            )


def _make_repository_tar(path: Path) -> bytes:
    payload = path.parent / "payload.txt"
    payload.write_text("exact immutable repository payload\n", encoding="ascii")
    with tarfile.open(path, "w:gz") as archive:
        archive.add(payload, arcname="src/payload.txt")
    return path.read_bytes()


def test_repository_archive_manifest_binds_one_uploaded_version(
    tmp_path: Path,
) -> None:
    """Break caught: fixed archive metadata is detached from uploaded bytes."""

    from glm52_enforcement.task13_fixed_artifacts import (
        publish_repository_archive,
        validate_repository_archive_payload,
    )

    archive = tmp_path / "repo.tar.gz"
    archive_raw = _make_repository_tar(archive)
    archive_sha = hashlib.sha256(archive_raw).hexdigest()
    assert validate_repository_archive_payload(archive) == (
        archive_raw,
        archive_sha,
    )
    s3 = ExactS3(page_size=1)
    manifest_key = f"task13/activations/{ACTIVATION}/archive/repo-tar.json"
    s3.seed(manifest_key + ".sibling", b"sibling")
    s3.seed(manifest_key + ".sibling-2", b"sibling-2")

    coordinate = publish_repository_archive(
        activation_id=ACTIVATION,
        archive_path=archive,
        bucket=BUCKET,
        services=_services(s3),
    )

    assert coordinate == {
        "artifact_kind": "REPOSITORY_ARCHIVE",
        "bucket": BUCKET,
        "key": manifest_key,
        "version_id": "version-0004",
        "file_sha256": coordinate["file_sha256"],
        "body_sha256": coordinate["body_sha256"],
    }
    manifest_raw = s3.current_raw(manifest_key)
    manifest = json.loads(manifest_raw)
    assert manifest["record_type"] == (
        "glm52_task13_repository_archive_manifest_v2"
    )
    assert manifest["activation_id"] == ACTIVATION
    assert manifest["archive_file_sha256"] == archive_sha
    assert manifest["archive_size_bytes"] == len(archive_raw)
    assert manifest["archive"] == {
        "bucket": BUCKET,
        "key": (
            f"campaigns/{RUN_ID}/repository/keep-{archive_sha}.tar.gz"
        ),
        "version_id": "version-0003",
        "file_sha256": archive_sha,
    }
    unsigned = dict(manifest)
    identity = unsigned.pop("canonical_identity_sha256")
    assert identity == hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    assert manifest_raw == canonical_json_bytes(manifest) + b"\n"
    assert coordinate["file_sha256"] == hashlib.sha256(
        manifest_raw
    ).hexdigest()
    assert coordinate["body_sha256"] == identity
    assert len(s3.list_requests) > 2
    archive_put = next(
        request
        for request in s3.put_requests
        if request["Key"] == manifest["archive"]["key"]
    )
    assert archive_put["ContentType"] == "application/gzip"
    manifest_put = next(
        request
        for request in s3.put_requests
        if request["Key"] == manifest_key
    )
    assert manifest_put["ContentType"] == "application/json"
    fixed_key_lists = [
        request
        for request in s3.list_requests
        if request["Prefix"] == manifest_key
    ]
    assert len(fixed_key_lists) == 5



@pytest.mark.parametrize("first_producer", ("stager", "task13"))
def test_repository_archive_adopts_one_version_in_either_producer_order(
    tmp_path: Path,
    first_producer: str,
) -> None:
    from glm52_enforcement.task13_fixed_artifacts import (
        publish_repository_archive,
    )

    stager = _load_bundle_stager()
    archive = tmp_path / "repo.tar.gz"
    archive_raw = _make_repository_tar(archive)
    archive_sha = hashlib.sha256(archive_raw).hexdigest()
    archive_key = (
        f"campaigns/{RUN_ID}/repository/keep-{archive_sha}.tar.gz"
    )
    s3 = InteropS3()

    if first_producer == "stager":
        staged = stager._put_immutable(
            profile="",
            region="",
            bucket=BUCKET,
            key=archive_key,
            path=archive,
            run_id=RUN_ID,
            record_type="glm52_task13_repository_archive_payload_v1",
            s3=s3,
        )
        assert staged.disposition == "uploaded"

    coordinate = publish_repository_archive(
        activation_id=ACTIVATION,
        archive_path=archive,
        bucket=BUCKET,
        services=_services(s3),
    )

    if first_producer == "task13":
        staged = stager._put_immutable(
            profile="",
            region="",
            bucket=BUCKET,
            key=archive_key,
            path=archive,
            run_id=RUN_ID,
            record_type="glm52_task13_repository_archive_payload_v1",
            s3=s3,
        )
        assert staged.disposition == "reused"

    assert len(s3.versions[archive_key]) == 1
    archive_row = s3.versions[archive_key][0]
    assert archive_row["Metadata"] == {
        "glm52-run-id": RUN_ID,
        "record-type": "glm52_task13_repository_archive_payload_v1",
        "file-sha256": archive_sha,
    }
    manifest = json.loads(s3.current_raw(str(coordinate["key"])))
    assert manifest["archive"]["version_id"] == archive_row["VersionId"]

def test_repository_archive_preserves_prior_activation_manifest(
    tmp_path: Path,
) -> None:
    from glm52_enforcement.task13_fixed_artifacts import (
        publish_repository_archive,
    )

    archive = tmp_path / "repo.tar.gz"
    _make_repository_tar(archive)
    s3 = ExactS3()
    prior_key = "task13/archive/repo-tar.json"
    prior = b'{"record_type":"prior-archive"}\n'
    s3.seed(
        prior_key,
        prior,
        metadata={
            "glm52-run-id": RUN_ID,
            "record-type": "glm52_task13_repository_archive_manifest_v1",
            "file-sha256": hashlib.sha256(prior).hexdigest(),
        },
    )

    coordinate = publish_repository_archive(
        activation_id=ACTIVATION,
        archive_path=archive,
        bucket=BUCKET,
        services=_services(s3),
    )

    new_key = f"task13/activations/{ACTIVATION}/archive/repo-tar.json"
    assert s3.current_raw(prior_key) == prior
    assert len(s3.versions[prior_key]) == 1
    assert coordinate["key"] == new_key
    assert len(s3.versions[new_key]) == 1


@pytest.mark.parametrize(
    ("operation_kind", "artifact_kind", "fixed_key"),
    [
        (
            "qualification-cache-seed",
            "QUALIFICATION_CACHE_SEED_INPUT",
            (
                f"task13/activations/{ACTIVATION}/"
                "qualification/cache-seed-input.json"
            ),
        ),
        (
            "h100-qualification",
            "H100_QUALIFICATION_INPUT",
            (
                f"task13/activations/{ACTIVATION}/"
                "qualification/h100-input.json"
            ),
        ),
    ],
)
def test_driver_builder_requires_concrete_sources_and_exact_fixed_key(
    tmp_path: Path,
    operation_kind: str,
    artifact_kind: str,
    fixed_key: str,
) -> None:
    """Break caught: reviewed driver input invents a path or publishes elsewhere."""

    from glm52_enforcement.task13_fixed_artifacts import (
        publish_repository_driver,
    )

    descriptor = tmp_path / "campaign-descriptor-v2.json"
    sources = [
        {
            "role": "campaign_descriptor",
            **_canonical_file(
                descriptor,
                {"record_type": "fixture-campaign-descriptor"},
            ),
        }
    ]
    if operation_kind == "qualification-cache-seed":
        flags = []
        sky_bin = tmp_path / "sky"
        sky_bin.write_text("#!/bin/sh\nexit 64\n", encoding="ascii")
        sky_bin.chmod(0o700)
        for role, flag in (
            ("approval", "--approval"),
            ("staged_ready", "--staged-ready"),
            ("rehearsal_evidence", "--rehearsal-evidence"),
            ("task", "--task"),
            ("config", "--config"),
        ):
            path = tmp_path / (role + ".json")
            sources.append(
                {
                    "role": role,
                    **_canonical_file(
                        path,
                        {"record_type": "fixture-" + role},
                    ),
                }
            )
            flags.extend([flag, str(path)])
        argv = [
            "aws/glm52-gpu/scripts/submit_sky_campaign.py",
            "cache-seed",
            "acquire-and-launch",
            "--profile",
            PROFILE,
            "--descriptor",
            str(descriptor),
            *flags,
            "--staged-ready-version-id",
            "opaque-version-1",
            "--sky-bin",
            str(sky_bin),
        ]
        environment = {
            "AWS_PROFILE": PROFILE,
            "AWS_REGION": REGION,
        }
    else:
        argv = [
            "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh"
        ]
        environment = {
            "AWS_PROFILE": PROFILE,
            "AWS_REGION": REGION,
            "CAMPAIGN_DESCRIPTOR": str(descriptor),
        }
    request = _driver_request(
        operation_kind=operation_kind,
        argv=argv,
        environment=environment,
        sources=sources,
    )
    s3 = ExactS3()

    coordinate = publish_repository_driver(
        request=request,
        bucket=BUCKET,
        services=_services(s3),
    )

    assert coordinate["artifact_kind"] == artifact_kind
    assert coordinate["key"] == fixed_key
    value = json.loads(s3.current_raw(fixed_key))
    assert set(value) == {
        "schema_version",
        "record_type",
        "activation_id",
        "operation_kind",
        "argv",
        "environment",
        "canonical_identity_sha256",
    }
    assert value["record_type"] == "glm52_task13_repository_driver_v2"
    assert value["activation_id"] == ACTIVATION
    assert value["operation_kind"] == operation_kind
    assert value["argv"] == argv
    assert value["environment"] == environment
    assert [
        request["Prefix"] for request in s3.list_requests
    ] == [fixed_key, fixed_key]
    assert s3.get_requests == [
        {
            "Bucket": BUCKET,
            "Key": fixed_key,
            "VersionId": coordinate["version_id"],
            "ExpectedBucketOwner": ACCOUNT,
            "ChecksumMode": "ENABLED",
        }
    ]

    descriptor.write_bytes(b'{"record_type":"mutated"}\n')
    with pytest.raises(ValueError, match="source"):
        publish_repository_driver(
            request=request,
            bucket=BUCKET,
            services=_services(ExactS3()),
        )


def test_lost_put_adopts_only_singular_byte_identical_fixed_truth(
    tmp_path: Path,
) -> None:
    """Break caught: ambiguous PutObject is retried or foreign history is adopted."""

    from glm52_enforcement.task13_fixed_artifacts import (
        publish_repository_driver,
    )

    descriptor = tmp_path / "descriptor.json"
    source = {
        "role": "campaign_descriptor",
        **_canonical_file(
            descriptor,
            {"record_type": "fixture-campaign-descriptor"},
        ),
    }
    request = _driver_request(
        operation_kind="h100-qualification",
        argv=[
            "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh"
        ],
        environment={
            "AWS_PROFILE": PROFILE,
            "AWS_REGION": REGION,
            "CAMPAIGN_DESCRIPTOR": str(descriptor),
        },
        sources=[source],
    )
    key = (
        f"task13/activations/{ACTIVATION}/qualification/h100-input.json"
    )
    s3 = ExactS3(lose_after_put_key=key)

    first = publish_repository_driver(
        request=request,
        bucket=BUCKET,
        services=_services(s3),
    )
    puts_after_lost_response = len(s3.put_requests)
    second = publish_repository_driver(
        request=request,
        bucket=BUCKET,
        services=_services(s3),
    )

    assert first == second
    assert len(s3.put_requests) == puts_after_lost_response
    s3.seed(key, s3.current_raw(key))
    with pytest.raises(ValueError, match="singular"):
        publish_repository_driver(
            request=request,
            bucket=BUCKET,
            services=_services(s3),
        )


def test_fixed_publish_rejects_foreign_bytes_and_transport_drift(
    tmp_path: Path,
) -> None:
    """Break caught: foreign fixed truth or retry-enabled transport is accepted."""

    from glm52_enforcement.task13_fixed_artifacts import (
        Task13FixedArtifactServices,
        publish_repository_driver,
    )

    descriptor = tmp_path / "descriptor.json"
    source = {
        "role": "campaign_descriptor",
        **_canonical_file(
            descriptor,
            {"record_type": "fixture-campaign-descriptor"},
        ),
    }
    request = _driver_request(
        operation_kind="h100-qualification",
        argv=[
            "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh"
        ],
        environment={
            "AWS_PROFILE": PROFILE,
            "AWS_REGION": REGION,
            "CAMPAIGN_DESCRIPTOR": str(descriptor),
        },
        sources=[source],
    )
    key = (
        f"task13/activations/{ACTIVATION}/qualification/h100-input.json"
    )
    foreign_s3 = ExactS3()
    foreign_s3.seed(
        key,
        b'{"record_type":"foreign"}\n',
        metadata={"record-type": "foreign"},
    )
    with pytest.raises(ValueError, match="foreign|drifted"):
        publish_repository_driver(
            request=request,
            bucket=BUCKET,
            services=_services(foreign_s3),
        )
    assert foreign_s3.put_requests == []

    retry_enabled_s3 = ExactS3()
    with pytest.raises(ValueError, match="transport"):
        publish_repository_driver(
            request=request,
            bucket=BUCKET,
            services=Task13FixedArtifactServices(
                sts=ExactSts(),
                s3=retry_enabled_s3,
                total_max_attempts=2,
            ),
        )
    assert retry_enabled_s3.list_requests == []
    assert retry_enabled_s3.put_requests == []


def test_public_validators_reject_foreign_archive_and_driver_truth(
    tmp_path: Path,
) -> None:
    """Break caught: builders emit records no independent reader can authenticate."""

    from glm52_enforcement.task13_fixed_artifacts import (
        build_repository_driver_manifest,
        validate_repository_archive_manifest,
        validate_repository_driver_manifest,
    )

    archive = {
        "schema_version": 2,
        "record_type": "glm52_task13_repository_archive_manifest_v2",
        "account_id": ACCOUNT,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "archive_file_sha256": "a" * 64,
        "archive_size_bytes": 123,
        "archive": {
            "bucket": BUCKET,
            "key": (
                f"campaigns/{RUN_ID}/repository/keep-"
                + "a" * 64
                + ".tar.gz"
            ),
            "version_id": "opaque-version-1",
            "file_sha256": "a" * 64,
        },
    }
    archive["canonical_identity_sha256"] = canonical_sha256(archive)
    assert validate_repository_archive_manifest(
        archive,
        expected_activation_id=ACTIVATION,
        expected_archive_sha256="a" * 64,
        expected_archive_size=123,
        expected_bucket=BUCKET,
    ) == archive
    foreign = json.loads(json.dumps(archive))
    foreign["archive"]["version_id"] = "null"
    with pytest.raises(ValueError):
        validate_repository_archive_manifest(
            foreign,
            expected_activation_id=ACTIVATION,
            expected_archive_sha256="a" * 64,
            expected_archive_size=123,
            expected_bucket=BUCKET,
        )

    descriptor = tmp_path / "descriptor.json"
    source = {
        "role": "campaign_descriptor",
        **_canonical_file(
            descriptor,
            {"record_type": "fixture-campaign-descriptor"},
        ),
    }
    request = _driver_request(
        operation_kind="h100-qualification",
        argv=[
            "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh"
        ],
        environment={
            "AWS_PROFILE": PROFILE,
            "AWS_REGION": REGION,
            "CAMPAIGN_DESCRIPTOR": str(descriptor),
        },
        sources=[source],
    )
    driver = build_repository_driver_manifest(request)
    assert validate_repository_driver_manifest(
        driver,
        expected_operation_kind="h100-qualification",
        expected_activation_id=ACTIVATION,
    ) == driver
    driver["invented"] = True
    with pytest.raises(ValueError):
        validate_repository_driver_manifest(
            driver,
            expected_operation_kind="h100-qualification",
            expected_activation_id=ACTIVATION,
        )


def _support_build_inputs_fixture() -> dict[str, object]:
    module_path = ROOT / "tests/test_glm52_enforcement_support_plane.py"
    spec = importlib.util.spec_from_file_location(
        "_task13_support_input_fixture",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = module._support_build_input_mapping()
    assert type(value) is dict
    return value


def test_support_build_inputs_publish_at_the_only_reviewed_key(
    tmp_path: Path,
) -> None:
    """Break caught: build-time support facts bypass their exact parser or key."""

    from glm52_enforcement.task13_fixed_artifacts import (
        publish_support_build_inputs,
    )

    path = tmp_path / "support-build-inputs.json"
    value = _support_build_inputs_fixture()
    assert value["schema_version"] == 2
    assert value["record_type"] == "glm52_h1g_support_build_inputs_v2"
    assert "fence_manifest_coordinate" not in value
    manifest_coordinate = value["bootstrap_manifest_coordinate"]
    assert type(manifest_coordinate) is dict
    assert manifest_coordinate["key"] == (
        "campaigns/glm52-sky-20260724/authorities/fence/manifests/"
        "glm52-v2-amber-quartz/00000001/FENCE_BOOTSTRAP_MANIFEST.json"
    )
    inventory = value["fence_template_inventory"]
    assert type(inventory) is list
    prepare = next(
        entry
        for entry in inventory
        if entry["slot"] == "PREPARE_GENESIS_LIVE_STATE"
    )
    assert value["prepare_entry_identity_sha256"] == (
        prepare["entry_identity_sha256"]
    )
    assert value["prepare_template_body_sha256"] == (
        prepare["template_body_sha256"]
    )
    assert value["prepare_policy_sha256"] == prepare["policy_sha256"]
    raw = canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    s3 = ExactS3()

    coordinate = publish_support_build_inputs(
        inputs_path=path,
        bucket=BUCKET,
        services=_services(s3),
    )

    assert coordinate == {
        "artifact_kind": "SUPPORT_INPUTS",
        "bucket": BUCKET,
        "key": "task13/inputs/support-build-inputs.json",
        "version_id": "version-0001",
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": hashlib.sha256(raw[:-1]).hexdigest(),
    }
    assert s3.current_raw(coordinate["key"]) == raw
    invented = dict(value)
    invented["invented_physical_id"] = "i-not-authenticated"
    path.unlink()
    path.write_bytes(canonical_json_bytes(invented) + b"\n")
    with pytest.raises(ValueError):
        publish_support_build_inputs(
            inputs_path=path,
            bucket=BUCKET,
            services=_services(ExactS3()),
        )


def _load_fixed_artifact_cli():
    spec = importlib.util.spec_from_file_location(
        "_task13_fixed_artifact_cli_under_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def _archive_cli_argv(
    archive: Path,
    coordinate_output: Path,
    manifest_output: Path,
) -> list[str]:
    return [
        "publish-archive",
        "--activation-id",
        ACTIVATION,
        "--archive",
        str(archive),
        "--bucket",
        BUCKET,
        "--coordinate-output",
        str(coordinate_output),
        "--manifest-output",
        str(manifest_output),
    ]


def test_cli_publishes_archive_under_exact_activation(
    tmp_path: Path,
) -> None:
    cli = _load_fixed_artifact_cli()
    archive = tmp_path / "repo.tar.gz"
    _make_repository_tar(archive)
    coordinate_output = tmp_path / "archive-coordinate.json"
    manifest_output = tmp_path / "archive-manifest.json"
    s3 = ExactS3()
    services = _services(s3)

    assert (
        cli.main(
            [
                "publish-archive",
                "--activation-id",
                ACTIVATION,
                "--archive",
                str(archive),
                "--bucket",
                BUCKET,
                "--coordinate-output",
                str(coordinate_output),
                "--manifest-output",
                str(manifest_output),
            ],
            services_factory=lambda: services,
        )
        == 0
    )
    coordinate_raw = coordinate_output.read_bytes()
    coordinate = json.loads(coordinate_raw)
    manifest_raw = manifest_output.read_bytes()
    manifest = json.loads(manifest_raw)
    assert coordinate_raw == canonical_json_bytes(coordinate) + b"\n"
    assert coordinate["key"] == (
        f"task13/activations/{ACTIVATION}/archive/repo-tar.json"
    )
    assert manifest_raw == s3.current_raw(coordinate["key"])
    assert hashlib.sha256(manifest_raw).hexdigest() == coordinate["file_sha256"]
    assert manifest["canonical_identity_sha256"] == coordinate["body_sha256"]
    assert coordinate_output.stat().st_mode & 0o777 == 0o600
    assert manifest_output.stat().st_mode & 0o777 == 0o600


def test_cli_writes_one_coordinate_without_overwrite(
    tmp_path: Path,
) -> None:
    """Break caught: the finite CLI can overwrite or bypass typed transport."""

    spec = importlib.util.spec_from_file_location(
        "_task13_fixed_artifact_cli",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    descriptor = tmp_path / "descriptor.json"
    source = {
        "role": "campaign_descriptor",
        **_canonical_file(
            descriptor,
            {"record_type": "fixture-campaign-descriptor"},
        ),
    }
    request = _driver_request(
        operation_kind="h100-qualification",
        argv=[
            "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh"
        ],
        environment={
            "AWS_PROFILE": PROFILE,
            "AWS_REGION": REGION,
            "CAMPAIGN_DESCRIPTOR": str(descriptor),
        },
        sources=[source],
    )
    request_path = tmp_path / "driver-request.json"
    request_path.write_bytes(canonical_json_bytes(request) + b"\n")
    output = tmp_path / "coordinate.json"
    services = _services(ExactS3())
    argv = [
        "publish-driver",
        "--request",
        str(request_path),
        "--bucket",
        BUCKET,
        "--coordinate-output",
        str(output),
    ]

    assert cli.main(argv, services_factory=lambda: services) == 0
    coordinate = json.loads(output.read_bytes())
    assert coordinate["artifact_kind"] == "H100_QUALIFICATION_INPUT"
    assert output.stat().st_mode & 0o777 == 0o600
    before = output.read_bytes()
    assert cli.main(argv, services_factory=lambda: services) == 64
    assert output.read_bytes() == before


def test_cli_archive_local_retry_adopts_the_same_exact_manifest(
    tmp_path: Path,
) -> None:
    cli = _load_fixed_artifact_cli()
    archive = tmp_path / "repo.tar.gz"
    _make_repository_tar(archive)
    s3 = ExactS3()
    services = _services(s3)
    first_coordinate = tmp_path / "first-coordinate.json"
    first_manifest = tmp_path / "first-manifest.json"
    second_coordinate = tmp_path / "second-coordinate.json"
    second_manifest = tmp_path / "second-manifest.json"

    assert (
        cli.main(
            _archive_cli_argv(
                archive,
                first_coordinate,
                first_manifest,
            ),
            services_factory=lambda: services,
        )
        == 0
    )
    assert (
        cli.main(
            _archive_cli_argv(
                archive,
                second_coordinate,
                second_manifest,
            ),
            services_factory=lambda: services,
        )
        == 0
    )

    assert second_coordinate.read_bytes() == first_coordinate.read_bytes()
    assert second_manifest.read_bytes() == first_manifest.read_bytes()
    assert len(s3.put_requests) == 2


@pytest.mark.parametrize(
    "preexisting_name",
    ["coordinate", "manifest"],
)
def test_cli_archive_preserves_preexisting_local_outputs_before_publication(
    tmp_path: Path,
    preexisting_name: str,
) -> None:
    cli = _load_fixed_artifact_cli()
    archive = tmp_path / "repo.tar.gz"
    _make_repository_tar(archive)
    coordinate_output = tmp_path / "coordinate.json"
    manifest_output = tmp_path / "manifest.json"
    preexisting = (
        coordinate_output
        if preexisting_name == "coordinate"
        else manifest_output
    )
    other = (
        manifest_output
        if preexisting_name == "coordinate"
        else coordinate_output
    )
    before = b"preexisting-local-authority\n"
    preexisting.write_bytes(before)
    s3 = ExactS3()
    factory_calls = 0

    def services_factory():
        nonlocal factory_calls
        factory_calls += 1
        return _services(s3)

    assert (
        cli.main(
            _archive_cli_argv(
                archive,
                coordinate_output,
                manifest_output,
            ),
            services_factory=services_factory,
        )
        == 64
    )
    assert preexisting.read_bytes() == before
    assert not other.exists()
    assert factory_calls == 0
    assert s3.put_requests == []


def test_archive_local_outputs_complete_short_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_fixed_artifact_cli()
    coordinate_output = tmp_path / "coordinate.json"
    manifest_output = tmp_path / "manifest.json"
    coordinate_raw = canonical_json_bytes({"kind": "coordinate"}) + b"\n"
    manifest_raw = canonical_json_bytes({"kind": "manifest"}) + b"\n"
    real_write = cli.os.write

    def short_write(descriptor: int, remaining: memoryview) -> int:
        count = min(3, len(remaining))
        return real_write(descriptor, remaining[:count])

    monkeypatch.setattr(cli.os, "write", short_write)
    cli._write_local_outputs_once(
        [
            (coordinate_output, coordinate_raw),
            (manifest_output, manifest_raw),
        ]
    )

    assert coordinate_output.read_bytes() == coordinate_raw
    assert manifest_output.read_bytes() == manifest_raw
    assert coordinate_output.stat().st_mode & 0o777 == 0o600
    assert manifest_output.stat().st_mode & 0o777 == 0o600


def test_archive_local_outputs_reject_zero_progress_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_fixed_artifact_cli()
    coordinate_output = tmp_path / "coordinate.json"
    manifest_output = tmp_path / "manifest.json"
    monkeypatch.setattr(cli.os, "write", lambda _descriptor, _raw: 0)

    with pytest.raises(OSError, match="made no progress"):
        cli._write_local_outputs_once(
            [
                (coordinate_output, b"coordinate\n"),
                (manifest_output, b"manifest\n"),
            ]
        )

    assert not coordinate_output.exists()
    assert not manifest_output.exists()


@pytest.mark.parametrize("failure_operation", ["fsync", "close"])
def test_archive_local_outputs_cleanup_on_durability_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_operation: str,
) -> None:
    cli = _load_fixed_artifact_cli()
    coordinate_output = tmp_path / "coordinate.json"
    manifest_output = tmp_path / "manifest.json"

    if failure_operation == "fsync":
        def fail_fsync(_descriptor: int) -> None:
            raise OSError("simulated fsync failure")

        monkeypatch.setattr(cli.os, "fsync", fail_fsync)
    else:
        real_close = cli.os.close
        failed = False

        def fail_close(descriptor: int) -> None:
            nonlocal failed
            real_close(descriptor)
            if not failed:
                failed = True
                raise OSError("simulated close failure")

        monkeypatch.setattr(cli.os, "close", fail_close)

    with pytest.raises(OSError, match=f"simulated {failure_operation} failure"):
        cli._write_local_outputs_once(
            [
                (coordinate_output, b"coordinate\n"),
                (manifest_output, b"manifest\n"),
            ]
        )

    assert not coordinate_output.exists()
    assert not manifest_output.exists()


def test_archive_local_output_race_preserves_foreign_path_and_rolls_back_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_fixed_artifact_cli()
    coordinate_output = tmp_path / "coordinate.json"
    manifest_output = tmp_path / "manifest.json"
    foreign = b"won-create-only-race\n"
    real_open = cli.os.open
    raced = False

    def race_second_open(
        path: object,
        flags: int,
        mode: int = 0o777,
    ) -> int:
        nonlocal raced
        if Path(path) == manifest_output and not raced:
            raced = True
            manifest_output.write_bytes(foreign)
        return real_open(path, flags, mode)

    monkeypatch.setattr(cli.os, "open", race_second_open)
    with pytest.raises(FileExistsError):
        cli._write_local_outputs_once(
            [
                (coordinate_output, b"coordinate\n"),
                (manifest_output, b"manifest\n"),
            ]
        )

    assert not coordinate_output.exists()
    assert manifest_output.read_bytes() == foreign
