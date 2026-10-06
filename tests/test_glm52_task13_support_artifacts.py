from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_json_bytes

ACCOUNT = "246813579024"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "aws/glm52-gpu/scripts/materialize_glm52_task13_support_artifacts.py"


def _wheel(
    path: Path,
    *,
    distribution: str,
    version: str,
    files: dict[str, bytes],
):
    from glm52_enforcement.task13_support_artifacts import PinnedWheel

    with zipfile.ZipFile(path, "x") as archive:
        for name, raw in reversed(list(files.items())):
            archive.writestr(name, raw)
    raw = path.read_bytes()
    return PinnedWheel(
        distribution=distribution,
        version=version,
        filename=path.name,
        url="https://files.pythonhosted.org/packages/exact/" + path.name,
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def test_cryptography_layer_refuses_unverified_wheel_bytes(
    tmp_path: Path,
) -> None:
    """Break caught: a filename alone can substitute foreign dependency bytes."""

    from glm52_enforcement.task13_support_artifacts import (
        PinnedWheel,
        Task13SupportArtifactError,
        build_cryptography_layer_archive,
    )

    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    wheel = wheelhouse / "cryptography-49.0.0-test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("cryptography/__init__.py", b"foreign = True\n")
    pin = PinnedWheel(
        distribution="cryptography",
        version="49.0.0",
        filename=wheel.name,
        url="https://files.pythonhosted.org/test/cryptography.whl",
        size_bytes=len(wheel.read_bytes()),
        sha256="0" * 64,
    )

    with pytest.raises(
        Task13SupportArtifactError,
        match="wheel SHA-256 drifted",
    ):
        build_cryptography_layer_archive(
            wheelhouse=wheelhouse,
            output=tmp_path / "layer.zip",
            wheel_records=(pin,),
        )


def test_cryptography_layer_is_deterministic_and_lambda_rooted(
    tmp_path: Path,
) -> None:
    """Break caught: wheel metadata/order leaks into the Lambda layer archive."""

    from glm52_enforcement.task13_support_artifacts import (
        build_cryptography_layer_archive,
    )

    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    cryptography = _wheel(
        wheelhouse / "cryptography-49.0.0-py3-none-any.whl",
        distribution="cryptography",
        version="49.0.0",
        files={
            "cryptography/__init__.py": b'__version__ = "49.0.0"\n',
            "cryptography-49.0.0.dist-info/METADATA": (
                b"Name: cryptography\nVersion: 49.0.0\n"
            ),
        },
    )
    cffi = _wheel(
        wheelhouse / "cffi-2.1.0-py3-none-any.whl",
        distribution="cffi",
        version="2.1.0",
        files={
            "cffi/__init__.py": b'__version__ = "2.1.0"\n',
            "cffi-2.1.0.data/purelib/_cffi_backend.py": b"ABI = 'test'\n",
            "cffi-2.1.0.dist-info/METADATA": (b"Name: cffi\nVersion: 2.1.0\n"),
        },
    )
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_manifest = build_cryptography_layer_archive(
        wheelhouse=wheelhouse,
        output=first,
        wheel_records=(cryptography, cffi),
    )
    second_manifest = build_cryptography_layer_archive(
        wheelhouse=wheelhouse,
        output=second,
        wheel_records=(cryptography, cffi),
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest == second_manifest
    assert first_manifest["runtime"] == "python3.12"
    assert first_manifest["architecture"] == "x86_64"
    assert first_manifest["cryptography_requirement"] == "cryptography==49.0.0"
    assert (
        first_manifest["file_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    )
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert set(archive.namelist()) == {
            "python/_cffi_backend.py",
            "python/cffi-2.1.0.dist-info/METADATA",
            "python/cffi/__init__.py",
            "python/cryptography-49.0.0.dist-info/METADATA",
            "python/cryptography/__init__.py",
            "python/glm52-cryptography-layer-manifest.json",
        }
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist()
        )
        embedded = json.loads(
            archive.read("python/glm52-cryptography-layer-manifest.json")
        )
        assert (
            archive.read("python/glm52-cryptography-layer-manifest.json")
            == canonical_json_bytes(embedded) + b"\n"
        )
        assert embedded["wheels"][0]["distribution"] == "cffi"
        assert embedded["wheels"][1]["distribution"] == "cryptography"


def test_support_lambda_archive_rebuilds_existing_packager_byte_exactly(
    tmp_path: Path,
) -> None:
    """Break caught: the artifact route substitutes a second Lambda packager."""

    from glm52_enforcement.task13_support_artifacts import (
        build_support_lambda_archive,
    )

    first = tmp_path / "first-support.zip"
    second = tmp_path / "second-support.zip"

    first_manifest = build_support_lambda_archive(first)
    second_manifest = build_support_lambda_archive(second)

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest == second_manifest
    assert first_manifest == {
        "size_bytes": len(first.read_bytes()),
        "file_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
        "entry_count": first_manifest["entry_count"],
    }
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert "support_custom_resource_handler.py" in names
        assert "glm52_enforcement/support_custom_resource_handler.py" in names


def test_materialized_manifest_authenticates_both_local_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: publication can consume a path whose bytes changed locally."""

    import glm52_enforcement.task13_support_artifacts as artifacts

    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    pin = _wheel(
        wheelhouse / "cryptography-49.0.0-py3-none-any.whl",
        distribution="cryptography",
        version="49.0.0",
        files={
            "cryptography/__init__.py": b'__version__ = "49.0.0"\n',
        },
    )
    monkeypatch.setattr(
        artifacts,
        "PINNED_CRYPTOGRAPHY_WHEELS",
        (pin,),
    )
    manifest_path = tmp_path / "manifest.json"
    support_path = tmp_path / "support.zip"
    layer_path = tmp_path / "layer.zip"

    manifest = artifacts.materialize_support_artifacts(
        wheelhouse=wheelhouse,
        support_output=support_path,
        layer_output=layer_path,
        manifest_output=manifest_path,
    )

    assert manifest_path.read_bytes() == canonical_json_bytes(manifest) + b"\n"
    assert artifacts.read_support_artifact_manifest(manifest_path) == manifest
    support_path.write_bytes(support_path.read_bytes() + b"foreign")
    with pytest.raises(
        artifacts.Task13SupportArtifactError,
        match="support Lambda archive bytes drifted",
    ):
        artifacts.read_support_artifact_manifest(manifest_path)


def test_wheel_fetch_requires_exact_url_size_and_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: a successful HTTPS response is accepted without its pin."""

    import glm52_enforcement.task13_support_artifacts as artifacts

    raw = b"one exact wheel body"
    pin = artifacts.PinnedWheel(
        distribution="cryptography",
        version="49.0.0",
        filename="cryptography-49.0.0-test.whl",
        url="https://files.pythonhosted.org/packages/exact/test.whl",
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(
        artifacts,
        "PINNED_CRYPTOGRAPHY_WHEELS",
        (pin,),
    )

    class Response(io.BytesIO):
        status = 200

        def geturl(self) -> str:
            return pin.url

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    monkeypatch.setattr(
        artifacts.urllib.request,
        "urlopen",
        lambda request, timeout: Response(raw),
    )
    wheelhouse = tmp_path / "wheelhouse"

    fetched = artifacts.fetch_pinned_wheels(wheelhouse)

    assert fetched == (
        {
            "distribution": "cryptography",
            "version": "49.0.0",
            "filename": pin.filename,
            "url": pin.url,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    )
    assert (wheelhouse / pin.filename).read_bytes() == raw


def _metadata(label: str) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RequestId": label,
        "RetryAttempts": 0,
    }


class _ExactSts:
    def get_caller_identity(self) -> dict[str, object]:
        return {
            "Account": ACCOUNT,
            "Arn": (
                "arn:aws:sts::246813579024:assumed-role/task13-support-artifacts/test"
            ),
            "UserId": "AROASUPPORT:test",
            "ResponseMetadata": _metadata("sts"),
        }


class _ExactS3:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}
        self.put_requests: list[dict[str, object]] = []

    def get_bucket_versioning(self, **request: object) -> dict[str, object]:
        assert request == {
            "Bucket": BUCKET,
            "ExpectedBucketOwner": ACCOUNT,
        }
        return {
            "Status": "Enabled",
            "ResponseMetadata": _metadata("versioning"),
        }

    def list_object_versions(self, **request: object) -> dict[str, object]:
        key = str(request["Prefix"])
        row = self.rows.get(key)
        versions = (
            []
            if row is None
            else [
                {
                    "Key": key,
                    "VersionId": row["VersionId"],
                    "IsLatest": True,
                    "Size": len(row["Body"]),
                }
            ]
        )
        return {
            "Versions": versions,
            "DeleteMarkers": [],
            "IsTruncated": False,
            "ResponseMetadata": _metadata("list"),
        }

    def put_object(self, **request: object) -> dict[str, object]:
        self.put_requests.append(dict(request))
        key = str(request["Key"])
        assert key not in self.rows
        version_id = "version-" + str(len(self.rows) + 1)
        self.rows[key] = {
            "VersionId": version_id,
            "Body": request["Body"],
            "Metadata": request["Metadata"],
        }
        return {
            "VersionId": version_id,
            "ChecksumSHA256": request["ChecksumSHA256"],
            "ResponseMetadata": _metadata("put"),
        }

    def get_object(self, **request: object) -> dict[str, object]:
        row = self.rows[str(request["Key"])]
        raw = row["Body"]
        assert type(raw) is bytes
        return {
            "Body": io.BytesIO(raw),
            "ContentLength": len(raw),
            "VersionId": row["VersionId"],
            "ChecksumSHA256": base64.b64encode(hashlib.sha256(raw).digest()).decode(
                "ascii"
            ),
            "Metadata": row["Metadata"],
            "ResponseMetadata": _metadata("get"),
        }


def test_publication_uses_two_content_addressed_versioned_s3_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: layer or Lambda bytes publish without an immutable VersionId."""

    import glm52_enforcement.task13_support_artifacts as artifacts
    from glm52_enforcement.task13_fixed_artifacts import (
        Task13FixedArtifactServices,
    )

    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    pin = _wheel(
        wheelhouse / "cryptography-49.0.0-py3-none-any.whl",
        distribution="cryptography",
        version="49.0.0",
        files={"cryptography/__init__.py": b"PINNED = True\n"},
    )
    monkeypatch.setattr(
        artifacts,
        "PINNED_CRYPTOGRAPHY_WHEELS",
        (pin,),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = artifacts.materialize_support_artifacts(
        wheelhouse=wheelhouse,
        support_output=tmp_path / "support.zip",
        layer_output=tmp_path / "layer.zip",
        manifest_output=manifest_path,
    )
    s3 = _ExactS3()
    services = Task13FixedArtifactServices(
        sts=_ExactSts(),
        s3=s3,
        total_max_attempts=1,
    )

    coordinate = artifacts.publish_support_artifacts(
        manifest_path=manifest_path,
        bucket=BUCKET,
        services=services,
    )

    assert len(s3.put_requests) == 2
    support_sha = manifest["support_lambda_archive"]["file_sha256"]
    layer_sha = manifest["cryptography_layer_archive"]["file_sha256"]
    assert {request["Key"] for request in s3.put_requests} == {
        f"task13/artifacts/support-lambda/{support_sha}.zip",
        (f"task13/artifacts/cryptography-layer-python312-x86_64/{layer_sha}.zip"),
    }
    assert all(
        request["IfNoneMatch"] == "*" and request["ExpectedBucketOwner"] == ACCOUNT
        for request in s3.put_requests
    )
    assert coordinate["support_lambda_archive"]["version_id"] == "version-1"
    assert coordinate["cryptography_layer_archive"]["version_id"] == "version-2"
    assert artifacts.validate_support_artifact_publication(coordinate) == coordinate
    foreign = json.loads(canonical_json_bytes(coordinate))
    foreign["cryptography_layer_archive"]["key"] = "foreign.zip"
    with pytest.raises(
        artifacts.Task13SupportArtifactError,
        match="publication identity drifted",
    ):
        artifacts.validate_support_artifact_publication(foreign)


def test_cli_builds_locally_without_constructing_aws_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: the local materialization verb silently opens AWS clients."""

    import glm52_enforcement.task13_support_artifacts as artifacts

    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    pin = _wheel(
        wheelhouse / "cryptography-49.0.0-py3-none-any.whl",
        distribution="cryptography",
        version="49.0.0",
        files={"cryptography/__init__.py": b"LOCAL = True\n"},
    )
    monkeypatch.setattr(
        artifacts,
        "PINNED_CRYPTOGRAPHY_WHEELS",
        (pin,),
    )
    spec = importlib.util.spec_from_file_location(
        "_task13_support_artifact_cli",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden_services():
        raise AssertionError("local materialization constructed AWS services")

    manifest = tmp_path / "manifest.json"
    result = module.main(
        [
            "materialize",
            "--wheelhouse",
            str(wheelhouse),
            "--support-output",
            str(tmp_path / "support.zip"),
            "--layer-output",
            str(tmp_path / "layer.zip"),
            "--manifest-output",
            str(manifest),
        ],
        services_factory=forbidden_services,
    )

    assert result == 0
    assert artifacts.read_support_artifact_manifest(manifest)["runtime"] == "python3.12"
