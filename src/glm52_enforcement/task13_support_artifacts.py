"""Deterministic Task 13 support-Lambda and cryptography-layer artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import stat
import tempfile
import urllib.request
import zipfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .canonical import canonical_json_bytes, canonical_sha256
from .task13_fixed_artifacts import (
    ACCOUNT_ID,
    CAMPAIGN_BUCKET,
    REGION,
    RUN_ID,
    Task13FixedArtifactServices,
    _guard_services,
    _publish_bytes,
)

FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
LAMBDA_RUNTIME = "python3.12"
LAMBDA_ARCHITECTURE = "x86_64"
CRYPTOGRAPHY_REQUIREMENT = "cryptography==49.0.0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class Task13SupportArtifactError(ValueError):
    """A local support artifact failed its exact identity contract."""


@dataclass(frozen=True)
class PinnedWheel:
    """One exact wheel admitted to the Lambda layer."""

    distribution: str
    version: str
    filename: str
    url: str
    size_bytes: int
    sha256: str


# Exact PyPI files resolved for CPython 3.12, x86_64, manylinux2014 and
# authenticated against PyPI's release JSON plus independently downloaded
# bytes on 2026-07-29.
PINNED_CRYPTOGRAPHY_WHEELS = (
    PinnedWheel(
        distribution="cryptography",
        version="49.0.0",
        filename=(
            "cryptography-49.0.0-cp311-abi3-manylinux2014_x86_64."
            "manylinux_2_17_x86_64.whl"
        ),
        url=(
            "https://files.pythonhosted.org/packages/e6/8b/"
            "43011f7ebe515a8aa20d61f290a326cd890c2e738e16e59eaff8d9c3a412/"
            "cryptography-49.0.0-cp311-abi3-manylinux2014_x86_64."
            "manylinux_2_17_x86_64.whl"
        ),
        size_bytes=4_716_422,
        sha256=("0e959b578856a3924bc0cbb710fc12c387b9412a951389f3ca61704a9e25f325"),
    ),
    PinnedWheel(
        distribution="cffi",
        version="2.1.0",
        filename=(
            "cffi-2.1.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
        ),
        url=(
            "https://files.pythonhosted.org/packages/62/f2/"
            "c9522a81c32132799a1972c39f5c5f8b4c8b9f00488a23feaa6c06f07741/"
            "cffi-2.1.0-cp312-cp312-manylinux2014_x86_64."
            "manylinux_2_17_x86_64.whl"
        ),
        size_bytes=221_844,
        sha256=("1e9f50d192a3e525b15a75ab5114e442d83d657b7ec29182a991bc9a88fd3a66"),
    ),
    PinnedWheel(
        distribution="typing_extensions",
        version="4.16.0",
        filename="typing_extensions-4.16.0-py3-none-any.whl",
        url=(
            "https://files.pythonhosted.org/packages/49/d3/"
            "b8441a820a491ddfc024b0b0cf0393375b75ea13866d9c66727e54c2fc80/"
            "typing_extensions-4.16.0-py3-none-any.whl"
        ),
        size_bytes=45_571,
        sha256=("481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8"),
    ),
    PinnedWheel(
        distribution="pycparser",
        version="3.0",
        filename="pycparser-3.0-py3-none-any.whl",
        url=(
            "https://files.pythonhosted.org/packages/0c/c3/"
            "44f3fbbfa403ea2a7c779186dc20772604442dde72947e7d01069cbe98e3/"
            "pycparser-3.0-py3-none-any.whl"
        ),
        size_bytes=48_172,
        sha256=("b727414169a36b7d524c1c3e31839a521725078d7b2ff038656844266160a992"),
    ),
)


def _fail(message: str) -> None:
    raise Task13SupportArtifactError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_once(path: Path, raw: bytes) -> None:
    path = Path(path)
    if (
        not path.is_absolute()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.exists()
        or path.is_symlink()
    ):
        _fail("artifact output must be one new absolute file")
    descriptor_number = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    descriptor = os.fdopen(descriptor_number, "wb")
    try:
        descriptor.write(raw)
        descriptor.flush()
        os.fsync(descriptor.fileno())
    except BaseException:
        descriptor.close()
        with suppress(FileNotFoundError):
            path.unlink()
        raise
    descriptor.close()
    if path.read_bytes() != raw:
        _fail("artifact local readback drifted")


def _record_projection(record: PinnedWheel) -> dict[str, object]:
    if (
        type(record) is not PinnedWheel
        or not record.distribution
        or not record.version
        or not record.filename.endswith(".whl")
        or Path(record.filename).name != record.filename
        or not record.url.startswith("https://files.pythonhosted.org/")
        or type(record.size_bytes) is not int
        or record.size_bytes < 1
        or len(record.sha256) != 64
        or any(character not in "0123456789abcdef" for character in record.sha256)
    ):
        _fail("pinned wheel record is malformed")
    return {
        "distribution": record.distribution,
        "version": record.version,
        "filename": record.filename,
        "url": record.url,
        "size_bytes": record.size_bytes,
        "sha256": record.sha256,
    }


def _wheel_target(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or ".." in path.parts
        or "." in path.parts
    ):
        _fail("wheel contains an unsafe member")
    parts = path.parts
    if parts[0].endswith(".data"):
        if len(parts) < 3 or parts[1] not in {"purelib", "platlib"}:
            _fail("wheel contains unsupported .data content")
        parts = parts[2:]
    if not parts:
        _fail("wheel member has no Lambda-layer target")
    return str(PurePosixPath("python", *parts))


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    sink = io.BytesIO()
    with zipfile.ZipFile(
        sink,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, entries[name])
    return sink.getvalue()


def _artifact_projection(path: Path, label: str) -> dict[str, object]:
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        _fail(label + " must be one regular absolute file")
    raw = path.read_bytes()
    if not raw:
        _fail(label + " is empty")
    return {
        "path": str(path),
        "size_bytes": len(raw),
        "file_sha256": _sha256(raw),
    }


def build_support_lambda_archive(output: Path) -> dict[str, object]:
    """Run the existing support packager twice and retain byte-identical output."""

    output = Path(output)
    if (
        not output.is_absolute()
        or not output.parent.is_dir()
        or output.parent.is_symlink()
        or output.exists()
        or output.is_symlink()
    ):
        _fail("support Lambda output must be one new absolute file")
    package_script = (
        REPOSITORY_ROOT / "aws/glm52-gpu/scripts/package_h1g_support_lambdas.py"
    )
    if not package_script.is_file() or package_script.is_symlink():
        _fail("existing support Lambda packager is absent")
    spec = importlib.util.spec_from_file_location(
        "_glm52_task13_support_lambda_packager",
        package_script,
    )
    if spec is None or spec.loader is None:
        _fail("existing support Lambda packager cannot be loaded")
    packager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packager)
    build = getattr(packager, "build", None)
    if not callable(build):
        _fail("existing support Lambda build entrypoint is absent")
    with tempfile.TemporaryDirectory(
        prefix=".glm52-support-build-",
        dir=output.parent,
    ) as temporary:
        temporary_path = Path(temporary)
        first = temporary_path / "first.zip"
        second = temporary_path / "second.zip"
        build(first)
        build(second)
        first_raw = first.read_bytes()
        second_raw = second.read_bytes()
    if not first_raw or first_raw != second_raw:
        _fail("existing support Lambda packager is not byte deterministic")
    try:
        with zipfile.ZipFile(io.BytesIO(first_raw)) as archive:
            names = archive.namelist()
            if (
                not names
                or len(names) != len(set(names))
                or any(
                    name.startswith("/")
                    or "\\" in name
                    or ".." in PurePosixPath(name).parts
                    for name in names
                )
                or any(
                    stat.S_ISLNK(info.external_attr >> 16)
                    for info in archive.infolist()
                )
            ):
                _fail("support Lambda archive inventory is unsafe")
    except zipfile.BadZipFile as exc:
        raise Task13SupportArtifactError(
            "existing support Lambda packager emitted an unreadable ZIP"
        ) from exc
    _write_once(output, first_raw)
    return {
        "size_bytes": len(first_raw),
        "file_sha256": _sha256(first_raw),
        "entry_count": len(names),
    }


def fetch_pinned_wheels(wheelhouse: Path) -> tuple[dict[str, object], ...]:
    """Download the exact locked wheels with one response and byte verification."""

    wheelhouse = Path(wheelhouse)
    if (
        not wheelhouse.is_absolute()
        or not wheelhouse.parent.is_dir()
        or wheelhouse.parent.is_symlink()
        or wheelhouse.exists()
        or wheelhouse.is_symlink()
    ):
        _fail("wheelhouse output must be one new absolute directory")
    projections = tuple(
        sorted(
            (_record_projection(record) for record in PINNED_CRYPTOGRAPHY_WHEELS),
            key=lambda item: (
                str(item["distribution"]),
                str(item["filename"]),
            ),
        )
    )
    if not projections:
        _fail("pinned wheel closure is empty")
    wheelhouse.mkdir(mode=0o700)
    try:
        for projection in projections:
            url = str(projection["url"])
            request = urllib.request.Request(
                url,
                headers={"Accept": "application/octet-stream"},
                method="GET",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                status_code = getattr(response, "status", None)
                final_url = response.geturl()
                raw = response.read(int(projection["size_bytes"]) + 1)
            if (
                status_code != 200
                or final_url != url
                or len(raw) != projection["size_bytes"]
                or _sha256(raw) != projection["sha256"]
            ):
                _fail(str(projection["filename"]) + " download identity drifted")
            _write_once(
                wheelhouse / str(projection["filename"]),
                raw,
            )
    except BaseException:
        for child in wheelhouse.iterdir():
            if child.is_file() and not child.is_symlink():
                child.unlink()
        wheelhouse.rmdir()
        raise
    return projections


def build_cryptography_layer_archive(
    *,
    wheelhouse: Path,
    output: Path,
    wheel_records: Sequence[PinnedWheel],
) -> dict[str, object]:
    """Build a layer only from byte-exact wheel records."""

    wheelhouse = Path(wheelhouse)
    if not wheelhouse.is_dir() or wheelhouse.is_symlink() or not wheel_records:
        _fail("wheelhouse must be one concrete nonempty directory")
    projections = sorted(
        (_record_projection(record) for record in wheel_records),
        key=lambda value: (
            str(value["distribution"]),
            str(value["filename"]),
        ),
    )
    if len({item["filename"] for item in projections}) != len(projections):
        _fail("pinned wheel filenames are not unique")
    if len({item["distribution"] for item in projections}) != len(projections):
        _fail("pinned wheel distributions are not unique")
    records_by_name = {record.filename: record for record in wheel_records}
    entries: dict[str, bytes] = {}
    for projection in projections:
        record = records_by_name[str(projection["filename"])]
        path = wheelhouse / record.filename
        if not path.is_file() or path.is_symlink():
            _fail(record.filename + " wheel is absent")
        raw = path.read_bytes()
        if len(raw) != record.size_bytes:
            _fail(record.filename + " wheel size drifted")
        if _sha256(raw) != record.sha256:
            _fail(record.filename + " wheel SHA-256 drifted")
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as wheel:
                seen_source: set[str] = set()
                for info in wheel.infolist():
                    if info.filename in seen_source:
                        _fail(record.filename + " wheel repeats a member")
                    seen_source.add(info.filename)
                    mode = info.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        _fail(record.filename + " wheel contains a symlink")
                    if info.is_dir():
                        continue
                    target = _wheel_target(info.filename)
                    if target in entries:
                        _fail("wheel layer target collision at " + target)
                    entries[target] = wheel.read(info)
        except (OSError, zipfile.BadZipFile) as exc:
            raise Task13SupportArtifactError(
                record.filename + " wheel is not a readable ZIP"
            ) from exc
    embedded_body = {
        "schema_version": 1,
        "record_type": "glm52_task13_cryptography_layer_manifest_v1",
        "runtime": LAMBDA_RUNTIME,
        "architecture": LAMBDA_ARCHITECTURE,
        "cryptography_requirement": CRYPTOGRAPHY_REQUIREMENT,
        "wheels": projections,
    }
    embedded_manifest = {
        **embedded_body,
        "canonical_identity_sha256": canonical_sha256(embedded_body),
    }
    embedded_raw = canonical_json_bytes(embedded_manifest) + b"\n"
    embedded_name = "python/glm52-cryptography-layer-manifest.json"
    if embedded_name in entries:
        _fail("wheel collides with the retained layer manifest")
    entries[embedded_name] = embedded_raw
    archive_raw = _zip_bytes(entries)
    _write_once(Path(output), archive_raw)
    return {
        "runtime": LAMBDA_RUNTIME,
        "architecture": LAMBDA_ARCHITECTURE,
        "cryptography_requirement": CRYPTOGRAPHY_REQUIREMENT,
        "size_bytes": len(archive_raw),
        "file_sha256": _sha256(archive_raw),
        "embedded_manifest_sha256": _sha256(embedded_raw),
        "wheels": projections,
    }


_ARTIFACT_FIELDS = frozenset(
    {
        "path",
        "size_bytes",
        "file_sha256",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "runtime",
        "architecture",
        "cryptography_requirement",
        "wheels",
        "support_lambda_archive",
        "cryptography_layer_archive",
        "canonical_identity_sha256",
    }
)


def _validate_local_coordinate(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _ARTIFACT_FIELDS:
        _fail(label + " coordinate schema drifted")
    path = Path(str(value.get("path")))
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] < 1
        or type(value.get("file_sha256")) is not str
        or len(value["file_sha256"]) != 64
    ):
        _fail(label + " coordinate is malformed")
    raw = path.read_bytes()
    if len(raw) != value["size_bytes"] or _sha256(raw) != value["file_sha256"]:
        _fail(label + " bytes drifted")
    return dict(value)


def validate_support_artifact_manifest(value: object) -> dict[str, object]:
    """Authenticate local archive paths and their fixed dependency closure."""

    if type(value) is not dict or set(value) != _MANIFEST_FIELDS:
        _fail("support artifact manifest schema drifted")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256", None)
    pinned = sorted(
        (_record_projection(record) for record in PINNED_CRYPTOGRAPHY_WHEELS),
        key=lambda item: (
            str(item["distribution"]),
            str(item["filename"]),
        ),
    )
    if (
        value.get("schema_version") != 1
        or value.get("record_type") != "glm52_task13_support_artifact_manifest_v1"
        or value.get("account_id") != ACCOUNT_ID
        or value.get("region") != REGION
        or value.get("run_id") != RUN_ID
        or value.get("runtime") != LAMBDA_RUNTIME
        or value.get("architecture") != LAMBDA_ARCHITECTURE
        or value.get("cryptography_requirement") != CRYPTOGRAPHY_REQUIREMENT
        or value.get("wheels") != pinned
        or identity != canonical_sha256(body)
    ):
        _fail("support artifact manifest identity drifted")
    _validate_local_coordinate(
        value["support_lambda_archive"],
        label="support Lambda archive",
    )
    _validate_local_coordinate(
        value["cryptography_layer_archive"],
        label="cryptography layer archive",
    )
    return json.loads(canonical_json_bytes(value))


def read_support_artifact_manifest(path: Path) -> dict[str, object]:
    """Read one canonical local artifact manifest and authenticate its files."""

    path = Path(path)
    if not path.is_file() or path.is_symlink():
        _fail("support artifact manifest must be one regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Task13SupportArtifactError(
            "support artifact manifest is not canonical ASCII JSON"
        ) from exc
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        _fail("support artifact manifest is not canonical JSON plus LF")
    return validate_support_artifact_manifest(value)


def materialize_support_artifacts(
    *,
    wheelhouse: Path,
    support_output: Path,
    layer_output: Path,
    manifest_output: Path,
) -> dict[str, object]:
    """Materialize both exact local ZIPs and their authenticated manifest."""

    support = build_support_lambda_archive(Path(support_output))
    layer = build_cryptography_layer_archive(
        wheelhouse=Path(wheelhouse),
        output=Path(layer_output),
        wheel_records=PINNED_CRYPTOGRAPHY_WHEELS,
    )
    body = {
        "schema_version": 1,
        "record_type": "glm52_task13_support_artifact_manifest_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "runtime": LAMBDA_RUNTIME,
        "architecture": LAMBDA_ARCHITECTURE,
        "cryptography_requirement": CRYPTOGRAPHY_REQUIREMENT,
        "wheels": layer["wheels"],
        "support_lambda_archive": {
            "path": str(Path(support_output)),
            "size_bytes": support["size_bytes"],
            "file_sha256": support["file_sha256"],
        },
        "cryptography_layer_archive": {
            "path": str(Path(layer_output)),
            "size_bytes": layer["size_bytes"],
            "file_sha256": layer["file_sha256"],
        },
    }
    manifest = {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }
    manifest = validate_support_artifact_manifest(manifest)
    _write_once(
        Path(manifest_output),
        canonical_json_bytes(manifest) + b"\n",
    )
    return manifest


def publish_support_artifacts(
    *,
    manifest_path: Path,
    bucket: str,
    services: Task13FixedArtifactServices,
) -> dict[str, object]:
    """Publish both exact ZIPs to content-addressed, versioned S3 keys."""

    manifest = read_support_artifact_manifest(Path(manifest_path))
    _guard_services(services, bucket)
    coordinates: dict[str, dict[str, object]] = {}
    specifications = (
        (
            "support_lambda_archive",
            "support-lambda",
            "glm52_task13_support_lambda_archive_v1",
        ),
        (
            "cryptography_layer_archive",
            "cryptography-layer-python312-x86_64",
            "glm52_task13_cryptography_layer_archive_v1",
        ),
    )
    for field, key_kind, record_type in specifications:
        source = manifest[field]
        source = _validate_local_coordinate(
            source,
            label=field.replace("_", " "),
        )
        raw = Path(str(source["path"])).read_bytes()
        key = f"task13/artifacts/{key_kind}/{source['file_sha256']}.zip"
        version_id = _publish_bytes(
            s3=services.s3,
            bucket=bucket,
            key=key,
            raw=raw,
            record_type=record_type,
            content_type="application/zip",
        )
        coordinates[field] = {
            "bucket": bucket,
            "key": key,
            "version_id": version_id,
            "size_bytes": source["size_bytes"],
            "file_sha256": source["file_sha256"],
        }
    body = {
        "schema_version": 1,
        "record_type": "glm52_task13_support_artifact_publication_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "runtime": LAMBDA_RUNTIME,
        "architecture": LAMBDA_ARCHITECTURE,
        "source_manifest_identity_sha256": (manifest["canonical_identity_sha256"]),
        **coordinates,
    }
    return validate_support_artifact_publication(
        {
            **body,
            "canonical_identity_sha256": canonical_sha256(body),
        }
    )


_PUBLICATION_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "runtime",
        "architecture",
        "source_manifest_identity_sha256",
        "support_lambda_archive",
        "cryptography_layer_archive",
        "canonical_identity_sha256",
    }
)
_PUBLISHED_ARTIFACT_FIELDS = frozenset(
    {
        "bucket",
        "key",
        "version_id",
        "size_bytes",
        "file_sha256",
    }
)


def _validate_published_coordinate(
    value: object,
    *,
    key_kind: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _PUBLISHED_ARTIFACT_FIELDS:
        _fail("support artifact publication coordinate schema drifted")
    sha = value.get("file_sha256")
    version = value.get("version_id")
    expected_key = f"task13/artifacts/{key_kind}/{sha}.zip"
    if (
        type(sha) is not str
        or len(sha) != 64
        or any(character not in "0123456789abcdef" for character in sha)
        or value.get("bucket") != CAMPAIGN_BUCKET
        or value.get("key") != expected_key
        or type(version) is not str
        or not version
        or version == "null"
        or not version.isascii()
        or version != version.strip()
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] < 1
    ):
        _fail("support artifact publication identity drifted")
    return dict(value)


def validate_support_artifact_publication(
    value: object,
) -> dict[str, object]:
    """Validate the two exact versioned S3 coordinates for downstream use."""

    if type(value) is not dict or set(value) != _PUBLICATION_FIELDS:
        _fail("support artifact publication schema drifted")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256", None)
    source_identity = value.get("source_manifest_identity_sha256")
    if (
        value.get("schema_version") != 1
        or value.get("record_type") != "glm52_task13_support_artifact_publication_v1"
        or value.get("account_id") != ACCOUNT_ID
        or value.get("region") != REGION
        or value.get("run_id") != RUN_ID
        or value.get("runtime") != LAMBDA_RUNTIME
        or value.get("architecture") != LAMBDA_ARCHITECTURE
        or type(source_identity) is not str
        or len(source_identity) != 64
        or any(character not in "0123456789abcdef" for character in source_identity)
        or identity != canonical_sha256(body)
    ):
        _fail("support artifact publication identity drifted")
    _validate_published_coordinate(
        value["support_lambda_archive"],
        key_kind="support-lambda",
    )
    _validate_published_coordinate(
        value["cryptography_layer_archive"],
        key_kind="cryptography-layer-python312-x86_64",
    )
    return json.loads(canonical_json_bytes(value))


__all__ = [
    "CRYPTOGRAPHY_REQUIREMENT",
    "LAMBDA_ARCHITECTURE",
    "LAMBDA_RUNTIME",
    "PINNED_CRYPTOGRAPHY_WHEELS",
    "PinnedWheel",
    "Task13SupportArtifactError",
    "build_cryptography_layer_archive",
    "build_support_lambda_archive",
    "fetch_pinned_wheels",
    "materialize_support_artifacts",
    "publish_support_artifacts",
    "read_support_artifact_manifest",
    "validate_support_artifact_manifest",
    "validate_support_artifact_publication",
]
