#!/usr/bin/env python3
"""Publish exact immutable Task 13 fixed-key reviewed artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import hashlib
import io
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task13_fixed_artifacts import (  # noqa: E402
    PROFILE,
    REGION,
    Task13FixedArtifactError,
    Task13FixedArtifactServices,
    publish_repository_archive,
    publish_repository_driver,
    publish_support_build_inputs,
    read_driver_materialization_request,
    repository_archive_manifest_key,
    write_coordinate_once,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    archive = subparsers.add_parser("publish-archive")
    archive.add_argument("--activation-id", required=True)
    archive.add_argument("--archive", required=True, type=Path)
    archive.add_argument("--manifest-output", required=True, type=Path)
    driver = subparsers.add_parser("publish-driver")
    driver.add_argument("--request", required=True, type=Path)
    support = subparsers.add_parser("publish-support-inputs")
    support.add_argument("--inputs", required=True, type=Path)
    for child in (archive, driver, support):
        child.add_argument("--bucket", required=True)
        child.add_argument(
            "--coordinate-output",
            required=True,
            type=Path,
        )
    return parser


class _ManifestCaptureS3:
    """Retain manifest bytes already used by the typed publication boundary."""

    def __init__(self, delegate: object, manifest_key: str) -> None:
        self._delegate = delegate
        self._manifest_key = manifest_key
        self._manifest_raw: bytes | None = None

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)

    def _retain(self, raw: object) -> None:
        if type(raw) is not bytes:
            raise Task13FixedArtifactError(
                "repository archive manifest publication bytes are invalid"
            )
        if self._manifest_raw is not None and self._manifest_raw != raw:
            raise Task13FixedArtifactError(
                "repository archive manifest publication bytes drifted"
            )
        self._manifest_raw = raw

    def put_object(self, **request: object) -> object:
        if request.get("Key") == self._manifest_key:
            self._retain(request.get("Body"))
        method = getattr(self._delegate, "put_object", None)
        if not callable(method):
            raise Task13FixedArtifactError(
                "PutObject typed client method is absent"
            )
        return method(**request)

    def get_object(self, **request: object) -> object:
        method = getattr(self._delegate, "get_object", None)
        if not callable(method):
            raise Task13FixedArtifactError(
                "GetObject typed client method is absent"
            )
        response = method(**request)
        if request.get("Key") != self._manifest_key:
            return response
        if type(request.get("VersionId")) is not str:
            raise Task13FixedArtifactError(
                "repository archive manifest readback is not version-pinned"
            )
        if type(response) is not dict:
            raise Task13FixedArtifactError(
                "repository archive manifest readback is malformed"
            )
        body = response.get("Body")
        read = getattr(body, "read", None)
        if not callable(read):
            raise Task13FixedArtifactError(
                "repository archive manifest readback body is malformed"
            )
        raw = read()
        self._retain(raw)
        replay = dict(response)
        replay["Body"] = io.BytesIO(raw)
        return replay

    def manifest_raw(self) -> bytes:
        if self._manifest_raw is None:
            raise Task13FixedArtifactError(
                "repository archive manifest publication bytes were not retained"
            )
        return self._manifest_raw


def _preflight_local_outputs(paths: list[Path]) -> None:
    resolved = [path.resolve(strict=False) for path in paths]
    if (
        len(set(resolved)) != len(paths)
        or paths != resolved
        or any(
            not path.is_absolute()
            or not path.parent.is_dir()
            or path.parent.is_symlink()
            or path.exists()
            or path.is_symlink()
            for path in paths
        )
    ):
        raise Task13FixedArtifactError(
            "local outputs must be distinct new absolute files"
        )


def _write_local_outputs_once(outputs: list[tuple[Path, bytes]]) -> None:
    """Create one local output set and roll it all back on any failure."""

    paths = [path for path, _raw in outputs]
    if not outputs or any(type(raw) is not bytes for _path, raw in outputs):
        raise Task13FixedArtifactError("local output bytes are invalid")
    _preflight_local_outputs(paths)
    created: list[Path] = []
    descriptors: list[int] = []
    try:
        for path, _raw in outputs:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            created.append(path)
            descriptors.append(descriptor)
            os.fchmod(descriptor, 0o600)
        for descriptor, (_path, raw) in zip(descriptors, outputs):
            remaining = memoryview(raw)
            while remaining:
                written = os.write(descriptor, remaining)
                if (
                    type(written) is not int
                    or written <= 0
                    or written > len(remaining)
                ):
                    raise OSError("local output write made no progress")
                remaining = remaining[written:]
        for descriptor in descriptors:
            os.fsync(descriptor)
        close_error: BaseException | None = None
        while descriptors:
            descriptor = descriptors.pop(0)
            try:
                os.close(descriptor)
            except BaseException as error:
                if close_error is None:
                    close_error = error
        if close_error is not None:
            raise close_error
    except BaseException as original_error:
        while descriptors:
            descriptor = descriptors.pop()
            try:
                os.close(descriptor)
            except BaseException:
                pass
        cleanup_errors: list[OSError] = []
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                cleanup_errors.append(cleanup_error)
        for cleanup_error in cleanup_errors:
            original_error.add_note(
                "local output cleanup failed: " + str(cleanup_error)
            )
        raise


def _validated_manifest_raw(
    *,
    raw: bytes,
    coordinate: Mapping[str, object],
    expected_key: str,
) -> bytes:
    try:
        manifest = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Task13FixedArtifactError(
            "retained repository archive manifest is not JSON"
        ) from error
    if (
        type(manifest) is not dict
        or raw != canonical_json_bytes(manifest) + b"\n"
        or coordinate.get("key") != expected_key
        or coordinate.get("file_sha256")
        != hashlib.sha256(raw).hexdigest()
        or coordinate.get("body_sha256")
        != manifest.get("canonical_identity_sha256")
    ):
        raise Task13FixedArtifactError(
            "retained repository archive manifest does not match coordinate"
        )
    return raw


def _build_services() -> Task13FixedArtifactServices:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    config = Config(
        region_name=REGION,
        connect_timeout=5,
        read_timeout=30,
        retries={"mode": "standard", "total_max_attempts": 1},
    )
    session = boto3.Session(
        profile_name=PROFILE,
        region_name=REGION,
    )
    if session.region_name != REGION:
        raise Task13FixedArtifactError("AWS session region drifted")
    return Task13FixedArtifactServices(
        sts=session.client("sts", config=config),
        s3=session.client("s3", config=config),
        total_max_attempts=1,
    )


def main(
    argv: list[str] | None = None,
    *,
    services_factory: Callable[[], Task13FixedArtifactServices] = (
        _build_services
    ),
) -> int:
    args = _parser().parse_args(argv)
    try:
        local_outputs = [args.coordinate_output]
        if args.operation == "publish-archive":
            local_outputs.append(args.manifest_output)
        _preflight_local_outputs(local_outputs)
        services = services_factory()
        if type(services) is not Task13FixedArtifactServices:
            raise Task13FixedArtifactError(
                "Task 13 fixed-artifact transport is not exact"
            )
        if args.operation == "publish-archive":
            manifest_key = repository_archive_manifest_key(
                args.activation_id
            )
            capture = _ManifestCaptureS3(services.s3, manifest_key)
            capturing_services = Task13FixedArtifactServices(
                sts=services.sts,
                s3=capture,
                total_max_attempts=services.total_max_attempts,
            )
            coordinate = publish_repository_archive(
                activation_id=args.activation_id,
                archive_path=args.archive,
                bucket=args.bucket,
                services=capturing_services,
            )
            manifest_raw = _validated_manifest_raw(
                raw=capture.manifest_raw(),
                coordinate=coordinate,
                expected_key=manifest_key,
            )
            _write_local_outputs_once(
                [
                    (
                        args.coordinate_output,
                        canonical_json_bytes(coordinate) + b"\n",
                    ),
                    (args.manifest_output, manifest_raw),
                ]
            )
        elif args.operation == "publish-driver":
            coordinate = publish_repository_driver(
                request=read_driver_materialization_request(args.request),
                bucket=args.bucket,
                services=services,
            )
            write_coordinate_once(args.coordinate_output, coordinate)
        else:
            coordinate = publish_support_build_inputs(
                inputs_path=args.inputs,
                bucket=args.bucket,
                services=services,
            )
            write_coordinate_once(args.coordinate_output, coordinate)
    except (
        FileExistsError,
        OSError,
        RuntimeError,
        Task13FixedArtifactError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "Task 13 fixed-artifact materialization refused: %s" % error,
            file=sys.stderr,
        )
        return 64
    print(str(args.coordinate_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
