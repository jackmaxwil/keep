#!/usr/bin/env python3
"""Build the canonical v2 request for Task 13 staged deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.task13_support_input_materialization import (
    parse_support_input_materialization_request,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ACTIVATION_ID = re.compile(r"[a-z0-9][a-z0-9-]{2,63}\Z")
_STACK_ID = re.compile(
    r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
    r"keep-glm52-gpu/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_INPUT_PATH_FIELDS = {
    "retained_bootstrap_runtime_deployment": (
        "retained_bootstrap_runtime_deployment_path"
    ),
    "bridge_seed_publication": "bridge_seed_publication_path",
    "retained_fence_runtime_deployment": ("retained_fence_runtime_deployment_path"),
    "bridge_seed": "bridge_seed_path",
    "migration_operations_1_to_6": "migration_operations_1_to_6_path",
    "bootstrap_fence_publication": "bootstrap_fence_publication_path",
    "prepare_execution": "prepare_execution_path",
    "support_input_materialization_request": (
        "support_input_materialization_request_path"
    ),
    "disabled_support_deployment": "disabled_support_deployment_path",
    "operation_7": "operation_7_path",
    "support_runtime_identity": "support_runtime_identity_path",
    "no_launch_evidence": "no_launch_evidence_path",
}
_INVOCATION_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "activation_id",
        "output_directory",
        "staged_infrastructure_evidence_output",
        "journal_path",
        "retained_stack_id",
        *tuple(_INPUT_PATH_FIELDS.values()),
        "request_output_path",
        "canonical_identity_sha256",
    }
)


class RequestBuildError(ValueError):
    """The local Task 13 v2 build invocation failed closed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or not value.isascii()
        or value != value.strip()
        or "\n" in value
        or "\r" in value
        or "\x00" in value
    ):
        raise RequestBuildError(label + " must be one exact ASCII string")
    return value


def _path(value: object, label: str) -> Path:
    return Path(_text(value, label))


def _exact_file(value: object, label: str) -> Path:
    path = _path(value, label)
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise RequestBuildError(label + " must be one exact regular non-symlink file")
    return path


def _canonical_file(value: object, label: str) -> tuple[dict[str, object], bytes]:
    path = _exact_file(value, label)
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise RequestBuildError(label + " must have exactly one trailing LF")
    try:
        parsed = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RequestBuildError(label + " is not canonical ASCII JSON") from exc
    if type(parsed) is not dict or canonical_json_bytes(parsed) + b"\n" != raw:
        raise RequestBuildError(label + " bytes are not canonical")
    return parsed, raw


def _guard_invocation(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _INVOCATION_FIELDS:
        missing = sorted(_INVOCATION_FIELDS - set(value)) if type(value) is dict else []
        unknown = sorted(set(value) - _INVOCATION_FIELDS) if type(value) is dict else []
        raise RequestBuildError(
            f"build invocation v2 schema mismatch: missing={missing}, unknown={unknown}"
        )
    if (
        value["schema_version"] != 2
        or value["record_type"] != "glm52_task13_staged_deployment_request_build_v2"
    ):
        raise RequestBuildError("build invocation identity is not exact v2")
    identity = value["canonical_identity_sha256"]
    unsigned = dict(value)
    unsigned.pop("canonical_identity_sha256")
    if (
        type(identity) is not str
        or _SHA256.fullmatch(identity) is None
        or _sha256(canonical_json_bytes(unsigned)) != identity
    ):
        raise RequestBuildError("build invocation canonical identity drifted")
    activation_id = _text(value["activation_id"], "activation_id")
    if _ACTIVATION_ID.fullmatch(activation_id) is None:
        raise RequestBuildError("activation_id grammar is unsafe")
    if (
        _STACK_ID.fullmatch(_text(value["retained_stack_id"], "retained_stack_id"))
        is None
    ):
        raise RequestBuildError("retained_stack_id is not exact")
    input_paths = tuple(
        _text(value[field], field) for field in _INPUT_PATH_FIELDS.values()
    )
    if len(set(input_paths)) != len(input_paths):
        raise RequestBuildError("operation input paths must be distinct")
    return dict(value)


def _new_output(value: object, label: str) -> Path:
    path = _path(value, label)
    if (
        not path.is_absolute()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.exists()
        or path.is_symlink()
    ):
        raise RequestBuildError(label + " must be one new exact absolute file")
    return path


def _write_new(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("request output write made no progress")
            offset += written
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_request(
    invocation: object,
) -> tuple[dict[str, object], Path, Path, Path]:
    exact = _guard_invocation(invocation)
    output_directory = _path(exact["output_directory"], "output_directory")
    if (
        not output_directory.is_absolute()
        or not output_directory.is_dir()
        or output_directory.is_symlink()
        or output_directory.resolve(strict=True) != output_directory
    ):
        raise RequestBuildError("output_directory is not exact")
    evidence_output = _new_output(
        exact["staged_infrastructure_evidence_output"],
        "staged_infrastructure_evidence_output",
    )
    expected_evidence = output_directory / "staged-infrastructure-evidence-v2.json"
    if evidence_output != expected_evidence:
        raise RequestBuildError("aggregate evidence target is not canonical v2")
    request_output = _new_output(exact["request_output_path"], "request_output_path")
    journal_path = _path(exact["journal_path"], "journal_path")
    if (
        not journal_path.is_absolute()
        or journal_path.parent != output_directory
        or journal_path.exists()
        or journal_path.is_symlink()
    ):
        raise RequestBuildError("journal_path is not one new output-local path")

    loaded: dict[str, dict[str, object]] = {}
    for destination, source in _INPUT_PATH_FIELDS.items():
        value, _raw = _canonical_file(exact[source], source)
        if (
            "activation_id" in value
            and value["activation_id"] != exact["activation_id"]
        ):
            raise RequestBuildError(source + " activation_id drifted")
        loaded[destination] = value
    try:
        support_request = parse_support_input_materialization_request(
            loaded["support_input_materialization_request"]
        )
    except (TypeError, ValueError) as exc:
        raise RequestBuildError(
            "support materialization request is not canonical v2"
        ) from exc
    if support_request.activation_id != exact["activation_id"]:
        raise RequestBuildError("support request activation identity drifted")

    production_request = {
        "schema_version": 2,
        "record_type": "glm52_task13_production_operations_v2",
        "activation_id": exact["activation_id"],
        "output_directory": str(output_directory),
        "retained_stack_id": exact["retained_stack_id"],
        **loaded,
    }
    request = {
        "schema_version": 2,
        "record_type": "glm52_task13_staged_deployment_request_v2",
        "activation_id": exact["activation_id"],
        "production_request": production_request,
    }
    return request, request_output, evidence_output, journal_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invocation", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        invocation, _raw = _canonical_file(args.invocation, "invocation")
        request, request_output, evidence_output, journal_path = build_request(
            invocation
        )
        _write_new(request_output, canonical_json_bytes(request) + b"\n")
    except (OSError, RequestBuildError, TypeError, ValueError) as exc:
        print("Task 13 staged request build refused: " + str(exc), file=sys.stderr)
        return 64
    print(
        "BUILT "
        f"request={request_output} evidence={evidence_output} journal={journal_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
