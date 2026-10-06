#!/usr/bin/env python3
"""Prepare the canonical inputs for a Task 13 staged request build.

This command consumes the two canonical retained-foundation transaction
records and explicit Phase Five input coordinates.  It emits the compact
foundation evidence, the foundation-only fixed-artifact set, and the
self-hashed invocation consumed by the guarded request builder.
"""
# ruff: noqa: E402

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
from glm52_enforcement.task13_migration_adapter import (
    validate_stack_migration_seed_projection,
)

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
STACK_NAME = "keep-glm52-gpu"
MODEL_BUCKET_NAME = "keep-glm52-models-246813579024-us-west-2"
FOUNDATION_TEMPLATE_KEY = "task13/templates/retained-foundation.yaml"
FOUNDATION_CHANGE_SET_NAME = "glm52-task13-retained-foundation-v1"
FOUNDATION_CHANGE_RECORD = "glm52_task13_retained_foundation_change_set_v1"
FOUNDATION_READBACK_RECORD = "glm52_task13_retained_foundation_readback_v1"
INVOCATION_RECORD = "glm52_task13_staged_deployment_request_build_v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ACTIVATION_ID = re.compile(r"[a-z0-9][a-z0-9-]{2,63}\Z")
_STACK_ID = re.compile(
    r"arn:aws:cloudformation:us-west-2:246813579024:"
    r"stack/keep-glm52-gpu/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_CHANGE_SET_ID = re.compile(
    r"arn:aws:cloudformation:us-west-2:246813579024:"
    r"changeSet/glm52-task13-retained-foundation-v1/"
    r"[A-Za-z0-9][-A-Za-z0-9]{0,127}\Z"
)
_COORDINATE_FIELDS = frozenset(
    {
        "artifact_kind",
        "bucket",
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
    }
)
_CLOUDTRAIL_EVENT_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_CLOUDTRAIL_EVENT_TIME = re.compile(
    r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z"
)
_CHANGE_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "profile",
        "stack_id",
        "stack_name",
        "termination_protection",
        "before_template_sha256",
        "after_template_sha256",
        "template_coordinate",
        "template_url",
        "change_set_id",
        "change_set_name",
        "client_token",
        "role_arn",
        "parameters",
        "capabilities",
        "changes",
    }
)
_READBACK_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "profile",
        "stack_id",
        "stack_name",
        "stack_status",
        "termination_protection",
        "role_arn",
        "template_sha256",
        "template_coordinate",
        "resource_count",
        "foundation_resources",
        "deployment_role",
        "fence_role",
        "kms_key_arn",
        "ledger_arn",
        "required_exports",
        "execute_response_was_ambiguous",
    }
)
_RECOVERY_READBACK_RECORD = "glm52_retained_foundation_recovery_readback_v1"
_RECOVERY_READBACK_EVIDENCE_FIELDS = _READBACK_EVIDENCE_FIELDS | {
    "source_change_set_evidence_sha256",
    "historical_termination_protection",
    "termination_protection_sealed_during_recovery",
    "termination_protection_update_was_ambiguous",
    "execute_cloudtrail_event_id",
    "execute_cloudtrail_event_time",
}


class InvocationPreparationError(ValueError):
    """The Phase Five invocation could not be prepared exactly."""


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
        raise InvocationPreparationError(
            label + " must be one exact nonempty ASCII string"
        )
    return value


def _exact_directory(path: Path, label: str) -> Path:
    if (
        not path.is_absolute()
        or not path.is_dir()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise InvocationPreparationError(
            label + " must be one exact absolute non-symlink directory"
        )
    return path


def _exact_file(path: Path, label: str) -> Path:
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise InvocationPreparationError(
            label + " must be one exact absolute regular non-symlink file"
        )
    return path


def _canonical_file(path: Path, label: str) -> tuple[object, bytes]:
    source = _exact_file(path, label)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise InvocationPreparationError(
            label + " is not canonical ASCII JSON"
        ) from exc
    if raw != canonical_json_bytes(value) + b"\n":
        raise InvocationPreparationError(label + " is not canonical JSON plus LF")
    return value, raw


def _new_target(path: Path, label: str) -> Path:
    if (
        not path.is_absolute()
        or path.resolve(strict=False) != path
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.parent.resolve(strict=True) != path.parent
        or path.exists()
        or path.is_symlink()
    ):
        raise InvocationPreparationError(label + " must be one new exact absolute file")
    return path


def _coordinate(value: object, *, label: str, kind: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != _COORDINATE_FIELDS:
        raise InvocationPreparationError(label + " coordinate schema mismatch")
    if (
        value["artifact_kind"] != kind
        or value["bucket"] != MODEL_BUCKET_NAME
        or type(value["key"]) is not str
        or not value["key"]
        or type(value["version_id"]) is not str
        or not value["version_id"]
        or value["version_id"] in {"null", "None"}
        or type(value["file_sha256"]) is not str
        or _SHA256.fullmatch(value["file_sha256"]) is None
        or type(value["body_sha256"]) is not str
        or _SHA256.fullmatch(value["body_sha256"]) is None
    ):
        raise InvocationPreparationError(label + " coordinate is not exact")
    return dict(value)


def _identity_fields_match(
    value: dict[str, object],
    *,
    record_type: str,
    stack_id: str,
) -> bool:
    return (
        value.get("schema_version") == 1
        and value.get("record_type") == record_type
        and value.get("account_id") == ACCOUNT_ID
        and value.get("region") == REGION
        and value.get("profile") == PROFILE
        and value.get("stack_id") == stack_id
        and value.get("stack_name") == STACK_NAME
    )


def _compact_foundation_evidence(
    *,
    change_value: object,
    readback_value: object,
    retained_stack_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    recovery_readback = (
        type(readback_value) is dict
        and readback_value.get("record_type") == _RECOVERY_READBACK_RECORD
    )
    expected_readback_fields = (
        _RECOVERY_READBACK_EVIDENCE_FIELDS
        if recovery_readback
        else _READBACK_EVIDENCE_FIELDS
    )
    if (
        type(change_value) is not dict
        or set(change_value) != _CHANGE_EVIDENCE_FIELDS
        or type(readback_value) is not dict
        or set(readback_value) != expected_readback_fields
    ):
        raise InvocationPreparationError(
            "foundation transaction evidence schema mismatch"
        )
    if not _identity_fields_match(
        change_value,
        record_type=FOUNDATION_CHANGE_RECORD,
        stack_id=retained_stack_id,
    ) or not _identity_fields_match(
        readback_value,
        record_type=(
            _RECOVERY_READBACK_RECORD
            if recovery_readback
            else FOUNDATION_READBACK_RECORD
        ),
        stack_id=retained_stack_id,
    ):
        raise InvocationPreparationError("foundation transaction identity mismatch")
    if readback_value.get("termination_protection") is not True or (
        not recovery_readback and change_value.get("termination_protection") is not True
    ):
        raise InvocationPreparationError(
            "retained termination protection is not enabled"
        )
    if recovery_readback and (
        type(change_value.get("termination_protection")) is not bool
        or readback_value.get("historical_termination_protection")
        is not change_value["termination_protection"]
        or readback_value.get("source_change_set_evidence_sha256")
        != _sha256(canonical_json_bytes(change_value))
        or readback_value.get("execute_response_was_ambiguous") is not None
        or type(readback_value.get("termination_protection_sealed_during_recovery"))
        is not bool
        or type(readback_value.get("termination_protection_update_was_ambiguous"))
        is not bool
        or type(readback_value.get("execute_cloudtrail_event_id")) is not str
        or _CLOUDTRAIL_EVENT_ID.fullmatch(readback_value["execute_cloudtrail_event_id"])
        is None
        or type(readback_value.get("execute_cloudtrail_event_time")) is not str
        or _CLOUDTRAIL_EVENT_TIME.fullmatch(
            readback_value["execute_cloudtrail_event_time"]
        )
        is None
    ):
        raise InvocationPreparationError(
            "foundation recovery evidence lineage mismatch"
        )

    change_coordinate = _coordinate(
        change_value["template_coordinate"],
        label="foundation change-set template",
        kind="RETAINED_FOUNDATION_TEMPLATE",
    )
    readback_coordinate = _coordinate(
        readback_value["template_coordinate"],
        label="foundation readback template",
        kind="RETAINED_FOUNDATION_TEMPLATE",
    )
    body_sha256 = change_coordinate["body_sha256"]
    change_set_id = change_value["change_set_id"]
    if (
        change_coordinate != readback_coordinate
        or change_coordinate["key"] != FOUNDATION_TEMPLATE_KEY
        or change_value.get("change_set_name") != FOUNDATION_CHANGE_SET_NAME
        or type(change_set_id) is not str
        or _CHANGE_SET_ID.fullmatch(change_set_id) is None
        or change_value.get("after_template_sha256") != body_sha256
        or readback_value.get("template_sha256") != body_sha256
        or readback_value.get("stack_status") != "UPDATE_COMPLETE"
        or change_value.get("role_arn") != readback_value.get("role_arn")
    ):
        raise InvocationPreparationError("foundation transaction identity mismatch")

    compact = {
        "stack_id": retained_stack_id,
        "stack_status": "UPDATE_COMPLETE",
        "change_set_id": change_set_id,
        "template_coordinate": change_coordinate,
        "template_body_sha256": body_sha256,
        "readback_sha256": _sha256(canonical_json_bytes(readback_value)),
    }
    return compact, change_coordinate


def _write_new(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("output write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _write_transaction(outputs: list[tuple[Path, bytes]]) -> None:
    created: list[Path] = []
    try:
        for path, raw in outputs:
            _write_new(path, raw)
            created.append(path)
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass
        for parent in {path.parent for path in created}:
            try:
                descriptor = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-id", required=True)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--journal-path", required=True, type=Path)
    parser.add_argument(
        "--staged-infrastructure-evidence-output",
        required=True,
        type=Path,
    )
    parser.add_argument("--retained-stack-id", required=True)
    parser.add_argument(
        "--bootstrap-template-coordinate",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--pre-support-runtime-inputs",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--support-input-materialization-request",
        required=True,
        type=Path,
    )
    parser.add_argument("--support-price-card", required=True, type=Path)
    parser.add_argument("--orphan-precreate", required=True, type=Path)
    parser.add_argument(
        "--support-artifact-publication",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--stack-migration-seed",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--retained-foundation-change-set-evidence",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--retained-foundation-readback-evidence",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--retained-foundation-evidence-output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--fixed-artifacts-output",
        required=True,
        type=Path,
    )
    parser.add_argument("--invocation-output", required=True, type=Path)
    parser.add_argument("--request-output", required=True, type=Path)
    return parser


def prepare(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    activation_id = _text(args.activation_id, "activation_id")
    if _ACTIVATION_ID.fullmatch(activation_id) is None:
        raise InvocationPreparationError("activation_id is not exact")
    output_directory = _exact_directory(args.output_directory, "output_directory")
    retained_stack_id = _text(args.retained_stack_id, "retained_stack_id")
    if _STACK_ID.fullmatch(retained_stack_id) is None:
        raise InvocationPreparationError("retained_stack_id is not exact")
    source_arguments = {
        "bootstrap_template_coordinate": (args.bootstrap_template_coordinate),
        "pre_support_runtime_inputs": args.pre_support_runtime_inputs,
        "support_input_materialization_request": (
            args.support_input_materialization_request
        ),
        "support_price_card": args.support_price_card,
        "orphan_precreate": args.orphan_precreate,
        "support_artifact_publication": (args.support_artifact_publication),
        "stack_migration_seed": args.stack_migration_seed,
        "retained_foundation_change_set_evidence": (
            args.retained_foundation_change_set_evidence
        ),
        "retained_foundation_readback_evidence": (
            args.retained_foundation_readback_evidence
        ),
    }
    source_values: dict[str, object] = {}
    source_paths: dict[str, Path] = {}
    for label, path in source_arguments.items():
        value, _raw = _canonical_file(path, label)
        source_values[label] = value
        source_paths[label] = path

    _coordinate(
        source_values["bootstrap_template_coordinate"],
        label="bootstrap template",
        kind="BOOTSTRAP_TEMPLATE",
    )
    try:
        validate_stack_migration_seed_projection(source_values["stack_migration_seed"])
    except (TypeError, ValueError) as exc:
        raise InvocationPreparationError("stack migration seed is not exact") from exc
    compact, foundation_coordinate = _compact_foundation_evidence(
        change_value=source_values["retained_foundation_change_set_evidence"],
        readback_value=source_values["retained_foundation_readback_evidence"],
        retained_stack_id=retained_stack_id,
    )

    compact_output = _new_target(
        args.retained_foundation_evidence_output,
        "retained_foundation_evidence_output",
    )
    fixed_output = _new_target(args.fixed_artifacts_output, "fixed_artifacts_output")
    invocation_output = _new_target(args.invocation_output, "invocation_output")
    request_output = _new_target(args.request_output, "request_output")
    journal_path = _new_target(args.journal_path, "journal_path")
    staged_evidence_output = _new_target(
        args.staged_infrastructure_evidence_output,
        "staged_infrastructure_evidence_output",
    )
    expected_staged_evidence = (
        output_directory / "staged-infrastructure-evidence-v1.json"
    )
    if staged_evidence_output != expected_staged_evidence:
        raise InvocationPreparationError(
            "staged_infrastructure_evidence_output does not match output_directory"
        )

    all_targets = {
        compact_output,
        fixed_output,
        invocation_output,
        request_output,
        journal_path,
        staged_evidence_output,
    }
    if len(all_targets) != 6 or all_targets & set(source_paths.values()):
        raise InvocationPreparationError("source and target paths must all be distinct")

    invocation_body = {
        "schema_version": 1,
        "record_type": INVOCATION_RECORD,
        "activation_id": activation_id,
        "output_directory": str(output_directory),
        "journal_path": str(journal_path),
        "staged_infrastructure_evidence_output": str(staged_evidence_output),
        "retained_stack_id": retained_stack_id,
        "bootstrap_template_coordinate_path": str(
            source_paths["bootstrap_template_coordinate"]
        ),
        "pre_support_runtime_inputs_path": str(
            source_paths["pre_support_runtime_inputs"]
        ),
        "support_input_materialization_request_path": str(
            source_paths["support_input_materialization_request"]
        ),
        "support_price_card_path": str(source_paths["support_price_card"]),
        "orphan_precreate_path": str(source_paths["orphan_precreate"]),
        "retained_foundation_evidence_path": str(compact_output),
        "fixed_artifacts_path": str(fixed_output),
        "support_artifact_publication_path": str(
            source_paths["support_artifact_publication"]
        ),
        "stack_migration_seed_path": str(source_paths["stack_migration_seed"]),
        "request_output_path": str(request_output),
    }
    invocation = {
        **invocation_body,
        "canonical_identity_sha256": _sha256(canonical_json_bytes(invocation_body)),
    }
    _write_transaction(
        [
            (
                compact_output,
                canonical_json_bytes(compact) + b"\n",
            ),
            (
                fixed_output,
                canonical_json_bytes([foundation_coordinate]) + b"\n",
            ),
            (
                invocation_output,
                canonical_json_bytes(invocation) + b"\n",
            ),
        ]
    )
    return compact_output, fixed_output, invocation_output


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        compact_output, fixed_output, invocation_output = prepare(args)
    except (
        InvocationPreparationError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            "Task 13 staged invocation preparation refused: " + str(exc),
            file=sys.stderr,
        )
        return 64
    print("retained_foundation_evidence=" + str(compact_output))
    print("fixed_artifacts=" + str(fixed_output))
    print("invocation=" + str(invocation_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
