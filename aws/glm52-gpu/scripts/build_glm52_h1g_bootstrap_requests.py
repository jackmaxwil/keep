#!/usr/bin/env python3
"""Materialize write-once H.1g requests from authenticated live readbacks."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.fence_bootstrap_publication import (
    parse_bootstrap_publication_authority_v2,
    parse_bridge_seed_publication_authority_v2,
)
from glm52_enforcement.support_plane import (
    retained_fence_bootstrap_inputs_from_mapping,
    retained_fence_runtime_authority_from_mapping,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INVOCATION_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "retained_bootstrap_inputs_path",
        "bridge_seed_authority_path",
        "bootstrap_publication_authority_path",
        "retained_runtime_authority_path",
        "retained_bootstrap_inputs_output",
        "bridge_seed_request_output",
        "bootstrap_publication_request_output",
        "retained_runtime_inputs_output",
        "canonical_identity_sha256",
    }
)
_SOURCE_PATH_FIELDS = (
    "retained_bootstrap_inputs_path",
    "bridge_seed_authority_path",
    "bootstrap_publication_authority_path",
    "retained_runtime_authority_path",
)
_OUTPUT_PATH_FIELDS = (
    "retained_bootstrap_inputs_output",
    "bridge_seed_request_output",
    "bootstrap_publication_request_output",
    "retained_runtime_inputs_output",
)
_SEED_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "activation_id",
        "generation",
        "expected_live_preseed_policy_sha256",
        "renderer_input",
    }
)
_BOOTSTRAP_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "activation_id",
        "generation",
        "renderer_inputs",
        "manifest_authority",
    }
)
_RUNTIME_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "activation_id",
        "retained_stack_id",
        "fence_service_role_arn",
        "model_bucket_arn",
        "retained_kms_key_arn",
        "ledger_table_arn",
        "lambda_code_bucket",
        "lambda_code_key",
        "lambda_code_version_id",
        "lambda_code_sha256",
    }
)


class BootstrapRequestMaterializationError(ValueError):
    """A request source or write-once target was not exact."""


def _text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or not value.isascii()
        or value != value.strip()
        or any(character in value for character in ("\n", "\r", "\x00"))
    ):
        raise BootstrapRequestMaterializationError(
            label + " must be one exact ASCII string"
        )
    return value


def _canonical_file(value: object, label: str) -> dict[str, object]:
    path = Path(_text(value, label))
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise BootstrapRequestMaterializationError(
            label + " must be one exact regular non-symlink file"
        )
    raw = path.read_bytes()
    try:
        parsed = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BootstrapRequestMaterializationError(
            label + " is not canonical ASCII JSON"
        ) from exc
    if type(parsed) is not dict or raw != canonical_json_bytes(parsed) + b"\n":
        raise BootstrapRequestMaterializationError(
            label + " is not canonical JSON plus LF"
        )
    return parsed


def _new_output(value: object, label: str) -> Path:
    path = Path(_text(value, label))
    if (
        not path.is_absolute()
        or path.resolve(strict=False) != path
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.parent.resolve(strict=True) != path.parent
        or path.exists()
        or path.is_symlink()
    ):
        raise BootstrapRequestMaterializationError(
            label + " must be one new exact absolute file"
        )
    return path


def _exact_authority(
    value: object,
    *,
    fields: frozenset[str],
    record_type: str,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise BootstrapRequestMaterializationError(label + " fields are not exact")
    if value.get("schema_version") != 2 or value.get("record_type") != record_type:
        raise BootstrapRequestMaterializationError(label + " identity is not exact")
    return dict(value)


def _validated(label: str, operation):
    try:
        return operation()
    except BootstrapRequestMaterializationError:
        raise
    except (TypeError, ValueError) as exc:
        raise BootstrapRequestMaterializationError(
            label + " identity is not exact"
        ) from exc


def _guard_invocation(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _INVOCATION_FIELDS:
        raise BootstrapRequestMaterializationError(
            "bootstrap materialization invocation fields are not exact"
        )
    if (
        value["schema_version"] != 2
        or value["record_type"] != "glm52_h1g_bootstrap_request_materialization_v2"
    ):
        raise BootstrapRequestMaterializationError(
            "bootstrap materialization invocation identity is not exact"
        )
    unsigned = dict(value)
    identity = unsigned.pop("canonical_identity_sha256")
    if (
        type(identity) is not str
        or _SHA256.fullmatch(identity) is None
        or canonical_sha256(unsigned) != identity
    ):
        raise BootstrapRequestMaterializationError(
            "bootstrap materialization invocation canonical identity drifted"
        )
    sources = tuple(_text(value[field], field) for field in _SOURCE_PATH_FIELDS)
    outputs = tuple(_text(value[field], field) for field in _OUTPUT_PATH_FIELDS)
    if len(set(sources + outputs)) != len(sources) + len(outputs):
        raise BootstrapRequestMaterializationError(
            "bootstrap materialization paths must be distinct"
        )
    return dict(value)


def _write_new_batch(items: tuple[tuple[Path, bytes], ...]) -> None:
    descriptors: list[tuple[Path, int]] = []
    try:
        for path, _raw in items:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            descriptors.append((path, descriptor))
        for (path, descriptor), (expected_path, raw) in zip(
            descriptors, items, strict=True
        ):
            if path != expected_path:
                raise AssertionError("write-once descriptor ordering drifted")
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise OSError("bootstrap request write made no progress")
                offset += written
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
    except Exception:
        for path, descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                path.unlink()
            except OSError:
                pass
        raise
    else:
        for _path, descriptor in descriptors:
            os.close(descriptor)


def build_requests(invocation: object) -> dict[str, object]:
    """Validate and copy only authority known before the staged effects."""

    exact = _guard_invocation(invocation)
    source = {
        field: _canonical_file(exact[field], field) for field in _SOURCE_PATH_FIELDS
    }
    outputs = tuple(_new_output(exact[field], field) for field in _OUTPUT_PATH_FIELDS)

    bootstrap_inputs_value = source["retained_bootstrap_inputs_path"]
    bootstrap_inputs = _validated(
        "retained bootstrap inputs",
        lambda: retained_fence_bootstrap_inputs_from_mapping(bootstrap_inputs_value),
    )
    seed_authority = _exact_authority(
        source["bridge_seed_authority_path"],
        fields=_SEED_AUTHORITY_FIELDS,
        record_type="glm52_h1g_bridge_seed_publication_authority_v2",
        label="bridge seed authority",
    )
    parsed_seed = _validated(
        "bridge seed authority",
        lambda: parse_bridge_seed_publication_authority_v2(seed_authority),
    )
    if parsed_seed.activation_id != bootstrap_inputs.activation_id:
        raise BootstrapRequestMaterializationError(
            "bridge seed activation identity drifted"
        )

    bootstrap_authority = _exact_authority(
        source["bootstrap_publication_authority_path"],
        fields=_BOOTSTRAP_AUTHORITY_FIELDS,
        record_type="glm52_fence_bootstrap_publication_authority_v2",
        label="bootstrap publication authority",
    )
    parsed_bootstrap = _validated(
        "bootstrap publication authority",
        lambda: parse_bootstrap_publication_authority_v2(bootstrap_authority),
    )
    if parsed_bootstrap.activation_id != bootstrap_inputs.activation_id:
        raise BootstrapRequestMaterializationError(
            "bootstrap publication activation identity drifted"
        )

    runtime_authority = _exact_authority(
        source["retained_runtime_authority_path"],
        fields=_RUNTIME_AUTHORITY_FIELDS,
        record_type="glm52_h1g_retained_fence_runtime_authority_v2",
        label="retained runtime authority",
    )
    runtime_inputs = _validated(
        "retained runtime authority",
        lambda: retained_fence_runtime_authority_from_mapping(runtime_authority),
    )
    if (
        runtime_inputs["activation_id"] != bootstrap_inputs.activation_id
        or runtime_inputs["retained_stack_id"] != bootstrap_inputs.retained_stack_id
        or runtime_inputs["lambda_code_sha256"] != bootstrap_inputs.lambda_code_sha256
        or runtime_inputs["lambda_code_version_id"]
        != bootstrap_inputs.lambda_code_version_id
        or runtime_inputs["retained_kms_key_arn"]
        != bootstrap_inputs.retained_kms_key_arn
    ):
        raise BootstrapRequestMaterializationError(
            "retained runtime bootstrap lineage drifted"
        )

    values = (
        dict(bootstrap_inputs_value),
        seed_authority,
        bootstrap_authority,
        runtime_authority,
    )
    raw_items = tuple(
        (path, canonical_json_bytes(value) + b"\n")
        for path, value in zip(outputs, values, strict=True)
    )
    _write_new_batch(raw_items)
    return {
        "retained_bootstrap_inputs": values[0],
        "bridge_seed_publication": values[1],
        "bootstrap_fence_publication": values[2],
        "retained_fence_runtime_deployment": values[3],
        "output_paths": tuple(str(path) for path in outputs),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invocation", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        invocation = _canonical_file(args.invocation, "invocation")
        built = build_requests(invocation)
    except (
        BootstrapRequestMaterializationError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            "H.1g bootstrap request materialization refused: " + str(exc),
            file=sys.stderr,
        )
        return 64
    print("BUILT " + " ".join(built["output_paths"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
