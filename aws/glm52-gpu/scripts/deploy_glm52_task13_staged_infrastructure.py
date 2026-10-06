#!/usr/bin/env python3
"""Execute the restart-safe Task 13 staged infrastructure route.

The executable adapter is pinned to the repository's concrete production
factory.  That factory composes the AWS helpers behind
``StagedDeploymentOperations``; the route itself remains the sole owner of
ordering and mutation restart semantics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.task13_production_operations import (  # noqa: E402
    build_staged_deployment_operations,
)
from glm52_enforcement.task13_staged_deployment import (  # noqa: E402
    StagedDeploymentError,
    StagedDeploymentOperations,
    StagedDeploymentRequest,
    run_staged_deployment,
)

_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "activation_id",
        "production_request",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _read_request(path: Path, journal: Path) -> StagedDeploymentRequest:
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise StagedDeploymentError(
            "staged deployment request must be one exact absolute file"
        )
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise StagedDeploymentError(
            "staged deployment request must end in exactly one LF"
        )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StagedDeploymentError(
            "staged deployment request is not ASCII JSON"
        ) from exc
    if (
        type(value) is not dict
        or set(value) != _REQUEST_FIELDS
        or _canonical_json_bytes(value) + b"\n" != raw
    ):
        raise StagedDeploymentError(
            "staged deployment request schema or canonical bytes mismatch"
        )
    return StagedDeploymentRequest(
        schema_version=value["schema_version"],
        record_type=value["record_type"],
        activation_id=value["activation_id"],
        journal_path=journal,
        production_request=value["production_request"],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request",
        required=True,
        type=Path,
        help="Canonical Task 13 staged deployment request",
    )
    parser.add_argument(
        "--journal",
        required=True,
        type=Path,
        help="Absolute append-only journal path",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    operations_factory: Callable[
        [Mapping[str, object]],
        StagedDeploymentOperations,
    ]
    | None = None,
) -> int:
    args = _parser().parse_args(argv)
    try:
        request = _read_request(
            args.request,
            args.journal,
        )
        factory = (
            operations_factory
            if operations_factory is not None
            else build_staged_deployment_operations
        )
        operations = factory(request.production_request)
        result = run_staged_deployment(request, operations)
    except (
        ImportError,
        OSError,
        RuntimeError,
        StagedDeploymentError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            "Task 13 staged infrastructure deployment refused: " + str(exc),
            file=sys.stderr,
        )
        return 64
    print(
        "COMMITTED "
        f"activation_id={result.activation_id} "
        f"journal_sha256={result.journal_sha256} "
        "staged_infrastructure_evidence="
        f"{result.staged_infrastructure_evidence_path} "
        "staged_infrastructure_identity_sha256="
        f"{result.staged_infrastructure_identity_sha256} "
        f"worker_activation_allowed={str(result.worker_activation_allowed).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
