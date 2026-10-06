#!/usr/bin/env python3
"""Project a validated production inventory into Sky object authorities."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "src/mlx_vq/quality/glm52_s3_artifact_audit.py"
SPEC = importlib.util.spec_from_file_location(
    "_glm52_project_production_inventory",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        inventory = AUDIT.validate_s3_artifact_inventory(
            json.loads(args.inventory.read_bytes())
        )
        objects = inventory["objects"]
        if not isinstance(objects, list) or not objects:
            raise ValueError("validated production inventory is empty")
        if any(item.get("run_scope") != "shared" for item in objects):
            raise ValueError(
                "production inventory projection accepts shared objects only"
            )
        if any(
            not isinstance(item.get("version_id"), str)
            or not item["version_id"]
            or item["version_id"] in {"null", "None"}
            for item in objects
        ):
            raise ValueError(
                "production inventory projection requires exact VersionIds"
            )
        repository_rows = [
            item for item in objects if item.get("kind") == "repository_tar"
        ]
        if len(repository_rows) != 1:
            raise ValueError(
                "production inventory requires exactly one repository tar"
            )
        repository = repository_rows[0]
        expected_repository_key = (
            f"campaigns/{inventory['run_id']}/repository/"
            f"keep-{repository['sha256']}.tar.gz"
        )
        if repository.get("key") != expected_repository_key:
            raise ValueError(
                "production inventory repository tar key is not content-addressed"
            )
        _write_atomic(args.output, objects)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
