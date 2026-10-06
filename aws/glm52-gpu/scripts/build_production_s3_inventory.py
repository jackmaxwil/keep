#!/usr/bin/env python3
"""Build the exact shared production artifact inventory from pinned S3 authorities."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import boto3


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "src/mlx_vq/quality/glm52_s3_artifact_audit.py"
SPEC = importlib.util.spec_from_file_location("_glm52_s3_inventory", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)

SOURCE_TREE_KEY = (
    "source-snapshot/.cache/huggingface/trees/"
    "6c9241aa05fb243a0edb7c804c213ec1cf5c920d.json"
)
NON_VQ_MANIFEST_KEY = "non-vq-package/non-vq-manifest.json"
BASELINE_READY_KEY = "training-baseline/ROUTED_BASELINE_READY.json"
BASELINE_MANIFEST_KEY = (
    "training-baseline/routed-artifacts/conversion-manifest.json"
)
SHARED_SMALL_OBJECTS = (
    ("training-baseline/non-vq-evidence.json", "training_baseline"),
    ("training-baseline/training-baseline.json", "training_baseline"),
    ("teich-pack/glm52-coding-agent-initial-v2-20260713.json", "prompt_pack"),
    ("quality/glm52-family-eval-prompts-20260709-v2.json", "prompt_pack"),
)
VALIDATION_PREFIX = "validation/"


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _version_id(value: object, *, key: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {"null", "None"}
    ):
        raise ValueError(f"{key} is not protected by an opaque S3 VersionId")
    return value


def _read_object(
    s3: Any,
    *,
    bucket: str,
    key: str,
) -> tuple[bytes, str, str]:
    response = s3.get_object(Bucket=bucket, Key=key)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    while True:
        block = response["Body"].read(8 * 1024 * 1024)
        if not block:
            break
        chunks.append(block)
        digest.update(block)
    return (
        b"".join(chunks),
        digest.hexdigest(),
        _version_id(response.get("VersionId"), key=key),
    )


def _head_size(
    s3: Any,
    *,
    bucket: str,
    key: str,
    expected: int | None,
) -> tuple[int, str]:
    head = s3.head_object(Bucket=bucket, Key=key)
    size = int(head["ContentLength"])
    if size <= 0:
        raise ValueError(f"{key} is empty")
    if expected is not None and size != expected:
        raise ValueError(f"{key} size mismatch: expected {expected}, got {size}")
    return size, _version_id(head.get("VersionId"), key=key)


def _record(
    *,
    key: str,
    size: int,
    sha256: str,
    kind: str,
    safetensors: bool,
    version_id: str,
) -> dict[str, object]:
    return {
        "key": key,
        "size": size,
        "sha256": sha256,
        "kind": kind,
        "safetensors": safetensors,
        "run_scope": "shared",
        "version_id": _version_id(version_id, key=key),
    }


def _non_vq_index_record(
    non_vq: dict[str, object],
    *,
    size: int,
    version_id: str,
) -> dict[str, object]:
    package_index_sha256 = non_vq.get("package_index_sha256")
    if (
        not isinstance(package_index_sha256, str)
        or len(package_index_sha256) != 64
        or any(character not in "0123456789abcdef" for character in package_index_sha256)
    ):
        raise ValueError("non-VQ package index SHA-256 authority is malformed")
    return _record(
        key="non-vq-package/model.safetensors.index.json",
        size=size,
        sha256=package_index_sha256,
        kind="non_vq_package",
        safetensors=False,
        version_id=version_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--qualification-cache-prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.profile == "default":
        parser.error("the default AWS profile is forbidden")
    subprocess.run(
        [str(REPO_ROOT / "aws/glm52-gpu/scripts/assert_rnd_aws_account.sh")],
        check=True,
        env={**os.environ, "AWS_PROFILE": args.profile},
    )

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3 = session.client("s3")
    objects: list[dict[str, object]] = []

    tree_raw, tree_sha256, tree_version_id = _read_object(
        s3,
        bucket=args.bucket,
        key=SOURCE_TREE_KEY,
    )
    tree = json.loads(tree_raw)
    files = tree.get("files") if isinstance(tree, dict) else None
    if not isinstance(files, dict):
        parser.error("source tree authority is malformed")
    objects.append(
        _record(
            key=SOURCE_TREE_KEY,
            size=len(tree_raw),
            sha256=tree_sha256,
            kind="source_model",
            safetensors=False,
            version_id=tree_version_id,
        )
    )
    for filename, authority in sorted(files.items()):
        if not isinstance(authority, dict):
            parser.error(f"source authority for {filename} is malformed")
        key = f"source-snapshot/{filename}"
        expected_size = int(authority["size"])
        if filename.endswith(".safetensors"):
            expected_sha256 = authority.get("lfs_sha256")
            expected_lfs_size = authority.get("lfs_size")
            if (
                not isinstance(expected_sha256, str)
                or expected_lfs_size != expected_size
            ):
                parser.error(f"{filename} lacks pinned LFS SHA-256 authority")
            size, version_id = _head_size(
                s3,
                bucket=args.bucket,
                key=key,
                expected=expected_size,
            )
            objects.append(
                _record(
                    key=key,
                    size=size,
                    sha256=expected_sha256,
                    kind="source_model",
                    safetensors=True,
                    version_id=version_id,
                )
            )
        else:
            raw, sha256, version_id = _read_object(
                s3,
                bucket=args.bucket,
                key=key,
            )
            if len(raw) != expected_size:
                raise ValueError(f"{key} source-tree size mismatch")
            objects.append(
                _record(
                    key=key,
                    size=len(raw),
                    sha256=sha256,
                    kind="source_model",
                    safetensors=False,
                    version_id=version_id,
                )
            )

    non_vq_raw, non_vq_sha256, non_vq_version_id = _read_object(
        s3,
        bucket=args.bucket,
        key=NON_VQ_MANIFEST_KEY,
    )
    non_vq = json.loads(non_vq_raw)
    if not isinstance(non_vq, dict) or non_vq.get("production_ready") is not True:
        parser.error("non-VQ package authority is not production-ready")
    objects.append(
        _record(
            key=NON_VQ_MANIFEST_KEY,
            size=len(non_vq_raw),
            sha256=non_vq_sha256,
            kind="non_vq_package",
            safetensors=False,
            version_id=non_vq_version_id,
        )
    )
    for shard in non_vq["shards"]:
        key = f"non-vq-package/{shard['filename']}"
        size, version_id = _head_size(
            s3,
            bucket=args.bucket,
            key=key,
            expected=int(shard["file_bytes"]),
        )
        objects.append(
            _record(
                key=key,
                size=size,
                sha256=str(shard["file_sha256"]),
                kind="non_vq_package",
                safetensors=True,
                version_id=version_id,
            )
        )
    non_vq_index_key = "non-vq-package/model.safetensors.index.json"
    non_vq_index_size, non_vq_index_version_id = _head_size(
        s3,
        bucket=args.bucket,
        key=non_vq_index_key,
        expected=None,
    )
    objects.append(
        _non_vq_index_record(
            non_vq,
            size=non_vq_index_size,
            version_id=non_vq_index_version_id,
        )
    )

    ready_raw, ready_sha256, ready_version_id = _read_object(
        s3,
        bucket=args.bucket,
        key=BASELINE_READY_KEY,
    )
    ready = json.loads(ready_raw)
    if (
        not isinstance(ready, dict)
        or ready.get("all_full_sha256_match") is not True
    ):
        parser.error("training baseline readiness authority is not accepted")
    objects.append(
        _record(
            key=BASELINE_READY_KEY,
            size=len(ready_raw),
            sha256=ready_sha256,
            kind="training_baseline",
            safetensors=False,
            version_id=ready_version_id,
        )
    )
    baseline_raw, baseline_sha256, baseline_version_id = _read_object(
        s3,
        bucket=args.bucket,
        key=BASELINE_MANIFEST_KEY,
    )
    if baseline_sha256 != ready["conversion_manifest_sha256"]:
        parser.error("training baseline conversion-manifest SHA-256 drift")
    baseline = json.loads(baseline_raw)
    groups = baseline.get("groups") if isinstance(baseline, dict) else None
    if not isinstance(groups, list) or len(groups) != ready["artifact_count"]:
        parser.error("training baseline group inventory mismatch")
    objects.append(
        _record(
            key=BASELINE_MANIFEST_KEY,
            size=len(baseline_raw),
            sha256=baseline_sha256,
            kind="training_baseline",
            safetensors=False,
            version_id=baseline_version_id,
        )
    )
    for group in groups:
        filename = Path(str(group["artifact_path"])).name
        key = f"training-baseline/routed-artifacts/{filename}"
        size, version_id = _head_size(
            s3,
            bucket=args.bucket,
            key=key,
            expected=int(group["artifact_bytes"]),
        )
        objects.append(
            _record(
                key=key,
                size=size,
                sha256=str(group["artifact_sha256"]),
                kind="training_baseline",
                safetensors=True,
                version_id=version_id,
            )
        )

    for key, kind in SHARED_SMALL_OBJECTS:
        raw, sha256, version_id = _read_object(
            s3,
            bucket=args.bucket,
            key=key,
        )
        objects.append(
            _record(
                key=key,
                size=len(raw),
                sha256=sha256,
                kind=kind,
                safetensors=False,
                version_id=version_id,
            )
        )

    prefixes = [(VALIDATION_PREFIX, "prompt_pack")]
    if args.qualification_cache_prefix:
        cache_prefix = args.qualification_cache_prefix.rstrip("/") + "/"
        if not cache_prefix.startswith("qualification-cache/"):
            parser.error("qualification cache prefix is outside its authority")
        prefixes.append((cache_prefix, "qualification_cache"))
    paginator = s3.get_paginator("list_objects_v2")
    for prefix, kind in prefixes:
        found = 0
        for page in paginator.paginate(Bucket=args.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item["Key"])
                raw, sha256, version_id = _read_object(
                    s3,
                    bucket=args.bucket,
                    key=key,
                )
                if not raw:
                    raise ValueError(f"{key} is empty")
                objects.append(
                    _record(
                        key=key,
                        size=len(raw),
                        sha256=sha256,
                        kind=kind,
                        safetensors=key.endswith(".safetensors"),
                        version_id=version_id,
                    )
                )
                found += 1
        if found == 0:
            parser.error(f"required S3 prefix is empty: {prefix}")

    inventory = AUDIT.build_s3_artifact_inventory(
        run_id=args.run_id,
        bucket=args.bucket,
        objects=objects,
    )
    _write_atomic(args.output, inventory)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
