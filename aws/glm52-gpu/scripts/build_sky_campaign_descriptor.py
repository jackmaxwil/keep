#!/usr/bin/env python3
"""Build the strict immutable AWS-only SkyPilot campaign descriptor v2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "src/mlx_vq/quality/glm52_sky_campaign.py"
SPEC = importlib.util.spec_from_file_location("_glm52_sky_campaign", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
SKY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SKY
SPEC.loader.exec_module(SKY)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--must-start-by", required=True)
    parser.add_argument("--controller-identity", required=True)
    parser.add_argument("--worker-identity", required=True)
    parser.add_argument("--vpc-name", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--jobs-bucket")
    parser.add_argument("--repo-tar", type=Path, required=True)
    parser.add_argument("--repo-tar-key", required=True)
    parser.add_argument("--campaign-descriptor-key", required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--approval-key", required=True)
    parser.add_argument("--source-snapshot-prefix", required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--non-vq-prefix", required=True)
    parser.add_argument("--non-vq-package-sha256", required=True)
    parser.add_argument("--teich-pack-key", required=True)
    parser.add_argument("--teich-pack-sha256", required=True)
    parser.add_argument("--frozen-prompt-pack-key", required=True)
    parser.add_argument("--frozen-prompt-pack-sha256", required=True)
    parser.add_argument("--training-baseline-prefix", required=True)
    parser.add_argument("--training-baseline-sha256", required=True)
    parser.add_argument("--training-config-key", required=True)
    parser.add_argument("--training-config-sha256", required=True)
    parser.add_argument("--artifact-inventory-key", required=True)
    parser.add_argument("--artifact-inventory-sha256", required=True)
    parser.add_argument("--qualification-cache-prefix", required=True)
    parser.add_argument("--qualification-cache-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        approval_value = json.loads(args.approval.read_bytes())
        if not isinstance(approval_value, dict):
            raise ValueError("GPU spend approval must contain an object")
        SKY.validate_gpu_spend_approval(approval_value)
        must_start_by = datetime.fromisoformat(
            args.must_start_by.replace("Z", "+00:00")
        )
        descriptor = SKY.build_sky_campaign_descriptor(
            run_id=args.run_id,
            must_start_by=must_start_by,
            controller_identity=args.controller_identity,
            worker_identity=args.worker_identity,
            vpc_name=args.vpc_name,
            image_id=args.image_id,
            bucket=args.bucket,
            jobs_bucket=args.jobs_bucket or args.bucket,
            repo_tar_key=args.repo_tar_key,
            repo_tar_sha256=_sha256_file(args.repo_tar),
            campaign_descriptor_key=args.campaign_descriptor_key,
            approval_key=args.approval_key,
            approval_sha256=_sha256_file(args.approval),
            artifacts={
                "source_snapshot_prefix": args.source_snapshot_prefix,
                "source_snapshot_sha256": args.source_snapshot_sha256,
                "non_vq_prefix": args.non_vq_prefix,
                "non_vq_package_sha256": args.non_vq_package_sha256,
                "teich_pack_key": args.teich_pack_key,
                "teich_pack_sha256": args.teich_pack_sha256,
                "frozen_prompt_pack_key": args.frozen_prompt_pack_key,
                "frozen_prompt_pack_sha256": args.frozen_prompt_pack_sha256,
                "training_baseline_prefix": args.training_baseline_prefix,
                "training_baseline_sha256": args.training_baseline_sha256,
                "training_config_key": args.training_config_key,
                "training_config_sha256": args.training_config_sha256,
                "artifact_inventory_key": args.artifact_inventory_key,
                "artifact_inventory_sha256": args.artifact_inventory_sha256,
                "qualification_cache_prefix": args.qualification_cache_prefix,
                "qualification_cache_manifest_sha256": (
                    args.qualification_cache_manifest_sha256
                ),
            },
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        parser.error(str(error))
    _write_atomic(args.output, descriptor)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
