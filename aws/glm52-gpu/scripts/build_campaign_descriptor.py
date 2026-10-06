#!/usr/bin/env python3
"""Build the canonical descriptor consumed by Lambda, EC2 user data, and systemd."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--account-id", default="246813579024")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--capacity-reservation-id", required=True)
    parser.add_argument("--capacity-block-end", required=True)
    parser.add_argument("--availability-zone", default="us-west-2a")
    parser.add_argument("--subnet-id", required=True)
    parser.add_argument("--security-group-id", required=True)
    parser.add_argument("--launch-template-id", required=True)
    parser.add_argument("--launch-template-version", required=True)
    parser.add_argument("--instance-type", default="p5.48xlarge")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--repo-tar", type=Path, required=True)
    parser.add_argument("--repo-tar-key", required=True)
    parser.add_argument("--campaign-descriptor-key", required=True)
    parser.add_argument("--source-snapshot-prefix", default="source-snapshot/")
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--non-vq-prefix", default="non-vq-package/")
    parser.add_argument("--non-vq-package-sha256", required=True)
    parser.add_argument(
        "--teich-pack-key",
        default="teich-pack/glm52-coding-agent-initial-v2-20260713.json",
    )
    parser.add_argument("--teich-pack-sha256", required=True)
    parser.add_argument(
        "--frozen-prompt-pack-key",
        default="quality/glm52-family-eval-prompts-20260709-v2.json",
    )
    parser.add_argument("--frozen-prompt-pack-sha256", required=True)
    parser.add_argument("--training-baseline-prefix", default="training-baseline/")
    parser.add_argument("--training-baseline-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_sha = hashlib.sha256(args.repo_tar.read_bytes()).hexdigest()
    body = {
        "schema_version": 1,
        "run_id": args.run_id,
        "account_id": args.account_id,
        "region": args.region,
        "capacity_reservation_id": args.capacity_reservation_id,
        "capacity_block_end": args.capacity_block_end,
        "availability_zone": args.availability_zone,
        "subnet_id": args.subnet_id,
        "security_group_id": args.security_group_id,
        "launch_template_id": args.launch_template_id,
        "launch_template_version": str(args.launch_template_version),
        "instance_type": args.instance_type,
        "instance_count": 1,
        "bucket": args.bucket,
        "repo_tar_key": args.repo_tar_key,
        "repo_tar_sha256": repo_sha,
        "campaign_descriptor_key": args.campaign_descriptor_key,
        "artifacts": {
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
        },
    }
    raw_body = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    descriptor = {
        **body,
        "descriptor_body_sha256": hashlib.sha256(raw_body).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
