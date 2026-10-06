#!/usr/bin/env python3
"""Rewrite the accepted baseline authority for the fixed campaign filesystem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-config", type=Path, required=True)
    parser.add_argument("--campaign-root", default="/mnt/nvme/glm52-campaign")
    parser.add_argument("--repo-root", default="/opt/keep-campaign/repo")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.accepted_config.read_bytes())
    campaign = Path(args.campaign_root)
    repo = Path(args.repo_root)
    rewritten = {
        **source,
        "tokenizer_dir": str(campaign / "source-snapshot"),
        "config_path": str(campaign / "source-snapshot/config.json"),
        "source_index_path": str(
            campaign / "source-snapshot/model.safetensors.index.json"
        ),
        "profile_path": str(repo / "models/glm52-reap-504b-v2.yaml"),
        "tokenizer_readiness_json": str(
            repo / "artifacts/quality/glm52-tokenizer-readiness-20260709.json"
        ),
        "family_policy_json": str(
            repo / "artifacts/quality/glm52-family-policy-20260709-v2.json"
        ),
        "non_vq_artifact_dir": str(campaign / "non-vq-package"),
        "non_vq_evidence_json": str(
            campaign / "training-baseline/non-vq-evidence.json"
        ),
        "routed_artifact_dir": str(
            campaign / "training-baseline/routed-artifacts"
        ),
        "composite_audit_json": str(
            repo / "artifacts/quality/glm52-wave6-full-artifact-audit-20260709.json"
        ),
        "materialization_runs_jsonl": str(
            repo
            / "artifacts/quality/glm52-wave6-full-materialization-20260709/materialization.jsonl"
        ),
        "full_bind_preflight_json": str(
            repo / "artifacts/quality/glm52-wave6-full-bind-preflight-20260709.json"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rewritten, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
