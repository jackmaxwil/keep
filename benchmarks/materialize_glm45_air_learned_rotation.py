#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from keep.convert.inspect_hf import GLM45_AIR_MODEL_ID
from keep.quality.learned_rotation_materialization import (
    materialize_learned_rotation_projection_candidates,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a GLM-4.5-Air VQ artifact with learned orthogonal rotations.",
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--seed-artifact-dir", required=True)
    parser.add_argument("--rotation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--code-bits", type=int, choices=(8, 16), default=8)
    parser.add_argument("--scale-estimator", default="max_abs")
    parser.add_argument("--expert-workers", type=int, default=1)
    parser.add_argument("--no-strict-config", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = json.loads(Path(args.config_path).read_text(encoding="utf-8"))
    manifest = materialize_learned_rotation_projection_candidates(
        config,
        source_dir=args.source_dir,
        index_path=args.index_path,
        seed_artifact_dir=args.seed_artifact_dir,
        rotation_manifest_path=args.rotation_manifest,
        output_dir=args.output_dir,
        model_id=args.model_id,
        group_size=args.group_size,
        code_bits=args.code_bits,
        scale_estimator=args.scale_estimator,
        expert_workers=args.expert_workers,
        strict_config=not args.no_strict_config,
    )
    print(json.dumps({"status": "ok", "manifest": manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
