#!/usr/bin/env python3
"""Build H100_RESUME_READY.json from authenticated qualification outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "src/mlx_vq/quality/glm52_h100_qualification.py"
SPEC = importlib.util.spec_from_file_location("_glm52_h100_ready", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_atomic(path: Path, value: object) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--campaign-identity-sha256", required=True)
    parser.add_argument("--repo-tar-sha256", required=True)
    parser.add_argument("--qualification-cache-manifest-sha256", required=True)
    parser.add_argument("--first-instance-id", required=True)
    parser.add_argument("--replacement-instance-id", required=True)
    parser.add_argument("--first-allocation-record-sha256", required=True)
    parser.add_argument("--replacement-allocation-record-sha256", required=True)
    parser.add_argument("--source-checkpoint-marker-sha256", required=True)
    parser.add_argument("--parity-report-sha256", required=True)
    parser.add_argument("--training-smoke-sha256", required=True)
    parser.add_argument("--resumed-capture-sha256", required=True)
    parser.add_argument("--peak-gpu-gib", required=True, type=float)
    parser.add_argument("--completed-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        value = MODULE.build_h100_resume_ready(
            run_id=args.run_id,
            campaign_identity_sha256=args.campaign_identity_sha256,
            repo_tar_sha256=args.repo_tar_sha256,
            qualification_cache_manifest_sha256=(
                args.qualification_cache_manifest_sha256
            ),
            first_instance_id=args.first_instance_id,
            replacement_instance_id=args.replacement_instance_id,
            first_allocation_record_sha256=args.first_allocation_record_sha256,
            replacement_allocation_record_sha256=(
                args.replacement_allocation_record_sha256
            ),
            source_checkpoint_marker_sha256=(
                args.source_checkpoint_marker_sha256
            ),
            parity_report_sha256=args.parity_report_sha256,
            training_smoke_sha256=args.training_smoke_sha256,
            resumed_capture_sha256=args.resumed_capture_sha256,
            peak_gpu_gib=args.peak_gpu_gib,
            completed_at=datetime.fromisoformat(
                args.completed_at.replace("Z", "+00:00")
            ),
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    _write_atomic(args.output, value)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
