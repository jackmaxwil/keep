#!/usr/bin/env python3
"""Write the immutable canonical GLM-5.2 Sky training configuration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "src/mlx_vq/quality/glm52_sky_training_config.py"
SPEC = importlib.util.spec_from_file_location("_glm52_sky_training_config", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
TRAINING_CONFIG = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAINING_CONFIG
SPEC.loader.exec_module(TRAINING_CONFIG)
build_sky_training_config = TRAINING_CONFIG.build_sky_training_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        build_sky_training_config(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode() + b"\n"
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
