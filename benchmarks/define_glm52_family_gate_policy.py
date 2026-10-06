from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mlx_vq.quality.glm52_family import (
    GLM52_FAMILY_POLICY_RECORD_TYPE as POLICY_RECORD_TYPE,
    GLM52_FAMILY_POLICY_STATUS as POLICY_STATUS,
    PINNED_GLM52_MODEL_ID as PINNED_MODEL_ID,
    PINNED_GLM52_REVISION as PINNED_REVISION,
    define_glm52_family_gate_policy,
)


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().absolute()


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = _absolute(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write the immutable GLM-5.2 REAP source-relative family policy."
        )
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = define_glm52_family_gate_policy(
            model_id=args.model_id,
            revision=args.revision,
        )
    except Exception as error:
        payload = {
            "schema_version": 1,
            "record_type": POLICY_RECORD_TYPE,
            "policy_status": "glm52_family_policy_failed",
            "model_id": args.model_id,
            "revision": args.revision,
            "thresholds_frozen_before_candidate_metrics": False,
            "input_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    _write_json_atomic(args.output_json, payload)
    return 0 if payload.get("policy_status") == POLICY_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
