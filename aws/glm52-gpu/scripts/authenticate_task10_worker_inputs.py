#!/usr/bin/env python3
"""Authenticate fixed worker-side Task 10 inputs before systemd startup."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import NoReturn


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from mlx_vq.quality.glm52_sky_production_submission import (  # noqa: E402
    production_submission_intent_file_bytes,
    validate_production_submission_intent,
)


INTENT = Path("/mnt/nvme/glm52-campaign/runtime/intent.json")


def _fail(message: str) -> NoReturn:
    print("Task 10 worker input authentication: " + message, file=sys.stderr)
    raise SystemExit(70)


def _expected(name: str) -> str:
    value = os.environ.get(name)
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(name + " is not a lowercase SHA-256")
    return value


def _authenticate_intent(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    expected_file = _expected("GLM52_SUBMISSION_INTENT_FILE_SHA256")
    expected_body = _expected("GLM52_SUBMISSION_INTENT_BODY_SHA256")
    if hashlib.sha256(raw).hexdigest() != expected_file:
        raise ValueError("production intent file identity drifted")
    value = json.loads(raw)
    if (
        type(value) is not dict
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        raise ValueError("production intent is not canonical JSON")
    validated = validate_production_submission_intent(value)
    if (
        validated != value
        or production_submission_intent_file_bytes(validated) != raw
        or validated.get("intent_body_sha256") != expected_body
    ):
        raise ValueError("production intent body identity drifted")
    return validated


def main(argv: list[str]) -> int:
    if argv:
        _fail("this authenticator accepts no arguments")
    try:
        _authenticate_intent(INTENT)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        _fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
