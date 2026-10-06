"""Deterministic JSON encoding and hashing for enforcement records."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number is not valid canonical JSON")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(key)
            _reject_non_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_finite(item)


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic, whitespace-free UTF-8 JSON bytes."""

    _reject_non_finite(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 of a value's canonical JSON bytes."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
