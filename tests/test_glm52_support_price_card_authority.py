from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.support_price_card_authority import (
    build_approved_support_price_card,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "aws/glm52-gpu/scripts/materialize_glm52_support_price_card.py"
)


def test_approved_support_price_card_is_exact_and_self_bound() -> None:
    card = build_approved_support_price_card()

    assert card["schema_version"] == 1
    assert card["record_type"] == "glm52_h1g_support_price_card_v1"
    assert card["region"] == "us-west-2"
    assert card["effective_date"] == "2026-07-28"
    assert card["currency"] == "USD"
    assert len(card["support_terms"]) == 15
    assert len(card["retained_terms"]) == 6
    assert card["estimated_support_total_usd"] == "17.28"

    unsigned = dict(card)
    identity = unsigned.pop("price_card_identity_sha256")
    assert identity == hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()


def test_price_card_materializer_writes_canonical_bytes_once(
    tmp_path: Path,
) -> None:
    output = tmp_path / "support-price-card.json"

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    expected = build_approved_support_price_card()
    assert output.read_bytes() == canonical_json_bytes(expected) + b"\n"

    duplicate = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert duplicate.returncode != 0
    assert json.loads(output.read_text("ascii")) == expected


@pytest.mark.parametrize(
    "flag",
    ["--effective-date", "--unit-price-usd", "--estimated-total"],
)
def test_price_card_materializer_has_no_operator_price_override(
    flag: str,
) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), flag, "0"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
