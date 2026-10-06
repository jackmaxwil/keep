"""Frozen, owner-approved GLM-5.2 support-plane price authority.

The owner approved these dated terms exactly as written.  This module makes
those terms executable production input without exposing operator price
overrides or importing test fixtures.
"""

from __future__ import annotations

from decimal import Decimal
import hashlib
from typing import Iterable, Mapping

from .canonical import canonical_json_bytes


_SUPPORT_TERMS = (
    ("combined_host_hours", "instance-hour", "72.000000", "0.100000"),
    ("root_ebs", "gib-hour", "2160.000000", "0.000500"),
    ("data_ebs", "gib-hour", "3600.000000", "0.000300"),
    ("nat_gateway_hours", "gateway-hour", "72.000000", "0.050000"),
    ("nat_processed_gib", "gib", "10.000000", "0.050000"),
    ("eip_hours", "ipv4-hour", "72.000000", "0.005000"),
    ("secrets_eight", "secret-hour", "576.000000", "0.002000"),
    ("secrets_api_calls", "api-call", "20000.000000", "0.000010"),
    (
        "interface_endpoint_two_eni_hours",
        "eni-hour",
        "144.000000",
        "0.010000",
    ),
    ("cross_az_bytes", "gib", "10.000000", "0.010000"),
    ("lambda", "invocation", "10000.000000", "0.000010"),
    ("step_functions", "state-transition", "12000.000000", "0.000010"),
    ("logs_ingestion", "gib", "0.500000", "0.500000"),
    ("logs_storage", "gib-day", "7.000000", "0.010000"),
    ("rehearsal_s3", "gib-day", "3.000000", "0.010000"),
)

_RETAINED_TERMS = (
    ("retained_s3", "gib-day", "30.000000", "0.010000"),
    ("retained_ledger", "request", "129600.000000", "0.000001"),
    ("retained_kms", "key-month", "1.000000", "1.000000"),
    ("liability_watcher", "scan", "43200.000000", "0.000010"),
    ("snapshot_cleanup", "delete-call", "12.000000", "0.010000"),
    ("snapshot_retention", "gib-day", "350.000000", "0.001000"),
)


def _rows(
    terms: Iterable[tuple[str, str, str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "term": term,
            "unit": unit,
            "quantity": quantity,
            "unit_price_usd": unit_price,
            "estimated_usd": str(
                (Decimal(quantity) * Decimal(unit_price)).quantize(
                    Decimal("0.01")
                )
            ),
        }
        for term, unit, quantity, unit_price in terms
    ]


def build_approved_support_price_card() -> Mapping[str, object]:
    """Return the immutable 2026-07-28 owner-approved price card."""

    support_rows = _rows(_SUPPORT_TERMS)
    retained_rows = _rows(_RETAINED_TERMS)
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_price_card_v1",
        "region": "us-west-2",
        "effective_date": "2026-07-28",
        "currency": "USD",
        "support_terms": support_rows,
        "retained_terms": retained_rows,
        "estimated_support_total_usd": str(
            sum(
                (
                    Decimal(row["estimated_usd"])
                    for row in support_rows
                ),
                Decimal("0.00"),
            )
        ),
    }
    return {
        **body,
        "price_card_identity_sha256": hashlib.sha256(
            canonical_json_bytes(body)
        ).hexdigest(),
    }


__all__ = ["build_approved_support_price_card"]
