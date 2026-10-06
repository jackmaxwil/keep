"""Descriptor dispatch that keeps SkyPilot isolated from Capacity Block launch code."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping

from mlx_vq.quality.glm52_sky_campaign import validate_sky_campaign_descriptor


_HEX64 = re.compile(r"[0-9a-f]{64}")


class RuntimeCampaignKind(str, Enum):
    CAPACITY_BLOCK = "capacity-block-v1"
    SKYPILOT = "skypilot-v2"


@dataclass(frozen=True)
class RuntimeAuthority:
    kind: RuntimeCampaignKind
    execution_deadline: datetime
    gpu_allocation_sha256: str | None
    gpu_spend_authority_sha256: str | None


def _parse_time(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 string") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def validate_runtime_descriptor(
    value: Mapping[str, object],
) -> tuple[dict[str, object], RuntimeCampaignKind]:
    """Validate v2 without importing the legacy controller on the Sky path."""

    if value.get("record_type") == "glm52_sky_campaign_descriptor_v2":
        return validate_sky_campaign_descriptor(value), RuntimeCampaignKind.SKYPILOT
    if value.get("schema_version") == 1 and "record_type" not in value:
        legacy = importlib.import_module(
            "mlx_vq.quality.glm52_capacity_block_controller"
        )
        return (
            legacy.validate_campaign_descriptor(value),
            RuntimeCampaignKind.CAPACITY_BLOCK,
        )
    raise ValueError("campaign descriptor is neither supported v1 nor SkyPilot v2")


def resolve_runtime_authority(
    descriptor: Mapping[str, object],
    *,
    environ: Mapping[str, str],
) -> RuntimeAuthority:
    """Resolve the deadline and ledger identities for the selected orchestrator."""

    if descriptor.get("record_type") == "glm52_sky_campaign_descriptor_v2":
        deadline_value = environ.get("GLM52_EXECUTION_DEADLINE")
        if not deadline_value:
            raise ValueError(
                "GLM52_EXECUTION_DEADLINE is required for a SkyPilot campaign"
            )
        allocation_sha = environ.get("GLM52_GPU_ALLOCATION_SHA256")
        if allocation_sha is None or _HEX64.fullmatch(allocation_sha) is None:
            raise ValueError(
                "GLM52_GPU_ALLOCATION_SHA256 is required and must be SHA-256"
            )
        spend_sha = environ.get("GLM52_GPU_SPEND_AUTHORITY_SHA256")
        if spend_sha is None or _HEX64.fullmatch(spend_sha) is None:
            raise ValueError(
                "GLM52_GPU_SPEND_AUTHORITY_SHA256 is required and must be SHA-256"
            )
        return RuntimeAuthority(
            kind=RuntimeCampaignKind.SKYPILOT,
            execution_deadline=_parse_time(
                deadline_value, field="GLM52_EXECUTION_DEADLINE"
            ),
            gpu_allocation_sha256=allocation_sha,
            gpu_spend_authority_sha256=spend_sha,
        )
    return RuntimeAuthority(
        kind=RuntimeCampaignKind.CAPACITY_BLOCK,
        execution_deadline=_parse_time(
            descriptor.get("capacity_block_end"), field="capacity_block_end"
        ),
        gpu_allocation_sha256=None,
        gpu_spend_authority_sha256=None,
    )


__all__ = [
    "RuntimeAuthority",
    "RuntimeCampaignKind",
    "resolve_runtime_authority",
    "validate_runtime_descriptor",
]
