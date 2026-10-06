"""Pure production-generation authority tests.

Every test names the production mutation that it kills.  The fixture reuses
the already accepted H.1c/H.1d fixture producers, but all H.1e expectations
are literal and are exercised through H.1e's public API.
"""

from __future__ import annotations

import ast
import base64
import builtins
import copy
import hashlib
import http.client
import importlib
import importlib.util
import inspect
import json
import math
import multiprocessing
import os
import random
import secrets
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.request
from collections.abc import Mapping
from dataclasses import MISSING, FrozenInstanceError, fields, replace
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Literal, Optional, Union, get_args, get_type_hints

import pytest

import mlx_vq.quality.glm52_sky_production_generation as generation
from mlx_vq.quality.glm52_gpu_spend_snapshot import validate_gpu_spend_snapshot
from mlx_vq.quality.glm52_sky_campaign import (
    validate_gpu_spend_approval,
    validate_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_production_acquisition import (
    production_submission_acquired_file_bytes,
    production_submission_acquired_s3_key,
)
from mlx_vq.quality.glm52_sky_production_submission import (
    VersionedJsonArtifact,
    build_production_submission_intent,
    production_submission_intent_file_bytes,
    production_submission_intent_s3_key,
    validate_production_submission_intent,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "glm52-sky-20260724"
ATTEMPT_A = "a1" * 32
ATTEMPT_B = "b2" * 32
CLAIM_AT = datetime(2026, 7, 26, 12, 11, 10, tzinfo=timezone.utc)
DECIDED_AT = datetime(2026, 7, 26, 12, 11, 20, tzinfo=timezone.utc)
MUST_START_BY = datetime(2026, 7, 26, 12, 11, 30, tzinfo=timezone.utc)

EXPECTED_ALL = [
    "GenerationInventoryEntry",
    "ModeledSubmitOnceValidation",
    "ProductionGenerationAction",
    "ProductionGenerationAuthorityError",
    "ProductionGenerationDecision",
    "ProductionGenerationReason",
    "UnambiguousStartDecisionCreateReceipt",
    "build_production_generation_claim",
    "production_generation_claim_file_bytes",
    "production_generation_claim_file_sha256",
    "production_generation_claim_s3_key",
    "validate_production_generation_claim",
    "build_production_generation_start_decision",
    "production_generation_start_decision_file_bytes",
    "production_generation_start_decision_file_sha256",
    "production_generation_start_decision_s3_key",
    "validate_production_generation_start_decision",
    "build_production_generation_terminal",
    "production_generation_terminal_file_bytes",
    "production_generation_terminal_file_sha256",
    "production_generation_terminal_s3_key",
    "validate_production_generation_terminal",
    "validate_production_generation_inventory",
    "decide_production_generation_action",
    "validate_modeled_submit_once",
]

CLAIM_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
    "generation",
    "generation_text",
    "submit_attempt_id",
    "sky_job_name",
    "sky_job_identity_sha256",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "descriptor_version_id",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "intent_version_id",
    "controller_baseline_key",
    "controller_baseline_file_sha256",
    "controller_baseline_body_sha256",
    "controller_baseline_version_id",
    "must_start_control_plane_ready_key",
    "must_start_control_plane_ready_file_sha256",
    "must_start_control_plane_ready_body_sha256",
    "must_start_control_plane_ready_version_id",
    "submission_acquisition_key",
    "submission_acquisition_file_sha256",
    "submission_acquisition_body_sha256",
    "submission_acquisition_version_id",
    "approval_key",
    "approval_file_sha256",
    "approval_body_sha256",
    "approval_version_id",
    "gpu_spend_snapshot_key",
    "gpu_spend_snapshot_file_sha256",
    "gpu_spend_snapshot_body_sha256",
    "gpu_spend_snapshot_version_id",
    "gpu_spend_ledger_genesis_sha256",
    "gpu_spend_ledger_record_count",
    "gpu_spend_ledger_tip_record_sha256",
    "ec2_allocation_history_sha256",
    "approved_gpu_runtime_seconds",
    "approved_gpu_cost_usd",
    "hourly_cost_usd",
    "consumed_gpu_seconds",
    "consumed_gpu_cost_usd",
    "remaining_gpu_seconds",
    "remaining_gpu_cost_usd",
    "open_allocation_count",
    "previous_generation",
    "previous_generation_terminal_key",
    "previous_generation_terminal_file_sha256",
    "previous_generation_terminal_body_sha256",
    "previous_generation_terminal_version_id",
    "must_start_by",
    "claim_created_at",
    "generation_claim_body_sha256",
}

DECISION_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
    "generation",
    "generation_text",
    "submit_attempt_id",
    "sky_job_name",
    "sky_job_identity_sha256",
    "generation_claim_key",
    "generation_claim_file_sha256",
    "generation_claim_body_sha256",
    "generation_claim_version_id",
    "decision",
    "must_start_by",
    "decided_at",
    "start_decision_body_sha256",
}

TERMINAL_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "campaign_identity_sha256",
    "generation",
    "generation_text",
    "submit_attempt_id",
    "sky_job_name",
    "sky_job_identity_sha256",
    "generation_claim_key",
    "generation_claim_file_sha256",
    "generation_claim_body_sha256",
    "generation_claim_version_id",
    "start_decision_key",
    "start_decision_file_sha256",
    "start_decision_body_sha256",
    "start_decision_version_id",
    "terminal_outcome",
    "must_start_by",
    "terminal_at",
    "final_gpu_spend_snapshot_key",
    "final_gpu_spend_snapshot_file_sha256",
    "final_gpu_spend_snapshot_body_sha256",
    "final_gpu_spend_snapshot_version_id",
    "final_gpu_spend_ledger_genesis_sha256",
    "final_gpu_spend_ledger_record_count",
    "final_gpu_spend_ledger_tip_record_sha256",
    "final_ec2_allocation_history_sha256",
    "final_approved_gpu_runtime_seconds",
    "final_approved_gpu_cost_usd",
    "final_hourly_cost_usd",
    "final_consumed_gpu_seconds",
    "final_consumed_gpu_cost_usd",
    "final_remaining_gpu_seconds",
    "final_remaining_gpu_cost_usd",
    "final_open_allocation_count",
    "generation_terminal_body_sha256",
}

APPROVAL_FIELDS = {
    "schema_version",
    "record_type",
    "approved_by",
    "approval_request",
    "approval_response",
    "approval_displayed_time",
    "approval_ingested_at",
    "evidence_source",
    "slack_permalink",
    "approved_hourly_usd",
    "approved_gpu_hours",
    "approved_gpu_cost_usd",
    "instance_type",
    "region",
    "includes_qualification",
    "includes_recovery_instances",
    "approval_body_sha256",
}

CLAIM_INT_FIELDS = {
    "schema_version",
    "generation",
    "gpu_spend_ledger_record_count",
    "approved_gpu_runtime_seconds",
    "consumed_gpu_seconds",
    "remaining_gpu_seconds",
    "open_allocation_count",
    "previous_generation",
}
CLAIM_FLOAT_FIELDS = {
    "approved_gpu_cost_usd",
    "hourly_cost_usd",
    "consumed_gpu_cost_usd",
    "remaining_gpu_cost_usd",
}
CLAIM_GENERATION_ONE_NULL_FIELDS = {
    "previous_generation_terminal_key",
    "previous_generation_terminal_file_sha256",
    "previous_generation_terminal_body_sha256",
    "previous_generation_terminal_version_id",
}
DECISION_INT_FIELDS = {"schema_version", "generation"}
TERMINAL_INT_FIELDS = {
    "schema_version",
    "generation",
    "final_gpu_spend_ledger_record_count",
    "final_approved_gpu_runtime_seconds",
    "final_consumed_gpu_seconds",
    "final_remaining_gpu_seconds",
    "final_open_allocation_count",
}
TERMINAL_FLOAT_FIELDS = {
    "final_approved_gpu_cost_usd",
    "final_hourly_cost_usd",
    "final_consumed_gpu_cost_usd",
    "final_remaining_gpu_cost_usd",
}


class DictSubclass(dict[str, object]):
    pass


class ListSubclass(list[object]):
    pass


class StringSubclass(str):
    pass


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


class BytesSubclass(bytes):
    pass


class TupleSubclass(tuple[tuple[str, str], ...]):
    pass


class ArtifactSubclass(VersionedJsonArtifact):
    pass


class EntrySubclass(generation.GenerationInventoryEntry):
    pass


class ReceiptSubclass(generation.UnambiguousStartDecisionCreateReceipt):
    pass


class DatetimeSubclass(datetime):
    pass


class TimedeltaSubclass(timedelta):
    pass


class StatefulTimezone(tzinfo):
    utc = timezone.utc

    def __init__(
        self,
        offsets: list[timedelta] | None = None,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.offsets = [timedelta(0)] if offsets is None else offsets
        self.failure = failure
        self.calls = 0

    def utcoffset(self, value: datetime | None) -> timedelta:
        del value
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return self.offsets[min(self.calls - 1, len(self.offsets) - 1)]

    def dst(self, value: datetime | None) -> timedelta:
        del value
        return timedelta(0)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _file_bytes(value: object) -> bytes:
    return _canonical(value) + b"\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _different_digest(value: object) -> str:
    assert type(value) is str and len(value) == 64
    return ("0" if value[0] != "0" else "1") + value[1:]


def _rehash(value: dict[str, object], digest_field: str) -> dict[str, object]:
    body = copy.deepcopy(value)
    body.pop(digest_field, None)
    return {**body, digest_field: _sha(_canonical(body))}


def _load_test_module(relative: str, name: str) -> Any:
    path = ROOT / relative
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    production_tests = _load_test_module(
        "tests/test_glm52_sky_production_submission.py",
        "_h1e_production_fixture",
    )
    source_chain = production_tests.source_chain.__wrapped__(tmp_path_factory)
    legacy_intent = build_production_submission_intent(
        **production_tests._kwargs(source_chain)
    )
    legacy_descriptor = source_chain["artifacts"]["descriptor"]
    legacy_approval = source_chain["artifacts"]["approval"]
    assert type(legacy_descriptor) is VersionedJsonArtifact
    assert type(legacy_approval) is VersionedJsonArtifact
    approval_file_sha = _sha(legacy_approval.raw)
    approval = replace(
        legacy_approval,
        key=(
            f"campaigns/{RUN_ID}/authorities/"
            f"GPU_SPEND_APPROVAL-{approval_file_sha}.json"
        ),
    )
    descriptor_record = copy.deepcopy(json.loads(legacy_descriptor.raw))
    descriptor_record["approval_key"] = approval.key
    descriptor_record["must_start_by"] = MUST_START_BY.isoformat().replace(
        "+00:00",
        "Z",
    )
    descriptor_record = production_tests._rehash_source_record(
        "descriptor",
        descriptor_record,
        "descriptor_body_sha256",
    )
    descriptor = replace(
        legacy_descriptor,
        key=str(descriptor_record["campaign_descriptor_key"]),
        raw=_file_bytes(descriptor_record),
    )
    assert validate_sky_campaign_descriptor(descriptor_record) == descriptor_record
    snapshot_record = copy.deepcopy(
        source_chain["records"]["closed_gpu_spend_snapshot"]
    )
    snapshot_record.update(
        {
            "campaign_identity_sha256": descriptor_record[
                "campaign_identity_sha256"
            ],
            "descriptor_sha256": _sha(descriptor.raw),
            "descriptor_body_sha256": descriptor_record[
                "descriptor_body_sha256"
            ],
            "approval_sha256": approval_file_sha,
            "approval_body_sha256": json.loads(approval.raw)[
                "approval_body_sha256"
            ],
            "observed_at": legacy_intent["intent_at"],
        }
    )
    snapshot_record = _rehash(snapshot_record, "snapshot_body_sha256")
    snapshot = _artifact(
        key=(
            f"campaigns/{RUN_ID}/spend-snapshots/"
            f"{snapshot_record['snapshot_body_sha256']}/GPU_SPEND_SNAPSHOT.json"
        ),
        value=snapshot_record,
        version_id="snapshot-version-1",
    )
    assert validate_gpu_spend_snapshot(snapshot_record) == snapshot_record
    intent = copy.deepcopy(legacy_intent)
    intent.update(
        {
            "campaign_identity_sha256": descriptor_record[
                "campaign_identity_sha256"
            ],
            "descriptor_key": descriptor.key,
            "descriptor_file_sha256": _sha(descriptor.raw),
            "descriptor_body_sha256": descriptor_record[
                "descriptor_body_sha256"
            ],
            "descriptor_version_id": descriptor.version_id,
            "approval_key": approval.key,
            "approval_file_sha256": approval_file_sha,
            "approval_body_sha256": json.loads(approval.raw)[
                "approval_body_sha256"
            ],
            "approval_version_id": approval.version_id,
            "must_start_by": descriptor_record["must_start_by"],
            "gpu_spend_snapshot_key": snapshot.key,
            "gpu_spend_snapshot_file_sha256": _sha(snapshot.raw),
            "gpu_spend_snapshot_body_sha256": snapshot_record[
                "snapshot_body_sha256"
            ],
            "gpu_spend_snapshot_version_id": snapshot.version_id,
            "spend_snapshot_observed_at": snapshot_record["observed_at"],
        }
    )
    intent = _rehash(intent, "intent_body_sha256")
    assert validate_production_submission_intent(intent) == intent
    intent_artifact = VersionedJsonArtifact(
        key=production_submission_intent_s3_key(
            run_id=str(intent["run_id"]),
            intent_body_sha256=str(intent["intent_body_sha256"]),
        ),
        raw=production_submission_intent_file_bytes(intent),
        version_id="production-intent-version-1",
    )
    authorities = {
        "descriptor": descriptor,
        "intent": intent_artifact,
        "descriptor_record": json.loads(descriptor.raw),
        "intent_record": intent,
    }
    acquisition_tests = _load_test_module(
        "tests/test_glm52_sky_production_acquisition.py",
        "_h1e_acquisition_fixture",
    )
    h1d = acquisition_tests._full_chain(
        sys.modules["mlx_vq.quality.glm52_sky_production_acquisition"],
        authorities,
    )
    acquisition_record = h1d["acquisition"]
    acquisition_artifact = VersionedJsonArtifact(
        key=production_submission_acquired_s3_key(
            run_id=RUN_ID,
            descriptor_file_sha256=_sha(descriptor.raw),
        ),
        raw=production_submission_acquired_file_bytes(acquisition_record),
        version_id="production-acquisition-version-1",
    )
    return {
        "descriptor": descriptor,
        "intent": intent_artifact,
        "approval": approval,
        "snapshot": snapshot,
        "baseline": h1d["baseline_artifact"],
        "ready": h1d["ready_artifact"],
        "acquisition": acquisition_artifact,
        "records": {
            "descriptor": descriptor_record,
            "intent": intent,
            "approval": json.loads(approval.raw),
            "snapshot": snapshot_record,
            "baseline": h1d["baseline"],
            "ready": h1d["ready"],
            "acquisition": acquisition_record,
        },
    }


def _claim_kwargs(
    chain: dict[str, object],
    *,
    generation_number: int = 1,
    inventory: list[generation.GenerationInventoryEntry] | None = None,
    previous_terminal: VersionedJsonArtifact | None = None,
    attempt: str = ATTEMPT_A,
    claim_at: datetime = CLAIM_AT,
) -> dict[str, object]:
    return {
        "generation": generation_number,
        "descriptor": chain["descriptor"],
        "intent": chain["intent"],
        "approval": chain["approval"],
        "controller_baseline": chain["baseline"],
        "must_start_control_plane_ready": chain["ready"],
        "submission_acquisition": chain["acquisition"],
        "gpu_spend_snapshot": chain["snapshot"],
        "generation_inventory": [] if inventory is None else inventory,
        "previous_generation_terminal": previous_terminal,
        "submit_attempt_id": attempt,
        "claim_created_at": claim_at,
    }


def _claim(chain: dict[str, object], **overrides: object) -> dict[str, object]:
    kwargs = _claim_kwargs(chain)
    kwargs.update(overrides)
    return generation.build_production_generation_claim(**kwargs)  # type: ignore[arg-type]


def _artifact(
    *,
    key: str,
    value: dict[str, object],
    version_id: str,
) -> VersionedJsonArtifact:
    return VersionedJsonArtifact(
        key=key,
        raw=_file_bytes(value),
        version_id=version_id,
    )


def _claim_artifact(
    value: dict[str, object],
    *,
    version_id: str = "generation-claim-version-1",
) -> VersionedJsonArtifact:
    return _artifact(
        key=generation.production_generation_claim_s3_key(
            run_id=str(value["run_id"]),
            generation=int(value["generation"]),
        ),
        value=value,
        version_id=version_id,
    )


def _decision(
    chain: dict[str, object],
    claim: dict[str, object],
    *,
    decision: str = "launch-once",
    decided_at: datetime = DECIDED_AT,
) -> dict[str, object]:
    return generation.build_production_generation_start_decision(
        generation_claim=_claim_artifact(claim),
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        decision=decision,  # type: ignore[arg-type]
        decided_at=decided_at,
    )


def _decision_artifact(
    value: dict[str, object],
    *,
    version_id: str = "generation-decision-version-1",
) -> VersionedJsonArtifact:
    return _artifact(
        key=generation.production_generation_start_decision_s3_key(
            run_id=str(value["run_id"]),
            generation=int(value["generation"]),
        ),
        value=value,
        version_id=version_id,
    )


def _entry(
    artifact: VersionedJsonArtifact,
) -> generation.GenerationInventoryEntry:
    return generation.GenerationInventoryEntry(
        key=artifact.key,
        raw=artifact.raw,
        version_id=artifact.version_id,
        is_latest=True,
        is_delete_marker=False,
    )


def _inventory(*artifacts: VersionedJsonArtifact) -> list[generation.GenerationInventoryEntry]:
    return sorted(
        [_entry(artifact) for artifact in artifacts],
        key=lambda entry: (entry.key, entry.version_id, entry.is_delete_marker),
    )


def _fresh_snapshot(
    chain: dict[str, object],
    observed_at: datetime,
    *,
    version_id: str,
) -> VersionedJsonArtifact:
    record = copy.deepcopy(chain["records"]["snapshot"])
    record["observed_at"] = observed_at.isoformat().replace("+00:00", "Z")
    record = _rehash(record, "snapshot_body_sha256")
    return _artifact(
        key=(
            f"campaigns/{RUN_ID}/spend-snapshots/"
            f"{record['snapshot_body_sha256']}/GPU_SPEND_SNAPSHOT.json"
        ),
        value=record,
        version_id=version_id,
    )


def _next_generation_chain(
    chain: dict[str, object],
    *,
    must_start_offset_seconds: int = 50,
) -> tuple[dict[str, object], datetime]:
    next_snapshot_at = MUST_START_BY + timedelta(seconds=42)
    next_intent_at = next_snapshot_at + timedelta(seconds=10)
    next_must_start = next_intent_at + timedelta(
        seconds=must_start_offset_seconds
    )
    next_claim_at = next_intent_at + timedelta(seconds=40)
    descriptor_record = copy.deepcopy(chain["records"]["descriptor"])
    descriptor_record.update(
        {
            "must_start_by": next_must_start.isoformat().replace("+00:00", "Z"),
            "campaign_descriptor_key": (
                f"campaigns/{RUN_ID}/submissions/production-generation-2/"
                "campaign-descriptor-v2.json"
            ),
        }
    )
    identity = {
        key: value
        for key, value in descriptor_record.items()
        if key
        not in {
            "campaign_identity_sha256",
            "descriptor_body_sha256",
            "must_start_by",
            "campaign_descriptor_key",
        }
    }
    descriptor_record["campaign_identity_sha256"] = _sha(_canonical(identity))
    descriptor_record = _rehash(descriptor_record, "descriptor_body_sha256")
    descriptor_artifact = _artifact(
        key=str(descriptor_record["campaign_descriptor_key"]),
        value=descriptor_record,
        version_id="descriptor-version-2",
    )
    assert validate_sky_campaign_descriptor(descriptor_record) == descriptor_record

    snapshot_record = copy.deepcopy(chain["records"]["snapshot"])
    snapshot_record.update(
        {
            "campaign_identity_sha256": descriptor_record[
                "campaign_identity_sha256"
            ],
            "descriptor_sha256": _sha(descriptor_artifact.raw),
            "descriptor_body_sha256": descriptor_record[
                "descriptor_body_sha256"
            ],
            "observed_at": next_snapshot_at.isoformat().replace("+00:00", "Z"),
        }
    )
    snapshot_record = _rehash(snapshot_record, "snapshot_body_sha256")
    snapshot_artifact = _artifact(
        key=(
            f"campaigns/{RUN_ID}/spend-snapshots/"
            f"{snapshot_record['snapshot_body_sha256']}/GPU_SPEND_SNAPSHOT.json"
        ),
        value=snapshot_record,
        version_id="snapshot-version-2",
    )
    assert validate_gpu_spend_snapshot(snapshot_record) == snapshot_record

    intent_record = copy.deepcopy(chain["records"]["intent"])
    intent_record.update(
        {
            "campaign_identity_sha256": descriptor_record[
                "campaign_identity_sha256"
            ],
            "descriptor_key": descriptor_artifact.key,
            "descriptor_file_sha256": _sha(descriptor_artifact.raw),
            "descriptor_body_sha256": descriptor_record[
                "descriptor_body_sha256"
            ],
            "descriptor_version_id": descriptor_artifact.version_id,
            "must_start_by": descriptor_record["must_start_by"],
            "intent_at": next_intent_at.isoformat().replace("+00:00", "Z"),
            "staged_readiness_key": (
                f"campaigns/{RUN_ID}/submissions/production-generation-2/"
                "STAGED_CONTROL_PLANE_READY.json"
            ),
            "bundle_manifest_key": (
                f"campaigns/{RUN_ID}/submissions/production-generation-2/"
                f"bundle-manifests/{intent_record['bundle_manifest_body_sha256']}/"
                "bundle-manifest-v1.json"
            ),
            "gpu_spend_snapshot_key": snapshot_artifact.key,
            "gpu_spend_snapshot_file_sha256": _sha(snapshot_artifact.raw),
            "gpu_spend_snapshot_body_sha256": snapshot_record[
                "snapshot_body_sha256"
            ],
            "gpu_spend_snapshot_version_id": snapshot_artifact.version_id,
            "spend_snapshot_observed_at": snapshot_record["observed_at"],
        }
    )
    intent_record = _rehash(intent_record, "intent_body_sha256")
    assert validate_production_submission_intent(intent_record) == intent_record
    intent_artifact = _artifact(
        key=production_submission_intent_s3_key(
            run_id=RUN_ID,
            intent_body_sha256=str(intent_record["intent_body_sha256"]),
        ),
        value=intent_record,
        version_id="production-intent-version-2",
    )
    authorities = {
        "descriptor": descriptor_artifact,
        "intent": intent_artifact,
        "descriptor_record": descriptor_record,
        "intent_record": intent_record,
    }
    acquisition_tests = _load_test_module(
        "tests/test_glm52_sky_production_acquisition.py",
        "_h1e_next_acquisition_fixture",
    )
    h1d = acquisition_tests._full_chain(
        sys.modules["mlx_vq.quality.glm52_sky_production_acquisition"],
        authorities,
    )
    acquisition_record = h1d["acquisition"]
    acquisition_artifact = _artifact(
        key=production_submission_acquired_s3_key(
            run_id=RUN_ID,
            descriptor_file_sha256=_sha(descriptor_artifact.raw),
        ),
        value=acquisition_record,
        version_id="production-acquisition-version-2",
    )
    return (
        {
            "descriptor": descriptor_artifact,
            "intent": intent_artifact,
            "approval": chain["approval"],
            "snapshot": snapshot_artifact,
            "baseline": h1d["baseline_artifact"],
            "ready": h1d["ready_artifact"],
            "acquisition": acquisition_artifact,
            "records": {
                "descriptor": descriptor_record,
                "intent": intent_record,
                "approval": chain["records"]["approval"],
                "snapshot": snapshot_record,
                "baseline": h1d["baseline"],
                "ready": h1d["ready"],
                "acquisition": acquisition_record,
            },
        },
        next_claim_at,
    )


def _coherent_approval_chain(
    chain: dict[str, object],
    *,
    field: str,
    value: object,
    approval_raw_override: bytes | None = None,
    approval_key_override: str | None = None,
    descriptor_approval_key_override: str | None = None,
    intent_approval_key_override: str | None = None,
) -> dict[str, object]:
    approval_record = copy.deepcopy(chain["records"]["approval"])
    approval_record[field] = value
    approval_record = _rehash(approval_record, "approval_body_sha256")
    approval_raw = (
        _file_bytes(approval_record)
        if approval_raw_override is None
        else approval_raw_override
    )
    approval_file_sha = _sha(approval_raw)
    canonical_approval_key = (
        f"campaigns/{RUN_ID}/authorities/"
        f"GPU_SPEND_APPROVAL-{approval_file_sha}.json"
    )
    approval_artifact = VersionedJsonArtifact(
        key=(
            canonical_approval_key
            if approval_key_override is None
            else approval_key_override
        ),
        raw=approval_raw,
        version_id="approval-version-type-mutant",
    )

    descriptor_record = copy.deepcopy(chain["records"]["descriptor"])
    descriptor_record.update(
        {
            "approval_key": (
                approval_artifact.key
                if descriptor_approval_key_override is None
                else descriptor_approval_key_override
            ),
            "approval_sha256": approval_file_sha,
        }
    )
    identity = {
        key: item
        for key, item in descriptor_record.items()
        if key
        not in {
            "campaign_identity_sha256",
            "descriptor_body_sha256",
            "must_start_by",
            "campaign_descriptor_key",
        }
    }
    descriptor_record["campaign_identity_sha256"] = _sha(_canonical(identity))
    descriptor_record = _rehash(descriptor_record, "descriptor_body_sha256")
    descriptor_artifact = _artifact(
        key=str(descriptor_record["campaign_descriptor_key"]),
        value=descriptor_record,
        version_id="descriptor-version-type-mutant",
    )

    snapshot_record = copy.deepcopy(chain["records"]["snapshot"])
    snapshot_record.update(
        {
            "campaign_identity_sha256": descriptor_record[
                "campaign_identity_sha256"
            ],
            "descriptor_sha256": _sha(descriptor_artifact.raw),
            "descriptor_body_sha256": descriptor_record[
                "descriptor_body_sha256"
            ],
            "approval_sha256": approval_file_sha,
            "approval_body_sha256": approval_record["approval_body_sha256"],
        }
    )
    snapshot_record = _rehash(snapshot_record, "snapshot_body_sha256")
    snapshot_artifact = _artifact(
        key=(
            f"campaigns/{RUN_ID}/spend-snapshots/"
            f"{snapshot_record['snapshot_body_sha256']}/GPU_SPEND_SNAPSHOT.json"
        ),
        value=snapshot_record,
        version_id="snapshot-version-type-mutant",
    )

    intent_record = copy.deepcopy(chain["records"]["intent"])
    intent_record.update(
        {
            "campaign_identity_sha256": descriptor_record[
                "campaign_identity_sha256"
            ],
            "descriptor_key": descriptor_artifact.key,
            "descriptor_file_sha256": _sha(descriptor_artifact.raw),
            "descriptor_body_sha256": descriptor_record[
                "descriptor_body_sha256"
            ],
            "descriptor_version_id": descriptor_artifact.version_id,
            "approval_key": (
                approval_artifact.key
                if intent_approval_key_override is None
                else intent_approval_key_override
            ),
            "approval_file_sha256": approval_file_sha,
            "approval_body_sha256": approval_record["approval_body_sha256"],
            "approval_version_id": approval_artifact.version_id,
            "gpu_spend_snapshot_key": snapshot_artifact.key,
            "gpu_spend_snapshot_file_sha256": _sha(snapshot_artifact.raw),
            "gpu_spend_snapshot_body_sha256": snapshot_record[
                "snapshot_body_sha256"
            ],
            "gpu_spend_snapshot_version_id": snapshot_artifact.version_id,
        }
    )
    intent_record = _rehash(intent_record, "intent_body_sha256")
    intent_artifact = _artifact(
        key=production_submission_intent_s3_key(
            run_id=RUN_ID,
            intent_body_sha256=str(intent_record["intent_body_sha256"]),
        ),
        value=intent_record,
        version_id="production-intent-version-type-mutant",
    )
    authorities = {
        "descriptor": descriptor_artifact,
        "intent": intent_artifact,
        "descriptor_record": descriptor_record,
        "intent_record": intent_record,
    }
    acquisition_tests = _load_test_module(
        "tests/test_glm52_sky_production_acquisition.py",
        f"_h1e_approval_acquisition_fixture_{field}",
    )
    h1d = acquisition_tests._full_chain(
        sys.modules["mlx_vq.quality.glm52_sky_production_acquisition"],
        authorities,
    )
    acquisition_record = h1d["acquisition"]
    acquisition_artifact = _artifact(
        key=production_submission_acquired_s3_key(
            run_id=RUN_ID,
            descriptor_file_sha256=_sha(descriptor_artifact.raw),
        ),
        value=acquisition_record,
        version_id="production-acquisition-version-type-mutant",
    )
    return {
        "descriptor": descriptor_artifact,
        "intent": intent_artifact,
        "approval": approval_artifact,
        "snapshot": snapshot_artifact,
        "baseline": h1d["baseline_artifact"],
        "ready": h1d["ready_artifact"],
        "acquisition": acquisition_artifact,
        "records": {
            "descriptor": descriptor_record,
            "intent": intent_record,
            "approval": approval_record,
            "snapshot": snapshot_record,
            "baseline": h1d["baseline"],
            "ready": h1d["ready"],
            "acquisition": acquisition_record,
        },
    }


def _coherently_readdressed_snapshot_chain(
    chain: dict[str, object],
    *,
    snapshot_updates: dict[str, object],
    copy_snapshot_semantics: bool = False,
    intent_updates: dict[str, object] | None = None,
) -> dict[str, object]:
    snapshot_record = copy.deepcopy(chain["records"]["snapshot"])
    snapshot_record.update(snapshot_updates)
    snapshot_record = _rehash(snapshot_record, "snapshot_body_sha256")
    snapshot_artifact = _artifact(
        key=(
            f"campaigns/{RUN_ID}/spend-snapshots/"
            f"{snapshot_record['snapshot_body_sha256']}/"
            "GPU_SPEND_SNAPSHOT.json"
        ),
        value=snapshot_record,
        version_id=(
            "snapshot-version-coherent-drift-"
            f"{_sha(_canonical(snapshot_updates))[:12]}"
        ),
    )
    assert validate_gpu_spend_snapshot(snapshot_record) == snapshot_record

    intent_record = copy.deepcopy(chain["records"]["intent"])
    intent_record.update(
        {
            "gpu_spend_snapshot_key": snapshot_artifact.key,
            "gpu_spend_snapshot_file_sha256": _sha(snapshot_artifact.raw),
            "gpu_spend_snapshot_body_sha256": snapshot_record[
                "snapshot_body_sha256"
            ],
            "gpu_spend_snapshot_version_id": snapshot_artifact.version_id,
        }
    )
    if copy_snapshot_semantics:
        for snapshot_field, intent_field in {
            "observed_at": "spend_snapshot_observed_at",
            "gpu_spend_ledger_tip_record_sha256": (
                "gpu_spend_ledger_tip_record_sha256"
            ),
            "gpu_spend_ledger_genesis_sha256": (
                "gpu_spend_ledger_genesis_sha256"
            ),
            "ec2_allocation_history_sha256": (
                "ec2_allocation_history_sha256"
            ),
            "open_allocation_count": "open_allocation_count",
            "approved_gpu_runtime_seconds": "approved_gpu_runtime_seconds",
            "approved_gpu_cost_usd": "approved_gpu_cost_usd",
            "hourly_cost_usd": "hourly_cost_usd",
            "consumed_gpu_seconds": "consumed_gpu_seconds",
            "consumed_gpu_cost_usd": "consumed_gpu_cost_usd",
            "remaining_gpu_seconds": "remaining_gpu_seconds",
            "remaining_gpu_cost_usd": "remaining_gpu_cost_usd",
        }.items():
            intent_record[intent_field] = snapshot_record[snapshot_field]
    if intent_updates is not None:
        intent_record.update(intent_updates)
    intent_record = _rehash(intent_record, "intent_body_sha256")
    assert validate_production_submission_intent(intent_record) == intent_record
    intent_artifact = _artifact(
        key=production_submission_intent_s3_key(
            run_id=RUN_ID,
            intent_body_sha256=str(intent_record["intent_body_sha256"]),
        ),
        value=intent_record,
        version_id=(
            "production-intent-version-coherent-drift-"
            f"{_sha(_canonical(snapshot_updates))[:12]}"
        ),
    )
    descriptor = chain["descriptor"]
    assert type(descriptor) is VersionedJsonArtifact
    authorities = {
        "descriptor": descriptor,
        "intent": intent_artifact,
        "descriptor_record": copy.deepcopy(chain["records"]["descriptor"]),
        "intent_record": intent_record,
    }
    acquisition_tests = _load_test_module(
        "tests/test_glm52_sky_production_acquisition.py",
        (
            "_h1e_snapshot_drift_acquisition_fixture_"
            f"{_sha(_canonical(snapshot_updates))[:12]}"
        ),
    )
    h1d = acquisition_tests._full_chain(
        sys.modules["mlx_vq.quality.glm52_sky_production_acquisition"],
        authorities,
    )
    acquisition_record = h1d["acquisition"]
    acquisition_artifact = _artifact(
        key=production_submission_acquired_s3_key(
            run_id=RUN_ID,
            descriptor_file_sha256=_sha(descriptor.raw),
        ),
        value=acquisition_record,
        version_id=(
            "production-acquisition-version-coherent-drift-"
            f"{_sha(_canonical(snapshot_updates))[:12]}"
        ),
    )
    return {
        "descriptor": descriptor,
        "intent": intent_artifact,
        "approval": chain["approval"],
        "snapshot": snapshot_artifact,
        "baseline": h1d["baseline_artifact"],
        "ready": h1d["ready_artifact"],
        "acquisition": acquisition_artifact,
        "records": {
            "descriptor": copy.deepcopy(chain["records"]["descriptor"]),
            "intent": intent_record,
            "approval": copy.deepcopy(chain["records"]["approval"]),
            "snapshot": snapshot_record,
            "baseline": h1d["baseline"],
            "ready": h1d["ready"],
            "acquisition": acquisition_record,
        },
    }


def _claim_for_chain(
    base_claim: dict[str, object],
    chain: dict[str, object],
) -> dict[str, object]:
    claim = copy.deepcopy(base_claim)
    descriptor = chain["descriptor"]
    intent = chain["intent"]
    approval = chain["approval"]
    baseline = chain["baseline"]
    ready = chain["ready"]
    acquisition = chain["acquisition"]
    snapshot = chain["snapshot"]
    assert all(
        type(artifact) is VersionedJsonArtifact
        for artifact in (
            descriptor,
            intent,
            approval,
            baseline,
            ready,
            acquisition,
            snapshot,
        )
    )
    records = chain["records"]
    claim.update(
        {
            "campaign_identity_sha256": records["intent"][
                "campaign_identity_sha256"
            ],
            "descriptor_key": descriptor.key,
            "descriptor_file_sha256": _sha(descriptor.raw),
            "descriptor_body_sha256": records["descriptor"][
                "descriptor_body_sha256"
            ],
            "descriptor_version_id": descriptor.version_id,
            "intent_key": intent.key,
            "intent_file_sha256": _sha(intent.raw),
            "intent_body_sha256": records["intent"]["intent_body_sha256"],
            "intent_version_id": intent.version_id,
            "controller_baseline_key": baseline.key,
            "controller_baseline_file_sha256": _sha(baseline.raw),
            "controller_baseline_body_sha256": records["baseline"][
                "baseline_body_sha256"
            ],
            "controller_baseline_version_id": baseline.version_id,
            "must_start_control_plane_ready_key": ready.key,
            "must_start_control_plane_ready_file_sha256": _sha(ready.raw),
            "must_start_control_plane_ready_body_sha256": records["ready"][
                "control_plane_ready_body_sha256"
            ],
            "must_start_control_plane_ready_version_id": ready.version_id,
            "submission_acquisition_key": acquisition.key,
            "submission_acquisition_file_sha256": _sha(acquisition.raw),
            "submission_acquisition_body_sha256": records["acquisition"][
                "acquisition_body_sha256"
            ],
            "submission_acquisition_version_id": acquisition.version_id,
            "approval_key": approval.key,
            "approval_file_sha256": _sha(approval.raw),
            "approval_body_sha256": records["approval"][
                "approval_body_sha256"
            ],
            "approval_version_id": approval.version_id,
            "gpu_spend_snapshot_key": snapshot.key,
            "gpu_spend_snapshot_file_sha256": _sha(snapshot.raw),
            "gpu_spend_snapshot_body_sha256": records["snapshot"][
                "snapshot_body_sha256"
            ],
            "gpu_spend_snapshot_version_id": snapshot.version_id,
        }
    )
    claim["sky_job_identity_sha256"] = _sha(
        _canonical(
            {
                "account_id": "246813579024",
                "campaign_identity_sha256": claim[
                    "campaign_identity_sha256"
                ],
                "generation": claim["generation"],
                "intent_body_sha256": claim["intent_body_sha256"],
                "managed_mode": "production",
                "run_id": claim["run_id"],
                "submit_attempt_id": claim["submit_attempt_id"],
            }
        )
    )
    return _rehash(claim, "generation_claim_body_sha256")


def _decision_for_claim(
    base_decision: dict[str, object],
    claim: dict[str, object],
    claim_artifact: VersionedJsonArtifact,
) -> dict[str, object]:
    decision = copy.deepcopy(base_decision)
    decision.update(
        {
            "campaign_identity_sha256": claim["campaign_identity_sha256"],
            "sky_job_identity_sha256": claim["sky_job_identity_sha256"],
            "generation_claim_key": claim_artifact.key,
            "generation_claim_file_sha256": _sha(claim_artifact.raw),
            "generation_claim_body_sha256": claim[
                "generation_claim_body_sha256"
            ],
            "generation_claim_version_id": claim_artifact.version_id,
        }
    )
    return _rehash(decision, "start_decision_body_sha256")


def _assert_approval_mutant_rejected_at_six_boundaries(
    *,
    chain: dict[str, object],
    claim: dict[str, object],
    decision: dict[str, object],
    public_validator_calls: list[object],
    local_gate_must_precede_public_validator: bool | None = True,
) -> None:
    claim_artifact = _claim_artifact(
        claim,
        version_id=str(decision["generation_claim_version_id"]),
    )
    decision_artifact = _decision_artifact(
        decision,
        version_id="generation-decision-version-approval-mutant",
    )

    def assert_local_gate_precedes_public_validator(
        operation: Any,
    ) -> None:
        public_validator_calls.clear()
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            operation()
        if local_gate_must_precede_public_validator is True:
            assert public_validator_calls == []
        elif local_gate_must_precede_public_validator is False:
            assert public_validator_calls

    assert_local_gate_precedes_public_validator(lambda: _claim(chain))
    assert_local_gate_precedes_public_validator(
        lambda: generation.validate_production_generation_claim(
            claim,
            descriptor=chain["descriptor"],  # type: ignore[arg-type]
            intent=chain["intent"],  # type: ignore[arg-type]
            approval=chain["approval"],  # type: ignore[arg-type]
            controller_baseline=chain["baseline"],  # type: ignore[arg-type]
            must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
            submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
            gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
            generation_inventory=[],
            previous_generation_terminal=None,
        )
    )
    assert_local_gate_precedes_public_validator(
        lambda: generation.build_production_generation_start_decision(
            generation_claim=claim_artifact,
            descriptor=chain["descriptor"],  # type: ignore[arg-type]
            intent=chain["intent"],  # type: ignore[arg-type]
            approval=chain["approval"],  # type: ignore[arg-type]
            controller_baseline=chain["baseline"],  # type: ignore[arg-type]
            must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
            submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
            decision="launch-once",
            decided_at=DECIDED_AT,
        )
    )
    assert_local_gate_precedes_public_validator(
        lambda: generation.validate_production_generation_start_decision(
            decision,
            generation_claim=claim_artifact,
            descriptor=chain["descriptor"],  # type: ignore[arg-type]
            intent=chain["intent"],  # type: ignore[arg-type]
            approval=chain["approval"],  # type: ignore[arg-type]
            controller_baseline=chain["baseline"],  # type: ignore[arg-type]
            must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
            submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
            now=DECIDED_AT,
        )
    )
    assert_local_gate_precedes_public_validator(
        lambda: generation.validate_modeled_submit_once(
            generation_claim=claim_artifact,
            start_decision=decision_artifact,
            descriptor=chain["descriptor"],  # type: ignore[arg-type]
            intent=chain["intent"],  # type: ignore[arg-type]
            approval=chain["approval"],  # type: ignore[arg-type]
            controller_baseline=chain["baseline"],  # type: ignore[arg-type]
            must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
            submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
            gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
            start_decision_receipt=_receipt(
                claim,
                decision,
                decision_artifact,
            ),
            post_create_generation_inventory=_inventory(
                claim_artifact,
                decision_artifact,
            ),
            post_create_audit_server_date=(
                "Sun, 26 Jul 2026 12:11:22 GMT"
            ),
            now=DECIDED_AT + timedelta(seconds=2),
        )
    )
    public_validator_calls.clear()
    assert generation.decide_production_generation_action(
        generation_inventory=[],
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
        submit_attempt_id=ATTEMPT_A,
        now=CLAIM_AT,
    ) == generation.ProductionGenerationDecision(
        action="fail-closed",
        reason="invalid-or-inconsistent-generation-authority",
        generation=None,
    )
    if local_gate_must_precede_public_validator is True:
        assert public_validator_calls == []
    elif local_gate_must_precede_public_validator is False:
        assert public_validator_calls


def _expiry_chain(
    chain: dict[str, object],
) -> tuple[
    dict[str, object],
    VersionedJsonArtifact,
    dict[str, object],
    VersionedJsonArtifact,
    VersionedJsonArtifact,
    dict[str, object],
    VersionedJsonArtifact,
]:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    decision = _decision(
        chain,
        claim,
        decision="expire-unstarted",
        decided_at=MUST_START_BY,
    )
    decision_artifact = _decision_artifact(decision)
    snapshot = _fresh_snapshot(
        chain,
        MUST_START_BY + timedelta(seconds=10),
        version_id="final-snapshot-version-1",
    )
    terminal = generation.build_production_generation_terminal(
        generation_claim=claim_artifact,
        start_decision=decision_artifact,
        final_gpu_spend_snapshot=snapshot,
        terminal_at=MUST_START_BY + timedelta(seconds=20),
    )
    terminal_artifact = _artifact(
        key=generation.production_generation_terminal_s3_key(
            run_id=RUN_ID,
            generation=1,
        ),
        value=terminal,
        version_id="generation-terminal-version-1",
    )
    return (
        claim,
        claim_artifact,
        decision,
        decision_artifact,
        snapshot,
        terminal,
        terminal_artifact,
    )


def _closed_generation_with_budget(
    chain: dict[str, object],
    *,
    claim_updates: dict[str, object],
) -> tuple[
    list[generation.GenerationInventoryEntry],
    VersionedJsonArtifact,
]:
    (
        claim,
        _,
        decision,
        _,
        _,
        terminal,
        _,
    ) = _expiry_chain(chain)
    claim.update(claim_updates)
    claim = _rehash(claim, "generation_claim_body_sha256")
    claim_artifact = _claim_artifact(
        claim,
        version_id="generation-claim-version-isolated-budget",
    )

    decision.update(
        {
            "generation_claim_file_sha256": _sha(claim_artifact.raw),
            "generation_claim_body_sha256": claim[
                "generation_claim_body_sha256"
            ],
            "generation_claim_version_id": claim_artifact.version_id,
        }
    )
    decision = _rehash(decision, "start_decision_body_sha256")
    decision_artifact = _decision_artifact(
        decision,
        version_id="generation-decision-version-isolated-budget",
    )

    terminal.update(
        {
            "generation_claim_file_sha256": _sha(claim_artifact.raw),
            "generation_claim_body_sha256": claim[
                "generation_claim_body_sha256"
            ],
            "generation_claim_version_id": claim_artifact.version_id,
            "start_decision_file_sha256": _sha(decision_artifact.raw),
            "start_decision_body_sha256": decision[
                "start_decision_body_sha256"
            ],
            "start_decision_version_id": decision_artifact.version_id,
            "final_consumed_gpu_seconds": claim["consumed_gpu_seconds"],
            "final_consumed_gpu_cost_usd": claim["consumed_gpu_cost_usd"],
            "final_remaining_gpu_seconds": claim["remaining_gpu_seconds"],
            "final_remaining_gpu_cost_usd": claim["remaining_gpu_cost_usd"],
        }
    )
    terminal = _rehash(terminal, "generation_terminal_body_sha256")
    terminal_artifact = _artifact(
        key=generation.production_generation_terminal_s3_key(
            run_id=RUN_ID,
            generation=1,
        ),
        value=terminal,
        version_id="generation-terminal-version-isolated-budget",
    )
    return (
        _inventory(claim_artifact, decision_artifact, terminal_artifact),
        terminal_artifact,
    )


def _receipt(
    claim: dict[str, object],
    decision: dict[str, object],
    decision_artifact: VersionedJsonArtifact,
    *,
    request_started_at: datetime = DECIDED_AT,
    response_received_at: datetime = DECIDED_AT + timedelta(seconds=1),
    server_date: str = "Sun, 26 Jul 2026 12:11:21 GMT",
) -> generation.UnambiguousStartDecisionCreateReceipt:
    raw = decision_artifact.raw
    digest = hashlib.sha256(raw).digest()
    file_sha = digest.hex()
    checksum = base64.b64encode(digest).decode("ascii")
    return generation.UnambiguousStartDecisionCreateReceipt(
        record_kind="generation-start-decision",
        account_id="246813579024",
        region="us-west-2",
        bucket=str(claim["bucket"]),
        expected_bucket_owner="246813579024",
        operation="PutObject",
        if_none_match="*",
        generation=int(claim["generation"]),
        generation_text=str(claim["generation_text"]),
        submit_attempt_id=str(claim["submit_attempt_id"]),
        key=decision_artifact.key,
        content_length=len(raw),
        candidate_file_sha256=file_sha,
        request_checksum_algorithm="SHA256",
        request_checksum_sha256_base64=checksum,
        immutable_metadata=(
            ("glm52-account-id", "246813579024"),
            ("glm52-region", "us-west-2"),
            ("glm52-run-id", str(claim["run_id"])),
            ("glm52-generation", str(claim["generation_text"])),
            ("glm52-submit-attempt-id", str(claim["submit_attempt_id"])),
            (
                "glm52-record-type",
                "glm52_sky_production_generation_start_decision_v1",
            ),
            ("glm52-body-sha256", str(decision["start_decision_body_sha256"])),
            ("glm52-file-sha256", file_sha),
        ),
        version_id=decision_artifact.version_id,
        etag='"0123456789abcdef0123456789abcdef"',
        response_checksum_sha256_base64=checksum,
        aws_request_id="A1B2C3D4E5F6G7H8",
        server_date=server_date,
        request_started_at=request_started_at,
        response_received_at=response_received_at,
        http_status=200,
        outcome="created",
        source="direct-response",
    )


def test_mutant_missing_module_or_public_surface_fails_exactly() -> None:
    assert generation.__all__ == EXPECTED_ALL
    assert issubclass(generation.ProductionGenerationAuthorityError, ValueError)
    assert generation.ProductionGenerationAuthorityError.__bases__ == (ValueError,)
    assert [field.name for field in fields(generation.GenerationInventoryEntry)] == [
        "key",
        "raw",
        "version_id",
        "is_latest",
        "is_delete_marker",
    ]
    forbidden = {
        "put_object",
        "list_object_versions",
        "submit",
        "launch",
        "retry",
        "serialize_receipt",
        "consume_receipt",
        "conditional_create",
    }
    assert forbidden.isdisjoint(generation.__all__)
    assert all(not hasattr(generation, name) for name in forbidden)


def test_mutant_public_literals_dataclasses_or_signatures_drift_from_exact_contract(
    chain: dict[str, object],
) -> None:
    assert get_args(generation.ProductionGenerationAction) == (
        "create-generation-claim",
        "create-launch-once-decision",
        "create-expire-unstarted-decision",
        "create-expired-unstarted-terminal",
        "reconcile-only",
        "advance-generation",
        "fail-closed",
    )
    assert get_args(generation.ProductionGenerationReason) == (
        "empty-inventory-ready-for-generation-one",
        "prior-expired-generation-ready-for-advance",
        "current-claim-ready-for-launch-decision",
        "current-claim-deadline-reached",
        "current-expiry-decision-ready-for-terminal",
        "stored-or-visible-generation-requires-reconciliation",
        "invalid-or-inconsistent-generation-authority",
    )
    exact_types: dict[type[object], list[tuple[str, object, object]]] = {
        generation.GenerationInventoryEntry: [
            ("key", str, MISSING),
            ("raw", Optional[bytes], MISSING),
            ("version_id", str, MISSING),
            ("is_latest", bool, MISSING),
            ("is_delete_marker", bool, MISSING),
        ],
        generation.UnambiguousStartDecisionCreateReceipt: [
            ("record_kind", Literal["generation-start-decision"], MISSING),
            ("account_id", str, MISSING),
            ("region", str, MISSING),
            ("bucket", str, MISSING),
            ("expected_bucket_owner", Literal["246813579024"], MISSING),
            ("operation", Literal["PutObject"], MISSING),
            ("if_none_match", Literal["*"], MISSING),
            ("generation", int, MISSING),
            ("generation_text", str, MISSING),
            ("submit_attempt_id", str, MISSING),
            ("key", str, MISSING),
            ("content_length", int, MISSING),
            ("candidate_file_sha256", str, MISSING),
            ("request_checksum_algorithm", Literal["SHA256"], MISSING),
            ("request_checksum_sha256_base64", str, MISSING),
            ("immutable_metadata", tuple[tuple[str, str], ...], MISSING),
            ("version_id", str, MISSING),
            ("etag", str, MISSING),
            ("response_checksum_sha256_base64", str, MISSING),
            ("aws_request_id", str, MISSING),
            ("server_date", str, MISSING),
            ("request_started_at", Union[datetime, str], MISSING),
            ("response_received_at", Union[datetime, str], MISSING),
            ("http_status", int, MISSING),
            ("outcome", Literal["created"], MISSING),
            ("source", Literal["direct-response"], MISSING),
        ],
        generation.ModeledSubmitOnceValidation: [
            (
                "validation_result",
                Literal["modeled-submit-once-valid"],
                MISSING,
            ),
            ("generation", int, MISSING),
            ("submit_attempt_id", str, MISSING),
            ("sky_job_identity_sha256", str, MISSING),
            ("claim_key", str, MISSING),
            ("claim_version_id", str, MISSING),
            ("start_decision_key", str, MISSING),
            ("start_decision_version_id", str, MISSING),
            ("validated_at", str, MISSING),
            ("conservative_validation_at", str, MISSING),
            ("must_start_by", str, MISSING),
        ],
        generation.ProductionGenerationDecision: [
            (
                "action",
                generation.ProductionGenerationAction,
                MISSING,
            ),
            (
                "reason",
                generation.ProductionGenerationReason,
                MISSING,
            ),
            ("generation", Optional[int], None),
        ],
    }
    for record_type, expected in exact_types.items():
        parameters = record_type.__dataclass_params__  # type: ignore[attr-defined]
        assert parameters.frozen is True
        assert parameters.order is False
        assert parameters.unsafe_hash is False
        assert parameters.eq is (
            record_type is not generation.UnambiguousStartDecisionCreateReceipt
        )
        hints = get_type_hints(record_type)
        assert [
            (field.name, hints[field.name], field.default)
            for field in fields(record_type)
        ] == expected

    claim = _claim(chain)
    decision = _decision(chain, claim)
    receipt = _receipt(claim, decision, _decision_artifact(decision))
    duplicate_receipt = replace(receipt)
    assert receipt is not duplicate_receipt
    assert receipt != duplicate_receipt
    with pytest.raises(FrozenInstanceError):
        receipt.account_id = "mutated"  # type: ignore[misc]
    resolution = generation.ProductionGenerationDecision(
        action="fail-closed",
        reason="invalid-or-inconsistent-generation-authority",
    )
    assert resolution.generation is None
    with pytest.raises(FrozenInstanceError):
        resolution.generation = 1  # type: ignore[misc]

    expected_signatures = {
        "build_production_generation_claim": (
            "generation",
            "descriptor",
            "intent",
            "approval",
            "controller_baseline",
            "must_start_control_plane_ready",
            "submission_acquisition",
            "gpu_spend_snapshot",
            "generation_inventory",
            "previous_generation_terminal",
            "submit_attempt_id",
            "claim_created_at",
        ),
        "validate_production_generation_claim": (
            "value",
            "descriptor",
            "intent",
            "approval",
            "controller_baseline",
            "must_start_control_plane_ready",
            "submission_acquisition",
            "gpu_spend_snapshot",
            "generation_inventory",
            "previous_generation_terminal",
        ),
        "production_generation_claim_file_bytes": ("value",),
        "production_generation_claim_file_sha256": ("value",),
        "production_generation_claim_s3_key": ("run_id", "generation"),
        "build_production_generation_start_decision": (
            "generation_claim",
            "descriptor",
            "intent",
            "approval",
            "controller_baseline",
            "must_start_control_plane_ready",
            "submission_acquisition",
            "decision",
            "decided_at",
        ),
        "validate_production_generation_start_decision": (
            "value",
            "generation_claim",
            "descriptor",
            "intent",
            "approval",
            "controller_baseline",
            "must_start_control_plane_ready",
            "submission_acquisition",
            "now",
        ),
        "production_generation_start_decision_file_bytes": ("value",),
        "production_generation_start_decision_file_sha256": ("value",),
        "production_generation_start_decision_s3_key": (
            "run_id",
            "generation",
        ),
        "build_production_generation_terminal": (
            "generation_claim",
            "start_decision",
            "final_gpu_spend_snapshot",
            "terminal_at",
        ),
        "validate_production_generation_terminal": (
            "value",
            "generation_claim",
            "start_decision",
            "final_gpu_spend_snapshot",
            "now",
        ),
        "production_generation_terminal_file_bytes": ("value",),
        "production_generation_terminal_file_sha256": ("value",),
        "production_generation_terminal_s3_key": ("run_id", "generation"),
        "validate_production_generation_inventory": ("entries", "run_id"),
        "decide_production_generation_action": (
            "generation_inventory",
            "descriptor",
            "intent",
            "approval",
            "controller_baseline",
            "must_start_control_plane_ready",
            "submission_acquisition",
            "gpu_spend_snapshot",
            "submit_attempt_id",
            "now",
        ),
        "validate_modeled_submit_once": (
            "generation_claim",
            "start_decision",
            "descriptor",
            "intent",
            "approval",
            "controller_baseline",
            "must_start_control_plane_ready",
            "submission_acquisition",
            "gpu_spend_snapshot",
            "start_decision_receipt",
            "post_create_generation_inventory",
            "post_create_audit_server_date",
            "now",
        ),
    }
    positional_first = {
        "production_generation_claim_file_bytes",
        "production_generation_claim_file_sha256",
        "production_generation_start_decision_file_bytes",
        "production_generation_start_decision_file_sha256",
        "production_generation_terminal_file_bytes",
        "production_generation_terminal_file_sha256",
        "validate_production_generation_claim",
        "validate_production_generation_start_decision",
        "validate_production_generation_terminal",
        "validate_production_generation_inventory",
    }
    expected_returns: dict[str, object] = {
        "build_production_generation_claim": dict[str, object],
        "production_generation_claim_file_bytes": bytes,
        "production_generation_claim_file_sha256": str,
        "production_generation_claim_s3_key": str,
        "validate_production_generation_claim": dict[str, object],
        "build_production_generation_start_decision": dict[str, object],
        "production_generation_start_decision_file_bytes": bytes,
        "production_generation_start_decision_file_sha256": str,
        "production_generation_start_decision_s3_key": str,
        "validate_production_generation_start_decision": dict[str, object],
        "build_production_generation_terminal": dict[str, object],
        "production_generation_terminal_file_bytes": bytes,
        "production_generation_terminal_file_sha256": str,
        "production_generation_terminal_s3_key": str,
        "validate_production_generation_terminal": dict[str, object],
        "validate_production_generation_inventory": dict[str, object],
        "decide_production_generation_action": (
            generation.ProductionGenerationDecision
        ),
        "validate_modeled_submit_once": (
            generation.ModeledSubmitOnceValidation
        ),
    }
    artifact = VersionedJsonArtifact
    artifact_list = list[generation.GenerationInventoryEntry]
    time_input = Union[datetime, str]
    exact_mapping = Mapping[str, object]
    expected_parameter_annotations: dict[str, dict[str, object]] = {
        "build_production_generation_claim": {
            "generation": int,
            "descriptor": artifact,
            "intent": artifact,
            "approval": artifact,
            "controller_baseline": artifact,
            "must_start_control_plane_ready": artifact,
            "submission_acquisition": artifact,
            "gpu_spend_snapshot": artifact,
            "generation_inventory": artifact_list,
            "previous_generation_terminal": Optional[artifact],
            "submit_attempt_id": str,
            "claim_created_at": time_input,
        },
        "validate_production_generation_claim": {
            "value": exact_mapping,
            "descriptor": artifact,
            "intent": artifact,
            "approval": artifact,
            "controller_baseline": artifact,
            "must_start_control_plane_ready": artifact,
            "submission_acquisition": artifact,
            "gpu_spend_snapshot": artifact,
            "generation_inventory": artifact_list,
            "previous_generation_terminal": Optional[artifact],
        },
        "production_generation_claim_file_bytes": {"value": exact_mapping},
        "production_generation_claim_file_sha256": {"value": exact_mapping},
        "production_generation_claim_s3_key": {
            "run_id": str,
            "generation": int,
        },
        "build_production_generation_start_decision": {
            "generation_claim": artifact,
            "descriptor": artifact,
            "intent": artifact,
            "approval": artifact,
            "controller_baseline": artifact,
            "must_start_control_plane_ready": artifact,
            "submission_acquisition": artifact,
            "decision": Literal["launch-once", "expire-unstarted"],
            "decided_at": time_input,
        },
        "validate_production_generation_start_decision": {
            "value": exact_mapping,
            "generation_claim": artifact,
            "descriptor": artifact,
            "intent": artifact,
            "approval": artifact,
            "controller_baseline": artifact,
            "must_start_control_plane_ready": artifact,
            "submission_acquisition": artifact,
            "now": time_input,
        },
        "production_generation_start_decision_file_bytes": {
            "value": exact_mapping
        },
        "production_generation_start_decision_file_sha256": {
            "value": exact_mapping
        },
        "production_generation_start_decision_s3_key": {
            "run_id": str,
            "generation": int,
        },
        "build_production_generation_terminal": {
            "generation_claim": artifact,
            "start_decision": artifact,
            "final_gpu_spend_snapshot": artifact,
            "terminal_at": time_input,
        },
        "validate_production_generation_terminal": {
            "value": exact_mapping,
            "generation_claim": artifact,
            "start_decision": artifact,
            "final_gpu_spend_snapshot": artifact,
            "now": time_input,
        },
        "production_generation_terminal_file_bytes": {"value": exact_mapping},
        "production_generation_terminal_file_sha256": {"value": exact_mapping},
        "production_generation_terminal_s3_key": {
            "run_id": str,
            "generation": int,
        },
        "validate_production_generation_inventory": {
            "entries": artifact_list,
            "run_id": str,
        },
        "decide_production_generation_action": {
            "generation_inventory": artifact_list,
            "descriptor": artifact,
            "intent": artifact,
            "approval": artifact,
            "controller_baseline": artifact,
            "must_start_control_plane_ready": artifact,
            "submission_acquisition": artifact,
            "gpu_spend_snapshot": artifact,
            "submit_attempt_id": str,
            "now": time_input,
        },
        "validate_modeled_submit_once": {
            "generation_claim": artifact,
            "start_decision": artifact,
            "descriptor": artifact,
            "intent": artifact,
            "approval": artifact,
            "controller_baseline": artifact,
            "must_start_control_plane_ready": artifact,
            "submission_acquisition": artifact,
            "gpu_spend_snapshot": artifact,
            "start_decision_receipt": (
                generation.UnambiguousStartDecisionCreateReceipt
            ),
            "post_create_generation_inventory": artifact_list,
            "post_create_audit_server_date": str,
            "now": time_input,
        },
    }
    assert set(expected_parameter_annotations) == set(expected_signatures)
    for name, parameter_names in expected_signatures.items():
        function = getattr(generation, name)
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == parameter_names
        assert all(
            parameter.default is inspect.Parameter.empty
            for parameter in signature.parameters.values()
        )
        first = next(iter(signature.parameters.values()))
        expected_first_kind = (
            inspect.Parameter.POSITIONAL_OR_KEYWORD
            if name in positional_first
            else inspect.Parameter.KEYWORD_ONLY
        )
        assert first.kind is expected_first_kind
        if first.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            assert all(
                parameter.kind is inspect.Parameter.KEYWORD_ONLY
                for parameter in tuple(signature.parameters.values())[1:]
            )
        else:
            assert all(
                parameter.kind is inspect.Parameter.KEYWORD_ONLY
                for parameter in signature.parameters.values()
            )
        hints = get_type_hints(function)
        assert set(hints) == {*parameter_names, "return"}
        assert {
            parameter: hints[parameter] for parameter in parameter_names
        } == expected_parameter_annotations[name]
        assert hints["return"] == expected_returns[name]


def test_mutant_open_registry_or_missing_generation_rows_is_killed() -> None:
    from mlx_vq.quality import glm52_sky_submission_modes as modes

    expected = {
        "generation-claim": (
            "glm52_sky_production_generation_claim_v1",
            "generation_claim_body_sha256",
            "generation-claim-singleton",
        ),
        "generation-start-decision": (
            "glm52_sky_production_generation_start_decision_v1",
            "start_decision_body_sha256",
            "generation-start-decision-singleton",
        ),
        "generation-terminal": (
            "glm52_sky_production_generation_terminal_v1",
            "generation_terminal_body_sha256",
            "generation-terminal-singleton",
        ),
    }
    assert len(modes._ROWS) == 17
    assert len({(row[0], row[1]) for row in modes._ROWS}) == 17
    assert len({(row[2], row[3], row[4]) for row in modes._ROWS}) == 17
    for kind, values in expected.items():
        contract = modes.record_contract(
            managed_mode="production",
            record_kind=kind,  # type: ignore[arg-type]
        )
        assert (
            contract.record_type,
            contract.digest_field,
            contract.address_kind,
        ) == values
        for other_mode in ("qualification", "cache-seed"):
            with pytest.raises(modes.SubmissionModeContractError):
                modes.record_contract(
                    managed_mode=other_mode,  # type: ignore[arg-type]
                    record_kind=kind,  # type: ignore[arg-type]
                )


def test_mutant_file_helper_signature_or_external_authority_input_is_killed() -> None:
    helpers = (
        generation.production_generation_claim_file_bytes,
        generation.production_generation_claim_file_sha256,
        generation.production_generation_start_decision_file_bytes,
        generation.production_generation_start_decision_file_sha256,
        generation.production_generation_terminal_file_bytes,
        generation.production_generation_terminal_file_sha256,
    )
    for helper in helpers:
        signature = inspect.signature(helper)
        assert list(signature.parameters) == ["value"]
        assert signature.parameters["value"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert get_type_hints(helper) == {
            "value": Mapping[str, object],
            "return": (
                bytes
                if helper.__name__.endswith("_file_bytes")
                else str
            ),
        }
    assert (
        inspect.signature(generation.validate_production_generation_inventory)
        .parameters.keys()
        == {"entries", "run_id"}
    )


def test_mutant_generation_padding_alias_or_pointer_keys_are_killed(
    chain: dict[str, object],
) -> None:
    claim_key = generation.production_generation_claim_s3_key(
        run_id=RUN_ID,
        generation=1,
    )
    assert claim_key == (
        f"campaigns/{RUN_ID}/submissions/production/generations/"
        "00000001/GENERATION_CLAIM.json"
    )
    decision_key = generation.production_generation_start_decision_s3_key(
        run_id=RUN_ID,
        generation=1,
    )
    terminal_key = generation.production_generation_terminal_s3_key(
        run_id=RUN_ID,
        generation=1,
    )
    assert decision_key == (
        f"campaigns/{RUN_ID}/submissions/production/generations/"
        "00000001/START_DECISION.json"
    )
    assert terminal_key == (
        f"campaigns/{RUN_ID}/submissions/production/generations/"
        "00000001/GENERATION_TERMINAL.json"
    )
    for helper in (
        generation.production_generation_claim_s3_key,
        generation.production_generation_start_decision_s3_key,
        generation.production_generation_terminal_s3_key,
    ):
        for invalid in (0, -1, 100_000_000, True, 1.0, IntSubclass(1)):
            with pytest.raises(generation.ProductionGenerationAuthorityError):
                helper(run_id=RUN_ID, generation=invalid)  # type: ignore[arg-type]
        for run_id in ("", "../run", "run/current", StringSubclass(RUN_ID)):
            with pytest.raises(generation.ProductionGenerationAuthorityError):
                helper(run_id=run_id, generation=1)  # type: ignore[arg-type]

    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    key_mutants = (
        claim_key.replace("/00000001/", "/0000001/"),
        claim_key.replace("/00000001/", "/000000001/"),
        claim_key.replace("/00000001/", "/1/"),
        claim_key.replace("/00000001/", "/+0000001/"),
        claim_key.replace("/00000001/", "/-0000001/"),
        claim_key.replace("/00000001/", "/0000000A/"),
        claim_key.replace(
            "/GENERATION_CLAIM.json",
            "/nested/GENERATION_CLAIM.json",
        ),
        claim_key.replace(
            "/GENERATION_CLAIM.json",
            "/GENERATION_CLAIM.json.tmp",
        ),
        claim_key.replace(
            "/GENERATION_CLAIM.json",
            "/GENERATION_CLAIM.json/child",
        ),
        claim_key.replace(
            "/GENERATION_CLAIM.json",
            "/generation_claim.json",
        ),
        claim_key.replace(
            "/GENERATION_CLAIM.json",
            "/GENERATION_CLAIM.JSON",
        ),
        claim_key.replace(
            "/GENERATION_CLAIM.json",
            "/.GENERATION_CLAIM.json",
        ),
        claim_key.replace(
            "/GENERATION_CLAIM.json",
            "/GENERATION_CLAIM.partial",
        ),
        claim_key.replace("/GENERATION_CLAIM.json", "/CLAIM.json"),
        claim_key.replace("/GENERATION_CLAIM.json", "/LATEST.json"),
        claim_key.replace("/GENERATION_CLAIM.json", "/CURRENT.json"),
        claim_key.replace(
            "/00000001/GENERATION_CLAIM.json",
            "/LATEST/GENERATION_CLAIM.json",
        ),
    )
    for key in key_mutants:
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_production_generation_inventory(
                [_entry(replace(claim_artifact, key=key))],
                run_id=RUN_ID,
            )


def test_mutant_record_field_set_self_hash_and_canonical_bytes_are_killed(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    launch = _decision(chain, claim)
    _, claim_artifact, expiry, expiry_artifact, snapshot, terminal, _ = (
        _expiry_chain(chain)
    )
    assert set(claim) == CLAIM_FIELDS
    assert set(launch) == DECISION_FIELDS
    assert set(terminal) == TERMINAL_FIELDS
    triples = (
        (
            claim,
            "generation_claim_body_sha256",
            generation.production_generation_claim_file_bytes,
            generation.production_generation_claim_file_sha256,
        ),
        (
            launch,
            "start_decision_body_sha256",
            generation.production_generation_start_decision_file_bytes,
            generation.production_generation_start_decision_file_sha256,
        ),
        (
            terminal,
            "generation_terminal_body_sha256",
            generation.production_generation_terminal_file_bytes,
            generation.production_generation_terminal_file_sha256,
        ),
    )
    for record, digest_field, byte_helper, sha_helper in triples:
        assert record[digest_field] == _sha(
            _canonical({k: v for k, v in record.items() if k != digest_field})
        )
        expected = _file_bytes(record)
        assert byte_helper(record) == expected
        assert sha_helper(record) == _sha(expected)
        for mutant in (
            {**record, "unknown": None},
            {k: v for k, v in record.items() if k != digest_field},
            {**record, digest_field: "0" * 64},
            DictSubclass(record),
        ):
            with pytest.raises(generation.ProductionGenerationAuthorityError):
                byte_helper(mutant)
            with pytest.raises(generation.ProductionGenerationAuthorityError):
                sha_helper(mutant)
    assert claim_artifact.version_id == "generation-claim-version-1"
    assert expiry_artifact.version_id == "generation-decision-version-1"
    assert snapshot.version_id == "final-snapshot-version-1"
    assert expiry["decision"] == "expire-unstarted"


@pytest.mark.parametrize(
    ("record_name", "field", "mutant"),
    [
        ("claim", "schema_version", True),
        ("claim", "generation", True),
        ("claim", "generation", 1.0),
        ("claim", "generation_text", StringSubclass("00000001")),
        ("claim", "approved_gpu_cost_usd", 1320),
        ("claim", "approved_gpu_cost_usd", FloatSubclass(1320.96)),
        ("claim", "gpu_spend_ledger_record_count", 8.0),
        ("claim", "open_allocation_count", False),
        ("claim", "previous_generation_terminal_key", ""),
        ("decision", "schema_version", True),
        ("decision", "generation", 1.0),
        ("decision", "decision", StringSubclass("launch-once")),
        ("terminal", "schema_version", True),
        ("terminal", "generation", 1.0),
        ("terminal", "final_approved_gpu_cost_usd", 1320),
        ("terminal", "final_gpu_spend_ledger_record_count", 8.0),
        ("terminal", "final_open_allocation_count", False),
    ],
)
def test_mutant_record_exact_type_tables_are_killed(
    chain: dict[str, object],
    record_name: str,
    field: str,
    mutant: object,
) -> None:
    claim = _claim(chain)
    decision = _decision(chain, claim)
    terminal = _expiry_chain(chain)[5]
    record, digest_field, helper = {
        "claim": (
            claim,
            "generation_claim_body_sha256",
            generation.production_generation_claim_file_bytes,
        ),
        "decision": (
            decision,
            "start_decision_body_sha256",
            generation.production_generation_start_decision_file_bytes,
        ),
        "terminal": (
            terminal,
            "generation_terminal_body_sha256",
            generation.production_generation_terminal_file_bytes,
        ),
    }[record_name]
    changed = copy.deepcopy(record)
    changed[field] = mutant
    changed = _rehash(changed, digest_field)
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        helper(changed)


def test_mutant_every_record_a_b_c_field_accepts_wrong_exact_builtin_type(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    decision = _decision(chain, claim)
    terminal = _expiry_chain(chain)[5]
    records = (
        (
            claim,
            CLAIM_FIELDS,
            CLAIM_INT_FIELDS,
            CLAIM_FLOAT_FIELDS,
            CLAIM_GENERATION_ONE_NULL_FIELDS,
            "generation_claim_body_sha256",
            generation.production_generation_claim_file_bytes,
            generation.production_generation_claim_file_sha256,
        ),
        (
            decision,
            DECISION_FIELDS,
            DECISION_INT_FIELDS,
            set(),
            set(),
            "start_decision_body_sha256",
            generation.production_generation_start_decision_file_bytes,
            generation.production_generation_start_decision_file_sha256,
        ),
        (
            terminal,
            TERMINAL_FIELDS,
            TERMINAL_INT_FIELDS,
            TERMINAL_FLOAT_FIELDS,
            set(),
            "generation_terminal_body_sha256",
            generation.production_generation_terminal_file_bytes,
            generation.production_generation_terminal_file_sha256,
        ),
    )
    for (
        record,
        field_set,
        integer_fields,
        float_fields,
        nullable_fields,
        digest_field,
        byte_helper,
        sha_helper,
    ) in records:
        string_fields = (
            field_set - integer_fields - float_fields - nullable_fields
        )
        assert set(record) == field_set
        assert (
            string_fields | integer_fields | float_fields | nullable_fields
        ) == field_set
        assert not (
            (string_fields & integer_fields)
            or (string_fields & float_fields)
            or (string_fields & nullable_fields)
            or (integer_fields & float_fields)
            or (integer_fields & nullable_fields)
            or (float_fields & nullable_fields)
        )
        for field in sorted(field_set):
            original = record[field]
            if field in string_fields:
                assert type(original) is str
                mutants: tuple[object, ...] = (
                    StringSubclass(original),
                    1,
                )
            elif field in integer_fields:
                assert type(original) is int
                mutants = (
                    True,
                    float(original),
                    IntSubclass(original),
                )
            elif field in float_fields:
                assert type(original) is float and math.isfinite(original)
                mutants = (
                    int(original),
                    True,
                    FloatSubclass(original),
                    math.nan,
                    math.inf,
                )
            else:
                assert field in nullable_fields and original is None
                mutants = ("not-null", StringSubclass("not-null"), 1)
            for mutant in mutants:
                changed = copy.deepcopy(record)
                changed[field] = mutant
                nonfinite = type(mutant) is float and not math.isfinite(mutant)
                if field != digest_field and not nonfinite:
                    changed = _rehash(changed, digest_field)
                for helper in (byte_helper, sha_helper):
                    with pytest.raises(
                        generation.ProductionGenerationAuthorityError
                    ):
                        helper(changed)

            missing = copy.deepcopy(record)
            del missing[field]
            if field != digest_field:
                missing = _rehash(missing, digest_field)
            for helper in (byte_helper, sha_helper):
                with pytest.raises(
                    generation.ProductionGenerationAuthorityError
                ):
                    helper(missing)

        constant_mutants = {
            "schema_version": 2,
            "record_type": "wrong-record-type",
            "managed_mode": "qualification",
        }
        if "decision" in record:
            constant_mutants["decision"] = "submit-again"
        if "terminal_outcome" in record:
            constant_mutants["terminal_outcome"] = "job-drained"
        for field, mutant in constant_mutants.items():
            changed = copy.deepcopy(record)
            changed[field] = mutant
            changed = _rehash(changed, digest_field)
            for helper in (byte_helper, sha_helper):
                with pytest.raises(
                    generation.ProductionGenerationAuthorityError
                ):
                    helper(changed)


def test_mutant_generation_n_record_a_previous_terminal_fields_accept_none_or_subclass(
    chain: dict[str, object],
) -> None:
    (
        _,
        claim_artifact,
        _,
        expiry_artifact,
        _,
        _,
        terminal_artifact,
    ) = _expiry_chain(chain)
    inventory = _inventory(
        claim_artifact,
        expiry_artifact,
        terminal_artifact,
    )
    next_chain, next_claim_at = _next_generation_chain(chain)
    second = generation.build_production_generation_claim(
        **_claim_kwargs(
            next_chain,
            generation_number=2,
            inventory=inventory,
            previous_terminal=terminal_artifact,
            attempt=ATTEMPT_B,
            claim_at=next_claim_at,
        )  # type: ignore[arg-type]
    )
    assert second["generation"] == 2
    for field in CLAIM_GENERATION_ONE_NULL_FIELDS:
        original = second[field]
        assert type(original) is str
        for mutant in (None, 1, StringSubclass(original)):
            changed = copy.deepcopy(second)
            changed[field] = mutant
            changed = _rehash(changed, "generation_claim_body_sha256")
            for helper in (
                generation.production_generation_claim_file_bytes,
                generation.production_generation_claim_file_sha256,
            ):
                with pytest.raises(
                    generation.ProductionGenerationAuthorityError
                ):
                    helper(changed)


def test_mutant_generation_one_previous_terminal_mixed_state_is_killed(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    for field in (
        "previous_generation_terminal_key",
        "previous_generation_terminal_file_sha256",
        "previous_generation_terminal_body_sha256",
        "previous_generation_terminal_version_id",
    ):
        changed = copy.deepcopy(claim)
        changed[field] = "x" * 64
        changed = _rehash(changed, "generation_claim_body_sha256")
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.production_generation_claim_file_bytes(changed)


def test_mutant_two_attempts_become_same_claim_or_different_key_is_killed(
    chain: dict[str, object],
) -> None:
    first = _claim(chain, submit_attempt_id=ATTEMPT_A)
    second = _claim(chain, submit_attempt_id=ATTEMPT_B)
    assert first != second
    assert generation.production_generation_claim_file_bytes(
        first
    ) != generation.production_generation_claim_file_bytes(second)
    assert generation.production_generation_claim_s3_key(
        run_id=RUN_ID,
        generation=1,
    ) == generation.production_generation_claim_s3_key(
        run_id=RUN_ID,
        generation=int(second["generation"]),
    )
    expected_identity = _sha(
        _canonical(
            {
                "account_id": "246813579024",
                "campaign_identity_sha256": first["campaign_identity_sha256"],
                "generation": 1,
                "intent_body_sha256": first["intent_body_sha256"],
                "managed_mode": "production",
                "run_id": RUN_ID,
                "submit_attempt_id": ATTEMPT_A,
            }
        )
    )
    assert first["sky_job_identity_sha256"] == expected_identity
    assert first["sky_job_name"] == second["sky_job_name"] == RUN_ID


def test_mutant_claim_roundtrip_recovery_requires_no_claim_receipt(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    artifact = _claim_artifact(claim)
    recovered = json.loads(artifact.raw)
    validated = generation.validate_production_generation_claim(
        recovered,
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
        generation_inventory=[],
        previous_generation_terminal=None,
    )
    assert validated == claim
    assert validated is not claim
    decision = _decision(chain, recovered)
    assert decision["decision"] == "launch-once"
    assert "claim_receipt" not in inspect.signature(
        generation.build_production_generation_start_decision
    ).parameters
    assert all("submit" not in name for name in generation.__all__ if name != "validate_modeled_submit_once")


@pytest.mark.parametrize(
    "artifact_name",
    [
        "descriptor",
        "intent",
        "approval",
        "baseline",
        "ready",
        "snapshot",
    ],
)
def test_mutant_source_version_or_exact_bytes_drift_is_killed(
    chain: dict[str, object],
    artifact_name: str,
) -> None:
    kwargs = _claim_kwargs(chain)
    parameter = {
        "descriptor": "descriptor",
        "intent": "intent",
        "approval": "approval",
        "baseline": "controller_baseline",
        "ready": "must_start_control_plane_ready",
        "acquisition": "submission_acquisition",
        "snapshot": "gpu_spend_snapshot",
    }[artifact_name]
    artifact = kwargs[parameter]
    assert type(artifact) is VersionedJsonArtifact
    kwargs[parameter] = replace(artifact, version_id=f"{artifact.version_id}-drift")
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_claim(  # type: ignore[arg-type]
            **kwargs
        )
    kwargs = _claim_kwargs(chain)
    kwargs[parameter] = replace(artifact, raw=artifact.raw + b" ")
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_claim(**kwargs)  # type: ignore[arg-type]
    kwargs = _claim_kwargs(chain)
    kwargs[parameter] = ArtifactSubclass(
        artifact.key,
        artifact.raw,
        artifact.version_id,
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_claim(**kwargs)  # type: ignore[arg-type]


def test_mutant_claim_validator_ignores_pinned_acquisition_version_drift(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    acquisition = chain["acquisition"]
    assert type(acquisition) is VersionedJsonArtifact
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_production_generation_claim(
            claim,
            descriptor=chain["descriptor"],  # type: ignore[arg-type]
            intent=chain["intent"],  # type: ignore[arg-type]
            approval=chain["approval"],  # type: ignore[arg-type]
            controller_baseline=chain["baseline"],  # type: ignore[arg-type]
            must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
            submission_acquisition=replace(
                acquisition,
                version_id=f"{acquisition.version_id}-drift",
            ),
            gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
            generation_inventory=[],
            previous_generation_terminal=None,
        )


@pytest.mark.parametrize(
    "version_id",
    [
        "",
        "null",
        "has space",
        "\n",
        "\x7f",
        "caf\u00e9",
        StringSubclass("version"),
    ],
)
def test_mutant_version_id_coercion_or_null_is_killed(
    chain: dict[str, object],
    version_id: object,
) -> None:
    artifact = chain["descriptor"]
    assert type(artifact) is VersionedJsonArtifact
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        _claim(
            chain,
            descriptor=replace(artifact, version_id=version_id),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value", "spy_kind"),
    [
        ("schema_version", True, "equality-only"),
        ("approved_gpu_hours", 24.0, "equality-only"),
        ("approved_hourly_usd", 55, "permissive"),
        ("approved_gpu_cost_usd", 1321, "permissive"),
        ("includes_qualification", 1, "equality-only"),
        ("includes_qualification", False, "permissive"),
        ("includes_recovery_instances", 1, "equality-only"),
        ("includes_recovery_instances", False, "permissive"),
        ("record_type", 1, "permissive"),
        ("approved_by", 1, "permissive"),
        ("approval_request", 1, "permissive"),
        ("approval_response", 1, "permissive"),
        ("approval_displayed_time", 1, "permissive"),
        ("approval_ingested_at", 1, "permissive"),
        ("evidence_source", 1, "permissive"),
        ("instance_type", 1, "permissive"),
        ("region", 1, "permissive"),
        (
            "approval_ingested_at",
            "2026-07-24T12:00:00+00:00",
            "permissive",
        ),
        ("slack_permalink", 1, "permissive"),
    ],
)
def test_mutant_coherently_readdressed_approval_type_gate_rejects_at_all_six_boundaries(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    spy_kind: str,
) -> None:
    base_claim = _claim(chain)
    base_decision = _decision(chain, base_claim)
    mutated_chain = _coherent_approval_chain(
        chain,
        field=field,
        value=value,
    )
    mutated_claim = _claim_for_chain(base_claim, mutated_chain)
    mutated_claim_artifact = _claim_artifact(
        mutated_claim,
        version_id="generation-claim-version-approval-mutant",
    )
    mutated_decision = _decision_for_claim(
        base_decision,
        mutated_claim,
        mutated_claim_artifact,
    )
    calls: list[object] = []
    original_value = chain["records"]["approval"][field]  # type: ignore[index]

    def equality_only(candidate: object) -> dict[str, object]:
        calls.append(candidate)
        assert type(candidate) is dict
        assert candidate[field] == original_value
        return dict(candidate)

    def permissive(candidate: object) -> dict[str, object]:
        calls.append(candidate)
        assert type(candidate) is dict
        return dict(candidate)

    if spy_kind == "equality-only":
        assert value == original_value
        validator = equality_only
    else:
        assert spy_kind == "permissive"
        validator = permissive
    monkeypatch.setattr(
        generation,
        "validate_gpu_spend_approval",
        validator,
    )
    _assert_approval_mutant_rejected_at_six_boundaries(
        chain=mutated_chain,
        claim=mutated_claim,
        decision=mutated_decision,
        public_validator_calls=calls,
    )


def test_mutant_invalid_exact_slack_permalink_bypasses_public_validator_at_six_boundaries(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_claim = _claim(chain)
    base_decision = _decision(chain, base_claim)
    mutated_chain = _coherent_approval_chain(
        chain,
        field="slack_permalink",
        value="not-a-slack-permalink",
    )
    mutated_claim = _claim_for_chain(base_claim, mutated_chain)
    mutated_claim_artifact = _claim_artifact(
        mutated_claim,
        version_id="generation-claim-version-invalid-permalink",
    )
    mutated_decision = _decision_for_claim(
        base_decision,
        mutated_claim,
        mutated_claim_artifact,
    )
    calls: list[object] = []

    def validating(candidate: object) -> dict[str, object]:
        calls.append(candidate)
        assert type(candidate) is dict
        return validate_gpu_spend_approval(candidate)

    monkeypatch.setattr(
        generation,
        "validate_gpu_spend_approval",
        validating,
    )
    _assert_approval_mutant_rejected_at_six_boundaries(
        chain=mutated_chain,
        claim=mutated_claim,
        decision=mutated_decision,
        public_validator_calls=calls,
        local_gate_must_precede_public_validator=False,
    )


def test_mutant_coherently_readdressed_approval_noncanonical_raw_bytes_reach_six_boundaries(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval_artifact = chain["approval"]
    approval_record = chain["records"]["approval"]
    assert type(approval_artifact) is VersionedJsonArtifact
    assert type(approval_record) is dict
    raw = approval_artifact.raw
    duplicate = raw[:-2] + b',"schema_version":1}\n'
    nonfinite = raw[:-2] + b',"unknown":NaN}\n'
    escaped = raw.replace(b"5:59 PM", b"5:59\\u0020PM", 1)
    unicode_raw = raw.replace(
        b"Alex Approver",
        "Alex Appr\u00f3ver".encode("utf-8"),
        1,
    )
    assert escaped != raw
    assert unicode_raw != raw
    mutants: dict[str, bytes] = {
        "missing-lf": raw[:-1],
        "double-lf": raw + b"\n",
        "leading-whitespace": b" " + raw,
        "pretty-json": (
            json.dumps(
                approval_record,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        ),
        "duplicate-key": duplicate,
        "nonfinite-number": nonfinite,
        "alternate-string-escape": escaped,
        "literal-unicode": unicode_raw,
        "malformed-utf8": b"\xff" + raw[1:],
        "bytes-subclass": BytesSubclass(raw),
    }
    base_claim = _claim(chain)
    base_decision = _decision(chain, base_claim)
    calls: list[object] = []

    def permissive(candidate: object) -> dict[str, object]:
        calls.append(candidate)
        assert type(candidate) is dict
        return dict(candidate)

    monkeypatch.setattr(
        generation,
        "validate_gpu_spend_approval",
        permissive,
    )
    for name, mutant_raw in mutants.items():
        mutated_chain = _coherent_approval_chain(
            chain,
            field="record_type",
            value=approval_record["record_type"],
            approval_raw_override=mutant_raw,
        )
        mutated_claim = _claim_for_chain(base_claim, mutated_chain)
        mutated_claim_artifact = _claim_artifact(
            mutated_claim,
            version_id=f"generation-claim-version-raw-{name}",
        )
        mutated_decision = _decision_for_claim(
            base_decision,
            mutated_claim,
            mutated_claim_artifact,
        )
        _assert_approval_mutant_rejected_at_six_boundaries(
            chain=mutated_chain,
            claim=mutated_claim,
            decision=mutated_decision,
            public_validator_calls=calls,
        )


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("schema_version", "int-subclass"),
        ("approved_gpu_hours", "int-subclass"),
        ("approved_hourly_usd", "float-subclass"),
        ("approved_gpu_cost_usd", "float-subclass"),
        ("approved_hourly_usd", "nan"),
        ("approved_hourly_usd", "positive-infinity"),
        ("approved_gpu_cost_usd", "negative-infinity"),
        ("record_type", "string-subclass"),
        ("approved_by", "string-subclass"),
        ("approval_request", "string-subclass"),
        ("approval_response", "string-subclass"),
        ("approval_displayed_time", "string-subclass"),
        ("approval_ingested_at", "string-subclass"),
        ("evidence_source", "string-subclass"),
        ("instance_type", "string-subclass"),
        ("region", "string-subclass"),
        ("approval_body_sha256", "string-subclass"),
        ("approval_body_sha256", "non-string"),
    ],
)
def test_mutant_parser_preserved_approval_subclass_or_nonfinite_type_gate_rejects_at_all_six_boundaries(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    mutation: str,
) -> None:
    approval_record = chain["records"]["approval"]
    assert type(approval_record) is dict
    original_value = approval_record[field]
    if mutation == "int-subclass":
        value: object = IntSubclass(int(original_value))
        spy_kind = "equality-only"
    elif mutation == "float-subclass":
        value = FloatSubclass(float(original_value))
        spy_kind = "equality-only"
    elif mutation == "string-subclass":
        value = StringSubclass(str(original_value))
        spy_kind = "equality-only"
    elif mutation == "nan":
        value = math.nan
        spy_kind = "permissive"
    elif mutation == "positive-infinity":
        value = math.inf
        spy_kind = "permissive"
    elif mutation == "negative-infinity":
        value = -math.inf
        spy_kind = "permissive"
    else:
        assert mutation == "non-string"
        value = 1
        spy_kind = "permissive"
    base_claim = _claim(chain)
    base_decision = _decision(chain, base_claim)
    original_parse = generation._parse_raw
    calls: list[object] = []

    def parse_with_runtime_type(
        raw: object,
        *,
        label: str,
    ) -> dict[str, object]:
        record = original_parse(raw, label=label)
        if label.startswith("approval"):
            record[field] = value
        return record

    def equality_only(candidate: object) -> dict[str, object]:
        calls.append(candidate)
        assert type(candidate) is dict
        assert candidate[field] == approval_record[field]
        return dict(candidate)

    def permissive(candidate: object) -> dict[str, object]:
        calls.append(candidate)
        assert type(candidate) is dict
        return dict(candidate)

    if spy_kind == "equality-only":
        assert value == approval_record[field]
        validator = equality_only
    else:
        assert spy_kind == "permissive"
        validator = permissive
    monkeypatch.setattr(generation, "_parse_raw", parse_with_runtime_type)
    monkeypatch.setattr(
        generation,
        "validate_gpu_spend_approval",
        validator,
    )
    _assert_approval_mutant_rejected_at_six_boundaries(
        chain=chain,
        claim=base_claim,
        decision=base_decision,
        public_validator_calls=calls,
    )


def test_mutant_slack_permalink_none_string_or_subclass_states_are_confused(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert chain["records"]["approval"]["slack_permalink"] is None  # type: ignore[index]
    assert _claim(chain)["generation"] == 1
    permalink = (
        "https://example.slack.com/archives/C01234567/"
        "p1722013140000100"
    )
    permalink_chain = _coherent_approval_chain(
        chain,
        field="slack_permalink",
        value=permalink,
    )
    claim = _claim(permalink_chain)
    decision = _decision(permalink_chain, claim)
    original_parse = generation._parse_raw
    calls: list[object] = []

    def parse_with_permalink_subclass(
        raw: object,
        *,
        label: str,
    ) -> dict[str, object]:
        record = original_parse(raw, label=label)
        if label.startswith("approval"):
            record["slack_permalink"] = StringSubclass(permalink)
        return record

    def equality_only(candidate: object) -> dict[str, object]:
        calls.append(candidate)
        assert type(candidate) is dict
        assert candidate["slack_permalink"] == permalink
        return dict(candidate)

    monkeypatch.setattr(
        generation,
        "_parse_raw",
        parse_with_permalink_subclass,
    )
    monkeypatch.setattr(
        generation,
        "validate_gpu_spend_approval",
        equality_only,
    )
    _assert_approval_mutant_rejected_at_six_boundaries(
        chain=permalink_chain,
        claim=claim,
        decision=decision,
        public_validator_calls=calls,
    )


def test_mutant_approval_validator_is_not_bypassed_at_six_boundaries(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def validating(value: object) -> dict[str, object]:
        calls.append("approval")
        assert type(value) is dict and set(value) == APPROVAL_FIELDS
        return validate_gpu_spend_approval(value)  # type: ignore[arg-type]

    monkeypatch.setattr(generation, "validate_gpu_spend_approval", validating)
    claim = _claim(chain)
    assert len(calls) >= 1
    calls.clear()
    generation.validate_production_generation_claim(
        claim,
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
        generation_inventory=[],
        previous_generation_terminal=None,
    )
    assert calls
    calls.clear()
    decision = _decision(chain, claim)
    assert calls
    calls.clear()
    generation.validate_production_generation_start_decision(
        decision,
        generation_claim=_claim_artifact(claim),
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        now=DECIDED_AT,
    )
    assert calls
    calls.clear()
    decision_artifact = _decision_artifact(decision)
    generation.validate_modeled_submit_once(
        generation_claim=_claim_artifact(claim),
        start_decision=decision_artifact,
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
        start_decision_receipt=_receipt(claim, decision, decision_artifact),
        post_create_generation_inventory=_inventory(
            _claim_artifact(claim),
            decision_artifact,
        ),
        post_create_audit_server_date="Sun, 26 Jul 2026 12:11:22 GMT",
        now=DECIDED_AT + timedelta(seconds=2),
    )
    assert calls
    calls.clear()
    generation.decide_production_generation_action(
        generation_inventory=[],
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
        submit_attempt_id=ATTEMPT_A,
        now=CLAIM_AT,
    )
    assert calls


def test_mutant_inventory_accepts_fork_delete_hole_or_unknown_key(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    valid = _inventory(claim_artifact)
    normalized = generation.validate_production_generation_inventory(
        valid,
        run_id=RUN_ID,
    )
    assert normalized["generation_count"] == 1
    assert normalized["highest_generation"] == 1
    assert normalized["open_generation"] == 1
    projection = normalized["generations"][0]["claim"]  # type: ignore[index]
    assert projection == {
        "record_type": "glm52_sky_production_generation_claim_v1",
        "key": claim_artifact.key,
        "raw": claim_artifact.raw,
        "version_id": claim_artifact.version_id,
        "file_sha256": _sha(claim_artifact.raw),
        "body_sha256": claim["generation_claim_body_sha256"],
    }
    assert generation.validate_production_generation_inventory(
        [],
        run_id=RUN_ID,
    ) == {
        "record_type": "glm52_sky_production_generation_inventory_validation_v1",
        "run_id": RUN_ID,
        "generation_count": 0,
        "highest_generation": None,
        "open_generation": None,
        "generations": [],
    }
    mutants = [
        valid + [replace(valid[0], version_id="second-version")],
        [replace(valid[0], is_latest=False)],
        [replace(valid[0], is_delete_marker=True, raw=None)],
        [replace(valid[0], key=f"{claim_artifact.key}.tmp")],
        [replace(valid[0], key=claim_artifact.key.replace("00000001", "00000002"))],
        [EntrySubclass(**valid[0].__dict__)],
    ]
    for mutant in mutants:
        mutant = sorted(
            mutant,
            key=lambda entry: (entry.key, entry.version_id, entry.is_delete_marker),
        )
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_production_generation_inventory(
                mutant,
                run_id=RUN_ID,
            )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_production_generation_inventory(
            ListSubclass(valid),  # type: ignore[arg-type]
            run_id=RUN_ID,
        )


def test_mutant_inventory_parses_noncanonical_claim_decision_or_terminal_raw(
    chain: dict[str, object],
) -> None:
    launch_claim = _claim(chain)
    launch_claim_artifact = _claim_artifact(launch_claim)
    launch_decision = _decision(chain, launch_claim)
    launch_decision_artifact = _decision_artifact(launch_decision)
    (
        _,
        expiry_claim_artifact,
        _,
        expiry_decision_artifact,
        _,
        _,
        terminal_artifact,
    ) = _expiry_chain(chain)
    cases = (
        (
            launch_claim_artifact,
            _inventory(launch_claim_artifact),
        ),
        (
            launch_decision_artifact,
            _inventory(launch_claim_artifact, launch_decision_artifact),
        ),
        (
            terminal_artifact,
            _inventory(
                expiry_claim_artifact,
                expiry_decision_artifact,
                terminal_artifact,
            ),
        ),
    )
    for artifact, valid_inventory in cases:
        raw = artifact.raw
        parsed = json.loads(raw)
        duplicate = raw[:-2] + b',"schema_version":1}\n'
        nonfinite = raw[:-2] + b',"unknown":NaN}\n'
        escaped = raw.replace(
            b'"production"',
            b'"\\u0070roduction"',
            1,
        )
        unicode_raw = raw.replace(
            RUN_ID.encode("ascii"),
            f"{RUN_ID}\u00e9".encode("utf-8"),
            1,
        )
        assert escaped != raw
        assert unicode_raw != raw
        mutants = (
            raw[:-1],
            raw + b"\n",
            b" " + raw,
            (
                json.dumps(
                    parsed,
                    sort_keys=True,
                    indent=2,
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
                + b"\n"
            ),
            duplicate,
            nonfinite,
            escaped,
            unicode_raw,
            b"\xff" + raw[1:],
            BytesSubclass(raw),
        )
        target_index = next(
            index
            for index, entry in enumerate(valid_inventory)
            if entry.key == artifact.key
        )
        for mutant_raw in mutants:
            mutated = list(valid_inventory)
            mutated[target_index] = replace(
                mutated[target_index],
                raw=mutant_raw,
            )
            with pytest.raises(generation.ProductionGenerationAuthorityError):
                generation.validate_production_generation_inventory(
                    mutated,
                    run_id=RUN_ID,
                )


def test_mutant_inventory_accepts_wrong_order_duplicate_second_open_or_missing_predecessor(
    chain: dict[str, object],
) -> None:
    launch_claim = _claim(chain)
    launch_claim_artifact = _claim_artifact(launch_claim)
    launch_decision = _decision(chain, launch_claim)
    launch_decision_artifact = _decision_artifact(launch_decision)
    (
        _,
        expiry_claim_artifact,
        _,
        expiry_decision_artifact,
        _,
        _,
        terminal_artifact,
    ) = _expiry_chain(chain)
    closed = _inventory(
        expiry_claim_artifact,
        expiry_decision_artifact,
        terminal_artifact,
    )
    next_chain, next_claim_at = _next_generation_chain(chain)
    second_claim = generation.build_production_generation_claim(
        **_claim_kwargs(
            next_chain,
            generation_number=2,
            inventory=closed,
            previous_terminal=terminal_artifact,
            attempt=ATTEMPT_B,
            claim_at=next_claim_at,
        )  # type: ignore[arg-type]
    )
    second_claim_artifact = _claim_artifact(
        second_claim,
        version_id="generation-claim-version-2",
    )
    mutants = (
        list(reversed(closed)),
        [closed[0], closed[0], *closed[1:]],
        [_entry(launch_decision_artifact)],
        _inventory(launch_claim_artifact, terminal_artifact),
        _inventory(
            launch_claim_artifact,
            launch_decision_artifact,
            terminal_artifact,
        ),
        _inventory(second_claim_artifact),
        _inventory(launch_claim_artifact, second_claim_artifact),
    )
    for mutant in mutants:
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_production_generation_inventory(
                mutant,
                run_id=RUN_ID,
            )

    valid_entry = _entry(launch_claim_artifact)
    entry_field_mutants = (
        replace(valid_entry, key=StringSubclass(valid_entry.key)),
        replace(valid_entry, raw=None),
        replace(valid_entry, version_id=1),  # type: ignore[arg-type]
        replace(valid_entry, is_latest=1),  # type: ignore[arg-type]
        replace(valid_entry, is_delete_marker=0),  # type: ignore[arg-type]
        generation.GenerationInventoryEntry(
            key=valid_entry.key,
            raw=valid_entry.raw,
            version_id=valid_entry.version_id,
            is_latest=True,
            is_delete_marker=True,
        ),
        generation.GenerationInventoryEntry(
            key=valid_entry.key,
            raw=b"",
            version_id=valid_entry.version_id,
            is_latest=True,
            is_delete_marker=True,
        ),
    )
    for mutant in entry_field_mutants:
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_production_generation_inventory(
                [mutant],
                run_id=RUN_ID,
            )


def test_mutant_decision_different_keys_or_deadline_boundary_is_killed(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    before = _decision(
        chain,
        claim,
        decision="launch-once",
        decided_at=MUST_START_BY - timedelta(seconds=1),
    )
    at = _decision(
        chain,
        claim,
        decision="expire-unstarted",
        decided_at=MUST_START_BY,
    )
    assert generation.production_generation_start_decision_s3_key(
        run_id=RUN_ID,
        generation=1,
    ) == generation.production_generation_start_decision_s3_key(
        run_id=RUN_ID,
        generation=int(at["generation"]),
    )
    assert before["decision"] == "launch-once"
    assert at["decision"] == "expire-unstarted"
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        _decision(
            chain,
            claim,
            decision="launch-once",
            decided_at=MUST_START_BY,
        )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        _decision(
            chain,
            claim,
            decision="expire-unstarted",
            decided_at=MUST_START_BY - timedelta(microseconds=1),
        )


def test_mutant_future_decision_or_expiry_requires_current_acquisition(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    launch = _decision(chain, claim)
    expiry = _decision(
        chain,
        claim,
        decision="expire-unstarted",
        decided_at=MUST_START_BY,
    )
    common = {
        "generation_claim": _claim_artifact(claim),
        "descriptor": chain["descriptor"],
        "intent": chain["intent"],
        "approval": chain["approval"],
        "controller_baseline": chain["baseline"],
        "must_start_control_plane_ready": chain["ready"],
        "submission_acquisition": chain["acquisition"],
    }
    assert generation.validate_production_generation_start_decision(
        launch,
        now=DECIDED_AT,
        **common,  # type: ignore[arg-type]
    ) == launch
    assert generation.validate_production_generation_start_decision(
        expiry,
        now=MUST_START_BY,
        **common,  # type: ignore[arg-type]
    ) == expiry
    for record, now in (
        (launch, DECIDED_AT - timedelta(microseconds=1)),
        (expiry, MUST_START_BY - timedelta(microseconds=1)),
    ):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_production_generation_start_decision(
                record,
                now=now,
                **common,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    "artifact_name",
    [
        "descriptor",
        "intent",
        "approval",
        "baseline",
        "ready",
        "acquisition",
        "snapshot",
    ],
)
def test_mutant_decision_or_modeled_validation_ignores_source_bytes_or_version(
    chain: dict[str, object],
    artifact_name: str,
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    decision = _decision(chain, claim)
    decision_artifact = _decision_artifact(decision)
    source_parameter = {
        "descriptor": "descriptor",
        "intent": "intent",
        "approval": "approval",
        "baseline": "controller_baseline",
        "ready": "must_start_control_plane_ready",
        "acquisition": "submission_acquisition",
        "snapshot": "gpu_spend_snapshot",
    }[artifact_name]
    artifact = chain[artifact_name]
    assert type(artifact) is VersionedJsonArtifact
    mutations = (
        replace(
            artifact,
            version_id=f"{artifact.version_id}-source-drift",
        ),
        replace(artifact, raw=artifact.raw + b" "),
        ArtifactSubclass(
            artifact.key,
            artifact.raw,
            artifact.version_id,
        ),
    )
    decision_sources = {
        "descriptor": chain["descriptor"],
        "intent": chain["intent"],
        "approval": chain["approval"],
        "controller_baseline": chain["baseline"],
        "must_start_control_plane_ready": chain["ready"],
        "submission_acquisition": chain["acquisition"],
    }
    modeled = {
        "generation_claim": claim_artifact,
        "start_decision": decision_artifact,
        **decision_sources,
        "gpu_spend_snapshot": chain["snapshot"],
        "start_decision_receipt": _receipt(
            claim,
            decision,
            decision_artifact,
        ),
        "post_create_generation_inventory": _inventory(
            claim_artifact,
            decision_artifact,
        ),
        "post_create_audit_server_date": (
            "Sun, 26 Jul 2026 12:11:22 GMT"
        ),
        "now": DECIDED_AT + timedelta(seconds=2),
    }
    for mutation in mutations:
        if artifact_name != "snapshot":
            with pytest.raises(generation.ProductionGenerationAuthorityError):
                generation.build_production_generation_start_decision(
                    generation_claim=claim_artifact,
                    decision="launch-once",
                    decided_at=DECIDED_AT,
                    **{
                        **decision_sources,
                        source_parameter: mutation,
                    },  # type: ignore[arg-type]
                )
            with pytest.raises(generation.ProductionGenerationAuthorityError):
                generation.validate_production_generation_start_decision(
                    decision,
                    generation_claim=claim_artifact,
                    now=DECIDED_AT,
                    **{
                        **decision_sources,
                        source_parameter: mutation,
                    },  # type: ignore[arg-type]
                )
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **modeled,
                    source_parameter: mutation,
                }  # type: ignore[arg-type]
            )


def test_mutant_h1d_freshness_uses_inclusive_60_and_rejects_61_at_decision_and_audit(
    chain: dict[str, object],
) -> None:
    fresh_chain, claim_at = _next_generation_chain(
        chain,
        must_start_offset_seconds=120,
    )
    acquired_text = fresh_chain["records"]["acquisition"]["acquired_at"]  # type: ignore[index]
    assert type(acquired_text) is str
    acquired_at = datetime.strptime(
        acquired_text,
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    oldest_observation_text = fresh_chain["records"]["acquisition"][  # type: ignore[index]
        "spend_snapshot_observed_at"
    ]
    assert type(oldest_observation_text) is str
    oldest_observation = datetime.strptime(
        oldest_observation_text,
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    fresh_60 = oldest_observation + timedelta(seconds=60)
    stale_61 = oldest_observation + timedelta(seconds=61)

    at_boundary = _claim(fresh_chain, claim_created_at=fresh_60)
    assert at_boundary["claim_created_at"] == fresh_60.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        _claim(fresh_chain, claim_created_at=stale_61)

    claim = _claim(fresh_chain, claim_created_at=claim_at)
    claim_artifact = _claim_artifact(
        claim,
        version_id="generation-claim-version-freshness",
    )
    decision_at = acquired_at + timedelta(seconds=15)
    decision = generation.build_production_generation_start_decision(
        generation_claim=claim_artifact,
        descriptor=fresh_chain["descriptor"],  # type: ignore[arg-type]
        intent=fresh_chain["intent"],  # type: ignore[arg-type]
        approval=fresh_chain["approval"],  # type: ignore[arg-type]
        controller_baseline=fresh_chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=fresh_chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=fresh_chain["acquisition"],  # type: ignore[arg-type]
        decision="launch-once",
        decided_at=decision_at,
    )
    decision_sources = {
        "generation_claim": claim_artifact,
        "descriptor": fresh_chain["descriptor"],
        "intent": fresh_chain["intent"],
        "approval": fresh_chain["approval"],
        "controller_baseline": fresh_chain["baseline"],
        "must_start_control_plane_ready": fresh_chain["ready"],
        "submission_acquisition": fresh_chain["acquisition"],
    }
    assert generation.validate_production_generation_start_decision(
        decision,
        now=fresh_60,
        **decision_sources,  # type: ignore[arg-type]
    ) == decision
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_production_generation_start_decision(
            decision,
            now=stale_61,
            **decision_sources,  # type: ignore[arg-type]
        )

    decision_artifact = _decision_artifact(
        decision,
        version_id="generation-decision-version-freshness",
    )
    response_at = fresh_60 - timedelta(seconds=1)
    server_at = fresh_60 - timedelta(seconds=5)
    receipt = _receipt(
        claim,
        decision,
        decision_artifact,
        request_started_at=response_at - timedelta(seconds=1),
        response_received_at=response_at,
        server_date=server_at.strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        ),
    )
    modeled = {
        "generation_claim": claim_artifact,
        "start_decision": decision_artifact,
        "descriptor": fresh_chain["descriptor"],
        "intent": fresh_chain["intent"],
        "approval": fresh_chain["approval"],
        "controller_baseline": fresh_chain["baseline"],
        "must_start_control_plane_ready": fresh_chain["ready"],
        "submission_acquisition": fresh_chain["acquisition"],
        "gpu_spend_snapshot": fresh_chain["snapshot"],
        "start_decision_receipt": receipt,
        "post_create_generation_inventory": _inventory(
            claim_artifact,
            decision_artifact,
        ),
    }
    boundary_result = generation.validate_modeled_submit_once(
        **modeled,  # type: ignore[arg-type]
        post_create_audit_server_date=server_at.strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        ),
        now=fresh_60,
    )
    assert boundary_result.conservative_validation_at == fresh_60.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            **modeled,  # type: ignore[arg-type]
            post_create_audit_server_date=(
                stale_61 - timedelta(seconds=5)
            ).strftime("%a, %d %b %Y %H:%M:%S GMT"),
            now=stale_61,
        )


def test_mutant_deadline_boundary_accepts_equality_for_launch_or_floors_fractional_time(
    chain: dict[str, object],
) -> None:
    deadline_chain, ordinary_claim_at = _next_generation_chain(chain)
    deadline_text = deadline_chain["records"]["intent"]["must_start_by"]  # type: ignore[index]
    assert type(deadline_text) is str
    deadline = datetime.strptime(
        deadline_text,
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    one_second_before = deadline - timedelta(seconds=1)
    one_microsecond_before = deadline - timedelta(microseconds=1)

    claim_at_last_whole_second = _claim(
        deadline_chain,
        claim_created_at=one_second_before,
    )
    assert claim_at_last_whole_second["claim_created_at"] == (
        one_second_before.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    for rejected_claim_at in (
        one_microsecond_before,
        deadline,
        deadline + timedelta(seconds=1),
    ):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            _claim(
                deadline_chain,
                claim_created_at=rejected_claim_at,
            )

    claim = _claim(
        deadline_chain,
        claim_created_at=ordinary_claim_at,
    )
    launch = _decision(
        deadline_chain,
        claim,
        decision="launch-once",
        decided_at=one_second_before,
    )
    assert launch["decided_at"] == one_second_before.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    for rejected_decision_at in (
        one_microsecond_before,
        deadline,
        deadline + timedelta(seconds=1),
    ):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            _decision(
                deadline_chain,
                claim,
                decision="launch-once",
                decided_at=rejected_decision_at,
            )
    assert _decision(
        deadline_chain,
        claim,
        decision="expire-unstarted",
        decided_at=deadline,
    )["decided_at"] == deadline_text
    assert _decision(
        deadline_chain,
        claim,
        decision="expire-unstarted",
        decided_at=deadline + timedelta(seconds=1),
    )["decision"] == "expire-unstarted"
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        _decision(
            deadline_chain,
            claim,
            decision="expire-unstarted",
            decided_at=one_second_before,
        )

    decision_at = deadline - timedelta(seconds=6)
    modeled_decision = _decision(
        deadline_chain,
        claim,
        decision="launch-once",
        decided_at=decision_at,
    )
    claim_artifact = _claim_artifact(
        claim,
        version_id="generation-claim-version-deadline",
    )
    modeled_decision = _decision_for_claim(
        modeled_decision,
        claim,
        claim_artifact,
    )
    decision_artifact = _decision_artifact(
        modeled_decision,
        version_id="generation-decision-version-deadline",
    )
    common = {
        "generation_claim": claim_artifact,
        "start_decision": decision_artifact,
        "descriptor": deadline_chain["descriptor"],
        "intent": deadline_chain["intent"],
        "approval": deadline_chain["approval"],
        "controller_baseline": deadline_chain["baseline"],
        "must_start_control_plane_ready": deadline_chain["ready"],
        "submission_acquisition": deadline_chain["acquisition"],
        "gpu_spend_snapshot": deadline_chain["snapshot"],
        "post_create_generation_inventory": _inventory(
            claim_artifact,
            decision_artifact,
        ),
    }
    server = deadline - timedelta(seconds=6)
    last_second_receipt = _receipt(
        claim,
        modeled_decision,
        decision_artifact,
        request_started_at=decision_at,
        response_received_at=one_second_before,
        server_date=server.strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        ),
    )
    accepted = generation.validate_modeled_submit_once(
        **common,  # type: ignore[arg-type]
        start_decision_receipt=last_second_receipt,
        post_create_audit_server_date=server.strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        ),
        now=one_second_before,
    )
    assert accepted.conservative_validation_at == (
        one_second_before.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    for response_at, audit_at in (
        (one_microsecond_before, one_microsecond_before),
        (deadline, deadline),
        (deadline + timedelta(seconds=1), deadline + timedelta(seconds=1)),
    ):
        request_at = response_at - timedelta(seconds=5)
        server_at = response_at - timedelta(seconds=5)
        receipt = _receipt(
            claim,
            modeled_decision,
            decision_artifact,
            request_started_at=request_at,
            response_received_at=response_at,
            server_date=server_at.strftime(
                "%a, %d %b %Y %H:%M:%S GMT"
            ),
        )
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **common,  # type: ignore[arg-type]
                start_decision_receipt=receipt,
                post_create_audit_server_date=server_at.strftime(
                    "%a, %d %b %Y %H:%M:%S GMT"
                ),
                now=audit_at,
            )

    audit_receipt = _receipt(
        claim,
        modeled_decision,
        decision_artifact,
        request_started_at=decision_at,
        response_received_at=deadline - timedelta(seconds=2),
        server_date=server.strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        ),
    )
    assert generation.validate_modeled_submit_once(
        **common,  # type: ignore[arg-type]
        start_decision_receipt=audit_receipt,
        post_create_audit_server_date=server.strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        ),
        now=one_second_before,
    ).validation_result == "modeled-submit-once-valid"
    for rejected_audit_at in (one_microsecond_before, deadline):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **common,  # type: ignore[arg-type]
                start_decision_receipt=audit_receipt,
                post_create_audit_server_date=(
                    deadline - timedelta(seconds=5)
                ).strftime("%a, %d %b %Y %H:%M:%S GMT"),
                now=rejected_audit_at,
            )

    post_audit_only_receipt = _receipt(
        claim,
        modeled_decision,
        decision_artifact,
        request_started_at=decision_at,
        response_received_at=deadline - timedelta(seconds=2),
        server_date=(deadline - timedelta(seconds=7)).strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        ),
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            **common,  # type: ignore[arg-type]
            start_decision_receipt=post_audit_only_receipt,
            post_create_audit_server_date=(
                deadline - timedelta(seconds=4)
            ).strftime("%a, %d %b %Y %H:%M:%S GMT"),
            now=deadline + timedelta(seconds=1),
        )


@pytest.mark.parametrize(
    "time_value",
    [
        "2026-07-26T12:11:10.000000Z",
        "2026-07-26T12:11:10+00:00",
        "2026-07-26T12:11:10z",
        "2026-07-26T12:11:60Z",
        datetime(2026, 7, 26, 12, 11, 10),
        DatetimeSubclass(2026, 7, 26, 12, 11, 10, tzinfo=timezone.utc),
    ],
)
def test_mutant_noncanonical_time_or_datetime_subclass_is_killed(
    chain: dict[str, object],
    time_value: object,
) -> None:
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        _claim(chain, claim_created_at=time_value)  # type: ignore[arg-type]


def test_mutant_microsecond_is_floored_or_interval_is_ceiled_before_measurement(
    chain: dict[str, object],
) -> None:
    one_microsecond_before = MUST_START_BY - timedelta(microseconds=1)
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        _claim(chain, claim_created_at=one_microsecond_before)
    exact_second = CLAIM_AT.astimezone(timezone(timedelta(hours=-7)))
    claim = _claim(chain, claim_created_at=exact_second)
    assert claim["claim_created_at"] == "2026-07-26T12:11:10Z"
    fractional = CLAIM_AT.replace(microsecond=1)
    rounded = _claim(chain, claim_created_at=fractional)
    assert rounded["claim_created_at"] == "2026-07-26T12:11:11Z"


def test_mutant_timezone_normalization_requeries_state_or_uses_astimezone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offset_value = datetime(
        2026,
        7,
        26,
        5,
        11,
        10,
        123456,
        tzinfo=timezone(timedelta(hours=-7)),
    )
    exact, ceiling = generation._normalize_time_input(
        offset_value,
        field="test_time",
    )
    assert exact == datetime(
        2026,
        7,
        26,
        12,
        11,
        10,
        123456,
        tzinfo=timezone.utc,
    )
    assert exact.microsecond == 123456
    assert ceiling == "2026-07-26T12:11:11Z"

    for invalid in (
        datetime(
            2026,
            7,
            26,
            12,
            11,
            10,
            tzinfo=timezone(timedelta(seconds=30)),
        ),
        datetime.min.replace(
            tzinfo=timezone(timedelta(hours=23, minutes=59))
        ),
        datetime.max.replace(
            tzinfo=timezone(-timedelta(hours=23, minutes=59))
        ),
    ):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation._normalize_time_input(invalid, field="test_time")

    ordinary_custom = StatefulTimezone([timedelta(0)])
    custom_value = datetime(
        2026,
        7,
        26,
        12,
        11,
        10,
        tzinfo=ordinary_custom,
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation._normalize_time_input(custom_value, field="test_time")
    assert ordinary_custom.calls == 0

    stateful = StatefulTimezone(
        [timedelta(hours=-7), timedelta(hours=3)]
    )
    stateful_value = datetime(
        2026,
        7,
        26,
        5,
        11,
        10,
        1,
        tzinfo=stateful,
    )
    monkeypatch.setattr(generation, "timezone", StatefulTimezone)
    exact, ceiling = generation._normalize_time_input(
        stateful_value,
        field="test_time",
    )
    assert stateful.calls == 1
    assert exact == datetime(
        2026,
        7,
        26,
        12,
        11,
        10,
        1,
        tzinfo=timezone.utc,
    )
    assert ceiling == "2026-07-26T12:11:11Z"

    invalid_offsets = (
        StatefulTimezone([timedelta(seconds=30)]),
        StatefulTimezone([timedelta(microseconds=1)]),
        StatefulTimezone([timedelta(hours=24)]),
        StatefulTimezone([TimedeltaSubclass(0)]),
        StatefulTimezone(failure=RuntimeError("state lookup failed")),
    )
    for invalid_zone in invalid_offsets:
        invalid_value = datetime(
            2026,
            7,
            26,
            12,
            11,
            10,
            tzinfo=invalid_zone,
        )
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation._normalize_time_input(
                invalid_value,
                field="test_time",
            )
        assert invalid_zone.calls == 1

    source_tree = ast.parse(
        Path(generation.__file__).read_text(encoding="utf-8")
    )
    normalize = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_normalize_time_input"
    )
    attribute_calls = {
        node.func.attr
        for node in ast.walk(normalize)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert "astimezone" not in attribute_calls
    assert sum(
        1
        for node in ast.walk(normalize)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "utcoffset"
    ) == 1


def test_mutant_time_normalization_accepts_ceiling_overflow_or_exact_fractional_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation._normalize_time_input(
            datetime.max.replace(tzinfo=timezone.utc),
            field="test_time",
        )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation._normalize_time_input(
            datetime(
                2026,
                7,
                26,
                12,
                11,
                10,
                tzinfo=timezone(timedelta(microseconds=1)),
            ),
            field="test_time",
        )
    negative_24_hours = StatefulTimezone([-timedelta(hours=24)])
    monkeypatch.setattr(generation, "timezone", StatefulTimezone)
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation._normalize_time_input(
            datetime(
                2026,
                7,
                26,
                12,
                11,
                10,
                tzinfo=negative_24_hours,
            ),
            field="test_time",
        )
    assert negative_24_hours.calls == 1


def test_mutant_bad_imf_dates_or_receipt_checksums_metadata_and_bounds_are_killed(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    decision = _decision(chain, claim)
    claim_artifact = _claim_artifact(claim)
    decision_artifact = _decision_artifact(decision)
    receipt = _receipt(claim, decision, decision_artifact)
    kwargs = {
        "generation_claim": claim_artifact,
        "start_decision": decision_artifact,
        "descriptor": chain["descriptor"],
        "intent": chain["intent"],
        "approval": chain["approval"],
        "controller_baseline": chain["baseline"],
        "must_start_control_plane_ready": chain["ready"],
        "submission_acquisition": chain["acquisition"],
        "gpu_spend_snapshot": chain["snapshot"],
        "start_decision_receipt": receipt,
        "post_create_generation_inventory": _inventory(
            claim_artifact,
            decision_artifact,
        ),
        "post_create_audit_server_date": "Sun, 26 Jul 2026 12:11:22 GMT",
        "now": DECIDED_AT + timedelta(seconds=2),
    }
    result = generation.validate_modeled_submit_once(**kwargs)  # type: ignore[arg-type]
    assert result == generation.ModeledSubmitOnceValidation(
        validation_result="modeled-submit-once-valid",
        generation=1,
        submit_attempt_id=ATTEMPT_A,
        sky_job_identity_sha256=str(claim["sky_job_identity_sha256"]),
        claim_key=claim_artifact.key,
        claim_version_id=claim_artifact.version_id,
        start_decision_key=decision_artifact.key,
        start_decision_version_id=decision_artifact.version_id,
        validated_at="2026-07-26T12:11:22Z",
        conservative_validation_at="2026-07-26T12:11:27Z",
        must_start_by="2026-07-26T12:11:30Z",
    )
    assert not hasattr(result, "file_bytes")
    receipt_mutants = (
        replace(receipt, expected_bucket_owner="000000000000"),
        replace(receipt, request_checksum_sha256_base64="A" * 43 + "="),
        replace(receipt, response_checksum_sha256_base64="A" * 43 + "="),
        replace(receipt, immutable_metadata=tuple(reversed(receipt.immutable_metadata))),
        replace(receipt, http_status=True),
        replace(receipt, source="readback"),
        replace(receipt, outcome="conflict"),
        replace(receipt, etag='"ABCDEF0123456789ABCDEF0123456789"'),
        replace(receipt, aws_request_id="short"),
        replace(
            receipt,
            response_received_at=DECIDED_AT + timedelta(seconds=5, microseconds=1),
        ),
        ReceiptSubclass(**receipt.__dict__),
    )
    for mutant in receipt_mutants:
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{**kwargs, "start_decision_receipt": mutant}  # type: ignore[arg-type]
            )
    for server_date in (
        "Mon, 26 Jul 2026 12:11:22 GMT",
        "Sun, 26 Jul 2026 12:11:22 UTC",
        "sun, 26 Jul 2026 12:11:22 GMT",
        "Sun, 26 Jul 2026 12:11:22.0 GMT",
        " Sun, 26 Jul 2026 12:11:22 GMT",
    ):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **kwargs,
                    "post_create_audit_server_date": server_date,
                }  # type: ignore[arg-type]
            )


def test_mutant_receipt_exact_type_or_every_modeled_field_is_ignored(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    decision = _decision(chain, claim)
    claim_artifact = _claim_artifact(claim)
    decision_artifact = _decision_artifact(decision)
    receipt = _receipt(claim, decision, decision_artifact)
    kwargs = {
        "generation_claim": claim_artifact,
        "start_decision": decision_artifact,
        "descriptor": chain["descriptor"],
        "intent": chain["intent"],
        "approval": chain["approval"],
        "controller_baseline": chain["baseline"],
        "must_start_control_plane_ready": chain["ready"],
        "submission_acquisition": chain["acquisition"],
        "gpu_spend_snapshot": chain["snapshot"],
        "start_decision_receipt": receipt,
        "post_create_generation_inventory": _inventory(
            claim_artifact,
            decision_artifact,
        ),
        "post_create_audit_server_date": "Sun, 26 Jul 2026 12:11:22 GMT",
        "now": DECIDED_AT + timedelta(seconds=2),
    }

    for field in fields(receipt):
        original = getattr(receipt, field.name)
        if type(original) is str:
            type_mutants: tuple[object, ...] = (
                StringSubclass(original),
                1,
            )
        elif type(original) is int:
            type_mutants = (
                True,
                float(original),
                IntSubclass(original),
            )
        elif type(original) is tuple:
            type_mutants = (
                TupleSubclass(original),
                list(original),
                (
                    (StringSubclass(original[0][0]), original[0][1]),
                    *original[1:],
                ),
                (
                    (original[0][0], StringSubclass(original[0][1])),
                    *original[1:],
                ),
                (
                    ListSubclass(list(original[0])),
                    *original[1:],
                ),
            )
        else:
            assert type(original) is datetime
            type_mutants = (
                DatetimeSubclass(
                    original.year,
                    original.month,
                    original.day,
                    original.hour,
                    original.minute,
                    original.second,
                    original.microsecond,
                    tzinfo=original.tzinfo,
                ),
                StringSubclass(
                    original.isoformat().replace("+00:00", "Z")
                ),
                1,
            )
        for mutant in type_mutants:
            with pytest.raises(generation.ProductionGenerationAuthorityError):
                generation.validate_modeled_submit_once(
                    **{
                        **kwargs,
                        "start_decision_receipt": replace(
                            receipt,
                            **{field.name: mutant},
                        ),
                    }  # type: ignore[arg-type]
                )

    semantic_mutants: dict[str, object] = {
        "record_kind": "generation-terminal",
        "account_id": "000000000000",
        "region": "us-east-1",
        "bucket": "other-bucket",
        "expected_bucket_owner": "000000000000",
        "operation": "GetObject",
        "if_none_match": "etag",
        "generation": 2,
        "generation_text": "00000002",
        "submit_attempt_id": ATTEMPT_B,
        "key": f"{decision_artifact.key}.other",
        "content_length": len(decision_artifact.raw) + 1,
        "candidate_file_sha256": "0" * 64,
        "request_checksum_algorithm": "MD5",
        "request_checksum_sha256_base64": "A" * 43 + "=",
        "immutable_metadata": tuple(reversed(receipt.immutable_metadata)),
        "version_id": "other-version",
        "etag": '"ABCDEF0123456789ABCDEF0123456789"',
        "response_checksum_sha256_base64": "A" * 43 + "=",
        "aws_request_id": "short",
        "server_date": "Mon, 26 Jul 2026 12:11:21 GMT",
        "request_started_at": DECIDED_AT - timedelta(microseconds=1),
        "response_received_at": DECIDED_AT + timedelta(seconds=6),
        "http_status": 201,
        "outcome": "conflict",
        "source": "readback",
    }
    assert set(semantic_mutants) == {
        field.name for field in fields(receipt)
    }
    for field, mutant in semantic_mutants.items():
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **kwargs,
                    "start_decision_receipt": replace(
                        receipt,
                        **{field: mutant},
                    ),
                }  # type: ignore[arg-type]
            )

    for non_receipt in (
        receipt.__dict__,
        json.loads(json.dumps(receipt.__dict__, default=str)),
        ReceiptSubclass(**receipt.__dict__),
        object(),
    ):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **kwargs,
                    "start_decision_receipt": non_receipt,
                }  # type: ignore[arg-type]
            )


def test_mutant_receipt_duration_audit_delay_or_either_server_skew_uses_open_bounds(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    decision = _decision(
        chain,
        claim,
        decided_at=CLAIM_AT,
    )
    claim_artifact = _claim_artifact(claim)
    decision_artifact = _decision_artifact(decision)
    base_kwargs = {
        "generation_claim": claim_artifact,
        "start_decision": decision_artifact,
        "descriptor": chain["descriptor"],
        "intent": chain["intent"],
        "approval": chain["approval"],
        "controller_baseline": chain["baseline"],
        "must_start_control_plane_ready": chain["ready"],
        "submission_acquisition": chain["acquisition"],
        "gpu_spend_snapshot": chain["snapshot"],
        "post_create_generation_inventory": _inventory(
            claim_artifact,
            decision_artifact,
        ),
    }

    exact_duration_receipt = _receipt(
        claim,
        decision,
        decision_artifact,
        request_started_at=CLAIM_AT,
        response_received_at=CLAIM_AT + timedelta(seconds=5),
        server_date="Sun, 26 Jul 2026 12:11:11 GMT",
    )
    assert generation.validate_modeled_submit_once(
        **base_kwargs,  # type: ignore[arg-type]
        start_decision_receipt=exact_duration_receipt,
        post_create_audit_server_date="Sun, 26 Jul 2026 12:11:20 GMT",
        now=CLAIM_AT + timedelta(seconds=5),
    ).validation_result == "modeled-submit-once-valid"
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            **base_kwargs,  # type: ignore[arg-type]
            start_decision_receipt=replace(
                exact_duration_receipt,
                response_received_at=(
                    CLAIM_AT + timedelta(seconds=5, microseconds=1)
                ),
            ),
            post_create_audit_server_date=(
                "Sun, 26 Jul 2026 12:11:20 GMT"
            ),
            now=CLAIM_AT + timedelta(seconds=5, microseconds=1),
        )

    audit_receipt = _receipt(
        claim,
        decision,
        decision_artifact,
        request_started_at=CLAIM_AT,
        response_received_at=CLAIM_AT + timedelta(seconds=1),
        server_date="Sun, 26 Jul 2026 12:11:11 GMT",
    )
    exact_audit_now = CLAIM_AT + timedelta(seconds=16)
    assert generation.validate_modeled_submit_once(
        **base_kwargs,  # type: ignore[arg-type]
        start_decision_receipt=audit_receipt,
        post_create_audit_server_date="Sun, 26 Jul 2026 12:11:22 GMT",
        now=exact_audit_now,
    ).validation_result == "modeled-submit-once-valid"
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            **base_kwargs,  # type: ignore[arg-type]
            start_decision_receipt=audit_receipt,
            post_create_audit_server_date=(
                "Sun, 26 Jul 2026 12:11:22 GMT"
            ),
            now=exact_audit_now + timedelta(microseconds=1),
        )

    skew_receipt = _receipt(
        claim,
        decision,
        decision_artifact,
        request_started_at=CLAIM_AT,
        response_received_at=CLAIM_AT + timedelta(seconds=1),
        server_date="Sun, 26 Jul 2026 12:11:16 GMT",
    )
    assert generation.validate_modeled_submit_once(
        **base_kwargs,  # type: ignore[arg-type]
        start_decision_receipt=skew_receipt,
        post_create_audit_server_date="Sun, 26 Jul 2026 12:11:17 GMT",
        now=CLAIM_AT + timedelta(seconds=2),
    ).validation_result == "modeled-submit-once-valid"
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            **base_kwargs,  # type: ignore[arg-type]
            start_decision_receipt=replace(
                skew_receipt,
                response_received_at=(
                    CLAIM_AT
                    + timedelta(seconds=1)
                    - timedelta(microseconds=1)
                ),
            ),
            post_create_audit_server_date=(
                "Sun, 26 Jul 2026 12:11:17 GMT"
            ),
            now=CLAIM_AT + timedelta(seconds=2),
        )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            **base_kwargs,  # type: ignore[arg-type]
            start_decision_receipt=skew_receipt,
            post_create_audit_server_date=(
                "Sun, 26 Jul 2026 12:11:17 GMT"
            ),
            now=(
                CLAIM_AT
                + timedelta(seconds=2)
                - timedelta(microseconds=1)
            ),
        )

    negative_skew_receipt = _receipt(
        claim,
        decision,
        decision_artifact,
        request_started_at=CLAIM_AT,
        response_received_at=CLAIM_AT + timedelta(seconds=1),
        server_date="Sun, 26 Jul 2026 12:11:06 GMT",
    )
    string_time_receipt = replace(
        negative_skew_receipt,
        request_started_at="2026-07-26T12:11:10Z",
        response_received_at="2026-07-26T12:11:11Z",
    )
    assert generation.validate_modeled_submit_once(
        **base_kwargs,  # type: ignore[arg-type]
        start_decision_receipt=string_time_receipt,
        post_create_audit_server_date="Sun, 26 Jul 2026 12:11:07 GMT",
        now=CLAIM_AT + timedelta(seconds=2),
    ).validation_result == "modeled-submit-once-valid"
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            **base_kwargs,  # type: ignore[arg-type]
            start_decision_receipt=replace(
                negative_skew_receipt,
                response_received_at=(
                    CLAIM_AT
                    + timedelta(seconds=1, microseconds=1)
                ),
            ),
            post_create_audit_server_date=(
                "Sun, 26 Jul 2026 12:11:07 GMT"
            ),
            now=CLAIM_AT + timedelta(seconds=2, microseconds=1),
        )


def test_mutant_offline_exact_receipt_is_treated_as_transport_authority(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    decision = _decision(chain, claim)
    decision_artifact = _decision_artifact(decision)
    offline = _receipt(claim, decision, decision_artifact)
    result = generation.validate_modeled_submit_once(
        generation_claim=_claim_artifact(claim),
        start_decision=decision_artifact,
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
        start_decision_receipt=offline,
        post_create_generation_inventory=_inventory(
            _claim_artifact(claim),
            decision_artifact,
        ),
        post_create_audit_server_date="Sun, 26 Jul 2026 12:11:22 GMT",
        now=DECIDED_AT + timedelta(seconds=2),
    )
    assert type(result) is generation.ModeledSubmitOnceValidation
    assert result.validation_result == "modeled-submit-once-valid"
    replay = generation.validate_modeled_submit_once(
        generation_claim=_claim_artifact(claim),
        start_decision=decision_artifact,
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
        start_decision_receipt=offline,
        post_create_generation_inventory=_inventory(
            _claim_artifact(claim),
            decision_artifact,
        ),
        post_create_audit_server_date="Sun, 26 Jul 2026 12:11:22 GMT",
        now=DECIDED_AT + timedelta(seconds=2),
    )
    assert replay == result
    assert replay is not result
    assert "non-authoritative" in (
        generation.validate_modeled_submit_once.__doc__ or ""
    ).lower()
    forbidden_methods = {
        "consume",
        "consume_receipt",
        "file_bytes",
        "file_sha256",
        "from_json",
        "serialize",
        "submit",
        "to_dict",
        "to_json",
    }
    for record_type in (
        generation.UnambiguousStartDecisionCreateReceipt,
        generation.ModeledSubmitOnceValidation,
    ):
        assert forbidden_methods.isdisjoint(vars(record_type))
    receipt_consumers: list[tuple[str, str]] = []
    validation_consumers: list[tuple[str, str]] = []
    for name in generation.__all__:
        value = getattr(generation, name)
        if not inspect.isfunction(value):
            continue
        hints = get_type_hints(value)
        for parameter, annotation in hints.items():
            if (
                annotation
                is generation.UnambiguousStartDecisionCreateReceipt
            ):
                receipt_consumers.append((name, parameter))
            if annotation is generation.ModeledSubmitOnceValidation:
                validation_consumers.append((name, parameter))
    assert receipt_consumers == [
        ("validate_modeled_submit_once", "start_decision_receipt")
    ]
    assert validation_consumers == [
        ("validate_modeled_submit_once", "return")
    ]
    assert not any(
        name.startswith(("consume_", "submit_", "serialize_"))
        for name in generation.__all__
    )


def test_mutant_post_create_inventory_missing_extra_or_historical_is_killed(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    decision = _decision(chain, claim)
    claim_artifact = _claim_artifact(claim)
    decision_artifact = _decision_artifact(decision)
    base = _inventory(claim_artifact, decision_artifact)
    kwargs = {
        "generation_claim": claim_artifact,
        "start_decision": decision_artifact,
        "descriptor": chain["descriptor"],
        "intent": chain["intent"],
        "approval": chain["approval"],
        "controller_baseline": chain["baseline"],
        "must_start_control_plane_ready": chain["ready"],
        "submission_acquisition": chain["acquisition"],
        "gpu_spend_snapshot": chain["snapshot"],
        "start_decision_receipt": _receipt(claim, decision, decision_artifact),
        "post_create_audit_server_date": "Sun, 26 Jul 2026 12:11:22 GMT",
        "now": DECIDED_AT + timedelta(seconds=2),
    }
    mutants = (
        base[:1],
        base[1:],
        base + [replace(base[0], version_id="historical", is_latest=False)],
        [replace(base[0], raw=base[0].raw + b" ")] + base[1:],
        [replace(base[0], is_delete_marker=True, raw=None)] + base[1:],
    )
    for mutant in mutants:
        mutant = sorted(
            mutant,
            key=lambda item: (item.key, item.version_id, item.is_delete_marker),
        )
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **kwargs,
                    "post_create_generation_inventory": mutant,
                }  # type: ignore[arg-type]
            )


def test_mutant_terminal_reuses_stale_snapshot_or_terminalizes_launch(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    launch = _decision(chain, claim)
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_terminal(
            generation_claim=claim_artifact,
            start_decision=_decision_artifact(launch),
            final_gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
            terminal_at=MUST_START_BY + timedelta(seconds=20),
        )
    expiry = _decision(
        chain,
        claim,
        decision="expire-unstarted",
        decided_at=MUST_START_BY,
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_terminal(
            generation_claim=claim_artifact,
            start_decision=_decision_artifact(expiry),
            final_gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
            terminal_at=MUST_START_BY + timedelta(seconds=20),
        )
    snapshot = _fresh_snapshot(
        chain,
        MUST_START_BY + timedelta(seconds=10),
        version_id="fresh-final",
    )
    terminal = generation.build_production_generation_terminal(
        generation_claim=claim_artifact,
        start_decision=_decision_artifact(expiry),
        final_gpu_spend_snapshot=snapshot,
        terminal_at=MUST_START_BY + timedelta(seconds=70),
    )
    assert terminal["terminal_outcome"] == "expired-unstarted"
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_terminal(
            generation_claim=claim_artifact,
            start_decision=_decision_artifact(expiry),
            final_gpu_spend_snapshot=snapshot,
            terminal_at=MUST_START_BY + timedelta(seconds=70, microseconds=1),
        )


def test_mutant_v1_terminal_accepts_job_drain_or_future_schema(
    chain: dict[str, object],
) -> None:
    claim, claim_artifact, decision, decision_artifact, snapshot, terminal, _ = (
        _expiry_chain(chain)
    )
    assert generation.validate_production_generation_terminal(
        terminal,
        generation_claim=claim_artifact,
        start_decision=decision_artifact,
        final_gpu_spend_snapshot=snapshot,
        now=MUST_START_BY + timedelta(seconds=20),
    ) == terminal
    for field, value in (
        ("terminal_outcome", "job-drained"),
        ("job_id", 7),
        ("binding", {}),
        ("drained_marker", None),
        ("schema_version", 2),
    ):
        changed = copy.deepcopy(terminal)
        changed[field] = value
        changed = _rehash(changed, "generation_terminal_body_sha256")
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.production_generation_terminal_file_bytes(changed)
    assert claim["generation"] == decision["generation"] == 1


def test_mutant_resolver_returns_illegal_pair_or_submit_authority(
    chain: dict[str, object],
) -> None:
    initial = generation.decide_production_generation_action(
        generation_inventory=[],
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
        submit_attempt_id=ATTEMPT_A,
        now=CLAIM_AT,
    )
    assert initial == generation.ProductionGenerationDecision(
        action="create-generation-claim",
        reason="empty-inventory-ready-for-generation-one",
        generation=1,
    )
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    claim_only = _inventory(claim_artifact)
    common = {
        "descriptor": chain["descriptor"],
        "intent": chain["intent"],
        "approval": chain["approval"],
        "controller_baseline": chain["baseline"],
        "must_start_control_plane_ready": chain["ready"],
        "submission_acquisition": chain["acquisition"],
        "gpu_spend_snapshot": chain["snapshot"],
        "submit_attempt_id": ATTEMPT_A,
    }
    before = generation.decide_production_generation_action(
        generation_inventory=claim_only,
        now=DECIDED_AT,
        **common,  # type: ignore[arg-type]
    )
    assert before == generation.ProductionGenerationDecision(
        action="create-launch-once-decision",
        reason="current-claim-ready-for-launch-decision",
        generation=1,
    )
    at = generation.decide_production_generation_action(
        generation_inventory=claim_only,
        now=MUST_START_BY,
        **common,  # type: ignore[arg-type]
    )
    assert at == generation.ProductionGenerationDecision(
        action="create-expire-unstarted-decision",
        reason="current-claim-deadline-reached",
        generation=1,
    )
    mismatch = generation.decide_production_generation_action(
        generation_inventory=claim_only,
        now=DECIDED_AT,
        **{**common, "submit_attempt_id": ATTEMPT_B},  # type: ignore[arg-type]
    )
    assert mismatch == generation.ProductionGenerationDecision(
        action="fail-closed",
        reason="invalid-or-inconsistent-generation-authority",
        generation=None,
    )
    launch = _decision(chain, claim)
    launch_inventory = _inventory(claim_artifact, _decision_artifact(launch))
    stored = generation.decide_production_generation_action(
        generation_inventory=launch_inventory,
        now=DECIDED_AT + timedelta(seconds=1),
        **common,  # type: ignore[arg-type]
    )
    assert stored == generation.ProductionGenerationDecision(
        action="reconcile-only",
        reason="stored-or-visible-generation-requires-reconciliation",
        generation=1,
    )
    assert "post_outcome" not in inspect.signature(
        generation.decide_production_generation_action
    ).parameters


def test_mutant_resolver_raises_on_malformed_inventory_or_skips_terminal_action(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        claim,
        claim_artifact,
        expiry,
        expiry_artifact,
        final_snapshot,
        _,
        _,
    ) = _expiry_chain(chain)
    common = {
        "descriptor": chain["descriptor"],
        "intent": chain["intent"],
        "approval": chain["approval"],
        "controller_baseline": chain["baseline"],
        "must_start_control_plane_ready": chain["ready"],
        "submission_acquisition": chain["acquisition"],
        "gpu_spend_snapshot": final_snapshot,
        "submit_attempt_id": ATTEMPT_A,
        "now": MUST_START_BY + timedelta(seconds=20),
    }
    resolution = generation.decide_production_generation_action(
        generation_inventory=_inventory(
            claim_artifact,
            expiry_artifact,
        ),
        **common,  # type: ignore[arg-type]
    )
    assert resolution == generation.ProductionGenerationDecision(
        action="create-expired-unstarted-terminal",
        reason="current-expiry-decision-ready-for-terminal",
        generation=1,
    )
    malformed_inventories = (
        [_entry(expiry_artifact)],
        _inventory(claim_artifact) + [_entry(claim_artifact)],
        [replace(_entry(claim_artifact), is_latest=False)],
        [replace(_entry(claim_artifact), raw=None)],
    )
    expected_fail_closed = generation.ProductionGenerationDecision(
        action="fail-closed",
        reason="invalid-or-inconsistent-generation-authority",
        generation=None,
    )
    for inventory in malformed_inventories:
        assert generation.decide_production_generation_action(
            generation_inventory=inventory,
            **common,  # type: ignore[arg-type]
        ) == expected_fail_closed
    assert generation.decide_production_generation_action(
        generation_inventory=_inventory(claim_artifact, expiry_artifact),
        **{
            **common,
            "gpu_spend_snapshot": chain["snapshot"],
        },  # type: ignore[arg-type]
    ) == expected_fail_closed
    assert claim["submit_attempt_id"] == expiry["submit_attempt_id"]

    def memory_failure(
        entries: object,
        *,
        run_id: str,
    ) -> dict[str, object]:
        del entries, run_id
        raise MemoryError("must not become fail-closed")

    monkeypatch.setattr(generation, "_validate_inventory", memory_failure)
    with pytest.raises(MemoryError):
        generation.decide_production_generation_action(
            generation_inventory=[],
            descriptor=chain["descriptor"],  # type: ignore[arg-type]
            intent=chain["intent"],  # type: ignore[arg-type]
            approval=chain["approval"],  # type: ignore[arg-type]
            controller_baseline=chain["baseline"],  # type: ignore[arg-type]
            must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
            submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
            gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
            submit_attempt_id=ATTEMPT_A,
            now=CLAIM_AT,
        )


def test_mutant_expiry_terminal_and_n_plus_one_budget_continuity_is_killed(
    chain: dict[str, object],
) -> None:
    (
        claim,
        claim_artifact,
        expiry,
        expiry_artifact,
        final_snapshot,
        terminal,
        terminal_artifact,
    ) = _expiry_chain(chain)
    inventory = _inventory(claim_artifact, expiry_artifact, terminal_artifact)
    normalized = generation.validate_production_generation_inventory(
        inventory,
        run_id=RUN_ID,
    )
    assert normalized["open_generation"] is None
    assert normalized["highest_generation"] == 1
    next_chain, next_claim_at = _next_generation_chain(chain)
    resolver = generation.decide_production_generation_action(
        generation_inventory=inventory,
        descriptor=next_chain["descriptor"],  # type: ignore[arg-type]
        intent=next_chain["intent"],  # type: ignore[arg-type]
        approval=next_chain["approval"],  # type: ignore[arg-type]
        controller_baseline=next_chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=next_chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=next_chain["acquisition"],  # type: ignore[arg-type]
        gpu_spend_snapshot=next_chain["snapshot"],  # type: ignore[arg-type]
        submit_attempt_id=ATTEMPT_B,
        now=next_claim_at,
    )
    assert resolver == generation.ProductionGenerationDecision(
        action="advance-generation",
        reason="prior-expired-generation-ready-for-advance",
        generation=2,
    )
    second = generation.build_production_generation_claim(
        **_claim_kwargs(
            next_chain,
            generation_number=2,
            inventory=inventory,
            previous_terminal=terminal_artifact,
            attempt=ATTEMPT_B,
            claim_at=next_claim_at,
        )  # type: ignore[arg-type]
    )
    assert second["previous_generation"] == 1
    assert second["previous_generation_terminal_body_sha256"] == terminal[
        "generation_terminal_body_sha256"
    ]
    changed = copy.deepcopy(terminal)
    changed["final_remaining_gpu_seconds"] = int(
        changed["final_remaining_gpu_seconds"]
    ) - 1
    changed = _rehash(changed, "generation_terminal_body_sha256")
    changed_artifact = replace(terminal_artifact, raw=_file_bytes(changed))
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_claim(
            **_claim_kwargs(
                next_chain,
                generation_number=2,
                inventory=inventory,
                previous_terminal=changed_artifact,
                attempt=ATTEMPT_B,
                claim_at=next_claim_at,
            )  # type: ignore[arg-type]
        )
    assert claim["open_allocation_count"] == terminal["final_open_allocation_count"] == 0
    assert final_snapshot.version_id != chain["snapshot"].version_id  # type: ignore[union-attr]
    assert expiry["decision"] == "expire-unstarted"


def test_mutant_n_plus_one_reuses_snapshot_drifts_approval_or_ignores_every_budget_edge(
    chain: dict[str, object],
) -> None:
    (
        _,
        claim_artifact,
        _,
        expiry_artifact,
        prior_final_snapshot,
        terminal,
        terminal_artifact,
    ) = _expiry_chain(chain)
    inventory = _inventory(
        claim_artifact,
        expiry_artifact,
        terminal_artifact,
    )
    next_chain, next_claim_at = _next_generation_chain(chain)
    valid_kwargs = _claim_kwargs(
        next_chain,
        generation_number=2,
        inventory=inventory,
        previous_terminal=terminal_artifact,
        attempt=ATTEMPT_B,
        claim_at=next_claim_at,
    )
    second = generation.build_production_generation_claim(
        **valid_kwargs  # type: ignore[arg-type]
    )
    assert second["generation"] == 2
    assert second["gpu_spend_snapshot_key"] != terminal[
        "final_gpu_spend_snapshot_key"
    ]
    assert second["descriptor_key"] != json.loads(
        claim_artifact.raw
    )["descriptor_key"]
    assert second["approval_key"] == json.loads(
        claim_artifact.raw
    )["approval_key"]

    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_claim(
            **{
                **valid_kwargs,
                "previous_generation_terminal": None,
            }  # type: ignore[arg-type]
        )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_claim(
            **{
                **valid_kwargs,
                "generation": 3,
            }  # type: ignore[arg-type]
        )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_claim(
            **{
                **valid_kwargs,
                "gpu_spend_snapshot": prior_final_snapshot,
            }  # type: ignore[arg-type]
        )
    noncurrent_inventory = [
        (
            replace(entry, is_latest=False)
            if entry.key == terminal_artifact.key
            else entry
        )
        for entry in inventory
    ]
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_claim(
            **{
                **valid_kwargs,
                "generation_inventory": noncurrent_inventory,
            }  # type: ignore[arg-type]
        )

    approval_drift_chain = _coherent_approval_chain(
        next_chain,
        field="slack_permalink",
        value=(
            "https://example.slack.com/archives/C01234567/"
            "p1722013140000200"
        ),
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_claim(
            **_claim_kwargs(
                approval_drift_chain,
                generation_number=2,
                inventory=inventory,
                previous_terminal=terminal_artifact,
                attempt=ATTEMPT_B,
                claim_at=next_claim_at,
            )  # type: ignore[arg-type]
        )

    continuity_fields = (
        "final_gpu_spend_ledger_genesis_sha256",
        "final_gpu_spend_ledger_record_count",
        "final_gpu_spend_ledger_tip_record_sha256",
        "final_ec2_allocation_history_sha256",
        "final_approved_gpu_runtime_seconds",
        "final_approved_gpu_cost_usd",
        "final_hourly_cost_usd",
        "final_consumed_gpu_seconds",
        "final_consumed_gpu_cost_usd",
        "final_remaining_gpu_seconds",
        "final_remaining_gpu_cost_usd",
        "final_open_allocation_count",
    )
    for field in continuity_fields:
        changed = copy.deepcopy(terminal)
        current = changed[field]
        if type(current) is str:
            changed[field] = (
                ("0" if current[0] != "0" else "1") + current[1:]
            )
        elif type(current) is int:
            changed[field] = current + 1
        else:
            assert type(current) is float
            changed[field] = current + 0.01
        changed = _rehash(changed, "generation_terminal_body_sha256")
        changed_artifact = replace(
            terminal_artifact,
            raw=_file_bytes(changed),
        )
        changed_inventory = _inventory(
            claim_artifact,
            expiry_artifact,
            changed_artifact,
        )
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.build_production_generation_claim(
                **_claim_kwargs(
                    next_chain,
                    generation_number=2,
                    inventory=changed_inventory,
                    previous_terminal=changed_artifact,
                    attempt=ATTEMPT_B,
                    claim_at=next_claim_at,
                )  # type: ignore[arg-type]
            )


def test_mutant_pure_module_imports_io_launch_or_claims_live_fence() -> None:
    path = Path(generation.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_import_roots = {
        "boto3",
        "botocore",
        "skypilot",
        "sky",
        "requests",
        "urllib",
        "socket",
        "subprocess",
        "mlx",
        "torch",
        "cuda",
    }
    imported: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert imported.isdisjoint(forbidden_import_roots)
    assert {
        "open",
        "read_bytes",
        "read_text",
        "write_bytes",
        "write_text",
        "mkdir",
        "exists",
        "stat",
        "glob",
        "unlink",
        "getenv",
        "listdir",
        "remove",
        "system",
        "popen",
        "putenv",
        "urandom",
        "uuid4",
        "now",
        "time",
        "monotonic",
        "perf_counter",
        "random",
        "getrandbits",
        "randbytes",
        "token_bytes",
        "token_hex",
        "token_urlsafe",
        "socket",
        "create_connection",
        "getaddrinfo",
        "urlopen",
        "Request",
        "HTTPConnection",
        "HTTPSConnection",
        "run",
        "Popen",
        "put_object",
        "list_object_versions",
        "submit",
    }.isdisjoint(calls)
    assert "ExpectedBucketOwner" not in generation.__all__


def test_mutant_python39_import_or_typing_contract_drift_is_killed(tmp_path: Path) -> None:
    source = Path(generation.__file__)
    completed = subprocess.run(
        ["/usr/bin/python3", "-m", "py_compile", str(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    import_code = """
import importlib.util
import json
import pathlib
import sys
import types
from dataclasses import dataclass

source = pathlib.Path(sys.argv[1])
mode = sys.argv[2]

def module(name):
    value = types.ModuleType(name)
    sys.modules[name] = value
    return value

def passthrough(value, *args, **kwargs):
    del args, kwargs
    return dict(value)

@dataclass(frozen=True)
class VersionedJsonArtifact:
    key: str
    raw: bytes
    version_id: str

campaign = module(
    "glm52_sky_campaign"
    if mode == "flat"
    else "mlx_vq.quality.glm52_sky_campaign"
)
campaign.validate_gpu_spend_approval = passthrough
campaign.validate_sky_campaign_descriptor = passthrough

snapshot = module(
    "glm52_gpu_spend_snapshot"
    if mode == "flat"
    else "mlx_vq.quality.glm52_gpu_spend_snapshot"
)
snapshot.validate_gpu_spend_snapshot = passthrough

submission = module(
    "glm52_sky_production_submission"
    if mode == "flat"
    else "mlx_vq.quality.glm52_sky_production_submission"
)
submission.VersionedJsonArtifact = VersionedJsonArtifact
submission.production_submission_intent_file_bytes = lambda value: b""
submission.production_submission_intent_file_sha256 = lambda value: "0" * 64
submission.production_submission_intent_s3_key = lambda **kwargs: ""
submission.validate_production_submission_intent = passthrough

acquisition = module(
    "glm52_sky_production_acquisition"
    if mode == "flat"
    else "mlx_vq.quality.glm52_sky_production_acquisition"
)
for name in (
    "production_controller_baseline_file_bytes",
    "production_controller_baseline_file_sha256",
    "production_controller_baseline_s3_key",
    "production_must_start_control_plane_ready_file_bytes",
    "production_must_start_control_plane_ready_file_sha256",
    "production_must_start_control_plane_ready_s3_key",
    "production_submission_acquired_file_bytes",
    "production_submission_acquired_file_sha256",
    "production_submission_acquired_s3_key",
):
    setattr(acquisition, name, lambda *args, **kwargs: "")
for name in (
    "validate_production_controller_baseline",
    "validate_production_must_start_control_plane_ready",
    "validate_production_submission_acquired",
):
    setattr(acquisition, name, passthrough)

modes = module(
    "glm52_sky_submission_modes"
    if mode == "flat"
    else "mlx_vq.quality.glm52_sky_submission_modes"
)
class SubmissionModeContractError(ValueError):
    pass
modes.SubmissionModeContractError = SubmissionModeContractError
modes.expected_sky_job_name = lambda **kwargs: ""
modes.record_contract = lambda **kwargs: None
modes.require_opaque_version_id = lambda value, **kwargs: value

if mode == "package":
    mlx_vq = module("mlx_vq")
    mlx_vq.__path__ = []
    quality = module("mlx_vq.quality")
    quality.__path__ = []
    target_name = "mlx_vq.quality.glm52_sky_production_generation"
else:
    target_name = "glm52_sky_production_generation"

spec = importlib.util.spec_from_file_location(target_name, source)
target = importlib.util.module_from_spec(spec)
sys.modules[target_name] = target
spec.loader.exec_module(target)
assert len(target.__all__) == 25
print(json.dumps({
    "mode": mode,
    "module": target.__name__,
    "python": list(sys.version_info[:2]),
}))
"""
    for import_mode in ("flat", "package"):
        imported = subprocess.run(
            [
                "/usr/bin/python3",
                "-I",
                "-c",
                import_code,
                str(source),
                import_mode,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert imported.returncode == 0, imported.stderr
        result = json.loads(imported.stdout)
        assert result["mode"] == import_mode
        assert tuple(result["python"]) == (3, 9)

    original_import = generation.importlib.import_module

    def nested_missing(name: str) -> object:
        del name
        error = ModuleNotFoundError("nested dependency is absent")
        error.name = "nested_dependency"
        raise error

    generation.importlib.import_module = nested_missing
    try:
        with pytest.raises(ModuleNotFoundError) as raised:
            generation._flat_module("requested_flat_module")
        assert raised.value.name == "nested_dependency"
    finally:
        generation.importlib.import_module = original_import

    hints = get_type_hints(generation.ProductionGenerationDecision)
    assert set(hints) == {"action", "reason", "generation"}
    assert len(fields(generation.ModeledSubmitOnceValidation)) == 11


def test_mutant_claim_decision_or_terminal_roundtrip_uses_process_identity_or_is_nondeterministic(
    chain: dict[str, object],
) -> None:
    (
        claim,
        claim_artifact,
        decision,
        decision_artifact,
        snapshot,
        terminal,
        _,
    ) = _expiry_chain(chain)
    parsed_claim = json.loads(
        generation.production_generation_claim_file_bytes(claim)
    )
    parsed_decision = json.loads(
        generation.production_generation_start_decision_file_bytes(decision)
    )
    parsed_terminal = json.loads(
        generation.production_generation_terminal_file_bytes(terminal)
    )
    assert parsed_claim == claim and parsed_claim is not claim
    assert parsed_decision == decision and parsed_decision is not decision
    assert parsed_terminal == terminal and parsed_terminal is not terminal

    assert generation.validate_production_generation_claim(
        parsed_claim,
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
        generation_inventory=[],
        previous_generation_terminal=None,
    ) == claim
    assert generation.validate_production_generation_start_decision(
        parsed_decision,
        generation_claim=claim_artifact,
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        now=MUST_START_BY,
    ) == decision
    assert generation.validate_production_generation_terminal(
        parsed_terminal,
        generation_claim=claim_artifact,
        start_decision=decision_artifact,
        final_gpu_spend_snapshot=snapshot,
        now=MUST_START_BY + timedelta(seconds=20),
    ) == terminal

    rebuilt_claim = _claim(chain)
    rebuilt_decision = generation.build_production_generation_start_decision(
        generation_claim=claim_artifact,
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        decision="expire-unstarted",
        decided_at=MUST_START_BY,
    )
    rebuilt_terminal = generation.build_production_generation_terminal(
        generation_claim=claim_artifact,
        start_decision=decision_artifact,
        final_gpu_spend_snapshot=snapshot,
        terminal_at=MUST_START_BY + timedelta(seconds=20),
    )
    assert rebuilt_claim == claim and rebuilt_claim is not claim
    assert rebuilt_decision == decision and rebuilt_decision is not decision
    assert rebuilt_terminal == terminal and rebuilt_terminal is not terminal
    assert (
        generation.production_generation_claim_file_bytes(rebuilt_claim)
        == claim_artifact.raw
    )
    assert generation.production_generation_claim_file_sha256(
        rebuilt_claim
    ) == _sha(claim_artifact.raw)
    assert _file_bytes(rebuilt_decision) == decision_artifact.raw
    assert _file_bytes(rebuilt_terminal) == _file_bytes(terminal)
    assert (
        generation.production_generation_start_decision_file_bytes(
            parsed_decision
        )
        == decision_artifact.raw
    )
    assert (
        generation.production_generation_terminal_file_bytes(parsed_terminal)
        == generation.production_generation_terminal_file_bytes(terminal)
    )


def test_mutant_recovered_claim_requires_creator_process_memory_before_resolver_selection(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)

    def pack(artifact: object) -> dict[str, str]:
        assert type(artifact) is VersionedJsonArtifact
        return {
            "key": artifact.key,
            "raw": base64.b64encode(artifact.raw).decode("ascii"),
            "version_id": artifact.version_id,
        }

    payload = {
        "claim": pack(claim_artifact),
        "descriptor": pack(chain["descriptor"]),
        "intent": pack(chain["intent"]),
        "approval": pack(chain["approval"]),
        "baseline": pack(chain["baseline"]),
        "ready": pack(chain["ready"]),
        "acquisition": pack(chain["acquisition"]),
        "snapshot": pack(chain["snapshot"]),
        "attempt": ATTEMPT_A,
        "now": DECIDED_AT.isoformat().replace("+00:00", "Z"),
    }
    code = """
import base64
import json
import sys
from mlx_vq.quality.glm52_sky_production_generation import (
    GenerationInventoryEntry,
    decide_production_generation_action,
)
from mlx_vq.quality.glm52_sky_production_submission import VersionedJsonArtifact

payload = json.load(sys.stdin)
def artifact(name):
    item = payload[name]
    return VersionedJsonArtifact(
        key=item["key"],
        raw=base64.b64decode(item["raw"]),
        version_id=item["version_id"],
    )
claim = artifact("claim")
decision = decide_production_generation_action(
    generation_inventory=[
        GenerationInventoryEntry(
            key=claim.key,
            raw=claim.raw,
            version_id=claim.version_id,
            is_latest=True,
            is_delete_marker=False,
        )
    ],
    descriptor=artifact("descriptor"),
    intent=artifact("intent"),
    approval=artifact("approval"),
    controller_baseline=artifact("baseline"),
    must_start_control_plane_ready=artifact("ready"),
    submission_acquisition=artifact("acquisition"),
    gpu_spend_snapshot=artifact("snapshot"),
    submit_attempt_id=payload["attempt"],
    now=payload["now"],
)
print(json.dumps({
    "action": decision.action,
    "reason": decision.reason,
    "generation": decision.generation,
}))
"""
    recovered = subprocess.run(
        [sys.executable, "-I", "-c", code],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        cwd=ROOT,
    )
    assert recovered.returncode == 0, recovered.stderr
    assert json.loads(recovered.stdout) == {
        "action": "create-launch-once-decision",
        "reason": "current-claim-ready-for-launch-decision",
        "generation": 1,
    }


def test_mutant_every_direct_source_accepts_alias_key_swapped_version_or_coherent_rehash(
    chain: dict[str, object],
) -> None:
    source_parameters = {
        "descriptor": "descriptor",
        "intent": "intent",
        "approval": "approval",
        "baseline": "controller_baseline",
        "ready": "must_start_control_plane_ready",
        "acquisition": "submission_acquisition",
        "snapshot": "gpu_spend_snapshot",
    }
    digest_fields = {
        "descriptor": "descriptor_body_sha256",
        "intent": "intent_body_sha256",
        "approval": "approval_body_sha256",
        "baseline": "baseline_body_sha256",
        "ready": "control_plane_ready_body_sha256",
        "acquisition": "acquisition_body_sha256",
        "snapshot": "snapshot_body_sha256",
    }
    source_versions = [
        chain[name].version_id  # type: ignore[union-attr]
        for name in source_parameters
    ]
    for index, (name, parameter) in enumerate(source_parameters.items()):
        source = chain[name]
        assert type(source) is VersionedJsonArtifact
        swapped = source_versions[(index + 1) % len(source_versions)]
        assert swapped != source.version_id
        record = json.loads(source.raw)
        record["record_type"] = f"{record['record_type']}-coherent-mutant"
        record = _rehash(record, digest_fields[name])
        mutants: tuple[VersionedJsonArtifact, ...] = (
            replace(source, key=f"{source.key}.alias"),
            replace(source, raw=_file_bytes(record)),
        )
        if name != "acquisition":
            mutants = (*mutants, replace(source, version_id=swapped))
        for mutant in mutants:
            kwargs = _claim_kwargs(chain)
            kwargs[parameter] = mutant
            with pytest.raises(generation.ProductionGenerationAuthorityError):
                generation.build_production_generation_claim(  # type: ignore[arg-type]
                    **kwargs
                )

        for bad_version in (
            None,
            "",
            " ",
            "has space",
            "line\nbreak",
            "nonascii-\u00e9",
            StringSubclass("visible"),
        ):
            kwargs = _claim_kwargs(chain)
            kwargs[parameter] = replace(
                source,
                version_id=bad_version,
            )
            with pytest.raises(generation.ProductionGenerationAuthorityError):
                generation.build_production_generation_claim(  # type: ignore[arg-type]
                    **kwargs
                )


def test_mutant_claim_validator_ignores_each_pinned_source_key_file_body_or_version_axis(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    prefixes = (
        "descriptor",
        "intent",
        "approval",
        "controller_baseline",
        "must_start_control_plane_ready",
        "submission_acquisition",
        "gpu_spend_snapshot",
    )
    for prefix in prefixes:
        for suffix in ("key", "file_sha256", "body_sha256", "version_id"):
            field = f"{prefix}_{suffix}"
            changed = copy.deepcopy(claim)
            original = changed[field]
            assert type(original) is str
            changed[field] = (
                f"{original}.alias"
                if suffix in {"key", "version_id"}
                else ("0" if original[0] != "0" else "1") + original[1:]
            )
            changed = _rehash(changed, "generation_claim_body_sha256")
            with pytest.raises(generation.ProductionGenerationAuthorityError):
                generation.validate_production_generation_claim(
                    changed,
                    descriptor=chain["descriptor"],  # type: ignore[arg-type]
                    intent=chain["intent"],  # type: ignore[arg-type]
                    approval=chain["approval"],  # type: ignore[arg-type]
                    controller_baseline=chain["baseline"],  # type: ignore[arg-type]
                    must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
                    submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
                    gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
                    generation_inventory=[],
                    previous_generation_terminal=None,
                )


def test_mutant_approval_alias_wrong_digest_or_descriptor_intent_disagreement_survives_six_boundaries(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval = chain["approval"]
    assert type(approval) is VersionedJsonArtifact
    approval_record = chain["records"]["approval"]  # type: ignore[index]
    aliases = (
        f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json",
        (
            f"campaigns/{RUN_ID}/authorities/"
            f"GPU_SPEND_APPROVAL-{'0' * 64}.json"
        ),
        (
            f"campaigns/{RUN_ID}/authorities/"
            f"GPU_SPEND_APPROVAL-{_sha(approval.raw)}.json.alias"
        ),
        (
            f"campaigns/{RUN_ID}/authorities/nested/"
            f"GPU_SPEND_APPROVAL-{_sha(approval.raw)}.json"
        ),
        (
            f"campaigns/{RUN_ID}/authorities/"
            f"gpu_spend_approval-{_sha(approval.raw)}.json"
        ),
    )

    calls: list[object] = []

    def validating(value: object) -> dict[str, object]:
        calls.append(value)
        return validate_gpu_spend_approval(value)  # type: ignore[arg-type]

    monkeypatch.setattr(generation, "validate_gpu_spend_approval", validating)
    base_claim = _claim(chain)
    base_decision = _decision(chain, base_claim)
    for alias in aliases:
        mutated = _coherent_approval_chain(
            chain,
            field="approval_response",
            value=approval_record["approval_response"],
            approval_key_override=alias,
        )
        claim = _claim_for_chain(base_claim, mutated)
        claim_artifact = _claim_artifact(
            claim,
            version_id="generation-claim-version-key-mutant",
        )
        decision = _decision_for_claim(base_decision, claim, claim_artifact)
        _assert_approval_mutant_rejected_at_six_boundaries(
            chain=mutated,
            claim=claim,
            decision=decision,
            public_validator_calls=calls,
        )

    disagreement = _coherent_approval_chain(
        chain,
        field="approval_response",
        value=approval_record["approval_response"],
        approval_key_override=approval.key,
        descriptor_approval_key_override=approval.key,
        intent_approval_key_override=aliases[0],
    )
    claim = _claim_for_chain(base_claim, disagreement)
    claim_artifact = _claim_artifact(
        claim,
        version_id="generation-claim-version-key-disagreement",
    )
    decision = _decision_for_claim(base_decision, claim, claim_artifact)
    _assert_approval_mutant_rejected_at_six_boundaries(
        chain=disagreement,
        claim=claim,
        decision=decision,
        public_validator_calls=calls,
        local_gate_must_precede_public_validator=None,
    )

    disagreement = _coherent_approval_chain(
        chain,
        field="approval_response",
        value=approval_record["approval_response"],
        approval_key_override=approval.key,
        descriptor_approval_key_override=aliases[0],
        intent_approval_key_override=approval.key,
    )
    claim = _claim_for_chain(base_claim, disagreement)
    claim_artifact = _claim_artifact(
        claim,
        version_id="generation-claim-version-descriptor-key-disagreement",
    )
    decision = _decision_for_claim(base_decision, claim, claim_artifact)
    _assert_approval_mutant_rejected_at_six_boundaries(
        chain=disagreement,
        claim=claim,
        decision=decision,
        public_validator_calls=calls,
        local_gate_must_precede_public_validator=None,
    )


def test_mutant_record_semantic_account_region_run_bucket_campaign_or_job_drift_is_ignored(
    chain: dict[str, object],
) -> None:
    (
        claim,
        claim_artifact,
        decision,
        decision_artifact,
        snapshot,
        terminal,
        _,
    ) = _expiry_chain(chain)
    semantic_mutants = {
        "account_id": "000000000000",
        "region": "us-east-1",
        "run_id": f"{RUN_ID}-foreign",
        "bucket": "foreign-bucket-123",
        "campaign_identity_sha256": "0" * 64,
        "sky_job_name": f"{RUN_ID}-other",
        "sky_job_identity_sha256": "1" * 64,
    }
    for field, mutant in semantic_mutants.items():
        changed_claim = _rehash(
            {**claim, field: mutant},
            "generation_claim_body_sha256",
        )
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_production_generation_claim(
                changed_claim,
                descriptor=chain["descriptor"],  # type: ignore[arg-type]
                intent=chain["intent"],  # type: ignore[arg-type]
                approval=chain["approval"],  # type: ignore[arg-type]
                controller_baseline=chain["baseline"],  # type: ignore[arg-type]
                must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
                submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
                gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
                generation_inventory=[],
                previous_generation_terminal=None,
            )

        changed_decision = _rehash(
            {**decision, field: mutant},
            "start_decision_body_sha256",
        )
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_production_generation_start_decision(
                changed_decision,
                generation_claim=claim_artifact,
                descriptor=chain["descriptor"],  # type: ignore[arg-type]
                intent=chain["intent"],  # type: ignore[arg-type]
                approval=chain["approval"],  # type: ignore[arg-type]
                controller_baseline=chain["baseline"],  # type: ignore[arg-type]
                must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
                submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
                now=MUST_START_BY,
            )

        changed_terminal = _rehash(
            {**terminal, field: mutant},
            "generation_terminal_body_sha256",
        )
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_production_generation_terminal(
                changed_terminal,
                generation_claim=claim_artifact,
                start_decision=decision_artifact,
                final_gpu_spend_snapshot=snapshot,
                now=MUST_START_BY + timedelta(seconds=20),
            )


@pytest.mark.parametrize(
    "snapshot_field",
    [
        "observed_at",
        "gpu_spend_ledger_tip_record_sha256",
    ],
)
def test_mutant_claim_accepts_snapshot_semantic_drift_from_intent(
    chain: dict[str, object],
    snapshot_field: str,
) -> None:
    snapshot_record = chain["records"]["snapshot"]  # type: ignore[index]
    if snapshot_field == "observed_at":
        original = datetime.strptime(
            str(snapshot_record[snapshot_field]),
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=timezone.utc)
        replacement: object = (
            original + timedelta(seconds=1)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        copied_intent_field = "spend_snapshot_observed_at"
    else:
        replacement = _different_digest(snapshot_record[snapshot_field])
        copied_intent_field = snapshot_field
    mutant_chain = _coherently_readdressed_snapshot_chain(
        chain,
        snapshot_updates={snapshot_field: replacement},
    )
    assert mutant_chain["records"]["snapshot"][snapshot_field] != (  # type: ignore[index]
        mutant_chain["records"]["intent"][copied_intent_field]  # type: ignore[index]
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        _claim(mutant_chain)


def _claim_with_spend_tip_drift(
    chain: dict[str, object],
) -> dict[str, object]:
    claim = _claim(chain)
    claim["gpu_spend_ledger_tip_record_sha256"] = _different_digest(
        claim["gpu_spend_ledger_tip_record_sha256"]
    )
    return _rehash(claim, "generation_claim_body_sha256")


def test_mutant_launch_decision_accepts_claim_spend_tip_drift(
    chain: dict[str, object],
) -> None:
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        _decision(chain, _claim_with_spend_tip_drift(chain))


def test_mutant_modeled_submit_accepts_claim_spend_tip_drift(
    chain: dict[str, object],
) -> None:
    original_claim = _claim(chain)
    mutant_claim = _claim_with_spend_tip_drift(chain)
    mutant_claim_artifact = _claim_artifact(
        mutant_claim,
        version_id="generation-claim-version-spend-tip-drift",
    )
    original_decision = _decision(chain, original_claim)
    mutant_decision = _decision_for_claim(
        original_decision,
        mutant_claim,
        mutant_claim_artifact,
    )
    mutant_decision_artifact = _decision_artifact(
        mutant_decision,
        version_id="generation-decision-version-spend-tip-drift",
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            generation_claim=mutant_claim_artifact,
            start_decision=mutant_decision_artifact,
            descriptor=chain["descriptor"],  # type: ignore[arg-type]
            intent=chain["intent"],  # type: ignore[arg-type]
            approval=chain["approval"],  # type: ignore[arg-type]
            controller_baseline=chain["baseline"],  # type: ignore[arg-type]
            must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
            submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
            gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
            start_decision_receipt=_receipt(
                mutant_claim,
                mutant_decision,
                mutant_decision_artifact,
            ),
            post_create_generation_inventory=_inventory(
                mutant_claim_artifact,
                mutant_decision_artifact,
            ),
            post_create_audit_server_date=(
                "Sun, 26 Jul 2026 12:11:22 GMT"
            ),
            now=DECIDED_AT + timedelta(seconds=2),
        )


def test_mutant_normalized_inventory_aliases_input_raw_bytes(
    chain: dict[str, object],
) -> None:
    claim_artifact = _claim_artifact(_claim(chain))
    normalized = generation.validate_production_generation_inventory(
        _inventory(claim_artifact),
        run_id=RUN_ID,
    )
    projection = normalized["generations"][0]["claim"]  # type: ignore[index]
    assert type(projection) is dict
    assert type(projection["raw"]) is bytes
    assert projection["raw"] == claim_artifact.raw
    assert projection["raw"] is not claim_artifact.raw


def _modeled_kwargs(
    chain: dict[str, object],
    *,
    claim: dict[str, object],
    claim_artifact: VersionedJsonArtifact,
    decision: dict[str, object],
    decision_artifact: VersionedJsonArtifact,
) -> dict[str, object]:
    return {
        "generation_claim": claim_artifact,
        "start_decision": decision_artifact,
        "descriptor": chain["descriptor"],
        "intent": chain["intent"],
        "approval": chain["approval"],
        "controller_baseline": chain["baseline"],
        "must_start_control_plane_ready": chain["ready"],
        "submission_acquisition": chain["acquisition"],
        "gpu_spend_snapshot": chain["snapshot"],
        "start_decision_receipt": _receipt(
            claim,
            decision,
            decision_artifact,
        ),
        "post_create_generation_inventory": _inventory(
            claim_artifact,
            decision_artifact,
        ),
        "post_create_audit_server_date": (
            "Sun, 26 Jul 2026 12:11:22 GMT"
        ),
        "now": DECIDED_AT + timedelta(seconds=2),
    }


def test_mutant_generation_one_accepts_nonempty_inventory_or_previous_terminal(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    assert claim["previous_generation"] == 0
    assert all(
        claim[field] is None
        for field in (
            "previous_generation_terminal_key",
            "previous_generation_terminal_file_sha256",
            "previous_generation_terminal_body_sha256",
            "previous_generation_terminal_version_id",
        )
    )
    claim_artifact = _claim_artifact(claim)
    (
        _,
        _,
        _,
        _,
        _,
        _,
        terminal_artifact,
    ) = _expiry_chain(chain)
    for kwargs in (
        {"generation_inventory": _inventory(claim_artifact)},
        {"previous_generation_terminal": terminal_artifact},
        {
            "generation": 2,
            "generation_inventory": [],
            "previous_generation_terminal": None,
        },
    ):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.build_production_generation_claim(
                **{
                    **_claim_kwargs(chain),
                    **kwargs,
                }  # type: ignore[arg-type]
            )


def test_mutant_inventory_normalization_drops_two_generation_projection_axes(
    chain: dict[str, object],
) -> None:
    (
        _,
        first_claim_artifact,
        _,
        first_decision_artifact,
        _,
        _,
        first_terminal_artifact,
    ) = _expiry_chain(chain)
    closed = _inventory(
        first_claim_artifact,
        first_decision_artifact,
        first_terminal_artifact,
    )
    next_chain, next_claim_at = _next_generation_chain(chain)
    second_claim = generation.build_production_generation_claim(
        **_claim_kwargs(
            next_chain,
            generation_number=2,
            inventory=closed,
            previous_terminal=first_terminal_artifact,
            attempt=ATTEMPT_B,
            claim_at=next_claim_at,
        )  # type: ignore[arg-type]
    )
    second_claim_artifact = _claim_artifact(
        second_claim,
        version_id="generation-claim-version-2-normalized",
    )
    second_decision = generation.build_production_generation_start_decision(
        generation_claim=second_claim_artifact,
        descriptor=next_chain["descriptor"],  # type: ignore[arg-type]
        intent=next_chain["intent"],  # type: ignore[arg-type]
        approval=next_chain["approval"],  # type: ignore[arg-type]
        controller_baseline=next_chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=next_chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=next_chain["acquisition"],  # type: ignore[arg-type]
        decision="launch-once",
        decided_at=next_claim_at + timedelta(seconds=1),
    )
    second_decision_artifact = _decision_artifact(
        second_decision,
        version_id="generation-decision-version-2-normalized",
    )
    artifacts = (
        first_claim_artifact,
        first_decision_artifact,
        first_terminal_artifact,
        second_claim_artifact,
        second_decision_artifact,
    )
    normalized = generation.validate_production_generation_inventory(
        _inventory(*artifacts),
        run_id=RUN_ID,
    )
    assert set(normalized) == {
        "record_type",
        "run_id",
        "generation_count",
        "highest_generation",
        "open_generation",
        "generations",
    }
    assert normalized["record_type"] == (
        "glm52_sky_production_generation_inventory_validation_v1"
    )
    assert normalized["generation_count"] == 2
    assert normalized["highest_generation"] == 2
    assert normalized["open_generation"] == 2
    rows = normalized["generations"]
    assert type(rows) is list
    assert [row["generation"] for row in rows] == [1, 2]
    assert [row["generation_text"] for row in rows] == [
        "00000001",
        "00000002",
    ]
    assert rows[0]["terminal"] is not None
    assert rows[1]["terminal"] is None
    for row in rows:
        assert set(row) == {
            "generation",
            "generation_text",
            "claim",
            "start_decision",
            "terminal",
        }
        for projection in (
            row["claim"],
            row["start_decision"],
            row["terminal"],
        ):
            if projection is not None:
                assert set(projection) == {
                    "record_type",
                    "key",
                    "raw",
                    "version_id",
                    "file_sha256",
                    "body_sha256",
                }
    projections = [
        rows[0]["claim"],
        rows[0]["start_decision"],
        rows[0]["terminal"],
        rows[1]["claim"],
        rows[1]["start_decision"],
    ]
    for projection, artifact in zip(projections, artifacts):
        assert projection["key"] == artifact.key
        assert projection["raw"] == artifact.raw
        assert projection["raw"] is not artifact.raw
        assert projection["version_id"] == artifact.version_id
        assert projection["file_sha256"] == _sha(artifact.raw)


@pytest.mark.parametrize(
    "version_id",
    [
        "",
        "null",
        " ",
        "has space",
        "\n",
        "\x7f",
        "caf\u00e9",
        StringSubclass("visible-version"),
    ],
)
def test_mutant_inventory_accepts_invalid_or_coercible_version_id(
    chain: dict[str, object],
    version_id: object,
) -> None:
    artifact = _claim_artifact(_claim(chain))
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_production_generation_inventory(
            [_entry(replace(artifact, version_id=version_id))],
            run_id=RUN_ID,
        )


def test_mutant_inventory_accepts_latest_plus_historical_version(
    chain: dict[str, object],
) -> None:
    artifact = _claim_artifact(_claim(chain))
    latest = _entry(artifact)
    historical = replace(
        latest,
        version_id="historical-version",
        is_latest=False,
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_production_generation_inventory(
            sorted(
                [latest, historical],
                key=lambda item: (
                    item.key,
                    item.version_id,
                    item.is_delete_marker,
                ),
            ),
            run_id=RUN_ID,
        )


def test_mutant_decision_or_terminal_copied_artifact_edges_are_ignored(
    chain: dict[str, object],
) -> None:
    (
        _,
        claim_artifact,
        decision,
        decision_artifact,
        snapshot,
        terminal,
        _,
    ) = _expiry_chain(chain)
    for field in (
        "generation_claim_key",
        "generation_claim_file_sha256",
        "generation_claim_body_sha256",
        "generation_claim_version_id",
    ):
        changed = copy.deepcopy(decision)
        original = changed[field]
        assert type(original) is str
        changed[field] = (
            f"{original}.alias"
            if field.endswith(("key", "version_id"))
            else _different_digest(original)
        )
        changed = _rehash(changed, "start_decision_body_sha256")
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_production_generation_start_decision(
                changed,
                generation_claim=claim_artifact,
                descriptor=chain["descriptor"],  # type: ignore[arg-type]
                intent=chain["intent"],  # type: ignore[arg-type]
                approval=chain["approval"],  # type: ignore[arg-type]
                controller_baseline=chain["baseline"],  # type: ignore[arg-type]
                must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
                submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
                now=MUST_START_BY,
            )

    for field in (
        "generation_claim_key",
        "generation_claim_file_sha256",
        "generation_claim_body_sha256",
        "generation_claim_version_id",
        "start_decision_key",
        "start_decision_file_sha256",
        "start_decision_body_sha256",
        "start_decision_version_id",
        "final_gpu_spend_snapshot_key",
        "final_gpu_spend_snapshot_file_sha256",
        "final_gpu_spend_snapshot_body_sha256",
        "final_gpu_spend_snapshot_version_id",
    ):
        changed = copy.deepcopy(terminal)
        original = changed[field]
        assert type(original) is str
        changed[field] = (
            f"{original}.alias"
            if field.endswith(("key", "version_id"))
            else _different_digest(original)
        )
        changed = _rehash(changed, "generation_terminal_body_sha256")
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_production_generation_terminal(
                changed,
                generation_claim=claim_artifact,
                start_decision=decision_artifact,
                final_gpu_spend_snapshot=snapshot,
                now=MUST_START_BY + timedelta(seconds=20),
            )


def test_mutant_direct_claim_decision_or_final_snapshot_artifact_drift_is_ignored(
    chain: dict[str, object],
) -> None:
    launch_claim = _claim(chain)
    launch_claim_artifact = _claim_artifact(launch_claim)
    launch_decision = _decision(chain, launch_claim)
    launch_decision_artifact = _decision_artifact(launch_decision)
    modeled = _modeled_kwargs(
        chain,
        claim=launch_claim,
        claim_artifact=launch_claim_artifact,
        decision=launch_decision,
        decision_artifact=launch_decision_artifact,
    )
    for mutant in (
        replace(launch_claim_artifact, key=f"{launch_claim_artifact.key}.alias"),
        replace(launch_claim_artifact, raw=launch_claim_artifact.raw + b" "),
        replace(launch_claim_artifact, version_id="claim-version-drift"),
    ):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_production_generation_start_decision(
                launch_decision,
                generation_claim=mutant,
                descriptor=chain["descriptor"],  # type: ignore[arg-type]
                intent=chain["intent"],  # type: ignore[arg-type]
                approval=chain["approval"],  # type: ignore[arg-type]
                controller_baseline=chain["baseline"],  # type: ignore[arg-type]
                must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
                submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
                now=DECIDED_AT,
            )
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **modeled,
                    "generation_claim": mutant,
                }  # type: ignore[arg-type]
            )

    for mutant in (
        replace(
            launch_decision_artifact,
            key=f"{launch_decision_artifact.key}.alias",
        ),
        replace(
            launch_decision_artifact,
            raw=launch_decision_artifact.raw + b" ",
        ),
        replace(
            launch_decision_artifact,
            version_id="decision-version-drift",
        ),
    ):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **modeled,
                    "start_decision": mutant,
                }  # type: ignore[arg-type]
            )

    (
        _,
        expiry_claim_artifact,
        _,
        expiry_decision_artifact,
        final_snapshot,
        terminal,
        _,
    ) = _expiry_chain(chain)
    for mutant in (
        replace(final_snapshot, key=f"{final_snapshot.key}.alias"),
        replace(final_snapshot, raw=final_snapshot.raw + b" "),
        replace(final_snapshot, version_id="final-snapshot-version-drift"),
    ):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_production_generation_terminal(
                terminal,
                generation_claim=expiry_claim_artifact,
                start_decision=expiry_decision_artifact,
                final_gpu_spend_snapshot=mutant,
                now=MUST_START_BY + timedelta(seconds=20),
            )


def test_mutant_terminal_launch_guard_depends_on_stale_snapshot_failure(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    fresh_snapshot = _fresh_snapshot(
        chain,
        MUST_START_BY + timedelta(seconds=10),
        version_id="fresh-launch-terminal-rejection",
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_terminal(
            generation_claim=_claim_artifact(claim),
            start_decision=_decision_artifact(_decision(chain, claim)),
            final_gpu_spend_snapshot=fresh_snapshot,
            terminal_at=MUST_START_BY + timedelta(seconds=20),
        )


def test_mutant_terminal_now_before_terminal_or_snapshot_validator_bypass(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        claim_artifact,
        _,
        decision_artifact,
        snapshot,
        terminal,
        _,
    ) = _expiry_chain(chain)
    terminal_at = MUST_START_BY + timedelta(seconds=20)
    assert generation.validate_production_generation_terminal(
        terminal,
        generation_claim=claim_artifact,
        start_decision=decision_artifact,
        final_gpu_spend_snapshot=snapshot,
        now=terminal_at,
    ) == terminal
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_production_generation_terminal(
            terminal,
            generation_claim=claim_artifact,
            start_decision=decision_artifact,
            final_gpu_spend_snapshot=snapshot,
            now=terminal_at - timedelta(microseconds=1),
        )

    calls: list[object] = []
    original = generation.validate_gpu_spend_snapshot

    def validating(value: object) -> dict[str, object]:
        calls.append(value)
        return original(value)  # type: ignore[arg-type]

    monkeypatch.setattr(generation, "validate_gpu_spend_snapshot", validating)
    generation.validate_production_generation_terminal(
        terminal,
        generation_claim=claim_artifact,
        start_decision=decision_artifact,
        final_gpu_spend_snapshot=snapshot,
        now=terminal_at,
    )
    assert calls


def test_mutant_stored_launch_attempt_mismatch_or_delete_marker_is_reconciled(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    decision_artifact = _decision_artifact(_decision(chain, claim))
    common = {
        "descriptor": chain["descriptor"],
        "intent": chain["intent"],
        "approval": chain["approval"],
        "controller_baseline": chain["baseline"],
        "must_start_control_plane_ready": chain["ready"],
        "submission_acquisition": chain["acquisition"],
        "gpu_spend_snapshot": chain["snapshot"],
        "now": DECIDED_AT + timedelta(seconds=1),
    }
    failed = generation.ProductionGenerationDecision(
        action="fail-closed",
        reason="invalid-or-inconsistent-generation-authority",
        generation=None,
    )
    assert generation.decide_production_generation_action(
        generation_inventory=_inventory(
            claim_artifact,
            decision_artifact,
        ),
        submit_attempt_id=ATTEMPT_B,
        **common,  # type: ignore[arg-type]
    ) == failed
    for artifact in (claim_artifact, decision_artifact):
        marker = generation.GenerationInventoryEntry(
            key=artifact.key,
            raw=None,
            version_id="delete-marker-version",
            is_latest=True,
            is_delete_marker=True,
        )
        inventory = _inventory(
            claim_artifact,
            decision_artifact,
        ) + [marker]
        inventory.sort(
            key=lambda item: (
                item.key,
                item.version_id,
                item.is_delete_marker,
            )
        )
        assert generation.decide_production_generation_action(
            generation_inventory=inventory,
            submit_attempt_id=ATTEMPT_A,
            **common,  # type: ignore[arg-type]
        ) == failed


def test_mutant_expiry_reauthenticates_stale_acquisition_as_current(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    stale_expiry_at = MUST_START_BY + timedelta(minutes=10)
    later_validation_at = stale_expiry_at + timedelta(hours=1)
    expiry = generation.build_production_generation_start_decision(
        generation_claim=_claim_artifact(claim),
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        decision="expire-unstarted",
        decided_at=stale_expiry_at,
    )
    assert generation.validate_production_generation_start_decision(
        expiry,
        generation_claim=_claim_artifact(claim),
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        now=later_validation_at,
    ) == expiry
    assert generation.decide_production_generation_action(
        generation_inventory=_inventory(_claim_artifact(claim)),
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
        submit_attempt_id=ATTEMPT_A,
        now=stale_expiry_at,
    ) == generation.ProductionGenerationDecision(
        action="create-expire-unstarted-decision",
        reason="current-claim-deadline-reached",
        generation=1,
    )


def test_mutant_runtime_performs_filesystem_environment_clock_random_network_or_process_io(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise AssertionError("pure H.1e attempted forbidden runtime I/O")

    with monkeypatch.context() as guard:
        for owner, name in (
            (builtins, "open"),
            (Path, "open"),
            (Path, "read_bytes"),
            (Path, "read_text"),
            (Path, "write_text"),
            (Path, "write_bytes"),
            (Path, "mkdir"),
            (Path, "exists"),
            (Path, "stat"),
            (Path, "glob"),
            (Path, "unlink"),
            (os, "getenv"),
            (os, "open"),
            (os, "listdir"),
            (os, "mkdir"),
            (os, "remove"),
            (os, "system"),
            (os, "popen"),
            (os, "putenv"),
            (time, "time"),
            (time, "monotonic"),
            (time, "perf_counter"),
            (random, "random"),
            (random, "getrandbits"),
            (random, "randbytes"),
            (secrets, "token_bytes"),
            (secrets, "token_hex"),
            (secrets, "token_urlsafe"),
            (socket, "socket"),
            (socket, "create_connection"),
            (socket, "getaddrinfo"),
            (urllib.request, "urlopen"),
            (urllib.request, "Request"),
            (http.client, "HTTPConnection"),
            (http.client, "HTTPSConnection"),
            (subprocess, "run"),
            (subprocess, "Popen"),
        ):
            guard.setattr(owner, name, forbidden)

        claim = _claim(chain)
        claim_artifact = _claim_artifact(claim)
        assert generation.production_generation_claim_file_bytes(claim) == (
            claim_artifact.raw
        )
        assert generation.production_generation_claim_file_sha256(claim) == (
            hashlib.sha256(claim_artifact.raw).hexdigest()
        )
        assert generation.production_generation_claim_s3_key(
            run_id=RUN_ID,
            generation=1,
        ) == claim_artifact.key
        assert generation.validate_production_generation_claim(
            claim,
            descriptor=chain["descriptor"],  # type: ignore[arg-type]
            intent=chain["intent"],  # type: ignore[arg-type]
            approval=chain["approval"],  # type: ignore[arg-type]
            controller_baseline=chain["baseline"],  # type: ignore[arg-type]
            must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
            submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
            gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
            generation_inventory=[],
            previous_generation_terminal=None,
        ) == claim
        decision = _decision(chain, claim)
        decision_artifact = _decision_artifact(decision)
        assert generation.production_generation_start_decision_file_bytes(
            decision
        ) == decision_artifact.raw
        assert generation.production_generation_start_decision_file_sha256(
            decision
        ) == hashlib.sha256(decision_artifact.raw).hexdigest()
        assert generation.production_generation_start_decision_s3_key(
            run_id=RUN_ID,
            generation=1,
        ) == decision_artifact.key
        (
            expiry_claim,
            expiry_claim_artifact,
            expiry,
            expiry_artifact,
            final_snapshot,
            terminal,
            terminal_artifact,
        ) = _expiry_chain(chain)
        assert generation.production_generation_terminal_file_bytes(
            terminal
        ) == terminal_artifact.raw
        assert generation.production_generation_terminal_file_sha256(
            terminal
        ) == hashlib.sha256(terminal_artifact.raw).hexdigest()
        assert generation.production_generation_terminal_s3_key(
            run_id=RUN_ID,
            generation=1,
        ) == terminal_artifact.key
        assert generation.validate_production_generation_terminal(
            terminal,
            generation_claim=expiry_claim_artifact,
            start_decision=expiry_artifact,
            final_gpu_spend_snapshot=final_snapshot,
            now=MUST_START_BY + timedelta(seconds=20),
        ) == terminal
        assert expiry_claim["generation"] == 1
        assert expiry["decision"] == "expire-unstarted"
        assert generation.validate_production_generation_inventory(
            _inventory(claim_artifact, decision_artifact),
            run_id=RUN_ID,
        )["open_generation"] == 1
        assert generation.validate_modeled_submit_once(
            **_modeled_kwargs(
                chain,
                claim=claim,
                claim_artifact=claim_artifact,
                decision=decision,
                decision_artifact=decision_artifact,
            )  # type: ignore[arg-type]
        ).validation_result == "modeled-submit-once-valid"
        assert generation.decide_production_generation_action(
            generation_inventory=_inventory(
                claim_artifact,
                decision_artifact,
            ),
            descriptor=chain["descriptor"],  # type: ignore[arg-type]
            intent=chain["intent"],  # type: ignore[arg-type]
            approval=chain["approval"],  # type: ignore[arg-type]
            controller_baseline=chain["baseline"],  # type: ignore[arg-type]
            must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
            submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
            gpu_spend_snapshot=chain["snapshot"],  # type: ignore[arg-type]
            submit_attempt_id=ATTEMPT_A,
            now=DECIDED_AT + timedelta(seconds=1),
        ).action == "reconcile-only"


def test_mutant_modeled_authority_revalidates_at_put_dominant_conservative_instant(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    decision = _decision(chain, claim)
    decision_artifact = _decision_artifact(decision)
    receipt = _receipt(
        claim,
        decision,
        decision_artifact,
        server_date="Sun, 26 Jul 2026 12:11:24 GMT",
    )
    authority_times: list[object] = []
    decision_validation_times: list[object] = []
    original_h1d = generation._authenticate_h1d
    original_decision_validator = (
        generation.validate_production_generation_start_decision
    )

    def authenticate_h1d(**kwargs: object) -> dict[str, object]:
        authority_times.append(kwargs["now"])
        return original_h1d(**kwargs)  # type: ignore[arg-type]

    def validate_decision(
        value: Mapping[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        decision_validation_times.append(kwargs["now"])
        return original_decision_validator(
            value,
            **kwargs,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(generation, "_authenticate_h1d", authenticate_h1d)
    monkeypatch.setattr(
        generation,
        "validate_production_generation_start_decision",
        validate_decision,
    )
    result = generation.validate_modeled_submit_once(
        **{
            **_modeled_kwargs(
                chain,
                claim=claim,
                claim_artifact=claim_artifact,
                decision=decision,
                decision_artifact=decision_artifact,
            ),
            "start_decision_receipt": receipt,
        }  # type: ignore[arg-type]
    )
    assert result.validated_at == "2026-07-26T12:11:22Z"
    assert result.conservative_validation_at == "2026-07-26T12:11:29Z"
    assert result.conservative_validation_at not in {
        result.validated_at,
        result.must_start_by,
    }
    assert authority_times[0] == "2026-07-26T12:11:29Z"
    assert decision_validation_times == ["2026-07-26T12:11:29Z"]


def test_mutant_ceiling_shrinks_fractional_request_or_audit_interval(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    decision = _decision(chain, claim, decided_at=CLAIM_AT)
    claim_artifact = _claim_artifact(claim)
    decision_artifact = _decision_artifact(decision)
    request = CLAIM_AT + timedelta(microseconds=900_000)
    response = CLAIM_AT + timedelta(seconds=5, microseconds=900_001)
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            **{
                **_modeled_kwargs(
                    chain,
                    claim=claim,
                    claim_artifact=claim_artifact,
                    decision=decision,
                    decision_artifact=decision_artifact,
                ),
                "start_decision_receipt": _receipt(
                    claim,
                    decision,
                    decision_artifact,
                    request_started_at=request,
                    response_received_at=response,
                    server_date="Sun, 26 Jul 2026 12:11:11 GMT",
                ),
                "post_create_audit_server_date": (
                    "Sun, 26 Jul 2026 12:11:16 GMT"
                ),
                "now": response,
            }  # type: ignore[arg-type]
        )

    fresh_chain, claim_at = _next_generation_chain(
        chain,
        must_start_offset_seconds=120,
    )
    fresh_claim = _claim(fresh_chain, claim_created_at=claim_at)
    fresh_claim_artifact = _claim_artifact(
        fresh_claim,
        version_id="fractional-audit-claim",
    )
    fresh_decision = _decision(
        fresh_chain,
        fresh_claim,
        decided_at=claim_at,
    )
    fresh_decision = _decision_for_claim(
        fresh_decision,
        fresh_claim,
        fresh_claim_artifact,
    )
    fresh_decision_artifact = _decision_artifact(
        fresh_decision,
        version_id="fractional-audit-decision",
    )
    audit_response = claim_at + timedelta(seconds=1, microseconds=900_000)
    audit_now = claim_at + timedelta(seconds=16, microseconds=900_001)
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            generation_claim=fresh_claim_artifact,
            start_decision=fresh_decision_artifact,
            descriptor=fresh_chain["descriptor"],  # type: ignore[arg-type]
            intent=fresh_chain["intent"],  # type: ignore[arg-type]
            approval=fresh_chain["approval"],  # type: ignore[arg-type]
            controller_baseline=fresh_chain["baseline"],  # type: ignore[arg-type]
            must_start_control_plane_ready=fresh_chain["ready"],  # type: ignore[arg-type]
            submission_acquisition=fresh_chain["acquisition"],  # type: ignore[arg-type]
            gpu_spend_snapshot=fresh_chain["snapshot"],  # type: ignore[arg-type]
            start_decision_receipt=_receipt(
                fresh_claim,
                fresh_decision,
                fresh_decision_artifact,
                request_started_at=claim_at,
                response_received_at=audit_response,
                server_date=claim_at.strftime(
                    "%a, %d %b %Y %H:%M:%S GMT"
                ),
            ),
            post_create_generation_inventory=_inventory(
                fresh_claim_artifact,
                fresh_decision_artifact,
            ),
            post_create_audit_server_date=audit_now.replace(
                microsecond=0
            ).strftime("%a, %d %b %Y %H:%M:%S GMT"),
            now=audit_now,
        )


def test_mutant_put_or_head_imf_alias_metadata_shape_and_receipt_chronology_are_accepted(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    decision = _decision(chain, claim)
    claim_artifact = _claim_artifact(claim)
    decision_artifact = _decision_artifact(decision)
    receipt = _receipt(claim, decision, decision_artifact)
    kwargs = _modeled_kwargs(
        chain,
        claim=claim,
        claim_artifact=claim_artifact,
        decision=decision,
        decision_artifact=decision_artifact,
    )
    imf_mutants = (
        "Mon, 26 Jul 2026 12:11:21 GMT",
        "sun, 26 Jul 2026 12:11:21 GMT",
        "Sun, 26 jul 2026 12:11:21 GMT",
        "Sun, 26 Jul 2026 12:11:21 UTC",
        "Sun, 26 Jul 2026 12:11:21 +0000",
        "Sun, 26 Jul 2026 12:11:21.0 GMT",
        " Sun, 26 Jul 2026 12:11:21 GMT",
        "Sun,  26 Jul 2026 12:11:21 GMT",
        "Sun, 26 Jul 2026 12:11:21 GMT ",
    )
    for server_date in imf_mutants:
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **kwargs,
                    "start_decision_receipt": replace(
                        receipt,
                        server_date=server_date,
                    ),
                }  # type: ignore[arg-type]
            )
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **kwargs,
                    "post_create_audit_server_date": server_date.replace(
                        "21",
                        "22",
                    ),
                }  # type: ignore[arg-type]
            )

    for metadata in (
        receipt.immutable_metadata[:-1],
        (*receipt.immutable_metadata, ("extra", "value")),
    ):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **kwargs,
                    "start_decision_receipt": replace(
                        receipt,
                        immutable_metadata=metadata,
                    ),
                }  # type: ignore[arg-type]
            )

    for changed_receipt, now in (
        (
            replace(
                receipt,
                request_started_at=DECIDED_AT + timedelta(seconds=2),
                response_received_at=DECIDED_AT + timedelta(seconds=1),
            ),
            DECIDED_AT + timedelta(seconds=2),
        ),
        (
            replace(
                receipt,
                response_received_at=DECIDED_AT + timedelta(seconds=3),
            ),
            DECIDED_AT + timedelta(seconds=2),
        ),
    ):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **kwargs,
                    "start_decision_receipt": changed_receipt,
                    "now": now,
                }  # type: ignore[arg-type]
            )

    for outcome, source in (
        ("prior-attempt", "direct-response"),
        ("timeout", "direct-response"),
        ("ambiguous", "direct-response"),
        ("created", "prior-attempt-readback"),
    ):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **kwargs,
                    "start_decision_receipt": replace(
                        receipt,
                        outcome=outcome,
                        source=source,
                    ),
                }  # type: ignore[arg-type]
            )

    class DuckReceipt:
        pass

    duck = DuckReceipt()
    duck.__dict__.update(receipt.__dict__)
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            **{
                **kwargs,
                "start_decision_receipt": duck,
            }  # type: ignore[arg-type]
        )


def test_mutant_each_immutable_metadata_key_or_value_and_conflict_status_is_accepted(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    decision = _decision(chain, claim)
    claim_artifact = _claim_artifact(claim)
    decision_artifact = _decision_artifact(decision)
    receipt = _receipt(claim, decision, decision_artifact)
    kwargs = _modeled_kwargs(
        chain,
        claim=claim,
        claim_artifact=claim_artifact,
        decision=decision,
        decision_artifact=decision_artifact,
    )
    assert len(receipt.immutable_metadata) == 8
    for index, (key, value) in enumerate(receipt.immutable_metadata):
        for changed_pair in (
            (f"{key}-tampered", value),
            (key, f"{value}-tampered"),
        ):
            changed_metadata = list(receipt.immutable_metadata)
            changed_metadata[index] = changed_pair
            with pytest.raises(generation.ProductionGenerationAuthorityError):
                generation.validate_modeled_submit_once(
                    **{
                        **kwargs,
                        "start_decision_receipt": replace(
                            receipt,
                            immutable_metadata=tuple(changed_metadata),
                        ),
                    }  # type: ignore[arg-type]
                )
    for http_status in (409, 412):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **kwargs,
                    "start_decision_receipt": replace(
                        receipt,
                        http_status=http_status,
                    ),
                }  # type: ignore[arg-type]
            )


def test_mutant_modeled_chronology_rejects_all_equal_inclusive_instants(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    decision = _decision(chain, claim)
    claim_artifact = _claim_artifact(claim)
    decision_artifact = _decision_artifact(decision)
    equal_time = DECIDED_AT
    result = generation.validate_modeled_submit_once(
        **{
            **_modeled_kwargs(
                chain,
                claim=claim,
                claim_artifact=claim_artifact,
                decision=decision,
                decision_artifact=decision_artifact,
            ),
            "start_decision_receipt": _receipt(
                claim,
                decision,
                decision_artifact,
                request_started_at=equal_time,
                response_received_at=equal_time,
                server_date="Sun, 26 Jul 2026 12:11:20 GMT",
            ),
            "post_create_audit_server_date": (
                "Sun, 26 Jul 2026 12:11:20 GMT"
            ),
            "now": equal_time,
        }  # type: ignore[arg-type]
    )
    assert result.validated_at == "2026-07-26T12:11:20Z"
    assert result.conservative_validation_at == "2026-07-26T12:11:25Z"


def test_mutant_fractional_modeled_output_is_floored_or_max_uses_wrong_operands(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    decision = _decision(chain, claim)
    claim_artifact = _claim_artifact(claim)
    decision_artifact = _decision_artifact(decision)
    result = generation.validate_modeled_submit_once(
        **{
            **_modeled_kwargs(
                chain,
                claim=claim,
                claim_artifact=claim_artifact,
                decision=decision,
                decision_artifact=decision_artifact,
            ),
            "start_decision_receipt": _receipt(
                claim,
                decision,
                decision_artifact,
                server_date="Sun, 26 Jul 2026 12:11:16 GMT",
            ),
            "post_create_audit_server_date": (
                "Sun, 26 Jul 2026 12:11:18 GMT"
            ),
            "now": DECIDED_AT + timedelta(seconds=2, microseconds=1),
        }  # type: ignore[arg-type]
    )
    assert result.validated_at == "2026-07-26T12:11:23Z"
    assert result.conservative_validation_at == "2026-07-26T12:11:23Z"

    tree = ast.parse(Path(generation.__file__).read_text(encoding="utf-8"))
    modeled = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_modeled_submit_once"
    )
    max_calls = [
        node
        for node in ast.walk(modeled)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "max"
    ]
    assert len(max_calls) == 1
    assert len(max_calls[0].args) == 3
    assert {ast.unparse(argument) for argument in max_calls[0].args} == {
        "_ceil_whole_second(now_exact)",
        "put_server + timedelta(seconds=5)",
        "audit_server + timedelta(seconds=5)",
    }


def test_mutant_conservative_server_skew_overflow_is_not_fail_closed(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    decision = _decision(chain, claim)
    claim_artifact = _claim_artifact(claim)
    decision_artifact = _decision_artifact(decision)
    last = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    receipt = _receipt(
        claim,
        decision,
        decision_artifact,
        request_started_at=last,
        response_received_at=last,
        server_date="Fri, 31 Dec 9999 23:59:59 GMT",
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            **{
                **_modeled_kwargs(
                    chain,
                    claim=claim,
                    claim_artifact=claim_artifact,
                    decision=decision,
                    decision_artifact=decision_artifact,
                ),
                "start_decision_receipt": receipt,
                "post_create_audit_server_date": (
                    "Fri, 31 Dec 9999 23:59:59 GMT"
                ),
                "now": last,
            }  # type: ignore[arg-type]
        )


def test_mutant_head_only_conservative_server_skew_overflow_is_not_fail_closed(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    decision = _decision(chain, claim)
    claim_artifact = _claim_artifact(claim)
    decision_artifact = _decision_artifact(decision)
    last = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    receipt = _receipt(
        claim,
        decision,
        decision_artifact,
        request_started_at=last,
        response_received_at=last,
        server_date="Fri, 31 Dec 9999 23:59:54 GMT",
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            **{
                **_modeled_kwargs(
                    chain,
                    claim=claim,
                    claim_artifact=claim_artifact,
                    decision=decision,
                    decision_artifact=decision_artifact,
                ),
                "start_decision_receipt": receipt,
                "post_create_audit_server_date": (
                    "Fri, 31 Dec 9999 23:59:59 GMT"
                ),
                "now": last,
            }  # type: ignore[arg-type]
        )


def test_mutant_terminal_ignores_each_snapshot_semantic_or_budget_axis(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        claim_artifact,
        _,
        decision_artifact,
        snapshot,
        _,
        _,
    ) = _expiry_chain(chain)
    record = json.loads(snapshot.raw)
    monkeypatch.setattr(
        generation,
        "validate_gpu_spend_snapshot",
        lambda value: dict(value),
    )
    fields_to_mutate = (
        "run_id",
        "campaign_identity_sha256",
        "descriptor_sha256",
        "descriptor_body_sha256",
        "approval_sha256",
        "approval_body_sha256",
        "gpu_spend_ledger_genesis_sha256",
        "gpu_spend_ledger_record_count",
        "gpu_spend_ledger_tip_record_sha256",
        "ec2_allocation_history_sha256",
        "approved_gpu_runtime_seconds",
        "approved_gpu_cost_usd",
        "hourly_cost_usd",
        "consumed_gpu_seconds",
        "consumed_gpu_cost_usd",
        "remaining_gpu_seconds",
        "remaining_gpu_cost_usd",
        "open_allocation_count",
    )
    for field in fields_to_mutate:
        changed = copy.deepcopy(record)
        original = changed[field]
        if type(original) is str:
            changed[field] = (
                f"{original}-foreign"
                if field == "run_id"
                else _different_digest(original)
            )
        elif type(original) is int:
            changed[field] = original + 1
        else:
            assert type(original) is float
            changed[field] = original + 0.01
        changed = _rehash(changed, "snapshot_body_sha256")
        artifact = _artifact(
            key=(
                f"campaigns/{RUN_ID}/spend-snapshots/"
                f"{changed['snapshot_body_sha256']}/"
                "GPU_SPEND_SNAPSHOT.json"
            ),
            value=changed,
            version_id=f"final-snapshot-semantic-{field}",
        )
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.build_production_generation_terminal(
                generation_claim=claim_artifact,
                start_decision=decision_artifact,
                final_gpu_spend_snapshot=artifact,
                terminal_at=MUST_START_BY + timedelta(seconds=20),
            )


def test_mutant_n_plus_one_accepts_foreign_non_v1_or_noninventory_predecessor(
    chain: dict[str, object],
) -> None:
    (
        _,
        claim_artifact,
        _,
        decision_artifact,
        _,
        terminal,
        terminal_artifact,
    ) = _expiry_chain(chain)
    inventory = _inventory(
        claim_artifact,
        decision_artifact,
        terminal_artifact,
    )
    next_chain, next_claim_at = _next_generation_chain(chain)
    base = _claim_kwargs(
        next_chain,
        generation_number=2,
        inventory=inventory,
        previous_terminal=terminal_artifact,
        attempt=ATTEMPT_B,
        claim_at=next_claim_at,
    )
    foreign = copy.deepcopy(terminal)
    foreign["run_id"] = f"{RUN_ID}-foreign"
    foreign = _rehash(foreign, "generation_terminal_body_sha256")
    non_v1 = copy.deepcopy(terminal)
    non_v1["schema_version"] = 2
    non_v1 = _rehash(non_v1, "generation_terminal_body_sha256")
    wrong_outcome = copy.deepcopy(terminal)
    wrong_outcome["terminal_outcome"] = "job-drained"
    wrong_outcome = _rehash(
        wrong_outcome,
        "generation_terminal_body_sha256",
    )
    predecessors = (
        replace(terminal_artifact, key=f"{terminal_artifact.key}.foreign"),
        replace(terminal_artifact, raw=_file_bytes(foreign)),
        replace(terminal_artifact, raw=_file_bytes(non_v1)),
        replace(terminal_artifact, raw=_file_bytes(wrong_outcome)),
        replace(
            terminal_artifact,
            version_id="terminal-version-not-in-inventory",
        ),
    )
    for predecessor in predecessors:
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.build_production_generation_claim(
                **{
                    **base,
                    "previous_generation_terminal": predecessor,
                }  # type: ignore[arg-type]
            )


def test_mutant_n_plus_one_ignores_terminal_to_new_snapshot_chronology(
    chain: dict[str, object],
) -> None:
    (
        _,
        claim_artifact,
        _,
        decision_artifact,
        _,
        _,
        terminal_artifact,
    ) = _expiry_chain(chain)
    inventory = _inventory(
        claim_artifact,
        decision_artifact,
        terminal_artifact,
    )
    next_chain, _ = _next_generation_chain(chain)
    terminal_at = MUST_START_BY + timedelta(seconds=20)
    for delta, accepted in (
        (timedelta(0), True),
        (-timedelta(microseconds=1), False),
    ):
        snapshot_at = terminal_at + delta
        chronological_chain = _coherently_readdressed_snapshot_chain(
            next_chain,
            snapshot_updates={
                "observed_at": snapshot_at.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            },
            copy_snapshot_semantics=True,
            intent_updates={
                "intent_at": (
                    snapshot_at + timedelta(seconds=30)
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
            },
        )
        acquired_at = datetime.strptime(
            str(
                chronological_chain["records"]["acquisition"][  # type: ignore[index]
                    "acquired_at"
                ]
            ),
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=timezone.utc)

        def operation() -> dict[str, object]:
            return generation.build_production_generation_claim(
                **_claim_kwargs(
                    chronological_chain,
                    generation_number=2,
                    inventory=inventory,
                    previous_terminal=terminal_artifact,
                    attempt=ATTEMPT_B,
                    claim_at=acquired_at,
                )  # type: ignore[arg-type]
            )

        if accepted:
            second = operation()
            assert second["gpu_spend_snapshot_key"] != json.loads(
                terminal_artifact.raw
            )["final_gpu_spend_snapshot_key"]
            assert second["previous_generation"] == 1
        else:
            with pytest.raises(
                generation.ProductionGenerationAuthorityError
            ):
                operation()


def test_mutant_inventory_ignores_adjacent_authority_budget_or_terminal_edge_drift(
    chain: dict[str, object],
) -> None:
    (
        _,
        first_claim_artifact,
        _,
        first_decision_artifact,
        _,
        _,
        first_terminal_artifact,
    ) = _expiry_chain(chain)
    closed = _inventory(
        first_claim_artifact,
        first_decision_artifact,
        first_terminal_artifact,
    )
    next_chain, next_claim_at = _next_generation_chain(chain)
    second = generation.build_production_generation_claim(
        **_claim_kwargs(
            next_chain,
            generation_number=2,
            inventory=closed,
            previous_terminal=first_terminal_artifact,
            attempt=ATTEMPT_B,
            claim_at=next_claim_at,
        )  # type: ignore[arg-type]
    )
    fields_to_mutate = (
        "previous_generation",
        "previous_generation_terminal_key",
        "previous_generation_terminal_file_sha256",
        "previous_generation_terminal_body_sha256",
        "previous_generation_terminal_version_id",
        "account_id",
        "region",
        "bucket",
        "run_id",
        "campaign_identity_sha256",
        "approval_key",
        "approval_file_sha256",
        "approval_body_sha256",
        "approval_version_id",
        "approved_gpu_runtime_seconds",
        "approved_gpu_cost_usd",
        "hourly_cost_usd",
        "gpu_spend_ledger_genesis_sha256",
        "gpu_spend_ledger_record_count",
        "gpu_spend_ledger_tip_record_sha256",
        "ec2_allocation_history_sha256",
        "consumed_gpu_seconds",
        "consumed_gpu_cost_usd",
        "remaining_gpu_seconds",
        "remaining_gpu_cost_usd",
        "open_allocation_count",
    )
    for field in fields_to_mutate:
        changed = copy.deepcopy(second)
        original = changed[field]
        if type(original) is str:
            changed[field] = (
                f"{original}.foreign"
                if field.endswith(("key", "version_id"))
                or field in {"region", "bucket", "run_id", "account_id"}
                else _different_digest(original)
            )
        elif type(original) is int:
            changed[field] = original + 1
        else:
            assert type(original) is float
            changed[field] = original + 0.01
        changed = _rehash(changed, "generation_claim_body_sha256")
        second_artifact = _claim_artifact(
            changed,
            version_id=f"generation-two-adjacent-{field}",
        )
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_production_generation_inventory(
                _inventory(*closed, second_artifact),
                run_id=RUN_ID,
            )


def test_mutant_h1e_claim_chronology_omits_readiness_or_accepts_inversion(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "validate_production_controller_baseline",
        "validate_production_must_start_control_plane_ready",
        "validate_production_submission_acquired",
    ):
        monkeypatch.setattr(
            generation,
            name,
            lambda value, **kwargs: dict(value),
        )
    for name in (
        "production_controller_baseline_file_bytes",
        "production_must_start_control_plane_ready_file_bytes",
        "production_submission_acquired_file_bytes",
    ):
        monkeypatch.setattr(
            generation,
            name,
            lambda value: _file_bytes(value),
        )
    for name in (
        "production_controller_baseline_file_sha256",
        "production_must_start_control_plane_ready_file_sha256",
        "production_submission_acquired_file_sha256",
    ):
        monkeypatch.setattr(
            generation,
            name,
            lambda value: _sha(_file_bytes(value)),
        )

    intent_record = chain["records"]["intent"]  # type: ignore[index]
    intent_at = datetime.strptime(
        str(intent_record["intent_at"]),
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    run_id = str(intent_record["run_id"])

    def chain_with_times(
        *,
        baseline_at: datetime,
        ready_at: datetime,
        acquired_at: datetime,
    ) -> dict[str, object]:
        baseline = copy.deepcopy(chain["records"]["baseline"])
        baseline["observed_at"] = baseline_at.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        baseline = _rehash(baseline, "baseline_body_sha256")
        baseline_artifact = _artifact(
            key=generation.production_controller_baseline_s3_key(
                run_id=run_id,
                baseline_body_sha256=str(
                    baseline["baseline_body_sha256"]
                ),
            ),
            value=baseline,
            version_id=f"chronology-baseline-{baseline_at.timestamp()}",
        )
        ready = copy.deepcopy(chain["records"]["ready"])
        ready["observed_at"] = ready_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        ready = _rehash(ready, "control_plane_ready_body_sha256")
        ready_artifact = _artifact(
            key=(
                generation.production_must_start_control_plane_ready_s3_key(
                    run_id=run_id,
                    intent_body_sha256=str(
                        intent_record["intent_body_sha256"]
                    ),
                    control_plane_ready_body_sha256=str(
                        ready["control_plane_ready_body_sha256"]
                    ),
                )
            ),
            value=ready,
            version_id=f"chronology-ready-{ready_at.timestamp()}",
        )
        acquisition = copy.deepcopy(chain["records"]["acquisition"])
        acquisition["acquired_at"] = acquired_at.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        acquisition = _rehash(acquisition, "acquisition_body_sha256")
        descriptor = chain["descriptor"]
        assert type(descriptor) is VersionedJsonArtifact
        acquisition_artifact = _artifact(
            key=production_submission_acquired_s3_key(
                run_id=run_id,
                descriptor_file_sha256=_sha(descriptor.raw),
            ),
            value=acquisition,
            version_id=f"chronology-acquisition-{acquired_at.timestamp()}",
        )
        return {
            **chain,
            "baseline": baseline_artifact,
            "ready": ready_artifact,
            "acquisition": acquisition_artifact,
            "records": {
                **chain["records"],  # type: ignore[dict-item]
                "baseline": baseline,
                "ready": ready,
                "acquisition": acquisition,
            },
        }

    equality_chain = chain_with_times(
        baseline_at=intent_at,
        ready_at=intent_at,
        acquired_at=intent_at,
    )
    equality_claim = _claim(
        equality_chain,
        claim_created_at=intent_at,
    )
    assert equality_claim["claim_created_at"] == intent_record["intent_at"]

    ordinary_baseline = datetime.strptime(
        str(chain["records"]["baseline"]["observed_at"]),  # type: ignore[index]
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    ordinary_ready = datetime.strptime(
        str(chain["records"]["ready"]["observed_at"]),  # type: ignore[index]
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    ordinary_acquired = datetime.strptime(
        str(chain["records"]["acquisition"]["acquired_at"]),  # type: ignore[index]
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=timezone.utc)
    chronology_mutants = (
        chain_with_times(
            baseline_at=intent_at - timedelta(seconds=1),
            ready_at=ordinary_ready,
            acquired_at=ordinary_acquired,
        ),
        chain_with_times(
            baseline_at=ordinary_ready + timedelta(seconds=1),
            ready_at=ordinary_ready,
            acquired_at=ordinary_acquired,
        ),
        chain_with_times(
            baseline_at=ordinary_baseline,
            ready_at=ordinary_acquired + timedelta(seconds=1),
            acquired_at=ordinary_acquired,
        ),
        chain_with_times(
            baseline_at=ordinary_baseline,
            ready_at=ordinary_ready,
            acquired_at=CLAIM_AT + timedelta(seconds=1),
        ),
    )
    for mutant in chronology_mutants:
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            _claim(mutant)


def test_mutant_launch_decision_skips_named_h1d_semantic_validators(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    names = (
        "validate_sky_campaign_descriptor",
        "validate_production_submission_intent",
        "validate_gpu_spend_approval",
        "validate_production_controller_baseline",
        "validate_production_must_start_control_plane_ready",
        "validate_production_submission_acquired",
    )
    calls: list[tuple[str, object]] = []
    for name in names:
        original = getattr(generation, name)

        def wrapping(
            *args: object,
            _name: str = name,
            _original: Any = original,
            **kwargs: object,
        ) -> object:
            calls.append((_name, kwargs.get("now")))
            return _original(*args, **kwargs)

        monkeypatch.setattr(generation, name, wrapping)

    decision = generation.build_production_generation_start_decision(
        generation_claim=claim_artifact,
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        decision="launch-once",
        decided_at=DECIDED_AT,
    )
    assert {name for name, _ in calls} == set(names)
    assert [
        now
        for name, now in calls
        if name == "validate_production_submission_acquired"
    ] == ["2026-07-26T12:11:20Z"]

    calls.clear()
    validation_now = DECIDED_AT + timedelta(seconds=1)
    assert generation.validate_production_generation_start_decision(
        decision,
        generation_claim=claim_artifact,
        descriptor=chain["descriptor"],  # type: ignore[arg-type]
        intent=chain["intent"],  # type: ignore[arg-type]
        approval=chain["approval"],  # type: ignore[arg-type]
        controller_baseline=chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=chain["acquisition"],  # type: ignore[arg-type]
        now=validation_now,
    ) == decision
    assert {name for name, _ in calls} == set(names)
    assert [
        now
        for name, now in calls
        if name == "validate_production_submission_acquired"
    ] == [
        "2026-07-26T12:11:20Z",
        "2026-07-26T12:11:21Z",
    ]


def test_mutant_initial_artifact_version_selection_is_rejected_instead_of_pinned(
    chain: dict[str, object],
) -> None:
    acquisition = chain["acquisition"]
    assert type(acquisition) is VersionedJsonArtifact
    selected_acquisition = replace(
        acquisition,
        version_id="new-visible-acquisition-version",
    )
    claim = generation.build_production_generation_claim(
        **{
            **_claim_kwargs(chain),
            "submission_acquisition": selected_acquisition,
        }  # type: ignore[arg-type]
    )
    assert claim["submission_acquisition_version_id"] == (
        selected_acquisition.version_id
    )

    claim_artifact = _claim_artifact(
        claim,
        version_id="new-visible-claim-version",
    )
    selected_chain = {
        **chain,
        "acquisition": selected_acquisition,
    }
    decision = generation.build_production_generation_start_decision(
        generation_claim=claim_artifact,
        descriptor=selected_chain["descriptor"],  # type: ignore[arg-type]
        intent=selected_chain["intent"],  # type: ignore[arg-type]
        approval=selected_chain["approval"],  # type: ignore[arg-type]
        controller_baseline=selected_chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=selected_chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=selected_chain["acquisition"],  # type: ignore[arg-type]
        decision="expire-unstarted",
        decided_at=MUST_START_BY,
    )
    assert decision["generation_claim_version_id"] == claim_artifact.version_id

    decision_artifact = _decision_artifact(
        decision,
        version_id="new-visible-decision-version",
    )
    final_snapshot = _fresh_snapshot(
        selected_chain,
        MUST_START_BY + timedelta(seconds=10),
        version_id="new-visible-final-snapshot-version",
    )
    terminal = generation.build_production_generation_terminal(
        generation_claim=claim_artifact,
        start_decision=decision_artifact,
        final_gpu_spend_snapshot=final_snapshot,
        terminal_at=MUST_START_BY + timedelta(seconds=20),
    )
    assert terminal["start_decision_version_id"] == (
        decision_artifact.version_id
    )
    assert terminal["final_gpu_spend_snapshot_version_id"] == (
        final_snapshot.version_id
    )


def test_mutant_post_create_inventory_substitutes_decision_raw_or_version(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    decision = _decision(chain, claim)
    decision_artifact = _decision_artifact(decision)
    kwargs = _modeled_kwargs(
        chain,
        claim=claim,
        claim_artifact=claim_artifact,
        decision=decision,
        decision_artifact=decision_artifact,
    )
    alternate = _decision(
        chain,
        claim,
        decided_at=DECIDED_AT + timedelta(seconds=1),
    )
    alternate_artifact = _decision_artifact(
        alternate,
        version_id=decision_artifact.version_id,
    )
    decision_entry = _entry(decision_artifact)
    mutants = (
        _inventory(claim_artifact, alternate_artifact),
        sorted(
            [
                _entry(claim_artifact),
                replace(
                    decision_entry,
                    version_id="other-visible-version",
                ),
            ],
            key=lambda item: (
                item.key,
                item.version_id,
                item.is_delete_marker,
            ),
        ),
        sorted(
            [
                _entry(claim_artifact),
                replace(
                    decision_entry,
                    version_id="historical-decision-version",
                    is_latest=False,
                ),
                decision_entry,
            ],
            key=lambda item: (
                item.key,
                item.version_id,
                item.is_delete_marker,
            ),
        ),
        sorted(
            [
                _entry(claim_artifact),
                decision_entry,
                generation.GenerationInventoryEntry(
                    key=decision_artifact.key,
                    raw=None,
                    version_id="decision-delete-marker",
                    is_latest=True,
                    is_delete_marker=True,
                ),
            ],
            key=lambda item: (
                item.key,
                item.version_id,
                item.is_delete_marker,
            ),
        ),
    )
    for inventory in mutants:
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **kwargs,
                    "post_create_generation_inventory": inventory,
                }  # type: ignore[arg-type]
            )


def test_mutant_modeled_accepts_claim_or_decision_artifact_subclass(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    decision = _decision(chain, claim)
    decision_artifact = _decision_artifact(decision)
    kwargs = _modeled_kwargs(
        chain,
        claim=claim,
        claim_artifact=claim_artifact,
        decision=decision,
        decision_artifact=decision_artifact,
    )
    for field, artifact in (
        ("generation_claim", claim_artifact),
        ("start_decision", decision_artifact),
    ):
        with pytest.raises(generation.ProductionGenerationAuthorityError):
            generation.validate_modeled_submit_once(
                **{
                    **kwargs,
                    field: ArtifactSubclass(
                        artifact.key,
                        artifact.raw,
                        artifact.version_id,
                    ),
                }  # type: ignore[arg-type]
            )


def _terminal_artifact_for(
    value: dict[str, object],
    *,
    version_id: str = "generation-terminal-version-round4",
) -> VersionedJsonArtifact:
    return _artifact(
        key=generation.production_generation_terminal_s3_key(
            run_id=str(value["run_id"]),
            generation=int(value["generation"]),
        ),
        value=value,
        version_id=version_id,
    )


def _build_decision_for_artifact(
    authority_chain: dict[str, object],
    claim_artifact: VersionedJsonArtifact,
    *,
    decision: str,
    decided_at: datetime,
) -> dict[str, object]:
    return generation.build_production_generation_start_decision(
        generation_claim=claim_artifact,
        descriptor=authority_chain["descriptor"],  # type: ignore[arg-type]
        intent=authority_chain["intent"],  # type: ignore[arg-type]
        approval=authority_chain["approval"],  # type: ignore[arg-type]
        controller_baseline=authority_chain["baseline"],  # type: ignore[arg-type]
        must_start_control_plane_ready=authority_chain["ready"],  # type: ignore[arg-type]
        submission_acquisition=authority_chain["acquisition"],  # type: ignore[arg-type]
        decision=decision,  # type: ignore[arg-type]
        decided_at=decided_at,
    )


def _generation_two_open_state(
    chain: dict[str, object],
) -> dict[str, object]:
    (
        claim_one,
        claim_one_artifact,
        decision_one,
        decision_one_artifact,
        _,
        terminal_one,
        terminal_one_artifact,
    ) = _expiry_chain(chain)
    closed_one = _inventory(
        claim_one_artifact,
        decision_one_artifact,
        terminal_one_artifact,
    )
    next_chain, next_claim_at = _next_generation_chain(chain)
    claim_two = generation.build_production_generation_claim(
        **_claim_kwargs(
            next_chain,
            generation_number=2,
            inventory=closed_one,
            previous_terminal=terminal_one_artifact,
            attempt=ATTEMPT_B,
            claim_at=next_claim_at,
        )  # type: ignore[arg-type]
    )
    claim_two_artifact = _claim_artifact(
        claim_two,
        version_id="generation-claim-version-2",
    )
    must_start_two = datetime.fromisoformat(
        str(claim_two["must_start_by"]).replace("Z", "+00:00")
    )
    return {
        "chain": next_chain,
        "closed_one": closed_one,
        "claim_one": claim_one,
        "decision_one": decision_one,
        "terminal_one": terminal_one,
        "terminal_one_artifact": terminal_one_artifact,
        "claim_two": claim_two,
        "claim_two_artifact": claim_two_artifact,
        "claim_two_at": next_claim_at,
        "must_start_two": must_start_two,
    }


def _resolver_kwargs(
    authority_chain: dict[str, object],
    *,
    inventory: list[generation.GenerationInventoryEntry],
    attempt: str,
    now: datetime,
    snapshot: VersionedJsonArtifact | None = None,
) -> dict[str, object]:
    return {
        "generation_inventory": inventory,
        "descriptor": authority_chain["descriptor"],
        "intent": authority_chain["intent"],
        "approval": authority_chain["approval"],
        "controller_baseline": authority_chain["baseline"],
        "must_start_control_plane_ready": authority_chain["ready"],
        "submission_acquisition": authority_chain["acquisition"],
        "gpu_spend_snapshot": (
            authority_chain["snapshot"] if snapshot is None else snapshot
        ),
        "submit_attempt_id": attempt,
        "now": now,
    }


@pytest.mark.parametrize(
    "source_name",
    [
        "descriptor",
        "intent",
        "approval",
        "baseline",
        "ready",
        "acquisition",
    ],
)
@pytest.mark.parametrize("component", ["key", "file", "body", "version"])
def test_mutant_terminal_resolver_ignores_isolated_claim_bound_coordinate(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    source_name: str,
    component: str,
) -> None:
    (
        claim,
        claim_artifact,
        _,
        decision_artifact,
        final_snapshot,
        _,
        _,
    ) = _expiry_chain(chain)
    authority = generation._authenticate_h1d(
        descriptor=chain["descriptor"],
        intent=chain["intent"],
        approval=chain["approval"],
        controller_baseline=chain["baseline"],
        must_start_control_plane_ready=chain["ready"],
        submission_acquisition=chain["acquisition"],
        now=claim["claim_created_at"],
    )
    artifact_field = f"{source_name}_artifact"
    artifact = authority[artifact_field]
    assert type(artifact) is VersionedJsonArtifact
    drifted_authority = dict(authority)
    body_fields = {
        "descriptor": "descriptor_body_sha256",
        "intent": "intent_body_sha256",
        "approval": "approval_body_sha256",
        "baseline": "baseline_body_sha256",
        "ready": "control_plane_ready_body_sha256",
        "acquisition": "acquisition_body_sha256",
    }
    if component == "key":
        drifted_authority[artifact_field] = replace(
            artifact,
            key=f"{artifact.key}.claim-edge-drift",
        )
    elif component == "file":
        file_field = f"{source_name}_file_sha256"
        drifted_authority[file_field] = _different_digest(
            authority[file_field]
        )
    elif component == "body":
        record = dict(authority[source_name])  # type: ignore[arg-type]
        body_field = body_fields[source_name]
        record[body_field] = _different_digest(record[body_field])
        drifted_authority[source_name] = record
    else:
        assert component == "version"
        drifted_authority[artifact_field] = replace(
            artifact,
            version_id=f"{artifact_field}-not-bound-to-claim",
        )
    h1d_calls: list[dict[str, object]] = []

    def isolated_h1d(**kwargs: object) -> dict[str, object]:
        h1d_calls.append(dict(kwargs))
        return drifted_authority

    monkeypatch.setattr(generation, "_authenticate_h1d", isolated_h1d)
    assert generation.decide_production_generation_action(
        **_resolver_kwargs(
            chain,
            inventory=_inventory(claim_artifact, decision_artifact),
            attempt=ATTEMPT_A,
            now=MUST_START_BY + timedelta(seconds=20),
            snapshot=final_snapshot,
        )  # type: ignore[arg-type]
    ) == generation.ProductionGenerationDecision(
        action="fail-closed",
        reason="invalid-or-inconsistent-generation-authority",
        generation=None,
    )
    assert len(h1d_calls) == 1


def test_mutant_inventory_accepts_decision_before_its_claim(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    decision = _decision(chain, claim)
    decision["decided_at"] = "2026-07-26T12:11:09Z"
    decision = _rehash(decision, "start_decision_body_sha256")
    decision_artifact = _decision_artifact(decision)
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_production_generation_inventory(
            _inventory(claim_artifact, decision_artifact),
            run_id=RUN_ID,
        )


def test_mutant_inventory_accepts_terminal_before_its_expiry_decision(
    chain: dict[str, object],
) -> None:
    (
        _,
        claim_artifact,
        decision,
        _,
        _,
        terminal,
        _,
    ) = _expiry_chain(chain)
    decision["decided_at"] = (
        MUST_START_BY + timedelta(seconds=30)
    ).isoformat().replace("+00:00", "Z")
    decision = _rehash(decision, "start_decision_body_sha256")
    decision_artifact = _decision_artifact(
        decision,
        version_id="generation-decision-version-late",
    )
    terminal.update(
        {
            "start_decision_file_sha256": _sha(decision_artifact.raw),
            "start_decision_body_sha256": decision[
                "start_decision_body_sha256"
            ],
            "start_decision_version_id": decision_artifact.version_id,
        }
    )
    terminal = _rehash(terminal, "generation_terminal_body_sha256")
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_production_generation_inventory(
            _inventory(
                claim_artifact,
                decision_artifact,
                _terminal_artifact_for(terminal),
            ),
            run_id=RUN_ID,
        )


@pytest.mark.parametrize(
    "terminal_updates",
    [
        {
            "final_gpu_spend_ledger_tip_record_sha256": "c3" * 32,
        },
        {
            "final_consumed_gpu_seconds": 1,
            "final_remaining_gpu_seconds": 86_399,
        },
        {
            "final_consumed_gpu_cost_usd": 0.01,
            "final_remaining_gpu_cost_usd": 1_320.95,
        },
    ],
)
def test_mutant_inventory_accepts_one_terminal_claim_continuity_drift(
    chain: dict[str, object],
    terminal_updates: dict[str, object],
) -> None:
    (
        _,
        claim_artifact,
        _,
        decision_artifact,
        _,
        terminal,
        _,
    ) = _expiry_chain(chain)
    terminal.update(terminal_updates)
    terminal = _rehash(terminal, "generation_terminal_body_sha256")
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_production_generation_inventory(
            _inventory(
                claim_artifact,
                decision_artifact,
                _terminal_artifact_for(terminal),
            ),
            run_id=RUN_ID,
        )


def test_mutant_public_result_class_adds_authority_property_or_callable() -> None:
    expected_bases = {
        generation.GenerationInventoryEntry: (object,),
        generation.UnambiguousStartDecisionCreateReceipt: (object,),
        generation.ModeledSubmitOnceValidation: (object,),
        generation.ProductionGenerationDecision: (object,),
        generation.ProductionGenerationAuthorityError: (ValueError,),
    }
    expected_public_names = {
        generation.GenerationInventoryEntry: set(),
        generation.UnambiguousStartDecisionCreateReceipt: set(),
        generation.ModeledSubmitOnceValidation: set(),
        generation.ProductionGenerationDecision: {"generation"},
        generation.ProductionGenerationAuthorityError: set(),
    }
    for public_class, expected in expected_public_names.items():
        assert public_class.__bases__ == expected_bases[public_class]
        assert {
            name
            for name in public_class.__dict__
            if not name.startswith("_")
        } == expected


@pytest.mark.parametrize(
    ("record_kind", "field"),
    [
        ("claim", "open_allocation_count"),
        ("terminal", "final_open_allocation_count"),
    ],
)
def test_mutant_standalone_helper_accepts_exact_integer_open_allocation_one(
    chain: dict[str, object],
    record_kind: str,
    field: str,
) -> None:
    if record_kind == "claim":
        record = _claim(chain)
        digest_field = "generation_claim_body_sha256"
        helper = generation.production_generation_claim_file_bytes
    else:
        record = _expiry_chain(chain)[5]
        digest_field = "generation_terminal_body_sha256"
        helper = generation.production_generation_terminal_file_bytes
    record[field] = 1
    record = _rehash(record, digest_field)
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        helper(record)


def test_mutant_inventory_accepts_exact_integer_open_allocation_one(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    claim["open_allocation_count"] = 1
    claim = _rehash(claim, "generation_claim_body_sha256")
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_production_generation_inventory(
            _inventory(_claim_artifact(claim)),
            run_id=RUN_ID,
        )


def test_mutant_timezone_memory_error_is_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fatal_zone = StatefulTimezone(failure=MemoryError("fatal-timezone"))
    value = datetime(2026, 7, 26, 12, 11, 10, tzinfo=fatal_zone)
    monkeypatch.setattr(generation, "timezone", StatefulTimezone)
    with pytest.raises(MemoryError, match="fatal-timezone"):
        generation._normalize_time_input(value, field="fatal_time")
    assert fatal_zone.calls == 1


@pytest.mark.parametrize("fatal_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_mutant_translate_catches_process_or_memory_base_exception(
    fatal_type: type[BaseException],
) -> None:
    fatal = fatal_type("fatal-translate")

    def operation() -> None:
        raise fatal

    with pytest.raises(fatal_type, match="fatal-translate") as raised:
        generation._translate(operation)
    assert raised.value is fatal


@pytest.mark.parametrize("fatal_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_mutant_resolver_catches_process_or_memory_base_exception(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    fatal_type: type[BaseException],
) -> None:
    fatal = fatal_type("fatal-resolver")

    def fail_inventory(
        entries: object,
        *,
        run_id: str,
    ) -> dict[str, object]:
        del entries, run_id
        raise fatal

    monkeypatch.setattr(generation, "_validate_inventory", fail_inventory)
    with pytest.raises(fatal_type, match="fatal-resolver") as raised:
        generation.decide_production_generation_action(
            **_resolver_kwargs(
                chain,
                inventory=[],
                attempt=ATTEMPT_A,
                now=CLAIM_AT,
            )  # type: ignore[arg-type]
        )
    assert raised.value is fatal


def test_mutant_stored_launch_after_deadline_or_stale_h1d_stops_reconciling(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    decision = _decision(chain, claim)
    decision_artifact = _decision_artifact(decision)
    expected = generation.ProductionGenerationDecision(
        action="reconcile-only",
        reason="stored-or-visible-generation-requires-reconciliation",
        generation=1,
    )
    for late_now in (
        MUST_START_BY,
        MUST_START_BY + timedelta(days=30),
    ):
        assert generation.decide_production_generation_action(
            **_resolver_kwargs(
                chain,
                inventory=_inventory(claim_artifact, decision_artifact),
                attempt=ATTEMPT_A,
                now=late_now,
            )  # type: ignore[arg-type]
        ) == expected


def test_mutant_put_behind_skew_is_masked_by_head_behind_skew(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    decision = _decision(chain, claim)
    decision_artifact = _decision_artifact(decision)
    kwargs = _modeled_kwargs(
        chain,
        claim=claim,
        claim_artifact=claim_artifact,
        decision=decision,
        decision_artifact=decision_artifact,
    )
    receipt = replace(
        kwargs["start_decision_receipt"],
        server_date="Sun, 26 Jul 2026 12:11:15 GMT",
        request_started_at=DECIDED_AT,
        response_received_at=DECIDED_AT + timedelta(seconds=1, microseconds=1),
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            **{
                **kwargs,
                "start_decision_receipt": receipt,
                "post_create_audit_server_date": (
                    "Sun, 26 Jul 2026 12:11:22 GMT"
                ),
                "now": DECIDED_AT + timedelta(seconds=2),
            }  # type: ignore[arg-type]
        )


def test_mutant_head_behind_skew_is_masked_by_put_behind_skew(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    decision = _decision(chain, claim)
    decision_artifact = _decision_artifact(decision)
    kwargs = _modeled_kwargs(
        chain,
        claim=claim,
        claim_artifact=claim_artifact,
        decision=decision,
        decision_artifact=decision_artifact,
    )
    receipt = replace(
        kwargs["start_decision_receipt"],
        server_date="Sun, 26 Jul 2026 12:11:16 GMT",
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_modeled_submit_once(
            **{
                **kwargs,
                "start_decision_receipt": receipt,
                "post_create_audit_server_date": (
                    "Sun, 26 Jul 2026 12:11:16 GMT"
                ),
                "now": DECIDED_AT + timedelta(seconds=1, microseconds=1),
            }  # type: ignore[arg-type]
        )


def test_mutant_terminal_rejects_inclusive_terminal_snapshot_equality(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    decision = _decision(
        chain,
        claim,
        decision="expire-unstarted",
        decided_at=MUST_START_BY,
    )
    decision_artifact = _decision_artifact(decision)
    snapshot_at_decision = _fresh_snapshot(
        chain,
        MUST_START_BY,
        version_id="snapshot-version-equal-decision",
    )
    terminal = generation.build_production_generation_terminal(
        generation_claim=claim_artifact,
        start_decision=decision_artifact,
        final_gpu_spend_snapshot=snapshot_at_decision,
        terminal_at=MUST_START_BY,
    )
    assert terminal["terminal_at"] == decision["decided_at"]

    snapshot_after_decision = _fresh_snapshot(
        chain,
        MUST_START_BY + timedelta(seconds=1),
        version_id="snapshot-version-equal-terminal",
    )
    terminal = generation.build_production_generation_terminal(
        generation_claim=claim_artifact,
        start_decision=decision_artifact,
        final_gpu_spend_snapshot=snapshot_after_decision,
        terminal_at=MUST_START_BY + timedelta(seconds=1),
    )
    assert terminal["terminal_at"] == json.loads(
        snapshot_after_decision.raw
    )["observed_at"]


def test_mutant_terminal_accepts_one_microsecond_before_final_snapshot(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    decision = _decision(
        chain,
        claim,
        decision="expire-unstarted",
        decided_at=MUST_START_BY,
    )
    decision_artifact = _decision_artifact(decision)
    snapshot_at = MUST_START_BY + timedelta(seconds=1)
    snapshot = _fresh_snapshot(
        chain,
        snapshot_at,
        version_id="snapshot-version-terminal-inversion",
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_terminal(
            generation_claim=claim_artifact,
            start_decision=decision_artifact,
            final_gpu_spend_snapshot=snapshot,
            terminal_at=snapshot_at - timedelta(microseconds=1),
        )


def test_mutant_terminal_accepts_snapshot_before_expiry_decision(
    chain: dict[str, object],
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    decision = _decision(
        chain,
        claim,
        decision="expire-unstarted",
        decided_at=MUST_START_BY + timedelta(seconds=1),
    )
    decision_artifact = _decision_artifact(decision)
    snapshot = _fresh_snapshot(
        chain,
        MUST_START_BY,
        version_id="snapshot-version-before-decision",
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_terminal(
            generation_claim=claim_artifact,
            start_decision=decision_artifact,
            final_gpu_spend_snapshot=snapshot,
            terminal_at=MUST_START_BY + timedelta(seconds=1),
        )


def test_mutant_resolver_current_generation_branches_are_hard_coded_to_one(
    chain: dict[str, object],
) -> None:
    state = _generation_two_open_state(chain)
    next_chain = state["chain"]
    claim_two_artifact = state["claim_two_artifact"]
    assert type(next_chain) is dict
    assert type(claim_two_artifact) is VersionedJsonArtifact
    claim_two_at = state["claim_two_at"]
    must_start_two = state["must_start_two"]
    assert type(claim_two_at) is datetime
    assert type(must_start_two) is datetime

    claim_only = _inventory(
        *[
            VersionedJsonArtifact(
                key=entry.key,
                raw=entry.raw,
                version_id=entry.version_id,
            )
            for entry in state["closed_one"]  # type: ignore[union-attr]
        ],
        claim_two_artifact,
    )
    assert generation.decide_production_generation_action(
        **_resolver_kwargs(
            next_chain,
            inventory=claim_only,
            attempt=ATTEMPT_B,
            now=claim_two_at,
        )  # type: ignore[arg-type]
    ) == generation.ProductionGenerationDecision(
        action="create-launch-once-decision",
        reason="current-claim-ready-for-launch-decision",
        generation=2,
    )
    assert generation.decide_production_generation_action(
        **_resolver_kwargs(
            next_chain,
            inventory=claim_only,
            attempt=ATTEMPT_B,
            now=must_start_two,
        )  # type: ignore[arg-type]
    ) == generation.ProductionGenerationDecision(
        action="create-expire-unstarted-decision",
        reason="current-claim-deadline-reached",
        generation=2,
    )

    launch = _build_decision_for_artifact(
        next_chain,
        claim_two_artifact,
        decision="launch-once",
        decided_at=claim_two_at,
    )
    launch_artifact = _decision_artifact(
        launch,
        version_id="generation-decision-version-2-launch",
    )
    assert generation.decide_production_generation_action(
        **_resolver_kwargs(
            next_chain,
            inventory=_inventory(
                *[
                    VersionedJsonArtifact(
                        key=entry.key,
                        raw=entry.raw,
                        version_id=entry.version_id,
                    )
                    for entry in state["closed_one"]  # type: ignore[union-attr]
                ],
                claim_two_artifact,
                launch_artifact,
            ),
            attempt=ATTEMPT_B,
            now=must_start_two + timedelta(days=1),
        )  # type: ignore[arg-type]
    ) == generation.ProductionGenerationDecision(
        action="reconcile-only",
        reason="stored-or-visible-generation-requires-reconciliation",
        generation=2,
    )

    expiry = _build_decision_for_artifact(
        next_chain,
        claim_two_artifact,
        decision="expire-unstarted",
        decided_at=must_start_two,
    )
    expiry_artifact = _decision_artifact(
        expiry,
        version_id="generation-decision-version-2-expiry",
    )
    final_snapshot = _fresh_snapshot(
        next_chain,
        must_start_two + timedelta(seconds=1),
        version_id="final-snapshot-version-2",
    )
    assert generation.decide_production_generation_action(
        **_resolver_kwargs(
            next_chain,
            inventory=_inventory(
                *[
                    VersionedJsonArtifact(
                        key=entry.key,
                        raw=entry.raw,
                        version_id=entry.version_id,
                    )
                    for entry in state["closed_one"]  # type: ignore[union-attr]
                ],
                claim_two_artifact,
                expiry_artifact,
            ),
            attempt=ATTEMPT_B,
            now=must_start_two + timedelta(seconds=1),
            snapshot=final_snapshot,
        )  # type: ignore[arg-type]
    ) == generation.ProductionGenerationDecision(
        action="create-expired-unstarted-terminal",
        reason="current-expiry-decision-ready-for-terminal",
        generation=2,
    )


@pytest.mark.parametrize("decision_outcome", ["launch-once", "expire-unstarted"])
def test_mutant_inventory_accepts_outcome_specific_decision_coordinate(
    chain: dict[str, object],
    decision_outcome: str,
) -> None:
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    decision = _decision(
        chain,
        claim,
        decision=decision_outcome,
        decided_at=(
            DECIDED_AT
            if decision_outcome == "launch-once"
            else MUST_START_BY
        ),
    )
    decision_artifact = _decision_artifact(decision)
    mutant_name = (
        "LAUNCH_DECISION.json"
        if decision_outcome == "launch-once"
        else "EXPIRE_DECISION.json"
    )
    alternate = replace(
        decision_artifact,
        key=decision_artifact.key.replace("START_DECISION.json", mutant_name),
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_production_generation_inventory(
            _inventory(claim_artifact, alternate),
            run_id=RUN_ID,
        )


def _chain_with_exact_approval_record(
    chain: dict[str, object],
    approval_record: dict[str, object],
    *,
    version_id: str,
) -> dict[str, object]:
    approval_raw = _file_bytes(approval_record)
    approval_file_sha = _sha(approval_raw)
    approval_artifact = VersionedJsonArtifact(
        key=(
            f"campaigns/{RUN_ID}/authorities/"
            f"GPU_SPEND_APPROVAL-{approval_file_sha}.json"
        ),
        raw=approval_raw,
        version_id=version_id,
    )
    descriptor_record = copy.deepcopy(chain["records"]["descriptor"])
    descriptor_record.update(
        {
            "approval_key": approval_artifact.key,
            "approval_sha256": approval_file_sha,
        }
    )
    identity = {
        key: item
        for key, item in descriptor_record.items()
        if key
        not in {
            "campaign_identity_sha256",
            "descriptor_body_sha256",
            "must_start_by",
            "campaign_descriptor_key",
        }
    }
    descriptor_record["campaign_identity_sha256"] = _sha(_canonical(identity))
    descriptor_record = _rehash(
        descriptor_record,
        "descriptor_body_sha256",
    )
    descriptor_artifact = _artifact(
        key=str(descriptor_record["campaign_descriptor_key"]),
        value=descriptor_record,
        version_id=f"descriptor-{version_id}",
    )

    snapshot_record = copy.deepcopy(chain["records"]["snapshot"])
    snapshot_record.update(
        {
            "campaign_identity_sha256": descriptor_record[
                "campaign_identity_sha256"
            ],
            "descriptor_sha256": _sha(descriptor_artifact.raw),
            "descriptor_body_sha256": descriptor_record[
                "descriptor_body_sha256"
            ],
            "approval_sha256": approval_file_sha,
            "approval_body_sha256": approval_record[
                "approval_body_sha256"
            ],
        }
    )
    snapshot_record = _rehash(snapshot_record, "snapshot_body_sha256")
    snapshot_artifact = _artifact(
        key=(
            f"campaigns/{RUN_ID}/spend-snapshots/"
            f"{snapshot_record['snapshot_body_sha256']}/"
            "GPU_SPEND_SNAPSHOT.json"
        ),
        value=snapshot_record,
        version_id=f"snapshot-{version_id}",
    )

    intent_record = copy.deepcopy(chain["records"]["intent"])
    intent_record.update(
        {
            "campaign_identity_sha256": descriptor_record[
                "campaign_identity_sha256"
            ],
            "descriptor_key": descriptor_artifact.key,
            "descriptor_file_sha256": _sha(descriptor_artifact.raw),
            "descriptor_body_sha256": descriptor_record[
                "descriptor_body_sha256"
            ],
            "descriptor_version_id": descriptor_artifact.version_id,
            "approval_key": approval_artifact.key,
            "approval_file_sha256": approval_file_sha,
            "approval_body_sha256": approval_record[
                "approval_body_sha256"
            ],
            "approval_version_id": approval_artifact.version_id,
            "gpu_spend_snapshot_key": snapshot_artifact.key,
            "gpu_spend_snapshot_file_sha256": _sha(snapshot_artifact.raw),
            "gpu_spend_snapshot_body_sha256": snapshot_record[
                "snapshot_body_sha256"
            ],
            "gpu_spend_snapshot_version_id": snapshot_artifact.version_id,
        }
    )
    intent_record = _rehash(intent_record, "intent_body_sha256")
    intent_artifact = _artifact(
        key=production_submission_intent_s3_key(
            run_id=RUN_ID,
            intent_body_sha256=str(intent_record["intent_body_sha256"]),
        ),
        value=intent_record,
        version_id=f"intent-{version_id}",
    )
    authorities = {
        "descriptor": descriptor_artifact,
        "intent": intent_artifact,
        "descriptor_record": descriptor_record,
        "intent_record": intent_record,
    }
    acquisition_tests = _load_test_module(
        "tests/test_glm52_sky_production_acquisition.py",
        f"_h1e_round4_approval_{version_id}",
    )
    h1d = acquisition_tests._full_chain(
        sys.modules["mlx_vq.quality.glm52_sky_production_acquisition"],
        authorities,
    )
    acquisition_record = h1d["acquisition"]
    acquisition_artifact = _artifact(
        key=production_submission_acquired_s3_key(
            run_id=RUN_ID,
            descriptor_file_sha256=_sha(descriptor_artifact.raw),
        ),
        value=acquisition_record,
        version_id=f"acquisition-{version_id}",
    )
    return {
        "descriptor": descriptor_artifact,
        "intent": intent_artifact,
        "approval": approval_artifact,
        "snapshot": snapshot_artifact,
        "baseline": h1d["baseline_artifact"],
        "ready": h1d["ready_artifact"],
        "acquisition": acquisition_artifact,
        "records": {
            "descriptor": descriptor_record,
            "intent": intent_record,
            "approval": approval_record,
            "snapshot": snapshot_record,
            "baseline": h1d["baseline"],
            "ready": h1d["ready"],
            "acquisition": acquisition_record,
        },
    }


@pytest.mark.parametrize(
    "snapshot_field",
    [
        "run_id",
        "campaign_identity_sha256",
        "descriptor_sha256",
        "descriptor_body_sha256",
        "approval_sha256",
        "approval_body_sha256",
        "gpu_spend_ledger_genesis_sha256",
        "ec2_allocation_history_sha256",
        "approved_gpu_runtime_seconds",
        "approved_gpu_cost_usd",
        "hourly_cost_usd",
        "consumed_gpu_seconds",
        "consumed_gpu_cost_usd",
        "remaining_gpu_seconds",
        "remaining_gpu_cost_usd",
        "open_allocation_count",
    ],
)
def test_mutant_claim_snapshot_ignores_one_semantic_axis(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    snapshot_field: str,
) -> None:
    current = chain["records"]["snapshot"][snapshot_field]
    if snapshot_field == "run_id":
        mutant: object = "glm52-sky-foreign-run"
    elif type(current) is str:
        mutant = _different_digest(current)
    elif type(current) is int:
        mutant = current + 1
    else:
        assert type(current) is float
        mutant = current + 0.01
    def permissive_snapshot(value: dict[str, object]) -> dict[str, object]:
        return dict(value)

    monkeypatch.setattr(
        sys.modules[__name__],
        "validate_gpu_spend_snapshot",
        permissive_snapshot,
    )
    monkeypatch.setattr(
        generation,
        "validate_gpu_spend_snapshot",
        permissive_snapshot,
    )
    drifted_chain = _coherently_readdressed_snapshot_chain(
        chain,
        snapshot_updates={snapshot_field: mutant},
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        _claim(drifted_chain)


def test_claim_copies_snapshot_record_count_without_an_intent_axis(
    chain: dict[str, object],
) -> None:
    record_count = chain["records"]["snapshot"]["gpu_spend_ledger_record_count"]
    assert type(record_count) is int
    instance_ids = chain["records"]["snapshot"]["ec2_allocation_instance_ids"]
    assert type(instance_ids) is list
    readdressed = _coherently_readdressed_snapshot_chain(
        chain,
        snapshot_updates={
            "gpu_spend_ledger_record_count": record_count + 2,
            "ec2_allocation_instance_ids": [
                *instance_ids,
                "i-0fedcba9876543210",
            ],
        },
    )

    claim = _claim(readdressed)

    assert claim["gpu_spend_ledger_record_count"] == record_count + 2


def test_mutant_final_snapshot_local_self_hash_is_delegated_to_public_validator(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        claim_artifact,
        _,
        decision_artifact,
        final_snapshot,
        _,
        _,
    ) = _expiry_chain(chain)
    record = json.loads(final_snapshot.raw)
    record["snapshot_body_sha256"] = _different_digest(
        record["snapshot_body_sha256"]
    )
    corrupt = _artifact(
        key=(
            f"campaigns/{RUN_ID}/spend-snapshots/"
            f"{record['snapshot_body_sha256']}/GPU_SPEND_SNAPSHOT.json"
        ),
        value=record,
        version_id=final_snapshot.version_id,
    )
    monkeypatch.setattr(
        generation,
        "validate_gpu_spend_snapshot",
        lambda value: dict(value),
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_terminal(
            generation_claim=claim_artifact,
            start_decision=decision_artifact,
            final_gpu_spend_snapshot=corrupt,
            terminal_at=MUST_START_BY + timedelta(seconds=20),
        )


@pytest.mark.parametrize("shifted_axis", ["seconds", "cost"])
def test_mutant_n_plus_one_accepts_internally_balanced_budget_shift(
    chain: dict[str, object],
    shifted_axis: str,
) -> None:
    next_budget = {
        "consumed_gpu_seconds": 1,
        "remaining_gpu_seconds": 86_399,
        "consumed_gpu_cost_usd": 0.02,
        "remaining_gpu_cost_usd": 1_320.94,
    }
    prior_updates = (
        {
            "consumed_gpu_cost_usd": next_budget["consumed_gpu_cost_usd"],
            "remaining_gpu_cost_usd": next_budget["remaining_gpu_cost_usd"],
        }
        if shifted_axis == "seconds"
        else {
            "consumed_gpu_seconds": next_budget["consumed_gpu_seconds"],
            "remaining_gpu_seconds": next_budget["remaining_gpu_seconds"],
        }
    )
    closed, terminal_artifact = _closed_generation_with_budget(
        chain,
        claim_updates=prior_updates,
    )
    next_chain, next_claim_at = _next_generation_chain(chain)
    shifted_chain = _coherently_readdressed_snapshot_chain(
        next_chain,
        snapshot_updates=next_budget,
        copy_snapshot_semantics=True,
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_claim(
            **_claim_kwargs(
                shifted_chain,
                generation_number=2,
                inventory=closed,
                previous_terminal=terminal_artifact,
                attempt=ATTEMPT_B,
                claim_at=next_claim_at,
            )  # type: ignore[arg-type]
        )


def test_mutant_n_plus_one_accepts_isolated_approval_version_adjacency_drift(
    chain: dict[str, object],
) -> None:
    (
        _,
        claim_artifact,
        _,
        decision_artifact,
        _,
        _,
        terminal_artifact,
    ) = _expiry_chain(chain)
    closed = _inventory(claim_artifact, decision_artifact, terminal_artifact)
    next_chain, next_claim_at = _next_generation_chain(chain)
    approval = next_chain["approval"]
    assert type(approval) is VersionedJsonArtifact
    different_version_chain = {
        **next_chain,
        "approval": replace(
            approval,
            version_id="approval-version-adjacency-only",
        ),
    }
    descriptor_record = copy.deepcopy(next_chain["records"]["descriptor"])
    snapshot_record = copy.deepcopy(next_chain["records"]["snapshot"])
    intent_record = copy.deepcopy(next_chain["records"]["intent"])
    intent_record["approval_version_id"] = "approval-version-adjacency-only"
    intent_record = _rehash(intent_record, "intent_body_sha256")
    intent_artifact = _artifact(
        key=production_submission_intent_s3_key(
            run_id=RUN_ID,
            intent_body_sha256=str(intent_record["intent_body_sha256"]),
        ),
        value=intent_record,
        version_id="intent-version-approval-adjacency-only",
    )
    authorities = {
        "descriptor": next_chain["descriptor"],
        "intent": intent_artifact,
        "descriptor_record": descriptor_record,
        "intent_record": intent_record,
    }
    acquisition_tests = _load_test_module(
        "tests/test_glm52_sky_production_acquisition.py",
        "_h1e_round4_approval_version_adjacency",
    )
    h1d = acquisition_tests._full_chain(
        sys.modules["mlx_vq.quality.glm52_sky_production_acquisition"],
        authorities,
    )
    acquisition_record = h1d["acquisition"]
    different_version_chain.update(
        {
            "intent": intent_artifact,
            "baseline": h1d["baseline_artifact"],
            "ready": h1d["ready_artifact"],
            "acquisition": _artifact(
                key=production_submission_acquired_s3_key(
                    run_id=RUN_ID,
                    descriptor_file_sha256=_sha(
                        next_chain["descriptor"].raw  # type: ignore[union-attr]
                    ),
                ),
                value=acquisition_record,
                version_id="acquisition-version-approval-adjacency-only",
            ),
            "records": {
                "descriptor": descriptor_record,
                "intent": intent_record,
                "approval": copy.deepcopy(next_chain["records"]["approval"]),
                "snapshot": snapshot_record,
                "baseline": h1d["baseline"],
                "ready": h1d["ready"],
                "acquisition": acquisition_record,
            },
        }
    )
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.build_production_generation_claim(
            **_claim_kwargs(
                different_version_chain,
                generation_number=2,
                inventory=closed,
                previous_terminal=terminal_artifact,
                attempt=ATTEMPT_B,
                claim_at=next_claim_at,
            )  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "approval_axis",
    [
        "approval_key",
        "approval_file_sha256",
        "approval_body_sha256",
    ],
)
def test_mutant_inventory_adjacency_ignores_one_approval_axis(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    approval_axis: str,
) -> None:
    state = _generation_two_open_state(chain)
    claim_two = copy.deepcopy(state["claim_two"])
    if approval_axis == "approval_key":
        claim_two[approval_axis] = (
            f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL-"
            f"{_different_digest(claim_two['approval_file_sha256'])}.json"
        )
    else:
        claim_two[approval_axis] = _different_digest(
            claim_two[approval_axis]
        )
    claim_two = _rehash(claim_two, "generation_claim_body_sha256")
    mutant_artifact = _claim_artifact(
        claim_two,
        version_id="generation-claim-version-2-isolated-approval-axis",
    )
    original_validate = generation._validate_claim_intrinsic

    def isolate_adjacency(
        value: Mapping[str, object],
    ) -> dict[str, object]:
        if (
            value.get("generation") == 2
            and value.get(approval_axis) == claim_two[approval_axis]
        ):
            return dict(value)
        return original_validate(value)

    monkeypatch.setattr(
        generation,
        "_validate_claim_intrinsic",
        isolate_adjacency,
    )
    inventory = [
        *state["closed_one"],  # type: ignore[misc]
        _entry(mutant_artifact),
    ]
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_production_generation_inventory(
            inventory,
            run_id=RUN_ID,
        )


def test_mutant_inventory_launch_terminal_guard_is_masked_by_stale_decision_edge(
    chain: dict[str, object],
) -> None:
    (
        claim,
        claim_artifact,
        _,
        _,
        _,
        terminal,
        _,
    ) = _expiry_chain(chain)
    launch = _decision(chain, claim)
    launch_artifact = _decision_artifact(
        launch,
        version_id="generation-decision-version-launch-terminal",
    )
    terminal.update(
        {
            "start_decision_key": launch_artifact.key,
            "start_decision_file_sha256": _sha(launch_artifact.raw),
            "start_decision_body_sha256": launch[
                "start_decision_body_sha256"
            ],
            "start_decision_version_id": launch_artifact.version_id,
        }
    )
    terminal = _rehash(terminal, "generation_terminal_body_sha256")
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        generation.validate_production_generation_inventory(
            _inventory(
                claim_artifact,
                launch_artifact,
                _terminal_artifact_for(terminal),
            ),
            run_id=RUN_ID,
        )


@pytest.mark.parametrize(
    ("record_kind", "field", "value"),
    [
        ("claim", "generation_text", "00000002"),
        ("claim", "previous_generation", 1),
        ("decision", "generation_text", "00000002"),
        ("terminal", "generation_text", "00000002"),
    ],
)
def test_mutant_standalone_record_semantic_value_guard_is_removed(
    chain: dict[str, object],
    record_kind: str,
    field: str,
    value: object,
) -> None:
    if record_kind == "claim":
        record = _claim(chain)
        digest_field = "generation_claim_body_sha256"
        helper = generation.production_generation_claim_file_bytes
    elif record_kind == "decision":
        claim = _claim(chain)
        record = _decision(chain, claim)
        digest_field = "start_decision_body_sha256"
        helper = generation.production_generation_start_decision_file_bytes
    else:
        record = _expiry_chain(chain)[5]
        digest_field = "generation_terminal_body_sha256"
        helper = generation.production_generation_terminal_file_bytes
    record[field] = value
    record = _rehash(record, digest_field)
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        helper(record)


def test_mutant_inventory_projection_substitutes_record_type_or_body_sha(
    chain: dict[str, object],
) -> None:
    (
        claim,
        claim_artifact,
        decision,
        decision_artifact,
        _,
        terminal,
        terminal_artifact,
    ) = _expiry_chain(chain)
    normalized = generation.validate_production_generation_inventory(
        _inventory(claim_artifact, decision_artifact, terminal_artifact),
        run_id=RUN_ID,
    )
    row = normalized["generations"][0]  # type: ignore[index]
    expected = (
        (
            row["claim"],
            "glm52_sky_production_generation_claim_v1",
            claim["generation_claim_body_sha256"],
        ),
        (
            row["start_decision"],
            "glm52_sky_production_generation_start_decision_v1",
            decision["start_decision_body_sha256"],
        ),
        (
            row["terminal"],
            "glm52_sky_production_generation_terminal_v1",
            terminal["generation_terminal_body_sha256"],
        ),
    )
    for projection, record_type, body_sha in expected:
        assert type(projection) is dict
        assert projection["record_type"] == record_type
        assert projection["body_sha256"] == body_sha


@pytest.mark.parametrize(
    ("mutation_name", "mutate"),
    [
        (
            "unknown-root-field",
            lambda record: record.update({"unexpected": "field"}),
        ),
        (
            "wrong-body-self-hash",
            lambda record: record.update(
                {
                    "approval_body_sha256": _different_digest(
                        record["approval_body_sha256"]
                    )
                }
            ),
        ),
        (
            "fractional-ingested-time",
            lambda record: record.update(
                {"approval_ingested_at": "2026-07-24T16:00:00.000000Z"}
            ),
        ),
        (
            "lowercase-z-ingested-time",
            lambda record: record.update(
                {"approval_ingested_at": "2026-07-24T16:00:00z"}
            ),
        ),
        (
            "leap-second-ingested-time",
            lambda record: record.update(
                {"approval_ingested_at": "2026-07-24T16:00:60Z"}
            ),
        ),
        (
            "whitespace-ingested-time",
            lambda record: record.update(
                {"approval_ingested_at": " 2026-07-24T16:00:00Z"}
            ),
        ),
        (
            "impossible-ingested-time",
            lambda record: record.update(
                {"approval_ingested_at": "2026-02-30T16:00:00Z"}
            ),
        ),
    ],
)
def test_mutant_approval_local_schema_hash_or_canonical_time_guard_is_delegated(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    mutation_name: str,
    mutate: Any,
) -> None:
    approval_record = copy.deepcopy(chain["records"]["approval"])
    mutate(approval_record)
    if mutation_name != "wrong-body-self-hash":
        approval_record = _rehash(
            approval_record,
            "approval_body_sha256",
        )
    mutant_chain = _chain_with_exact_approval_record(
        chain,
        approval_record,
        version_id=f"approval-{mutation_name}",
    )
    base_claim = _claim(chain)
    mutant_claim = _claim_for_chain(base_claim, mutant_chain)
    mutant_claim_artifact = _claim_artifact(
        mutant_claim,
        version_id="generation-claim-version-approval-local-guard",
    )
    mutant_decision = _decision_for_claim(
        _decision(chain, base_claim),
        mutant_claim,
        mutant_claim_artifact,
    )
    calls: list[object] = []

    def permissive(value: object) -> dict[str, object]:
        calls.append(value)
        return dict(value)  # type: ignore[arg-type]

    monkeypatch.setattr(generation, "validate_gpu_spend_approval", permissive)
    _assert_approval_mutant_rejected_at_six_boundaries(
        chain=mutant_chain,
        claim=mutant_claim,
        decision=mutant_decision,
        public_validator_calls=calls,
        local_gate_must_precede_public_validator=True,
    )


@pytest.mark.parametrize(
    "hostile_result",
    [
        None,
        {},
        {"unexpected": "value"},
        DictSubclass(),
    ],
)
def test_mutant_approval_public_validator_hostile_return_is_accepted_at_boundary(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    hostile_result: object,
) -> None:
    claim = _claim(chain)
    decision = _decision(chain, claim)
    calls: list[object] = []

    def hostile(value: object) -> object:
        calls.append(value)
        if type(hostile_result) is DictSubclass:
            return DictSubclass(dict(value))  # type: ignore[arg-type]
        if hostile_result == {"unexpected": "value"}:
            altered = dict(value)  # type: ignore[arg-type]
            altered["approval_response"] = "not-approved"
            return altered
        return hostile_result

    monkeypatch.setattr(generation, "validate_gpu_spend_approval", hostile)
    _assert_approval_mutant_rejected_at_six_boundaries(
        chain=chain,
        claim=claim,
        decision=decision,
        public_validator_calls=calls,
        local_gate_must_precede_public_validator=False,
    )


@pytest.mark.parametrize(
    "approval_axis",
    [
        "key",
        "version_id",
    ],
)
def test_mutant_approval_coordinate_axis_is_ignored_at_one_of_six_boundaries(
    chain: dict[str, object],
    approval_axis: str,
) -> None:
    approval = chain["approval"]
    assert type(approval) is VersionedJsonArtifact
    if approval_axis == "key":
        mutant_approval = replace(
            approval,
            key=approval.key.replace(
                "GPU_SPEND_APPROVAL-",
                "GPU_SPEND_APPROVAL_ALIAS-",
            ),
        )
    else:
        assert approval_axis == "version_id"
        mutant_approval = replace(
            approval,
            version_id="approval-version-axis-mutant",
        )
    mutant_chain = {**chain, "approval": mutant_approval}
    calls: list[object] = []
    _assert_approval_mutant_rejected_at_six_boundaries(
        chain=mutant_chain,
        claim=_claim(chain),
        decision=_decision(chain, _claim(chain)),
        public_validator_calls=calls,
        local_gate_must_precede_public_validator=None,
    )


def test_mutant_resolver_approval_preflight_is_masked_by_nested_branch_validation(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        claim_artifact,
        _,
        expiry_artifact,
        final_snapshot,
        _,
        terminal_artifact,
    ) = _expiry_chain(chain)
    launch = _decision_artifact(_decision(chain, _claim(chain)))
    inventories = [
        ([], CLAIM_AT, chain["snapshot"]),
        (_inventory(claim_artifact), DECIDED_AT, chain["snapshot"]),
        (_inventory(claim_artifact), MUST_START_BY, chain["snapshot"]),
        (
            _inventory(claim_artifact, launch),
            MUST_START_BY + timedelta(days=1),
            chain["snapshot"],
        ),
        (
            _inventory(claim_artifact, expiry_artifact),
            MUST_START_BY + timedelta(seconds=20),
            final_snapshot,
        ),
        (
            _inventory(claim_artifact, expiry_artifact, terminal_artifact),
            MUST_START_BY + timedelta(seconds=40),
            chain["snapshot"],
        ),
    ]
    original = generation._authenticate_approval
    calls: list[str] = []

    def approval_spy(*args: object, **kwargs: object) -> object:
        calls.append("approval")
        return original(*args, **kwargs)

    monkeypatch.setattr(generation, "_authenticate_approval", approval_spy)
    monkeypatch.setattr(generation, "_build_claim", lambda **kwargs: {})
    monkeypatch.setattr(generation, "_build_start_decision", lambda **kwargs: {})
    monkeypatch.setattr(generation, "_build_terminal", lambda **kwargs: {})
    monkeypatch.setattr(generation, "_authenticate_h1d", lambda **kwargs: {})
    monkeypatch.setattr(generation, "_claim_edges_match", lambda *args, **kwargs: None)
    for inventory, now, snapshot in inventories:
        calls.clear()
        result = generation.decide_production_generation_action(
            **_resolver_kwargs(
                chain,
                inventory=inventory,
                attempt=(
                    ATTEMPT_B
                    if len(inventory) == 3
                    and inventory[-1].key.endswith(
                        "GENERATION_TERMINAL.json"
                    )
                    else ATTEMPT_A
                ),
                now=now,
                snapshot=snapshot,  # type: ignore[arg-type]
            )  # type: ignore[arg-type]
        )
        assert result.action != "fail-closed"
        assert calls == ["approval"]


def test_mutant_flat_and_package_both_present_choose_package_or_misbind_adapter() -> None:
    flat_to_package = {
        "glm52_sky_campaign": "mlx_vq.quality.glm52_sky_campaign",
        "glm52_gpu_spend_snapshot": "mlx_vq.quality.glm52_gpu_spend_snapshot",
        "glm52_sky_production_submission": (
            "mlx_vq.quality.glm52_sky_production_submission"
        ),
        "glm52_sky_production_acquisition": (
            "mlx_vq.quality.glm52_sky_production_acquisition"
        ),
        "glm52_sky_submission_modes": (
            "mlx_vq.quality.glm52_sky_submission_modes"
        ),
    }
    original_modules = {
        name: sys.modules.get(name) for name in flat_to_package
    }
    flat_modules: dict[str, types.ModuleType] = {}
    target_name = "_h1e_round4_both_present"
    try:
        for flat_name, package_name in flat_to_package.items():
            package = importlib.import_module(package_name)
            flat = types.ModuleType(flat_name)
            for name in dir(package):
                if not name.startswith("__"):
                    setattr(flat, name, getattr(package, name))
            flat_modules[flat_name] = flat
            sys.modules[flat_name] = flat

        adapter_names = {
            "glm52_sky_campaign": (
                "validate_gpu_spend_approval",
                "validate_sky_campaign_descriptor",
            ),
            "glm52_gpu_spend_snapshot": (
                "validate_gpu_spend_snapshot",
            ),
            "glm52_sky_production_submission": (
                "VersionedJsonArtifact",
                "production_submission_intent_file_bytes",
                "production_submission_intent_file_sha256",
                "production_submission_intent_s3_key",
                "validate_production_submission_intent",
            ),
            "glm52_sky_production_acquisition": (
                "production_controller_baseline_file_bytes",
                "production_controller_baseline_file_sha256",
                "production_controller_baseline_s3_key",
                "production_must_start_control_plane_ready_file_bytes",
                "production_must_start_control_plane_ready_file_sha256",
                "production_must_start_control_plane_ready_s3_key",
                "production_submission_acquired_file_bytes",
                "production_submission_acquired_file_sha256",
                "production_submission_acquired_s3_key",
                "validate_production_controller_baseline",
                "validate_production_must_start_control_plane_ready",
                "validate_production_submission_acquired",
            ),
            "glm52_sky_submission_modes": (
                "SubmissionModeContractError",
                "expected_sky_job_name",
                "record_contract",
                "require_opaque_version_id",
            ),
        }
        sentinels: dict[str, object] = {}
        for module_name, attributes in adapter_names.items():
            for attribute in attributes:
                if attribute == "validate_gpu_spend_approval":

                    def sentinel(value: object) -> dict[str, object]:
                        return {"flat-adapter": value}

                    replacement: object = sentinel
                else:
                    replacement = object()
                setattr(flat_modules[module_name], attribute, replacement)
                sentinels[attribute] = replacement

        specification = importlib.util.spec_from_file_location(
            target_name,
            Path(generation.__file__),
        )
        assert specification is not None and specification.loader is not None
        target = importlib.util.module_from_spec(specification)
        sys.modules[target_name] = target
        specification.loader.exec_module(target)
        for attribute, sentinel in sentinels.items():
            assert getattr(target, attribute) is sentinel
        sample = {"x": "y"}
        assert target.validate_gpu_spend_approval(sample) == {
            "flat-adapter": sample
        }
    finally:
        sys.modules.pop(target_name, None)
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_mutant_module_initialization_swallows_nested_module_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = importlib.import_module

    def nested_missing(name: str, package: str | None = None) -> object:
        if name == "glm52_sky_campaign":
            error = ModuleNotFoundError("nested dependency missing")
            error.name = "nested_dependency"
            raise error
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", nested_missing)
    target_name = "_h1e_round4_nested_missing"
    specification = importlib.util.spec_from_file_location(
        target_name,
        Path(generation.__file__),
    )
    assert specification is not None and specification.loader is not None
    target = importlib.util.module_from_spec(specification)
    sys.modules[target_name] = target
    try:
        with pytest.raises(ModuleNotFoundError) as raised:
            specification.loader.exec_module(target)
        assert raised.value.name == "nested_dependency"
    finally:
        sys.modules.pop(target_name, None)


def test_mutant_forbidden_io_is_hidden_in_alias_or_unscanned_equivalent() -> None:
    tree = ast.parse(
        Path(generation.__file__).read_text(encoding="utf-8")
    )
    allowed_import_roots = {
        "__future__",
        "base64",
        "collections",
        "dataclasses",
        "datetime",
        "decimal",
        "hashlib",
        "importlib",
        "json",
        "math",
        "mlx_vq",
        "re",
        "typing",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                alias.name.split(".")[0] in allowed_import_roots
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module.split(".")[0] in allowed_import_roots
            assert not (
                node.module == "builtins"
                and any(alias.name == "open" for alias in node.names)
            )

    def dotted(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = dotted(node.value)
            return None if parent is None else f"{parent}.{node.attr}"
        return None

    forbidden_paths = {
        "builtins.open",
        "datetime.now",
        "datetime.utcnow",
        "os.environ",
        "os.getenv",
        "random.random",
        "random.randrange",
        "random.getrandbits",
        "secrets.token_bytes",
        "secrets.token_hex",
        "time.time",
        "time.monotonic",
        "time.perf_counter",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Call, ast.Attribute)):
            target = node.func if isinstance(node, ast.Call) else node
            path = dotted(target)
            if path is not None:
                assert path not in forbidden_paths
        if isinstance(node, ast.Name):
            assert node.id not in {
                "_cached_open",
                "open",
                "getenv",
                "urandom",
                "uuid4",
            }
    forbidden_values = {
        builtins.open,
        os.getenv,
        random.random,
        random.randrange,
        secrets.token_bytes,
        secrets.token_hex,
        time.time,
        time.monotonic,
        time.perf_counter,
    }
    assert all(
        value not in forbidden_values
        for name, value in vars(generation).items()
        if not name.startswith("__")
    )


def test_mutant_import_time_or_uncovered_resolver_branch_performs_io(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = Path(generation.__file__)
    module_code = compile(
        source_path.read_text(encoding="utf-8"),
        str(source_path),
        "exec",
    )
    claim = _claim(chain)
    claim_artifact = _claim_artifact(claim)
    launch_artifact = _decision_artifact(_decision(chain, claim))
    (
        _,
        _,
        _,
        expiry_artifact,
        final_snapshot,
        _,
        terminal_artifact,
    ) = _expiry_chain(chain)
    fail_closed_case = _resolver_kwargs(
        chain,
        inventory=[
            generation.GenerationInventoryEntry(
                key=generation.production_generation_claim_s3_key(
                    run_id=RUN_ID,
                    generation=1,
                ),
                raw=b"{}\n",
                version_id="malformed-inventory-version",
                is_latest=True,
                is_delete_marker=False,
            )
        ],
        attempt=ATTEMPT_A,
        now=CLAIM_AT,
    )
    resolver_cases = [
        _resolver_kwargs(
            chain,
            inventory=[],
            attempt=ATTEMPT_A,
            now=CLAIM_AT,
        ),
        _resolver_kwargs(
            chain,
            inventory=_inventory(claim_artifact),
            attempt=ATTEMPT_A,
            now=DECIDED_AT,
        ),
        _resolver_kwargs(
            chain,
            inventory=_inventory(claim_artifact),
            attempt=ATTEMPT_A,
            now=MUST_START_BY,
        ),
        _resolver_kwargs(
            chain,
            inventory=_inventory(claim_artifact, launch_artifact),
            attempt=ATTEMPT_A,
            now=MUST_START_BY + timedelta(days=1),
        ),
        _resolver_kwargs(
            chain,
            inventory=_inventory(claim_artifact, expiry_artifact),
            attempt=ATTEMPT_A,
            now=MUST_START_BY + timedelta(seconds=20),
            snapshot=final_snapshot,
        ),
        _resolver_kwargs(
            chain,
            inventory=_inventory(
                claim_artifact,
                expiry_artifact,
                terminal_artifact,
            ),
            attempt=ATTEMPT_B,
            now=MUST_START_BY + timedelta(seconds=40),
        ),
        fail_closed_case,
    ]
    for case in resolver_cases:
        generation.decide_production_generation_action(
            **case  # type: ignore[arg-type]
        )
    module_names_before = set(sys.modules)
    threads_before = tuple(thread.ident for thread in threading.enumerate())
    children_before = tuple(
        child.pid for child in multiprocessing.active_children()
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("pure resolver performed forbidden I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(time, "time", forbidden)
    monkeypatch.setattr(time, "monotonic", forbidden)
    monkeypatch.setattr(time, "perf_counter", forbidden)
    monkeypatch.setattr(random, "random", forbidden)
    monkeypatch.setattr(random, "randrange", forbidden)
    monkeypatch.setattr(random, "getrandbits", forbidden)
    monkeypatch.setattr(secrets, "token_bytes", forbidden)
    monkeypatch.setattr(secrets, "token_hex", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(http.client, "HTTPConnection", forbidden)
    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    allowed_flat_modules = {
        "glm52_sky_campaign": sys.modules[
            "mlx_vq.quality.glm52_sky_campaign"
        ],
        "glm52_gpu_spend_snapshot": sys.modules[
            "mlx_vq.quality.glm52_gpu_spend_snapshot"
        ],
        "glm52_sky_production_submission": sys.modules[
            "mlx_vq.quality.glm52_sky_production_submission"
        ],
        "glm52_sky_production_acquisition": sys.modules[
            "mlx_vq.quality.glm52_sky_production_acquisition"
        ],
        "glm52_sky_submission_modes": sys.modules[
            "mlx_vq.quality.glm52_sky_submission_modes"
        ],
    }

    def guarded_import_module(
        name: str,
        package: str | None = None,
    ) -> object:
        if package is None and name in allowed_flat_modules:
            return allowed_flat_modules[name]
        raise AssertionError(
            f"pure module performed unexpected dynamic import: {name}"
        )

    monkeypatch.setattr(
        importlib,
        "import_module",
        guarded_import_module,
    )
    target_name = "_h1e_round4_no_io_import"
    target = types.ModuleType(target_name)
    target.__file__ = str(source_path)
    sys.modules[target_name] = target
    try:
        exec(module_code, target.__dict__)
    finally:
        sys.modules.pop(target_name, None)
    for case in resolver_cases:
        result = generation.decide_production_generation_action(
            **case  # type: ignore[arg-type]
        )
        assert type(result) is generation.ProductionGenerationDecision
        if case is fail_closed_case:
            assert result == generation.ProductionGenerationDecision(
                action="fail-closed",
                reason="invalid-or-inconsistent-generation-authority",
                generation=None,
            )
    assert set(sys.modules) == module_names_before
    assert tuple(thread.ident for thread in threading.enumerate()) == threads_before
    assert tuple(
        child.pid for child in multiprocessing.active_children()
    ) == children_before


@pytest.mark.parametrize(
    ("adapter_name", "hostile_kind"),
    [
        ("validate_sky_campaign_descriptor", "none"),
        ("validate_production_submission_intent", "subclass"),
        ("validate_gpu_spend_approval", "altered"),
        ("validate_production_controller_baseline", "none"),
        ("validate_production_must_start_control_plane_ready", "subclass"),
        ("validate_production_submission_acquired", "altered"),
        ("validate_gpu_spend_snapshot", "none"),
        ("validate_sky_campaign_descriptor", "in-place"),
        ("validate_production_submission_intent", "in-place"),
        ("validate_gpu_spend_approval", "in-place"),
        ("validate_production_controller_baseline", "in-place"),
        ("validate_production_must_start_control_plane_ready", "in-place"),
        ("validate_production_submission_acquired", "in-place"),
        ("validate_gpu_spend_snapshot", "in-place"),
        ("validate_sky_campaign_descriptor", "nested-in-place"),
    ],
)
def test_mutant_hostile_dependency_adapter_return_is_trusted(
    chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    adapter_name: str,
    hostile_kind: str,
) -> None:
    def hostile(value: object, *args: object, **kwargs: object) -> object:
        del args, kwargs
        if hostile_kind == "nested-in-place":
            assert adapter_name == "validate_sky_campaign_descriptor"
            assert type(value) is dict
            artifacts = value["artifacts"]
            assert type(artifacts) is dict
            artifacts[next(iter(artifacts))] = _different_digest(
                next(iter(artifacts.values()))
            )
            return value
        if hostile_kind == "in-place":
            assert type(value) is dict
            field = {
                "validate_sky_campaign_descriptor": "task_name",
                "validate_gpu_spend_approval": "approval_response",
            }.get(adapter_name, "record_type")
            value[field] = object()
            return value
        if hostile_kind == "none":
            return None
        if hostile_kind == "subclass":
            return DictSubclass(dict(value))  # type: ignore[arg-type]
        altered = dict(value)  # type: ignore[arg-type]
        first_key = next(iter(altered))
        altered[first_key] = object()
        return altered

    monkeypatch.setattr(generation, adapter_name, hostile)
    with pytest.raises(generation.ProductionGenerationAuthorityError):
        _claim(chain)
