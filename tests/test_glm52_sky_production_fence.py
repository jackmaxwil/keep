"""Closed production-fence model tests for frozen H.1f."""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import mlx_vq.quality.glm52_sky_production_fence as fence
from mlx_vq.quality.glm52_sky_production_submission import (
    VersionedJsonArtifact,
)

EXPECTED_ALL = [
    "FenceAuthorityError",
    "FenceControlArtifact",
    "FenceLiveStateIdentity",
    "GenerationReservation",
    "ModeledFenceHead",
    "SourceEnrollment",
    "build_fence_genesis",
    "build_fence_successor",
    "fence_control_file_bytes",
    "fence_control_file_sha256",
    "fence_genesis_s3_key",
    "fence_successor_s3_key",
    "validate_fence_genesis",
    "validate_fence_successor",
    "walk_modeled_fence_chain",
]

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "glm52-sky-20260724"
CREATED_AT = datetime(2026, 7, 26, 12, 11, 25, tzinfo=timezone.utc)
GENESIS_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "campaign_identity_sha256",
    "genesis_key",
    "enrolled_sources",
    "reserved_generations",
    "live_state",
    "created_at",
    "fence_body_sha256",
}
SUCCESSOR_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "campaign_identity_sha256",
    "successor_key",
    "predecessor_control_key",
    "predecessor_control_version_id",
    "predecessor_control_file_sha256",
    "predecessor_control_body_sha256",
    "enrolled_sources",
    "reserved_generations",
    "added_sources",
    "added_generation_reservations",
    "live_state",
    "selected_at",
    "fence_body_sha256",
}


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


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _checksum(file_sha256: str) -> str:
    return base64.b64encode(bytes.fromhex(file_sha256)).decode("ascii")


def _rehash(record: dict[str, object]) -> dict[str, object]:
    body = copy.deepcopy(record)
    body.pop("fence_body_sha256", None)
    return {**body, "fence_body_sha256": _sha(_canonical(body))}


def _load_test_module(relative: str, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def authority_chain(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    generation_tests = _load_test_module(
        "tests/test_glm52_sky_production_generation.py",
        "_h1f_generation_fixture",
    )
    return generation_tests.chain.__wrapped__(tmp_path_factory)


def _source_enrollment(
    *,
    source_kind: str,
    artifact: VersionedJsonArtifact,
    body_sha256: str,
    metadata: tuple[tuple[str, str], ...] = (("glm52-run-id", RUN_ID),),
) -> fence.SourceEnrollment:
    file_sha256 = _sha(artifact.raw)
    return fence.SourceEnrollment(
        source_kind=source_kind,
        key=artifact.key,
        version_id=artifact.version_id,
        file_sha256=file_sha256,
        body_sha256=body_sha256,
        content_length=len(artifact.raw),
        checksum_sha256_base64=_checksum(file_sha256),
        checksum_type="FULL_OBJECT",
        content_type="application/json",
        metadata=metadata,
        publisher_principal_arn=(
            "arn:aws:iam::246813579024:role/glm52-source-publisher"
        ),
        publisher_principal_id="AROATEST:glm52-source-publisher",
        publisher_policy_sha256=_digest(f"{source_kind}-publisher-policy"),
        publisher_state="explicitly-denied",
    )


def _genesis_sources(
    chain: dict[str, object],
) -> tuple[fence.SourceEnrollment, ...]:
    records = chain["records"]
    assert isinstance(records, dict)
    entries = (
        _source_enrollment(
            source_kind="campaign-descriptor",
            artifact=chain["descriptor"],  # type: ignore[arg-type]
            body_sha256=records["descriptor"]["descriptor_body_sha256"],
        ),
        _source_enrollment(
            source_kind="production-submission-intent",
            artifact=chain["intent"],  # type: ignore[arg-type]
            body_sha256=records["intent"]["intent_body_sha256"],
        ),
        _source_enrollment(
            source_kind="production-controller-baseline",
            artifact=chain["baseline"],  # type: ignore[arg-type]
            body_sha256=records["baseline"]["baseline_body_sha256"],
        ),
        _source_enrollment(
            source_kind="production-control-plane-readiness",
            artifact=chain["ready"],  # type: ignore[arg-type]
            body_sha256=records["ready"]["control_plane_ready_body_sha256"],
        ),
        _source_enrollment(
            source_kind="production-submission-acquisition",
            artifact=chain["acquisition"],  # type: ignore[arg-type]
            body_sha256=records["acquisition"]["acquisition_body_sha256"],
        ),
        _source_enrollment(
            source_kind="gpu-spend-approval",
            artifact=chain["approval"],  # type: ignore[arg-type]
            body_sha256=records["approval"]["approval_body_sha256"],
        ),
        _source_enrollment(
            source_kind="gpu-spend-snapshot",
            artifact=chain["snapshot"],  # type: ignore[arg-type]
            body_sha256=records["snapshot"]["snapshot_body_sha256"],
        ),
    )
    return tuple(sorted(entries, key=lambda item: (item.source_kind, item.key)))


def _reservation(generation: int) -> fence.GenerationReservation:
    return fence.GenerationReservation(
        generation=generation,
        generation_text=f"{generation:08d}",
        claim_key=(
            f"campaigns/{RUN_ID}/submissions/production/generations/"
            f"{generation:08d}/GENERATION_CLAIM.json"
        ),
        claim_write_rule="conditional-single-part-put-if-absent",
        decision_key=(
            f"campaigns/{RUN_ID}/submissions/production/generations/"
            f"{generation:08d}/START_DECISION.json"
        ),
        decision_write_rule="conditional-single-part-put-if-absent",
        terminal_key=(
            f"campaigns/{RUN_ID}/submissions/production/generations/"
            f"{generation:08d}/GENERATION_TERMINAL.json"
        ),
        terminal_write_rule="conditional-single-part-put-if-absent",
    )


def _live_state(
    chain: dict[str, object],
) -> fence.FenceLiveStateIdentity:
    records = chain["records"]
    assert isinstance(records, dict)
    return fence.FenceLiveStateIdentity(
        account_id="246813579024",
        region="us-west-2",
        bucket=records["descriptor"]["bucket"],
        bucket_versioning_status="Enabled",
        bucket_mfa_delete_status="Disabled",
        bucket_policy_sha256=_digest("bucket-policy-1"),
        lifecycle_configuration_state="absent",
        lifecycle_configuration_sha256=_sha(b'{"state":"absent"}\n'),
        replication_configuration_state="absent",
        replication_configuration_sha256=_sha(b'{"state":"absent"}\n'),
        cloudformation_stack_id=(
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "stack/glm52-campaign/11111111-1111-1111-1111-111111111111"
        ),
        cloudformation_template_sha256=_digest("template-1"),
        cloudformation_parameters_sha256=_digest("parameters-1"),
        executor_identity_sha256=_digest("executor-1"),
        publisher_deny_policy_sha256=_digest("publisher-deny-1"),
        h1d_control_plane_ready_body_sha256=records["ready"][
            "control_plane_ready_body_sha256"
        ],
    )


def _source_kwargs(chain: dict[str, object]) -> dict[str, VersionedJsonArtifact]:
    return {
        "descriptor": chain["descriptor"],  # type: ignore[dict-item]
        "intent": chain["intent"],  # type: ignore[dict-item]
        "approval": chain["approval"],  # type: ignore[dict-item]
        "controller_baseline": chain["baseline"],  # type: ignore[dict-item]
        "must_start_control_plane_ready": chain["ready"],  # type: ignore[dict-item]
        "submission_acquisition": chain["acquisition"],  # type: ignore[dict-item]
        "gpu_spend_snapshot": chain["snapshot"],  # type: ignore[dict-item]
    }


def _build_genesis(chain: dict[str, object]) -> dict[str, object]:
    return fence.build_fence_genesis(
        **_source_kwargs(chain),
        source_enrollments=_genesis_sources(chain),
        generation_one_reservation=_reservation(1),
        live_state=_live_state(chain),
        created_at=CREATED_AT,
    )


def _control_artifact(
    record: dict[str, object],
    *,
    version_id: str = "fence-control-version-1",
    last_modified: str = "2026-07-26T12:11:25Z",
) -> fence.FenceControlArtifact:
    raw = _file_bytes(record)
    file_sha256 = _sha(raw)
    record_type = record["record_type"]
    key_field = (
        "genesis_key"
        if record_type == "glm52_sky_production_fence_genesis_v1"
        else "successor_key"
    )
    metadata = tuple(
        sorted(
            (
                ("glm52-body-sha256", record["fence_body_sha256"]),
                (
                    "glm52-campaign-identity-sha256",
                    record["campaign_identity_sha256"],
                ),
                ("glm52-file-sha256", file_sha256),
                ("glm52-record-type", record_type),
                ("glm52-run-id", record["run_id"]),
            )
        )
    )
    return fence.FenceControlArtifact(
        key=record[key_field],  # type: ignore[arg-type]
        raw=raw,
        version_id=version_id,
        file_sha256=file_sha256,
        body_sha256=record["fence_body_sha256"],  # type: ignore[arg-type]
        content_length=len(raw),
        etag='"11111111111111111111111111111111"',
        last_modified=last_modified,
        checksum_sha256_base64=_checksum(file_sha256),
        checksum_type="FULL_OBJECT",
        content_type="application/json",
        metadata=metadata,  # type: ignore[arg-type]
        missing_meta=0,
    )


def _added_source(
    *,
    label: str = "second-snapshot",
    source_kind: str = "gpu-spend-snapshot",
    key: str | None = None,
    body_sha256: str | None = None,
) -> fence.SourceEnrollment:
    artifact = VersionedJsonArtifact(
        key=(
            key
            if key is not None
            else f"campaigns/{RUN_ID}/enrolled-sources/{label}.json"
        ),
        raw=_file_bytes({"candidate": label}),
        version_id=f"{label}-version",
    )
    return _source_enrollment(
        source_kind=source_kind,
        artifact=artifact,
        body_sha256=body_sha256 or _digest(f"{label}-body"),
    )


def _successor_live_state(
    predecessor: fence.FenceLiveStateIdentity,
    *,
    label: str,
    added_sources: bool,
) -> fence.FenceLiveStateIdentity:
    return replace(
        predecessor,
        bucket_policy_sha256=_digest(f"{label}-bucket-policy"),
        publisher_deny_policy_sha256=(
            _digest(f"{label}-publisher-deny")
            if added_sources
            else predecessor.publisher_deny_policy_sha256
        ),
    )


def _build_successor(
    chain: dict[str, object],
    predecessor_controls: tuple[fence.FenceControlArtifact, ...],
    *,
    added_sources: tuple[fence.SourceEnrollment, ...] = (),
    added_reservations: tuple[fence.GenerationReservation, ...] = (),
    live_state: fence.FenceLiveStateIdentity | None = None,
    label: str = "successor-1",
) -> dict[str, object]:
    predecessor_record = json.loads(predecessor_controls[-1].raw)
    predecessor_live = fence.FenceLiveStateIdentity(
        **predecessor_record["live_state"]
    )
    return fence.build_fence_successor(
        **_source_kwargs(chain),
        predecessor_controls=predecessor_controls,
        added_sources=added_sources,
        added_generation_reservations=added_reservations,
        live_state=live_state
        or _successor_live_state(
            predecessor_live,
            label=label,
            added_sources=bool(added_sources),
        ),
        selected_at=CREATED_AT,
    )


@pytest.fixture(scope="module")
def modeled_chain(
    authority_chain: dict[str, object],
) -> dict[str, object]:
    genesis_record = _build_genesis(authority_chain)
    genesis_control = _control_artifact(genesis_record)
    first_added_source = _added_source(label="chain-source-2")
    first_record = _build_successor(
        authority_chain,
        (genesis_control,),
        added_sources=(first_added_source,),
        label="chain-successor-1",
    )
    first_control = _control_artifact(
        first_record,
        version_id="fence-control-version-2",
        last_modified="2026-07-26T12:11:26Z",
    )
    second_record = _build_successor(
        authority_chain,
        (genesis_control, first_control),
        added_reservations=(_reservation(2),),
        label="chain-successor-2",
    )
    second_control = _control_artifact(
        second_record,
        version_id="fence-control-version-3",
        last_modified="2026-07-26T12:11:27Z",
    )
    return {
        "genesis_record": genesis_record,
        "genesis_control": genesis_control,
        "first_added_source": first_added_source,
        "first_record": first_record,
        "first_control": first_control,
        "second_record": second_record,
        "second_control": second_control,
    }


def test_model_public_surface_is_exact() -> None:
    """A missing or widened fence model surface must fail closed."""

    module = importlib.import_module(
        "mlx_vq.quality.glm52_sky_production_fence"
    )
    assert module.__all__ == EXPECTED_ALL
    expected_signatures = {
        "build_fence_genesis": "(*, descriptor: 'VersionedJsonArtifact', intent: 'VersionedJsonArtifact', approval: 'VersionedJsonArtifact', controller_baseline: 'VersionedJsonArtifact', must_start_control_plane_ready: 'VersionedJsonArtifact', submission_acquisition: 'VersionedJsonArtifact', gpu_spend_snapshot: 'VersionedJsonArtifact', source_enrollments: 'tuple[SourceEnrollment, ...]', generation_one_reservation: 'GenerationReservation', live_state: 'FenceLiveStateIdentity', created_at: 'Union[datetime, str]') -> 'dict[str, object]'",
        "build_fence_successor": "(*, predecessor_controls: 'tuple[FenceControlArtifact, ...]', descriptor: 'VersionedJsonArtifact', intent: 'VersionedJsonArtifact', approval: 'VersionedJsonArtifact', controller_baseline: 'VersionedJsonArtifact', must_start_control_plane_ready: 'VersionedJsonArtifact', submission_acquisition: 'VersionedJsonArtifact', gpu_spend_snapshot: 'VersionedJsonArtifact', added_sources: 'tuple[SourceEnrollment, ...]', added_generation_reservations: 'tuple[GenerationReservation, ...]', live_state: 'FenceLiveStateIdentity', selected_at: 'Union[datetime, str]') -> 'dict[str, object]'",
        "fence_control_file_bytes": "(value: 'Mapping[str, object]') -> 'bytes'",
        "fence_control_file_sha256": "(value: 'Mapping[str, object]') -> 'str'",
        "fence_genesis_s3_key": "(*, run_id: 'str') -> 'str'",
        "fence_successor_s3_key": "(*, run_id: 'str', predecessor_body_sha256: 'str') -> 'str'",
        "validate_fence_genesis": "(value: 'Mapping[str, object]', *, descriptor: 'VersionedJsonArtifact', intent: 'VersionedJsonArtifact', approval: 'VersionedJsonArtifact', controller_baseline: 'VersionedJsonArtifact', must_start_control_plane_ready: 'VersionedJsonArtifact', submission_acquisition: 'VersionedJsonArtifact', gpu_spend_snapshot: 'VersionedJsonArtifact') -> 'dict[str, object]'",
        "validate_fence_successor": "(value: 'Mapping[str, object]', *, predecessor_controls: 'tuple[FenceControlArtifact, ...]', descriptor: 'VersionedJsonArtifact', intent: 'VersionedJsonArtifact', approval: 'VersionedJsonArtifact', controller_baseline: 'VersionedJsonArtifact', must_start_control_plane_ready: 'VersionedJsonArtifact', submission_acquisition: 'VersionedJsonArtifact', gpu_spend_snapshot: 'VersionedJsonArtifact') -> 'dict[str, object]'",
        "walk_modeled_fence_chain": "(*, run_id: 'str', controls: 'tuple[FenceControlArtifact, ...]', descriptor: 'VersionedJsonArtifact', intent: 'VersionedJsonArtifact', approval: 'VersionedJsonArtifact', controller_baseline: 'VersionedJsonArtifact', must_start_control_plane_ready: 'VersionedJsonArtifact', submission_acquisition: 'VersionedJsonArtifact', gpu_spend_snapshot: 'VersionedJsonArtifact') -> 'ModeledFenceHead'",
    }
    assert {
        name: str(inspect.signature(getattr(module, name)))
        for name in expected_signatures
    } == expected_signatures
    assert [item.name for item in fields(fence.SourceEnrollment)] == [
        "source_kind",
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
        "content_length",
        "checksum_sha256_base64",
        "checksum_type",
        "content_type",
        "metadata",
        "publisher_principal_arn",
        "publisher_principal_id",
        "publisher_policy_sha256",
        "publisher_state",
    ]
    assert [item.name for item in fields(fence.GenerationReservation)] == [
        "generation",
        "generation_text",
        "claim_key",
        "claim_write_rule",
        "decision_key",
        "decision_write_rule",
        "terminal_key",
        "terminal_write_rule",
    ]
    assert [item.name for item in fields(fence.FenceLiveStateIdentity)] == [
        "account_id",
        "region",
        "bucket",
        "bucket_versioning_status",
        "bucket_mfa_delete_status",
        "bucket_policy_sha256",
        "lifecycle_configuration_state",
        "lifecycle_configuration_sha256",
        "replication_configuration_state",
        "replication_configuration_sha256",
        "cloudformation_stack_id",
        "cloudformation_template_sha256",
        "cloudformation_parameters_sha256",
        "executor_identity_sha256",
        "publisher_deny_policy_sha256",
        "h1d_control_plane_ready_body_sha256",
    ]
    assert [item.name for item in fields(fence.FenceControlArtifact)] == [
        "key",
        "raw",
        "version_id",
        "file_sha256",
        "body_sha256",
        "content_length",
        "etag",
        "last_modified",
        "checksum_sha256_base64",
        "checksum_type",
        "content_type",
        "metadata",
        "missing_meta",
    ]
    assert [item.name for item in fields(fence.ModeledFenceHead)] == [
        "genesis",
        "successors",
        "head",
        "enrolled_sources",
        "reserved_generations",
        "next_successor_key",
    ]
    for data_class in (
        fence.SourceEnrollment,
        fence.GenerationReservation,
        fence.FenceLiveStateIdentity,
        fence.FenceControlArtifact,
        fence.ModeledFenceHead,
    ):
        assert data_class.__dataclass_params__.frozen


def test_genesis_schema_fields_and_canonical_bytes_are_exact(
    authority_chain: dict[str, object],
) -> None:
    record = _build_genesis(authority_chain)
    assert set(record) == GENESIS_FIELDS
    assert record["schema_version"] == 1
    assert record["record_type"] == "glm52_sky_production_fence_genesis_v1"
    body = copy.deepcopy(record)
    digest = body.pop("fence_body_sha256")
    assert digest == _sha(_canonical(body))
    assert fence.fence_control_file_bytes(record) == _file_bytes(record)
    assert fence.fence_control_file_sha256(record) == _sha(_file_bytes(record))


def test_successor_schema_fields_and_canonical_bytes_are_exact(
    modeled_chain: dict[str, object],
) -> None:
    record = modeled_chain["first_record"]
    assert isinstance(record, dict)
    assert set(record) == SUCCESSOR_FIELDS
    assert record["schema_version"] == 1
    assert record["record_type"] == "glm52_sky_production_fence_successor_v1"
    body = copy.deepcopy(record)
    digest = body.pop("fence_body_sha256")
    assert digest == _sha(_canonical(body))
    assert fence.fence_control_file_bytes(record) == _file_bytes(record)
    assert fence.fence_control_file_sha256(record) == _sha(_file_bytes(record))


def test_genesis_requires_exactly_seven_mandatory_sources(
    authority_chain: dict[str, object],
) -> None:
    sources = _genesis_sources(authority_chain)
    with pytest.raises(fence.FenceAuthorityError):
        fence.build_fence_genesis(
            **_source_kwargs(authority_chain),
            source_enrollments=sources[:-1],
            generation_one_reservation=_reservation(1),
            live_state=_live_state(authority_chain),
            created_at=CREATED_AT,
        )
    assert len(_build_genesis(authority_chain)["enrolled_sources"]) == 7


def test_genesis_live_state_binds_authenticated_readiness_body(
    authority_chain: dict[str, object],
) -> None:
    with pytest.raises(fence.FenceAuthorityError):
        fence.build_fence_genesis(
            **_source_kwargs(authority_chain),
            source_enrollments=_genesis_sources(authority_chain),
            generation_one_reservation=_reservation(1),
            live_state=replace(
                _live_state(authority_chain),
                h1d_control_plane_ready_body_sha256=_digest("foreign-ready"),
            ),
            created_at=CREATED_AT,
        )


def test_genesis_source_kind_maps_to_exact_record_validator_and_key(
    authority_chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = list(_genesis_sources(authority_chain))
    for index in range(len(sources)):
        wrong = list(sources)
        wrong[index] = replace(
            wrong[index],
            key=f"campaigns/{RUN_ID}/wrong-{index}.json",
        )
        with pytest.raises(fence.FenceAuthorityError):
            fence.build_fence_genesis(
                **_source_kwargs(authority_chain),
                source_enrollments=tuple(wrong),
                generation_one_reservation=_reservation(1),
                live_state=_live_state(authority_chain),
                created_at=CREATED_AT,
            )

    validator_names = (
        "validate_sky_campaign_descriptor",
        "validate_production_submission_intent",
        "validate_gpu_spend_approval",
        "validate_production_controller_baseline",
        "validate_production_must_start_control_plane_ready",
        "validate_production_submission_acquired",
        "validate_gpu_spend_snapshot",
    )
    for validator_name in validator_names:
        original = getattr(fence, validator_name)
        calls = 0

        def observe(
            *args: object,
            __original: Any = original,
            **kwargs: object,
        ) -> object:
            nonlocal calls
            calls += 1
            return __original(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(fence, validator_name, observe)
            _build_genesis(authority_chain)
        assert calls == 1


def test_source_metadata_is_captured_exactly_without_inventing_fields(
    authority_chain: dict[str, object],
) -> None:
    sources = list(_genesis_sources(authority_chain))
    sources[0] = replace(
        sources[0],
        metadata=(
            ("glm52-observed-publisher", "publisher-a"),
            ("glm52-run-id", RUN_ID),
        ),
    )
    record = fence.build_fence_genesis(
        **_source_kwargs(authority_chain),
        source_enrollments=tuple(sources),
        generation_one_reservation=_reservation(1),
        live_state=_live_state(authority_chain),
        created_at=CREATED_AT,
    )
    assert record["enrolled_sources"][0]["metadata"] == [
        ["glm52-observed-publisher", "publisher-a"],
        ["glm52-run-id", RUN_ID],
    ]
    with pytest.raises(fence.FenceAuthorityError):
        fence.build_fence_genesis(
            **_source_kwargs(authority_chain),
            source_enrollments=tuple(
                [replace(sources[0], metadata=tuple(reversed(sources[0].metadata)))]
                + sources[1:]
            ),
            generation_one_reservation=_reservation(1),
            live_state=_live_state(authority_chain),
            created_at=CREATED_AT,
        )
    wrong_body_metadata = replace(
        sources[0],
        metadata=(
            ("glm52-body-sha256", _digest("wrong-metadata-body")),
            ("glm52-run-id", RUN_ID),
        ),
    )
    with pytest.raises(fence.FenceAuthorityError):
        fence.build_fence_genesis(
            **_source_kwargs(authority_chain),
            source_enrollments=tuple([wrong_body_metadata] + sources[1:]),
            generation_one_reservation=_reservation(1),
            live_state=_live_state(authority_chain),
            created_at=CREATED_AT,
        )


def test_genesis_requires_only_generation_one_reservation(
    authority_chain: dict[str, object],
) -> None:
    with pytest.raises(fence.FenceAuthorityError):
        fence.build_fence_genesis(
            **_source_kwargs(authority_chain),
            source_enrollments=_genesis_sources(authority_chain),
            generation_one_reservation=_reservation(2),
            live_state=_live_state(authority_chain),
            created_at=CREATED_AT,
        )


def test_successor_key_derives_only_from_predecessor_body_sha(
    authority_chain: dict[str, object],
) -> None:
    genesis = _build_genesis(authority_chain)
    control = _control_artifact(genesis)
    successor = _build_successor(
        authority_chain,
        (control,),
        added_reservations=(_reservation(2),),
    )
    assert successor["successor_key"] == (
        f"campaigns/{RUN_ID}/authorities/fence/successors/"
        f"{genesis['fence_body_sha256']}/FENCE_SUCCESSOR.json"
    )


def test_successor_requires_nonempty_purely_additive_delta(
    authority_chain: dict[str, object],
) -> None:
    genesis_control = _control_artifact(_build_genesis(authority_chain))
    with pytest.raises(fence.FenceAuthorityError):
        _build_successor(authority_chain, (genesis_control,))


def test_successor_source_enrollment_is_structural_until_h1g_exact_read(
    authority_chain: dict[str, object],
) -> None:
    genesis_control = _control_artifact(_build_genesis(authority_chain))
    added = _added_source(
        key=f"campaigns/{RUN_ID}/structural-only/not-a-canonical-snapshot.json"
    )
    successor = _build_successor(
        authority_chain,
        (genesis_control,),
        added_sources=(added,),
    )
    assert successor["added_sources"][0]["key"] == added.key


@pytest.mark.parametrize(
    "bad_key",
    [
        "",
        "campaigns/foreign-run/source.json",
        f"campaigns/{RUN_ID}/authorities/fence/source.json",
        f"campaigns/{RUN_ID}/bad//source.json",
        f"campaigns/{RUN_ID}/bad/../source.json",
    ],
)
def test_successor_source_key_is_safe_run_scoped_and_outside_fence_namespace(
    authority_chain: dict[str, object],
    bad_key: str,
) -> None:
    genesis_control = _control_artifact(_build_genesis(authority_chain))
    with pytest.raises(fence.FenceAuthorityError):
        _build_successor(
            authority_chain,
            (genesis_control,),
            added_sources=(_added_source(key=bad_key),),
        )


def test_evidence_graph_contains_only_deeply_immutable_values(
    authority_chain: dict[str, object],
) -> None:
    head = fence.walk_modeled_fence_chain(
        run_id=RUN_ID,
        controls=(_control_artifact(_build_genesis(authority_chain)),),
        **_source_kwargs(authority_chain),
    )
    with pytest.raises(FrozenInstanceError):
        head.head = head.genesis  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        head.genesis.key = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        head.enrolled_sources[0].key = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        head.reserved_generations[0].generation = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        replace(_live_state(authority_chain)).bucket = "mutated"  # type: ignore[misc]
    assert isinstance(head.successors, tuple)
    assert isinstance(head.enrolled_sources, tuple)
    assert isinstance(head.reserved_generations, tuple)
    assert isinstance(head.head.raw, bytes)
    assert all(
        isinstance(source.metadata, tuple)
        and all(isinstance(pair, tuple) for pair in source.metadata)
        for source in head.enrolled_sources
    )


@pytest.mark.parametrize(
    "field",
    [
        "file_sha256",
        "body_sha256",
        "checksum_sha256_base64",
        "checksum_type",
        "content_type",
    ],
)
def test_control_artifact_rejects_scalar_subclasses_and_bool_integer_confusion(
    authority_chain: dict[str, object],
    field: str,
) -> None:
    class StringSubclass(str):
        pass

    control = _control_artifact(_build_genesis(authority_chain))
    with pytest.raises(fence.FenceAuthorityError):
        fence.walk_modeled_fence_chain(
            run_id=RUN_ID,
            controls=(
                replace(
                    control,
                    **{field: StringSubclass(getattr(control, field))},
                ),
            ),
            **_source_kwargs(authority_chain),
        )
    with pytest.raises(fence.FenceAuthorityError):
        fence.walk_modeled_fence_chain(
            run_id=RUN_ID,
            controls=(replace(control, content_length=True),),
            **_source_kwargs(authority_chain),
        )
    with pytest.raises(fence.FenceAuthorityError):
        fence.walk_modeled_fence_chain(
            run_id=RUN_ID,
            controls=(replace(control, missing_meta=False),),
            **_source_kwargs(authority_chain),
        )


def test_successor_preserves_complete_predecessor_sets_byte_for_byte(
    authority_chain: dict[str, object],
    modeled_chain: dict[str, object],
) -> None:
    genesis_record = modeled_chain["genesis_record"]
    first_record = modeled_chain["first_record"]
    assert isinstance(genesis_record, dict)
    assert isinstance(first_record, dict)
    assert first_record["enrolled_sources"] == sorted(
        genesis_record["enrolled_sources"]
        + first_record["added_sources"],
        key=lambda item: (item["source_kind"], item["key"]),
    )
    assert first_record["reserved_generations"] == genesis_record[
        "reserved_generations"
    ]
    validated = fence.validate_fence_successor(
        first_record,
        predecessor_controls=(modeled_chain["genesis_control"],),  # type: ignore[arg-type]
        **_source_kwargs(authority_chain),
    )
    assert validated == first_record


def test_successor_rejects_removal_mutation_weakening_duplicate_and_noop(
    authority_chain: dict[str, object],
    modeled_chain: dict[str, object],
) -> None:
    first_record = modeled_chain["first_record"]
    genesis_control = modeled_chain["genesis_control"]
    assert isinstance(first_record, dict)
    assert isinstance(genesis_control, fence.FenceControlArtifact)

    mutations: list[dict[str, object]] = []
    removed = copy.deepcopy(first_record)
    removed["enrolled_sources"] = removed["enrolled_sources"][1:]
    mutations.append(_rehash(removed))

    weakened = copy.deepcopy(first_record)
    weakened["enrolled_sources"][0]["publisher_state"] = "retired"
    mutations.append(_rehash(weakened))

    duplicate = copy.deepcopy(first_record)
    duplicate["enrolled_sources"].append(
        copy.deepcopy(duplicate["enrolled_sources"][0])
    )
    duplicate["enrolled_sources"] = sorted(
        duplicate["enrolled_sources"],
        key=lambda item: (item["source_kind"], item["key"]),
    )
    mutations.append(_rehash(duplicate))

    noop = copy.deepcopy(first_record)
    noop["added_sources"] = []
    noop["added_generation_reservations"] = []
    mutations.append(_rehash(noop))

    for mutation in mutations:
        with pytest.raises(fence.FenceAuthorityError):
            fence.validate_fence_successor(
                mutation,
                predecessor_controls=(genesis_control,),
                **_source_kwargs(authority_chain),
            )


def test_successor_intrinsic_delta_marks_only_newest_contiguous_reservation(
    modeled_chain: dict[str, object],
) -> None:
    second_record = copy.deepcopy(modeled_chain["second_record"])
    second_record["added_generation_reservations"] = [
        copy.deepcopy(second_record["reserved_generations"][0])
    ]
    second_record = _rehash(second_record)
    with pytest.raises(fence.FenceAuthorityError):
        fence.fence_control_file_bytes(second_record)


def test_successor_live_state_transition_is_exact_and_never_rolls_back(
    authority_chain: dict[str, object],
    modeled_chain: dict[str, object],
) -> None:
    genesis_control = modeled_chain["genesis_control"]
    first_control = modeled_chain["first_control"]
    assert isinstance(genesis_control, fence.FenceControlArtifact)
    assert isinstance(first_control, fence.FenceControlArtifact)
    first_live = fence.FenceLiveStateIdentity(
        **json.loads(first_control.raw)["live_state"]
    )
    genesis_live = fence.FenceLiveStateIdentity(
        **json.loads(genesis_control.raw)["live_state"]
    )

    with pytest.raises(fence.FenceAuthorityError):
        _build_successor(
            authority_chain,
            (genesis_control, first_control),
            added_reservations=(_reservation(2),),
            live_state=replace(
                first_live,
                bucket_policy_sha256=genesis_live.bucket_policy_sha256,
            ),
        )
    with pytest.raises(fence.FenceAuthorityError):
        _build_successor(
            authority_chain,
            (genesis_control, first_control),
            added_reservations=(_reservation(2),),
            live_state=replace(
                first_live,
                bucket="foreign-bucket",
                bucket_policy_sha256=_digest("new-policy"),
            ),
        )
    with pytest.raises(fence.FenceAuthorityError):
        _build_successor(
            authority_chain,
            (genesis_control, first_control),
            added_reservations=(_reservation(2),),
            live_state=replace(
                first_live,
                bucket_policy_sha256=_digest("new-policy"),
                publisher_deny_policy_sha256=_digest("unexpected-deny"),
            ),
        )
    with pytest.raises(fence.FenceAuthorityError):
        _build_successor(
            authority_chain,
            (genesis_control, first_control),
            added_reservations=(_reservation(2),),
            live_state=first_live,
        )
    with pytest.raises(fence.FenceAuthorityError):
        _build_successor(
            authority_chain,
            (genesis_control, first_control),
            added_sources=(_added_source(label="unchanged-publisher-deny"),),
            live_state=replace(
                first_live,
                bucket_policy_sha256=_digest("unchanged-publisher-policy"),
            ),
        )

    for index, field in enumerate(
        (
            "cloudformation_template_sha256",
            "cloudformation_parameters_sha256",
            "h1d_control_plane_ready_body_sha256",
        ),
        start=1,
    ):
        with pytest.raises(fence.FenceAuthorityError):
            _build_successor(
                authority_chain,
                (genesis_control, first_control),
                added_reservations=(_reservation(2),),
                live_state=replace(
                    first_live,
                    bucket_policy_sha256=_digest(f"unauthorized-ready-{index}"),
                    **{field: _digest(f"unauthorized-{field}")},
                ),
            )

    mismatched_readiness = _added_source(
        label="readiness-mismatch",
        source_kind="production-control-plane-readiness",
        body_sha256=_digest("readiness-enrollment"),
    )
    with pytest.raises(fence.FenceAuthorityError):
        _build_successor(
            authority_chain,
            (genesis_control, first_control),
            added_sources=(mismatched_readiness,),
            live_state=replace(
                first_live,
                bucket_policy_sha256=_digest("readiness-mismatch-policy"),
                publisher_deny_policy_sha256=_digest("readiness-mismatch-deny"),
                h1d_control_plane_ready_body_sha256=_digest("different-readiness"),
            ),
        )

    with pytest.raises(fence.FenceAuthorityError):
        _build_successor(
            authority_chain,
            (genesis_control, first_control),
            added_sources=(_added_source(label="publisher-rollback"),),
            live_state=replace(
                first_live,
                bucket_policy_sha256=_digest("publisher-rollback-policy"),
                publisher_deny_policy_sha256=(
                    genesis_live.publisher_deny_policy_sha256
                ),
            ),
        )

    readiness_body = _digest("readiness-v2")
    readiness_source = _added_source(
        label="readiness-v2",
        source_kind="production-control-plane-readiness",
        body_sha256=readiness_body,
    )
    readiness_live = replace(
        first_live,
        bucket_policy_sha256=_digest("readiness-v2-policy"),
        publisher_deny_policy_sha256=_digest("readiness-v2-deny"),
        cloudformation_template_sha256=_digest("readiness-v2-template"),
        cloudformation_parameters_sha256=_digest("readiness-v2-parameters"),
        h1d_control_plane_ready_body_sha256=readiness_body,
    )
    for index, (change_template, change_parameters) in enumerate(
        (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ),
        start=1,
    ):
        optional_body = _digest(f"readiness-optional-{index}")
        optional_source = _added_source(
            label=f"readiness-optional-{index}",
            source_kind="production-control-plane-readiness",
            body_sha256=optional_body,
        )
        optional_live = replace(
            first_live,
            bucket_policy_sha256=_digest(f"readiness-optional-policy-{index}"),
            publisher_deny_policy_sha256=_digest(
                f"readiness-optional-deny-{index}"
            ),
            cloudformation_template_sha256=(
                _digest(f"readiness-optional-template-{index}")
                if change_template
                else first_live.cloudformation_template_sha256
            ),
            cloudformation_parameters_sha256=(
                _digest(f"readiness-optional-parameters-{index}")
                if change_parameters
                else first_live.cloudformation_parameters_sha256
            ),
            h1d_control_plane_ready_body_sha256=optional_body,
        )
        optional_record = _build_successor(
            authority_chain,
            (genesis_control, first_control),
            added_sources=(optional_source,),
            live_state=optional_live,
        )
        optional_state = optional_record["live_state"]
        assert isinstance(optional_state, dict)
        assert optional_state["cloudformation_template_sha256"] == (
            optional_live.cloudformation_template_sha256
        )
        assert optional_state["cloudformation_parameters_sha256"] == (
            optional_live.cloudformation_parameters_sha256
        )
    readiness_record = _build_successor(
        authority_chain,
        (genesis_control, first_control),
        added_sources=(readiness_source,),
        live_state=readiness_live,
    )
    readiness_control = _control_artifact(
        readiness_record,
        version_id="readiness-v2-control",
    )

    for index, field in enumerate(
        (
            "cloudformation_template_sha256",
            "cloudformation_parameters_sha256",
            "h1d_control_plane_ready_body_sha256",
        ),
        start=1,
    ):
        rollback_body = (
            genesis_live.h1d_control_plane_ready_body_sha256
            if field == "h1d_control_plane_ready_body_sha256"
            else _digest(f"readiness-v3-{index}")
        )
        rollback_source = _added_source(
            label=f"readiness-rollback-{index}",
            source_kind="production-control-plane-readiness",
            body_sha256=rollback_body,
        )
        rollback_live = replace(
            readiness_live,
            bucket_policy_sha256=_digest(f"readiness-rollback-policy-{index}"),
            publisher_deny_policy_sha256=_digest(
                f"readiness-rollback-deny-{index}"
            ),
            h1d_control_plane_ready_body_sha256=rollback_body,
            **(
                {field: getattr(genesis_live, field)}
                if field != "h1d_control_plane_ready_body_sha256"
                else {}
            ),
        )
        with pytest.raises(fence.FenceAuthorityError):
            _build_successor(
                authority_chain,
                (genesis_control, first_control, readiness_control),
                added_sources=(rollback_source,),
                live_state=rollback_live,
            )


def test_build_successor_rewalks_incomplete_reordered_orphan_and_forked_predecessors(
    authority_chain: dict[str, object],
    modeled_chain: dict[str, object],
) -> None:
    genesis_control = modeled_chain["genesis_control"]
    first_control = modeled_chain["first_control"]
    second_control = modeled_chain["second_control"]
    assert isinstance(genesis_control, fence.FenceControlArtifact)
    assert isinstance(first_control, fence.FenceControlArtifact)
    assert isinstance(second_control, fence.FenceControlArtifact)

    reordered = fence.build_fence_successor(
        **_source_kwargs(authority_chain),
        predecessor_controls=(first_control, genesis_control),
        added_sources=(_added_source(label="rewalk-build"),),
        added_generation_reservations=(),
        live_state=_successor_live_state(
            fence.FenceLiveStateIdentity(
                **json.loads(first_control.raw)["live_state"]
            ),
            label="rewalk-build",
            added_sources=True,
        ),
        selected_at=CREATED_AT,
    )
    assert reordered["predecessor_control_key"] == first_control.key

    second_candidate = _added_source(label="forked-build")
    fork_record = _build_successor(
        authority_chain,
        (genesis_control,),
        added_sources=(second_candidate,),
        label="forked-build",
    )
    fork_control = _control_artifact(
        fork_record,
        version_id="forked-build-version",
    )
    invalid_predecessors = (
        (first_control,),
        (genesis_control, second_control),
        (genesis_control, first_control, fork_control),
    )
    for controls in invalid_predecessors:
        with pytest.raises(fence.FenceAuthorityError):
            fence.build_fence_successor(
                **_source_kwargs(authority_chain),
                predecessor_controls=controls,
                added_sources=(_added_source(label="unreachable-build"),),
                added_generation_reservations=(),
                live_state=_live_state(authority_chain),
                selected_at=CREATED_AT,
            )


def test_validate_successor_rewalks_incomplete_reordered_orphan_and_forked_predecessors(
    authority_chain: dict[str, object],
    modeled_chain: dict[str, object],
) -> None:
    genesis_control = modeled_chain["genesis_control"]
    first_control = modeled_chain["first_control"]
    second_control = modeled_chain["second_control"]
    second_record = modeled_chain["second_record"]
    assert isinstance(genesis_control, fence.FenceControlArtifact)
    assert isinstance(first_control, fence.FenceControlArtifact)
    assert isinstance(second_control, fence.FenceControlArtifact)
    assert isinstance(second_record, dict)

    fork_record = _build_successor(
        authority_chain,
        (genesis_control,),
        added_sources=(_added_source(label="forked-validate"),),
        label="forked-validate",
    )
    fork_control = _control_artifact(
        fork_record,
        version_id="forked-validate-version",
    )
    assert fence.validate_fence_successor(
        second_record,
        predecessor_controls=(first_control, genesis_control),
        **_source_kwargs(authority_chain),
    ) == second_record
    for controls in (
        (first_control,),
        (genesis_control, second_control),
        (genesis_control, first_control, fork_control),
    ):
        with pytest.raises(fence.FenceAuthorityError):
            fence.validate_fence_successor(
                second_record,
                predecessor_controls=controls,
                **_source_kwargs(authority_chain),
            )


def test_unknown_alias_duplicate_json_subclass_and_noncanonical_input_fails(
    authority_chain: dict[str, object],
) -> None:
    genesis = _build_genesis(authority_chain)

    class DictSubclass(dict[str, object]):
        pass

    with pytest.raises(fence.FenceAuthorityError):
        fence.validate_fence_genesis(
            DictSubclass(genesis),
            **_source_kwargs(authority_chain),
        )

    nested_subclass = copy.deepcopy(genesis)
    nested_subclass["live_state"] = DictSubclass(nested_subclass["live_state"])
    with pytest.raises(fence.FenceAuthorityError):
        fence.validate_fence_genesis(
            nested_subclass,
            **_source_kwargs(authority_chain),
        )

    alias = copy.deepcopy(genesis)
    alias["record_type"] = "FENCE_GENESIS"
    alias = _rehash(alias)
    with pytest.raises(fence.FenceAuthorityError):
        fence.fence_control_file_bytes(alias)

    control = _control_artifact(genesis)
    duplicate_raw = control.raw.replace(
        b'{"account_id":',
        b'{"schema_version":1,"account_id":',
        1,
    )
    duplicate_sha = _sha(duplicate_raw)
    with pytest.raises(fence.FenceAuthorityError):
        fence.walk_modeled_fence_chain(
            run_id=RUN_ID,
            controls=(
                replace(
                    control,
                    raw=duplicate_raw,
                    file_sha256=duplicate_sha,
                    content_length=len(duplicate_raw),
                    checksum_sha256_base64=_checksum(duplicate_sha),
                ),
            ),
            **_source_kwargs(authority_chain),
        )

    noncanonical_raw = control.raw[:-1] + b" \n"
    noncanonical_sha = _sha(noncanonical_raw)
    with pytest.raises(fence.FenceAuthorityError):
        fence.walk_modeled_fence_chain(
            run_id=RUN_ID,
            controls=(
                replace(
                    control,
                    raw=noncanonical_raw,
                    file_sha256=noncanonical_sha,
                    content_length=len(noncanonical_raw),
                    checksum_sha256_base64=_checksum(noncanonical_sha),
                ),
            ),
            **_source_kwargs(authority_chain),
        )


def test_fence_records_do_not_extend_h1e_registry() -> None:
    from mlx_vq.quality.glm52_sky_submission_modes import (
        SubmissionModeContractError,
        record_contract,
    )

    for record_kind in (
        "fence-genesis",
        "fence-successor",
    ):
        with pytest.raises(SubmissionModeContractError):
            record_contract(
                managed_mode="production",
                record_kind=record_kind,  # type: ignore[arg-type]
            )


def test_chain_walk_reaches_unique_zero_child_key_from_genesis(
    authority_chain: dict[str, object],
    modeled_chain: dict[str, object],
) -> None:
    controls = (
        modeled_chain["genesis_control"],
        modeled_chain["first_control"],
        modeled_chain["second_control"],
    )
    head = fence.walk_modeled_fence_chain(
        run_id=RUN_ID,
        controls=controls,  # type: ignore[arg-type]
        **_source_kwargs(authority_chain),
    )
    second_control = modeled_chain["second_control"]
    assert isinstance(second_control, fence.FenceControlArtifact)
    assert head.head is second_control
    assert head.successors == controls[1:]
    assert head.next_successor_key == (
        f"campaigns/{RUN_ID}/authorities/fence/successors/"
        f"{second_control.body_sha256}/FENCE_SUCCESSOR.json"
    )


def test_chain_rejects_orphan_alternate_key_unknown_record_and_unreachable(
    authority_chain: dict[str, object],
    modeled_chain: dict[str, object],
) -> None:
    genesis_control = modeled_chain["genesis_control"]
    first_control = modeled_chain["first_control"]
    second_control = modeled_chain["second_control"]
    assert isinstance(genesis_control, fence.FenceControlArtifact)
    assert isinstance(first_control, fence.FenceControlArtifact)
    assert isinstance(second_control, fence.FenceControlArtifact)

    cases = (
        (genesis_control, second_control),
        (
            replace(
                genesis_control,
                key=f"campaigns/{RUN_ID}/authorities/fence/ALTERNATE.json",
            ),
        ),
    )
    for controls in cases:
        with pytest.raises(fence.FenceAuthorityError):
            fence.walk_modeled_fence_chain(
                run_id=RUN_ID,
                controls=controls,
                **_source_kwargs(authority_chain),
            )

    unknown = json.loads(genesis_control.raw)
    unknown["record_type"] = "glm52_unknown_fence_record_v1"
    unknown = _rehash(unknown)
    unknown_raw = _file_bytes(unknown)
    unknown_sha = _sha(unknown_raw)
    unknown_control = replace(
        genesis_control,
        raw=unknown_raw,
        file_sha256=unknown_sha,
        body_sha256=unknown["fence_body_sha256"],
        content_length=len(unknown_raw),
        checksum_sha256_base64=_checksum(unknown_sha),
    )
    with pytest.raises(fence.FenceAuthorityError):
        fence.walk_modeled_fence_chain(
            run_id=RUN_ID,
            controls=(unknown_control,),
            **_source_kwargs(authority_chain),
        )


def test_chain_rejects_sibling_fork_cycle_and_repeated_digest(
    authority_chain: dict[str, object],
    modeled_chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    genesis_control = modeled_chain["genesis_control"]
    first_control = modeled_chain["first_control"]
    second_control = modeled_chain["second_control"]
    assert isinstance(genesis_control, fence.FenceControlArtifact)
    assert isinstance(first_control, fence.FenceControlArtifact)
    assert isinstance(second_control, fence.FenceControlArtifact)
    fork_record = _build_successor(
        authority_chain,
        (genesis_control,),
        added_sources=(_added_source(label="fork-branch"),),
        label="fork-branch",
    )
    fork_control = _control_artifact(
        fork_record,
        version_id="fork-branch-version",
    )
    for controls in (
        (genesis_control, first_control, fork_control),
        (genesis_control, first_control, first_control),
        (
            genesis_control,
            first_control,
            replace(
                first_control,
                key=first_control.key + ".alternate",
            ),
        ),
    ):
        with pytest.raises(fence.FenceAuthorityError):
            fence.walk_modeled_fence_chain(
                run_id=RUN_ID,
                controls=controls,
                **_source_kwargs(authority_chain),
            )

    original_artifact_validator = fence._validate_control_artifact
    _, second_record = original_artifact_validator(second_control)
    collision_control = replace(
        second_control,
        body_sha256=first_control.body_sha256,
    )

    def inject_validated_digest_collision(
        value: object,
    ) -> tuple[fence.FenceControlArtifact, dict[str, object]]:
        if value is collision_control:
            return collision_control, second_record
        return original_artifact_validator(value)

    with monkeypatch.context() as collision_patch:
        collision_patch.setattr(
            fence,
            "_validate_control_artifact",
            inject_validated_digest_collision,
        )
        with pytest.raises(
            fence.FenceAuthorityError,
            match="repeated body digest",
        ):
            fence.walk_modeled_fence_chain(
                run_id=RUN_ID,
                controls=(
                    genesis_control,
                    first_control,
                    collision_control,
                ),
                **_source_kwargs(authority_chain),
            )

    original_successor_key = fence._fence_successor_key

    def inject_cycle(run_id: str, predecessor_body_sha256: str) -> str:
        if predecessor_body_sha256 == second_control.body_sha256:
            return first_control.key
        return original_successor_key(run_id, predecessor_body_sha256)

    with monkeypatch.context() as cycle_patch:
        cycle_patch.setattr(fence, "_fence_successor_key", inject_cycle)
        with pytest.raises(fence.FenceAuthorityError, match="cycle"):
            fence.walk_modeled_fence_chain(
                run_id=RUN_ID,
                controls=(
                    genesis_control,
                    first_control,
                    second_control,
                ),
                **_source_kwargs(authority_chain),
            )


@pytest.mark.parametrize(
    "field",
    [
        "predecessor_control_version_id",
        "predecessor_control_file_sha256",
        "predecessor_control_body_sha256",
    ],
)
def test_chain_rejects_predecessor_version_file_or_body_drift(
    authority_chain: dict[str, object],
    modeled_chain: dict[str, object],
    field: str,
) -> None:
    record = copy.deepcopy(modeled_chain["first_record"])
    record[field] = (
        "foreign-version"
        if field.endswith("version_id")
        else _digest(f"foreign-{field}")
    )
    if field == "predecessor_control_body_sha256":
        record["successor_key"] = fence.fence_successor_s3_key(
            run_id=RUN_ID,
            predecessor_body_sha256=record[field],
        )
    record = _rehash(record)
    control = _control_artifact(record, version_id="drifted-successor-version")
    with pytest.raises(fence.FenceAuthorityError):
        fence.walk_modeled_fence_chain(
            run_id=RUN_ID,
            controls=(modeled_chain["genesis_control"], control),  # type: ignore[arg-type]
            **_source_kwargs(authority_chain),
        )


def test_chain_rejects_noncontiguous_reservation(
    authority_chain: dict[str, object],
    modeled_chain: dict[str, object],
) -> None:
    genesis_control = modeled_chain["genesis_control"]
    first_control = modeled_chain["first_control"]
    assert isinstance(genesis_control, fence.FenceControlArtifact)
    assert isinstance(first_control, fence.FenceControlArtifact)
    with pytest.raises(fence.FenceAuthorityError):
        _build_successor(
            authority_chain,
            (genesis_control, first_control),
            added_reservations=(_reservation(3),),
        )


def test_chain_ignores_discovery_time_lexical_and_latest_heuristics(
    authority_chain: dict[str, object],
    modeled_chain: dict[str, object],
) -> None:
    genesis_control = modeled_chain["genesis_control"]
    first_control = modeled_chain["first_control"]
    second_control = modeled_chain["second_control"]
    assert isinstance(genesis_control, fence.FenceControlArtifact)
    assert isinstance(first_control, fence.FenceControlArtifact)
    assert isinstance(second_control, fence.FenceControlArtifact)
    reordered = (
        replace(second_control, etag='"00000000000000000000000000000000"'),
        replace(
            first_control,
            etag='"ffffffffffffffffffffffffffffffff"',
        ),
        replace(genesis_control, last_modified="2026-07-26T12:11:59Z"),
    )
    head = fence.walk_modeled_fence_chain(
        run_id=RUN_ID,
        controls=reordered,
        **_source_kwargs(authority_chain),
    )
    assert head.head.key == second_control.key
    assert tuple(item.key for item in head.successors) == (
        first_control.key,
        second_control.key,
    )


def test_modeled_head_is_not_accepted_as_transport_authority(
    authority_chain: dict[str, object],
    modeled_chain: dict[str, object],
) -> None:
    head = fence.walk_modeled_fence_chain(
        run_id=RUN_ID,
        controls=(
            modeled_chain["genesis_control"],
            modeled_chain["first_control"],
        ),  # type: ignore[arg-type]
        **_source_kwargs(authority_chain),
    )
    for public_name in EXPECTED_ALL:
        public = getattr(fence, public_name)
        if not callable(public) or isinstance(public, type):
            continue
        signature = inspect.signature(public)
        assert "modeled_head" not in signature.parameters
        assert all(
            annotation is not fence.ModeledFenceHead
            and annotation != "ModeledFenceHead"
            for annotation in (
                parameter.annotation
                for parameter in signature.parameters.values()
            )
        )
        assert all(
            parameter.kind is not inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    with pytest.raises(fence.FenceAuthorityError):
        fence.walk_modeled_fence_chain(
            run_id=RUN_ID,
            controls=(head,),  # type: ignore[arg-type]
            **_source_kwargs(authority_chain),
        )


class _FatalFenceTest(BaseException):
    pass


@pytest.mark.parametrize(
    "fatal",
    [MemoryError("fatal-memory"), KeyboardInterrupt(), SystemExit(17), _FatalFenceTest()],
)
def test_fatal_process_exceptions_propagate_unchanged(
    authority_chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    fatal: BaseException,
) -> None:
    def explode(value: object) -> object:
        raise fatal

    monkeypatch.setattr(fence, "validate_sky_campaign_descriptor", explode)
    with pytest.raises(type(fatal)) as captured:
        _build_genesis(authority_chain)
    assert captured.value is fatal


def test_pure_model_ast_has_no_side_effect_or_external_authority_surface() -> None:
    source = inspect.getsource(fence)
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(
        {
            "boto3",
            "botocore",
            "http",
            "os",
            "pathlib",
            "random",
            "requests",
            "secrets",
            "shutil",
            "sky",
            "skypilot",
            "socket",
            "subprocess",
            "tempfile",
            "time",
            "urllib",
            "uuid",
        }
    )

    forbidden_calls = {
        "abort_multipart_upload",
        "client",
        "complete_multipart_upload",
        "copy_object",
        "delete_object",
        "delete_objects",
        "eval",
        "exec",
        "getenv",
        "list_object_versions",
        "list_objects_v2",
        "makedirs",
        "mkdir",
        "open",
        "put_object",
        "remove",
        "removedirs",
        "rename",
        "rmdir",
        "system",
        "touch",
        "unlink",
        "upload_part",
        "upload_part_copy",
        "write",
        "write_bytes",
        "write_text",
        "writelines",
    }
    forbidden_clock_random_or_environment = {
        "choice",
        "environ",
        "monotonic",
        "now",
        "perf_counter",
        "randbytes",
        "randint",
        "random",
        "time",
        "today",
        "token_bytes",
        "token_hex",
        "token_urlsafe",
        "urandom",
        "utcnow",
        "uuid1",
        "uuid4",
    }
    forbidden_h1e_resolvers = {
        "decide_production_generation_action",
        "validate_modeled_submit_once",
    }
    observed_call_names: set[str] = set()
    observed_string_literals: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                observed_call_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                observed_call_names.add(node.func.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            observed_string_literals.add(node.value)
    forbidden_names = (
        forbidden_calls
        | forbidden_clock_random_or_environment
        | forbidden_h1e_resolvers
    )
    assert observed_call_names.isdisjoint(forbidden_names)
    assert observed_string_literals.isdisjoint(forbidden_names)
    assert all(name not in source for name in forbidden_h1e_resolvers)
    assert not hasattr(fence, "main")
