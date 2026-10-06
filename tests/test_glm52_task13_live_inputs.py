from __future__ import annotations

from dataclasses import asdict
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.decision_closure import build_closure_request
from glm52_enforcement.task13_fixed_artifacts import (
    Task13FixedArtifactError,
    parse_driver_materialization_request,
)
from glm52_enforcement.task13_campaign_package import (
    build_campaign_package,
    derive_staged_infrastructure_successor_evidence,
)
from glm52_enforcement.task13_transport_gates import SEMANTIC_GATE_PINS
from glm52_task13_staged_fixture_support import (
    build_staged_infrastructure_evidence,
)


ACCOUNT = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
RUN_ID = "glm52-sky-20260724"
ACTIVATION = "approved-20260728"
BUCKET = "keep-glm52-models-246813579024-us-west-2"

PREQUALIFICATION_KEYS = {
    "TASK11_REVIEW_APPROVAL": "reviews/task11/approval.json",
    "TASK12_REVIEW_APPROVAL": "reviews/task12/approval.json",
    "RETAINED_FOUNDATION_TEMPLATE": (
        "task13/templates/retained-foundation.yaml"
    ),
    "RETAINED_PRE_SUPPORT_TEMPLATE": (
        "task13/templates/retained-pre-support.yaml"
    ),
    "RETAINED_TEMPLATE": "task13/templates/retained.yaml",
    "FENCE_TEMPLATE": "task13/migration/fence-transfer.json",
    "SUPPORT_TEMPLATE": "task13/templates/support-disabled.yaml",
    "SUPPORT_INPUTS": "task13/inputs/support-build-inputs.json",
    "BOOTSTRAP_TEMPLATE": "task13/templates/container-bootstrap-v1.json",
    "QUALIFICATION_CACHE_SEED_INPUT": (
        f"task13/activations/{ACTIVATION}/"
        "qualification/cache-seed-input.json"
    ),
    "H100_QUALIFICATION_INPUT": (
        f"task13/activations/{ACTIVATION}/qualification/h100-input.json"
    ),
    "T01_T25_GATE": "task13/gates/t01-t25.json",
    "TRANSPORT_22_MUTANT_GATE": "task13/gates/transport-22-mutants.json",
    "REPOSITORY_ARCHIVE": (
        f"task13/activations/{ACTIVATION}/archive/repo-tar.json"
    ),
    "ACCEPTED_BASELINE": "task13/inputs/accepted-baseline.json",
    "PROMPT_PACK": "task13/inputs/prompt-pack.json",
    "TRAINING_CONFIGURATION": "task13/inputs/training-configuration.json",
    "GPU_SPEND_APPROVAL": "task13/approvals/gpu-spend.json",
    "SUPPORT_APPROVAL": "task13/approvals/support-plane.json",
    "RESIDUAL_LIABILITY_APPROVAL": (
        "task13/approvals/residual-liability.json"
    ),
    "PRODUCTION_DESCRIPTOR": (
        f"task13/activations/{ACTIVATION}/inputs/campaign-descriptor-v2.json"
    ),
}
RETAINED_STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:stack/"
    "keep-glm52-gpu/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
)
FENCE_STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:stack/"
    "keep-glm52-h1g-fence/bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
)
SUPPORT_STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:stack/"
    "keep-glm52-h1g-support/cccccccc-dddd-4eee-8fff-aaaaaaaaaaaa"
)
DEPLOYMENT_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/"
    "keep-glm52-h1g-cloudformation-deployment"
)
STAGED_DEPLOYMENT_STEPS = [
    "ACCOUNT_LIVE_BASELINE",
    "BOOTSTRAP_INERT_ANCHORS",
    "PROVE_BUCKET_POLICY_ABSENT",
    "CREATE_FENCE_CHANGE_SET",
    "INSPECT_FENCE_CHANGE_SET",
    "EXECUTE_FENCE_UPDATE",
    "MATERIALIZE_PRE_SUPPORT",
    "UPDATE_RETAINED_PRE_SUPPORT",
    "MATERIALIZE_FULL_SUPPORT_INPUTS",
    "BUILD_PUBLISH_SUPPORT",
    "CAPTURE_PRECREATE_ORPHAN_AUTHORITY",
    "CREATE_SUPPORT_CHANGE_SET",
    "INSPECT_SUPPORT_CHANGE_SET",
    "EXECUTE_SUPPORT_CHANGE_SET",
    "MATERIALIZE_POSTCREATE_FRAGMENT",
    "UPDATE_RETAINED_FINAL",
    "POSTPUBLICATION_AUTHORITY",
    "FINAL_EXACT_READBACK",
    "PROVE_NO_WORKER_ACTIVATION",
]

_STALE_CAMPAIGN_BUCKET = (
    "keep-glm52-h1g-campaign-246813579024-us-west-2"
)
_OWNED_BUCKET_SURFACES = (
    "src/glm52_enforcement/support_plane.py",
    "src/glm52_enforcement/task10_capacity_reconciliation.py",
    "src/glm52_enforcement/task10_production.py",
    "src/glm52_enforcement/task10_support_plane.py",
    "src/glm52_enforcement/task10_task13_bridge.py",
    "src/glm52_enforcement/task13_authority_materialization.py",
    "src/glm52_enforcement/task13_campaign_package.py",
    "src/glm52_enforcement/task13_campaign_runner.py",
    "src/glm52_enforcement/task13_fixed_artifacts.py",
    "src/glm52_enforcement/task13_live_inputs.py",
    "aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py",
    "aws/glm52-gpu/scripts/publish_glm52_task13_transport_gates.py",
    "aws/glm52-gpu/cfn/h1g/support-task10-production-v1.json",
    (
        ".superpowers/sdd/"
        "2026-07-28-glm52-candidate13-implementation-freeze-addendum/"
        "task-13-live-execution-runbook.md"
    ),
)


def test_owned_task10_task13_surfaces_have_no_absent_campaign_bucket_pin(
) -> None:
    """Break caught: an executable surface revives the absent S3 identity."""

    root = Path(__file__).resolve().parents[1]
    stale = [
        relative
        for relative in _OWNED_BUCKET_SURFACES
        if _STALE_CAMPAIGN_BUCKET
        in (root / relative).read_text(encoding="utf-8")
    ]
    assert stale == []


def _write_canonical(path: Path, value: object) -> bytes:
    raw = canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return raw


def _task11_request() -> dict[str, object]:
    return asdict(
        build_closure_request(
            activation_id=ACTIVATION,
            generation=1,
            candidate_identity_sha256="9" * 64,
            initial_source_predecessor_version_id="3LgSourcePredecessor",
            admission_version_arn=(
                "arn:aws:lambda:us-west-2:246813579024:function:"
                "keep-glm52-h1g-launch-admission:17"
            ),
            numeric_binding_version_arn=(
                "arn:aws:lambda:us-west-2:246813579024:function:"
                "keep-glm52-h1g-numeric-binding:23"
            ),
            closure_budget_status="CLOSURE_BUDGET_PROVEN",
        )
    )


def _task11_boundary() -> dict[str, object]:
    return {
        "bucket": BUCKET,
        "key": (
            "campaigns/glm52-sky-20260724/authorities/task11/"
            "approved-20260728/00000001/TASK11_BOUNDARY.json"
        ),
        "version_id": "3LgTask11BoundaryVersion",
        "file_sha256": "7" * 64,
        "body_sha256": "8" * 64,
    }


def _prequalification_coordinates() -> list[dict[str, object]]:
    result = []
    for kind, key in PREQUALIFICATION_KEYS.items():
        semantic = SEMANTIC_GATE_PINS.get(kind)
        result.append(
            {
                "artifact_kind": kind,
                "bucket": BUCKET,
                "key": key,
                "version_id": "3LgExactImmutableVersion",
                "file_sha256": (
                    semantic["file_sha256"]
                    if semantic is not None
                    else hashlib.sha256((kind + "-file").encode()).hexdigest()
                ),
                "body_sha256": (
                    semantic["body_sha256"]
                    if semantic is not None
                    else hashlib.sha256(kind.encode()).hexdigest()
                ),
            }
        )
    return result


def _post_h100_task10_coordinates() -> list[dict[str, object]]:
    return [
        {
            "artifact_kind": kind,
            "bucket": BUCKET,
            "key": key,
            "version_id": "3Lg" + kind.title().replace("_", "") + "Version",
            "file_sha256": hashlib.sha256(
                (kind + "-file").encode()
            ).hexdigest(),
            "body_sha256": hashlib.sha256(kind.encode()).hexdigest(),
        }
        for kind, key in (
            (
                "TASK10_PRODUCTION_AUTHORITY",
                "task13/production/task10-production-authority.json",
            ),
            (
                "TASK10_WORKER_DESCRIPTOR",
                "task13/production/task10-worker-descriptor.json",
            ),
            (
                "TASK10_TASK_INPUTS",
                "task13/production/task10-task-inputs.json",
            ),
        )
    ]


def _staged_infrastructure_evidence(
    coordinates: list[dict[str, object]],
) -> dict[str, object]:
    return build_staged_infrastructure_evidence(
        coordinates,
        activation_id=ACTIVATION,
        retained_stack_id=RETAINED_STACK_ID,
        fence_stack_id=FENCE_STACK_ID,
        support_stack_id=SUPPORT_STACK_ID,
    )


def _prequalification_bundle(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    from glm52_enforcement.task13_live_inputs import (
        assemble_prequalification_inputs,
        build_prequalification_base_request,
    )

    coordinates = _prequalification_coordinates()
    base = build_prequalification_base_request(
        activation_id=ACTIVATION,
        collector_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-rehearsal-collector:19"
        ),
        task11_request=_task11_request(),
        task11_boundary=_task11_boundary(),
        staged_infrastructure_evidence=(
            _staged_infrastructure_evidence(coordinates)
        ),
        retained_stack_id=RETAINED_STACK_ID,
        fence_change_set_name="glm52-task13-fence-disabled-0001",
        support_change_set_name="glm52-task13-support-disabled-0001",
        monitor_descriptor_path=str(
            (tmp_path / "production/campaign-descriptor-v2.json").resolve()
        ),
    )
    assembled = assemble_prequalification_inputs(
        base_request=base,
        coordinate_documents=coordinates,
    )
    return assembled, build_campaign_package(assembled["request"])


def _h100_stage_result(
    package: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    marker_body = {
        "key": (
            f"campaigns/{RUN_ID}/qualification/"
            "H100_RESUME_READY.json"
        ),
        "version_id": "3LgH100ResumeReadyVersion",
        "file_sha256": "a" * 64,
        "body_sha256": "b" * 64,
    }
    marker = {
        **marker_body,
        "canonical_identity_sha256": canonical_sha256(marker_body),
    }
    summary = {
        "distinct_instance_count": 2,
        "peak_memory_gib": 69,
        "training_steps": 2,
        "h100_resume_ready": marker,
    }
    body = {
        "status": "STAGE_COMMITTED",
        "stage": "h100-qualification",
        "package_identity_sha256": package[
            "canonical_identity_sha256"
        ],
        **summary,
    }
    result = {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }
    request_identity = canonical_sha256(
        {
            "stage": "h100-qualification",
            "package_identity_sha256": package[
                "canonical_identity_sha256"
            ],
            "reviewed_artifacts_identity_sha256": canonical_sha256(
                package["reviewed_artifacts"]
            ),
        }
    )
    rows = []
    for sequence, state in enumerate(
        ("PREPARED", "POSSIBLY_SENT", "COMMITTED"),
        1,
    ):
        journal_body = {
            "schema_version": 1,
            "record_type": "glm52_task13_runner_journal_v1",
            "operation_id": "stage:h100-qualification",
            "operation_kind": "stage-completion",
            "stage": "h100-qualification",
            "sequence": sequence,
            "state": state,
            "request_identity_sha256": request_identity,
            "result_identity_sha256": (
                canonical_sha256(summary)
                if state == "COMMITTED"
                else None
            ),
        }
        rows.append(
            {
                **journal_body,
                "canonical_identity_sha256": canonical_sha256(
                    journal_body
                ),
            }
        )
    return result, rows


def test_cache_seed_driver_request_binds_exact_sources_and_argv(
    tmp_path: Path,
) -> None:
    """Break caught: the cache driver invents argv or omits source-byte pins."""

    from glm52_enforcement.task13_live_inputs import (
        build_cache_seed_driver_request,
    )

    paths: dict[str, Path] = {}
    raws: dict[str, bytes] = {}
    for role in (
        "campaign_descriptor",
        "approval",
        "staged_ready",
        "rehearsal_evidence",
        "task",
        "config",
    ):
        path = (tmp_path / f"{role}.json").resolve()
        paths[role] = path
        raws[role] = _write_canonical(
            path,
            {"record_type": f"exact-{role}"},
        )
    sky_bin = (tmp_path / "sky").resolve()
    sky_bin.write_text("#!/bin/sh\nexit 64\n", encoding="ascii")
    sky_bin.chmod(0o700)

    value = build_cache_seed_driver_request(
        activation_id=ACTIVATION,
        campaign_descriptor=paths["campaign_descriptor"],
        approval=paths["approval"],
        staged_ready=paths["staged_ready"],
        staged_ready_version_id="opaque-staged-version-1",
        rehearsal_evidence=paths["rehearsal_evidence"],
        task=paths["task"],
        config=paths["config"],
        sky_bin=sky_bin,
    )

    expected_sources = [
        {
            "role": role,
            "path": str(paths[role]),
            "size_bytes": len(raws[role]),
            "file_sha256": hashlib.sha256(raws[role]).hexdigest(),
        }
        for role in (
            "campaign_descriptor",
            "approval",
            "staged_ready",
            "rehearsal_evidence",
            "task",
            "config",
        )
    ]
    expected_body = {
        "schema_version": 2,
        "record_type": "glm52_task13_driver_materialization_request_v2",
        "account_id": ACCOUNT,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "operation_kind": "qualification-cache-seed",
        "argv": [
            "aws/glm52-gpu/scripts/submit_sky_campaign.py",
            "cache-seed",
            "acquire-and-launch",
            "--profile",
            PROFILE,
            "--descriptor",
            str(paths["campaign_descriptor"]),
            "--approval",
            str(paths["approval"]),
            "--staged-ready",
            str(paths["staged_ready"]),
            "--staged-ready-version-id",
            "opaque-staged-version-1",
            "--rehearsal-evidence",
            str(paths["rehearsal_evidence"]),
            "--task",
            str(paths["task"]),
            "--config",
            str(paths["config"]),
            "--sky-bin",
            str(sky_bin),
        ],
        "environment": {
            "AWS_PROFILE": PROFILE,
            "AWS_REGION": REGION,
        },
        "source_coordinates": expected_sources,
    }
    assert value == {
        **expected_body,
        "canonical_identity_sha256": canonical_sha256(expected_body),
    }
    assert parse_driver_materialization_request(value) == value

    paths["campaign_descriptor"].write_bytes(b'{"mutated":true}\n')
    with pytest.raises(Task13FixedArtifactError, match="source bytes"):
        parse_driver_materialization_request(value)


def test_h100_driver_request_has_one_descriptor_source(
    tmp_path: Path,
) -> None:
    """Break caught: the H100 driver admits an unpinned or extra input."""

    from glm52_enforcement.task13_live_inputs import (
        build_h100_driver_request,
    )

    descriptor = (tmp_path / "campaign-descriptor-v2.json").resolve()
    raw = _write_canonical(
        descriptor,
        {"record_type": "exact-production-descriptor"},
    )

    value = build_h100_driver_request(
        activation_id=ACTIVATION,
        campaign_descriptor=descriptor,
    )

    expected_body = {
        "schema_version": 2,
        "record_type": "glm52_task13_driver_materialization_request_v2",
        "account_id": ACCOUNT,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "operation_kind": "h100-qualification",
        "argv": [
            "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh"
        ],
        "environment": {
            "AWS_PROFILE": PROFILE,
            "AWS_REGION": REGION,
            "CAMPAIGN_DESCRIPTOR": str(descriptor),
        },
        "source_coordinates": [
            {
                "role": "campaign_descriptor",
                "path": str(descriptor),
                "size_bytes": len(raw),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ],
    }
    assert value == {
        **expected_body,
        "canonical_identity_sha256": canonical_sha256(expected_body),
    }
    assert parse_driver_materialization_request(value) == value


def test_driver_request_cli_writes_once_without_replacement(
    tmp_path: Path,
) -> None:
    """Break caught: the driver CLI overwrites a reviewed request."""

    descriptor = (tmp_path / "campaign-descriptor-v2.json").resolve()
    _write_canonical(
        descriptor,
        {"record_type": "exact-production-descriptor"},
    )
    output = (tmp_path / "h100-driver-request.json").resolve()
    script = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/scripts/build_glm52_task13_driver_request.py"
    )
    command = [
        sys.executable,
        str(script),
        "h100",
        "--activation-id",
        ACTIVATION,
        "--campaign-descriptor",
        str(descriptor),
        "--output",
        str(output),
    ]

    first = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    raw = output.read_bytes()
    value = json.loads(raw)
    assert raw == canonical_json_bytes(value) + b"\n"
    assert value["operation_kind"] == "h100-qualification"
    assert output.stat().st_mode & 0o777 == 0o600

    second = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 64
    assert output.read_bytes() == raw


def _write_fresh_repository_tar(
    path: Path,
    *,
    coordinator_raw: bytes,
    omit: str | None = None,
    duplicate: str | None = None,
    symlink: str | None = None,
    noncanonical_member: bool = False,
    unsafe_member: bool = False,
) -> bytes:
    from glm52_enforcement.task13_live_inputs import (
        REQUIRED_EXECUTABLE_PATHS,
    )

    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for relative_path in REQUIRED_EXECUTABLE_PATHS:
            if relative_path == omit:
                continue
            if relative_path == symlink:
                member = tarfile.TarInfo(relative_path)
                member.type = tarfile.SYMTYPE
                member.linkname = REQUIRED_EXECUTABLE_PATHS[-1]
                member.mode = 0o755
                member.mtime = 0
                archive.addfile(member)
                continue
            raw = (
                coordinator_raw
                if relative_path.endswith(
                    "glm52_task13_production_coordinator.py"
                )
                else f"fixture for {relative_path}\n".encode()
            )
            member = tarfile.TarInfo(relative_path)
            member.size = len(raw)
            member.mode = 0o755
            member.mtime = 0
            archive.addfile(member, io.BytesIO(raw))
        if duplicate is not None:
            raw = b"duplicate repository member\n"
            member = tarfile.TarInfo(duplicate)
            member.size = len(raw)
            member.mode = 0o755
            member.mtime = 0
            archive.addfile(member, io.BytesIO(raw))
        if noncanonical_member:
            raw = b"noncanonical repository member\n"
            member = tarfile.TarInfo("./member-drift.txt")
            member.size = len(raw)
            member.mode = 0o644
            member.mtime = 0
            archive.addfile(member, io.BytesIO(raw))
        if unsafe_member:
            raw = b"escaped repository member\n"
            member = tarfile.TarInfo("../escape.txt")
            member.size = len(raw)
            member.mode = 0o644
            member.mtime = 0
            archive.addfile(member, io.BytesIO(raw))
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0) as output:
        output.write(stream.getvalue())
    raw = compressed.getvalue()
    path.write_bytes(raw)
    return raw


def _fresh_joiner_kwargs(
    tmp_path: Path,
    *,
    archive_coordinator_raw: bytes | None = None,
    omit: str | None = None,
    duplicate: str | None = None,
    symlink: str | None = None,
    noncanonical_member: bool = False,
    unsafe_member: bool = False,
) -> dict[str, object]:
    source_root = tmp_path / "fresh-sources"
    source_root.mkdir(parents=True)
    paths = {}
    for role in (
        "h100_campaign_descriptor",
        "cache_seed_campaign_descriptor",
        "approval",
        "staged_ready",
        "rehearsal_evidence",
        "task",
        "config",
    ):
        path = (source_root / f"{role}.json").resolve()
        _write_canonical(path, {"record_type": f"exact-{role}"})
        paths[role] = path
    sky_bin = (source_root / "sky").resolve()
    sky_bin.write_text("#!/bin/sh\nexit 64\n", encoding="ascii")
    sky_bin.chmod(0o700)
    coordinator = (source_root / "coordinator.py").resolve()
    coordinator_raw = b"#!/usr/bin/env python3\nraise SystemExit(64)\n"
    coordinator.write_bytes(coordinator_raw)
    coordinator.chmod(0o700)
    repository_archive = (source_root / "repository.tar.gz").resolve()
    _write_fresh_repository_tar(
        repository_archive,
        coordinator_raw=(
            coordinator_raw
            if archive_coordinator_raw is None
            else archive_coordinator_raw
        ),
        omit=omit,
        duplicate=duplicate,
        symlink=symlink,
        noncanonical_member=noncanonical_member,
        unsafe_member=unsafe_member,
    )
    return {
        "activation_id": ACTIVATION,
        "repository_archive_path": repository_archive,
        "coordinator_path": coordinator,
        "h100_campaign_descriptor": paths["h100_campaign_descriptor"],
        "cache_seed_campaign_descriptor": paths[
            "cache_seed_campaign_descriptor"
        ],
        "approval": paths["approval"],
        "staged_ready": paths["staged_ready"],
        "staged_ready_version_id": "opaque-staged-version-1",
        "rehearsal_evidence": paths["rehearsal_evidence"],
        "task": paths["task"],
        "config": paths["config"],
        "sky_bin": sky_bin,
    }


def test_fresh_archive_joiner_closes_activation_archive_and_drivers(
    tmp_path: Path,
) -> None:
    """Break caught: fresh drivers detach from their reviewed repository bytes."""

    from glm52_enforcement.task13_live_inputs import (
        FreshArchiveActivationDriverInputs,
        REQUIRED_EXECUTABLE_PATHS,
        build_cache_seed_driver_request,
        build_fresh_archive_activation_driver_inputs,
        build_h100_driver_request,
    )

    inputs = _fresh_joiner_kwargs(tmp_path)
    result = build_fresh_archive_activation_driver_inputs(**inputs)

    assert isinstance(result, FreshArchiveActivationDriverInputs)
    assert result.activation_id == ACTIVATION
    archive_path = inputs["repository_archive_path"]
    coordinator_path = inputs["coordinator_path"]
    assert isinstance(archive_path, Path)
    assert isinstance(coordinator_path, Path)
    archive_raw = archive_path.read_bytes()
    coordinator_raw = coordinator_path.read_bytes()
    assert result.repository_archive_path == archive_path
    assert result.repository_archive_size_bytes == len(archive_raw)
    assert result.repository_archive_file_sha256 == hashlib.sha256(
        archive_raw
    ).hexdigest()
    assert result.coordinator_path == coordinator_path
    assert result.coordinator_size_bytes == len(coordinator_raw)
    assert result.coordinator_file_sha256 == hashlib.sha256(
        coordinator_raw
    ).hexdigest()
    assert result.required_executable_paths == REQUIRED_EXECUTABLE_PATHS
    assert set(result.repository_executable_file_sha256) == set(
        REQUIRED_EXECUTABLE_PATHS
    )
    assert result.repository_executable_file_sha256[
        "aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py"
    ] == result.coordinator_file_sha256
    assert result.h100_driver_request == build_h100_driver_request(
        activation_id=ACTIVATION,
        campaign_descriptor=inputs["h100_campaign_descriptor"],
    )
    assert (
        result.cache_seed_driver_request
        == build_cache_seed_driver_request(
            activation_id=ACTIVATION,
            campaign_descriptor=inputs["cache_seed_campaign_descriptor"],
            approval=inputs["approval"],
            staged_ready=inputs["staged_ready"],
            staged_ready_version_id=inputs["staged_ready_version_id"],
            rehearsal_evidence=inputs["rehearsal_evidence"],
            task=inputs["task"],
            config=inputs["config"],
            sky_bin=inputs["sky_bin"],
        )
    )
    assert parse_driver_materialization_request(
        result.h100_driver_request
    ) == result.h100_driver_request
    assert parse_driver_materialization_request(
        result.cache_seed_driver_request
    ) == result.cache_seed_driver_request
    assert result.h100_driver_request["activation_id"] == ACTIVATION
    assert result.cache_seed_driver_request["activation_id"] == ACTIVATION

    descriptor = inputs["h100_campaign_descriptor"]
    assert isinstance(descriptor, Path)
    descriptor.write_bytes(b'{"mutated":true}\n')
    with pytest.raises(Task13FixedArtifactError, match="source bytes"):
        parse_driver_materialization_request(result.h100_driver_request)


def test_fresh_archive_joiner_rejects_coordinator_and_inventory_drift(
    tmp_path: Path,
) -> None:
    """Break caught: a stale or incomplete repository can reach fresh drivers."""

    from glm52_enforcement.task13_live_inputs import (
        REQUIRED_EXECUTABLE_PATHS,
        Task13LiveInputError,
        build_fresh_archive_activation_driver_inputs,
    )

    mismatch = _fresh_joiner_kwargs(
        tmp_path / "mismatch",
        archive_coordinator_raw=b"stale coordinator\n",
    )
    with pytest.raises(Task13LiveInputError, match="coordinator"):
        build_fresh_archive_activation_driver_inputs(**mismatch)

    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    missing = _fresh_joiner_kwargs(
        missing_root,
        omit=REQUIRED_EXECUTABLE_PATHS[-1],
    )
    with pytest.raises(Task13LiveInputError, match="missing required executable"):
        build_fresh_archive_activation_driver_inputs(**missing)


@pytest.mark.parametrize(
    "archive_mutation",
    ("unsafe", "duplicate", "symlink", "noncanonical"),
)
def test_fresh_archive_joiner_reuses_safe_tar_policy_and_rejects_member_drift(
    tmp_path: Path,
    archive_mutation: str,
) -> None:
    """Break caught: the fresh join accepts an ambiguous extraction inventory."""

    from glm52_enforcement.task13_live_inputs import (
        REQUIRED_EXECUTABLE_PATHS,
        Task13LiveInputError,
        build_fresh_archive_activation_driver_inputs,
    )

    options: dict[str, object] = {}
    if archive_mutation == "unsafe":
        options["unsafe_member"] = True
    elif archive_mutation == "duplicate":
        options["duplicate"] = REQUIRED_EXECUTABLE_PATHS[0]
    elif archive_mutation == "symlink":
        options["symlink"] = REQUIRED_EXECUTABLE_PATHS[-1]
    else:
        options["noncanonical_member"] = True
    inputs = _fresh_joiner_kwargs(tmp_path, **options)

    with pytest.raises(Task13LiveInputError):
        build_fresh_archive_activation_driver_inputs(**inputs)


def test_fresh_archive_joiner_uses_prequalification_activation_boundary(
    tmp_path: Path,
) -> None:
    """Break caught: activation syntax or Task11 ancestry bypasses packaging."""

    from glm52_enforcement.task13_live_inputs import (
        Task13LiveInputError,
        assemble_prequalification_inputs,
        build_fresh_archive_activation_driver_inputs,
        build_prequalification_base_request,
    )

    malformed = _fresh_joiner_kwargs(tmp_path)
    malformed["activation_id"] = "INVALID_ACTIVATION"
    with pytest.raises(Task13LiveInputError, match="activation_id is invalid"):
        build_fresh_archive_activation_driver_inputs(**malformed)

    coordinates = _prequalification_coordinates()
    base = build_prequalification_base_request(
        activation_id="different-valid-activation",
        collector_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-rehearsal-collector:19"
        ),
        task11_request=_task11_request(),
        task11_boundary=_task11_boundary(),
        staged_infrastructure_evidence=(
            _staged_infrastructure_evidence(coordinates)
        ),
        retained_stack_id=RETAINED_STACK_ID,
        fence_change_set_name="glm52-task13-fence-disabled-0001",
        support_change_set_name="glm52-task13-support-disabled-0001",
        monitor_descriptor_path=str(
            (tmp_path / "production/campaign-descriptor-v2.json").resolve()
        ),
    )
    with pytest.raises(ValueError, match="task11_request activation drifted"):
        assemble_prequalification_inputs(
            base_request=base,
            coordinate_documents=coordinates,
        )


def _fresh_prequalification_fixture(
    tmp_path: Path,
) -> dict[str, object]:
    from glm52_enforcement.task13_fixed_artifacts import (
        build_repository_driver_manifest,
    )
    from glm52_enforcement.task13_live_inputs import (
        build_fresh_archive_activation_driver_inputs,
    )

    source_arguments = _fresh_joiner_kwargs(tmp_path)
    fresh_inputs = build_fresh_archive_activation_driver_inputs(
        **source_arguments
    )
    archive_body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_task13_repository_archive_manifest_v2",
        "account_id": "246813579024",
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "archive_file_sha256": (
            fresh_inputs.repository_archive_file_sha256
        ),
        "archive_size_bytes": fresh_inputs.repository_archive_size_bytes,
        "archive": {
            "bucket": BUCKET,
            "key": (
                f"campaigns/{RUN_ID}/repository/keep-"
                f"{fresh_inputs.repository_archive_file_sha256}.tar.gz"
            ),
            "version_id": "archive-payload-version-0001",
            "file_sha256": fresh_inputs.repository_archive_file_sha256,
        },
    }
    archive_manifest = {
        **archive_body,
        "canonical_identity_sha256": canonical_sha256(archive_body),
    }
    archive_raw = canonical_json_bytes(archive_manifest) + b"\n"
    archive_coordinate = {
        "artifact_kind": "REPOSITORY_ARCHIVE",
        "bucket": BUCKET,
        "key": f"task13/activations/{ACTIVATION}/archive/repo-tar.json",
        "version_id": "archive-manifest-version-0001",
        "file_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "body_sha256": archive_manifest["canonical_identity_sha256"],
    }
    driver_coordinates = {}
    for operation_kind, artifact_kind, key, request_value in (
        (
            "h100-qualification",
            "H100_QUALIFICATION_INPUT",
            (
                f"task13/activations/{ACTIVATION}/"
                "qualification/h100-input.json"
            ),
            fresh_inputs.h100_driver_request,
        ),
        (
            "qualification-cache-seed",
            "QUALIFICATION_CACHE_SEED_INPUT",
            (
                f"task13/activations/{ACTIVATION}/"
                "qualification/cache-seed-input.json"
            ),
            fresh_inputs.cache_seed_driver_request,
        ),
    ):
        manifest = build_repository_driver_manifest(request_value)
        raw = canonical_json_bytes(manifest) + b"\n"
        driver_coordinates[operation_kind] = {
            "artifact_kind": artifact_kind,
            "bucket": BUCKET,
            "key": key,
            "version_id": operation_kind + "-version-0001",
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "body_sha256": manifest["canonical_identity_sha256"],
        }
    return {
        "source_arguments": source_arguments,
        "fresh_inputs": fresh_inputs,
        "archive_manifest": archive_manifest,
        "archive_raw": archive_raw,
        "archive_coordinate": archive_coordinate,
        "h100_coordinate": driver_coordinates["h100-qualification"],
        "cache_seed_coordinate": driver_coordinates[
            "qualification-cache-seed"
        ],
    }


def test_fresh_coordinates_bind_recomputed_archive_and_driver_bytes(
    tmp_path: Path,
) -> None:
    from glm52_enforcement.task13_live_inputs import (
        build_fresh_archive_prequalification_coordinates,
    )

    fixture = _fresh_prequalification_fixture(tmp_path)
    result = build_fresh_archive_prequalification_coordinates(
        fresh_inputs=fixture["fresh_inputs"],
        repository_archive_coordinate=fixture["archive_coordinate"],
        repository_archive_manifest=fixture["archive_manifest"],
        repository_archive_manifest_bytes=fixture["archive_raw"],
        h100_driver_coordinate=fixture["h100_coordinate"],
        cache_seed_driver_coordinate=fixture["cache_seed_coordinate"],
    )

    assert [row["artifact_kind"] for row in result] == [
        "H100_QUALIFICATION_INPUT",
        "QUALIFICATION_CACHE_SEED_INPUT",
        "REPOSITORY_ARCHIVE",
    ]
    assert result == sorted(result, key=lambda row: row["artifact_kind"])


@pytest.mark.parametrize(
    "mutation",
    ("activation", "driver-hash", "mutable-version"),
)
def test_fresh_coordinates_reject_lineage_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    from glm52_enforcement.task13_live_inputs import (
        Task13LiveInputError,
        build_fresh_archive_prequalification_coordinates,
    )

    fixture = _fresh_prequalification_fixture(tmp_path)
    if mutation == "activation":
        manifest = json.loads(canonical_json_bytes(fixture["archive_manifest"]))
        manifest["activation_id"] = "approved-20260729"
        unsigned = dict(manifest)
        unsigned.pop("canonical_identity_sha256")
        manifest["canonical_identity_sha256"] = canonical_sha256(unsigned)
        fixture["archive_manifest"] = manifest
        fixture["archive_raw"] = canonical_json_bytes(manifest) + b"\n"
    elif mutation == "driver-hash":
        coordinate = dict(fixture["h100_coordinate"])
        coordinate["body_sha256"] = "f" * 64
        fixture["h100_coordinate"] = coordinate
    else:
        coordinate = dict(fixture["cache_seed_coordinate"])
        coordinate["version_id"] = "null"
        fixture["cache_seed_coordinate"] = coordinate

    with pytest.raises(Task13LiveInputError):
        build_fresh_archive_prequalification_coordinates(
            fresh_inputs=fixture["fresh_inputs"],
            repository_archive_coordinate=fixture["archive_coordinate"],
            repository_archive_manifest=fixture["archive_manifest"],
            repository_archive_manifest_bytes=fixture["archive_raw"],
            h100_driver_coordinate=fixture["h100_coordinate"],
            cache_seed_driver_coordinate=fixture["cache_seed_coordinate"],
        )


def test_fresh_coordinate_builder_writes_one_canonical_create_only_file(
    tmp_path: Path,
) -> None:
    fixture = _fresh_prequalification_fixture(tmp_path)
    paths = {}
    for label, value in (
        ("archive-coordinate", fixture["archive_coordinate"]),
        ("archive-manifest", fixture["archive_manifest"]),
        ("h100-coordinate", fixture["h100_coordinate"]),
        ("cache-seed-coordinate", fixture["cache_seed_coordinate"]),
    ):
        path = (tmp_path / (label + ".json")).resolve()
        _write_canonical(path, value)
        paths[label] = path
    output = (tmp_path / "fresh-coordinates.json").resolve()
    script = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/scripts/"
        "build_glm52_fresh_archive_prequalification_coordinates.py"
    )
    source = fixture["source_arguments"]
    command = [
        sys.executable,
        str(script),
        "--activation-id",
        str(source["activation_id"]),
        "--repository-archive",
        str(source["repository_archive_path"]),
        "--coordinator",
        str(source["coordinator_path"]),
        "--h100-campaign-descriptor",
        str(source["h100_campaign_descriptor"]),
        "--cache-seed-campaign-descriptor",
        str(source["cache_seed_campaign_descriptor"]),
        "--approval",
        str(source["approval"]),
        "--staged-ready",
        str(source["staged_ready"]),
        "--staged-ready-version-id",
        str(source["staged_ready_version_id"]),
        "--rehearsal-evidence",
        str(source["rehearsal_evidence"]),
        "--task",
        str(source["task"]),
        "--config",
        str(source["config"]),
        "--sky-bin",
        str(source["sky_bin"]),
        "--repository-archive-coordinate",
        str(paths["archive-coordinate"]),
        "--repository-archive-manifest",
        str(paths["archive-manifest"]),
        "--h100-driver-coordinate",
        str(paths["h100-coordinate"]),
        "--cache-seed-driver-coordinate",
        str(paths["cache-seed-coordinate"]),
        "--output",
        str(output),
    ]

    first = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    raw = output.read_bytes()
    assert raw == canonical_json_bytes(json.loads(raw)) + b"\n"
    assert output.stat().st_mode & 0o777 == 0o600
    assert subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    ).returncode == 64
    assert output.read_bytes() == raw


def _load_fresh_driver_cli() -> object:
    script = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/scripts/"
        "build_glm52_task13_fresh_activation_driver_inputs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "glm52_task13_fresh_activation_driver_inputs_test",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_driver_pair_completes_short_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_fresh_driver_cli()
    original_write = module.os.write
    calls = 0

    def short_write(descriptor: int, value: bytes) -> int:
        nonlocal calls
        calls += 1
        return original_write(descriptor, value[:3])

    monkeypatch.setattr(module.os, "write", short_write)
    h100 = (tmp_path / "h100.json").resolve()
    cache_seed = (tmp_path / "cache-seed.json").resolve()

    module._write_new_pair(
        h100_path=h100,
        h100_value={"driver": "h100"},
        cache_seed_path=cache_seed,
        cache_seed_value={"driver": "cache-seed"},
    )

    assert calls > 2
    assert h100.read_bytes() == canonical_json_bytes(
        {"driver": "h100"}
    ) + b"\n"
    assert cache_seed.read_bytes() == canonical_json_bytes(
        {"driver": "cache-seed"}
    ) + b"\n"


def test_fresh_driver_pair_removes_outputs_after_midwrite_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_fresh_driver_cli()
    h100 = (tmp_path / "h100.json").resolve()
    cache_seed = (tmp_path / "cache-seed.json").resolve()

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(module.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="synthetic fsync failure"):
        module._write_new_pair(
            h100_path=h100,
            h100_value={"driver": "h100"},
            cache_seed_path=cache_seed,
            cache_seed_value={"driver": "cache-seed"},
        )

    assert not h100.exists()
    assert not cache_seed.exists()


def test_fresh_driver_pair_closes_descriptor_when_fchmod_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_fresh_driver_cli()
    original_open = module.os.open
    original_close = module.os.close
    opened: list[int] = []
    closed: list[int] = []

    def capture_open(*args: object, **kwargs: object) -> int:
        descriptor = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def capture_close(descriptor: int) -> None:
        closed.append(descriptor)
        original_close(descriptor)

    def fail_fchmod(_descriptor: int, _mode: int) -> None:
        raise OSError("synthetic fchmod failure")

    monkeypatch.setattr(module.os, "open", capture_open)
    monkeypatch.setattr(module.os, "close", capture_close)
    monkeypatch.setattr(module.os, "fchmod", fail_fchmod)
    h100 = (tmp_path / "h100.json").resolve()
    cache_seed = (tmp_path / "cache-seed.json").resolve()

    try:
        with pytest.raises(OSError, match="synthetic fchmod failure"):
            module._write_new_pair(
                h100_path=h100,
                h100_value={"driver": "h100"},
                cache_seed_path=cache_seed,
                cache_seed_value={"driver": "cache-seed"},
            )
        assert closed == opened
    finally:
        for descriptor in set(opened) - set(closed):
            original_close(descriptor)

    assert not h100.exists()
    assert not cache_seed.exists()


def test_fresh_driver_pair_rolls_back_after_first_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_fresh_driver_cli()
    original_open = module.os.open
    original_close = module.os.close
    opened: list[int] = []
    closed: list[int] = []

    def capture_open(*args: object, **kwargs: object) -> int:
        descriptor = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def fail_first_close(descriptor: int) -> None:
        original_close(descriptor)
        closed.append(descriptor)
        if len(closed) == 1:
            raise OSError("synthetic close failure")

    monkeypatch.setattr(module.os, "open", capture_open)
    monkeypatch.setattr(module.os, "close", fail_first_close)
    h100 = (tmp_path / "h100.json").resolve()
    cache_seed = (tmp_path / "cache-seed.json").resolve()

    try:
        with pytest.raises(OSError, match="synthetic close failure"):
            module._write_new_pair(
                h100_path=h100,
                h100_value={"driver": "h100"},
                cache_seed_path=cache_seed,
                cache_seed_value={"driver": "cache-seed"},
            )
        assert closed == opened
    finally:
        for descriptor in set(opened) - set(closed):
            original_close(descriptor)

    assert not h100.exists()
    assert not cache_seed.exists()


def test_fresh_archive_driver_cli_writes_only_two_files_create_only(
    tmp_path: Path,
) -> None:
    """Break caught: the fresh join leaks a composite or replaces driver truth."""

    inputs = _fresh_joiner_kwargs(tmp_path)
    output_dir = tmp_path / "fresh-driver-outputs"
    output_dir.mkdir()
    h100_output = (output_dir / "h100-driver-request.json").resolve()
    cache_output = (output_dir / "cache-seed-driver-request.json").resolve()
    script = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/scripts/"
        "build_glm52_task13_fresh_activation_driver_inputs.py"
    )
    command = [
        sys.executable,
        str(script),
        "--activation-id",
        str(inputs["activation_id"]),
        "--repository-archive",
        str(inputs["repository_archive_path"]),
        "--coordinator",
        str(inputs["coordinator_path"]),
        "--h100-campaign-descriptor",
        str(inputs["h100_campaign_descriptor"]),
        "--cache-seed-campaign-descriptor",
        str(inputs["cache_seed_campaign_descriptor"]),
        "--approval",
        str(inputs["approval"]),
        "--staged-ready",
        str(inputs["staged_ready"]),
        "--staged-ready-version-id",
        str(inputs["staged_ready_version_id"]),
        "--rehearsal-evidence",
        str(inputs["rehearsal_evidence"]),
        "--task",
        str(inputs["task"]),
        "--config",
        str(inputs["config"]),
        "--sky-bin",
        str(inputs["sky_bin"]),
        "--h100-output",
        str(h100_output),
        "--cache-seed-output",
        str(cache_output),
    ]

    first = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    assert {path.name for path in output_dir.iterdir()} == {
        h100_output.name,
        cache_output.name,
    }
    before = {}
    for path, operation_kind in (
        (h100_output, "h100-qualification"),
        (cache_output, "qualification-cache-seed"),
    ):
        raw = path.read_bytes()
        value = json.loads(raw)
        assert raw == canonical_json_bytes(value) + b"\n"
        assert value["record_type"] == (
            "glm52_task13_driver_materialization_request_v2"
        )
        assert value["operation_kind"] == operation_kind
        assert value["activation_id"] == ACTIVATION
        assert parse_driver_materialization_request(value) == value
        assert path.stat().st_mode & 0o777 == 0o600
        before[path] = raw

    second = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 64
    assert {path: path.read_bytes() for path in before} == before


def test_prequalification_builder_closes_base_and_21_coordinate_sources(
    tmp_path: Path,
) -> None:
    """Break caught: package inputs omit, duplicate, or add a coordinate."""

    from glm52_enforcement.task13_live_inputs import (
        assemble_prequalification_inputs,
        build_prequalification_base_request,
    )

    coordinates = _prequalification_coordinates()
    base = build_prequalification_base_request(
        activation_id=ACTIVATION,
        collector_version_arn=(
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-rehearsal-collector:19"
        ),
        task11_request=_task11_request(),
        task11_boundary=_task11_boundary(),
        staged_infrastructure_evidence=(
            _staged_infrastructure_evidence(coordinates)
        ),
        retained_stack_id=RETAINED_STACK_ID,
        fence_change_set_name="glm52-task13-fence-disabled-0001",
        support_change_set_name="glm52-task13-support-disabled-0001",
        monitor_descriptor_path=(
            str((tmp_path / "production/campaign-descriptor-v2.json").resolve())
        ),
    )
    documents: list[object] = [
        coordinates[:2],
        *coordinates[2:],
    ]

    result = assemble_prequalification_inputs(
        base_request=base,
        coordinate_documents=documents,
    )

    expected_kinds = sorted(PREQUALIFICATION_KEYS)
    assert list(base) == [
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "profile",
        "run_id",
        "activation_id",
        "collector_version_arn",
        "task11_request",
        "task11_boundary",
        "staged_infrastructure_evidence",
        "retained_stack_id",
        "fence_stack_name",
        "support_stack_name",
        "fence_change_set_name",
        "support_change_set_name",
        "retained_role_arn",
        "fence_role_arn",
        "support_role_arn",
        "monitor_descriptor_path",
    ]
    assert [row["artifact_kind"] for row in result["coordinates"]] == (
        expected_kinds
    )
    assert result["request"] == {
        **base,
        "artifacts": result["coordinates"],
    }
    assert result["reviewed_artifacts"] == result["coordinates"]

    duplicate = [*documents, coordinates[-1]]
    with pytest.raises(ValueError, match="artifact kinds are not exact"):
        assemble_prequalification_inputs(
            base_request=base,
            coordinate_documents=duplicate,
        )


def test_prequalification_source_cli_emits_all_outputs_exclusively(
    tmp_path: Path,
) -> None:
    """Break caught: the 21-way CLI leaves partial or replaceable inputs."""

    task11_request = (tmp_path / "task11-request.json").resolve()
    task11_boundary = (tmp_path / "task11-boundary.json").resolve()
    _write_canonical(task11_request, _task11_request())
    _write_canonical(task11_boundary, _task11_boundary())
    coordinates = _prequalification_coordinates()
    staged_evidence = (
        tmp_path / "staged-infrastructure-evidence.json"
    ).resolve()
    _write_canonical(
        staged_evidence,
        _staged_infrastructure_evidence(coordinates),
    )
    coordinate_paths = []
    gate_pair = (tmp_path / "gate-pair.json").resolve()
    _write_canonical(gate_pair, coordinates[:2])
    coordinate_paths.append(gate_pair)
    for index, coordinate in enumerate(coordinates[2:], 1):
        path = (tmp_path / f"coordinate-{index:02d}.json").resolve()
        _write_canonical(path, coordinate)
        coordinate_paths.append(path)
    outputs = {
        name: (tmp_path / f"{name}.json").resolve()
        for name in (
            "base",
            "coordinates",
            "request",
            "reviewed",
        )
    }
    script = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/scripts/"
        "build_glm52_task13_prequalification_sources.py"
    )
    command = [
        sys.executable,
        str(script),
        "--activation-id",
        ACTIVATION,
        "--collector-version-arn",
        (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-rehearsal-collector:19"
        ),
        "--task11-request",
        str(task11_request),
        "--task11-boundary",
        str(task11_boundary),
        "--staged-infrastructure-evidence",
        str(staged_evidence),
        "--retained-stack-id",
        RETAINED_STACK_ID,
        "--fence-change-set-name",
        "glm52-task13-fence-disabled-0001",
        "--support-change-set-name",
        "glm52-task13-support-disabled-0001",
        "--monitor-descriptor-path",
        str((tmp_path / "production/campaign-descriptor-v2.json").resolve()),
        *[
            argument
            for path in coordinate_paths
            for argument in ("--coordinate-source", str(path))
        ],
        "--base-request-output",
        str(outputs["base"]),
        "--coordinates-output",
        str(outputs["coordinates"]),
        "--request-output",
        str(outputs["request"]),
        "--reviewed-artifacts-output",
        str(outputs["reviewed"]),
    ]

    first = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    for path in outputs.values():
        raw = path.read_bytes()
        assert raw == canonical_json_bytes(json.loads(raw)) + b"\n"
        assert path.stat().st_mode & 0o777 == 0o600
    request = json.loads(outputs["request"].read_bytes())
    assert len(request["artifacts"]) == 21
    assert json.loads(outputs["coordinates"].read_bytes()) == (
        json.loads(outputs["reviewed"].read_bytes())
    )

    before = {name: path.read_bytes() for name, path in outputs.items()}
    second = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 64
    assert {name: path.read_bytes() for name, path in outputs.items()} == before


def test_task10_materialization_manifest_binds_committed_h100_result(
    tmp_path: Path,
) -> None:
    """Break caught: Task10 materialization guesses the current H100 marker."""

    from glm52_enforcement.task13_live_inputs import (
        build_task10_materialization_manifest,
        validate_task10_materialization_manifest,
    )

    assembled, package = _prequalification_bundle(tmp_path)
    stage_result, journal_rows = _h100_stage_result(package)
    post_h100 = _post_h100_task10_coordinates()
    repo_root = Path(__file__).resolve().parents[1]

    manifest = build_task10_materialization_manifest(
        prequalification_package=package,
        prequalification_reviewed_artifacts=assembled[
            "reviewed_artifacts"
        ],
        task10_task_inputs=post_h100[2],
        task10_worker_descriptor=post_h100[1],
        h100_stage_result=stage_result,
        campaign_journal_rows=journal_rows,
        repository_root=repo_root,
    )

    assert validate_task10_materialization_manifest(
        manifest,
        repository_root=repo_root,
    ) == manifest
    assert manifest["h100_resume_ready"] == {
        "bucket": BUCKET,
        "key": (
            f"campaigns/{RUN_ID}/qualification/"
            "H100_RESUME_READY.json"
        ),
        "version_id": "3LgH100ResumeReadyVersion",
        "file_sha256": "a" * 64,
        "body_sha256": "b" * 64,
    }
    assert set(manifest["approval_coordinates"]) == {
        "GPU_SPEND_APPROVAL",
        "SUPPORT_APPROVAL",
        "RESIDUAL_LIABILITY_APPROVAL",
    }
    assert manifest["task10_task_inputs"]["artifact_kind"] == (
        "TASK10_TASK_INPUTS"
    )
    assert manifest["task10_worker_descriptor"]["artifact_kind"] == (
        "TASK10_WORKER_DESCRIPTOR"
    )
    assert manifest["reviewed_artifacts_without_task10_authority"] == (
        package["reviewed_artifacts"]
    )
    assert len(package["reviewed_artifacts"]) == 21
    assert "CLEAN_REHEARSAL" not in {
        row["artifact_kind"] for row in package["reviewed_artifacts"]
    }

    drifted = json.loads(canonical_json_bytes(journal_rows))
    drifted[-1]["result_identity_sha256"] = "f" * 64
    drifted_body = {
        key: value
        for key, value in drifted[-1].items()
        if key != "canonical_identity_sha256"
    }
    drifted[-1]["canonical_identity_sha256"] = canonical_sha256(
        drifted_body
    )
    with pytest.raises(ValueError, match="stage result identity"):
        build_task10_materialization_manifest(
            prequalification_package=package,
            prequalification_reviewed_artifacts=assembled[
                "reviewed_artifacts"
            ],
            task10_task_inputs=post_h100[2],
            task10_worker_descriptor=post_h100[1],
            h100_stage_result=stage_result,
            campaign_journal_rows=drifted,
            repository_root=repo_root,
        )


def _clean_successor_inputs() -> dict[str, object]:
    from glm52_enforcement.task13_clean_rehearsal import (
        clean_rehearsal_coordinate,
    )
    from test_glm52_task13_campaign_package import (
        _staged_evidence,
        request,
    )
    from test_glm52_task13_clean_rehearsal import _build, _fixture

    fixture = _fixture()
    evidence = _build(fixture)
    prequalification_request = request()
    prequalification_request["artifacts"] = [
        fixture["repository_archive_coordinate"]
        if row["artifact_kind"] == "REPOSITORY_ARCHIVE"
        else row
        for row in prequalification_request["artifacts"]
    ]
    prequalification_request["staged_infrastructure_evidence"] = (
        _staged_evidence(prequalification_request["artifacts"])
    )
    clean_coordinate = clean_rehearsal_coordinate(
        evidence=evidence,
        version_id="clean-evidence-version-0001",
    )
    production_reviewed = [
        *fixture["predecessor_reviewed_artifacts"],
        clean_coordinate,
    ]
    production_reviewed.extend(_post_h100_task10_coordinates())
    production_reviewed.sort(key=lambda row: row["artifact_kind"])
    return {
        "prequalification_request": prequalification_request,
        "prequalification_package": fixture["predecessor_package"],
        "prequalification_reviewed_artifacts": fixture[
            "predecessor_reviewed_artifacts"
        ],
        "production_reviewed_artifacts": production_reviewed,
        "clean_rehearsal_evidence": evidence,
    }


def test_task10_manifest_rejects_clean_as_prequalification(
    tmp_path: Path,
) -> None:
    from glm52_enforcement.task13_live_inputs import (
        build_task10_materialization_manifest,
        validate_task10_materialization_manifest,
    )

    assembled, package = _prequalification_bundle(tmp_path)
    stage_result, journal_rows = _h100_stage_result(package)
    task10 = _post_h100_task10_coordinates()
    root = Path(__file__).resolve().parents[1]
    manifest = build_task10_materialization_manifest(
        prequalification_package=package,
        prequalification_reviewed_artifacts=assembled["reviewed_artifacts"],
        task10_task_inputs=task10[2],
        task10_worker_descriptor=task10[1],
        h100_stage_result=stage_result,
        campaign_journal_rows=journal_rows,
        repository_root=root,
    )
    clean = next(
        row
        for row in _clean_successor_inputs()[
            "production_reviewed_artifacts"
        ]
        if row["artifact_kind"] == "CLEAN_REHEARSAL"
    )
    reviewed = json.loads(
        canonical_json_bytes(
            manifest["reviewed_artifacts_without_task10_authority"]
        )
    )
    reviewed[0] = clean
    reviewed.sort(key=lambda row: row["artifact_kind"])
    manifest["reviewed_artifacts_without_task10_authority"] = reviewed

    with pytest.raises(ValueError, match="prequalification reviewed set"):
        validate_task10_materialization_manifest(
            manifest,
            repository_root=root,
        )


def test_production_successor_adds_clean_and_three_task10_coordinates(
) -> None:
    from glm52_enforcement.task13_live_inputs import (
        build_production_successor_request,
    )

    inputs = _clean_successor_inputs()
    result = build_production_successor_request(**inputs)

    assert result == {
        **inputs["prequalification_request"],
        "artifacts": inputs["production_reviewed_artifacts"],
        "staged_infrastructure_evidence": (
            derive_staged_infrastructure_successor_evidence(
                inputs["prequalification_request"][
                    "staged_infrastructure_evidence"
                ],
                inputs["production_reviewed_artifacts"],
            )
        ),
    }
    assert len(result["artifacts"]) == 25
    predecessor_by_kind = {
        row["artifact_kind"]: row
        for row in inputs["prequalification_reviewed_artifacts"]
    }
    successor_by_kind = {
        row["artifact_kind"]: row for row in result["artifacts"]
    }
    assert set(successor_by_kind) - set(predecessor_by_kind) == {
        "CLEAN_REHEARSAL",
        "TASK10_PRODUCTION_AUTHORITY",
        "TASK10_TASK_INPUTS",
        "TASK10_WORKER_DESCRIPTOR",
    }
    assert all(
        successor_by_kind[kind] == coordinate
        for kind, coordinate in predecessor_by_kind.items()
    )


@pytest.mark.parametrize(
    "mutation",
    ("missing-evidence", "unused-evidence", "second-drift", "cross-activation"),
)
def test_production_successor_rejects_unproven_or_extra_drift(
    mutation: str,
) -> None:
    from glm52_enforcement.task13_live_inputs import (
        Task13LiveInputError,
        build_production_successor_request,
    )

    inputs = _clean_successor_inputs()
    if mutation == "missing-evidence":
        inputs["clean_rehearsal_evidence"] = None
    elif mutation == "unused-evidence":
        inputs["production_reviewed_artifacts"] = [
            *inputs["prequalification_reviewed_artifacts"],
            *_post_h100_task10_coordinates(),
            _post_h100_task10_coordinates()[0],
        ]
        inputs["production_reviewed_artifacts"].sort(
            key=lambda row: row["artifact_kind"]
        )
    elif mutation == "second-drift":
        drifted = json.loads(
            canonical_json_bytes(inputs["production_reviewed_artifacts"])
        )
        next(
            row
            for row in drifted
            if row["artifact_kind"] == "ACCEPTED_BASELINE"
        )["body_sha256"] = "e" * 64
        inputs["production_reviewed_artifacts"] = drifted
    else:
        drifted = json.loads(
            canonical_json_bytes(inputs["production_reviewed_artifacts"])
        )
        next(
            row
            for row in drifted
            if row["artifact_kind"] == "CLEAN_REHEARSAL"
        )["key"] = (
            "task13/gates/clean-rehearsal/approved-20260729/"
            "clean-rehearsal.json"
        )
        inputs["production_reviewed_artifacts"] = drifted

    with pytest.raises(Task13LiveInputError):
        build_production_successor_request(**inputs)


def test_task10_manifest_and_production_successor_clis_write_once(
    tmp_path: Path,
) -> None:
    """Break caught: live builders require hand-edited intermediate JSON."""

    assembled, package = _prequalification_bundle(tmp_path)
    stage_result, journal_rows = _h100_stage_result(package)
    post_h100 = _post_h100_task10_coordinates()
    inputs = {
        "prequalification-request": assembled["request"],
        "prequalification-package": package,
        "prequalification-reviewed": assembled["reviewed_artifacts"],
        "h100-stage-result": stage_result,
        "task10-worker-descriptor": post_h100[1],
        "task10-task-inputs": post_h100[2],
    }
    paths = {}
    for name, value in inputs.items():
        path = (tmp_path / f"{name}.json").resolve()
        _write_canonical(path, value)
        paths[name] = path
    journal_path = (tmp_path / "campaign-journal.jsonl").resolve()
    journal_path.write_bytes(
        b"".join(
            canonical_json_bytes(row) + b"\n"
            for row in journal_rows
        )
    )
    manifest_output = (tmp_path / "task10-manifest.json").resolve()
    manifest_script = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/scripts/"
        "build_glm52_task10_materialization_manifest.py"
    )
    manifest_command = [
        sys.executable,
        str(manifest_script),
        "--prequalification-package",
        str(paths["prequalification-package"]),
        "--prequalification-reviewed-artifacts",
        str(paths["prequalification-reviewed"]),
        "--task10-task-inputs",
        str(paths["task10-task-inputs"]),
        "--task10-worker-descriptor",
        str(paths["task10-worker-descriptor"]),
        "--h100-stage-result",
        str(paths["h100-stage-result"]),
        "--campaign-journal",
        str(journal_path),
        "--output",
        str(manifest_output),
    ]

    first_manifest = subprocess.run(
        manifest_command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert first_manifest.returncode == 0, first_manifest.stderr
    assert manifest_output.stat().st_mode & 0o777 == 0o600
    manifest_raw = manifest_output.read_bytes()
    assert manifest_raw == canonical_json_bytes(
        json.loads(manifest_raw)
    ) + b"\n"
    assert subprocess.run(
        manifest_command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    ).returncode == 64
    assert manifest_output.read_bytes() == manifest_raw

    successor_inputs = _clean_successor_inputs()
    successor_paths = {}
    for label, field in (
        ("prequalification-request", "prequalification_request"),
        ("prequalification-package", "prequalification_package"),
        (
            "prequalification-reviewed",
            "prequalification_reviewed_artifacts",
        ),
        ("production-reviewed", "production_reviewed_artifacts"),
        ("clean-evidence", "clean_rehearsal_evidence"),
    ):
        path = (tmp_path / (label + "-successor.json")).resolve()
        _write_canonical(path, successor_inputs[field])
        successor_paths[label] = path
    successor_output = (tmp_path / "production-request.json").resolve()
    successor_script = (
        Path(__file__).resolve().parents[1]
        / "aws/glm52-gpu/scripts/"
        "build_glm52_task13_production_successor_request.py"
    )
    successor_command = [
        sys.executable,
        str(successor_script),
        "--prequalification-request",
        str(successor_paths["prequalification-request"]),
        "--prequalification-package",
        str(successor_paths["prequalification-package"]),
        "--prequalification-reviewed-artifacts",
        str(successor_paths["prequalification-reviewed"]),
        "--production-reviewed-artifacts",
        str(successor_paths["production-reviewed"]),
        "--clean-rehearsal-evidence",
        str(successor_paths["clean-evidence"]),
        "--output",
        str(successor_output),
    ]

    first_successor = subprocess.run(
        successor_command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert first_successor.returncode == 0, first_successor.stderr
    assert successor_output.stat().st_mode & 0o777 == 0o600
    successor_raw = successor_output.read_bytes()
    assert successor_raw == canonical_json_bytes(
        json.loads(successor_raw)
    ) + b"\n"
    assert len(json.loads(successor_raw)["artifacts"]) == 25
    assert subprocess.run(
        successor_command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    ).returncode == 64
    assert successor_output.read_bytes() == successor_raw
