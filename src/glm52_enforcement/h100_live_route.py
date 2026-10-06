"""Executable guarded route for the real H100 recovery qualification.

This module deliberately owns no AWS or SkyPilot mutation API.  It can only
invoke the repository's sealed qualification submitter, first to prepare the
reviewed live authority and then to acquire and launch the Managed Job.
Source-worker termination remains exclusively owned by the deployed
qualification watchdog.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .canonical import canonical_json_bytes
from .glm52_sky_campaign import validate_sky_campaign_descriptor
from mlx_vq.quality.glm52_sky_submission_lifecycle import (
    submission_intent_s3_key,
)
from mlx_vq.quality.glm52_sky_submission_live_authority import (
    controller_baseline_s3_key,
    must_start_control_plane_ready_s3_key,
)

APPROVED_ACCOUNT_ID = "246813579024"
APPROVED_REGION = "us-west-2"
APPROVED_PROFILE = "keep-gpu"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GUARDED_SUBMITTER = (
    _REPO_ROOT / "aws/glm52-gpu/scripts/submit_sky_campaign.sh"
)
_GUARDED_SUBMITTER_SHA256 = (
    "33a4520edeef8cf907fc637fd3119fba227f9f9cf81c41ca1b1d5a8abbf791c3"
)

_HANDOFF_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "bucket",
        "run_id",
        "managed_mode",
        "descriptor_s3_uri",
        "descriptor_file_sha256",
        "intent_s3_uri",
        "intent_file_sha256",
        "intent_body_sha256",
        "controller_baseline_s3_uri",
        "controller_baseline_file_sha256",
        "controller_baseline_body_sha256",
        "must_start_ready_s3_uri",
        "must_start_ready_file_sha256",
        "must_start_ready_body_sha256",
        "requires_separate_active_deployment",
        "created_at",
        "handoff_body_sha256",
    }
)


class GuardedQualificationError(RuntimeError):
    """The sealed qualification route did not establish a launch."""


@dataclass(frozen=True)
class QualificationAuthorities:
    """Exact local files consumed by the guarded qualification submitter."""

    descriptor: Path
    seed_descriptor: Path
    approval: Path
    staged_ready: Path
    rehearsal_evidence: Path
    cache_seed_accepted: Path
    gpu_spend_snapshot: Path
    qualification_ready: Path
    task: Path
    config: Path


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _require_regular(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise GuardedQualificationError(
            f"{label} must be a regular non-symlink file"
        )
    return path.resolve()


def _validated_authorities(
    authorities: QualificationAuthorities,
) -> QualificationAuthorities:
    return QualificationAuthorities(
        **{
            field: _require_regular(
                getattr(authorities, field),
                label=field.replace("_", " "),
            )
            for field in QualificationAuthorities.__dataclass_fields__
        }
    )


def _canonical_mapping(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    raw = _require_regular(path, label=label).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GuardedQualificationError(f"{label} is not JSON") from error
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise GuardedQualificationError(
            f"{label} is not canonical JSON plus LF"
        )
    return dict(value), raw


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _workspace_files(workspace: Path, name: str) -> tuple[Path, ...]:
    root = workspace.resolve()
    if workspace.is_symlink() or not root.is_dir():
        raise GuardedQualificationError(
            "qualification workspace must be a regular directory"
        )
    candidates = []
    for path in root.rglob(name):
        if (
            path.is_symlink()
            or not path.is_file()
            or not _is_beneath(path, root)
        ):
            continue
        candidates.append(path.resolve())
    return tuple(sorted(set(candidates)))


def _file_by_sha256(
    *,
    workspace: Path,
    name: str,
    expected_sha256: object,
    label: str,
) -> Path:
    if (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise GuardedQualificationError(f"{label} identity is invalid")
    winners = [
        path
        for path in _workspace_files(workspace, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
    ]
    if not winners:
        raise GuardedQualificationError(
            f"exact {label} is absent from the qualification workspace"
        )
    # Duplicate paths with byte-identical authority do not create an authority
    # ambiguity; the bytes, not an ambient pathname, are the reviewed pin.
    return winners[0]


def _matching_readiness(
    *,
    workspace: Path,
    descriptor: Mapping[str, object],
    descriptor_raw: bytes,
) -> tuple[Path, dict[str, object]]:
    expected = {
        "schema_version": 1,
        "record_type": "glm52_qualification_submission_ready_v1",
        "account_id": APPROVED_ACCOUNT_ID,
        "region": APPROVED_REGION,
        "run_id": descriptor["run_id"],
        "managed_mode": "qualification",
        "descriptor_key": descriptor["campaign_descriptor_key"],
        "descriptor_file_sha256": hashlib.sha256(descriptor_raw).hexdigest(),
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "approval_sha256": descriptor["approval_sha256"],
    }
    matches: list[tuple[Path, dict[str, object], bytes]] = []
    for path in _workspace_files(
        workspace,
        "QUALIFICATION_SUBMISSION_READY.json",
    ):
        try:
            value, raw = _canonical_mapping(
                path,
                label="qualification submission readiness",
            )
        except GuardedQualificationError:
            continue
        if all(value.get(field) == wanted for field, wanted in expected.items()):
            matches.append((path, value, raw))
    if not matches:
        raise GuardedQualificationError(
            "no qualification readiness binds the exact campaign descriptor"
        )
    unique_raw = {raw for _path, _value, raw in matches}
    if len(unique_raw) != 1:
        raise GuardedQualificationError(
            "multiple distinct readiness authorities bind the campaign descriptor"
        )
    path, value, _raw = matches[0]
    return path, value


def _materialize_seed_descriptor(
    *,
    readiness: Mapping[str, object],
    directory: Path,
) -> Path:
    value = readiness.get("seed_descriptor")
    if type(value) is not dict:
        raise GuardedQualificationError(
            "qualification readiness omits the seed descriptor"
        )
    raw = canonical_json_bytes(value) + b"\n"
    expected = readiness.get("seed_descriptor_file_sha256")
    if (
        type(expected) is not str
        or hashlib.sha256(raw).hexdigest() != expected
    ):
        raise GuardedQualificationError(
            "qualification seed descriptor identity drifted"
        )
    if directory.exists() or directory.is_symlink():
        raise GuardedQualificationError(
            "authority materialization directory must not already exist"
        )
    directory.mkdir(parents=True, mode=0o700)
    path = directory / "seed-campaign-descriptor-v2.json"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path.resolve()


def resolve_qualification_authorities(
    *,
    descriptor: Path,
    config: Path,
    repo_root: Path,
    workspace: Path,
    materialization_dir: Path,
) -> QualificationAuthorities:
    """Resolve local reviewed pins and materialize the embedded seed descriptor."""

    descriptor_path = _require_regular(
        descriptor,
        label="campaign descriptor",
    )
    if not _is_beneath(descriptor_path, workspace):
        raise GuardedQualificationError(
            "campaign descriptor is outside the qualification workspace"
        )
    descriptor_value, descriptor_raw = _canonical_mapping(
        descriptor_path,
        label="campaign descriptor",
    )
    try:
        descriptor_value = validate_sky_campaign_descriptor(descriptor_value)
    except (TypeError, ValueError) as error:
        raise GuardedQualificationError(
            "campaign descriptor is not an approved GLM-5.2 descriptor"
        ) from error
    readiness_path, readiness = _matching_readiness(
        workspace=workspace,
        descriptor=descriptor_value,
        descriptor_raw=descriptor_raw,
    )
    resolved = QualificationAuthorities(
        descriptor=descriptor_path,
        seed_descriptor=_materialize_seed_descriptor(
            readiness=readiness,
            directory=materialization_dir,
        ),
        approval=_file_by_sha256(
            workspace=workspace,
            name="GPU_SPEND_APPROVAL.json",
            expected_sha256=descriptor_value["approval_sha256"],
            label="GPU spend approval",
        ),
        staged_ready=_file_by_sha256(
            workspace=workspace,
            name="STAGED_CONTROL_PLANE_READY.json",
            expected_sha256=readiness.get("staged_readiness_file_sha256"),
            label="staged readiness",
        ),
        rehearsal_evidence=_file_by_sha256(
            workspace=workspace,
            name="GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json",
            expected_sha256=readiness.get("rehearsal_evidence_sha256"),
            label="rehearsal evidence",
        ),
        cache_seed_accepted=_file_by_sha256(
            workspace=workspace,
            name="QUALIFICATION_CACHE_SEED_ACCEPTED.json",
            expected_sha256=readiness.get(
                "cache_seed_acceptance_file_sha256"
            ),
            label="cache-seed acceptance",
        ),
        gpu_spend_snapshot=_file_by_sha256(
            workspace=workspace,
            name="GPU_SPEND_SNAPSHOT.json",
            expected_sha256=readiness.get("gpu_spend_snapshot_sha256"),
            label="GPU spend snapshot",
        ),
        qualification_ready=readiness_path,
        task=_require_regular(
            repo_root / "aws/glm52-gpu/skypilot/glm52-campaign.yaml",
            label="repository Sky task",
        ),
        config=_require_regular(config, label="pinned SkyPilot config"),
    )
    return _validated_authorities(resolved)


def _load_handoff(
    path: Path,
    *,
    descriptor: Mapping[str, object],
    descriptor_raw: bytes,
) -> dict[str, object]:
    raw = _require_regular(path, label="qualification handoff").read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GuardedQualificationError(
            "qualification handoff is not JSON"
        ) from error
    if (
        type(value) is not dict
        or raw != canonical_json_bytes(value) + b"\n"
        or set(value) != _HANDOFF_FIELDS
    ):
        raise GuardedQualificationError(
            "qualification handoff is not the exact canonical schema"
        )
    unsigned = dict(value)
    identity = unsigned.pop("handoff_body_sha256")
    descriptor_file_sha256 = hashlib.sha256(descriptor_raw).hexdigest()
    bucket = descriptor.get("bucket")
    run_id = descriptor.get("run_id")
    descriptor_key = descriptor.get("campaign_descriptor_key")
    if type(bucket) is not str or type(run_id) is not str or type(descriptor_key) is not str:
        raise GuardedQualificationError(
            "qualification descriptor coordinates are malformed"
        )
    for field in (
        "intent_file_sha256",
        "intent_body_sha256",
        "controller_baseline_file_sha256",
        "controller_baseline_body_sha256",
        "must_start_ready_file_sha256",
        "must_start_ready_body_sha256",
    ):
        candidate = value.get(field)
        if (
            type(candidate) is not str
            or len(candidate) != 64
            or any(character not in "0123456789abcdef" for character in candidate)
        ):
            raise GuardedQualificationError(
                "qualification handoff " + field + " is invalid"
            )
    expected_uris = {
        "descriptor_s3_uri": f"s3://{bucket}/{descriptor_key}",
        "intent_s3_uri": (
            f"s3://{bucket}/"
            + submission_intent_s3_key(
                run_id=run_id,
                managed_mode="qualification",
                intent_body_sha256=str(value["intent_body_sha256"]),
            )
        ),
        "controller_baseline_s3_uri": (
            f"s3://{bucket}/"
            + controller_baseline_s3_key(
                run_id=run_id,
                baseline_body_sha256=str(
                    value["controller_baseline_body_sha256"]
                ),
            )
        ),
        "must_start_ready_s3_uri": (
            f"s3://{bucket}/"
            + must_start_control_plane_ready_s3_key(
                run_id=run_id,
                managed_mode="qualification",
                intent_body_sha256=str(value["intent_body_sha256"]),
                control_plane_ready_body_sha256=str(
                    value["must_start_ready_body_sha256"]
                ),
            )
        ),
    }
    if (
        value["schema_version"] != 1
        or value["record_type"] != "glm52_sky_submission_handoff_v1"
        or value["account_id"] != APPROVED_ACCOUNT_ID
        or value["region"] != APPROVED_REGION
        or value["bucket"] != bucket
        or value["run_id"] != run_id
        or value["managed_mode"] != "qualification"
        or any(value[field] != expected for field, expected in expected_uris.items())
        or value["descriptor_file_sha256"] != descriptor_file_sha256
        or value["requires_separate_active_deployment"] is not True
        or identity != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    ):
        raise GuardedQualificationError(
            "qualification handoff authority is foreign or malformed"
        )
    for field in (
        "descriptor_s3_uri",
        "intent_s3_uri",
        "controller_baseline_s3_uri",
        "must_start_ready_s3_uri",
    ):
        if (
            type(value[field]) is not str
            or not str(value[field]).startswith("s3://")
        ):
            raise GuardedQualificationError(
                f"qualification handoff {field} is invalid"
            )
    return dict(value)


def _validated_submitter(path: Path) -> Path:
    submitter = _require_regular(path, label="guarded submitter")
    expected = _GUARDED_SUBMITTER.resolve()
    if (
        submitter != expected
        or hashlib.sha256(submitter.read_bytes()).hexdigest()
        != _GUARDED_SUBMITTER_SHA256
    ):
        raise GuardedQualificationError(
            "guarded submitter is not the pinned repository executable"
        )
    return submitter


def _common_argv(
    *,
    submitter: Path,
    action: str,
    authorities: QualificationAuthorities,
    sky_bin: Path,
    profile: str,
) -> list[str]:
    return [
        str(submitter),
        "--qualification",
        action,
        "--profile",
        profile,
        "--descriptor",
        str(authorities.descriptor),
        "--seed-descriptor",
        str(authorities.seed_descriptor),
        "--approval",
        str(authorities.approval),
        "--staged-ready",
        str(authorities.staged_ready),
        "--rehearsal-evidence",
        str(authorities.rehearsal_evidence),
        "--cache-seed-accepted",
        str(authorities.cache_seed_accepted),
        "--gpu-spend-snapshot",
        str(authorities.gpu_spend_snapshot),
        "--qualification-ready",
        str(authorities.qualification_ready),
        "--task",
        str(authorities.task),
        "--config",
        str(authorities.config),
        "--sky-bin",
        str(sky_bin),
    ]


def _run(
    runner: Runner,
    argv: list[str],
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            shell=False,
        )
    except OSError as error:
        raise GuardedQualificationError(
            "guarded qualification submitter could not execute"
        ) from error


def run_guarded_qualification(
    *,
    authorities: QualificationAuthorities,
    submitter: Path,
    sky_bin: Path,
    profile: str,
    work_dir: Path,
    runner: Runner = subprocess.run,
) -> dict[str, object]:
    """Prepare and acquire exactly one guarded qualification Managed Job."""

    if profile != APPROVED_PROFILE:
        raise GuardedQualificationError(
            f"profile must be exactly {APPROVED_PROFILE}"
        )
    submitter = _validated_submitter(submitter)
    sky_bin = _require_regular(sky_bin, label="pinned SkyPilot executable")
    exact = _validated_authorities(authorities)
    descriptor, descriptor_raw = _canonical_mapping(
        exact.descriptor,
        label="campaign descriptor",
    )
    try:
        descriptor = validate_sky_campaign_descriptor(descriptor)
    except (TypeError, ValueError) as error:
        raise GuardedQualificationError(
            "campaign descriptor is not an approved GLM-5.2 descriptor"
        ) from error
    if work_dir.exists() or work_dir.is_symlink():
        raise GuardedQualificationError(
            "qualification route work directory must not already exist"
        )
    work_dir.mkdir(parents=True, mode=0o700)
    handoff_path = work_dir / "qualification-submission-handoff.json"
    prepare_argv = _common_argv(
        submitter=submitter,
        action="prepare-intent",
        authorities=exact,
        sky_bin=sky_bin,
        profile=profile,
    )
    prepare_argv.extend(["--output-handoff", str(handoff_path)])
    prepared = _run(runner, prepare_argv)
    if prepared.returncode == 78:
        raise GuardedQualificationError(
            "deployed qualification supervisor is not positively ready"
        )
    if prepared.returncode != 0:
        detail = (prepared.stderr or prepared.stdout).strip()
        raise GuardedQualificationError(
            "guarded qualification preparation failed"
            + (f": {detail}" if detail else "")
        )
    handoff = _load_handoff(
        handoff_path,
        descriptor=descriptor,
        descriptor_raw=descriptor_raw,
    )
    acquire_argv = _common_argv(
        submitter=submitter,
        action="acquire-and-launch",
        authorities=exact,
        sky_bin=sky_bin,
        profile=profile,
    )
    acquire_argv.extend(
        [
            "--intent-s3-uri",
            str(handoff["intent_s3_uri"]),
            "--must-start-ready-s3-uri",
            str(handoff["must_start_ready_s3_uri"]),
        ]
    )
    for attempt in range(3):
        acquired = _run(runner, acquire_argv)
        if acquired.returncode == 0:
            return {
                "status": "accepted",
                "attempts": attempt + 1,
                "intent_s3_uri": handoff["intent_s3_uri"],
                "must_start_ready_s3_uri": handoff[
                    "must_start_ready_s3_uri"
                ],
            }
        if acquired.returncode != 75:
            detail = (acquired.stderr or acquired.stdout).strip()
            raise GuardedQualificationError(
                "guarded qualification acquisition failed"
                + (f": {detail}" if detail else "")
            )
    raise GuardedQualificationError(
        "guarded qualification reconciliation remained pending after two retries"
    )
