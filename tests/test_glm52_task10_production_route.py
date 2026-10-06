"""Task 10 guarded production submission-route contracts."""

from __future__ import annotations

import base64
import importlib.util
from dataclasses import asdict, replace
from datetime import UTC, datetime
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.task10_production import (
    ProductionRequest,
    ProductionRouteError,
    assert_no_raw_effect_surface,
    build_execution_input_identity,
    build_production_authority,
    run_production,
)
from glm52_enforcement.task10_worker import (
    H100_READY_KEY,
    MountFreeTaskInputs,
    build_worker_bootstrap_descriptor,
    build_jobs_launch_body,
    render_mount_free_task,
    worker_bootstrap_descriptor_from_mapping,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = REPO_ROOT / "aws/glm52-gpu/scripts/submit_sky_campaign.py"


def _submitter() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "_glm52_task10_submitter_under_test",
        SUBMITTER,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _authority_materializer() -> ModuleType:
    path = (
        REPO_ROOT
        / "aws/glm52-gpu/scripts/materialize_glm52_task10_authority.py"
    )
    specification = importlib.util.spec_from_file_location(
        "_glm52_task10_authority_materializer_under_test",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _launch_bridge_builder() -> ModuleType:
    path = (
        REPO_ROOT
        / "aws/glm52-gpu/scripts/build_glm52_task13_launch_bridge.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_glm52_task13_launch_bridge_builder_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_mode_has_genuine_guarded_validate_start_and_reconcile_route(
    tmp_path: Path,
) -> None:
    """The public submitter must expose the frozen production vocabulary."""

    module = _submitter()
    descriptor = tmp_path / "descriptor.json"

    for action in ("validate-only", "start", "reconcile"):
        parsed = module._parser().parse_args(
            [
                "production",
                action,
                "--profile",
                "keep-gpu",
                "--descriptor",
                str(descriptor),
            ]
        )
        assert parsed.mode == "production"
        assert parsed.action == action


def _task_inputs(descriptor_sha256: str = "1" * 64) -> MountFreeTaskInputs:
    return MountFreeTaskInputs(
        job_name="glm52-sky-20260724",
        descriptor_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/descriptor.json"
        ),
        descriptor_version_id="descriptor-version-opaque-1",
        descriptor_file_sha256=descriptor_sha256,
        approval_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/authorities/approval.json"
        ),
        approval_version_id="approval-version-opaque-1",
        approval_file_sha256="2" * 64,
        intent_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/submissions/production/intent.json"
        ),
        intent_version_id="intent-version-opaque-1",
        intent_file_sha256="3" * 64,
        intent_body_sha256="4" * 64,
        repository_archive_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/repository/repo.tar.gz"
        ),
        repository_archive_version_id="archive-version-opaque-1",
        repository_archive_file_sha256="5" * 64,
    )


def _authority(
    task_inputs: MountFreeTaskInputs | None = None,
    *,
    descriptor_identity_sha256: str = "7" * 64,
):
    inputs = task_inputs or _task_inputs()
    task_yaml = render_mount_free_task(inputs)
    body = build_jobs_launch_body(
        task_yaml=task_yaml,
        job_name=inputs.job_name,
    )
    common = {
        "schema_version": 1,
        "record_type": "glm52_task10_production_authority_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "managed_mode": "production",
        "campaign_identity_sha256": "6" * 64,
        "activation_id": "activation-0001",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "sky_job_name": inputs.job_name,
        "execution_deadline": "2026-07-31T12:00:00Z",
        "gpu_allocation_sha256": "8" * 64,
        "executor_epoch": 1,
        "action_key": "ACTIVATION#activation-0001#ACTION#SKY_POST#00000001",
        "action_state": "CONSUMED",
        "current_activation": True,
        "descriptor_identity_sha256": descriptor_identity_sha256,
        "archive_identity_sha256": (
            inputs.repository_archive_file_sha256
        ),
        "repository_archive_version_id": (
            inputs.repository_archive_version_id
        ),
        "approval_identity_sha256": inputs.approval_file_sha256,
        "approval_version_id": inputs.approval_version_id,
        "intent_identity_sha256": inputs.intent_body_sha256,
        "intent_version_id": inputs.intent_version_id,
        "task11_boundary_bucket": "keep-glm52-us-west-2-246813579024",
        "task11_boundary_key": (
            "campaigns/glm52-sky-20260724/authorities/task11/"
            "activation-0001/00000001.json"
        ),
        "task11_boundary_version_id": "task11-boundary-version-1",
        "task11_boundary_file_sha256": "9" * 64,
        "task11_boundary_body_sha256": "a" * 64,
        "h100_resume_ready_bucket": (
            "keep-glm52-models-246813579024-us-west-2"
        ),
        "h100_resume_ready_key": H100_READY_KEY,
        "h100_resume_ready_version_id": "h100-version-opaque-1",
        "h100_resume_ready_file_sha256": "b" * 64,
        "h100_resume_ready_body_sha256": "c" * 64,
        "capacity_retry_availability_zones": [
            "us-west-2" + letter for letter in "abcdef"
        ],
        "capacity_retry_outcome_rules": [
            {
                "availability_zone": "us-west-2" + letter,
                "on_capacity_failure": (
                    "ROTATE_NEXT_AZ" if index < 6
                    else "STOP_CAPACITY_EXHAUSTED"
                ),
            }
            for index, letter in enumerate("abcdef", 1)
        ],
        "maximum_capacity_attempts": 6,
        "same_token_identity_sha256": "d" * 64,
        "stop_on_first_worker_success": True,
        "automatic_spot_fallback": False,
        "capacity_block_allowed": False,
        "task8_live_h1d_identity_sha256": "d" * 64,
        "task8_spend_authority_identity_sha256": "e" * 64,
        "task8_spend_reserve_identity_sha256": "f" * 64,
        "task9_launch_identity_sha256": "0" * 64,
        "task9_admission_identity_sha256": "1" * 64,
        "task9_custody_identity_sha256": "2" * 64,
        "attestation_identity_sha256": "3" * 64,
        "decision_seal_identity_sha256": "4" * 64,
        "relay_envelope_sha256": "5" * 64,
        "task_yaml_sha256": hashlib.sha256(task_yaml.encode()).hexdigest(),
        "request_body_sha256": canonical_sha256(body),
        "workflow_version_arn": (
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-production:7"
        ),
        "execution_name": "activation-0001-epoch-00000001",
        "expected_execution_arn": (
            "arn:aws:states:us-west-2:246813579024:execution:"
            "keep-glm52-h1g-production:activation-0001-epoch-00000001"
        ),
        "task13_route_binding_identity_sha256": "e" * 64,
    }
    placeholder = build_production_authority(
        **common,
        execution_input_sha256="a" * 64,
    )
    input_sha = build_execution_input_identity(placeholder, inputs)
    return build_production_authority(
        **common,
        execution_input_sha256=input_sha,
    )


def _main_inputs():
    preliminary_inputs = _task_inputs()
    preliminary_authority = _authority(preliminary_inputs)
    wrapper = build_worker_bootstrap_descriptor(
        schema_version=1,
        record_type="glm52_task10_worker_bootstrap_descriptor_v1",
        account_id="246813579024",
        region="us-west-2",
        run_id="glm52-sky-20260724",
        campaign_identity_sha256=(
            preliminary_authority.campaign_identity_sha256
        ),
        activation_id=preliminary_authority.activation_id,
        activation_ordinal=preliminary_authority.activation_ordinal,
        generation=preliminary_authority.generation,
        generation_text=preliminary_authority.generation_text,
        action_key=preliminary_authority.action_key,
        sky_job_name=preliminary_inputs.job_name,
        execution_deadline="2026-07-31T12:00:00Z",
        gpu_allocation_sha256=(
            preliminary_authority.gpu_allocation_sha256
        ),
        base_descriptor_s3_uri=(
            "s3://keep-glm52-us-west-2-246813579024/"
            "campaigns/glm52-sky-20260724/base-descriptor.json"
        ),
        base_descriptor_version_id="base-descriptor-version-1",
        base_descriptor_file_sha256="6" * 64,
        base_descriptor_body_sha256="7" * 64,
        archive_identity_sha256=(
            preliminary_authority.archive_identity_sha256
        ),
        repository_archive_version_id=(
            preliminary_authority.repository_archive_version_id
        ),
        approval_identity_sha256=(
            preliminary_authority.approval_identity_sha256
        ),
        approval_version_id=preliminary_authority.approval_version_id,
        intent_identity_sha256=preliminary_authority.intent_identity_sha256,
        intent_version_id=preliminary_authority.intent_version_id,
        task8_live_h1d_identity_sha256=(
            preliminary_authority.task8_live_h1d_identity_sha256
        ),
        task8_spend_authority_identity_sha256=(
            preliminary_authority.task8_spend_authority_identity_sha256
        ),
        task9_launch_identity_sha256=(
            preliminary_authority.task9_launch_identity_sha256
        ),
        task9_admission_identity_sha256=(
            preliminary_authority.task9_admission_identity_sha256
        ),
        task9_custody_identity_sha256=(
            preliminary_authority.task9_custody_identity_sha256
        ),
    )
    wrapper_raw = (
        json.dumps(
            asdict(wrapper),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    inputs = _task_inputs(hashlib.sha256(wrapper_raw).hexdigest())
    authority = _authority(
        inputs,
        descriptor_identity_sha256=wrapper.descriptor_body_sha256,
    )
    return inputs, authority, wrapper_raw


def _request(action: str = "start") -> ProductionRequest:
    inputs, authority, wrapper_raw = _main_inputs()
    wrapper = worker_bootstrap_descriptor_from_mapping(
        json.loads(wrapper_raw)
    )
    return ProductionRequest(
        action=action,
        profile="keep-gpu",
        authority=authority,
        task_inputs=inputs,
        worker_descriptor=wrapper,
    )


def _rebuild_authority(
    authority,
    inputs: MountFreeTaskInputs,
    **changes: object,
):
    task_yaml = render_mount_free_task(inputs)
    body = build_jobs_launch_body(
        task_yaml=task_yaml,
        job_name=inputs.job_name,
    )
    values = asdict(authority)
    values.pop("canonical_identity_sha256")
    values.update(changes)
    values["task_yaml_sha256"] = hashlib.sha256(
        task_yaml.encode()
    ).hexdigest()
    values["request_body_sha256"] = canonical_sha256(body)
    values["execution_input_sha256"] = "a" * 64
    placeholder = build_production_authority(**values)
    values["execution_input_sha256"] = build_execution_input_identity(
        placeholder,
        inputs,
    )
    return build_production_authority(**values)


class _Boundary:
    def __init__(self, *, classification: str = "STARTED") -> None:
        self.classification = classification
        self.inspect_calls = 0
        self.start_calls = 0
        self.reconcile_calls = 0

    def inspect(self, authority, execution_input):
        del execution_input
        self.inspect_calls += 1
        body = {
            "current_activation": True,
            "action_consumed": True,
            "task8_live_authority": (
                authority.task8_live_h1d_identity_sha256
            ),
            "task8_spend_authority": (
                authority.task8_spend_authority_identity_sha256
            ),
            "task8_spend_reserve": (
                authority.task8_spend_reserve_identity_sha256
            ),
            "task9_launch_admission": (
                authority.task9_admission_identity_sha256
            ),
            "task9_liability_custody": (
                authority.task9_custody_identity_sha256
            ),
            "workflow_version_arn": authority.workflow_version_arn,
        }
        return {
            **body,
            "inspection_identity_sha256": canonical_sha256(body),
        }

    def start_once(self, authority, execution_input):
        del execution_input
        self.start_calls += 1
        body = {
            "classification": self.classification,
            "execution_arn": authority.expected_execution_arn,
            "reserve_identity_sha256": (
                authority.task8_spend_reserve_identity_sha256
            ),
            "task9_custody_identity_sha256": (
                authority.task9_custody_identity_sha256
            ),
        }
        return {**body, "observation_identity_sha256": canonical_sha256(body)}

    def reconcile(self, authority, execution_input):
        del execution_input
        self.reconcile_calls += 1
        body = {
            "classification": "RUNNING",
            "execution_arn": authority.expected_execution_arn,
            "capacity_outcomes": [
                {
                    "attempt": 1,
                    "availability_zone": "us-west-2a",
                    "outcome": "CAPACITY_REJECTED",
                }
            ],
            "instance_id": None,
            "availability_zone": None,
            "workflow_output": {
                "bucket": authority.h100_resume_ready_bucket,
                "key": "campaigns/glm52-sky-20260724/production/output.json",
                "version_id": "workflow-output-version-1",
                "file_sha256": "a" * 64,
                "body_sha256": "b" * 64,
            },
            "same_token_identity_sha256": authority.same_token_identity_sha256,
            "reserve_identity_sha256": (
                authority.task8_spend_reserve_identity_sha256
            ),
            "spend_authority_identity_sha256": (
                authority.task8_spend_authority_identity_sha256
            ),
            "action_identity_sha256": authority.task9_launch_identity_sha256,
            "task9_custody_identity_sha256": (
                authority.task9_custody_identity_sha256
            ),
        }
        return {**body, "observation_identity_sha256": canonical_sha256(body)}


def test_validate_only_authenticates_but_cannot_reserve_or_post() -> None:
    boundary = _Boundary()
    request = _request("validate-only")

    result = run_production(request, boundary=boundary)

    assert result.status == "validated"
    assert boundary.inspect_calls == 1
    assert boundary.start_calls == 0
    assert boundary.reconcile_calls == 0


@pytest.mark.parametrize(
    ("classification", "status"),
    [
        ("STARTED", "started"),
        ("KNOWN_REJECTED", "known-rejected"),
        ("AMBIGUOUS", "reconcile-required"),
    ],
)
def test_start_uses_one_guarded_boundary_call_and_never_replays(
    classification: str,
    status: str,
) -> None:
    boundary = _Boundary(classification=classification)
    request = _request("start")

    result = run_production(request, boundary=boundary)

    assert result.status == status
    assert boundary.start_calls == 1
    assert boundary.reconcile_calls == 0


def test_reconcile_observes_same_execution_without_start_or_reserve() -> None:
    boundary = _Boundary()
    request = _request("reconcile")
    authority = request.authority

    result = run_production(
        request,
        boundary=boundary,
    )

    assert result.status == "reconcile-required"
    assert result.detail["expected_execution_arn"] == (
        authority.expected_execution_arn
    )
    assert boundary.start_calls == 0
    assert boundary.reconcile_calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_state", "ARMED"),
        ("current_activation", False),
        ("h100_resume_ready_key", "H100_RESUME_READY.json"),
        ("h100_resume_ready_version_id", "null"),
        ("managed_mode", "qualification"),
        ("task8_spend_reserve_identity_sha256", "x" * 64),
        ("task9_custody_identity_sha256", "x" * 64),
    ],
)
def test_foreign_stale_missing_and_mutated_authority_fails_before_start(
    field: str,
    value: object,
) -> None:
    boundary = _Boundary()
    request = _request("start")
    authority = replace(request.authority, **{field: value})

    with pytest.raises(ProductionRouteError):
        run_production(
            replace(request, authority=authority),
            boundary=boundary,
        )

    assert boundary.inspect_calls == 0
    assert boundary.start_calls == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"campaign_identity_sha256": "a" * 64},
        {"activation_id": "activation-foreign"},
        {"activation_ordinal": 2},
        {"generation": 2, "generation_text": "00000002"},
        {"action_key": "ACTION#00000002#SKY_POST#00000002"},
        {"execution_deadline": "2026-07-31T11:59:59Z"},
        {"gpu_allocation_sha256": "9" * 64},
        {"archive_identity_sha256": "a" * 64},
        {"approval_identity_sha256": "b" * 64},
        {"intent_identity_sha256": "c" * 64},
        {"task8_live_h1d_identity_sha256": "e" * 64},
        {"task8_spend_authority_identity_sha256": "f" * 64},
        {"task9_launch_identity_sha256": "f" * 64},
        {"task9_admission_identity_sha256": "a" * 64},
        {"task9_custody_identity_sha256": "b" * 64},
    ],
)
def test_foreign_worker_wrapper_identity_fails_before_inspect_or_start(
    changes: dict[str, object],
) -> None:
    request = _request("start")
    wrapper_values = asdict(request.worker_descriptor)
    wrapper_values.pop("descriptor_body_sha256")
    wrapper_values.update(changes)
    mutant = build_worker_bootstrap_descriptor(**wrapper_values)
    mutant_raw = (
        json.dumps(
            asdict(mutant),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    inputs = replace(
        request.task_inputs,
        descriptor_file_sha256=hashlib.sha256(mutant_raw).hexdigest(),
    )
    authority = _rebuild_authority(
        request.authority,
        inputs,
        descriptor_identity_sha256=mutant.descriptor_body_sha256,
    )
    boundary = _Boundary()

    with pytest.raises(ProductionRouteError):
        run_production(
            replace(
                request,
                authority=authority,
                task_inputs=inputs,
                worker_descriptor=mutant,
            ),
            boundary=boundary,
        )

    assert boundary.inspect_calls == 0
    assert boundary.start_calls == 0


def test_raw_sky_http_subprocess_and_ec2_surface_is_rejected() -> None:
    class RawBoundary(_Boundary):
        def jobs_launch(self) -> None:
            raise AssertionError

    with pytest.raises(ProductionRouteError, match="raw effects"):
        assert_no_raw_effect_surface(RawBoundary())


def test_python_main_writes_one_canonical_nonoverwrite_handoff(
    tmp_path: Path,
) -> None:
    module = _submitter()
    inputs, authority, descriptor_raw = _main_inputs()
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_bytes(descriptor_raw)
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(
        json.dumps(
            asdict(authority),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    inputs_path = tmp_path / "task-inputs.json"
    inputs_path.write_text(
        json.dumps(
            asdict(inputs),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    output = tmp_path / "out.json"
    boundary = _Boundary()
    argv = [
        "production",
        "validate-only",
        "--profile",
        "keep-gpu",
        "--descriptor",
        str(descriptor),
        "--production-authority",
        str(authority_path),
        "--production-task-inputs",
        str(inputs_path),
        "--output-handoff",
        str(output),
    ]

    assert (
        module.main(
            argv,
            production_boundary_factory=lambda **_: boundary,
        )
        == 0
    )
    assert output.read_bytes() == (
        json.dumps(
            json.loads(output.read_bytes()),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    assert (
        module.main(
            argv,
            production_boundary_factory=lambda **_: boundary,
        )
        == 70
    )


@pytest.mark.parametrize(
    "sink_failure",
    (
        "pre-existing",
        "symlink",
        "invalid-parent",
        "open-refused",
        "write-refused",
    ),
)
def test_python_main_refuses_unwritable_outcome_sink_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sink_failure: str,
) -> None:
    """Break caught: a known bad handoff sink still allowed production start."""

    module = _submitter()
    inputs, authority, descriptor_raw = _main_inputs()
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_bytes(descriptor_raw)
    authority_path = tmp_path / "authority.json"
    authority_path.write_bytes(
        canonical_json_bytes(asdict(authority)) + b"\n"
    )
    inputs_path = tmp_path / "task-inputs.json"
    inputs_path.write_bytes(canonical_json_bytes(asdict(inputs)) + b"\n")
    output = tmp_path / "out.json"
    expected_existing: bytes | None = None
    if sink_failure == "pre-existing":
        expected_existing = b"existing outcome\n"
        output.write_bytes(expected_existing)
    elif sink_failure == "symlink":
        target = tmp_path / "existing-target.json"
        expected_existing = b"existing target\n"
        target.write_bytes(expected_existing)
        output.symlink_to(target)
    elif sink_failure == "invalid-parent":
        invalid_parent = tmp_path / "not-a-directory"
        invalid_parent.write_text("regular file\n")
        output = invalid_parent / "out.json"
    elif sink_failure == "open-refused":
        real_open = module.os.open

        def refusing_open(
            path: object,
            flags: int,
            mode: int = 0o777,
        ) -> int:
            if os.fspath(path) == os.fspath(output):
                raise PermissionError("injected outcome sink refusal")
            return real_open(path, flags, mode)

        monkeypatch.setattr(module.os, "open", refusing_open)
    else:
        real_write = module.os.write

        def refusing_write(descriptor: int, raw: bytes) -> int:
            if raw == b"\n":
                raise OSError("injected outcome sink write refusal")
            return real_write(descriptor, raw)

        monkeypatch.setattr(module.os, "write", refusing_write)
    boundary = _Boundary()

    result = module.main(
        [
            "production",
            "start",
            "--profile",
            "keep-gpu",
            "--descriptor",
            str(descriptor),
            "--production-authority",
            str(authority_path),
            "--production-task-inputs",
            str(inputs_path),
            "--output-handoff",
            str(output),
        ],
        production_boundary_factory=lambda **_: boundary,
    )

    assert result == 70
    assert boundary.inspect_calls == 0
    assert boundary.start_calls == 0
    assert boundary.reconcile_calls == 0
    if sink_failure == "pre-existing":
        assert output.read_bytes() == expected_existing
    elif sink_failure == "symlink":
        assert output.is_symlink()
        assert output.read_bytes() == expected_existing
    elif sink_failure in {"open-refused", "write-refused"}:
        assert not output.exists()


def test_default_route_uses_one_sha_pinned_task13_launch_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-injected route can only accept one Task 13 workflow start."""

    module = _submitter()
    inputs, authority, descriptor_raw = _main_inputs()

    def write_canonical(name: str, value: object) -> tuple[Path, bytes, str]:
        path = tmp_path / name
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        path.write_bytes(raw)
        return path, raw, canonical_sha256(value)

    package_path, package_raw, package_body = write_canonical(
        "package.json", {"package": "sealed"}
    )
    reviewed_path, reviewed_raw, reviewed_body = write_canonical(
        "reviewed.json", []
    )
    controller_path, controller_raw, controller_body = write_canonical(
        "controller.json", {"stage": "launch"}
    )
    runner = REPO_ROOT / "aws/glm52-gpu/scripts/run_glm52_task13_campaign.py"
    coordinator = REPO_ROOT / "aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py"
    bridge_body = {
        "schema_version": 1,
        "record_type": "glm52_task10_task13_launch_bridge_v1",
        "account_id": authority.account_id,
        "region": authority.region,
        "profile": "keep-gpu",
        "run_id": authority.run_id,
        "activation_id": authority.activation_id,
        "package_path": str(package_path),
        "package_file_sha256": hashlib.sha256(package_raw).hexdigest(),
        "package_body_sha256": package_body,
        "reviewed_artifacts_path": str(reviewed_path),
        "reviewed_artifacts_file_sha256": hashlib.sha256(reviewed_raw).hexdigest(),
        "reviewed_artifacts_body_sha256": reviewed_body,
        "controller_authority_path": str(controller_path),
        "controller_authority_file_sha256": hashlib.sha256(controller_raw).hexdigest(),
        "controller_authority_body_sha256": controller_body,
        "fence_journal_path": str(tmp_path / "fence-journal.jsonl"),
        "support_journal_path": str(tmp_path / "support-journal.jsonl"),
        "campaign_journal_path": str(tmp_path / "campaign-journal.jsonl"),
        "runner_relative_path": "aws/glm52-gpu/scripts/run_glm52_task13_campaign.py",
        "runner_file_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "coordinator_relative_path": "aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py",
        "coordinator_file_sha256": hashlib.sha256(coordinator.read_bytes()).hexdigest(),
        "h100_resume_ready_bucket": authority.h100_resume_ready_bucket,
        "h100_resume_ready_key": authority.h100_resume_ready_key,
        "h100_resume_ready_version_id": authority.h100_resume_ready_version_id,
        "h100_resume_ready_file_sha256": authority.h100_resume_ready_file_sha256,
        "h100_resume_ready_body_sha256": authority.h100_resume_ready_body_sha256,
        "task13_route_binding_identity_sha256": (
            authority.task13_route_binding_identity_sha256
        ),
    }
    bridge = {
        **bridge_body,
        "canonical_identity_sha256": canonical_sha256(bridge_body),
    }
    bridge_path, _bridge_raw, _bridge_body = write_canonical("bridge.json", bridge)
    authority = _rebuild_authority(
        authority,
        inputs,
        task13_route_binding_identity_sha256=(
            bridge["task13_route_binding_identity_sha256"]
        ),
    )
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_bytes(descriptor_raw)
    authority_path, _authority_raw, _authority_body = write_canonical(
        "authority.json", asdict(authority)
    )
    inputs_path, _inputs_raw, _inputs_body = write_canonical(
        "task-inputs.json", asdict(inputs)
    )
    output = tmp_path / "out.json"
    calls: list[list[str]] = []
    reconciliation_body = {
        "classification": "RUNNING",
        "execution_arn": authority.expected_execution_arn,
        "capacity_outcomes": [
            {
                "attempt": 1,
                "availability_zone": "us-west-2a",
                "outcome": "CAPACITY_REJECTED",
            }
        ],
        "instance_id": None,
        "availability_zone": None,
        "workflow_output": {
            "bucket": authority.h100_resume_ready_bucket,
            "key": "campaigns/glm52-sky-20260724/production/output.json",
            "version_id": "workflow-output-version-1",
            "file_sha256": "a" * 64,
            "body_sha256": "b" * 64,
        },
        "same_token_identity_sha256": authority.same_token_identity_sha256,
        "reserve_identity_sha256": authority.task8_spend_reserve_identity_sha256,
        "spend_authority_identity_sha256": authority.task8_spend_authority_identity_sha256,
        "action_identity_sha256": authority.task9_launch_identity_sha256,
        "task9_custody_identity_sha256": authority.task9_custody_identity_sha256,
    }
    reconciliation = {
        **reconciliation_body,
        "observation_identity_sha256": canonical_sha256(reconciliation_body),
    }

    def run_task13(command, **kwargs):
        del kwargs
        calls.append(list(command))
        result_body = {
            "status": "STAGE_COMMITTED",
            "stage": "launch",
            "accepted_execution_arn": authority.expected_execution_arn,
            "workflow_invocation_count": 1,
            "workflow_reconciliation": reconciliation,
        }
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(result_body, sort_keys=True, separators=(",", ":")).encode() + b"\n",
            b"",
        )

    monkeypatch.setattr(module.subprocess, "run", run_task13)
    assert module.main([
        "production", "start", "--profile", "keep-gpu",
        "--descriptor", str(descriptor), "--production-authority", str(authority_path),
        "--production-task-inputs", str(inputs_path), "--task13-launch-bridge", str(bridge_path),
        "--output-handoff", str(output),
    ]) == 0
    assert len(calls) == 1
    assert calls[0][:3] == ["/usr/bin/python3", "-I", str(runner)]
    assert calls[0][calls[0].index("--stage") + 1] == "launch"
    assert calls[0][calls[0].index("--fence-journal") + 1] == (
        str(tmp_path / "fence-journal.jsonl")
    )
    assert calls[0][calls[0].index("--support-journal") + 1] == (
        str(tmp_path / "support-journal.jsonl")
    )
    assert calls[0][calls[0].index("--campaign-journal") + 1] == (
        str(tmp_path / "campaign-journal.jsonl")
    )
    assert json.loads(output.read_bytes())["status"] == "started"
    reconcile_output = tmp_path / "reconcile.json"
    assert module.main([
        "production", "reconcile", "--profile", "keep-gpu",
        "--descriptor", str(descriptor), "--production-authority", str(authority_path),
        "--production-task-inputs", str(inputs_path), "--task13-launch-bridge", str(bridge_path),
        "--output-handoff", str(reconcile_output),
    ]) == 75
    assert len(calls) == 2
    assert calls[1][calls[1].index("--stage") + 1] == "launch"
    assert json.loads(reconcile_output.read_bytes())["classification"] == "RUNNING"


def test_shell_routes_exact_production_argv_without_sky_client(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "aws/glm52-gpu/scripts"
    scripts.mkdir(parents=True)
    shell = scripts / "submit_sky_campaign.sh"
    shutil.copy2(
        REPO_ROOT / "aws/glm52-gpu/scripts/submit_sky_campaign.sh",
        shell,
    )
    shell.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    for helper_name in (
        "assert_rnd_aws_account.sh",
        "assert_sns_email_confirmed.sh",
    ):
        helper = scripts / helper_name
        helper.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' '"
            + helper_name
            + "' >> \"$TASK10_SHELL_LOG\"\n"
        )
        helper.chmod(0o755)
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['TASK10_SHELL_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    fake_python.chmod(0o755)
    production_authority = tmp_path / "authority.json"
    production_authority.write_text('{"h100":"H100_RESUME_READY.json"}\n')
    args = [
        "--production",
        "start",
        "--profile",
        "keep-gpu",
        "--descriptor",
        "/exact/descriptor.json",
        "--production-authority",
        str(production_authority),
        "--production-task-inputs",
        "/exact/task-inputs.json",
        "--output-handoff",
        "/exact/output.json",
    ]

    result = subprocess.run(
        ["bash", str(shell), *args],
        env={
            **os.environ,
            "KEEP_PRODUCTION_PYTHON": str(fake_python),
            "TASK10_SHELL_LOG": str(log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = log.read_text().splitlines()
    assert lines[:2] == [
        "assert_rnd_aws_account.sh",
        "assert_sns_email_confirmed.sh",
    ]
    assert json.loads(lines[2]) == [
        str(scripts / "submit_sky_campaign.py"),
        "production",
        *args[1:],
    ]
    assert "SKY_BIN" not in (
        REPO_ROOT / "aws/glm52-gpu/scripts/submit_sky_campaign.sh"
    ).read_text().split('if [ "$MODE" = "--production" ]')[1].split(
        'if [ "$MODE" != "--qualification" ]'
    )[0]
    assert shell.stat().st_mode & stat.S_IXUSR


def test_shell_forwards_production_authority_environment_fallback(
    tmp_path: Path,
) -> None:
    """Break caught: the advertised environment fallback was not forwarded."""

    scripts = tmp_path / "aws/glm52-gpu/scripts"
    scripts.mkdir(parents=True)
    shell = scripts / "submit_sky_campaign.sh"
    shutil.copy2(
        REPO_ROOT / "aws/glm52-gpu/scripts/submit_sky_campaign.sh",
        shell,
    )
    shell.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    for helper_name in (
        "assert_rnd_aws_account.sh",
        "assert_sns_email_confirmed.sh",
    ):
        helper = scripts / helper_name
        helper.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' '"
            + helper_name
            + "' >> \"$TASK10_SHELL_LOG\"\n"
        )
        helper.chmod(0o755)
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['TASK10_SHELL_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    fake_python.chmod(0o755)
    authority = tmp_path / "authority.json"
    authority.write_text('{"h100":"H100_RESUME_READY.json"}\n')

    result = subprocess.run(
        [
            "bash",
            str(shell),
            "--production",
            "start",
            "--profile",
            "keep-gpu",
            "--descriptor",
            "/exact/descriptor.json",
            "--production-task-inputs",
            "/exact/task-inputs.json",
            "--output-handoff",
            "/exact/output.json",
        ],
        env={
            **os.environ,
            "GLM52_PRODUCTION_AUTHORITY": str(authority),
            "KEEP_PRODUCTION_PYTHON": str(fake_python),
            "TASK10_SHELL_LOG": str(log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    forwarded = json.loads(log.read_text().splitlines()[2])
    assert forwarded[-2:] == ["--production-authority", str(authority)]


def test_shell_rejects_wrong_production_action_before_any_helper(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "aws/glm52-gpu/scripts"
    scripts.mkdir(parents=True)
    shell = scripts / "submit_sky_campaign.sh"
    shutil.copy2(
        REPO_ROOT / "aws/glm52-gpu/scripts/submit_sky_campaign.sh",
        shell,
    )
    shell.chmod(0o755)
    log = tmp_path / "calls"
    result = subprocess.run(
        ["bash", str(shell), "--production", "acquire-and-launch"],
        env={**os.environ, "TASK10_SHELL_LOG": str(log)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 64
    assert "usage:" in result.stderr
    assert not log.exists()


def test_prepackage_authority_breaks_package_bridge_cycle_from_exact_sources(
) -> None:
    """Break caught: authority identity depends on the not-yet-built package."""

    from glm52_enforcement.task11_boundary import (
        BOUNDARY_INPUT_KINDS,
        build_task11_boundary_document,
        build_task11_input_coordinate,
    )
    from glm52_enforcement.task10_worker import (
        worker_bootstrap_descriptor_from_mapping,
    )
    from glm52_enforcement.task13_authority_materialization import (
        AuthorityMaterializationError,
        attach_task10_authority_coordinate,
        build_task10_authority_from_sources,
        build_task13_route_binding_identity,
    )

    inputs, expected, descriptor_raw = _main_inputs()
    descriptor = worker_bootstrap_descriptor_from_mapping(
        json.loads(descriptor_raw)
    )
    coordinates = tuple(
        build_task11_input_coordinate(
            input_kind=kind,
            bucket=expected.task11_boundary_bucket,
            key=(
                f"campaigns/{expected.run_id}/authorities/task9/"
                f"{expected.activation_id}/TASK9_DEPLOYED_IDENTITY.json"
                if kind == "TASK9_DEPLOYED_IDENTITY_COORDINATE"
                else (
                    f"campaigns/{expected.run_id}/authorities/task11/"
                    f"{expected.activation_id}/{expected.generation_text}/"
                    f"{index:02d}-"
                    + kind.lower().replace("_", "-")
                    + ".json"
                )
            ),
            version_id=f"source-version-{index:02d}",
            file_sha256=hashlib.sha256(
                f"file-{index}".encode()
            ).hexdigest(),
            body_sha256=hashlib.sha256(
                f"body-{index}".encode()
            ).hexdigest(),
        )
        for index, kind in enumerate(BOUNDARY_INPUT_KINDS, 1)
    )
    boundary = build_task11_boundary_document(
        activation_id=expected.activation_id,
        generation=expected.generation,
        campaign_identity_sha256=expected.campaign_identity_sha256,
        state_machine_version_arn=expected.workflow_version_arn,
        action_key=expected.action_key,
        inputs=coordinates,
    )
    boundary_coordinate = {
        "bucket": expected.task11_boundary_bucket,
        "key": expected.task11_boundary_key,
        "version_id": expected.task11_boundary_version_id,
        "file_sha256": expected.task11_boundary_file_sha256,
        "body_sha256": expected.task11_boundary_body_sha256,
    }
    approvals = {
        kind: {
            "artifact_kind": kind,
            "bucket": "approval-bucket",
            "key": kind.lower() + ".json",
            "version_id": kind.lower() + "-version",
            "file_sha256": hashlib.sha256(
                (kind + "-file").encode()
            ).hexdigest(),
            "body_sha256": hashlib.sha256(kind.encode()).hexdigest(),
        }
        for kind in (
            "GPU_SPEND_APPROVAL",
            "SUPPORT_APPROVAL",
            "RESIDUAL_LIABILITY_APPROVAL",
        )
    }
    workflow_inventory = {
        "state_machine_logical_id": "Task10ProductionStateMachine",
        "state_machine_version_logical_id": (
            "Task10ProductionStateMachineVersion"
        ),
        "state_machine_name": "keep-glm52-h1g-production",
        "reconciliation_function_logical_id": (
            "Task10ProductionReconciliationFunction"
        ),
        "reconciliation_function_version_logical_id": (
            "Task10ProductionReconciliationFunctionVersion"
        ),
        "reconciliation_function_name": (
            "keep-glm52-h1g-task10-capacity-reconciliation"
        ),
        "terminal_writer_state": "RunInternalSixAzSoleSender",
        "writer_handler": (
            "aws/glm52-gpu/lambda/"
            "task10_sole_sender_handler.py"
        ),
    }
    route_identity = build_task13_route_binding_identity(
        task11_boundary_coordinate=boundary_coordinate,
        task11_input_coordinates=[asdict(value) for value in coordinates],
        approval_coordinates=approvals,
        workflow_inventory=workflow_inventory,
        runner_file_sha256="a" * 64,
        coordinator_file_sha256="b" * 64,
    )
    source = {
        "activation_ordinal": expected.activation_ordinal,
        "execution_deadline": expected.execution_deadline,
        "gpu_allocation_sha256": expected.gpu_allocation_sha256,
        "executor_epoch": expected.executor_epoch,
        "same_token_identity_sha256": expected.same_token_identity_sha256,
        "task8_live_h1d_identity_sha256": (
            expected.task8_live_h1d_identity_sha256
        ),
        "task8_spend_authority_identity_sha256": (
            expected.task8_spend_authority_identity_sha256
        ),
        "task8_spend_reserve_identity_sha256": (
            expected.task8_spend_reserve_identity_sha256
        ),
        "task9_launch_identity_sha256": (
            expected.task9_launch_identity_sha256
        ),
        "task9_admission_identity_sha256": (
            expected.task9_admission_identity_sha256
        ),
        "task9_custody_identity_sha256": (
            expected.task9_custody_identity_sha256
        ),
        "attestation_identity_sha256": (
            expected.attestation_identity_sha256
        ),
        "decision_seal_identity_sha256": (
            expected.decision_seal_identity_sha256
        ),
        "relay_envelope_sha256": expected.relay_envelope_sha256,
        "qualification_cache_manifest_sha256": "b" * 64,
    }
    h100_marker = _h100_marker(
        campaign_identity_sha256=expected.campaign_identity_sha256,
        repo_tar_sha256=inputs.repository_archive_file_sha256,
        qualification_cache_manifest_sha256="b" * 64,
    )
    built = build_task10_authority_from_sources(
        boundary=boundary,
        source_documents={
            kind: dict(source) for kind in BOUNDARY_INPUT_KINDS
        },
        task_inputs=inputs,
        worker_descriptor=descriptor,
        task11_boundary_coordinate=boundary_coordinate,
        h100_resume_ready_coordinate={
            "bucket": expected.h100_resume_ready_bucket,
            "key": expected.h100_resume_ready_key,
            "version_id": expected.h100_resume_ready_version_id,
            "file_sha256": expected.h100_resume_ready_file_sha256,
            "body_sha256": h100_marker["ready_body_sha256"],
        },
        h100_resume_ready=h100_marker,
        workflow_readback={
            "state_machine_version_arn": expected.workflow_version_arn,
            "state_machine_name": "keep-glm52-h1g-production",
            "status": "ACTIVE",
            "terminal_writer_state": "RunInternalSixAzSoleSender",
            "reconciliation_function_name": (
                "keep-glm52-h1g-task10-capacity-reconciliation"
            ),
        },
        task13_route_binding_identity_sha256=route_identity,
    )
    assert built.task13_route_binding_identity_sha256 == route_identity
    assert built.activation_id == expected.activation_id
    assert built.execution_input_sha256 != "0" * 64
    mutant = dict(h100_marker)
    mutant["campaign_identity_sha256"] = "f" * 64
    mutant_body = dict(mutant)
    mutant_body.pop("ready_body_sha256")
    mutant["ready_body_sha256"] = canonical_sha256(mutant_body)
    with pytest.raises(
        AuthorityMaterializationError,
        match="campaign",
    ):
        build_task10_authority_from_sources(
            boundary=boundary,
            source_documents={
                kind: dict(source) for kind in BOUNDARY_INPUT_KINDS
            },
            task_inputs=inputs,
            worker_descriptor=descriptor,
            task11_boundary_coordinate=boundary_coordinate,
            h100_resume_ready_coordinate={
                "bucket": expected.h100_resume_ready_bucket,
                "key": expected.h100_resume_ready_key,
                "version_id": expected.h100_resume_ready_version_id,
                "file_sha256": expected.h100_resume_ready_file_sha256,
                "body_sha256": mutant["ready_body_sha256"],
            },
            h100_resume_ready=mutant,
            workflow_readback={
                "state_machine_version_arn": expected.workflow_version_arn,
                "state_machine_name": "keep-glm52-h1g-production",
                "status": "ACTIVE",
                "terminal_writer_state": "RunInternalSixAzSoleSender",
                "reconciliation_function_name": (
                    "keep-glm52-h1g-task10-capacity-reconciliation"
                ),
            },
            task13_route_binding_identity_sha256=route_identity,
        )
    reviewed = attach_task10_authority_coordinate(
        [
            {"artifact_kind": "PRODUCTION_DESCRIPTOR"},
            {"artifact_kind": "TASK10_WORKER_DESCRIPTOR"},
        ],
        {
            "artifact_kind": "TASK10_PRODUCTION_AUTHORITY",
            "bucket": "authority-bucket",
            "key": "task13/production/task10-production-authority.json",
            "version_id": "authority-version",
            "file_sha256": "c" * 64,
            "body_sha256": built.canonical_identity_sha256,
        },
    )
    assert [item["artifact_kind"] for item in reviewed] == [
        "PRODUCTION_DESCRIPTOR",
        "TASK10_PRODUCTION_AUTHORITY",
        "TASK10_WORKER_DESCRIPTOR",
    ]


def _h100_marker(
    *,
    campaign_identity_sha256: str = "6" * 64,
    repo_tar_sha256: str = "5" * 64,
    qualification_cache_manifest_sha256: str = "b" * 64,
) -> dict[str, object]:
    from glm52_enforcement.glm52_h100_qualification import (
        build_h100_resume_ready,
    )

    return build_h100_resume_ready(
        run_id="glm52-sky-20260724",
        campaign_identity_sha256=campaign_identity_sha256,
        repo_tar_sha256=repo_tar_sha256,
        qualification_cache_manifest_sha256=(
            qualification_cache_manifest_sha256
        ),
        first_instance_id="i-00000000000000001",
        replacement_instance_id="i-00000000000000002",
        first_allocation_record_sha256="c" * 64,
        replacement_allocation_record_sha256="d" * 64,
        source_checkpoint_marker_sha256="e" * 64,
        parity_report_sha256="f" * 64,
        training_smoke_sha256="1" * 64,
        resumed_capture_sha256="2" * 64,
        peak_gpu_gib=69.0,
        completed_at=datetime(2026, 7, 29, 12, 0, tzinfo=UTC),
    )


class _ExactH100S3:
    def __init__(
        self,
        raw: bytes,
        *,
        version_id: str = "h100-version-opaque-1",
        include_body: bool = True,
        checksum: str | None = None,
        retry_attempts: int | None = 0,
    ) -> None:
        self.raw = raw
        self.version_id = version_id
        self.include_body = include_body
        self.checksum = checksum
        self.retry_attempts = retry_attempts

    def get_object(self, **values: object) -> dict[str, object]:
        assert values == {
            "Bucket": (
                "keep-glm52-models-246813579024-us-west-2"
            ),
            "Key": (
                "campaigns/glm52-sky-20260724/qualification/"
                "H100_RESUME_READY.json"
            ),
            "VersionId": "h100-version-opaque-1",
            "ExpectedBucketOwner": "246813579024",
            "ChecksumMode": "ENABLED",
        }
        response_metadata = {
                "HTTPStatusCode": 200,
                "RequestId": "h100-get-1",
        }
        if self.retry_attempts is not None:
            response_metadata["RetryAttempts"] = self.retry_attempts
        result: dict[str, object] = {
            "ResponseMetadata": response_metadata,
            "VersionId": self.version_id,
            "ChecksumSHA256": (
                self.checksum
                if self.checksum is not None
                else base64.b64encode(
                    hashlib.sha256(self.raw).digest()
                ).decode("ascii")
            ),
        }
        if self.include_body:
            result["Body"] = io.BytesIO(self.raw)
        return result


def _h100_coordinate(
    raw: bytes,
    marker: dict[str, object],
) -> dict[str, object]:
    return {
        "bucket": (
            "keep-glm52-models-246813579024-us-west-2"
        ),
        "key": (
            "campaigns/glm52-sky-20260724/qualification/"
            "H100_RESUME_READY.json"
        ),
        "version_id": "h100-version-opaque-1",
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": marker["ready_body_sha256"],
    }


def test_prepackage_cli_exactly_validates_and_cross_binds_h100_marker() -> None:
    """Break caught: the authority builder trusts an unread H100 coordinate."""

    module = _authority_materializer()
    marker = _h100_marker()
    raw = json.dumps(
        marker,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"
    result = module._read_validated_h100_resume_ready(
        _ExactH100S3(raw),
        coordinate=_h100_coordinate(raw, marker),
        campaign_identity_sha256="6" * 64,
        repository_archive_file_sha256="5" * 64,
        qualification_cache_manifest_sha256="b" * 64,
    )
    assert result == marker


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "body"),
        ("version", "VersionId"),
        ("retry-missing", "VersionId"),
        ("retry-nonzero", "VersionId"),
        ("checksum", "checksum"),
        ("schema", "schema"),
        ("campaign-substitution", "campaign"),
        ("archive-substitution", "archive"),
        ("cache-substitution", "qualification cache"),
    ],
)
def test_prepackage_cli_rejects_h100_marker_mutants(
    mutation: str,
    message: str,
) -> None:
    """Break caught: missing or substituted H100 proof enters Task10 authority."""

    module = _authority_materializer()
    marker = _h100_marker(
        campaign_identity_sha256=(
            "7" * 64
            if mutation == "campaign-substitution"
            else "6" * 64
        ),
        repo_tar_sha256=(
            "8" * 64
            if mutation == "archive-substitution"
            else "5" * 64
        ),
        qualification_cache_manifest_sha256=(
            "9" * 64
            if mutation == "cache-substitution"
            else "b" * 64
        ),
    )
    if mutation == "schema":
        marker["training_sample_steps"] = 3
        body = dict(marker)
        body.pop("ready_body_sha256")
        marker["ready_body_sha256"] = canonical_sha256(body)
    raw = json.dumps(
        marker,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"
    client = _ExactH100S3(
        raw,
        include_body=mutation != "missing",
        version_id=(
            "foreign-version"
            if mutation == "version"
            else "h100-version-opaque-1"
        ),
        checksum=(
            base64.b64encode(b"x" * 32).decode("ascii")
            if mutation == "checksum"
            else None
        ),
        retry_attempts=(
            None
            if mutation == "retry-missing"
            else (1 if mutation == "retry-nonzero" else 0)
        ),
    )
    with pytest.raises(
        module.AuthorityMaterializationError,
        match=message,
    ):
        module._read_validated_h100_resume_ready(
            client,
            coordinate=_h100_coordinate(raw, marker),
            campaign_identity_sha256="6" * 64,
            repository_archive_file_sha256="5" * 64,
            qualification_cache_manifest_sha256="b" * 64,
        )


def test_materializer_emits_complete_public_task10_local_bundle(
    tmp_path: Path,
) -> None:
    """Break caught: public Task10 requires hand-built local bridge inputs."""

    from glm52_enforcement.task13_authority_materialization import (
        validate_public_task10_inputs_manifest,
    )

    module = _authority_materializer()
    task_inputs, authority, descriptor_raw = _main_inputs()
    descriptor = worker_bootstrap_descriptor_from_mapping(
        json.loads(descriptor_raw)
    )
    h100_coordinate = {
        "bucket": authority.h100_resume_ready_bucket,
        "key": authority.h100_resume_ready_key,
        "version_id": authority.h100_resume_ready_version_id,
        "file_sha256": authority.h100_resume_ready_file_sha256,
        "body_sha256": authority.h100_resume_ready_body_sha256,
    }
    authority_coordinate = {
        "artifact_kind": "TASK10_PRODUCTION_AUTHORITY",
        "bucket": authority.h100_resume_ready_bucket,
        "key": "task13/production/task10-production-authority.json",
        "version_id": "task10-authority-version-1",
        "file_sha256": hashlib.sha256(
            canonical_json_bytes(asdict(authority)) + b"\n"
        ).hexdigest(),
        "body_sha256": authority.canonical_identity_sha256,
    }
    output = (tmp_path / "public-task10").resolve()
    output.mkdir()

    manifest = module._write_public_outputs(
        output,
        authority=authority,
        task_inputs=task_inputs,
        worker_descriptor=descriptor,
        h100_resume_ready_coordinate=h100_coordinate,
        task10_authority_coordinate=authority_coordinate,
        task13_route_binding_identity_sha256=(
            authority.task13_route_binding_identity_sha256
        ),
    )

    manifest_path = output / "public-task10-inputs.json"
    assert json.loads(manifest_path.read_bytes()) == manifest
    assert validate_public_task10_inputs_manifest(manifest) == manifest
    assert set(path.name for path in output.iterdir()) == {
        "task10-production-authority.json",
        "task10-task-inputs.json",
        "task10-worker-descriptor.json",
        "h100-resume-ready-coordinate.json",
        "task13-route-binding-identity.json",
        "public-task10-inputs.json",
    }
    assert all(
        path.stat().st_mode & 0o777 == 0o600
        for path in output.iterdir()
    )
    assert manifest["production_authority_path"] == str(
        output / "task10-production-authority.json"
    )
    assert manifest["production_task_inputs_path"] == str(
        output / "task10-task-inputs.json"
    )
    assert manifest["production_descriptor_path"] == str(
        output / "task10-worker-descriptor.json"
    )
    assert manifest["h100_resume_ready_coordinate_path"] == str(
        output / "h100-resume-ready-coordinate.json"
    )
    bridge_builder = _launch_bridge_builder()
    h100, route_identity, loaded_manifest = bridge_builder._public_inputs(
        manifest_path
    )
    assert h100 == h100_coordinate
    assert route_identity == authority.task13_route_binding_identity_sha256
    assert loaded_manifest == manifest
    package_path = (tmp_path / "production-package.json").resolve()
    reviewed_path = (tmp_path / "production-reviewed.json").resolve()
    controller_path = (tmp_path / "controller-authority.json").resolve()
    reviewed = [authority_coordinate]
    package = {
        "activation_id": authority.activation_id,
        "reviewed_artifacts": reviewed,
    }
    package_path.write_bytes(canonical_json_bytes(package) + b"\n")
    reviewed_path.write_bytes(canonical_json_bytes(reviewed) + b"\n")
    controller_path.write_bytes(
        canonical_json_bytes(
            {"activation_id": authority.activation_id}
        )
        + b"\n"
    )
    bridge_path = (tmp_path / "task13-launch-bridge.json").resolve()
    assert bridge_builder.main(
        [
            "--package",
            str(package_path),
            "--reviewed-artifacts",
            str(reviewed_path),
            "--controller-authority",
            str(controller_path),
            "--public-task10-inputs",
            str(manifest_path),
            "--fence-journal",
            str((tmp_path / "fence.jsonl").resolve()),
            "--support-journal",
            str((tmp_path / "support.jsonl").resolve()),
            "--campaign-journal",
            str((tmp_path / "campaign.jsonl").resolve()),
            "--output",
            str(bridge_path),
        ]
    ) == 0
    bridge = json.loads(bridge_path.read_bytes())
    assert bridge["h100_resume_ready_version_id"] == (
        h100_coordinate["version_id"]
    )
    assert bridge["task13_route_binding_identity_sha256"] == (
        authority.task13_route_binding_identity_sha256
    )

    with pytest.raises(
        module.AuthorityMaterializationError,
        match="new empty directory",
    ):
        module._write_public_outputs(
            output,
            authority=authority,
            task_inputs=task_inputs,
            worker_descriptor=descriptor,
            h100_resume_ready_coordinate=h100_coordinate,
            task10_authority_coordinate=authority_coordinate,
            task13_route_binding_identity_sha256=(
                authority.task13_route_binding_identity_sha256
            ),
        )
