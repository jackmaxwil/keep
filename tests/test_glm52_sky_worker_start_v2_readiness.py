"""Pure worker-start-v2 source-readiness authority contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from mlx_vq.quality.glm52_sky_worker_start_v2_readiness import (
    WorkerStartV2ReadinessArtifact,
    WorkerStartV2ReadinessAuthorities,
    WorkerStartV2ReadinessError,
    WorkerStartV2SourceSnapshot,
    build_worker_start_v2_readiness,
    validate_worker_start_v2_readiness,
    worker_start_v2_readiness_file_bytes,
    worker_start_v2_readiness_file_sha256,
    worker_start_v2_readiness_s3_key,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_TEST = REPO_ROOT / "tests/test_glm52_sky_submission_integration.py"

EXPECTED_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "bucket",
    "run_id",
    "managed_mode",
    "descriptor_key",
    "descriptor_file_sha256",
    "descriptor_body_sha256",
    "campaign_identity_sha256",
    "repo_tar_key",
    "repo_tar_sha256",
    "qualification_submission_ready_key",
    "qualification_submission_ready_file_sha256",
    "qualification_submission_ready_body_sha256",
    "rehearsal_evidence_key",
    "rehearsal_evidence_file_sha256",
    "rehearsal_evidence_body_sha256",
    "intent_key",
    "intent_file_sha256",
    "intent_body_sha256",
    "skypilot_version",
    "sky_task_name",
    "sky_task_file_sha256",
    "sky_job_name",
    "must_start_by",
    "instance_type",
    "instance_count",
    "use_spot",
    "image_id",
    "controller_identity",
    "worker_identity",
    "source_contract_sha256",
    "source_closure_sha256",
    "source_files",
    "worker_start_v2_readiness_body_sha256",
}

EXPECTED_SOURCE_FILES = [
    {
        "relative_path": "aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py",
        "file_sha256": (
            "efa7c48ad259e10c6ce0e4cb1db9ff0a6e6a9fa203274329a7c2eb5203366406"
        ),
    },
    {
        "relative_path": "aws/glm52-gpu/skypilot/bootstrap_campaign.sh",
        "file_sha256": (
            "8073ec6e55184073eece202d1726cbb1e7fe83c6f95abee927114b713f4366fc"
        ),
    },
    {
        "relative_path": "aws/glm52-gpu/skypilot/glm52-campaign.yaml",
        "file_sha256": (
            "cc0a56a34684d297e55827cf017802f4e8b16bd8dd0bb18fa1e1c0a197d787e5"
        ),
    },
    {
        "relative_path": "aws/glm52-gpu/skypilot/publish_worker_start_v2.py",
        "file_sha256": (
            "4d1d47aef6dd211c20b52f05ede9bd6d32c35433e305f2b1bd56b48c83b9051e"
        ),
    },
    {
        "relative_path": "aws/glm52-gpu/skypilot/run_h100_qualification.sh",
        "file_sha256": (
            "870d85585ea8037b9e6a92647da585ce537bd15768fa298c330e6894ddd87686"
        ),
    },
    {
        "relative_path": "aws/glm52-gpu/skypilot/run_managed_campaign.sh",
        "file_sha256": (
            "ac2400441f2b6ecac6bf6ad9cc46fba5abb0386e1b2cf6046527f8c3c3ba9a9c"
        ),
    },
    {
        "relative_path": "src/glm52_enforcement/glm52_h100_qualification.py",
        "file_sha256": (
            "f1720b675a2ae84b23be6a34a8bd6b84e0e65cabab260c5a1c79707096c11ff5"
        ),
    },
    {
        "relative_path": "src/glm52_enforcement/glm52_sky_campaign.py",
        "file_sha256": (
            "f023eaeddf8fac73fa6546e3c4a0b60dd32e5266f06e1519fe21e0444d445d03"
        ),
    },
    {
        "relative_path": "src/glm52_enforcement/glm52_sky_must_start.py",
        "file_sha256": (
            "e910d3d03f7b30b7b9b668e214b61ecc7cbd641a53dedabd620b8e14573d33f3"
        ),
    },
    {
        "relative_path": "src/mlx_vq/quality/glm52_h100_qualification.py",
        "file_sha256": (
            "b9418985afdd3f9ef44a3dbf44731cb149a2bc8e17194c0bf9fab9d6a88b329a"
        ),
    },
    {
        "relative_path": "src/mlx_vq/quality/glm52_sky_campaign.py",
        "file_sha256": (
            "77412115f6b8ce325177864418d838e0335d536329e8e7565d7811bcc806cd94"
        ),
    },
    {
        "relative_path": "src/mlx_vq/quality/glm52_sky_must_start.py",
        "file_sha256": (
            "478ac3a16ce8a77a4e844c1e0a3dca99454d50e7233fd7cd67dfa184b486860a"
        ),
    },
    {
        "relative_path": "src/mlx_vq/quality/glm52_sky_must_start_dynamic.py",
        "file_sha256": (
            "527019a41b54f810e9f98734c2ca99acc5a7b370344adf48c80f46f925396e18"
        ),
    },
    {
        "relative_path": "src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py",
        "file_sha256": (
            "8dfad6682d8afc9774dc31385cfc9ecc8c8e2f80d9a535e3c8d4cf6cf0fa97df"
        ),
    },
]


@dataclass(frozen=True)
class ReadinessFixture:
    authorities: WorkerStartV2ReadinessAuthorities
    source_snapshot: WorkerStartV2SourceSnapshot


def _canonical(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return raw + (b"\n" if newline else b"")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _integration_module() -> ModuleType:
    name = "_glm52_source_readiness_integration_fixture"
    if name in sys.modules:
        return sys.modules[name]
    specification = importlib.util.spec_from_file_location(name, INTEGRATION_TEST)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _source_snapshot() -> WorkerStartV2SourceSnapshot:
    return WorkerStartV2SourceSnapshot(
        publisher_raw=(
            REPO_ROOT / "aws/glm52-gpu/skypilot/publish_worker_start_v2.py"
        ).read_bytes(),
        h100_qualification_worker_raw=(
            REPO_ROOT / "aws/glm52-gpu/skypilot/run_h100_qualification.sh"
        ).read_bytes(),
        h100_qualification_native_raw=(
            REPO_ROOT / "src/glm52_enforcement/glm52_h100_qualification.py"
        ).read_bytes(),
        h100_qualification_raw=(
            REPO_ROOT / "src/mlx_vq/quality/glm52_h100_qualification.py"
        ).read_bytes(),
        campaign_policy_raw=(
            REPO_ROOT / "src/mlx_vq/quality/glm52_sky_campaign.py"
        ).read_bytes(),
        campaign_policy_native_raw=(
            REPO_ROOT / "src/glm52_enforcement/glm52_sky_campaign.py"
        ).read_bytes(),
        must_start_policy_raw=(
            REPO_ROOT / "src/mlx_vq/quality/glm52_sky_must_start.py"
        ).read_bytes(),
        must_start_policy_native_raw=(
            REPO_ROOT / "src/glm52_enforcement/glm52_sky_must_start.py"
        ).read_bytes(),
        dynamic_policy_raw=(
            REPO_ROOT / "src/mlx_vq/quality/glm52_sky_must_start_dynamic.py"
        ).read_bytes(),
        worker_policy_raw=(
            REPO_ROOT / "src/mlx_vq/quality/glm52_sky_worker_must_start_v2.py"
        ).read_bytes(),
        coordinator_raw=(
            REPO_ROOT / "aws/glm52-gpu/lambda/sky_worker_start_v2_coordinator.py"
        ).read_bytes(),
        sky_task_raw=(
            REPO_ROOT / "aws/glm52-gpu/skypilot/glm52-campaign.yaml"
        ).read_bytes(),
        bootstrap_raw=(
            REPO_ROOT / "aws/glm52-gpu/skypilot/bootstrap_campaign.sh"
        ).read_bytes(),
        managed_entrypoint_raw=(
            REPO_ROOT / "aws/glm52-gpu/skypilot/run_managed_campaign.sh"
        ).read_bytes(),
    )


def _fixture(tmp_path: Path) -> ReadinessFixture:
    integration = _integration_module()
    fixture = integration._fixture(tmp_path)
    services = integration._services(fixture)
    integration._prepare(fixture, services, tmp_path)
    intent_key, intent = integration._remote_record_by_suffix(
        services,
        "/SKYPILOT_SUBMISSION_INTENT.json",
    )
    descriptor = fixture.values["descriptor"]
    readiness = fixture.values["readiness"]
    rehearsal = fixture.values["rehearsal"]
    assert isinstance(descriptor, dict)
    assert isinstance(readiness, dict)
    assert isinstance(rehearsal, dict)
    return ReadinessFixture(
        authorities=WorkerStartV2ReadinessAuthorities(
            descriptor=WorkerStartV2ReadinessArtifact(
                key=str(descriptor["campaign_descriptor_key"]),
                raw=fixture.paths.descriptor.read_bytes(),
            ),
            qualification_submission_ready=WorkerStartV2ReadinessArtifact(
                key=str(fixture.values["readiness_key"]),
                raw=fixture.paths.qualification_ready.read_bytes(),
            ),
            rehearsal_evidence=WorkerStartV2ReadinessArtifact(
                key=str(fixture.values["rehearsal_key"]),
                raw=fixture.paths.rehearsal_evidence.read_bytes(),
            ),
            intent=WorkerStartV2ReadinessArtifact(
                key=intent_key,
                raw=_canonical(intent, newline=True),
            ),
        ),
        source_snapshot=_source_snapshot(),
    )


def _rehash(value: dict[str, object], digest_field: str) -> dict[str, object]:
    body = dict(value)
    body.pop(digest_field, None)
    return {**body, digest_field: _sha(_canonical(body))}


def _different(value: object, *, field: str) -> object:
    if field.endswith("_sha256"):
        return ("e" if value == "f" * 64 else "f") * 64
    if field.endswith("_key"):
        assert isinstance(value, str)
        return f"campaigns/foreign/{field}.json"
    if field in {"must_start_by", "built_at", "intent_at"}:
        return "2026-07-26T12:01:00Z"
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is float:
        return value + 0.01
    assert isinstance(value, str)
    return f"foreign-{field}"


def _mutate_authority(
    fixture: ReadinessFixture,
    *,
    artifact_name: str,
    field: str,
) -> WorkerStartV2ReadinessAuthorities:
    artifact = getattr(fixture.authorities, artifact_name)
    value = json.loads(artifact.raw)
    assert isinstance(value, dict)
    value[field] = _different(value[field], field=field)
    digest_fields = {
        "qualification_submission_ready": "readiness_body_sha256",
        "rehearsal_evidence": "rehearsal_body_sha256",
        "intent": "intent_body_sha256",
    }
    digest_field = digest_fields[artifact_name]
    value = _rehash(value, digest_field)
    run_id = str(value["run_id"])
    if artifact_name == "qualification_submission_ready":
        key = (
            f"campaigns/{run_id}/qualification/submission-ready/"
            f"{value[digest_field]}/QUALIFICATION_SUBMISSION_READY.json"
        )
    elif artifact_name == "rehearsal_evidence":
        key = (
            f"campaigns/{run_id}/qualification/rehearsals/"
            f"{value[digest_field]}/GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
        )
    else:
        key = (
            f"campaigns/{run_id}/submissions/qualification/intents/"
            f"{value[digest_field]}/SKYPILOT_SUBMISSION_INTENT.json"
        )
    return replace(
        fixture.authorities,
        **{
            artifact_name: WorkerStartV2ReadinessArtifact(
                key=key,
                raw=_canonical(value, newline=True),
            )
        },
    )


def _mutate_record(
    value: dict[str, object],
    *,
    field: str,
    replacement: object | None = None,
) -> dict[str, object]:
    mutated = deepcopy(value)
    mutated[field] = (
        _different(mutated[field], field=field) if replacement is None else replacement
    )
    return _rehash(mutated, "worker_start_v2_readiness_body_sha256")


def test_public_api_exposes_frozen_dataclasses_and_helpers() -> None:
    assert issubclass(WorkerStartV2ReadinessError, ValueError)
    assert is_dataclass(WorkerStartV2ReadinessArtifact)
    assert is_dataclass(WorkerStartV2ReadinessAuthorities)
    assert is_dataclass(WorkerStartV2SourceSnapshot)
    assert callable(build_worker_start_v2_readiness)
    assert callable(validate_worker_start_v2_readiness)
    assert callable(worker_start_v2_readiness_file_bytes)
    assert callable(worker_start_v2_readiness_file_sha256)
    assert callable(worker_start_v2_readiness_s3_key)


def test_readiness_imports_actual_frozen_dependency_chain_on_system_python_39() -> None:
    python = Path("/usr/bin/python3")
    if not python.is_file():
        pytest.skip("system Python is unavailable")
    version = subprocess.run(
        [
            str(python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if version.returncode != 0 or version.stdout.strip() != "3.9":
        pytest.skip("system Python is not Python 3.9")

    bootstrap = """
import datetime as datetime_module
import importlib
import pathlib
import sys
import types

assert not hasattr(datetime_module, "UTC")
repo_root = pathlib.Path(sys.argv[1]).resolve()
source_root = repo_root / "src"
mlx_vq_root = source_root / "mlx_vq"
quality_root = mlx_vq_root / "quality"

mlx_vq = types.ModuleType("mlx_vq")
mlx_vq.__path__ = [str(mlx_vq_root)]
mlx_vq.__package__ = "mlx_vq"
quality = types.ModuleType("mlx_vq.quality")
quality.__path__ = [str(quality_root)]
quality.__package__ = "mlx_vq.quality"
sys.modules["mlx_vq"] = mlx_vq
sys.modules["mlx_vq.quality"] = quality

numpy = types.ModuleType("numpy")
numpy.int32 = "int32"
numpy.float16 = "float16"
numpy.float32 = "float32"
numpy.dtype = lambda value: value
safetensors = types.ModuleType("safetensors")
safetensors.__path__ = []
safetensors_numpy = types.ModuleType("safetensors.numpy")
safetensors_numpy.load = lambda *_args, **_kwargs: None
safetensors_numpy.save = lambda *_args, **_kwargs: b""
sys.modules["numpy"] = numpy
sys.modules["safetensors"] = safetensors
sys.modules["safetensors.numpy"] = safetensors_numpy

module = importlib.import_module(
    "mlx_vq.quality.glm52_sky_worker_start_v2_readiness"
)
assert datetime_module.UTC is datetime_module.timezone.utc
for name, loaded in tuple(sys.modules.items()):
    if name.startswith("mlx_vq.quality."):
        loaded_path = pathlib.Path(loaded.__file__).resolve()
        loaded_path.relative_to(quality_root)
print(module.__name__)
for name in (
    "glm52_qualification_submission_ready",
    "glm52_qualification_cache_seed",
    "glm52_teich_training_cache",
    "glm52_sky_must_start_dynamic",
):
    dependency = importlib.import_module(f"mlx_vq.quality.{name}")
    print(pathlib.Path(dependency.__file__).resolve())
"""
    result = subprocess.run(
        [str(python), "-c", bootstrap, str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "mlx_vq.quality.glm52_sky_worker_start_v2_readiness",
        *[
            str((REPO_ROOT / "src/mlx_vq/quality" / f"{name}.py").resolve())
            for name in (
                "glm52_qualification_submission_ready",
                "glm52_qualification_cache_seed",
                "glm52_teich_training_cache",
                "glm52_sky_must_start_dynamic",
            )
        ],
    ]


@pytest.mark.parametrize(
    "family",
    ["artifact", "authorities", "source-snapshot"],
)
def test_exact_dataclass_inventories_reject_dynamically_injected_members(
    tmp_path: Path,
    family: str,
) -> None:
    fixture = _fixture(tmp_path)
    if family == "artifact":
        target = fixture.authorities.descriptor
    elif family == "authorities":
        target = fixture.authorities
    else:
        target = fixture.source_snapshot
    object.__setattr__(target, "attacker_injected", b"foreign")

    with pytest.raises(WorkerStartV2ReadinessError):
        build_worker_start_v2_readiness(
            authorities=fixture.authorities,
            source_snapshot=fixture.source_snapshot,
        )


def test_deterministic_build_validate_file_transport_and_key(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    first = build_worker_start_v2_readiness(
        authorities=fixture.authorities,
        source_snapshot=fixture.source_snapshot,
    )
    second = build_worker_start_v2_readiness(
        authorities=fixture.authorities,
        source_snapshot=fixture.source_snapshot,
    )

    assert first == second
    assert set(first) == EXPECTED_FIELDS
    assert first["schema_version"] == 1
    assert first["record_type"] == "glm52_sky_worker_start_v2_readiness_v1"
    assert first["account_id"] == "246813579024"
    assert first["region"] == "us-west-2"
    assert first["managed_mode"] == "qualification"
    assert first["skypilot_version"] == "0.13.0"
    assert first["sky_task_name"] == "glm52-campaign"
    assert first["instance_type"] == "p5.48xlarge"
    assert first["instance_count"] == 1
    assert first["use_spot"] is False
    assert first["source_files"] == EXPECTED_SOURCE_FILES
    assert (
        first["source_closure_sha256"]
        == "6ec127962854b61002eec7f04bef935edb35ef6acd84e2b1c814beb0de822b2c"
    )
    assert (
        first["source_contract_sha256"]
        == "fdb6cbd36dc852426334bab72b6f3dddb98c1b931442dfb21e373419efa962b2"
    )
    body = dict(first)
    digest = body.pop("worker_start_v2_readiness_body_sha256")
    assert digest == _sha(_canonical(body))
    raw = worker_start_v2_readiness_file_bytes(first)
    assert raw == _canonical(first, newline=True)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert worker_start_v2_readiness_file_sha256(first) == _sha(raw)
    assert worker_start_v2_readiness_s3_key(first) == (
        f"campaigns/{first['run_id']}/submissions/qualification/intents/"
        f"{first['intent_body_sha256']}/worker-start-v2-readiness/"
        f"{digest}/WORKER_START_V2_READINESS.json"
    )
    assert (
        validate_worker_start_v2_readiness(
            first,
            authorities=fixture.authorities,
            source_snapshot=fixture.source_snapshot,
        )
        == first
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("sky_job_name", "foreign-qualification"),
        ("sky_job_name", 7),
        ("image_id", 7),
        ("image_id", "ami-NOT-LOWER-HEX"),
        ("controller_identity", 7),
        ("controller_identity", "arn:aws:iam::135792468013:role/foreign"),
        (
            "controller_identity",
            "arn:aws:iam::246813579024:role/path/",
        ),
        ("worker_identity", 7),
        ("worker_identity", "arn:aws:iam::135792468013:role/foreign"),
        (
            "worker_identity",
            "arn:aws:iam::246813579024:role//foreign",
        ),
        (
            "descriptor_key",
            "campaigns/glm52-sky-20260723/foreign/campaign-descriptor-v2.json",
        ),
        (
            "repo_tar_key",
            "campaigns/glm52-sky-20260723/foreign/repository.tar.gz",
        ),
        (
            "qualification_submission_ready_key",
            (
                "campaigns/glm52-sky-20260723/qualification/submission-ready/"
                f"{'0' * 64}/QUALIFICATION_SUBMISSION_READY.json"
            ),
        ),
        (
            "rehearsal_evidence_key",
            (
                "campaigns/glm52-sky-20260723/qualification/rehearsals/"
                f"{'0' * 64}/GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
            ),
        ),
        (
            "intent_key",
            (
                "campaigns/glm52-sky-20260723/submissions/qualification/intents/"
                f"{'0' * 64}/SKYPILOT_SUBMISSION_INTENT.json"
            ),
        ),
        ("descriptor_file_sha256", "descriptor_body_sha256"),
        (
            "qualification_submission_ready_file_sha256",
            "qualification_submission_ready_body_sha256",
        ),
        ("rehearsal_evidence_file_sha256", "rehearsal_evidence_body_sha256"),
        ("intent_file_sha256", "intent_body_sha256"),
        ("sky_task_file_sha256", "0" * 64),
        ("source_contract_sha256", "0" * 64),
        ("source_closure_sha256", "0" * 64),
    ],
)
@pytest.mark.parametrize(
    "helper",
    [
        worker_start_v2_readiness_file_bytes,
        worker_start_v2_readiness_file_sha256,
        worker_start_v2_readiness_s3_key,
    ],
)
def test_standalone_helpers_reject_self_rehashed_intrinsic_contradictions(
    tmp_path: Path,
    field: str,
    replacement: object,
    helper: Any,
) -> None:
    fixture = _fixture(tmp_path)
    value = build_worker_start_v2_readiness(
        authorities=fixture.authorities,
        source_snapshot=fixture.source_snapshot,
    )
    if (
        isinstance(replacement, str)
        and replacement.endswith("_sha256")
        and replacement in value
    ):
        replacement = value[replacement]
    mutated = _mutate_record(value, field=field, replacement=replacement)

    with pytest.raises(WorkerStartV2ReadinessError):
        helper(mutated)


@pytest.mark.parametrize(
    "helper",
    [
        worker_start_v2_readiness_file_bytes,
        worker_start_v2_readiness_file_sha256,
        worker_start_v2_readiness_s3_key,
    ],
)
def test_standalone_helpers_reject_foreign_repo_key_with_embedded_exact_digest(
    tmp_path: Path,
    helper: Any,
) -> None:
    fixture = _fixture(tmp_path)
    value = build_worker_start_v2_readiness(
        authorities=fixture.authorities,
        source_snapshot=fixture.source_snapshot,
    )
    foreign_key = (
        f"campaigns/{value['run_id']}/repository/"
        f"attacker-{value['repo_tar_sha256']}.tar.gz"
    )
    mutated = _mutate_record(
        value,
        field="repo_tar_key",
        replacement=foreign_key,
    )

    with pytest.raises(WorkerStartV2ReadinessError):
        helper(mutated)


@pytest.mark.parametrize(
    "consumer",
    [
        "validate",
        "file-bytes",
        "file-sha",
        "s3-key",
    ],
)
def test_public_record_consumers_translate_recursive_values_to_domain_errors(
    tmp_path: Path,
    consumer: str,
) -> None:
    fixture = _fixture(tmp_path)
    value = build_worker_start_v2_readiness(
        authorities=fixture.authorities,
        source_snapshot=fixture.source_snapshot,
    )
    value["source_files"] = [value]

    with pytest.raises(WorkerStartV2ReadinessError):
        if consumer == "validate":
            validate_worker_start_v2_readiness(
                value,
                authorities=fixture.authorities,
                source_snapshot=fixture.source_snapshot,
            )
        elif consumer == "file-bytes":
            worker_start_v2_readiness_file_bytes(value)
        elif consumer == "file-sha":
            worker_start_v2_readiness_file_sha256(value)
        else:
            worker_start_v2_readiness_s3_key(value)


@pytest.mark.parametrize(
    ("family", "mutation"),
    [
        ("artifact", "missing"),
        ("authorities", "missing"),
        ("source-snapshot", "missing"),
        ("artifact", "subclass"),
        ("authorities", "subclass"),
        ("source-snapshot", "subclass"),
    ],
)
def test_exact_dataclass_inventories_reject_missing_and_subclassed_state(
    tmp_path: Path,
    family: str,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path)
    authorities: object = fixture.authorities
    source_snapshot: object = fixture.source_snapshot
    if mutation == "missing":
        target = {
            "artifact": fixture.authorities.descriptor,
            "authorities": fixture.authorities,
            "source-snapshot": fixture.source_snapshot,
        }[family]
        field = {
            "artifact": "raw",
            "authorities": "intent",
            "source-snapshot": "sky_task_raw",
        }[family]
        object.__delattr__(target, field)
    elif family == "artifact":

        class ArtifactSubclass(WorkerStartV2ReadinessArtifact):
            pass

        authorities = replace(
            fixture.authorities,
            descriptor=ArtifactSubclass(**vars(fixture.authorities.descriptor)),
        )
    elif family == "authorities":

        class AuthoritiesSubclass(WorkerStartV2ReadinessAuthorities):
            pass

        authorities = AuthoritiesSubclass(
            descriptor=fixture.authorities.descriptor,
            qualification_submission_ready=(
                fixture.authorities.qualification_submission_ready
            ),
            rehearsal_evidence=fixture.authorities.rehearsal_evidence,
            intent=fixture.authorities.intent,
        )
    else:

        class SnapshotSubclass(WorkerStartV2SourceSnapshot):
            pass

        source_snapshot = SnapshotSubclass(**vars(fixture.source_snapshot))

    with pytest.raises(WorkerStartV2ReadinessError):
        build_worker_start_v2_readiness(
            authorities=authorities,  # type: ignore[arg-type]
            source_snapshot=source_snapshot,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("family", "mutation"),
    [
        ("artifact", "renamed"),
        ("artifact", "non-byte"),
        ("authorities", "renamed"),
        ("authorities", "non-artifact"),
        ("source-snapshot", "renamed"),
        ("source-snapshot", "non-byte"),
    ],
)
def test_exact_dataclass_inventories_reject_renamed_or_wrong_typed_members(
    tmp_path: Path,
    family: str,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path)
    if family == "artifact":
        target = fixture.authorities.descriptor
        field = "raw"
        replacement: object = bytearray(target.raw)
    elif family == "authorities":
        target = fixture.authorities
        field = "descriptor"
        replacement = bytearray(b"not-an-artifact")
    else:
        target = fixture.source_snapshot
        field = "coordinator_raw"
        replacement = bytearray(target.coordinator_raw)

    if mutation == "renamed":
        original = getattr(target, field)
        object.__delattr__(target, field)
        object.__setattr__(target, f"renamed_{field}", original)
    else:
        object.__setattr__(target, field, replacement)

    with pytest.raises(WorkerStartV2ReadinessError):
        build_worker_start_v2_readiness(
            authorities=fixture.authorities,
            source_snapshot=fixture.source_snapshot,
        )


@pytest.mark.parametrize("field", sorted(EXPECTED_FIELDS))
def test_exact_schema_rejects_each_missing_field(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _fixture(tmp_path)
    value = build_worker_start_v2_readiness(
        authorities=fixture.authorities,
        source_snapshot=fixture.source_snapshot,
    )
    value.pop(field)

    with pytest.raises(WorkerStartV2ReadinessError):
        validate_worker_start_v2_readiness(
            value,
            authorities=fixture.authorities,
            source_snapshot=fixture.source_snapshot,
        )


def test_exact_schema_rejects_unknown_field_and_mixed_version(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    value = build_worker_start_v2_readiness(
        authorities=fixture.authorities,
        source_snapshot=fixture.source_snapshot,
    )
    unknown = {**value, "capacity_reservation_id": "cr-foreign"}
    mixed = _mutate_record(value, field="record_type", replacement="legacy_v1")

    for candidate in (unknown, mixed):
        with pytest.raises(WorkerStartV2ReadinessError):
            validate_worker_start_v2_readiness(
                candidate,
                authorities=fixture.authorities,
                source_snapshot=fixture.source_snapshot,
            )


@pytest.mark.parametrize(
    ("artifact_name", "field"),
    [
        *[
            ("qualification_submission_ready", field)
            for field in (
                "account_id",
                "region",
                "run_id",
                "descriptor_key",
                "descriptor_file_sha256",
                "descriptor_body_sha256",
                "campaign_identity_sha256",
                "approval_sha256",
                "repo_tar_sha256",
                "sky_task_name",
                "sky_job_name",
                "must_start_by",
                "managed_mode",
            )
        ],
        *[
            ("rehearsal_evidence", field)
            for field in (
                "run_id",
                "campaign_identity_sha256",
                "descriptor_key",
                "descriptor_file_sha256",
                "descriptor_body_sha256",
                "repo_tar_sha256",
                "artifact_inventory_key",
                "artifact_inventory_file_sha256",
                "skypilot_version",
                "skypilot_task_name",
                "staged_readiness_key",
                "staged_readiness_file_sha256",
                "staged_readiness_body_sha256",
            )
        ],
        *[
            ("qualification_submission_ready", field)
            for field in (
                "rehearsal_evidence_key",
                "rehearsal_evidence_sha256",
                "rehearsal_evidence_body_sha256",
                "staged_readiness_key",
                "staged_readiness_file_sha256",
                "staged_readiness_body_sha256",
            )
        ],
        *[
            ("intent", field)
            for field in (
                "account_id",
                "region",
                "run_id",
                "managed_mode",
                "descriptor_key",
                "descriptor_file_sha256",
                "descriptor_body_sha256",
                "campaign_identity_sha256",
                "approval_sha256",
                "approval_body_sha256",
                "repo_tar_sha256",
                "cache_seed_acceptance_key",
                "cache_seed_acceptance_file_sha256",
                "cache_seed_acceptance_body_sha256",
                "gpu_spend_snapshot_key",
                "gpu_spend_snapshot_sha256",
                "gpu_spend_snapshot_body_sha256",
                "gpu_spend_ledger_tip_record_sha256",
                "remaining_gpu_seconds",
                "remaining_gpu_cost_usd",
                "qualification_allowance_seconds",
                "qualification_allowance_cost_usd",
                "sky_job_name",
                "must_start_by",
            )
        ],
    ],
)
def test_every_shared_authority_binding_rejects_drift(
    tmp_path: Path,
    artifact_name: str,
    field: str,
) -> None:
    fixture = _fixture(tmp_path)
    authorities = _mutate_authority(
        fixture,
        artifact_name=artifact_name,
        field=field,
    )

    with pytest.raises(WorkerStartV2ReadinessError):
        build_worker_start_v2_readiness(
            authorities=authorities,
            source_snapshot=fixture.source_snapshot,
        )


def test_artifact_transport_rejects_wrong_classes_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    descriptor = fixture.authorities.descriptor

    class ArtifactSubclass(WorkerStartV2ReadinessArtifact):
        pass

    candidates: list[Any] = [
        replace(
            fixture.authorities,
            descriptor=ArtifactSubclass(descriptor.key, descriptor.raw),
        ),
        replace(
            fixture.authorities,
            descriptor=WorkerStartV2ReadinessArtifact(
                descriptor.key,
                bytearray(descriptor.raw),  # type: ignore[arg-type]
            ),
        ),
        replace(
            fixture.authorities,
            descriptor=WorkerStartV2ReadinessArtifact(
                descriptor.key,
                descriptor.raw.removesuffix(b"\n"),
            ),
        ),
        replace(
            fixture.authorities,
            descriptor=WorkerStartV2ReadinessArtifact(
                descriptor.key.replace("/", "//", 1),
                descriptor.raw,
            ),
        ),
    ]
    for authorities in candidates:
        with pytest.raises(WorkerStartV2ReadinessError):
            build_worker_start_v2_readiness(
                authorities=authorities,
                source_snapshot=fixture.source_snapshot,
            )


def test_artifact_transport_rejects_nonfinite_numeric_json(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    artifact = fixture.authorities.qualification_submission_ready
    assert b'"remaining_gpu_cost_usd":1155.84' in artifact.raw
    nonfinite = artifact.raw.replace(
        b'"remaining_gpu_cost_usd":1155.84',
        b'"remaining_gpu_cost_usd":NaN',
    )
    authorities = replace(
        fixture.authorities,
        qualification_submission_ready=WorkerStartV2ReadinessArtifact(
            key=artifact.key,
            raw=nonfinite,
        ),
    )

    with pytest.raises(WorkerStartV2ReadinessError):
        build_worker_start_v2_readiness(
            authorities=authorities,
            source_snapshot=fixture.source_snapshot,
        )


@pytest.mark.parametrize(
    ("artifact_name", "field", "replacement"),
    [
        (
            "qualification_submission_ready",
            "built_at",
            "2026-07-26T04:56:00-07:00",
        ),
        (
            "qualification_submission_ready",
            "staged_readiness_key",
            "campaigns//unsafe key",
        ),
        (
            "rehearsal_evidence",
            "completed_at",
            "2026-07-26T11:50:00+00:00",
        ),
    ],
)
def test_authorities_reject_noncanonical_times_and_unsafe_keys(
    tmp_path: Path,
    artifact_name: str,
    field: str,
    replacement: str,
) -> None:
    fixture = _fixture(tmp_path)
    artifact = getattr(fixture.authorities, artifact_name)
    value = json.loads(artifact.raw)
    assert isinstance(value, dict)
    value[field] = replacement
    digest_field = (
        "readiness_body_sha256"
        if artifact_name == "qualification_submission_ready"
        else "rehearsal_body_sha256"
    )
    value = _rehash(value, digest_field)
    mutated = replace(
        fixture.authorities,
        **{
            artifact_name: WorkerStartV2ReadinessArtifact(
                key=artifact.key,
                raw=_canonical(value, newline=True),
            )
        },
    )

    with pytest.raises(WorkerStartV2ReadinessError):
        build_worker_start_v2_readiness(
            authorities=mutated,
            source_snapshot=fixture.source_snapshot,
        )


def test_body_and_file_hashes_cannot_be_confused(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    value = build_worker_start_v2_readiness(
        authorities=fixture.authorities,
        source_snapshot=fixture.source_snapshot,
    )
    confused = _mutate_record(
        value,
        field="descriptor_file_sha256",
        replacement=value["descriptor_body_sha256"],
    )

    with pytest.raises(WorkerStartV2ReadinessError):
        validate_worker_start_v2_readiness(
            confused,
            authorities=fixture.authorities,
            source_snapshot=fixture.source_snapshot,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("instance_count", True),
        ("use_spot", 0),
        ("schema_version", True),
    ],
)
def test_record_rejects_boolean_integer_aliases(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    fixture = _fixture(tmp_path)
    value = build_worker_start_v2_readiness(
        authorities=fixture.authorities,
        source_snapshot=fixture.source_snapshot,
    )
    mutated = _mutate_record(value, field=field, replacement=replacement)

    with pytest.raises(WorkerStartV2ReadinessError):
        validate_worker_start_v2_readiness(
            mutated,
            authorities=fixture.authorities,
            source_snapshot=fixture.source_snapshot,
        )


@pytest.mark.parametrize(
    "field",
    [
        "publisher_raw",
        "campaign_policy_raw",
        "must_start_policy_raw",
        "dynamic_policy_raw",
        "worker_policy_raw",
        "coordinator_raw",
        "sky_task_raw",
        "bootstrap_raw",
        "managed_entrypoint_raw",
    ],
)
def test_each_frozen_source_hash_is_independently_enforced(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _fixture(tmp_path)
    drifted = replace(
        fixture.source_snapshot,
        **{field: getattr(fixture.source_snapshot, field) + b"\n"},
    )

    with pytest.raises(WorkerStartV2ReadinessError):
        build_worker_start_v2_readiness(
            authorities=fixture.authorities,
            source_snapshot=drifted,
        )


@pytest.mark.parametrize(
    "source_files",
    [
        EXPECTED_SOURCE_FILES[:-1],
        list(reversed(EXPECTED_SOURCE_FILES)),
        [*EXPECTED_SOURCE_FILES, EXPECTED_SOURCE_FILES[0]],
        [
            *EXPECTED_SOURCE_FILES,
            {"relative_path": "extra.py", "file_sha256": "0" * 64},
        ],
    ],
    ids=["missing", "reordered", "duplicate", "extra"],
)
def test_source_closure_rejects_missing_reordered_duplicate_or_extra_members(
    tmp_path: Path,
    source_files: list[dict[str, str]],
) -> None:
    fixture = _fixture(tmp_path)
    value = build_worker_start_v2_readiness(
        authorities=fixture.authorities,
        source_snapshot=fixture.source_snapshot,
    )
    mutated = _mutate_record(
        value,
        field="source_files",
        replacement=deepcopy(source_files),
    )

    with pytest.raises(WorkerStartV2ReadinessError):
        validate_worker_start_v2_readiness(
            mutated,
            authorities=fixture.authorities,
            source_snapshot=fixture.source_snapshot,
        )


def test_file_helpers_reject_self_consistent_foreign_source_contract(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    value = build_worker_start_v2_readiness(
        authorities=fixture.authorities,
        source_snapshot=fixture.source_snapshot,
    )
    foreign = deepcopy(value)
    foreign["source_contract_sha256"] = "f" * 64
    foreign = _rehash(foreign, "worker_start_v2_readiness_body_sha256")

    with pytest.raises(WorkerStartV2ReadinessError):
        worker_start_v2_readiness_file_bytes(foreign)


@pytest.mark.parametrize(
    ("field", "old", "new"),
    [
        (
            "publisher_raw",
            b"def publish_worker_start_latch_v2(",
            b"def publish_worker_start_latch_v1(",
        ),
        (
            "campaign_policy_raw",
            b"def validate_sky_campaign_descriptor(",
            b"def validate_other_descriptor(",
        ),
        (
            "must_start_policy_raw",
            b"def build_must_start_controller_observation(",
            b"def build_other_observation(",
        ),
        (
            "dynamic_policy_raw",
            b"def dynamic_v2_job_binding_s3_key(",
            b"def other_job_binding_s3_key(",
        ),
        (
            "worker_policy_raw",
            b"def build_worker_start_latch_v2(",
            b"def build_worker_start_latch_v1(",
        ),
        (
            "coordinator_raw",
            b"def coordinate_worker_start_acceptance_v2(",
            b"def coordinate_worker_start_acceptance_v1(",
        ),
        (
            "sky_task_raw",
            b"publish-latch-v2",
            b"publish-latch-v1",
        ),
        (
            "bootstrap_raw",
            b"verify-receipt-v2",
            b"verify-receipt-v1",
        ),
        (
            "managed_entrypoint_raw",
            b"verify-receipt-v2",
            b"verify-receipt-v1",
        ),
    ],
)
def test_required_source_contract_tokens_are_enforced(
    tmp_path: Path,
    field: str,
    old: bytes,
    new: bytes,
) -> None:
    fixture = _fixture(tmp_path)
    raw = getattr(fixture.source_snapshot, field)
    assert raw.count(old) >= 1
    drifted = replace(fixture.source_snapshot, **{field: raw.replace(old, new, 1)})

    with pytest.raises(WorkerStartV2ReadinessError, match="source contract"):
        build_worker_start_v2_readiness(
            authorities=fixture.authorities,
            source_snapshot=drifted,
        )


@pytest.mark.parametrize(
    "field",
    ["sky_task_raw", "bootstrap_raw", "managed_entrypoint_raw"],
)
def test_forbidden_legacy_tokens_are_rejected(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _fixture(tmp_path)
    raw = getattr(fixture.source_snapshot, field)
    drifted = replace(
        fixture.source_snapshot,
        **{field: raw + b"\n# IMMUTABLE_SUBMISSION.json\n"},
    )

    with pytest.raises(WorkerStartV2ReadinessError, match="source contract"):
        build_worker_start_v2_readiness(
            authorities=fixture.authorities,
            source_snapshot=drifted,
        )


@pytest.mark.parametrize(
    ("field", "first", "second"),
    [
        ("sky_task_raw", b"publish-latch-v2", b"replay-accepted-v2"),
        ("bootstrap_raw", b"verify-receipt-v2", b"prepare_nvme_storage.sh"),
        (
            "managed_entrypoint_raw",
            b"verify-receipt-v2",
            b"GLM52_SKY_CONTROLLER_JOB_ID=",
        ),
    ],
)
def test_required_source_contract_order_is_enforced(
    tmp_path: Path,
    field: str,
    first: bytes,
    second: bytes,
) -> None:
    fixture = _fixture(tmp_path)
    raw = getattr(fixture.source_snapshot, field)
    first_at = raw.index(first)
    second_at = raw.index(second)
    assert first_at < second_at
    drifted_raw = (
        raw[:first_at]
        + second
        + raw[first_at + len(first) : second_at]
        + first
        + raw[second_at + len(second) :]
    )
    drifted = replace(fixture.source_snapshot, **{field: drifted_raw})

    with pytest.raises(WorkerStartV2ReadinessError, match="source contract"):
        build_worker_start_v2_readiness(
            authorities=fixture.authorities,
            source_snapshot=drifted,
        )


def test_task_sha_must_match_rehearsal_pin(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authorities = _mutate_authority(
        fixture,
        artifact_name="rehearsal_evidence",
        field="skypilot_task_file_sha256",
    )

    with pytest.raises(WorkerStartV2ReadinessError):
        build_worker_start_v2_readiness(
            authorities=authorities,
            source_snapshot=fixture.source_snapshot,
        )


def test_inputs_are_not_mutated(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authority_raw = tuple(
        (artifact.key, artifact.raw)
        for artifact in (
            fixture.authorities.descriptor,
            fixture.authorities.qualification_submission_ready,
            fixture.authorities.rehearsal_evidence,
            fixture.authorities.intent,
        )
    )
    source_raw = tuple(vars(fixture.source_snapshot).values())

    value = build_worker_start_v2_readiness(
        authorities=fixture.authorities,
        source_snapshot=fixture.source_snapshot,
    )
    validate_worker_start_v2_readiness(
        deepcopy(value),
        authorities=fixture.authorities,
        source_snapshot=fixture.source_snapshot,
    )

    assert authority_raw == tuple(
        (artifact.key, artifact.raw)
        for artifact in (
            fixture.authorities.descriptor,
            fixture.authorities.qualification_submission_ready,
            fixture.authorities.rehearsal_evidence,
            fixture.authorities.intent,
        )
    )
    assert source_raw == tuple(vars(fixture.source_snapshot).values())
