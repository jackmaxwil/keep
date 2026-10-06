from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.decision_closure import (
    build_deployed_gate_document,
    rehearsal_measurement_from_mapping,
)
from glm52_enforcement.task13_campaign_package import (
    build_campaign_package,
    canonical_campaign_package_bytes,
)
from glm52_enforcement.task13_campaign_runner import FinalizationCapture
from test_glm52_task13_campaign_package import _staged_evidence, request
from test_glm52_task13_campaign_runner import (
    ACTIVATION,
    COLLECTOR_ARN,
    _measurement,
)


ACCOUNT = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
REHEARSAL_BUCKET = "keep-glm52-h1g-rehearsal-246813579024-us-west-2"
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "aws/glm52-gpu/scripts/build_glm52_task13_clean_rehearsal.py"
)
RUNNER_SCRIPT = (
    ROOT
    / "aws/glm52-gpu/scripts/run_glm52_task13_campaign.py"
)
def _load_cli(path: Path, module_name: str) -> object:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture(fixture: dict[str, object]) -> FinalizationCapture:
    result = fixture["finalize_invoke_result"]
    assert type(result) is dict
    payload = result["Payload"]
    payload.seek(0)
    return FinalizationCapture(
        status_code=result["StatusCode"],
        executed_version=result["ExecutedVersion"],
        payload_bytes=payload.read(),
        gate_bytes=fixture["gate_readback_bytes"],
    )




def _fixture() -> dict[str, object]:
    archive_sha = "a" * 64
    archive_body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_task13_repository_archive_manifest_v2",
        "account_id": ACCOUNT,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "archive_file_sha256": archive_sha,
        "archive_size_bytes": 123,
        "archive": {
            "bucket": BUCKET,
            "key": (
                f"campaigns/{RUN_ID}/repository/keep-{archive_sha}.tar.gz"
            ),
            "version_id": "archive-payload-version",
            "file_sha256": archive_sha,
        },
    }
    archive_manifest = {
        **archive_body,
        "canonical_identity_sha256": canonical_sha256(archive_body),
    }
    archive_manifest_raw = canonical_json_bytes(archive_manifest) + b"\n"
    archive_coordinate = {
        "artifact_kind": "REPOSITORY_ARCHIVE",
        "bucket": BUCKET,
        "key": f"task13/activations/{ACTIVATION}/archive/repo-tar.json",
        "version_id": "archive-manifest-version",
        "file_sha256": hashlib.sha256(archive_manifest_raw).hexdigest(),
        "body_sha256": archive_manifest["canonical_identity_sha256"],
    }

    package_request = request()
    package_request["artifacts"] = [
        archive_coordinate
        if row["artifact_kind"] == "REPOSITORY_ARCHIVE"
        else row
        for row in package_request["artifacts"]
    ]
    package_request["staged_infrastructure_evidence"] = _staged_evidence(
        package_request["artifacts"]
    )
    predecessor = build_campaign_package(package_request)
    predecessor_raw = canonical_campaign_package_bytes(predecessor)

    measurements = tuple(
        rehearsal_measurement_from_mapping(_measurement(index))
        for index in range(1, 21)
    )
    deployment_identity = "d" * 64
    gate_raw = build_deployed_gate_document(
        account_id=ACCOUNT,
        region=REGION,
        run_id=RUN_ID,
        activation_id=ACTIVATION,
        deployment_identity_sha256=deployment_identity,
        measurements=measurements,
    )
    gate_value = json.loads(gate_raw)
    finalizer_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task11_finalize_rehearsal_gate_result_v1",
        "status": "CLOSURE_BUDGET_PROVEN",
        "account_id": ACCOUNT,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "collector_function_version_arn": COLLECTOR_ARN,
        "key": f"rehearsal/gates/{ACTIVATION}/CLOSURE_BUDGET.json",
        "version_id": "gate-version-0001",
        "file_sha256": hashlib.sha256(gate_raw).hexdigest(),
        "body_sha256": gate_value["canonical_body_sha256"],
        "checksum_sha256_base64": base64.b64encode(
            hashlib.sha256(gate_raw).digest()
        ).decode("ascii"),
        "measurement_count": 20,
        "cold_environment_count": 5,
        "measurements_identity_sha256": gate_value[
            "measurements_identity_sha256"
        ],
    }
    finalizer_payload = {
        **finalizer_body,
        "canonical_identity_sha256": canonical_sha256(finalizer_body),
    }
    finalizer_result = {
        "StatusCode": 200,
        "ExecutedVersion": "19",
        "Payload": io.BytesIO(canonical_json_bytes(finalizer_payload)),
    }
    return {
        "predecessor_package": predecessor,
        "predecessor_package_bytes": predecessor_raw,
        "predecessor_reviewed_artifacts": predecessor[
            "reviewed_artifacts"
        ],
        "repository_archive_coordinate": archive_coordinate,
        "repository_archive_manifest": archive_manifest,
        "repository_archive_manifest_bytes": archive_manifest_raw,
        "finalize_invoke_result": finalizer_result,
        "gate_readback": gate_value,
        "gate_readback_bytes": gate_raw,
    }


def _build(fixture: dict[str, object]) -> dict[str, object]:
    from glm52_enforcement.task13_clean_rehearsal import (
        build_clean_rehearsal_evidence,
    )

    result = fixture["finalize_invoke_result"]
    assert type(result) is dict
    payload = result["Payload"]
    if hasattr(payload, "seek"):
        payload.seek(0)
    return build_clean_rehearsal_evidence(**fixture)


def test_clean_rehearsal_evidence_is_canonical_self_hashed_and_bound() -> None:
    from glm52_enforcement.task13_clean_rehearsal import (
        canonical_clean_rehearsal_evidence_bytes,
        clean_rehearsal_artifact_key,
        validate_clean_rehearsal_evidence,
    )

    fixture = _fixture()
    evidence = _build(fixture)
    raw = canonical_clean_rehearsal_evidence_bytes(evidence)

    assert evidence["record_type"] == (
        "glm52_task13_clean_rehearsal_evidence_v1"
    )
    assert evidence["activation_id"] == ACTIVATION
    assert evidence["predecessor_package"][
        "package_identity_sha256"
    ] == fixture["predecessor_package"]["canonical_identity_sha256"]
    assert evidence["repository_archive"]["coordinate"] == fixture[
        "repository_archive_coordinate"
    ]
    assert evidence["finalizer"]["measurements_identity_sha256"] == (
        evidence["immutable_gate"]["measurements_identity_sha256"]
    )
    assert evidence["canonical_identity_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in evidence.items()
            if key != "canonical_identity_sha256"
        }
    )
    assert raw == canonical_json_bytes(evidence) + b"\n"
    assert validate_clean_rehearsal_evidence(evidence) == evidence
    assert clean_rehearsal_artifact_key(evidence) == (
        f"task13/gates/clean-rehearsal/{ACTIVATION}/"
        f"{evidence['canonical_identity_sha256']}.json"
    )


def test_standalone_evidence_rejects_cross_activation_archive() -> None:
    from glm52_enforcement.task13_clean_rehearsal import (
        validate_clean_rehearsal_evidence,
    )

    evidence = _build(_fixture())
    archive = evidence["repository_archive"]["coordinate"]
    archive["key"] = (
        "task13/activations/approved-20260729/archive/repo-tar.json"
    )
    evidence.pop("canonical_identity_sha256")
    evidence["canonical_identity_sha256"] = canonical_sha256(evidence)

    with pytest.raises(ValueError, match="activation drifted"):
        validate_clean_rehearsal_evidence(evidence)


def test_standalone_evidence_recomputes_archive_manifest_hashes() -> None:
    from glm52_enforcement.task13_clean_rehearsal import (
        validate_clean_rehearsal_evidence,
    )

    evidence = _build(_fixture())
    archive = evidence["repository_archive"]
    foreign_body = {
        "schema_version": 2,
        "record_type": "glm52_task13_repository_archive_manifest_v2",
        "account_id": evidence["account_id"],
        "region": evidence["region"],
        "run_id": evidence["run_id"],
        "activation_id": "approved-20260729",
        "archive_file_sha256": archive["archive_file_sha256"],
        "archive_size_bytes": archive["archive_size_bytes"],
        "archive": archive["archive_payload"],
    }
    foreign_manifest = {
        **foreign_body,
        "canonical_identity_sha256": canonical_sha256(foreign_body),
    }
    foreign_file_sha = hashlib.sha256(
        canonical_json_bytes(foreign_manifest) + b"\n"
    ).hexdigest()
    archive["manifest_identity_sha256"] = foreign_manifest[
        "canonical_identity_sha256"
    ]
    archive["manifest_file_sha256"] = foreign_file_sha
    archive["coordinate"]["body_sha256"] = foreign_manifest[
        "canonical_identity_sha256"
    ]
    archive["coordinate"]["file_sha256"] = foreign_file_sha
    evidence.pop("canonical_identity_sha256")
    evidence["canonical_identity_sha256"] = canonical_sha256(evidence)

    with pytest.raises(ValueError, match="manifest identity drifted"):
        validate_clean_rehearsal_evidence(evidence)


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        (
            "package bytes",
            lambda value: value.__setitem__(
                "predecessor_package_bytes",
                value["predecessor_package_bytes"] + b" ",
            ),
        ),
        (
            "reviewed list",
            lambda value: value["predecessor_reviewed_artifacts"].reverse(),
        ),
        (
            "archive coordinate",
            lambda value: value["repository_archive_coordinate"].__setitem__(
                "version_id", "foreign-version"
            ),
        ),
        (
            "archive manifest bytes",
            lambda value: value.__setitem__(
                "repository_archive_manifest_bytes",
                value["repository_archive_manifest_bytes"] + b" ",
            ),
        ),
        (
            "finalizer",
            lambda value: value["finalize_invoke_result"].__setitem__(
                "ExecutedVersion", "18"
            ),
        ),
        (
            "gate bytes",
            lambda value: value.__setitem__(
                "gate_readback_bytes", value["gate_readback_bytes"] + b"\n"
            ),
        ),
        (
            "gate value",
            lambda value: value["gate_readback"].__setitem__(
                "activation_id", "foreign-activation"
            ),
        ),
    ],
)
def test_builder_rejects_binding_or_canonicality_drift(
    field: str,
    mutate: object,
) -> None:
    fixture = _fixture()
    mutate(fixture)

    with pytest.raises(ValueError, match="drift|canonical|identity|exact"):
        _build(fixture)


def test_validated_evidence_coordinate_uses_identity_key_and_dual_hashes() -> None:
    from glm52_enforcement.task13_clean_rehearsal import (
        clean_rehearsal_coordinate,
        validate_clean_rehearsal_coordinate,
    )

    evidence = _build(_fixture())
    coordinate = clean_rehearsal_coordinate(
        evidence=evidence,
        version_id="clean-version-0001",
    )

    assert validate_clean_rehearsal_coordinate(
        coordinate,
        activation_id=ACTIVATION,
        evidence=evidence,
    ) == coordinate
    assert coordinate["key"].startswith(
        f"task13/gates/clean-rehearsal/{ACTIVATION}/"
    )
    assert coordinate["file_sha256"] != evidence[
        "canonical_identity_sha256"
    ]


def test_clean_coordinate_rejects_legacy_fixed_key() -> None:
    import glm52_enforcement.task13_clean_rehearsal as clean_api

    evidence = _build(_fixture())
    coordinate = clean_api.clean_rehearsal_coordinate(
        evidence=evidence,
        version_id="clean-version-0001",
    )
    coordinate["key"] = "task13/gates/clean-rehearsal.json"

    with pytest.raises(ValueError, match="key is not exact"):
        clean_api.validate_clean_rehearsal_coordinate(coordinate)
    assert not hasattr(clean_api, "LEGACY_CLEAN_REHEARSAL_KEY")


def test_local_cli_creates_one_private_evidence_file(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "_task13_clean_rehearsal_cli", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    fixture = _fixture()
    paths: dict[str, Path] = {}
    for name in (
        "predecessor_package",
        "predecessor_reviewed_artifacts",
        "repository_archive_coordinate",
        "repository_archive_manifest",
        "gate_readback",
    ):
        path = tmp_path / (name + ".json")
        path.write_bytes(canonical_json_bytes(fixture[name]) + b"\n")
        paths[name] = path
    finalizer_result = fixture["finalize_invoke_result"]
    assert type(finalizer_result) is dict
    payload = finalizer_result["Payload"]
    payload.seek(0)
    finalizer_payload_raw = payload.read()
    metadata_path = (tmp_path / "finalizer-metadata.json").resolve()
    payload_path = (tmp_path / "finalizer-payload.json").resolve()
    gate_path = (tmp_path / "gate-readback.json").resolve()
    runner_cli = _load_cli(RUNNER_SCRIPT, "_finalization_capture_cli")
    outputs = runner_cli._FinalizationCaptureFiles(
        metadata_path=metadata_path,
        payload_path=payload_path,
        gate_path=gate_path,
    )
    outputs(
        FinalizationCapture(
            status_code=finalizer_result["StatusCode"],
            executed_version=finalizer_result["ExecutedVersion"],
            payload_bytes=finalizer_payload_raw,
            gate_bytes=fixture["gate_readback_bytes"],
        )
    )
    outputs.commit()
    assert all(
        path.stat().st_mode & 0o777 == 0o600
        for path in (metadata_path, payload_path, gate_path)
    )
    assert metadata_path.read_bytes() == canonical_json_bytes(
        {"StatusCode": 200, "ExecutedVersion": "19"}
    ) + b"\n"
    assert payload_path.read_bytes() == finalizer_payload_raw
    assert not payload_path.read_bytes().endswith(b"\n")
    assert gate_path.read_bytes() == fixture["gate_readback_bytes"]
    assert not gate_path.read_bytes().endswith(b"\n")
    output = tmp_path / "clean-evidence.json"
    argv = [
        "--predecessor-package", str(paths["predecessor_package"]),
        "--predecessor-reviewed-artifacts",
        str(paths["predecessor_reviewed_artifacts"]),
        "--repository-archive-coordinate",
        str(paths["repository_archive_coordinate"]),
        "--repository-archive-manifest",
        str(paths["repository_archive_manifest"]),
        "--finalize-invoke-metadata", str(metadata_path),
        "--finalize-payload", str(payload_path),
        "--gate-readback", str(gate_path),
        "--output", str(output),
    ]

    assert cli.main(argv) == 0
    assert output.stat().st_mode & 0o777 == 0o600
    assert output.read_bytes() == canonical_json_bytes(
        json.loads(output.read_bytes())
    ) + b"\n"
    assert cli.main(argv) == 64


def test_capture_output_arguments_are_forbidden_outside_finalization(
    tmp_path: Path,
) -> None:
    runner_cli = _load_cli(RUNNER_SCRIPT, "_capture_argument_cli")
    fixture = _fixture()
    package_path = (tmp_path / "package.json").resolve()
    reviewed_path = (tmp_path / "reviewed.json").resolve()
    package_path.write_bytes(fixture["predecessor_package_bytes"])
    reviewed_path.write_bytes(
        canonical_json_bytes(fixture["predecessor_reviewed_artifacts"])
        + b"\n"
    )
    outputs = [
        (tmp_path / "metadata.json").resolve(),
        (tmp_path / "payload.json").resolve(),
        (tmp_path / "gate.json").resolve(),
    ]

    result = runner_cli.main(
        [
            "--package",
            str(package_path),
            "--reviewed-artifacts",
            str(reviewed_path),
            "--stage",
            "validate",
            "--finalize-invoke-metadata-output",
            str(outputs[0]),
            "--finalize-payload-output",
            str(outputs[1]),
            "--gate-readback-output",
            str(outputs[2]),
        ]
    )

    assert result == 64
    assert all(not path.exists() for path in outputs)


def test_capture_output_collision_preserves_foreign_file(
    tmp_path: Path,
) -> None:
    runner_cli = _load_cli(RUNNER_SCRIPT, "_capture_collision_cli")
    metadata = (tmp_path / "metadata.json").resolve()
    payload = (tmp_path / "payload.json").resolve()
    gate = (tmp_path / "gate.json").resolve()
    payload.write_bytes(b"foreign")

    with pytest.raises(FileExistsError):
        runner_cli._FinalizationCaptureFiles(
            metadata_path=metadata,
            payload_path=payload,
            gate_path=gate,
        )

    assert payload.read_bytes() == b"foreign"
    assert not metadata.exists()
    assert not gate.exists()


def test_capture_partial_write_removes_only_created_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_cli = _load_cli(RUNNER_SCRIPT, "_capture_partial_cli")
    paths = [
        (tmp_path / "metadata.json").resolve(),
        (tmp_path / "payload.json").resolve(),
        (tmp_path / "gate.json").resolve(),
    ]
    outputs = runner_cli._FinalizationCaptureFiles(
        metadata_path=paths[0],
        payload_path=paths[1],
        gate_path=paths[2],
    )
    original = runner_cli._write_descriptor
    calls = 0

    def fail_second(descriptor: int, raw: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated local write failure")
        original(descriptor, raw)

    monkeypatch.setattr(runner_cli, "_write_descriptor", fail_second)

    with pytest.raises(OSError, match="simulated"):
        outputs(_capture(_fixture()))

    assert all(not path.exists() for path in paths)
