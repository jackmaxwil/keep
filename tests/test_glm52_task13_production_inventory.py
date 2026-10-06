from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.task13_campaign_package import (
    PRODUCTION_ONLY_ARTIFACT_KINDS,
    build_campaign_package,
    canonical_campaign_package_bytes,
)
from glm52_enforcement.task13_transport_gates import SEMANTIC_GATE_PINS
from mlx_vq.quality.glm52_s3_artifact_audit import (
    build_s3_artifact_inventory,
    validate_s3_artifact_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "aws/glm52-gpu/scripts/build_glm52_task13_production_inventory.py"
)
ACCOUNT = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
SUPPORT_LAMBDA_ARCHIVE_RAW = b"s" * 4096
CRYPTOGRAPHY_LAYER_ARCHIVE_RAW = b"c" * 8192


def _api():
    if not SCRIPT.exists():
        pytest.fail(
            "Task 13 production-inventory bridge is missing",
            pytrace=False,
        )
    spec = importlib.util.spec_from_file_location(
        "_glm52_task13_production_inventory_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _campaign_fixture_api():
    path = ROOT / "tests/test_glm52_task13_campaign_package.py"
    spec = importlib.util.spec_from_file_location(
        "_glm52_task13_campaign_package_fixture",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metadata(request_id: str) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RetryAttempts": 0,
        "RequestId": request_id,
    }


class ExactReadS3:
    def __init__(
        self,
        objects: dict[tuple[str, str], bytes],
    ) -> None:
        self.objects = objects
        self.requests: list[dict[str, object]] = []

    def get_object(self, **request: object) -> object:
        self.requests.append(dict(request))
        key = str(request["Key"])
        version_id = str(request["VersionId"])
        raw = self.objects[(key, version_id)]
        return {
            "Body": io.BytesIO(raw),
            "ContentLength": len(raw),
            "VersionId": version_id,
            "ResponseMetadata": _metadata("get-" + str(len(self.requests))),
        }


class VersionDriftS3(ExactReadS3):
    def get_object(self, **request: object) -> object:
        response = super().get_object(**request)
        assert isinstance(response, dict)
        response["VersionId"] = "foreign-version"
        return response


class MissingSizeS3(ExactReadS3):
    def get_object(self, **request: object) -> object:
        response = super().get_object(**request)
        assert isinstance(response, dict)
        response.pop("ContentLength")
        return response


def _canonical_record(kind: str) -> bytes:
    return canonical_json_bytes(
        {
            "artifact_kind": kind,
            "record_type": "glm52_task13_test_reviewed_artifact_v1",
        }
    ) + b"\n"


def _production_package_and_objects(
    *,
    phase: str = "PRODUCTION",
) -> tuple[dict[str, object], dict[tuple[str, str], bytes]]:
    import glm52_task13_staged_fixture_support as staged_fixture

    fixture = _campaign_fixture_api()
    original_runtime_coordinate = (
        staged_fixture._published_runtime_coordinate
    )
    runtime_objects: dict[tuple[str, str], bytes] = {}
    runtime_coordinates: dict[str, dict[str, object]] = {}

    def exact_runtime_coordinate(
        *,
        key_kind: str,
        seed: bytes,
        version_id: str,
        size_bytes: int,
    ) -> dict[str, object]:
        del seed, size_bytes
        raw = {
            "support-lambda": SUPPORT_LAMBDA_ARCHIVE_RAW,
            "cryptography-layer-python312-x86_64": (
                CRYPTOGRAPHY_LAYER_ARCHIVE_RAW
            ),
        }[key_kind]
        coordinate = original_runtime_coordinate(
            key_kind=key_kind,
            seed=raw,
            version_id=version_id,
            size_bytes=len(raw),
        )
        runtime_coordinates[key_kind] = dict(coordinate)
        runtime_objects[
            (str(coordinate["key"]), str(coordinate["version_id"]))
        ] = raw
        return coordinate

    staged_fixture._published_runtime_coordinate = (
        exact_runtime_coordinate
    )
    try:
        production_request = fixture._production_request()
    finally:
        staged_fixture._published_runtime_coordinate = (
            original_runtime_coordinate
        )
    coordinates: list[dict[str, object]] = []
    objects: dict[tuple[str, str], bytes] = dict(runtime_objects)
    archive_raw = b"exact immutable repository archive bytes\n"
    archive_sha256 = hashlib.sha256(archive_raw).hexdigest()
    archive_manifest_value: object = None
    archive_manifest_raw: object = None
    archive_coordinate: object = None
    archive_key = (
        f"campaigns/{RUN_ID}/repository/keep-{archive_sha256}.tar.gz"
    )
    archive_version_id = "repository-payload-version"

    for source in production_request["artifacts"]:
        kind = str(source["artifact_kind"])
        version_id = "reviewed-" + kind.lower().replace("_", "-")
        if kind in SEMANTIC_GATE_PINS:
            filename = (
                "t01-t25.json"
                if kind == "T01_T25_GATE"
                else "transport-22-mutants.json"
            )
            raw = (ROOT / "task13/gates" / filename).read_bytes()
        elif kind == "FENCE_TEMPLATE":
            raw = canonical_json_bytes(
                {
                    "Resources": {
                        "H1gProductionFenceBucketPolicy": {
                            "Type": "AWS::S3::BucketPolicy",
                        }
                    }
                }
            ) + b"\n"
        elif kind == "SUPPORT_TEMPLATE":
            support_lambda = runtime_coordinates["support-lambda"]
            raw = canonical_json_bytes(
                {
                    "Resources": {
                        "SupportFunctionRole": {
                            "Type": "AWS::IAM::Role",
                            "Properties": {
                                "RoleName": (
                                    "keep-glm52-h1g-support-function-role"
                                ),
                                "AssumeRolePolicyDocument": {
                                    "Version": "2012-10-17",
                                    "Statement": [
                                        {
                                            "Effect": "Allow",
                                            "Principal": {
                                                "Service": (
                                                    "lambda.amazonaws.com"
                                                )
                                            },
                                            "Action": "sts:AssumeRole",
                                        }
                                    ],
                                },
                            },
                        },
                        "SupportFunction": {
                            "Type": "AWS::Lambda::Function",
                            "Properties": {
                                "FunctionName": (
                                    "keep-glm52-h1g-support-function"
                                ),
                                "Code": {
                                    "S3Bucket": support_lambda["bucket"],
                                    "S3Key": support_lambda["key"],
                                    "S3ObjectVersion": support_lambda[
                                        "version_id"
                                    ],
                                },
                                "Handler": (
                                    "support_custom_resource_handler.main"
                                ),
                                "Runtime": "python3.12",
                                "Architectures": ["x86_64"],
                                "MemorySize": 256,
                                "Timeout": 840,
                                "ReservedConcurrentExecutions": 1,
                                "Role": {
                                    "Fn::GetAtt": [
                                        "SupportFunctionRole",
                                        "Arn",
                                    ]
                                },
                                "Environment": {
                                    "Variables": {
                                        "GLM52_ACTIVATION_ID": (
                                            "approved-20260728"
                                        ),
                                        "GLM52_RUN_ID": RUN_ID,
                                    }
                                },
                                "Layers": [
                                    "arn:aws:lambda:us-west-2:"
                                    "246813579024:layer:"
                                    "keep-glm52-h1g-cryptography-"
                                    "py312-x86-64:7"
                                ],
                            },
                        },
                        "SupportVersion": {
                            "Type": "AWS::Lambda::Version",
                            "Properties": {
                                "FunctionName": {
                                    "Ref": "SupportFunction"
                                },
                                "Description": (
                                    "code="
                                    + str(support_lambda["file_sha256"])
                                ),
                            },
                        },
                    }
                }
            ) + b"\n"
        elif kind == "REPOSITORY_ARCHIVE":
            body = {
                "schema_version": 2,
                "record_type": (
                    "glm52_task13_repository_archive_manifest_v2"
                ),
                "account_id": ACCOUNT,
                "region": REGION,
                "run_id": RUN_ID,
                "activation_id": fixture.ACTIVATION,
                "archive_file_sha256": archive_sha256,
                "archive_size_bytes": len(archive_raw),
                "archive": {
                    "bucket": BUCKET,
                    "key": archive_key,
                    "version_id": archive_version_id,
                    "file_sha256": archive_sha256,
                },
            }
            raw = canonical_json_bytes(
                {
                    **body,
                    "canonical_identity_sha256": canonical_sha256(body),
                }
            ) + b"\n"
            objects[(archive_key, archive_version_id)] = archive_raw
            archive_manifest_value = json.loads(raw)
            archive_manifest_raw = raw
        else:
            raw = _canonical_record(kind)
        coordinate = {
            "artifact_kind": kind,
            "bucket": BUCKET,
            "key": (
                "task13/migration/fence-transfer.json"
                if kind == "FENCE_TEMPLATE"
                else source["key"]
            ),
            "version_id": version_id,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "body_sha256": (
                SEMANTIC_GATE_PINS[kind]["body_sha256"]
                if kind in SEMANTIC_GATE_PINS
                else (
                    json.loads(raw)["canonical_identity_sha256"]
                    if kind == "REPOSITORY_ARCHIVE"
                    else hashlib.sha256(raw[:-1]).hexdigest()
                )
            ),
        }
        coordinates.append(coordinate)
        objects[(str(coordinate["key"]), version_id)] = raw
        if kind == "REPOSITORY_ARCHIVE":
            archive_coordinate = coordinate

    staged_fixture._published_runtime_coordinate = (
        exact_runtime_coordinate
    )
    try:
        predecessor_request = dict(production_request)
        predecessor_request["artifacts"] = [
            row
            for row in coordinates
            if row["artifact_kind"]
            not in PRODUCTION_ONLY_ARTIFACT_KINDS
        ]
        predecessor_request["staged_infrastructure_evidence"] = (
            fixture._staged_evidence(
                predecessor_request["artifacts"]
            )
        )
        predecessor = build_campaign_package(predecessor_request)

        from glm52_enforcement.task13_clean_rehearsal import (
            build_clean_rehearsal_evidence,
            canonical_clean_rehearsal_evidence_bytes,
            clean_rehearsal_coordinate,
        )
        from test_glm52_task13_clean_rehearsal import (
            _fixture as clean_fixture,
        )

        clean_inputs = clean_fixture()
        assert type(archive_coordinate) is dict
        assert type(archive_manifest_value) is dict
        assert type(archive_manifest_raw) is bytes
        evidence = build_clean_rehearsal_evidence(
            predecessor_package=predecessor,
            predecessor_package_bytes=canonical_campaign_package_bytes(
                predecessor
            ),
            predecessor_reviewed_artifacts=predecessor[
                "reviewed_artifacts"
            ],
            repository_archive_coordinate=archive_coordinate,
            repository_archive_manifest=archive_manifest_value,
            repository_archive_manifest_bytes=archive_manifest_raw,
            finalize_invoke_result=clean_inputs[
                "finalize_invoke_result"
            ],
            gate_readback=clean_inputs["gate_readback"],
            gate_readback_bytes=clean_inputs["gate_readback_bytes"],
        )
        clean_raw = canonical_clean_rehearsal_evidence_bytes(evidence)
        clean_coordinate = clean_rehearsal_coordinate(
            evidence=evidence,
            version_id="reviewed-clean-rehearsal",
        )
        coordinates.append(clean_coordinate)
        objects[
            (
                str(clean_coordinate["key"]),
                str(clean_coordinate["version_id"]),
            )
        ] = clean_raw
        production_request["artifacts"] = coordinates
        production_request["staged_infrastructure_evidence"] = (
            fixture._staged_evidence(coordinates)
        )
        package = build_campaign_package(
            production_request,
            predecessor_package=predecessor,
            predecessor_reviewed_artifacts=predecessor[
                "reviewed_artifacts"
            ],
            clean_rehearsal_evidence=evidence,
        )
    finally:
        staged_fixture._published_runtime_coordinate = (
            original_runtime_coordinate
        )
    if phase == "PREQUALIFICATION":
        return predecessor, objects
    assert phase == "PRODUCTION"
    return package, objects


def _base_inventory() -> dict[str, object]:
    return build_s3_artifact_inventory(
        run_id=RUN_ID,
        bucket=BUCKET,
        objects=[
            {
                "key": "source-snapshot/config.json",
                "size": 12,
                "sha256": "a" * 64,
                "kind": "source_model",
                "safetensors": False,
                "run_scope": "shared",
                "version_id": "source-config-version",
            }
        ],
    )


def _staged_coordinate(
    package: dict[str, object],
    objects: dict[tuple[str, str], bytes],
) -> dict[str, object]:
    deployment = package["disabled_deployment"]
    assert isinstance(deployment, dict)
    evidence = deployment["staged_infrastructure_evidence"]
    raw = canonical_json_bytes(evidence) + b"\n"
    key = (
        f"campaigns/{RUN_ID}/task13/"
        f"{package['canonical_identity_sha256']}/"
        "staged-infrastructure-evidence-v1.json"
    )
    version_id = "staged-evidence-version"
    objects[(key, version_id)] = raw
    return {
        "schema_version": 1,
        "record_type": "glm52_task13_staged_evidence_coordinate_v1",
        "account_id": ACCOUNT,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": package["activation_id"],
        "package_identity_sha256": package["canonical_identity_sha256"],
        "bucket": BUCKET,
        "key": key,
        "version_id": version_id,
        "size_bytes": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_bridge_exact_reads_reviewed_archive_payload_and_staged_evidence() -> None:
    """Break caught: the inventory lists only Task 13 manifest aliases."""

    api = _api()
    package, objects = _production_package_and_objects()
    staged = _staged_coordinate(package, objects)
    s3 = ExactReadS3(objects)

    inventory = api.build_task13_production_inventory(
        base_inventory=_base_inventory(),
        campaign_package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        staged_evidence_coordinate=staged,
        services=api.Task13ProductionInventoryServices(
            s3=s3,
            total_max_attempts=1,
        ),
    )

    assert validate_s3_artifact_inventory(inventory) == inventory
    rows = {row["key"]: row for row in inventory["objects"]}
    archive_manifest = next(
        row
        for row in package["reviewed_artifacts"]
        if row["artifact_kind"] == "REPOSITORY_ARCHIVE"
    )
    manifest = json.loads(
        objects[
            (
                str(archive_manifest["key"]),
                str(archive_manifest["version_id"]),
            )
        ]
    )
    archive = manifest["archive"]
    assert rows[archive["key"]] == {
        "key": archive["key"],
        "size": manifest["archive_size_bytes"],
        "sha256": archive["file_sha256"],
        "kind": "repository_tar",
        "safetensors": False,
        "run_scope": "shared",
        "version_id": archive["version_id"],
    }
    assert rows[str(staged["key"])]["kind"] == "approval"
    assert rows[str(staged["key"])]["version_id"] == staged["version_id"]
    staged_value = json.loads(
        objects[(str(staged["key"]), str(staged["version_id"]))]
    )
    for field, raw in (
        ("support_lambda_archive", SUPPORT_LAMBDA_ARCHIVE_RAW),
        (
            "cryptography_layer_archive",
            CRYPTOGRAPHY_LAYER_ARCHIVE_RAW,
        ),
    ):
        coordinate = staged_value["support_stack"][field]
        assert rows[coordinate["key"]] == {
            "key": coordinate["key"],
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "kind": "watchdog_code",
            "safetensors": False,
            "run_scope": "shared",
            "version_id": coordinate["version_id"],
        }
    expected_kinds = {
        "ACCEPTED_BASELINE": "training_baseline",
        "BOOTSTRAP_TEMPLATE": "watchdog_code",
        "CLEAN_REHEARSAL": "approval",
        "FENCE_TEMPLATE": "watchdog_code",
        "GPU_SPEND_APPROVAL": "approval",
        "H100_QUALIFICATION_INPUT": "qualification_cache",
        "PRODUCTION_DESCRIPTOR": "campaign_descriptor",
        "PROMPT_PACK": "prompt_pack",
        "QUALIFICATION_CACHE_SEED_INPUT": "qualification_cache",
        "REPOSITORY_ARCHIVE": "campaign_descriptor",
        "RESIDUAL_LIABILITY_APPROVAL": "approval",
        "RETAINED_FOUNDATION_TEMPLATE": "watchdog_code",
        "RETAINED_PRE_SUPPORT_TEMPLATE": "watchdog_code",
        "RETAINED_TEMPLATE": "watchdog_code",
        "SUPPORT_APPROVAL": "approval",
        "SUPPORT_INPUTS": "training_configuration",
        "SUPPORT_TEMPLATE": "watchdog_code",
        "T01_T25_GATE": "approval",
        "TASK10_PRODUCTION_AUTHORITY": "approval",
        "TASK10_TASK_INPUTS": "campaign_descriptor",
        "TASK10_WORKER_DESCRIPTOR": "campaign_descriptor",
        "TASK11_REVIEW_APPROVAL": "approval",
        "TASK12_REVIEW_APPROVAL": "approval",
        "TRAINING_CONFIGURATION": "training_configuration",
        "TRANSPORT_22_MUTANT_GATE": "approval",
    }
    assert {
        coordinate["artifact_kind"]: rows[str(coordinate["key"])]["kind"]
        for coordinate in package["reviewed_artifacts"]
    } == expected_kinds
    assert len(s3.requests) == len(package["reviewed_artifacts"]) + 4
    assert all(
        request
        == {
            "Bucket": BUCKET,
            "Key": request["Key"],
            "VersionId": request["VersionId"],
            "ExpectedBucketOwner": ACCOUNT,
            "ChecksumMode": "ENABLED",
        }
        for request in s3.requests
    )


def test_complete_inventory_projects_into_sky_object_authorities(
    tmp_path: Path,
) -> None:
    """Break caught: campaign-scoped rows make the Sky projection unusable."""

    api = _api()
    package, objects = _production_package_and_objects()
    inventory = api.build_task13_production_inventory(
        base_inventory=_base_inventory(),
        campaign_package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        staged_evidence_coordinate=_staged_coordinate(package, objects),
        services=api.Task13ProductionInventoryServices(
            s3=ExactReadS3(objects),
            total_max_attempts=1,
        ),
    )
    inventory_path = tmp_path / "inventory.json"
    projected_path = tmp_path / "objects.json"
    inventory_path.write_bytes(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )

    completed = subprocess.run(
        [
            str(
                ROOT
                / "aws/glm52-gpu/scripts/"
                "project_glm52_production_inventory_objects.py"
            ),
            "--inventory",
            str(inventory_path),
            "--output",
            str(projected_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    projected = json.loads(projected_path.read_bytes())
    repository_rows = [
        row
        for row in inventory["objects"]
        if row["kind"] == "repository_tar"
    ]
    assert len(repository_rows) == 1
    assert projected == inventory["objects"]


def _project_inventory(
    tmp_path: Path,
    inventory: dict[str, object],
) -> subprocess.CompletedProcess[str]:
    inventory_path = tmp_path / "project-input.json"
    projected_path = tmp_path / "projected-objects.json"
    inventory_path.write_bytes(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    return subprocess.run(
        [
            str(
                ROOT
                / "aws/glm52-gpu/scripts/"
                "project_glm52_production_inventory_objects.py"
            ),
            "--inventory",
            str(inventory_path),
            "--output",
            str(projected_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("failure", ("zero", "multiple", "mismatched"))
def test_sky_projection_requires_one_canonical_repository_owner(
    tmp_path: Path,
    failure: str,
) -> None:
    """Break caught: the projector drops an absent or ambiguous repo row."""

    base_row = dict(_base_inventory()["objects"][0])
    repository_row = {
        "key": (
            f"campaigns/{RUN_ID}/repository/keep-"
            f"{'b' * 64}.tar.gz"
        ),
        "size": 20,
        "sha256": "b" * 64,
        "kind": "repository_tar",
        "safetensors": False,
        "run_scope": "shared",
        "version_id": "repo-version",
    }
    rows = [base_row]
    if failure != "zero":
        rows.append(repository_row)
    if failure == "multiple":
        rows.append(
            {
                **repository_row,
                "key": (
                    f"campaigns/{RUN_ID}/repository/keep-"
                    f"{'c' * 64}.tar.gz"
                ),
                "sha256": "c" * 64,
                "version_id": "other-repo-version",
            }
        )
    elif failure == "mismatched":
        rows[-1] = {
            **repository_row,
            "key": "task13/archive/repo.tar.gz",
        }
    inventory = build_s3_artifact_inventory(
        run_id=RUN_ID,
        bucket=BUCKET,
        objects=rows,
    )

    completed = _project_inventory(tmp_path, inventory)

    assert completed.returncode != 0
    assert "repository tar" in completed.stderr


def test_projected_inventory_drives_real_sky_bundle_without_repo_duplicate(
    tmp_path: Path,
) -> None:
    """Break caught: the complete bridge inventory duplicates the bundle repo."""

    api = _api()
    package, objects = _production_package_and_objects()
    inventory = api.build_task13_production_inventory(
        base_inventory=_base_inventory(),
        campaign_package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        staged_evidence_coordinate=_staged_coordinate(
            package,
            objects,
        ),
        services=api.Task13ProductionInventoryServices(
            s3=ExactReadS3(objects),
            total_max_attempts=1,
        ),
    )
    inventory_path = tmp_path / "inventory.json"
    projected_path = tmp_path / "objects.json"
    inventory_path.write_bytes(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    projected = subprocess.run(
        [
            str(
                ROOT
                / "aws/glm52-gpu/scripts/"
                "project_glm52_production_inventory_objects.py"
            ),
            "--inventory",
            str(inventory_path),
            "--output",
            str(projected_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert projected.returncode == 0, projected.stderr

    repository_manifest_coordinate = next(
        row
        for row in package["reviewed_artifacts"]
        if row["artifact_kind"] == "REPOSITORY_ARCHIVE"
    )
    repository_manifest = json.loads(
        objects[
            (
                str(repository_manifest_coordinate["key"]),
                str(repository_manifest_coordinate["version_id"]),
            )
        ]
    )
    repository_coordinate = repository_manifest["archive"]
    repository_source = tmp_path / "reviewed-repo.tar.gz"
    repository_source.write_bytes(
        objects[
            (
                str(repository_coordinate["key"]),
                str(repository_coordinate["version_id"]),
            )
        ]
    )
    approval_source = tmp_path / "GPU_SPEND_APPROVAL.json"
    approval = subprocess.run(
        [
            str(
                ROOT
                / "aws/glm52-gpu/scripts/"
                "build_gpu_spend_approval.py"
            ),
            "--ingested-at",
            "2026-07-24T01:05:00Z",
            "--output",
            str(approval_source),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert approval.returncode == 0, approval.stderr
    output = tmp_path / "bundle"
    clean_env = {
        "PATH": os.environ["PATH"],
        **(
            {"TMPDIR": os.environ["TMPDIR"]}
            if os.environ.get("TMPDIR")
            else {}
        ),
    }
    bundle = subprocess.run(
        [
            str(
                ROOT
                / "aws/glm52-gpu/scripts/build_sky_campaign_bundle.sh"
            )
        ],
        cwd=ROOT,
        env={
            **clean_env,
            "RUN_ID": RUN_ID,
            "BUCKET": BUCKET,
            "MUST_START_BY": "2026-07-30T01:05:00Z",
            "IMAGE_ID": "ami-0123456789abcdef0",
            "OBJECT_AUTHORITIES_JSON": str(projected_path),
            "SOURCE_SNAPSHOT_PREFIX": "source-snapshot/",
            "SOURCE_SNAPSHOT_SHA256": "2" * 64,
            "NON_VQ_PREFIX": "non-vq-package/",
            "NON_VQ_PACKAGE_SHA256": "3" * 64,
            "TEICH_PACK_KEY": "teich-pack/pack.json",
            "TEICH_PACK_SHA256": "4" * 64,
            "FROZEN_PROMPT_PACK_KEY": "quality/frozen.json",
            "FROZEN_PROMPT_PACK_SHA256": "5" * 64,
            "TRAINING_BASELINE_PREFIX": "training-baseline/",
            "TRAINING_BASELINE_SHA256": "6" * 64,
            "QUALIFICATION_CACHE_PREFIX": (
                f"qualification-cache/seeds/{RUN_ID}/"
                f"{'7' * 64}/"
            ),
            "QUALIFICATION_CACHE_MANIFEST_SHA256": "7" * 64,
            "SUBMISSION_ID": "task13-inventory-integration",
            "OUTPUT_DIR": str(output),
            "APPROVAL_SOURCE": str(approval_source),
            "EXPECTED_APPROVAL_SHA256": hashlib.sha256(
                approval_source.read_bytes()
            ).hexdigest(),
            "REPO_TAR_SOURCE": str(repository_source),
            "EXPECTED_REPO_TAR_SHA256": str(
                repository_coordinate["file_sha256"]
            ),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert bundle.returncode == 0, bundle.stderr
    final_inventory = validate_s3_artifact_inventory(
        json.loads(
            (output / "artifact-inventory-v1.json").read_bytes()
        )
    )
    repo_rows = [
        row
        for row in final_inventory["objects"]
        if row["kind"] == "repository_tar"
    ]
    expected_repository_row = next(
        row
        for row in inventory["objects"]
        if row["kind"] == "repository_tar"
    )
    assert repo_rows == [expected_repository_row]
    shared_inventory = build_s3_artifact_inventory(
        run_id=RUN_ID,
        bucket=BUCKET,
        objects=[
            row
            for row in final_inventory["objects"]
            if row["run_scope"] == "shared"
        ],
    )
    assert shared_inventory == inventory
    assert (
        canonical_json_bytes(shared_inventory) + b"\n"
        == inventory_path.read_bytes()
    )

    checksum_body = {
        "schema_version": 1,
        "record_type": "glm52_s3_batch_checksum_authority_v1",
        "account_id": ACCOUNT,
        "region": REGION,
        "bucket": BUCKET,
        "job_id": "task13-inventory-integration",
        "inventory_body_sha256": inventory["inventory_body_sha256"],
        "checksum_algorithm": "SHA256",
        "checksum_type": "FULL_OBJECT",
        "checksums": {
            str(row["key"]): str(row["sha256"])
            for row in inventory["objects"]
        },
    }
    checksum_authority = {
        **checksum_body,
        "authority_body_sha256": hashlib.sha256(
            canonical_json_bytes(checksum_body)
        ).hexdigest(),
    }
    checksum_path = tmp_path / "checksum-authority.json"
    checksum_path.write_bytes(
        canonical_json_bytes(checksum_authority) + b"\n"
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_aws = fake_bin / "aws"
    fake_aws.write_text(
        """#!/usr/bin/env python3
import base64
import json
import os
import pathlib
import sys

arguments = sys.argv[1:]
if arguments[:2] == ["sts", "get-caller-identity"]:
    print(json.dumps({
        "Account": "246813579024",
        "Arn": "arn:aws:sts::246813579024:assumed-role/test/session",
    }))
elif arguments[:2] == ["s3api", "list-multipart-uploads"]:
    print(json.dumps({"Uploads": []}))
elif arguments[:2] == ["s3api", "head-object"]:
    key = arguments[arguments.index("--key") + 1]
    inventory = json.loads(
        pathlib.Path(os.environ["FAKE_INVENTORY"]).read_bytes()
    )
    item = next(row for row in inventory["objects"] if row["key"] == key)
    response = {
        "ContentLength": item["size"],
        "ETag": "\\"fixture\\"",
        "ChecksumSHA256": base64.b64encode(
            bytes.fromhex(item["sha256"])
        ).decode("ascii"),
        "ChecksumType": "FULL_OBJECT",
        "Metadata": {},
    }
    if "version_id" in item:
        response["VersionId"] = item["version_id"]
    print(json.dumps(response))
else:
    raise SystemExit("unexpected fake AWS command: " + repr(arguments))
""",
        encoding="utf-8",
    )
    fake_aws.chmod(0o755)
    audit_output = tmp_path / "checksum-audit.json"
    audited = subprocess.run(
        [
            str(
                ROOT
                / "aws/glm52-gpu/scripts/audit_s3_campaign_artifacts.py"
            ),
            "--inventory",
            str(output / "artifact-inventory-v1.json"),
            "--output",
            str(audit_output),
            "--profile",
            "keep-gpu",
            "--region",
            REGION,
            "--checksum-authority",
            str(checksum_path),
        ],
        cwd=ROOT,
        env={
            **clean_env,
            "PATH": str(fake_bin) + os.pathsep + clean_env["PATH"],
            "AWS_PROFILE": "keep-gpu",
            "FAKE_INVENTORY": str(
                output / "artifact-inventory-v1.json"
            ),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert audited.returncode == 0, audited.stderr
    assert json.loads(audit_output.read_bytes())["audit_pass"] is True


def test_bridge_accepts_prequalification_post_deploy_inventory() -> None:
    """Break caught: post-deploy inventory waits for Task10 successor rows."""

    api = _api()
    package, objects = _production_package_and_objects(
        phase="PREQUALIFICATION",
    )
    inventory = api.build_task13_production_inventory(
        base_inventory=_base_inventory(),
        campaign_package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        staged_evidence_coordinate=_staged_coordinate(
            package,
            objects,
        ),
        services=api.Task13ProductionInventoryServices(
            s3=ExactReadS3(objects),
            total_max_attempts=1,
        ),
    )

    reviewed_kinds = {
        row["artifact_kind"]
        for row in package["reviewed_artifacts"]
    }
    assert package["package_phase"] == "PREQUALIFICATION"
    assert not reviewed_kinds.intersection(
        PRODUCTION_ONLY_ARTIFACT_KINDS
    )
    assert validate_s3_artifact_inventory(inventory) == inventory


def test_bridge_rejects_staged_support_archive_byte_drift() -> None:
    """Break caught: deployed support code is listed but never authenticated."""

    api = _api()
    package, objects = _production_package_and_objects()
    deployment = package["disabled_deployment"]
    coordinate = deployment["staged_infrastructure_evidence"][
        "support_stack"
    ]["support_lambda_archive"]
    object_key = (
        str(coordinate["key"]),
        str(coordinate["version_id"]),
    )
    original = objects[object_key]
    objects[object_key] = original[:-1] + bytes([original[-1] ^ 1])

    with pytest.raises(
        api.Task13ProductionInventoryError,
        match="file SHA-256 drifted",
    ):
        api.build_task13_production_inventory(
            base_inventory=_base_inventory(),
            campaign_package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            staged_evidence_coordinate=_staged_coordinate(
                package,
                objects,
            ),
            services=api.Task13ProductionInventoryServices(
                s3=ExactReadS3(objects),
                total_max_attempts=1,
            ),
        )


def test_bridge_rejects_unversioned_base_object() -> None:
    """Break caught: Task 13 is merged into a Batch-ineligible base inventory."""

    api = _api()
    package, objects = _production_package_and_objects()
    base = _base_inventory()
    del base["objects"][0]["version_id"]
    unsigned = dict(base)
    unsigned.pop("inventory_body_sha256")
    base["inventory_body_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()

    with pytest.raises(
        api.Task13ProductionInventoryError,
        match="VersionId",
    ):
        api.build_task13_production_inventory(
            base_inventory=base,
            campaign_package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            staged_evidence_coordinate=_staged_coordinate(
                package,
                objects,
            ),
            services=api.Task13ProductionInventoryServices(
                s3=ExactReadS3(objects),
                total_max_attempts=1,
            ),
        )


@pytest.mark.parametrize(
    ("s3_type", "message"),
    [
        (VersionDriftS3, "VersionId drifted"),
        (MissingSizeS3, "size is missing"),
    ],
)
def test_bridge_rejects_incomplete_exact_readback(
    s3_type: type[ExactReadS3],
    message: str,
) -> None:
    """Break caught: current-version or sizeless reads become authority."""

    api = _api()
    package, objects = _production_package_and_objects()
    with pytest.raises(
        api.Task13ProductionInventoryError,
        match=message,
    ):
        api.build_task13_production_inventory(
            base_inventory=_base_inventory(),
            campaign_package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            staged_evidence_coordinate=_staged_coordinate(
                package,
                objects,
            ),
            services=api.Task13ProductionInventoryServices(
                s3=s3_type(objects),
                total_max_attempts=1,
            ),
        )


def test_bridge_rejects_foreign_package_binding_and_duplicate_key_drift() -> None:
    """Break caught: staged or base truth aliases a different package/version."""

    api = _api()
    package, objects = _production_package_and_objects()
    staged = _staged_coordinate(package, objects)
    staged["package_identity_sha256"] = "f" * 64
    with pytest.raises(
        api.Task13ProductionInventoryError,
        match="staged evidence coordinate identity",
    ):
        api.build_task13_production_inventory(
            base_inventory=_base_inventory(),
            campaign_package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            staged_evidence_coordinate=staged,
            services=api.Task13ProductionInventoryServices(
                s3=ExactReadS3(objects),
                total_max_attempts=1,
            ),
        )

    duplicate = package["reviewed_artifacts"][0]
    base = build_s3_artifact_inventory(
        run_id=RUN_ID,
        bucket=BUCKET,
        objects=[
            {
                "key": duplicate["key"],
                "size": 1,
                "sha256": "a" * 64,
                "kind": "approval",
                "safetensors": False,
                "run_scope": "shared",
                "version_id": "foreign-version",
            }
        ],
    )
    with pytest.raises(
        api.Task13ProductionInventoryError,
        match="duplicate inventory key.*drift",
    ):
        api.build_task13_production_inventory(
            base_inventory=base,
            campaign_package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            staged_evidence_coordinate=_staged_coordinate(
                package,
                objects,
            ),
            services=api.Task13ProductionInventoryServices(
                s3=ExactReadS3(objects),
                total_max_attempts=1,
            ),
        )


def test_file_inputs_are_canonical_and_inventory_output_is_create_only(
    tmp_path: Path,
) -> None:
    """Break caught: pretty JSON or a replacement output is silently adopted."""

    api = _api()
    package, _objects = _production_package_and_objects()
    noncanonical = (tmp_path / "package.json").resolve()
    noncanonical.write_text(
        json.dumps(package, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        api.Task13ProductionInventoryError,
        match="canonical",
    ):
        api._parse_package(noncanonical)

    output = (tmp_path / "inventory.json").resolve()
    api.write_inventory_once(output, _base_inventory())
    assert os.stat(output).st_mode & 0o777 == 0o600
    with pytest.raises(
        api.Task13ProductionInventoryError,
        match="new absolute file",
    ):
        api.write_inventory_once(output, _base_inventory())
