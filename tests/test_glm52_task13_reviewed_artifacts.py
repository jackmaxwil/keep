from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_json_bytes


ACCOUNT = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
RUN_ID = "glm52-sky-20260724"
ACTIVATION = "approved-20260728"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "aws/glm52-gpu/scripts/"
    "publish_glm52_task13_reviewed_artifact.py"
)

ARTIFACT_KEYS = {
    "TASK11_REVIEW_APPROVAL": "reviews/task11/approval.json",
    "TASK12_REVIEW_APPROVAL": "reviews/task12/approval.json",
    "RETAINED_TEMPLATE": "task13/templates/retained.yaml",
    "FENCE_TEMPLATE": "task13/migration/fence-transfer.json",
    "SUPPORT_TEMPLATE": "task13/templates/support-disabled.yaml",
    "BOOTSTRAP_TEMPLATE": "task13/templates/container-bootstrap-v1.json",
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
    "TASK10_WORKER_DESCRIPTOR": (
        "task13/production/task10-worker-descriptor.json"
    ),
    "TASK10_TASK_INPUTS": "task13/production/task10-task-inputs.json",
}


def _metadata(label: str) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RequestId": label,
        "RetryAttempts": 0,
    }


class ExactSts:
    def get_caller_identity(self) -> object:
        return {
            "Account": ACCOUNT,
            "Arn": (
                "arn:aws:sts::246813579024:"
                "assumed-role/task13-reviewed-artifacts/test"
            ),
            "UserId": "AROATASK13:test",
            "ResponseMetadata": _metadata("sts"),
        }


class FalseRetrySts(ExactSts):
    def get_caller_identity(self) -> object:
        response = super().get_caller_identity()
        assert type(response) is dict
        response["ResponseMetadata"]["RetryAttempts"] = False
        return response


class LostPut(RuntimeError):
    pass


class ExactS3:
    def __init__(
        self,
        *,
        lose_after_put_key: str | None = None,
        page_size: int = 1000,
        versioning_status: str = "Enabled",
    ) -> None:
        self.versions: dict[str, list[dict[str, object]]] = {}
        self.delete_markers: list[dict[str, object]] = []
        self.put_requests: list[dict[str, object]] = []
        self.list_requests: list[dict[str, object]] = []
        self.get_requests: list[dict[str, object]] = []
        self.lose_after_put_key = lose_after_put_key
        self.page_size = page_size
        self.versioning_status = versioning_status
        self._version = 0

    def seed(
        self,
        key: str,
        raw: bytes,
        *,
        metadata: dict[str, str],
    ) -> str:
        self._version += 1
        version_id = f"version-{self._version:04d}"
        self.versions.setdefault(key, []).append(
            {
                "VersionId": version_id,
                "raw": raw,
                "Metadata": dict(metadata),
            }
        )
        return version_id

    def get_bucket_versioning(self, **request: object) -> object:
        assert request == {
            "Bucket": BUCKET,
            "ExpectedBucketOwner": ACCOUNT,
        }
        return {
            "Status": self.versioning_status,
            "ResponseMetadata": _metadata("versioning"),
        }

    def list_object_versions(self, **request: object) -> object:
        self.list_requests.append(dict(request))
        assert request["Bucket"] == BUCKET
        assert request["ExpectedBucketOwner"] == ACCOUNT
        prefix = str(request["Prefix"])
        flattened = [
            {
                "Key": key,
                "VersionId": row["VersionId"],
                "Size": len(row["raw"]),
            }
            for key, rows in sorted(self.versions.items())
            if key.startswith(prefix)
            for row in rows
        ]
        start = int(str(request.get("KeyMarker", "0")))
        versions = flattened[start : start + self.page_size]
        next_index = start + len(versions)
        truncated = next_index < len(flattened)
        return {
            "Versions": versions,
            "DeleteMarkers": [
                item
                for item in self.delete_markers
                if str(item.get("Key", "")).startswith(prefix)
            ],
            "IsTruncated": truncated,
            **(
                {
                    "NextKeyMarker": str(next_index),
                    "NextVersionIdMarker": "opaque-" + str(next_index),
                }
                if truncated
                else {}
            ),
            "ResponseMetadata": _metadata("list"),
        }

    def put_object(self, **request: object) -> object:
        self.put_requests.append(dict(request))
        assert request["Bucket"] == BUCKET
        assert request["ExpectedBucketOwner"] == ACCOUNT
        assert request["IfNoneMatch"] == "*"
        assert request["ChecksumAlgorithm"] == "SHA256"
        raw = request["Body"]
        assert type(raw) is bytes
        assert request["ChecksumSHA256"] == base64.b64encode(
            hashlib.sha256(raw).digest()
        ).decode("ascii")
        key = str(request["Key"])
        assert not self.versions.get(key)
        version_id = self.seed(
            key,
            raw,
            metadata=dict(request["Metadata"]),
        )
        if key == self.lose_after_put_key:
            self.lose_after_put_key = None
            raise LostPut("simulated lost PutObject response")
        return {
            "VersionId": version_id,
            "ChecksumSHA256": request["ChecksumSHA256"],
            "ResponseMetadata": _metadata("put-" + version_id),
        }

    def get_object(self, **request: object) -> object:
        self.get_requests.append(dict(request))
        assert request["Bucket"] == BUCKET
        assert request["ExpectedBucketOwner"] == ACCOUNT
        assert request["ChecksumMode"] == "ENABLED"
        key = str(request["Key"])
        version_id = str(request["VersionId"])
        row = next(
            item
            for item in self.versions[key]
            if item["VersionId"] == version_id
        )
        raw = row["raw"]
        assert type(raw) is bytes
        return {
            "Body": io.BytesIO(raw),
            "ContentLength": len(raw),
            "VersionId": version_id,
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(raw).digest()
            ).decode("ascii"),
            "Metadata": row["Metadata"],
            "ResponseMetadata": _metadata("get-" + version_id),
        }


class MissingVersionPutS3(ExactS3):
    def put_object(self, **request: object) -> object:
        response = super().put_object(**request)
        assert type(response) is dict
        response.pop("VersionId")
        return response


def _pins(raw: bytes) -> dict[str, str]:
    return {
        "expected_file_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_body_sha256": hashlib.sha256(raw[:-1]).hexdigest(),
    }


def _review_metadata(artifact_kind: str, raw: bytes) -> dict[str, str]:
    return {
        "artifact-kind": artifact_kind,
        "body-sha256": hashlib.sha256(raw[:-1]).hexdigest(),
        "file-sha256": hashlib.sha256(raw).hexdigest(),
        "glm52-run-id": RUN_ID,
    }


def _services(s3: ExactS3):
    from glm52_enforcement.task13_reviewed_artifacts import (
        Task13ReviewedArtifactServices,
    )

    return Task13ReviewedArtifactServices(
        sts=ExactSts(),
        s3=s3,
        total_max_attempts=1,
    )


@pytest.mark.parametrize(
    ("artifact_kind", "key"),
    sorted(
        item
        for item in ARTIFACT_KEYS.items()
        if item[0] != "CLEAN_REHEARSAL"
    ),
)
def test_closed_reviewed_artifact_map_publishes_exact_canonical_version(
    tmp_path: Path,
    artifact_kind: str,
    key: str,
) -> None:
    """Break caught: an allowed kind publishes mutable or noncanonical truth."""

    from glm52_enforcement.task13_reviewed_artifacts import (
        publish_reviewed_artifact,
    )

    value = (
        {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Resources": {
                "Anchor": {
                    "Type": "AWS::CloudFormation::WaitConditionHandle"
                }
            },
        }
        if artifact_kind.endswith("_TEMPLATE")
        else {
            "artifact_kind": artifact_kind,
            "record_type": "fixture-reviewed-artifact",
            "schema_version": 1,
        }
    )
    raw = canonical_json_bytes(value) + b"\n"
    source = tmp_path / (artifact_kind.lower() + ".json")
    source.write_bytes(raw)
    s3 = ExactS3()

    coordinate = publish_reviewed_artifact(
        artifact_kind=artifact_kind,
        source_path=source,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        expected_body_sha256=hashlib.sha256(raw[:-1]).hexdigest(),
        bucket=BUCKET,
        services=_services(s3),
        activation_id=(
            ACTIVATION if artifact_kind == "PRODUCTION_DESCRIPTOR" else None
        ),
    )

    assert coordinate == {
        "artifact_kind": artifact_kind,
        "bucket": BUCKET,
        "key": key,
        "version_id": "version-0001",
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": hashlib.sha256(raw[:-1]).hexdigest(),
    }
    request = s3.put_requests[0]
    assert request["Key"] == key
    assert request["ContentType"] == "application/json"
    assert request["Metadata"] == {
        "artifact-kind": artifact_kind,
        "body-sha256": coordinate["body_sha256"],
        "file-sha256": coordinate["file_sha256"],
        "glm52-run-id": RUN_ID,
    }
    assert [item["Prefix"] for item in s3.list_requests] == [key, key]
    assert s3.get_requests == [
        {
            "Bucket": BUCKET,
            "Key": key,
            "VersionId": "version-0001",
            "ExpectedBucketOwner": ACCOUNT,
            "ChecksumMode": "ENABLED",
        }
    ]


def test_production_descriptor_requires_exact_activation_lineage(
    tmp_path: Path,
) -> None:
    """Break caught: stale global or foreign-activation descriptors are adopted."""

    from glm52_enforcement.task13_reviewed_artifacts import (
        publish_reviewed_artifact,
        write_coordinate_once,
    )

    raw = canonical_json_bytes(
        {
            "record_type": "glm52_sky_campaign_descriptor_v2",
            "schema_version": 2,
        }
    ) + b"\n"
    source = tmp_path / "campaign-descriptor-v2.json"
    source.write_bytes(raw)
    s3 = ExactS3()

    coordinate = publish_reviewed_artifact(
        artifact_kind="PRODUCTION_DESCRIPTOR",
        activation_id=ACTIVATION,
        source_path=source,
        **_pins(raw),
        bucket=BUCKET,
        services=_services(s3),
    )

    assert coordinate["key"] == ARTIFACT_KEYS["PRODUCTION_DESCRIPTOR"]
    assert all(
        request.get("Prefix") != "task13/inputs/campaign-descriptor-v2.json"
        and request.get("Key") != "task13/inputs/campaign-descriptor-v2.json"
        for request in (*s3.list_requests, *s3.put_requests, *s3.get_requests)
    )
    with pytest.raises(ValueError, match="activation"):
        publish_reviewed_artifact(
            artifact_kind="PRODUCTION_DESCRIPTOR",
            source_path=source,
            **_pins(raw),
            bucket=BUCKET,
            services=_services(ExactS3()),
        )

    legacy_coordinate = {
        **coordinate,
        "key": "task13/inputs/campaign-descriptor-v2.json",
    }
    with pytest.raises(ValueError, match="coordinate.*exact|activation|key"):
        write_coordinate_once(
            tmp_path / "legacy-coordinate.json",
            legacy_coordinate,
        )

def test_clean_rehearsal_uses_generic_protocol_at_new_identity_key(
    tmp_path: Path,
) -> None:
    from glm52_enforcement.task13_clean_rehearsal import (
        canonical_clean_rehearsal_evidence_bytes,
        clean_rehearsal_artifact_key,
    )
    from glm52_enforcement.task13_reviewed_artifacts import (
        REVIEWED_ARTIFACT_KEYS,
        publish_reviewed_artifact,
    )
    from test_glm52_task13_clean_rehearsal import _build, _fixture

    evidence = _build(_fixture())
    raw = canonical_clean_rehearsal_evidence_bytes(evidence)
    source = tmp_path / "clean-rehearsal.json"
    source.write_bytes(raw)
    s3 = ExactS3()

    coordinate = publish_reviewed_artifact(
        artifact_kind="CLEAN_REHEARSAL",
        source_path=source,
        **_pins(raw),
        bucket=BUCKET,
        services=_services(s3),
    )

    expected_key = clean_rehearsal_artifact_key(evidence)
    assert coordinate["key"] == expected_key
    assert "CLEAN_REHEARSAL" not in REVIEWED_ARTIFACT_KEYS
    assert s3.versions[expected_key][0]["raw"] == raw
    assert all(
        request["Key"] != "task13/gates/clean-rehearsal.json"
        for request in (*s3.put_requests, *s3.get_requests)
    )


def test_coherent_rehash_cannot_replace_externally_reviewed_bytes(
    tmp_path: Path,
) -> None:
    """Break caught: a rewritten self-hashed record replaces reviewed bytes."""

    from glm52_enforcement.task13_reviewed_artifacts import (
        publish_reviewed_artifact,
    )

    reviewed_body = {
        "decision": "APPROVED",
        "record_type": "glm52_task13_review_approval_v1",
        "schema_version": 1,
    }
    reviewed_body["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(reviewed_body)
    ).hexdigest()
    reviewed_raw = canonical_json_bytes(reviewed_body) + b"\n"
    rewritten_body = {**reviewed_body, "decision": "REWRITTEN"}
    rewritten_unsigned = dict(rewritten_body)
    rewritten_unsigned.pop("canonical_identity_sha256")
    rewritten_body["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(rewritten_unsigned)
    ).hexdigest()
    source = tmp_path / "task11-review.json"
    source.write_bytes(canonical_json_bytes(rewritten_body) + b"\n")
    s3 = ExactS3()

    with pytest.raises(ValueError, match="reviewed.*SHA-256|identity"):
        publish_reviewed_artifact(
            artifact_kind="TASK11_REVIEW_APPROVAL",
            source_path=source,
            expected_file_sha256=hashlib.sha256(reviewed_raw).hexdigest(),
            expected_body_sha256=hashlib.sha256(reviewed_raw[:-1]).hexdigest(),
            bucket=BUCKET,
            services=_services(s3),
        )

    assert s3.list_requests == []
    assert s3.put_requests == []


def test_lost_put_reconciles_one_version_through_paginated_siblings(
    tmp_path: Path,
) -> None:
    """Break caught: ambiguous publication retries or ignores later pages."""

    from glm52_enforcement.task13_reviewed_artifacts import (
        publish_reviewed_artifact,
    )

    artifact_kind = "GPU_SPEND_APPROVAL"
    key = ARTIFACT_KEYS[artifact_kind]
    value = {
        "record_type": "glm52_gpu_spend_approval_v1",
        "schema_version": 1,
    }
    raw = canonical_json_bytes(value) + b"\n"
    source = tmp_path / "gpu-spend.json"
    source.write_bytes(raw)
    s3 = ExactS3(lose_after_put_key=key, page_size=1)
    s3.seed(
        key + ".sibling-a",
        b"first sibling",
        metadata={"foreign": "true"},
    )
    s3.seed(
        key + ".sibling-b",
        b"second sibling",
        metadata={"foreign": "true"},
    )

    first = publish_reviewed_artifact(
        artifact_kind=artifact_kind,
        source_path=source,
        **_pins(raw),
        bucket=BUCKET,
        services=_services(s3),
    )
    puts_after_lost_response = len(s3.put_requests)
    second = publish_reviewed_artifact(
        artifact_kind=artifact_kind,
        source_path=source,
        **_pins(raw),
        bucket=BUCKET,
        services=_services(s3),
    )

    assert first == second
    assert first["version_id"] == "version-0003"
    assert puts_after_lost_response == 1
    assert len(s3.put_requests) == 1
    assert any("KeyMarker" in request for request in s3.list_requests)


def test_malformed_success_response_reconciles_the_single_durable_version(
    tmp_path: Path,
) -> None:
    """Break caught: a persisted PUT with malformed 200 is not reconciled."""

    from glm52_enforcement.task13_reviewed_artifacts import (
        publish_reviewed_artifact,
    )

    raw = canonical_json_bytes({"record_type": "fixture"}) + b"\n"
    source = tmp_path / "task12-review.json"
    source.write_bytes(raw)
    s3 = MissingVersionPutS3()

    coordinate = publish_reviewed_artifact(
        artifact_kind="TASK12_REVIEW_APPROVAL",
        source_path=source,
        **_pins(raw),
        bucket=BUCKET,
        services=_services(s3),
    )

    assert coordinate["version_id"] == "version-0001"
    assert len(s3.put_requests) == 1


@pytest.mark.parametrize("history_kind", ["multiple", "delete-marker"])
def test_non_singular_fixed_key_history_is_never_adopted(
    tmp_path: Path,
    history_kind: str,
) -> None:
    """Break caught: multiple versions or deleted truth is treated as immutable."""

    from glm52_enforcement.task13_reviewed_artifacts import (
        publish_reviewed_artifact,
    )

    artifact_kind = "SUPPORT_APPROVAL"
    key = ARTIFACT_KEYS[artifact_kind]
    raw = canonical_json_bytes(
        {"record_type": "PRODUCTION_SUPPORT_PLANE_APPROVAL"}
    ) + b"\n"
    source = tmp_path / "support-approval.json"
    source.write_bytes(raw)
    s3 = ExactS3()
    if history_kind == "multiple":
        metadata = _review_metadata(artifact_kind, raw)
        s3.seed(key, raw, metadata=metadata)
        s3.seed(key, raw, metadata=metadata)
        match = "singular"
    else:
        s3.delete_markers.append(
            {"Key": key, "VersionId": "deleted-version"}
        )
        match = "delete marker"

    with pytest.raises(ValueError, match=match):
        publish_reviewed_artifact(
            artifact_kind=artifact_kind,
            source_path=source,
            **_pins(raw),
            bucket=BUCKET,
            services=_services(s3),
        )

    assert s3.put_requests == []


def test_foreign_kind_bucket_source_and_existing_bytes_fail_before_put(
    tmp_path: Path,
) -> None:
    """Break caught: caller-selected keys or foreign bytes enter the fixed map."""

    from glm52_enforcement.task13_reviewed_artifacts import (
        publish_reviewed_artifact,
    )

    raw = canonical_json_bytes(
        {"record_type": "glm52_task13_review_approval_v1"}
    ) + b"\n"
    source = tmp_path / "review.json"
    source.write_bytes(raw)
    unknown_s3 = ExactS3()
    with pytest.raises(ValueError, match="allowlist"):
        publish_reviewed_artifact(
            artifact_kind="INVENTED_FIXED_KEY",
            source_path=source,
            **_pins(raw),
            bucket=BUCKET,
            services=_services(unknown_s3),
        )
    with pytest.raises(ValueError, match="retained models bucket"):
        publish_reviewed_artifact(
            artifact_kind="TASK11_REVIEW_APPROVAL",
            source_path=source,
            **_pins(raw),
            bucket="keep-glm52-h1g-campaign-246813579024-us-west-2",
            services=_services(unknown_s3),
        )
    assert unknown_s3.list_requests == []
    assert unknown_s3.put_requests == []

    source.write_bytes(b'{"record_type": "not-canonical"}\n')
    with pytest.raises(ValueError, match="canonical"):
        publish_reviewed_artifact(
            artifact_kind="TASK11_REVIEW_APPROVAL",
            source_path=source,
            **_pins(source.read_bytes()),
            bucket=BUCKET,
            services=_services(ExactS3()),
        )
    source.write_text(
        "AWSTemplateFormatVersion: '2010-09-09'\nResources: {}\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="canonical"):
        publish_reviewed_artifact(
            artifact_kind="RETAINED_TEMPLATE",
            source_path=source,
            **_pins(source.read_bytes()),
            bucket=BUCKET,
            services=_services(ExactS3()),
        )

    source.write_bytes(raw)
    foreign = canonical_json_bytes(
        {"record_type": "foreign-reviewed-artifact"}
    ) + b"\n"
    foreign_s3 = ExactS3()
    foreign_s3.seed(
        ARTIFACT_KEYS["TASK11_REVIEW_APPROVAL"],
        foreign,
        metadata=_review_metadata("TASK11_REVIEW_APPROVAL", foreign),
    )
    with pytest.raises(ValueError, match="foreign|drifted"):
        publish_reviewed_artifact(
            artifact_kind="TASK11_REVIEW_APPROVAL",
            source_path=source,
            **_pins(raw),
            bucket=BUCKET,
            services=_services(foreign_s3),
        )
    assert foreign_s3.put_requests == []


def test_retry_enabled_or_unversioned_transport_is_refused(
    tmp_path: Path,
) -> None:
    """Break caught: publication proceeds without one-attempt versioned truth."""

    from glm52_enforcement.task13_reviewed_artifacts import (
        Task13ReviewedArtifactServices,
        publish_reviewed_artifact,
    )

    raw = canonical_json_bytes({"record_type": "fixture"}) + b"\n"
    source = tmp_path / "review.json"
    source.write_bytes(raw)
    retry_s3 = ExactS3()
    with pytest.raises(ValueError, match="transport"):
        publish_reviewed_artifact(
            artifact_kind="TASK12_REVIEW_APPROVAL",
            source_path=source,
            **_pins(raw),
            bucket=BUCKET,
            services=Task13ReviewedArtifactServices(
                sts=ExactSts(),
                s3=retry_s3,
                total_max_attempts=2,
            ),
        )
    suspended_s3 = ExactS3(versioning_status="Suspended")
    with pytest.raises(ValueError, match="versioning"):
        publish_reviewed_artifact(
            artifact_kind="TASK12_REVIEW_APPROVAL",
            source_path=source,
            **_pins(raw),
            bucket=BUCKET,
            services=_services(suspended_s3),
        )
    assert retry_s3.put_requests == []
    assert suspended_s3.put_requests == []


def test_boolean_retry_metadata_is_not_authenticated_zero_retry(
    tmp_path: Path,
) -> None:
    """Break caught: bool False is accepted as the integer SDK retry count."""

    from glm52_enforcement.task13_reviewed_artifacts import (
        Task13ReviewedArtifactServices,
        publish_reviewed_artifact,
    )

    raw = canonical_json_bytes({"record_type": "fixture"}) + b"\n"
    source = tmp_path / "review.json"
    source.write_bytes(raw)
    s3 = ExactS3()
    with pytest.raises(ValueError, match="zero-retry"):
        publish_reviewed_artifact(
            artifact_kind="TASK12_REVIEW_APPROVAL",
            source_path=source,
            **_pins(raw),
            bucket=BUCKET,
            services=Task13ReviewedArtifactServices(
                sts=FalseRetrySts(),
                s3=s3,
                total_max_attempts=1,
            ),
        )
    assert s3.put_requests == []


def test_canonical_utf8_json_is_preserved_byte_exact(
    tmp_path: Path,
) -> None:
    """Break caught: canonical prompt text is silently limited to ASCII."""

    from glm52_enforcement.task13_reviewed_artifacts import (
        publish_reviewed_artifact,
    )

    raw = canonical_json_bytes(
        {"prompt": "Résumé — 東京", "record_type": "glm52_prompt_pack"}
    ) + b"\n"
    source = tmp_path / "prompt-pack.json"
    source.write_bytes(raw)
    s3 = ExactS3()

    coordinate = publish_reviewed_artifact(
        artifact_kind="PROMPT_PACK",
        source_path=source,
        **_pins(raw),
        bucket=BUCKET,
        services=_services(s3),
    )

    assert coordinate["file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert s3.versions[coordinate["key"]][0]["raw"] == raw


def test_cli_writes_one_private_coordinate_and_refuses_replacement(
    tmp_path: Path,
) -> None:
    """Break caught: executable publisher bypasses typed transport or overwrites."""

    spec = importlib.util.spec_from_file_location(
        "_task13_reviewed_artifact_cli",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    raw = canonical_json_bytes({"record_type": "fixture"}) + b"\n"
    source = tmp_path / "review.json"
    source.write_bytes(raw)
    output = tmp_path / "coordinate.json"
    s3 = ExactS3()
    argv = [
        "--artifact-kind",
        "TASK12_REVIEW_APPROVAL",
        "--source",
        str(source),
        "--expected-file-sha256",
        hashlib.sha256(raw).hexdigest(),
        "--expected-body-sha256",
        hashlib.sha256(raw[:-1]).hexdigest(),
        "--bucket",
        BUCKET,
        "--coordinate-output",
        str(output),
    ]

    assert cli.main(argv, services_factory=lambda: _services(s3)) == 0
    coordinate = json.loads(output.read_bytes())
    assert output.read_bytes() == canonical_json_bytes(coordinate) + b"\n"
    assert output.stat().st_mode & 0o777 == 0o600
    assert coordinate["artifact_kind"] == "TASK12_REVIEW_APPROVAL"
    assert len(s3.put_requests) == 1

    assert cli.main(argv, services_factory=lambda: _services(s3)) == 64
    assert len(s3.put_requests) == 1


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("bucket", "keep-glm52-h1g-campaign-246813579024-us-west-2"),
        ("key", "task13/approvals/foreign.json"),
    ],
)
def test_coordinate_writer_rejects_coherently_rehashed_foreign_route(
    tmp_path: Path,
    field: str,
    foreign: str,
) -> None:
    """Break caught: write-once output launders a foreign bucket or key."""

    from glm52_enforcement.task13_reviewed_artifacts import (
        write_coordinate_once,
    )

    coordinate = {
        "artifact_kind": "SUPPORT_APPROVAL",
        "bucket": BUCKET,
        "key": ARTIFACT_KEYS["SUPPORT_APPROVAL"],
        "version_id": "opaque-version",
        "file_sha256": "a" * 64,
        "body_sha256": "b" * 64,
    }
    coordinate[field] = foreign
    output = tmp_path / "coordinate.json"

    with pytest.raises(ValueError, match="coordinate.*exact|bucket|key"):
        write_coordinate_once(output, coordinate)

    assert not output.exists()
