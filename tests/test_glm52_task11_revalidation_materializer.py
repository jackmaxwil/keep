"""Production materialization for Task 11 runtime revalidation authority."""

from __future__ import annotations

import base64
import importlib.util
import hashlib
import io
import json
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "aws/glm52-gpu/scripts/materialize_h1g_support_plane.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "_glm52_runtime_revalidation_materializer",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _semantic_authorities(
    task7_identity: str,
) -> dict[str, object]:
    path = ROOT / "tests/test_glm52_task11_runtime_revalidation.py"
    spec = importlib.util.spec_from_file_location(
        "_glm52_runtime_revalidation_test_fixtures",
        path,
    )
    assert spec is not None and spec.loader is not None
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    documents = fixture_module._valid_semantic_documents(
        task7_inventory_identity_sha256=task7_identity
    )
    return {
        kind: document["authority"]
        for kind, document in documents.items()
    }


def _identified(body: dict[str, object]) -> dict[str, object]:
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


def _source_bundle() -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    bucket = "keep-glm52-models-246813579024-us-west-2"
    source_documents = {
        "task6_manifest": _identified(
            {
                "record_type": "glm52_h1g_stack_migration_manifest_v1",
                "account_id": "246813579024",
                "region": "us-west-2",
                "run_id": "glm52-sky-20260724",
                "bucket_name": bucket,
                "retained_deployment_role_id": "AROADEPLOYMENTROLE123",
                "stacks": [{"stage": str(index)} for index in range(3)],
                "artifacts": [{"key": "migration"}],
                "ownership": [{"owner": "campaign"}],
            }
        ),
        "task6_templates": _identified(
            {
                "record_type": (
                    "glm52_h1d_task6_task7_template_bundle_v1"
                ),
                "task6_templates": {
                    stage: {"Resources": {stage: {"Type": "Test"}}}
                    for stage in (
                        "retention-only",
                        "post-retain",
                        "fence-import",
                        "final-fence",
                    )
                },
                "task7_support_template": {
                    "Resources": {"Support": {"Type": "Test"}}
                },
            }
        ),
        "task7_postcreate_manifest": _identified(
            {
                "record_type": (
                    "glm52_h1g_support_postcreate_manifest_v1"
                ),
                "account_id": "246813579024",
                "region": "us-west-2",
                "run_id": "glm52-sky-20260724",
                "support_stack_id": (
                    "arn:aws:cloudformation:us-west-2:246813579024:"
                    "stack/keep-glm52-h1g-support/"
                    "00000000-0000-0000-0000-000000000001"
                ),
                "support_template_body_sha256": "1" * 64,
                "support_postcreate_inventory_sha256": "2" * 64,
            }
        ),
        "task7_inventory": _identified(
            {
                "record_type": (
                    "glm52_h1g_support_postcreate_inventory_v1"
                ),
                "support_stack_id": (
                    "arn:aws:cloudformation:us-west-2:246813579024:"
                    "stack/keep-glm52-h1g-support/"
                    "00000000-0000-0000-0000-000000000001"
                ),
                "support_template_body_sha256": "1" * 64,
                "stack_resources": [{"logical_id": "Support"}],
                "action_resources": {"Support": {"physical_id": "support"}},
                "describe_stacks_response_sha256": "3" * 64,
                "list_stack_resources_response_sha256": ["4" * 64],
            }
        ),
    }
    coordinates = {}
    for name, document in source_documents.items():
        body = dict(document)
        body_identity = body.pop("canonical_identity_sha256")
        coordinates[name] = {
            "bucket": bucket,
            "key": "campaigns/glm52-sky-20260724/" + name + ".json",
            "version_id": "version-" + name,
            "file_sha256": hashlib.sha256(
                canonical_json_bytes(document) + b"\n"
            ).hexdigest(),
            "body_sha256": body_identity,
        }
    return source_documents, coordinates


def _h1d_base() -> dict[str, object]:
    body = {
        "schema_version": 1,
        "record_type": "glm52_task11_h1d_live_input_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "activation_id": "activation-0001",
        "generation": 1,
        "expected_state": {},
        "expected_state_authentication": {},
        "runtime_revalidation_sources": {},
        "spend_request_template": {},
        "sky_probe_request": {},
        "sky_probe_admission_identity_sha256": "5" * 64,
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


class _S3:
    def __init__(self, *, ambiguous_put: bool = False) -> None:
        self.objects: dict[str, dict[str, object]] = {}
        self.put_calls = 0
        self.ambiguous_put = ambiguous_put
        self._request = 0

    def _metadata(self) -> dict[str, object]:
        self._request += 1
        return {
            "HTTPStatusCode": 200,
            "RequestId": f"request-{self._request}",
            "HostId": f"host-{self._request}",
            "RetryAttempts": 0,
            "HTTPHeaders": {
                "date": "Wed, 29 Jul 2026 20:00:00 GMT"
            },
        }

    def get_paginator(self, operation: str):
        assert operation == "list_object_versions"
        outer = self

        class Paginator:
            def paginate(self, **request: object):
                key = request["Prefix"]
                item = outer.objects.get(key)
                versions = []
                if item is not None:
                    versions.append(
                        {
                            "Key": key,
                            "VersionId": item["VersionId"],
                            "ETag": item["ETag"],
                            "Size": len(item["Body"]),
                            "IsLatest": True,
                        }
                    )
                return (
                    {
                        "Versions": versions,
                        "DeleteMarkers": [],
                        "IsTruncated": False,
                        "ResponseMetadata": outer._metadata(),
                    },
                )

        return Paginator()

    def put_object(self, **request: object):
        self.put_calls += 1
        key = request["Key"]
        assert request["IfNoneMatch"] == "*"
        assert key not in self.objects
        raw = request["Body"]
        version_id = f"version-{self.put_calls}"
        etag = '"' + hashlib.md5(raw).hexdigest() + '"'
        self.objects[key] = {
            **request,
            "VersionId": version_id,
            "ETag": etag,
        }
        if self.ambiguous_put:
            raise TimeoutError("response lost after accepted write")
        return {
            "VersionId": version_id,
            "ETag": etag,
            "ChecksumSHA256": request["ChecksumSHA256"],
            "ResponseMetadata": self._metadata(),
        }

    def get_object(self, **request: object):
        item = self.objects[request["Key"]]
        assert request["VersionId"] == item["VersionId"]
        assert request["ChecksumMode"] == "ENABLED"
        return {
            "Body": io.BytesIO(item["Body"]),
            "VersionId": item["VersionId"],
            "ContentLength": len(item["Body"]),
            "ChecksumSHA256": item["ChecksumSHA256"],
            "ETag": item["ETag"],
            "ContentType": item["ContentType"],
            "Metadata": item["Metadata"],
            "ServerSideEncryption": item["ServerSideEncryption"],
            "SSEKMSKeyId": item["SSEKMSKeyId"],
            "BucketKeyEnabled": item["BucketKeyEnabled"],
            "ResponseMetadata": self._metadata(),
        }


def test_postcreate_materializer_has_real_runtime_revalidation_route() -> None:
    module = _module()

    source_documents, source_coordinates = _source_bundle()
    task7_identity = source_coordinates["task7_inventory"]["body_sha256"]
    result = module.materialize_runtime_revalidation_documents(
        activation_id="activation-0001",
        generation=1,
        source_documents=source_documents,
        source_coordinates=source_coordinates,
        semantic_authorities=_semantic_authorities(task7_identity),
    )

    assert tuple(result) == module.RUNTIME_REVALIDATION_SOURCE_KINDS
    assert len(result) == 7
    for kind, document in result.items():
        body = dict(document)
        identity = body.pop("canonical_identity_sha256")
        assert identity == canonical_sha256(body)
        assert document["input_kind"] == kind
        assert document["activation_id"] == "activation-0001"


def test_postcreate_materializer_exposes_immutable_publication_route() -> None:
    module = _module()

    assert callable(module.publish_runtime_revalidation_bundle)
    assert callable(module.inject_runtime_sources_into_h1d_live_request)


@pytest.mark.parametrize("ambiguous_put", (False, True))
def test_runtime_revalidation_publication_is_exact_and_never_resent(
    ambiguous_put: bool,
) -> None:
    module = _module()
    source_documents, source_coordinates = _source_bundle()
    task7_identity = source_coordinates["task7_inventory"]["body_sha256"]
    s3 = _S3(ambiguous_put=ambiguous_put)
    result = module.publish_runtime_revalidation_bundle(
        s3=s3,
        bucket="keep-glm52-models-246813579024-us-west-2",
        kms_key_arn=(
            "arn:aws:kms:us-west-2:246813579024:key/"
            "00000000-0000-0000-0000-000000000001"
        ),
        activation_id="activation-0001",
        generation=1,
        source_documents=source_documents,
        source_coordinates=source_coordinates,
        semantic_authorities=_semantic_authorities(task7_identity),
        h1d_live_request=_h1d_base(),
    )

    assert s3.put_calls == 8
    assert len(result["runtime_revalidation_sources"]) == 7
    assert result["h1d_live_request"]["input_kind"] == "H1D_LIVE_REQUEST"
    assert all(
        evidence["ambiguity_reconciled"] is ambiguous_put
        for evidence in result["publication_evidence"].values()
    )
    for item in s3.objects.values():
        assert item["ExpectedBucketOwner"] == "246813579024"
        assert item["ServerSideEncryption"] == "aws:kms"
        assert item["ContentType"] == "application/json"
        assert item["ChecksumSHA256"] == base64.b64encode(
            hashlib.sha256(item["Body"]).digest()
        ).decode("ascii")

    with pytest.raises(ValueError, match="already materialized"):
        module.publish_runtime_revalidation_bundle(
            s3=s3,
            bucket="keep-glm52-models-246813579024-us-west-2",
            kms_key_arn=(
                "arn:aws:kms:us-west-2:246813579024:key/"
                "00000000-0000-0000-0000-000000000001"
            ),
            activation_id="activation-0001",
            generation=1,
            source_documents=source_documents,
            source_coordinates=source_coordinates,
            semantic_authorities=_semantic_authorities(task7_identity),
            h1d_live_request=_h1d_base(),
        )
    assert s3.put_calls == 8


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_put_evidence",
        "hidden_version",
        "hidden_delete_marker",
        "wrong_get_metadata",
        "wrong_get_kms",
        "missing_exact_get",
        "fabricated_version_id",
        "lost_before_acceptance",
    ),
)
def test_immutable_transport_mutants_fail_without_resend(
    mutation: str,
) -> None:
    module = _module()
    source_documents, source_coordinates = _source_bundle()
    task7_identity = source_coordinates["task7_inventory"]["body_sha256"]
    documents = module.materialize_runtime_revalidation_documents(
        activation_id="activation-0001",
        generation=1,
        source_documents=source_documents,
        source_coordinates=source_coordinates,
        semantic_authorities=_semantic_authorities(task7_identity),
    )

    class Mutant(_S3):
        def put_object(self, **request: object):
            if mutation == "lost_before_acceptance":
                self.put_calls += 1
                raise TimeoutError("request not accepted")
            response = super().put_object(**request)
            if mutation == "missing_put_evidence":
                response["ResponseMetadata"].pop("HostId")
            elif mutation == "fabricated_version_id":
                response["VersionId"] = "fabricated-version"
            return response

        def get_object(self, **request: object):
            response = super().get_object(**request)
            if mutation == "wrong_get_metadata":
                response["Metadata"] = {"wrong": "metadata"}
            elif mutation == "wrong_get_kms":
                response["SSEKMSKeyId"] = (
                    "arn:aws:kms:us-west-2:246813579024:key/"
                    "ffffffff-ffff-ffff-ffff-ffffffffffff"
                )
            return response

        def get_paginator(self, operation: str):
            paginator = super().get_paginator(operation)
            outer = self

            class MutantPaginator:
                def paginate(self, **request: object):
                    pages = list(paginator.paginate(**request))
                    if (
                        mutation
                        in {"hidden_delete_marker", "hidden_version"}
                        and request["Prefix"] in outer.objects
                    ):
                        if mutation == "hidden_delete_marker":
                            pages[0]["DeleteMarkers"] = [
                                {
                                    "Key": request["Prefix"],
                                    "VersionId": "hidden-delete",
                                }
                            ]
                        else:
                            pages[0]["Versions"].append(
                                {
                                    "Key": request["Prefix"],
                                    "VersionId": "hidden-version",
                                    "ETag": '"hidden"',
                                    "Size": 1,
                                    "IsLatest": False,
                                }
                            )
                    return tuple(pages)

            return MutantPaginator()

    s3 = Mutant()
    if mutation == "missing_exact_get":
        s3.get_object = None
    with pytest.raises(ValueError):
        module._publish_immutable_document(
            s3=s3,
            bucket="keep-glm52-models-246813579024-us-west-2",
            key=module._runtime_revalidation_key(
                activation_id="activation-0001",
                generation=1,
                kind="RUNTIME_CUTOFF_AUTHORITY",
            ),
            kms_key_arn=(
                "arn:aws:kms:us-west-2:246813579024:key/"
                "00000000-0000-0000-0000-000000000001"
            ),
            input_kind="RUNTIME_CUTOFF_AUTHORITY",
            document=documents["RUNTIME_CUTOFF_AUTHORITY"],
        )
    assert s3.put_calls == 1


def test_semantic_drift_fails_before_any_publication() -> None:
    module = _module()
    source_documents, source_coordinates = _source_bundle()
    task7_identity = source_coordinates["task7_inventory"]["body_sha256"]
    authorities = _semantic_authorities(task7_identity)
    authorities["RUNTIME_ATTACHMENT_INVENTORY"]["attachments"] = []
    s3 = _S3()

    with pytest.raises(ValueError):
        module.publish_runtime_revalidation_bundle(
            s3=s3,
            bucket="keep-glm52-models-246813579024-us-west-2",
            kms_key_arn=(
                "arn:aws:kms:us-west-2:246813579024:key/"
                "00000000-0000-0000-0000-000000000001"
            ),
            activation_id="activation-0001",
            generation=1,
            source_documents=source_documents,
            source_coordinates=source_coordinates,
            semantic_authorities=authorities,
            h1d_live_request=_h1d_base(),
        )
    assert s3.put_calls == 0


def test_runtime_publication_cli_is_explicit_closed_and_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    source_documents, source_coordinates = _source_bundle()
    task7_identity = source_coordinates["task7_inventory"]["body_sha256"]
    request_body = {
        "schema_version": 1,
        "record_type": (
            "glm52_h1g_runtime_revalidation_publication_request_v1"
        ),
        "bucket": "keep-glm52-models-246813579024-us-west-2",
        "kms_key_arn": (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "00000000-0000-0000-0000-000000000001"
        ),
        "activation_id": "activation-0001",
        "generation": 1,
        "source_documents": source_documents,
        "source_coordinates": source_coordinates,
        "semantic_authorities": _semantic_authorities(task7_identity),
        "h1d_live_request": _h1d_base(),
    }
    request = {
        **request_body,
        "canonical_identity_sha256": canonical_sha256(request_body),
    }
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "publication.json"
    request_path.write_bytes(canonical_json_bytes(request) + b"\n")
    s3 = _S3()

    class Runner:
        def __init__(self, *, profile: str, region: str) -> None:
            assert profile == "keep-gpu"
            assert region == "us-west-2"
            self.s3 = s3

        def get_caller_identity(self):
            return {
                "Account": "246813579024",
                "Arn": (
                    "arn:aws:sts::246813579024:"
                    "assumed-role/runtime-publication/session"
                ),
                "UserId": "AROAPUBLICATION:session",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "caller-identity",
                },
            }

    monkeypatch.setenv("AWS_PAGER", "")
    assert (
        module.run(
            [
                "--profile",
                "keep-gpu",
                "--region",
                "us-west-2",
                "--publish-runtime-revalidation",
                "--runtime-revalidation-request",
                str(request_path),
                "--runtime-revalidation-output",
                str(output_path),
            ],
            runner_factory=Runner,
        )
        == 0
    )
    raw = output_path.read_bytes()
    assert raw == canonical_json_bytes(
        json.loads(raw.decode("ascii"))
    ) + b"\n"
    assert s3.put_calls == 8

    with pytest.raises(ValueError, match="closed argument"):
        module.run(
            [
                "--profile",
                "keep-gpu",
                "--region",
                "us-west-2",
                "--publish-runtime-revalidation",
                "--runtime-revalidation-request",
                str(request_path),
                "--runtime-revalidation-output",
                str(tmp_path / "other.json"),
                "--support-stack-id",
                "forbidden",
            ],
            runner_factory=Runner,
        )


def test_h1d_injection_rejects_stale_self_identity() -> None:
    module = _module()
    stale = _h1d_base()
    stale["generation"] = 2

    with pytest.raises(ValueError, match="base identity"):
        module.inject_runtime_sources_into_h1d_live_request(
            h1d_live_request=stale,
            runtime_revalidation_sources={},
            activation_id="activation-0001",
            generation=1,
        )


def test_postcreate_materializer_rejects_detached_task6_coordinate() -> None:
    module = _module()
    source_documents, source_coordinates = _source_bundle()
    task7_identity = source_coordinates["task7_inventory"]["body_sha256"]
    source_coordinates["task6_manifest"]["version_id"] = "null"

    with pytest.raises(ValueError, match="authenticated source"):
        module.materialize_runtime_revalidation_documents(
            activation_id="activation-0001",
            generation=1,
            source_documents=source_documents,
            source_coordinates=source_coordinates,
            semantic_authorities=_semantic_authorities(task7_identity),
        )
