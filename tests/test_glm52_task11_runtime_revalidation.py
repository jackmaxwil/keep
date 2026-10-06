"""Task 11 executable Task 6/7 seal and attachment revalidation."""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from glm52_enforcement import live_authority
from glm52_enforcement import task11_production
from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.task11_boundary import build_task11_input_coordinate


ACTIVATION_ID = "activation-0001"
SOURCE_KINDS = (
    "TASK6_MIGRATION_MANIFEST",
    "TASK6_TEMPLATE_INVENTORY",
    "TASK7_POSTCREATE_MANIFEST",
    "TASK7_SUPPORT_INVENTORY",
    "RUNTIME_CUTOFF_AUTHORITY",
    "RUNTIME_ATTACHMENT_INVENTORY",
    "CREDENTIAL_PROBE_INVENTORY",
)


def _source_coordinates() -> dict[str, object]:
    result = {}
    for index, kind in enumerate(SOURCE_KINDS, 1):
        body = {
            "input_kind": kind,
            "bucket": "keep-glm52-campaign",
            "key": (
                "campaigns/glm52-sky-20260724/authorities/task11/"
                + ACTIVATION_ID
                + "/00000001/runtime-revalidation/"
                + f"{index:02d}-"
                + kind.lower().replace("_", "-")
                + ".json"
            ),
            "version_id": "source-version-" + str(index),
            "file_sha256": hashlib.sha256(
                ("file-" + kind).encode("ascii")
            ).hexdigest(),
            "body_sha256": hashlib.sha256(
                ("body-" + kind).encode("ascii")
            ).hexdigest(),
        }
        result[kind] = {
            **body,
            "canonical_identity_sha256": canonical_sha256(body),
        }
    return result


def test_runtime_revalidation_source_coordinates_are_exact_and_ordered() -> None:
    parsed = live_authority.runtime_revalidation_sources_from_mapping(
        _source_coordinates(),
        activation_id=ACTIVATION_ID,
        generation=1,
    )

    assert tuple(item.input_kind for item in parsed) == SOURCE_KINDS
    assert parsed[0].version_id == "source-version-1"
    assert parsed[-1].version_id == "source-version-7"


@pytest.mark.parametrize(
    "retired_kind",
    (
        "BASELINE_SCP_AUTHORITY",
        "MAINTENANCE_SEAL_SCP_AUTHORITY",
        "POST_SEAL_ATTACHMENT_INVENTORY",
    ),
)
def test_runtime_revalidation_coordinates_reject_retired_source_kinds(
    retired_kind: str,
) -> None:
    coordinates = _source_coordinates()
    coordinates[retired_kind] = dict(
        coordinates["RUNTIME_ATTACHMENT_INVENTORY"]
    )

    with pytest.raises(
        live_authority.H1dLiveAuthorityError,
        match="source inventory",
    ):
        live_authority.runtime_revalidation_sources_from_mapping(
            coordinates,
            activation_id=ACTIVATION_ID,
            generation=1,
        )


def test_runtime_revalidation_sources_use_exact_versioned_service_reads() -> None:
    coordinate_values = _source_coordinates()
    raw_by_key = {}
    for kind in SOURCE_KINDS:
        coordinate = coordinate_values[kind]
        body = {
            "schema_version": 1,
            "record_type": (
                "fixture_" + kind.lower() + "_v1"
            ),
            "input_kind": kind,
        }
        document = {
            **body,
            "canonical_identity_sha256": canonical_sha256(body),
        }
        raw = canonical_json_bytes(document) + b"\n"
        raw_by_key[coordinate["key"]] = raw
        coordinate_body = dict(coordinate)
        coordinate_body.pop("canonical_identity_sha256")
        coordinate_body["file_sha256"] = hashlib.sha256(raw).hexdigest()
        coordinate_body["body_sha256"] = document[
            "canonical_identity_sha256"
        ]
        coordinate_values[kind] = {
            **coordinate_body,
            "canonical_identity_sha256": canonical_sha256(
                coordinate_body
            ),
        }
    coordinates = live_authority.runtime_revalidation_sources_from_mapping(
        coordinate_values,
        activation_id=ACTIVATION_ID,
        generation=1,
    )

    class S3:
        def __init__(self) -> None:
            self.calls = []

        def get_object(self, **request: object):
            self.calls.append(dict(request))
            raw = raw_by_key[request["Key"]]
            checksum = base64.b64encode(
                hashlib.sha256(raw).digest()
            ).decode("ascii")
            return {
                "Body": io.BytesIO(raw),
                "VersionId": request["VersionId"],
                "ContentLength": len(raw),
                "ChecksumSHA256": checksum,
                "ETag": '"service-etag"',
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "get-" + str(len(self.calls)),
                },
            }

    s3 = S3()
    reads = task11_production.load_runtime_revalidation_sources(
        s3=s3,
        coordinates=coordinates,
    )

    assert len(reads) == 7
    assert tuple(read.request_id for read in reads) == tuple(
        "get-" + str(index) for index in range(1, 8)
    )
    assert len({read.response_identity_sha256 for read in reads}) == 7
    assert all(
        call["ExpectedBucketOwner"] == "246813579024"
        and call["ChecksumMode"] == "ENABLED"
        for call in s3.calls
    )


def test_current_closure_session_is_bound_after_runtime_cutoff() -> None:
    binding = live_authority.bind_current_closure_session_to_cutoff(
        activation_id=ACTIVATION_ID,
        decision_role_arn=(
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-support-decision"
        ),
        closure_role_arn=(
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-closure-session-daa92ceb457a0cfb"
        ),
        closure_role_id="AROACLOSURESESSION123",
        cutoff_at="2026-07-29T12:00:00Z",
        credential_issue_time="2026-07-29T12:01:00Z",
        credential_expiration="2026-07-29T12:16:00Z",
        observed_at="2026-07-29T12:05:00Z",
        assume_role_request_id="assume-role-1",
        source_caller_identity={
            "Account": "246813579024",
            "Arn": (
                "arn:aws:sts::246813579024:assumed-role/"
                "keep-glm52-h1g-support-decision/lambda-session"
            ),
            "UserId": "AROADECISIONROLE123:lambda-session",
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "source-caller-1",
            },
        },
        closure_caller_identity={
            "Account": "246813579024",
            "Arn": (
                "arn:aws:sts::246813579024:assumed-role/"
                "keep-glm52-h1g-closure-session-daa92ceb457a0cfb/"
                "h1g-decision-session"
            ),
            "UserId": (
                "AROACLOSURESESSION123:h1g-decision-session"
            ),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "closure-caller-1",
            },
        },
    )

    assert binding.current_session_proven is True
    assert binding.credential_expiration == "2026-07-29T12:16:00Z"
    assert binding.source_request_id == "source-caller-1"
    assert binding.closure_request_id == "closure-caller-1"


@pytest.mark.parametrize("mutation", ("page_cycle", "unbounded"))
def test_runtime_inventory_pagination_rejects_cycle_and_unbounded_walk(
    mutation: str,
) -> None:
    def response(index: int) -> dict[str, object]:
        return {
            "Functions": [
                {
                    "FunctionArn": (
                        "arn:aws:lambda:us-west-2:246813579024:"
                        "function:keep-glm52-h1g-decision:"
                        + str(index + 1)
                    )
                }
            ],
            "NextMarker": (
                "same-marker"
                if mutation == "page_cycle"
                else "marker-" + str(index + 1)
            ),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "lambda-list-" + str(index + 1),
            },
        }

    class Paginator:
        def paginate(self, **request: object):
            del request
            count = 2 if mutation == "page_cycle" else 65
            return tuple(response(index) for index in range(count))

    class Lambda:
        def list_functions(self, **request: object):
            raise AssertionError("paginated read called direct method")

        def get_paginator(self, operation: str):
            assert operation == "list_functions"
            return Paginator()

    spec = live_authority.LiveReadSpec(
        family="lambda",
        operation="lambda.inspect_complete",
        parameters={
            "calls": [
                {
                    "method": "list_functions",
                    "request": {},
                    "items_path": ["Functions"],
                    "field_paths": {
                        "function_arn": ["FunctionArn"],
                    },
                    "paginate": True,
                }
            ]
        },
        identity_field="function_arn",
        expected_items=(),
    )
    reader = task11_production.Task11H1dReader({"lambda": Lambda()})

    with pytest.raises(RuntimeError, match="pagination|repeated|bound"):
        reader._read_spec(spec)


def test_runtime_attachment_inventory_accepts_exact_two_page_service_truth() -> None:
    pages = (
        {
            "Functions": [
                {
                    "FunctionArn": (
                        "arn:aws:lambda:us-west-2:246813579024:"
                        "function:keep-glm52-h1g-attestation:3"
                    ),
                    "Role": (
                        "arn:aws:iam::246813579024:role/"
                        "keep-glm52-h1g-support-attestation"
                    ),
                }
            ],
            "NextMarker": "lambda-page-2",
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "lambda-list-1",
            },
        },
        {
            "Functions": [
                {
                    "FunctionArn": (
                        "arn:aws:lambda:us-west-2:246813579024:"
                        "function:keep-glm52-h1g-decision:11"
                    ),
                    "Role": (
                        "arn:aws:iam::246813579024:role/"
                        "keep-glm52-h1g-support-decision"
                    ),
                }
            ],
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "lambda-list-2",
            },
        },
    )

    class Paginator:
        def paginate(self, **request: object):
            assert request == {}
            return pages

    class Lambda:
        def list_functions(self, **request: object):
            raise AssertionError("pagination boundary was bypassed")

        def get_paginator(self, operation: str):
            assert operation == "list_functions"
            return Paginator()

    expected = (
        {
            "function_arn": pages[0]["Functions"][0]["FunctionArn"],
            "role_arn": pages[0]["Functions"][0]["Role"],
        },
        {
            "function_arn": pages[1]["Functions"][0]["FunctionArn"],
            "role_arn": pages[1]["Functions"][0]["Role"],
        },
    )
    spec = live_authority.LiveReadSpec(
        family="lambda",
        operation="lambda.inspect_complete",
        parameters={
            "calls": [
                {
                    "method": "list_functions",
                    "request": {},
                    "items_path": ["Functions"],
                    "field_paths": {
                        "function_arn": ["FunctionArn"],
                        "role_arn": ["Role"],
                    },
                    "paginate": True,
                }
            ]
        },
        identity_field="function_arn",
        expected_items=expected,
    )

    evidence = task11_production.revalidate_runtime_attachment_specs(
        specs=(spec,),
        clients={"lambda": Lambda()},
        deadline=task11_production._utc_now()
        + task11_production.timedelta(seconds=5),
    )

    assert evidence.request_ids == ("lambda-list-1", "lambda-list-2")
    assert len(evidence.response_identities) == 2
    assert evidence.family_identities["lambda"] == canonical_sha256(
        {
            "family": "lambda",
            "items": tuple(
                sorted(expected, key=lambda row: row["function_arn"])
            ),
        }
    )


def test_runtime_attachment_inventory_supports_closed_states_reads() -> None:
    """Production revalidation may make several exact reads per AWS family."""

    def response(request_id: str, **body: object) -> dict[str, object]:
        return {
            **body,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": request_id,
            },
        }

    class StepFunctions:
        def describe_state_machine(self, **request: object):
            assert request == {
                "stateMachineArn": (
                    "arn:aws:states:us-west-2:246813579024:"
                    "stateMachine:keep-glm52-h1g-support"
                )
            }
            return response(
                "states-describe",
                stateMachineArn=request["stateMachineArn"],
                roleArn=(
                    "arn:aws:iam::246813579024:"
                    "role/keep-glm52-h1g-support-workflow"
                ),
            )

        def list_state_machine_versions(self, **request: object):
            raise AssertionError("paginated read called direct method")

        def get_paginator(self, operation: str):
            assert operation == "list_state_machine_versions"

            class Paginator:
                def paginate(self, **request: object):
                    assert request["stateMachineArn"].endswith(
                        ":keep-glm52-h1g-support"
                    )
                    return (
                        response(
                            "states-list-versions",
                            stateMachineVersions=[
                                {
                                    "stateMachineVersionArn": (
                                        request["stateMachineArn"] + ":1"
                                    )
                                }
                            ],
                        ),
                    )

            return Paginator()

    state_machine_arn = (
        "arn:aws:states:us-west-2:246813579024:"
        "stateMachine:keep-glm52-h1g-support"
    )
    specs = (
        live_authority.LiveReadSpec(
            family="stepfunctions",
            operation="stepfunctions.describe_state_machine",
            parameters={
                "calls": [
                    {
                        "method": "describe_state_machine",
                        "request": {"stateMachineArn": state_machine_arn},
                        "items_path": [],
                        "field_paths": {
                            "state_machine_arn": ["stateMachineArn"],
                            "role_arn": ["roleArn"],
                        },
                        "paginate": False,
                    }
                ]
            },
            identity_field="state_machine_arn",
            expected_items=(
                {
                    "state_machine_arn": state_machine_arn,
                    "role_arn": (
                        "arn:aws:iam::246813579024:"
                        "role/keep-glm52-h1g-support-workflow"
                    ),
                },
            ),
        ),
        live_authority.LiveReadSpec(
            family="stepfunctions",
            operation="stepfunctions.list_state_machine_versions",
            parameters={
                "calls": [
                    {
                        "method": "list_state_machine_versions",
                        "request": {"stateMachineArn": state_machine_arn},
                        "items_path": ["stateMachineVersions"],
                        "field_paths": {
                            "version_arn": [
                                "stateMachineVersionArn"
                            ]
                        },
                        "paginate": True,
                    }
                ]
            },
            identity_field="version_arn",
            expected_items=(
                {"version_arn": state_machine_arn + ":1"},
            ),
        ),
    )

    evidence = task11_production.revalidate_runtime_attachment_specs(
        specs=specs,
        clients={"stepfunctions": StepFunctions()},
        deadline=task11_production._utc_now()
        + task11_production.timedelta(seconds=5),
    )

    assert set(evidence.family_identities) == {
        spec.operation for spec in specs
    }
    assert evidence.request_ids == (
        "states-describe",
        "states-list-versions",
    )


def test_runtime_attachment_inventory_rejects_organizations_reads() -> None:
    spec = live_authority.LiveReadSpec(
        family="organizations",
        operation="organizations.describe_policy",
        parameters={
            "calls": [
                {
                    "method": "describe_policy",
                    "request": {"PolicyId": "p-retired"},
                    "items_path": ["Policy"],
                    "field_paths": {"policy_id": ["PolicySummary", "Id"]},
                    "paginate": False,
                }
            ]
        },
        identity_field="policy_id",
        expected_items=({"policy_id": "p-retired"},),
    )

    with pytest.raises(
        (KeyError, RuntimeError),
        match="organizations|allowlist",
    ):
        task11_production.revalidate_runtime_attachment_specs(
            specs=(spec,),
            clients={"organizations": object()},
            deadline=task11_production._utc_now()
            + task11_production.timedelta(seconds=5),
        )


def test_step3_and_task8_bundle_reads_are_independent_and_session_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_identity = hashlib.sha256(b"task7-manifest").hexdigest()
    source_identities = {
        "TASK6_MIGRATION_MANIFEST": hashlib.sha256(
            b"task6-manifest"
        ).hexdigest(),
        "TASK6_TEMPLATE_INVENTORY": hashlib.sha256(
            b"task6-templates"
        ).hexdigest(),
        "TASK7_POSTCREATE_MANIFEST": manifest_identity,
        "TASK7_SUPPORT_INVENTORY": hashlib.sha256(
            b"task7-inventory"
        ).hexdigest(),
    }
    specs = tuple(
        live_authority.LiveReadSpec(
            family=family,
            operation=family + ".inspect_complete",
            parameters={"calls": []},
            identity_field="id",
            expected_items=({"id": family},),
        )
        for family in live_authority.REQUIRED_LIVE_FAMILIES
    )
    expected = live_authority.build_h1d_expected_state(
        account_id="246813579024",
        region="us-west-2",
        run_id="glm52-sky-20260724",
        activation_id=ACTIVATION_ID,
        manifest_identity_sha256=manifest_identity,
        support_host_instance_id="i-0123456789abcdef0",
        must_start_by="2026-07-29T12:30:00Z",
        execution_deadline="2026-08-01T12:30:00Z",
        specs=specs,
    )
    authentication = live_authority.build_expected_state_authentication(
        expected_state_identity_sha256=(
            expected.canonical_identity_sha256
        ),
        task6_manifest_identity_sha256=source_identities[
            "TASK6_MIGRATION_MANIFEST"
        ],
        task6_templates_identity_sha256=source_identities[
            "TASK6_TEMPLATE_INVENTORY"
        ],
        task7_postcreate_manifest_identity_sha256=manifest_identity,
        task7_inventory_identity_sha256=source_identities[
            "TASK7_SUPPORT_INVENTORY"
        ],
        direct_read_request_ids=("task6-1", "task6-2", "task7-1", "task7-2"),
        observed_at="2026-07-29T11:59:00Z",
    )
    expected_mapping = live_authority.h1d_expected_state_to_mapping(
        expected
    )
    expected_mapping["specs"] = [
        {
            **spec,
            "expected_items": list(spec["expected_items"]),
        }
        for spec in expected_mapping["specs"]
    ]
    coordinate_values = _source_coordinates()
    for kind, identity in source_identities.items():
        body = dict(coordinate_values[kind])
        body.pop("canonical_identity_sha256")
        body["body_sha256"] = identity
        coordinate_values[kind] = {
            **body,
            "canonical_identity_sha256": canonical_sha256(body),
        }
    coordinates = live_authority.runtime_revalidation_sources_from_mapping(
        coordinate_values,
        activation_id=ACTIVATION_ID,
        generation=1,
    )
    cutoff = {
        "schema_version": 1,
        "record_type": "glm52_task11_runtime_cutoff_authority_v1",
        "input_kind": "RUNTIME_CUTOFF_AUTHORITY",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "activation_id": ACTIVATION_ID,
        "generation": 1,
        "decision_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-support-decision"
        ),
        "closure_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-closure-session-daa92ceb457a0cfb"
        ),
        "closure_role_id": "AROACLOSURESESSION123",
        "cutoff_at": "2026-07-29T12:00:00Z",
    }
    cutoff["canonical_identity_sha256"] = canonical_sha256(cutoff)
    semantic_documents = _valid_semantic_documents(
        task7_inventory_identity_sha256=source_identities[
            "TASK7_SUPPORT_INVENTORY"
        ]
    )
    source_reads = tuple(
        task11_production.RuntimeRevalidationSourceRead(
            coordinate=coordinate,
            document=(
                semantic_documents[coordinate.input_kind]
                if coordinate.input_kind in semantic_documents
                else {
                    "input_kind": coordinate.input_kind,
                    "canonical_identity_sha256": (
                        coordinate.body_sha256
                    ),
                }
            ),
            request_id="runtime-get-" + str(index),
            response_identity_sha256=hashlib.sha256(
                ("runtime-response-" + str(index)).encode()
            ).hexdigest(),
        )
        for index, coordinate in enumerate(coordinates, 1)
    )
    source_mapping = {
        kind: asdict(coordinate)
        for kind, coordinate in zip(SOURCE_KINDS, coordinates)
    }
    h1d_document = {
        "schema_version": 1,
        "record_type": "glm52_task11_h1d_live_input_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "activation_id": ACTIVATION_ID,
        "generation": 1,
        "expected_state": expected_mapping,
        "expected_state_authentication": {
            **asdict(authentication),
            "direct_read_request_ids": list(
                authentication.direct_read_request_ids
            ),
        },
        "runtime_revalidation_sources": source_mapping,
        "spend_request_template": {},
        "sky_probe_request": {},
        "sky_probe_admission_identity_sha256": hashlib.sha256(
            b"sky-probe"
        ).hexdigest(),
    }
    h1d_document["canonical_identity_sha256"] = canonical_sha256(
        h1d_document
    )
    h1d_coordinate = build_task11_input_coordinate(
        input_kind="H1D_LIVE_REQUEST",
        bucket="keep-glm52-campaign",
        key=(
            "campaigns/glm52-sky-20260724/authorities/task11/"
            + ACTIVATION_ID
            + "/00000001/11-h1d-live-request.json"
        ),
        version_id="h1d-input-version-1",
        file_sha256=hashlib.sha256(b"h1d-file").hexdigest(),
        body_sha256=h1d_document["canonical_identity_sha256"],
    )
    service = object.__new__(
        task11_production.Task11ProductionServices
    )
    service._config = SimpleNamespace(
        account_id="246813579024",
        region="us-west-2",
        run_id="glm52-sky-20260724",
        activation_id=ACTIVATION_ID,
        closure_role_arn=cutoff["closure_role_arn"],
    )
    inputs = [None] * 14
    inputs[10] = h1d_coordinate
    service._boundary = SimpleNamespace(inputs=tuple(inputs))
    service._adapters = SimpleNamespace(
        clients=SimpleNamespace(
            s3=object(),
            credential_issue_time="2026-07-29T12:01:00Z",
            credential_expiration="2026-07-29T12:16:00Z",
            assume_role_request_id="assume-role-1",
            assumed_role_id=(
                "AROACLOSURESESSION123:h1g-decision-session"
            ),
            source_caller_identity={
                "Account": "246813579024",
                "Arn": (
                    "arn:aws:sts::246813579024:assumed-role/"
                    "keep-glm52-h1g-support-decision/lambda-session"
                ),
                "UserId": "AROADECISIONROLE123:lambda-session",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "source-caller-1",
                },
            },
            caller_identity={
                "Account": "246813579024",
                "Arn": (
                    "arn:aws:sts::246813579024:assumed-role/"
                    "keep-glm52-h1g-closure-session-daa92ceb457a0cfb/"
                    "h1g-decision-session"
                ),
                "UserId": (
                    "AROACLOSURESESSION123:h1g-decision-session"
                ),
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "closure-caller-1",
                },
            },
        )
    )
    calls = []
    monkeypatch.setattr(
        task11_production,
        "load_exact_input",
        lambda **_kwargs: dict(h1d_document),
    )

    def load_sources(**_kwargs: object):
        calls.append("fresh-read")
        return source_reads

    monkeypatch.setattr(
        task11_production,
        "load_runtime_revalidation_sources",
        load_sources,
    )
    monkeypatch.setattr(
        task11_production,
        "_utc_now",
        lambda: datetime(
            2026, 7, 29, 12, 5, tzinfo=timezone.utc
        ),
    )

    first = service._load_h1d_runtime_bundle(
        request=SimpleNamespace(generation=1)
    )
    second = service._load_h1d_runtime_bundle(
        request=SimpleNamespace(generation=1)
    )

    assert calls == ["fresh-read", "fresh-read"]
    assert first.current_session.current_session_proven is True
    assert first.current_session.absence_observed is False
    assert second.canonical_identity_sha256 == (
        first.canonical_identity_sha256
    )


def _semantic_document(
    kind: str,
    authority: dict[str, object],
) -> dict[str, object]:
    body = {
        "schema_version": 1,
        "record_type": (
            "glm52_task11_"
            + kind.lower()
            + "_v1"
        ),
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "activation_id": ACTIVATION_ID,
        "generation": 1,
        "input_kind": kind,
        "authority": authority,
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


def _valid_semantic_documents(
    *,
    task7_inventory_identity_sha256: str,
) -> dict[str, dict[str, object]]:
    roles = [
        {
            "role_arn": (
                "arn:aws:iam::246813579024:role/"
                "keep-glm52-h1g-support-decision"
            ),
            "role_id": "AROADECISIONROLE123",
            "trust_policy_sha256": "1" * 64,
            "inline_policy_identities": {"RuntimeCutoff": "2" * 64},
            "managed_policy_versions": [],
        },
        {
            "role_arn": (
                "arn:aws:iam::246813579024:role/"
                "keep-glm52-h1g-closure-session-daa92ceb457a0cfb"
            ),
            "role_id": "AROACLOSURESESSION123",
            "trust_policy_sha256": "3" * 64,
            "inline_policy_identities": {
                "ExecuteExactPrivateClosureOnly": "4" * 64,
            },
            "managed_policy_versions": [],
        },
    ]
    cutoff = _semantic_document(
        "RUNTIME_CUTOFF_AUTHORITY",
        {
            "runtime_credential_cutoff_at": "2026-07-29T12:00:00Z",
            "roles": roles,
            "roles_identity_sha256": canonical_sha256(roles),
        },
    )
    attachments = [
        {
            "attachment_kind": "LAMBDA_VERSION",
            "resource_arn": (
                "arn:aws:lambda:us-west-2:246813579024:"
                "function:keep-glm52-h1g-decision:11"
            ),
            "role_arn": roles[0]["role_arn"],
            "role_id": roles[0]["role_id"],
            "association_id": None,
        },
        {
            "attachment_kind": "STEP_FUNCTIONS_VERSION",
            "resource_arn": (
                "arn:aws:states:us-west-2:246813579024:"
                "stateMachine:keep-glm52-h1g-production:7"
            ),
            "role_arn": roles[1]["role_arn"],
            "role_id": roles[1]["role_id"],
            "association_id": None,
        },
    ]
    runtime_attachments = _semantic_document(
        "RUNTIME_ATTACHMENT_INVENTORY",
        {
            "task7_inventory_identity_sha256": (
                task7_inventory_identity_sha256
            ),
            "runtime_cutoff_identity_sha256": cutoff[
                "canonical_identity_sha256"
            ],
            "attachments": attachments,
            "cardinality_by_kind": {
                "LAMBDA_VERSION": 1,
                "STEP_FUNCTIONS_VERSION": 1,
            },
            "attachments_identity_sha256": canonical_sha256(attachments),
        },
    )
    credentials = _semantic_document(
        "CREDENTIAL_PROBE_INVENTORY",
        {
            "runtime_cutoff_identity_sha256": cutoff[
                "canonical_identity_sha256"
            ],
            "runtime_attachment_inventory_identity_sha256": runtime_attachments[
                "canonical_identity_sha256"
            ],
            "runtime_credential_cutoff_at": "2026-07-29T12:00:00Z",
            "old_session_denials": [
                {
                    "role_id": "AROADECISIONROLE123",
                    "credential_issue_time": "2026-07-29T11:59:00Z",
                    "observed_at": "2026-07-29T12:02:00Z",
                    "error_code": "AccessDenied",
                    "request_id": "old-session-deny-1",
                }
            ],
            "missing_issue_time_denials": [
                {
                    "role_id": "AROADECISIONROLE123",
                    "observed_at": "2026-07-29T12:02:30Z",
                    "error_code": "AccessDenied",
                    "request_id": "missing-time-deny-1",
                }
            ],
            "post_cutoff_positive_probes": [
                {
                    "role_id": "AROACLOSURESESSION123",
                    "credential_issue_time": "2026-07-29T12:01:00Z",
                    "credential_expiration": "2026-07-29T12:16:00Z",
                    "observed_at": "2026-07-29T12:03:00Z",
                    "request_id": "current-session-positive-1",
                }
            ],
            "absence_observed": False,
        },
    )
    return {
        "RUNTIME_CUTOFF_AUTHORITY": cutoff,
        "RUNTIME_ATTACHMENT_INVENTORY": runtime_attachments,
        "CREDENTIAL_PROBE_INVENTORY": credentials,
    }


def test_runtime_source_semantics_cross_bind_attachments_and_probes() -> None:
    documents = _valid_semantic_documents(
        task7_inventory_identity_sha256="5" * 64
    )
    semantic = live_authority.validate_runtime_revalidation_source_documents(
        documents,
        activation_id=ACTIVATION_ID,
        generation=1,
        task7_inventory_identity_sha256="5" * 64,
    )

    assert set(semantic.source_identities) == {
        "RUNTIME_CUTOFF_AUTHORITY",
        "RUNTIME_ATTACHMENT_INVENTORY",
        "CREDENTIAL_PROBE_INVENTORY",
    }
    assert set(type(semantic).__dataclass_fields__) == {
        "runtime_credential_cutoff_at",
        "role_ids_by_arn",
        "attachment_cardinality",
        "attachment_cardinality_by_kind",
        "current_positive_role_ids",
        "absence_observed",
        "source_identities",
        "canonical_identity_sha256",
    }
    assert semantic.attachment_cardinality == 2
    assert semantic.current_positive_role_ids == (
        "AROACLOSURESESSION123",
    )
    assert semantic.absence_observed is False


@pytest.mark.parametrize(
    "mutation",
    (
        "opaque_authority",
        "retired_scp_identity_field",
        "retired_source_kind",
        "reused_role_id",
        "duplicate_attachment",
        "stale_positive_probe",
        "absence_claim",
    ),
)
def test_runtime_source_semantic_mutants_fail_closed(
    mutation: str,
) -> None:
    documents = _valid_semantic_documents(
        task7_inventory_identity_sha256="5" * 64
    )

    def rehash(kind: str) -> None:
        document = documents[kind]
        body = dict(document)
        body.pop("canonical_identity_sha256")
        documents[kind] = {
            **body,
            "canonical_identity_sha256": canonical_sha256(body),
        }

    if mutation == "opaque_authority":
        documents["RUNTIME_ATTACHMENT_INVENTORY"]["authority"] = {}
        rehash("RUNTIME_ATTACHMENT_INVENTORY")
    elif mutation == "retired_scp_identity_field":
        documents["RUNTIME_ATTACHMENT_INVENTORY"]["authority"][
            "baseline_scp_identity_sha256"
        ] = "9" * 64
        rehash("RUNTIME_ATTACHMENT_INVENTORY")
    elif mutation == "retired_source_kind":
        documents["BASELINE_SCP_AUTHORITY"] = dict(
            documents["RUNTIME_CUTOFF_AUTHORITY"]
        )
    elif mutation == "reused_role_id":
        cutoff = documents["RUNTIME_CUTOFF_AUTHORITY"]["authority"]
        cutoff["roles"][1]["role_id"] = cutoff["roles"][0]["role_id"]
        cutoff["roles_identity_sha256"] = canonical_sha256(
            cutoff["roles"]
        )
        rehash("RUNTIME_CUTOFF_AUTHORITY")
    elif mutation == "duplicate_attachment":
        runtime_attachments = documents["RUNTIME_ATTACHMENT_INVENTORY"][
            "authority"
        ]
        runtime_attachments["attachments"].append(
            dict(runtime_attachments["attachments"][0])
        )
        runtime_attachments["cardinality_by_kind"]["LAMBDA_VERSION"] = 2
        runtime_attachments["attachments_identity_sha256"] = canonical_sha256(
            runtime_attachments["attachments"]
        )
        rehash("RUNTIME_ATTACHMENT_INVENTORY")
    elif mutation == "stale_positive_probe":
        documents["CREDENTIAL_PROBE_INVENTORY"]["authority"][
            "post_cutoff_positive_probes"
        ][0]["credential_issue_time"] = "2026-07-29T11:59:59Z"
        rehash("CREDENTIAL_PROBE_INVENTORY")
    else:
        documents["CREDENTIAL_PROBE_INVENTORY"]["authority"][
            "absence_observed"
        ] = True
        rehash("CREDENTIAL_PROBE_INVENTORY")

    with pytest.raises(live_authority.H1dLiveAuthorityError):
        live_authority.validate_runtime_revalidation_source_documents(
            documents,
            activation_id=ACTIVATION_ID,
            generation=1,
            task7_inventory_identity_sha256="5" * 64,
        )
