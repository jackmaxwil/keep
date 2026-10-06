"""Deployed, canary-only Task 11 closure rehearsal collector."""

from __future__ import annotations

import base64
from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable, Mapping, Optional

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.decision_closure import (
    CLOSURE_PHASE_CEILINGS,
    CLOSURE_STEPS,
    REHEARSAL_PATH_PROOFS,
    SUFFIX_PHASE_CEILINGS,
    ZERO_PRODUCTION_EFFECTS,
    ClosureRequest,
    PhaseSpan,
    StepReceipt,
    build_deployed_gate_document,
    build_rehearsal_measurement,
    build_step_receipt,
    rehearsal_measurement_from_mapping,
    validate_closure_request,
    verify_no_post_rehearsals,
)


COLLECTOR_MEASUREMENT_IDS = tuple(
    f"measurement-{index:02d}" for index in range(1, 21)
)
_COLLECT_FIELDS = {
    "schema_version",
    "record_type",
    "activation_id",
    "measurement_id",
    "scenario",
    "task11_request",
    "task11_boundary",
}
_FINALIZE_FIELDS = {
    "schema_version",
    "record_type",
    "activation_id",
}
_SCENARIOS = {
    "measurement-01": "THROTTLING",
    "measurement-02": "PAGINATION",
    "measurement-03": "NETWORK_AMBIGUITY",
    **{
        f"measurement-{index:02d}": "NONE"
        for index in range(4, 21)
    },
}
_SHA = re.compile(r"^[0-9a-f]{64}$")
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_VERSION_ARN = re.compile(
    r"^arn:aws:lambda:us-west-2:246813579024:function:"
    r"keep-glm52-h1g-rehearsal-collector:[1-9][0-9]*$"
)
_PROBE_VERSION_ARN = re.compile(
    r"^arn:aws:lambda:us-west-2:246813579024:function:"
    r"keep-glm52-h1g-rehearsal-probe:[1-9][0-9]*$"
)
_ARTIFACT_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "run_id",
    "activation_id",
    "deployment_identity_sha256",
    "collector_function_version_arn",
    "lambda_request_id",
    "measurement_id",
    "input_identity_sha256",
    "probe_identity_sha256",
    "probe_evidence",
    "wire_ledger",
    "wire_ledger_identity_sha256",
    "measurement",
    "canonical_identity_sha256",
}


class RehearsalCollectorError(ValueError):
    """The deployed rehearsal input or evidence failed closed."""


@dataclass(frozen=True)
class CollectorConfig:
    account_id: str
    region: str
    run_id: str
    activation_id: str
    deployment_identity_sha256: str
    bucket: str
    expected_bucket_owner: str
    function_version_arn: str
    probe_version_arn: str


@dataclass(frozen=True)
class CollectorRuntimeIdentity:
    function_version_arn: str
    lambda_request_id: str
    lambda_environment_id: str
    cold_start: bool


@dataclass(frozen=True)
class CollectorServices:
    s3: object
    probe: object
    production_phases: object
    faults: object
    monotonic: Callable[[], float]
    sleeper: Callable[[float], None]
    utcnow: Callable[[], object]


def _fail(message: str) -> RehearsalCollectorError:
    return RehearsalCollectorError(message)


def _validate_config_runtime(
    config: CollectorConfig,
    runtime: CollectorRuntimeIdentity,
) -> None:
    if (
        type(config) is not CollectorConfig
        or config.account_id != "246813579024"
        or config.region != "us-west-2"
        or config.run_id != "glm52-sky-20260724"
        or type(config.activation_id) is not str
        or not config.activation_id
        or not _SHA.fullmatch(config.deployment_identity_sha256)
        or config.bucket
        != "keep-glm52-h1g-rehearsal-246813579024-us-west-2"
        or config.expected_bucket_owner != config.account_id
        or not _VERSION_ARN.fullmatch(config.function_version_arn)
        or not _PROBE_VERSION_ARN.fullmatch(config.probe_version_arn)
    ):
        raise _fail("collector configuration coordinates drifted")
    if (
        type(runtime) is not CollectorRuntimeIdentity
        or runtime.function_version_arn != config.function_version_arn
        or not _UUID.fullmatch(runtime.lambda_request_id)
        or not _SHA.fullmatch(runtime.lambda_environment_id)
        or type(runtime.cold_start) is not bool
    ):
        raise _fail("collector runtime identity drifted")


def _http_success(value: object, operation: str) -> dict[str, object]:
    if type(value) is not dict:
        raise _fail(operation + " response was not an exact object")
    metadata = value.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
    ):
        raise _fail(operation + " response metadata was not authenticated")
    return value


def _checksum(raw: bytes) -> str:
    return base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")


def _response_version_checksum(
    value: dict[str, object],
    *,
    expected_checksum: str,
    operation: str,
) -> str:
    version = value.get("VersionId")
    metadata = value["ResponseMetadata"]
    assert type(metadata) is dict
    headers = metadata.get("HTTPHeaders")
    if (
        type(version) is not str
        or not version
        or value.get("ChecksumSHA256") != expected_checksum
        or type(headers) is not dict
        or headers.get("x-amz-version-id") != version
        or headers.get("x-amz-checksum-sha256") != expected_checksum
        or type(headers.get("date")) is not str
        or not headers["date"]
    ):
        raise _fail(operation + " version or checksum was not authenticated")
    return version


def _put_immutable(
    services: CollectorServices,
    config: CollectorConfig,
    *,
    key: str,
    raw: bytes,
) -> tuple[str, str, str]:
    checksum = _checksum(raw)
    try:
        response = services.s3.put_object(
            Bucket=config.bucket,
            Key=key,
            Body=raw,
            ContentType="application/json",
            IfNoneMatch="*",
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=checksum,
            ExpectedBucketOwner=config.expected_bucket_owner,
        )
    except Exception as exc:
        raise _fail("conditional canary write failed closed") from exc
    value = _http_success(response, "PutObject")
    version = _response_version_checksum(
        value,
        expected_checksum=checksum,
        operation="PutObject",
    )
    etag = value.get("ETag")
    if type(etag) is not str or not etag.startswith('"') or not etag.endswith('"'):
        raise _fail("PutObject ETag was not authenticated")
    metadata = value["ResponseMetadata"]
    assert type(metadata) is dict
    return version, checksum, str(metadata["RequestId"])


def _list_versions(
    services: CollectorServices,
    config: CollectorConfig,
    *,
    prefix: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    key_marker: Optional[str] = None
    version_marker: Optional[str] = None
    versions: list[dict[str, object]] = []
    deletes: list[dict[str, object]] = []
    request_ids: list[str] = []
    seen_markers: set[tuple[str, str]] = set()
    for _page in range(1, 1001):
        kwargs: dict[str, object] = {
            "Bucket": config.bucket,
            "Prefix": prefix,
            "ExpectedBucketOwner": config.expected_bucket_owner,
            "MaxKeys": 1000,
        }
        if key_marker is not None and version_marker is not None:
            kwargs["KeyMarker"] = key_marker
            kwargs["VersionIdMarker"] = version_marker
        try:
            response = services.s3.list_object_versions(**kwargs)
        except Exception as exc:
            raise _fail("version pagination failed closed") from exc
        value = _http_success(response, "ListObjectVersions")
        metadata = value["ResponseMetadata"]
        assert type(metadata) is dict
        request_ids.append(str(metadata["RequestId"]))
        if key_marker is not None and (
            value.get("KeyMarker") != key_marker
            or value.get("VersionIdMarker") != version_marker
        ):
            raise _fail("version pagination marker echo drifted")
        page_versions = value.get("Versions")
        page_deletes = value.get("DeleteMarkers")
        if type(page_versions) is not list or type(page_deletes) is not list:
            raise _fail("version pagination rows drifted")
        if any(type(row) is not dict for row in page_versions + page_deletes):
            raise _fail("version pagination row type drifted")
        versions.extend(page_versions)
        deletes.extend(page_deletes)
        if value.get("IsTruncated") is False:
            return versions, deletes, request_ids
        if value.get("IsTruncated") is not True:
            raise _fail("version pagination truncation flag drifted")
        next_key = value.get("NextKeyMarker")
        next_version = value.get("NextVersionIdMarker")
        if (
            type(next_key) is not str
            or not next_key
            or type(next_version) is not str
            or not next_version
            or (next_key, next_version) in seen_markers
        ):
            raise _fail("version pagination did not advance")
        seen_markers.add((next_key, next_version))
        key_marker = next_key
        version_marker = next_version
    raise _fail("version pagination exceeded the frozen page bound")


def _read_exact(
    services: CollectorServices,
    config: CollectorConfig,
    *,
    key: str,
    version_id: str,
    expected_raw: Optional[bytes] = None,
) -> tuple[bytes, str, str, str]:
    kwargs = {
        "Bucket": config.bucket,
        "Key": key,
        "VersionId": version_id,
        "ChecksumMode": "ENABLED",
        "ExpectedBucketOwner": config.expected_bucket_owner,
    }
    try:
        get_response = services.s3.get_object(**kwargs)
        head_response = services.s3.head_object(
            Bucket=config.bucket,
            Key=key,
            VersionId=version_id,
            ChecksumMode="ENABLED",
            ExpectedBucketOwner=config.expected_bucket_owner,
        )
    except Exception as exc:
        raise _fail("exact version readback failed closed") from exc
    get_value = _http_success(get_response, "GetObject")
    head_value = _http_success(head_response, "HeadObject")
    body = get_value.get("Body")
    if not hasattr(body, "read"):
        raise _fail("GetObject body was not readable")
    raw = body.read()
    if type(raw) is not bytes or (expected_raw is not None and raw != expected_raw):
        raise _fail("GetObject bytes drifted")
    checksum = _checksum(raw)
    get_version = _response_version_checksum(
        get_value,
        expected_checksum=checksum,
        operation="GetObject",
    )
    head_version = _response_version_checksum(
        head_value,
        expected_checksum=checksum,
        operation="HeadObject",
    )
    if (
        get_version != version_id
        or head_version != version_id
        or get_value.get("ContentLength") != len(raw)
        or head_value.get("ContentLength") != len(raw)
        or get_value.get("ETag") != head_value.get("ETag")
    ):
        raise _fail("exact version GET/HEAD identity drifted")
    get_metadata = get_value["ResponseMetadata"]
    head_metadata = head_value["ResponseMetadata"]
    assert type(get_metadata) is dict and type(head_metadata) is dict
    return (
        raw,
        checksum,
        str(get_metadata["RequestId"]),
        str(head_metadata["RequestId"]),
    )


def _self_hashed(body: dict[str, object]) -> dict[str, object]:
    if "canonical_identity_sha256" in body:
        raise _fail("self-hashed body already contained an identity")
    result = dict(body)
    result["canonical_identity_sha256"] = canonical_sha256(body)
    return result


def _parse_self_hashed(
    value: object,
    *,
    expected_fields: set[str],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected_fields:
        raise _fail(label + " fields drifted")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256")
    if type(identity) is not str or identity != canonical_sha256(body):
        raise _fail(label + " canonical identity drifted")
    return value


def _parse_probe(
    value: object,
    config: CollectorConfig,
    request: Mapping[str, object],
) -> dict[str, object]:
    fields = {
        "schema_version",
        "record_type",
        "activation_id",
        "measurement_id",
        "candidate_bucket",
        "candidate_key",
        "candidate_version_id",
        "candidate_body_sha256",
        "executed_version_arn",
        "probe_lambda_request_id",
        "health_path",
        "health_request_identity_sha256",
        "health_response_request_id",
        "health_status_code",
        "health_response_body_sha256",
        "role_path",
        "role_request_identity_sha256",
        "role_response_request_id",
        "role_status_code",
        "role_response_body_sha256",
        "relay_call_count",
        "sky_post_call_count",
        "canonical_identity_sha256",
    }
    result = _parse_self_hashed(
        value,
        expected_fields=fields,
        label="rehearsal probe",
    )
    request_binding = {
        "activation_id": request["activation_id"],
        "measurement_id": request["measurement_id"],
        "candidate_bucket": request["candidate_bucket"],
        "candidate_key": request["candidate_key"],
        "candidate_version_id": request["candidate_version_id"],
        "candidate_body_sha256": request["candidate_body_sha256"],
    }
    health_request_identity = canonical_sha256(
        {
            "method": "GET",
            "path": "/api/health",
            **request_binding,
        }
    )
    role_request_identity = canonical_sha256(
        {
            "method": "GET",
            "path": "/users/role",
            **request_binding,
        }
    )
    if (
        result["schema_version"] != 1
        or result["record_type"]
        != "glm52_task11_rehearsal_probe_result_v1"
        or any(
            result[field] != expected
            for field, expected in request_binding.items()
        )
        or result["executed_version_arn"] != config.probe_version_arn
        or _UUID.fullmatch(str(result["probe_lambda_request_id"])) is None
        or result["health_path"] != "/api/health"
        or result["health_request_identity_sha256"]
        != health_request_identity
        or type(result["health_response_request_id"]) is not str
        or not result["health_response_request_id"]
        or result["health_status_code"] != 200
        or _SHA.fullmatch(
            str(result["health_response_body_sha256"])
        )
        is None
        or result["role_path"] != "/users/role"
        or result["role_request_identity_sha256"]
        != role_request_identity
        or type(result["role_response_request_id"]) is not str
        or not result["role_response_request_id"]
        or result["role_status_code"] != 200
        or _SHA.fullmatch(
            str(result["role_response_body_sha256"])
        )
        is None
        or result["relay_call_count"] != 0
        or result["sky_post_call_count"] != 0
    ):
        raise _fail("rehearsal probe request binding drifted")
    return result


def _read_bucket_policy(
    services: CollectorServices,
    config: CollectorConfig,
) -> tuple[str, str]:
    try:
        response = services.s3.get_bucket_policy(
            Bucket=config.bucket,
            ExpectedBucketOwner=config.expected_bucket_owner,
        )
    except Exception as exc:
        raise _fail("canary bucket policy readback failed closed") from exc
    value = _http_success(response, "GetBucketPolicy")
    policy = value.get("Policy")
    if type(policy) is not str or not policy:
        raise _fail("canary bucket policy was absent")
    try:
        parsed = json.loads(policy)
    except json.JSONDecodeError as exc:
        raise _fail("canary bucket policy was not JSON") from exc
    if canonical_json_bytes(parsed).decode("utf-8") != policy:
        raise _fail("canary bucket policy was not canonical")
    metadata = value["ResponseMetadata"]
    assert type(metadata) is dict
    return policy, str(metadata["RequestId"])


def _spans(
    names: tuple[str, ...],
    boundaries: tuple[float, ...],
) -> tuple[PhaseSpan, ...]:
    if len(boundaries) != len(names) + 1:
        raise _fail("timing boundary cardinality drifted")
    return tuple(
        PhaseSpan(names[index], boundaries[index], boundaries[index + 1])
        for index in range(len(names))
    )


_PRODUCTION_PHASE_METHODS = (
    (
        "TEMPLATE_CHANGE_SET_QUIESCENCE_MAINTENANCE_SEAL_PREFLIGHT",
        (
            ("prove_preauthorized_batch_template", CLOSURE_STEPS[0]),
            ("acquire_cfn_quiescence", CLOSURE_STEPS[1]),
            (
                "revalidate_runtime_attachments_and_seals",
                CLOSURE_STEPS[2],
            ),
        ),
    ),
    (
        "CLIENT_WARMING_AND_IMMUTABLE_CONSTRUCTION",
        (
            (
                "warm_clients_and_construct",
                "CLIENT_WARMING_AND_IMMUTABLE_CONSTRUCTION",
            ),
        ),
    ),
    (
        "NON_AUTHORITATIVE_SIZING",
        (
            (
                "size_non_authoritative",
                "NON_AUTHORITATIVE_SIZING",
            ),
        ),
    ),
    (
        "STABLE_TLS_SKY_IDENTITY_PREFLIGHT",
        (
            (
                "stable_tls_sky_identity_preflight",
                "STABLE_TLS_SKY_IDENTITY_PREFLIGHT",
            ),
        ),
    ),
)
_PRODUCTION_RECEIPT_SEQUENCE = tuple(
    (phase_name, method_name, step_name)
    for phase_name, methods in _PRODUCTION_PHASE_METHODS
    for method_name, step_name in methods
)


def _task11_request(value: object, config: CollectorConfig) -> ClosureRequest:
    if (
        type(value) is not dict
        or set(value) != set(ClosureRequest.__dataclass_fields__)
    ):
        raise _fail("deployed Task 11 request field set drifted")
    try:
        request = validate_closure_request(ClosureRequest(**value))
    except (TypeError, ValueError) as exc:
        raise _fail("deployed Task 11 request identity drifted") from exc
    if request.activation_id != config.activation_id:
        raise _fail("deployed Task 11 request scope drifted")
    return request


def _task11_boundary(value: object, config: CollectorConfig) -> None:
    if (
        type(value) is not dict
        or set(value)
        != {
            "bucket",
            "key",
            "version_id",
            "file_sha256",
            "body_sha256",
        }
        or value["bucket"] != "keep-glm52-campaign"
        or value["key"]
        != (
            f"campaigns/{config.run_id}/authorities/task11/"
            f"{config.activation_id}/00000001.json"
        )
        or type(value["version_id"]) is not str
        or not value["version_id"]
        or value["version_id"] == "null"
        or type(value["file_sha256"]) is not str
        or _SHA.fullmatch(value["file_sha256"]) is None
        or type(value["body_sha256"]) is not str
        or _SHA.fullmatch(value["body_sha256"]) is None
    ):
        raise _fail("deployed Task 11 boundary coordinate drifted")


def _run_production_equivalent_preflight(
    services: CollectorServices,
    config: CollectorConfig,
    runtime: CollectorRuntimeIdentity,
    measurement_id: str,
    request: ClosureRequest,
) -> tuple[tuple[float, ...], list[dict[str, object]]]:
    """Measure the exact named Task 11 production read-only phase methods."""

    boundaries = [services.monotonic()]
    wire = []
    receipt_identities = set()
    operation_identities = set()
    custody_nonce_sha256 = hashlib.sha256(
        (
            runtime.lambda_request_id
            + ":"
            + config.deployment_identity_sha256
            + ":"
            + measurement_id
        ).encode("ascii")
    ).hexdigest()
    for phase_name, methods in _PRODUCTION_PHASE_METHODS:
        started = boundaries[-1]
        for method_name, step_name in methods:
            method = getattr(services.production_phases, method_name, None)
            if not callable(method):
                raise _fail("named production phase method is absent")
            try:
                receipt = method(
                    request=request,
                    custody_nonce_sha256=custody_nonce_sha256,
                )
            except Exception as exc:
                raise _fail(
                    "named production phase failed closed"
                ) from exc
            if type(receipt) is not StepReceipt:
                raise _fail("typed production receipt was absent")
            try:
                expected = build_step_receipt(
                    step_name=step_name,
                    operation_identity_sha256=(
                        receipt.operation_identity_sha256
                    ),
                )
            except ValueError as exc:
                raise _fail(
                    "typed production receipt identity drifted"
                ) from exc
            if receipt != expected:
                raise _fail("typed production receipt identity drifted")
            if (
                receipt.canonical_identity_sha256 in receipt_identities
                or receipt.operation_identity_sha256 in operation_identities
            ):
                raise _fail("production receipt identity was replayed")
            receipt_identities.add(receipt.canonical_identity_sha256)
            operation_identities.add(receipt.operation_identity_sha256)
            wire.append(
                {
                    "operation": method_name,
                    "step_name": step_name,
                    "operation_identity_sha256": (
                        receipt.operation_identity_sha256
                    ),
                    "receipt_identity_sha256": (
                        receipt.canonical_identity_sha256
                    ),
                    "phase_name": phase_name,
                }
            )
        ended = services.monotonic()
        if ended <= started:
            raise _fail("production phase timing did not advance")
        boundaries.append(ended)
    return tuple(boundaries), wire


def collect_rehearsal(
    event: object,
    *,
    config: CollectorConfig,
    runtime: CollectorRuntimeIdentity,
    services: CollectorServices,
) -> dict[str, object]:
    if type(event) is not dict or set(event) != _COLLECT_FIELDS:
        raise RehearsalCollectorError("collector event field set drifted")
    _validate_config_runtime(config, runtime)
    measurement_id = event["measurement_id"]
    scenario = event["scenario"]
    if (
        event["schema_version"] != 1
        or event["record_type"] != "glm52_task11_collect_rehearsal_v1"
        or event["activation_id"] != config.activation_id
        or type(measurement_id) is not str
        or measurement_id not in COLLECTOR_MEASUREMENT_IDS
        or scenario != _SCENARIOS[measurement_id]
    ):
        raise _fail("collector event identity or scenario drifted")
    request = _task11_request(event["task11_request"], config)
    _task11_boundary(event["task11_boundary"], config)

    event_identity = canonical_sha256(event)
    candidate_key = (
        f"rehearsal/canary/{config.activation_id}/"
        f"{config.deployment_identity_sha256}/{measurement_id}.json"
    )
    measurement_key = (
        f"rehearsal/measurements/{config.activation_id}/"
        f"{config.deployment_identity_sha256}/{measurement_id}.json"
    )
    candidate = _self_hashed(
        {
            "schema_version": 1,
            "record_type": "glm52_task11_rehearsal_canary_v1",
            "account_id": config.account_id,
            "region": config.region,
            "run_id": config.run_id,
            "activation_id": config.activation_id,
            "deployment_identity_sha256": config.deployment_identity_sha256,
            "collector_function_version_arn": config.function_version_arn,
            "lambda_request_id": runtime.lambda_request_id,
            "measurement_id": measurement_id,
            "input_identity_sha256": event_identity,
        }
    )
    candidate_raw = canonical_json_bytes(candidate)
    candidate_body_sha256 = hashlib.sha256(candidate_raw).hexdigest()

    preflight_boundaries, wire = _run_production_equivalent_preflight(
        services,
        config,
        runtime,
        str(measurement_id),
        request,
    )
    closure_started = preflight_boundaries[0]
    suffix_started = preflight_boundaries[-1]
    candidate_version, candidate_checksum, put_request = _put_immutable(
        services,
        config,
        key=candidate_key,
        raw=candidate_raw,
    )
    wire.append(
        {
            "operation": "PutObject",
            "bucket": config.bucket,
            "key": candidate_key,
            "version_id": candidate_version,
            "request_id": put_request,
            "body_sha256": candidate_body_sha256,
            "checksum_sha256_base64": candidate_checksum,
        }
    )
    versions, deletes, list_requests = _list_versions(
        services,
        config,
        prefix=candidate_key,
    )
    if deletes or len(versions) != 1:
        raise _fail("canary version inventory drifted")
    candidate_row = versions[0]
    if (
        candidate_row.get("Key") != candidate_key
        or candidate_row.get("VersionId") != candidate_version
        or candidate_row.get("IsLatest") is not True
        or candidate_row.get("Size") != len(candidate_raw)
    ):
        raise _fail("canary current version identity drifted")
    for request_id in list_requests:
        wire.append(
            {
                "operation": "ListObjectVersions",
                "bucket": config.bucket,
                "key": candidate_key,
                "version_id": candidate_version,
                "request_id": request_id,
            }
        )
    _, read_checksum, get_request, head_request = _read_exact(
        services,
        config,
        key=candidate_key,
        version_id=candidate_version,
        expected_raw=candidate_raw,
    )
    if read_checksum != candidate_checksum:
        raise _fail("canary readback checksum drifted")
    wire.extend(
        [
            {
                "operation": "GetObject",
                "bucket": config.bucket,
                "key": candidate_key,
                "version_id": candidate_version,
                "request_id": get_request,
            },
            {
                "operation": "HeadObject",
                "bucket": config.bucket,
                "key": candidate_key,
                "version_id": candidate_version,
                "request_id": head_request,
            },
        ]
    )
    suffix_phase_0_end = services.monotonic()
    suffix_phase_1_end = suffix_phase_0_end

    first_policy = services.monotonic()
    policy_one, policy_request_one = _read_bucket_policy(services, config)
    services.sleeper(10.0)
    second_policy = services.monotonic()
    policy_two, policy_request_two = _read_bucket_policy(services, config)
    if policy_one != policy_two:
        raise _fail("canary bucket policy changed between readbacks")
    wire.extend(
        [
            {
                "operation": "GetBucketPolicy",
                "resource": config.bucket,
                "request_id": policy_request_one,
                "policy_identity_sha256": canonical_sha256(
                    json.loads(policy_one)
                ),
            },
            {
                "operation": "GetBucketPolicy",
                "resource": config.bucket,
                "request_id": policy_request_two,
                "policy_identity_sha256": canonical_sha256(
                    json.loads(policy_two)
                ),
            },
        ]
    )
    suffix_phase_2_end = services.monotonic()
    suffix_phase_3_end = suffix_phase_2_end

    fault_started = services.monotonic()
    try:
        unwind_reported = services.faults.exercise(str(scenario))
    except Exception as exc:
        raise _fail("injected fault exercise escaped its unwind") from exc
    fault_ended = services.monotonic()
    if (
        type(unwind_reported) not in (int, float)
        or type(unwind_reported) is bool
        or float(unwind_reported) != fault_ended - fault_started
        or (
            scenario == "NONE"
            and float(unwind_reported) != 0.0
        )
        or (
            scenario != "NONE"
            and not (0.0 < float(unwind_reported) <= 55.0)
        )
    ):
        raise _fail("injected fault unwind evidence drifted")
    wire.append(
        {
            "operation": "InjectedFault",
            "scenario": scenario,
            "unwind_seconds": float(unwind_reported),
        }
    )
    suffix_phase_4_end = services.monotonic()

    probe_request = {
        "schema_version": 1,
        "record_type": "glm52_task11_rehearsal_probe_v1",
        "activation_id": config.activation_id,
        "measurement_id": measurement_id,
        "candidate_bucket": config.bucket,
        "candidate_key": candidate_key,
        "candidate_version_id": candidate_version,
        "candidate_body_sha256": candidate_body_sha256,
        "candidate_checksum_sha256_base64": candidate_checksum,
    }
    try:
        probe = _parse_probe(
            services.probe.inspect(probe_request),
            config,
            probe_request,
        )
    except RehearsalCollectorError:
        raise
    except Exception as exc:
        raise _fail("version-pinned rehearsal probe failed closed") from exc
    wire.append(
        {
            "operation": "Invoke",
            "resource": config.probe_version_arn,
            "probe_lambda_request_id": probe["probe_lambda_request_id"],
            "health_response_request_id": (
                probe["health_response_request_id"]
            ),
            "role_response_request_id": probe["role_response_request_id"],
            "relay_call_count": 0,
            "sky_post_call_count": 0,
        }
    )
    suffix_phase_5_end = services.monotonic()
    suffix_phase_6_end = suffix_phase_5_end
    suffix_phase_7_end = suffix_phase_6_end
    closure_phase_4_end = suffix_phase_7_end

    closure_names = tuple(CLOSURE_PHASE_CEILINGS)
    suffix_names = tuple(SUFFIX_PHASE_CEILINGS)
    closure_spans = _spans(
        closure_names,
        (
            closure_started,
            preflight_boundaries[1],
            preflight_boundaries[2],
            preflight_boundaries[3],
            suffix_started,
            closure_phase_4_end,
            closure_phase_4_end,
            closure_phase_4_end,
            closure_phase_4_end,
        ),
    )
    suffix_spans = _spans(
        suffix_names,
        (
            suffix_started,
            suffix_phase_0_end,
            suffix_phase_1_end,
            suffix_phase_2_end,
            suffix_phase_3_end,
            suffix_phase_4_end,
            suffix_phase_5_end,
            suffix_phase_6_end,
            suffix_phase_7_end,
        ),
    )
    try:
        measurement = build_rehearsal_measurement(
            rehearsal_id=str(measurement_id),
            provenance="DEPLOYED_REHEARSAL",
            lambda_environment_id=runtime.lambda_environment_id,
            cold_start=runtime.cold_start,
            path_proofs=REHEARSAL_PATH_PROOFS,
            canary_bucket="keep-glm52-h1g-rehearsal",
            canary_key=candidate_key,
            canary_can_satisfy_production_authority=False,
            canary_worker_readable=False,
            canary_admission_readable=False,
            effect_counts=ZERO_PRODUCTION_EFFECTS,
            relay_call_count=0,
            closure_spans=closure_spans,
            suffix_spans=suffix_spans,
            first_policy_readback_monotonic_seconds=first_policy,
            second_policy_readback_monotonic_seconds=second_policy,
            injected_failure_kind=str(scenario),
            failure_unwind_seconds=float(unwind_reported),
        )
    except ValueError as exc:
        raise _fail("derived rehearsal measurement failed validation") from exc

    wire_identity = canonical_sha256(wire)
    artifact = _self_hashed(
        {
            "schema_version": 1,
            "record_type": "glm52_task11_rehearsal_measurement_artifact_v1",
            "account_id": config.account_id,
            "region": config.region,
            "run_id": config.run_id,
            "activation_id": config.activation_id,
            "deployment_identity_sha256": config.deployment_identity_sha256,
            "collector_function_version_arn": config.function_version_arn,
            "lambda_request_id": runtime.lambda_request_id,
            "measurement_id": measurement_id,
            "input_identity_sha256": event_identity,
            "probe_identity_sha256": probe["canonical_identity_sha256"],
            "probe_evidence": probe,
            "wire_ledger": wire,
            "wire_ledger_identity_sha256": wire_identity,
            "measurement": asdict(measurement),
        }
    )
    artifact_raw = canonical_json_bytes(artifact)
    measurement_version, measurement_checksum, _request_id = _put_immutable(
        services,
        config,
        key=measurement_key,
        raw=artifact_raw,
    )
    readback, readback_checksum, _get_request, _head_request = _read_exact(
        services,
        config,
        key=measurement_key,
        version_id=measurement_version,
        expected_raw=artifact_raw,
    )
    if readback != artifact_raw or readback_checksum != measurement_checksum:
        raise _fail("measurement artifact readback drifted")

    return _self_hashed(
        {
            "schema_version": 1,
            "record_type": "glm52_task11_collect_rehearsal_result_v1",
            "status": "DEPLOYED_REHEARSAL_RECORDED",
            "account_id": config.account_id,
            "region": config.region,
            "run_id": config.run_id,
            "activation_id": config.activation_id,
            "measurement_id": measurement_id,
            "collector_function_version_arn": config.function_version_arn,
            "key": measurement_key,
            "version_id": measurement_version,
            "file_sha256": hashlib.sha256(artifact_raw).hexdigest(),
            "checksum_sha256_base64": measurement_checksum,
        }
    )


def finalize_rehearsal_gate(
    event: object,
    *,
    config: CollectorConfig,
    runtime: CollectorRuntimeIdentity,
    services: CollectorServices,
) -> dict[str, object]:
    if type(event) is not dict or set(event) != _FINALIZE_FIELDS:
        raise _fail("finalizer event field set drifted")
    _validate_config_runtime(config, runtime)
    if (
        event["schema_version"] != 1
        or event["record_type"]
        != "glm52_task11_finalize_rehearsal_gate_v1"
        or event["activation_id"] != config.activation_id
    ):
        raise _fail("finalizer event identity drifted")
    prefix = (
        f"rehearsal/measurements/{config.activation_id}/"
        f"{config.deployment_identity_sha256}/"
    )
    versions, deletes, _list_requests = _list_versions(
        services,
        config,
        prefix=prefix,
    )
    expected_keys = {
        prefix + measurement_id + ".json"
        for measurement_id in COLLECTOR_MEASUREMENT_IDS
    }
    actual_keys = [row.get("Key") for row in versions]
    if (
        deletes
        or len(versions) != 20
        or set(actual_keys) != expected_keys
        or len(set(actual_keys)) != len(actual_keys)
        or any(
            row.get("IsLatest") is not True
            or type(row.get("VersionId")) is not str
            or not row["VersionId"]
            for row in versions
        )
    ):
        raise _fail("measurement version inventory drifted")

    measurements = []
    probe_identities: set[str] = set()
    probe_lambda_request_ids: set[str] = set()
    health_request_identities: set[str] = set()
    health_response_request_ids: set[str] = set()
    role_request_identities: set[str] = set()
    role_response_request_ids: set[str] = set()
    production_operation_identities: set[str] = set()
    production_receipt_identities: set[str] = set()
    for row in sorted(versions, key=lambda item: str(item["Key"])):
        key = str(row["Key"])
        measurement_id = key.removeprefix(prefix).removesuffix(".json")
        version_id = str(row["VersionId"])
        raw, _checksum_value, _get_request, _head_request = _read_exact(
            services,
            config,
            key=key,
            version_id=version_id,
        )
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise _fail("measurement artifact was not JSON") from exc
        if canonical_json_bytes(value) != raw:
            raise _fail("measurement artifact bytes were not canonical")
        artifact = _parse_self_hashed(
            value,
            expected_fields=_ARTIFACT_FIELDS,
            label="measurement artifact",
        )
        wire = artifact["wire_ledger"]
        candidate_key = (
            f"rehearsal/canary/{config.activation_id}/"
            f"{config.deployment_identity_sha256}/{measurement_id}.json"
        )
        candidate_writes = [
            item
            for item in wire
            if type(item) is dict
            and item.get("operation") == "PutObject"
            and item.get("key") == candidate_key
        ]
        production_receipts = [
            item
            for item in wire
            if (
                type(item) is dict
                and item.get("operation")
                in {
                    method_name
                    for _phase_name, method_name, _step_name
                    in _PRODUCTION_RECEIPT_SEQUENCE
                }
            )
        ]
        if (
            artifact["schema_version"] != 1
            or artifact["record_type"]
            != "glm52_task11_rehearsal_measurement_artifact_v1"
            or artifact["account_id"] != config.account_id
            or artifact["region"] != config.region
            or artifact["run_id"] != config.run_id
            or artifact["activation_id"] != config.activation_id
            or artifact["deployment_identity_sha256"]
            != config.deployment_identity_sha256
            or artifact["collector_function_version_arn"]
            != config.function_version_arn
            or artifact["measurement_id"] != measurement_id
            or type(artifact["lambda_request_id"]) is not str
            or not _UUID.fullmatch(str(artifact["lambda_request_id"]))
            or type(wire) is not list
            or not wire
            or artifact["wire_ledger_identity_sha256"]
            != canonical_sha256(wire)
            or len(candidate_writes) != 1
            or len(production_receipts)
            != len(_PRODUCTION_RECEIPT_SEQUENCE)
            or production_receipts
            != wire[: len(_PRODUCTION_RECEIPT_SEQUENCE)]
            or any(
                set(receipt)
                != {
                    "operation",
                    "step_name",
                    "operation_identity_sha256",
                    "receipt_identity_sha256",
                    "phase_name",
                }
                or receipt["phase_name"] != expected_phase
                or receipt["operation"] != expected_operation
                or receipt["step_name"] != expected_step
                or type(receipt["operation_identity_sha256"]) is not str
                or _SHA.fullmatch(
                    receipt["operation_identity_sha256"]
                )
                is None
                or type(receipt["receipt_identity_sha256"]) is not str
                or _SHA.fullmatch(receipt["receipt_identity_sha256"])
                is None
                for receipt, (
                    expected_phase,
                    expected_operation,
                    expected_step,
                ) in zip(
                    production_receipts,
                    _PRODUCTION_RECEIPT_SEQUENCE,
                    strict=True,
                )
            )
            or any(
                type(item) is not dict
                or (
                    "bucket" in item
                    and (
                        item["bucket"] != config.bucket
                        or type(item.get("key")) is not str
                        or not str(item["key"]).startswith("rehearsal/")
                    )
                )
                or item.get("relay_call_count", 0) != 0
                or item.get("sky_post_call_count", 0) != 0
                for item in wire
            )
        ):
            raise _fail("measurement artifact authority drifted")
        production_operation_identities.update(
            receipt["operation_identity_sha256"]
            for receipt in production_receipts
        )
        production_receipt_identities.update(
            receipt["receipt_identity_sha256"]
            for receipt in production_receipts
        )
        candidate_write = candidate_writes[0]
        probe_request = {
            "schema_version": 1,
            "record_type": "glm52_task11_rehearsal_probe_v1",
            "activation_id": config.activation_id,
            "measurement_id": measurement_id,
            "candidate_bucket": config.bucket,
            "candidate_key": candidate_key,
            "candidate_version_id": candidate_write.get("version_id"),
            "candidate_body_sha256": candidate_write.get("body_sha256"),
            "candidate_checksum_sha256_base64": candidate_write.get(
                "checksum_sha256_base64"
            ),
        }
        probe = _parse_probe(
            artifact["probe_evidence"],
            config,
            probe_request,
        )
        if (
            artifact["probe_identity_sha256"]
            != probe["canonical_identity_sha256"]
        ):
            raise _fail("measurement probe identity drifted")
        probe_identities.add(probe["canonical_identity_sha256"])
        probe_lambda_request_ids.add(probe["probe_lambda_request_id"])
        health_request_identities.add(
            probe["health_request_identity_sha256"]
        )
        health_response_request_ids.add(
            probe["health_response_request_id"]
        )
        role_request_identities.add(probe["role_request_identity_sha256"])
        role_response_request_ids.add(probe["role_response_request_id"])
        try:
            measurement = rehearsal_measurement_from_mapping(
                artifact["measurement"]
            )
        except ValueError as exc:
            raise _fail("measurement payload failed authentication") from exc
        if measurement.rehearsal_id != measurement_id:
            raise _fail("measurement id binding drifted")
        measurements.append(measurement)
    if any(
        len(values) != len(COLLECTOR_MEASUREMENT_IDS)
        for values in (
            probe_identities,
            probe_lambda_request_ids,
            health_request_identities,
            health_response_request_ids,
            role_request_identities,
            role_response_request_ids,
        )
    ):
        raise _fail("measurement probe uniqueness drifted")
    expected_production_receipts = (
        len(COLLECTOR_MEASUREMENT_IDS)
        * len(_PRODUCTION_RECEIPT_SEQUENCE)
    )
    if (
        len(production_operation_identities)
        != expected_production_receipts
        or len(production_receipt_identities)
        != expected_production_receipts
    ):
        raise _fail("measurement production receipt uniqueness drifted")
    typed_measurements = tuple(measurements)
    try:
        gate = verify_no_post_rehearsals(typed_measurements)
    except ValueError as exc:
        raise _fail("measurement set did not prove cold/fault closure") from exc
    if gate.measurement_count != 20 or gate.cold_environment_count != 5:
        raise _fail("measurement set cardinality drifted")
    try:
        gate_raw = build_deployed_gate_document(
            account_id=config.account_id,
            region=config.region,
            run_id=config.run_id,
            activation_id=config.activation_id,
            deployment_identity_sha256=config.deployment_identity_sha256,
            measurements=typed_measurements,
        )
    except ValueError as exc:
        raise _fail("deployed gate construction failed closed") from exc
    gate_key = (
        f"rehearsal/gates/{config.activation_id}/CLOSURE_BUDGET.json"
    )
    gate_version, gate_checksum, _request_id = _put_immutable(
        services,
        config,
        key=gate_key,
        raw=gate_raw,
    )
    readback, readback_checksum, _get_request, _head_request = _read_exact(
        services,
        config,
        key=gate_key,
        version_id=gate_version,
        expected_raw=gate_raw,
    )
    if readback != gate_raw or readback_checksum != gate_checksum:
        raise _fail("deployed gate readback drifted")
    parsed_gate = json.loads(gate_raw)
    return _self_hashed(
        {
            "schema_version": 1,
            "record_type": "glm52_task11_finalize_rehearsal_gate_result_v1",
            "status": "CLOSURE_BUDGET_PROVEN",
            "account_id": config.account_id,
            "region": config.region,
            "run_id": config.run_id,
            "activation_id": config.activation_id,
            "collector_function_version_arn": config.function_version_arn,
            "key": gate_key,
            "version_id": gate_version,
            "file_sha256": hashlib.sha256(gate_raw).hexdigest(),
            "body_sha256": parsed_gate["canonical_body_sha256"],
            "checksum_sha256_base64": gate_checksum,
            "measurement_count": gate.measurement_count,
            "cold_environment_count": gate.cold_environment_count,
            "measurements_identity_sha256": (
                gate.measurements_identity_sha256
            ),
        }
    )


__all__ = [
    "COLLECTOR_MEASUREMENT_IDS",
    "CollectorConfig",
    "CollectorRuntimeIdentity",
    "CollectorServices",
    "RehearsalCollectorError",
    "collect_rehearsal",
    "finalize_rehearsal_gate",
]
