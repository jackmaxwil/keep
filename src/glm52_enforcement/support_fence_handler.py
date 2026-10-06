"""Strict request-coordinate-only entrypoint for H.1g fence transitions."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from time import sleep

from .canonical import canonical_json_bytes, canonical_sha256
from .cloudformation_stacks import (
    parse_stack_migration_transfer_checkpoint_v2,
)
from .fence_artifacts import (
    ArtifactCoordinate,
    ExecutorAuthorityClass,
    FenceSlot,
    FenceTransitionRequest,
    parse_artifact_coordinate,
    parse_fence_transition_request,
    parse_support_runtime_identity_coordinate,
)
from .fence_bootstrap_publication import (
    parse_bootstrap_publication_request_v2,
    parse_bridge_seed_publication_request_v2,
    publish_bootstrap_fence_artifacts_v2,
    publish_bridge_seed_policy_v2,
)
from .fence_executor import (
    FenceExecutionResult,
    FenceExecutor,
    PreparedFenceChangeSet,
    load_pinned_fence_entry_template,
    load_pinned_fence_manifest,
)
from .support_plane import (
    parse_support_runtime_identity,
    require_live_support_runtime_identity,
)
from .task13_fixed_artifacts import (
    ACCOUNT_ID,
    CAMPAIGN_BUCKET,
    REGION,
    RUN_ID,
    Task13FixedArtifactServices,
)

_ACCOUNT_ID = ACCOUNT_ID
_BUCKET = CAMPAIGN_BUCKET
_MAX_HISTORY_PAGES = 16
_ACTIVATION_ID = "glm52-v2-amber-quartz"
_BOOTSTRAP_PUBLISHER_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/keep-glm52-h1g-fence-bootstrap-artifact-publisher"
)
_BOOTSTRAP_PUBLISHER_SESSION_NAME = "glm52-h1g-bootstrap-publisher"
_SOURCE_PUBLISHER_ROLE_ARN = (
    "arn:aws:iam::246813579024:"
    "role/keep-glm52-h1g-fence-source-settled-artifact-publisher"
)
_SOURCE_PUBLISHER_SESSION_NAME = "glm52-h1g-source-settled-publisher"
_RETAINED_KMS_KEY_ARN_PREFIX = "arn:aws:kms:us-west-2:246813579024:key/"


@dataclass(frozen=True)
class FenceHandlerServices:
    """Injected v2 executor and fresh support-runtime read boundary."""

    s3: object
    executor: FenceExecutor
    authority_class: ExecutorAuthorityClass
    live_support_runtime_identity: Callable[[], Mapping[str, object]] | None = None


@dataclass(frozen=True)
class BootstrapPublicationHandlerServices:
    """Split read and assumed-role write boundary for bootstrap publication."""

    fixed_artifacts: Task13FixedArtifactServices

    def __post_init__(self) -> None:
        fixed = self.fixed_artifacts
        if (
            type(fixed) is not Task13FixedArtifactServices
            or fixed.total_max_attempts != 1
            or fixed.publisher_s3 is None
            or fixed.publisher_s3 is fixed.s3
        ):
            raise TypeError(
                "bootstrap publication services must use exact split clients"
            )


@dataclass(frozen=True)
class SourceSettlementHandlerServices:
    """Live evidence reads plus split-client source-settled publication."""

    evidence_reader_factory: Callable[[object], object]
    s3: object
    artifact_services: Task13FixedArtifactServices
    now_utc: Callable[[], datetime]
    sleep: Callable[[float], None]

    def __post_init__(self) -> None:
        fixed = self.artifact_services
        if (
            not callable(self.evidence_reader_factory)
            or self.s3 is None
            or not callable(self.now_utc)
            or not callable(self.sleep)
            or type(fixed) is not Task13FixedArtifactServices
            or fixed.total_max_attempts != 1
            or fixed.s3 is not self.s3
            or fixed.publisher_s3 is None
            or fixed.publisher_s3 is fixed.s3
        ):
            raise TypeError(
                "source settlement services must use live reads and split clients"
            )


def _fail(message: str) -> None:
    raise ValueError(message)


def _success_response(value: object, operation: str) -> Mapping[str, object]:
    if type(value) is not dict:
        _fail(operation + " response is not exact")
    metadata = value.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
    ):
        _fail(operation + " response metadata is unauthenticated")
    return value


def _service_method(service: object, name: str) -> Callable[..., object]:
    method = getattr(service, name, None)
    if not callable(method):
        _fail("handler service method is absent: " + name)
    return method


def _required_environment(name: str, *, expected: str | None = None) -> str:
    value = os.environ.get(name)
    if (
        type(value) is not str
        or not value
        or (expected is not None and value != expected)
    ):
        _fail(name + " is not exact")
    return value


def _build_bootstrap_publication_handler_services() -> (
    BootstrapPublicationHandlerServices
):
    """Build the zero-retry Lambda boundary and authenticate its writer role."""

    _required_environment("GLM52_ACCOUNT_ID", expected=ACCOUNT_ID)
    _required_environment("GLM52_REGION", expected=REGION)
    _required_environment("GLM52_RUN_ID", expected=RUN_ID)
    _required_environment("GLM52_ACTIVATION_ID", expected=_ACTIVATION_ID)
    _required_environment(
        "GLM52_MODEL_BUCKET_ARN",
        expected=f"arn:aws:s3:::{CAMPAIGN_BUCKET}",
    )
    kms_key_arn = _required_environment("GLM52_RETAINED_KMS_KEY_ARN")
    if not kms_key_arn.startswith(_RETAINED_KMS_KEY_ARN_PREFIX):
        _fail("GLM52_RETAINED_KMS_KEY_ARN is not exact")
    publisher_role_arn = _required_environment(
        "GLM52_BOOTSTRAP_PUBLISHER_ROLE_ARN",
        expected=_BOOTSTRAP_PUBLISHER_ROLE_ARN,
    )
    try:  # pragma: no cover - Lambda runtime dependency
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - Lambda runtime dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    config = Config(
        retries={"mode": "standard", "total_max_attempts": 1},
        connect_timeout=2,
        read_timeout=15,
    )
    base_session = boto3.Session(region_name=REGION)
    base_sts = base_session.client("sts", config=config)
    base_s3 = base_session.client("s3", config=config)
    assumed = _success_response(
        _service_method(base_sts, "assume_role")(
            RoleArn=publisher_role_arn,
            RoleSessionName=_BOOTSTRAP_PUBLISHER_SESSION_NAME,
            DurationSeconds=900,
        ),
        "AssumeRole",
    )
    credentials = assumed.get("Credentials")
    assumed_user = assumed.get("AssumedRoleUser")
    assumed_arn = (
        "arn:aws:sts::246813579024:assumed-role/"
        "keep-glm52-h1g-fence-bootstrap-artifact-publisher/"
        + _BOOTSTRAP_PUBLISHER_SESSION_NAME
    )
    if (
        type(credentials) is not dict
        or type(assumed_user) is not dict
        or assumed_user.get("Arn") != assumed_arn
        or type(assumed_user.get("AssumedRoleId")) is not str
        or not assumed_user["AssumedRoleId"].endswith(
            ":" + _BOOTSTRAP_PUBLISHER_SESSION_NAME
        )
    ):
        _fail("bootstrap publisher assumption identity is not exact")
    for field in ("AccessKeyId", "SecretAccessKey", "SessionToken"):
        if type(credentials.get(field)) is not str or not credentials[field]:
            _fail("bootstrap publisher temporary credentials are incomplete")
    publisher_session = boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=REGION,
    )
    publisher_sts = publisher_session.client("sts", config=config)
    publisher_s3 = publisher_session.client("s3", config=config)
    caller = _success_response(
        _service_method(publisher_sts, "get_caller_identity")(),
        "GetCallerIdentity",
    )
    if (
        caller.get("Account") != ACCOUNT_ID
        or caller.get("Arn") != assumed_arn
        or type(caller.get("UserId")) is not str
        or not caller["UserId"].endswith(":" + _BOOTSTRAP_PUBLISHER_SESSION_NAME)
    ):
        _fail("bootstrap publisher caller identity is not exact")
    return BootstrapPublicationHandlerServices(
        fixed_artifacts=Task13FixedArtifactServices(
            sts=base_sts,
            s3=base_s3,
            publisher_s3=publisher_s3,
            total_max_attempts=1,
        )
    )


def _build_source_settlement_handler_services(
    request: object,
) -> SourceSettlementHandlerServices:
    """Build zero-retry live readers and authenticate the source writer."""

    from .fence_source_settlement import SourceSettlementRequest

    if type(request) is not SourceSettlementRequest:
        raise TypeError("source settlement request must be exact")
    _required_environment("GLM52_ACCOUNT_ID", expected=ACCOUNT_ID)
    _required_environment("GLM52_REGION", expected=REGION)
    _required_environment("GLM52_RUN_ID", expected=RUN_ID)
    _required_environment(
        "GLM52_ACTIVATION_ID",
        expected=request.bootstrap_manifest.activation_id,
    )
    _required_environment(
        "GLM52_MODEL_BUCKET_ARN",
        expected=request.bootstrap_manifest.bucket_arn,
    )
    _required_environment(
        "GLM52_BOOTSTRAP_MANIFEST_COORDINATE",
        expected=canonical_json_bytes(
            request.bootstrap_manifest_coordinate.to_dict()
        ).decode("ascii"),
    )
    _required_environment(
        "GLM52_AUTHORITY_CLASS",
        expected="SOURCE_SETTLEMENT_MATERIALIZER",
    )
    kms_key_arn = _required_environment(
        "GLM52_RETAINED_KMS_KEY_ARN",
        expected=request.kms_key_arn,
    )
    if not kms_key_arn.startswith(_RETAINED_KMS_KEY_ARN_PREFIX):
        _fail("GLM52_RETAINED_KMS_KEY_ARN is not exact")
    publisher_role_arn = _required_environment(
        "GLM52_SOURCE_PUBLISHER_ROLE_ARN",
        expected=_SOURCE_PUBLISHER_ROLE_ARN,
    )
    try:  # pragma: no cover - Lambda runtime dependency
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - Lambda runtime dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    config = Config(
        retries={"mode": "standard", "total_max_attempts": 1},
        connect_timeout=2,
        read_timeout=15,
    )
    base_session = boto3.Session(region_name=REGION)
    base_sts = base_session.client("sts", config=config)
    base_s3 = base_session.client("s3", config=config)
    assumed = _success_response(
        _service_method(base_sts, "assume_role")(
            RoleArn=publisher_role_arn,
            RoleSessionName=_SOURCE_PUBLISHER_SESSION_NAME,
            DurationSeconds=900,
        ),
        "AssumeRole",
    )
    credentials = assumed.get("Credentials")
    assumed_user = assumed.get("AssumedRoleUser")
    assumed_arn = (
        "arn:aws:sts::246813579024:assumed-role/"
        "keep-glm52-h1g-fence-source-settled-artifact-publisher/"
        + _SOURCE_PUBLISHER_SESSION_NAME
    )
    if (
        type(credentials) is not dict
        or type(assumed_user) is not dict
        or assumed_user.get("Arn") != assumed_arn
        or type(assumed_user.get("AssumedRoleId")) is not str
        or not assumed_user["AssumedRoleId"].endswith(
            ":" + _SOURCE_PUBLISHER_SESSION_NAME
        )
    ):
        _fail("source publisher assumption identity is not exact")
    for field in ("AccessKeyId", "SecretAccessKey", "SessionToken"):
        if type(credentials.get(field)) is not str or not credentials[field]:
            _fail("source publisher temporary credentials are incomplete")
    publisher_session = boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=REGION,
    )
    publisher_sts = publisher_session.client("sts", config=config)
    publisher_s3 = publisher_session.client("s3", config=config)
    caller = _success_response(
        _service_method(publisher_sts, "get_caller_identity")(),
        "GetCallerIdentity",
    )
    if (
        caller.get("Account") != ACCOUNT_ID
        or caller.get("Arn") != assumed_arn
        or type(caller.get("UserId")) is not str
        or not caller["UserId"].endswith(":" + _SOURCE_PUBLISHER_SESSION_NAME)
    ):
        _fail("source publisher caller identity is not exact")
    fixed_artifacts = Task13FixedArtifactServices(
        sts=base_sts,
        s3=base_s3,
        publisher_s3=publisher_s3,
        total_max_attempts=1,
    )
    return SourceSettlementHandlerServices(
        evidence_reader_factory=lambda settlement_request: _CoordinateEvidenceReader(
            request=settlement_request,
            s3=base_s3,
        ),
        s3=base_s3,
        artifact_services=fixed_artifacts,
        now_utc=lambda: datetime.now(UTC),
        sleep=sleep,
    )


def _read_body(value: object) -> bytes:
    if type(value) is bytes:
        return value
    read_body = getattr(value, "read", None)
    if not callable(read_body):
        _fail("pinned artifact body is unreadable")
    raw = read_body()
    if type(raw) is not bytes:
        _fail("pinned artifact body is not bytes")
    return raw


def _json_lists_to_tuples(value: object) -> object:
    if type(value) is list:
        return tuple(_json_lists_to_tuples(item) for item in value)
    if type(value) is dict:
        return {key: _json_lists_to_tuples(item) for key, item in value.items()}
    return value


class _CoordinateEvidenceReader:
    def __init__(self, *, request: object, s3: object) -> None:
        from .fence_source_settlement import SourceSettlementRequest

        if type(request) is not SourceSettlementRequest or s3 is None:
            raise TypeError("source settlement evidence reader is not exact")
        self._request = request
        self._s3 = s3

    def _load(self, name: str) -> object:
        from .fence_source_settlement import (
            SOURCE_SETTLEMENT_EVIDENCE_READER_NAMES,
        )

        if name not in SOURCE_SETTLEMENT_EVIDENCE_READER_NAMES:
            _fail("source settlement evidence reader name is foreign")
        coordinate = self._request.evidence_coordinates[name]
        response = _success_response(
            _service_method(self._s3, "get_object")(
                Bucket=coordinate.bucket,
                Key=coordinate.key,
                VersionId=coordinate.version_id,
                ExpectedBucketOwner=_ACCOUNT_ID,
                ChecksumMode="ENABLED",
            ),
            "GetSourceSettlementEvidence",
        )
        if (
            response.get("VersionId") != coordinate.version_id
            or response.get("ServerSideEncryption") != "aws:kms"
            or response.get("SSEKMSKeyId") != self._request.kms_key_arn
        ):
            _fail(name + " live evidence object identity drifted")
        raw = _read_body(response.get("Body"))
        if hashlib.sha256(raw).hexdigest() != coordinate.file_sha256:
            _fail(name + " live evidence file hash drifted")
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            _fail(name + " live evidence is not singular canonical JSON")
        try:
            value = json.loads(
                raw[:-1],
                object_pairs_hook=_reject_duplicates,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(name + " live evidence is malformed") from exc
        if raw != canonical_json_bytes(value) + b"\n":
            _fail(name + " live evidence bytes are not canonical")
        expected_identity = coordinate.canonical_identity_sha256
        if type(value) is dict and "canonical_identity_sha256" in value:
            identity = value.get("canonical_identity_sha256")
            unsigned = dict(value)
            unsigned.pop("canonical_identity_sha256")
            if identity != expected_identity or canonical_sha256(unsigned) != identity:
                _fail(name + " signed live evidence identity drifted")
        elif canonical_sha256(value) != expected_identity:
            _fail(name + " live evidence identity drifted")
        return _json_lists_to_tuples(value)

    def __getattr__(self, name: str) -> Callable[[], object]:
        from .fence_source_settlement import (
            SOURCE_SETTLEMENT_EVIDENCE_READER_NAMES,
        )

        if name not in SOURCE_SETTLEMENT_EVIDENCE_READER_NAMES:
            raise AttributeError(name)
        return lambda: self._load(name)


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            _fail("pinned artifact contains a duplicate JSON key")
        value[key] = item
    return value


def _load_exact_json(
    *, s3: object, coordinate: ArtifactCoordinate, label: str
) -> Mapping[str, object]:
    """Load one immutable fixed-key version after complete history proof."""

    if not isinstance(coordinate, ArtifactCoordinate):
        _fail(label + " coordinate is not one parsed ArtifactCoordinate")
    if coordinate.bucket != _BUCKET:
        _fail(label + " bucket is foreign")
    versions: list[Mapping[str, object]] = []
    delete_markers: list[Mapping[str, object]] = []
    key_marker: str | None = None
    version_marker: str | None = None
    seen: set[tuple[str, str]] = set()
    for _ in range(_MAX_HISTORY_PAGES):
        request: dict[str, object] = {
            "Bucket": coordinate.bucket,
            "Prefix": coordinate.key,
            "ExpectedBucketOwner": _ACCOUNT_ID,
        }
        if key_marker is not None:
            request["KeyMarker"] = key_marker
            request["VersionIdMarker"] = version_marker
        response = _success_response(
            _service_method(s3, "list_object_versions")(**request),
            "ListObjectVersions",
        )
        page_versions = response.get("Versions")
        page_markers = response.get("DeleteMarkers")
        if (
            type(page_versions) is not list
            or type(page_markers) is not list
            or any(type(row) is not dict for row in page_versions)
            or any(type(row) is not dict for row in page_markers)
        ):
            _fail(label + " fixed-key history is malformed")
        versions.extend(
            row for row in page_versions if row.get("Key") == coordinate.key
        )
        delete_markers.extend(
            row for row in page_markers if row.get("Key") == coordinate.key
        )
        if response.get("IsTruncated") is False:
            break
        next_key = response.get("NextKeyMarker")
        next_version = response.get("NextVersionIdMarker")
        if (
            type(next_key) is not str
            or not next_key
            or type(next_version) is not str
            or not next_version
            or (next_key, next_version) in seen
        ):
            _fail(label + " fixed-key history pagination is ambiguous")
        seen.add((next_key, next_version))
        key_marker = next_key
        version_marker = next_version
    else:
        _fail(label + " fixed-key history pagination is unbounded")
    if (
        delete_markers
        or len(versions) != 1
        or versions[0].get("VersionId") != coordinate.version_id
        or versions[0].get("IsLatest") is not True
    ):
        _fail(label + " FIXED_KEY_HISTORY_DRIFT")
    response = _success_response(
        _service_method(s3, "get_object")(
            Bucket=coordinate.bucket,
            Key=coordinate.key,
            VersionId=coordinate.version_id,
            ExpectedBucketOwner=_ACCOUNT_ID,
            ChecksumMode="ENABLED",
        ),
        "GetObjectVersion",
    )
    if response.get("VersionId") != coordinate.version_id:
        _fail(label + " returned another VersionId")
    raw = _read_body(response.get("Body"))
    if hashlib.sha256(raw).hexdigest() != coordinate.file_sha256:
        _fail(label + " file hash drifted")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail(label + " is not singular LF-terminated canonical JSON")
    try:
        value = json.loads(raw[:-1], object_pairs_hook=_reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(label + " is not canonical JSON") from exc
    if type(value) is not dict:
        _fail(label + " is not one object")
    return value


def _load_request(
    *, s3: object, coordinate: ArtifactCoordinate
) -> FenceTransitionRequest:
    value = _load_exact_json(
        s3=s3, coordinate=coordinate, label="fence transition request"
    )
    request = parse_fence_transition_request(dict(value))
    if request.canonical_identity_sha256 != coordinate.canonical_identity_sha256:
        _fail("transition request coordinate identity drifted")
    return request


def _select_entry(manifest: object, request: FenceTransitionRequest):
    entries = getattr(manifest, "entries", None)
    selected_identity = request.to_dict()["selected_entry_identity_sha256"]
    matches = [
        entry
        for entry in entries or ()
        if getattr(entry, "entry_identity_sha256", None) == selected_identity
    ]
    if len(matches) != 1 or matches[0].slot is not request.slot:
        _fail("selected manifest entry is absent or ambiguous")
    return matches[0]


def _expected_support_contract(manifest: object) -> Mapping[str, object]:
    value = manifest.to_dict()
    matches = [
        row
        for row in value.get("executor_inventory", [])
        if type(row) is dict and row.get("authority_class") == "SUPPORT_RUNTIME"
    ]
    if len(matches) != 1 or type(matches[0].get("expected_contract")) is not dict:
        _fail("manifest SUPPORT_RUNTIME expected contract is absent or ambiguous")
    return matches[0]["expected_contract"]


def _authenticate_runtime_identity(
    *,
    services: FenceHandlerServices,
    request: FenceTransitionRequest,
    manifest: object,
) -> None:
    request_value = request.to_dict()
    coordinate_value = request_value["support_runtime_identity_coordinate"]
    if request.slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE:
        if coordinate_value is not None:
            _fail("PREPARE embeds a support runtime identity")
        if services.authority_class is not ExecutorAuthorityClass.RETAINED_PRE_SUPPORT:
            _fail("PREPARE requires retained pre-support authority")
        if services.live_support_runtime_identity is not None:
            _fail("retained authority cannot impersonate support live equality")
        return
    coordinate = parse_support_runtime_identity_coordinate(coordinate_value)
    identity_value = _load_exact_json(
        s3=services.s3,
        coordinate=coordinate,
        label="support runtime identity",
    )
    identity = parse_support_runtime_identity(dict(identity_value))
    if identity.canonical_identity_sha256 != coordinate.canonical_identity_sha256:
        _fail("support runtime identity coordinate drifted")
    expected = _expected_support_contract(manifest)
    if services.authority_class is ExecutorAuthorityClass.SUPPORT_RUNTIME:
        live_reader = services.live_support_runtime_identity
        if not callable(live_reader):
            _fail("SUPPORT_RUNTIME fresh live equality reader is absent")
        require_live_support_runtime_identity(
            identity,
            expected_contract=dict(expected),
            live_readback=dict(live_reader()),
        )
        return
    if services.authority_class is not ExecutorAuthorityClass.RETAINED_PRE_SUPPORT:
        _fail("executor authority class is foreign")
    if request.slot not in {
        FenceSlot.SOURCE_FAMILIES_FROZEN,
        FenceSlot.CLOSED_SOURCE,
    }:
        _fail("retained failover may execute only freeze or CLOSED_SOURCE")
    if services.live_support_runtime_identity is not None:
        _fail("retained failover cannot use support live authority")


def _require_prepared_binding(
    prepared: object,
    *,
    request: FenceTransitionRequest,
    manifest: object,
    entry: object,
) -> PreparedFenceChangeSet:
    if (
        type(prepared) is not PreparedFenceChangeSet
        or prepared.request_identity_sha256 != request.canonical_identity_sha256
        or prepared.manifest_identity_sha256 != manifest.canonical_identity_sha256
        or prepared.entry_identity_sha256 != entry.entry_identity_sha256
        or prepared.slot is not request.slot
    ):
        _fail("prepared change-set custody binding drifted")
    return prepared


def handle_request_coordinate(
    *, coordinate: ArtifactCoordinate, services: FenceHandlerServices
) -> FenceExecutionResult:
    """Authenticate one coordinate and execute only Task-3 loaded custody."""

    if type(services) is not FenceHandlerServices:
        raise TypeError("handler services must be exact FenceHandlerServices")
    if type(services.executor) is not FenceExecutor:
        raise TypeError("handler executor must be the exact v2 FenceExecutor")
    if type(services.authority_class) is not ExecutorAuthorityClass:
        raise TypeError("handler authority class must be exact")
    request = _load_request(s3=services.s3, coordinate=coordinate)
    request_value = request.to_dict()
    manifest_coordinate = parse_artifact_coordinate(
        request_value["manifest_coordinate"]
    )
    manifest = load_pinned_fence_manifest(
        s3=services.s3, coordinate=manifest_coordinate
    )
    if (
        manifest.manifest_stage.value != request_value["manifest_stage"]
        or manifest.canonical_identity_sha256
        != manifest_coordinate.canonical_identity_sha256
    ):
        _fail("request manifest identity or stage drifted")
    entry = _select_entry(manifest, request)
    entry = load_pinned_fence_entry_template(s3=services.s3, entry=entry)
    _authenticate_runtime_identity(
        services=services, request=request, manifest=manifest
    )
    retained_failover = False
    if services.authority_class is ExecutorAuthorityClass.RETAINED_PRE_SUPPORT:
        allowed_create_authority_classes = getattr(
            entry, "allowed_create_authority_classes", None
        )
        known_authority_classes = (
            ExecutorAuthorityClass.RETAINED_PRE_SUPPORT.value,
            ExecutorAuthorityClass.SUPPORT_RUNTIME.value,
        )
        if type(allowed_create_authority_classes) is not tuple or any(
            type(authority_class) is not str
            or authority_class not in known_authority_classes
            for authority_class in allowed_create_authority_classes
        ):
            _fail("selected entry create authority classes are not exact")
        retained_failover = (
            ExecutorAuthorityClass.RETAINED_PRE_SUPPORT.value
            not in allowed_create_authority_classes
        )
    if retained_failover:
        prepared = _require_prepared_binding(
            services.executor.load_prepared_change_set(request),
            request=request,
            manifest=manifest,
            entry=entry,
        )
    else:
        create_authority = services.executor.load_create_authority(request)
        if create_authority.authority_class is not services.authority_class:
            _fail("create authority class does not match this runtime")
        create_prestate = services.executor.load_prestate_snapshot(
            request, phase="CREATE"
        )
        created = services.executor.prepare_change_set(
            request=request,
            manifest=manifest,
            entry=entry,
            authority=create_authority,
            prestate=create_prestate,
        )
        prepared = services.executor.load_prepared_change_set(request)
        if (
            type(created) is not PreparedFenceChangeSet
            or type(prepared) is not PreparedFenceChangeSet
            or created.to_dict() != prepared.to_dict()
        ):
            _fail("prepared change-set custody readback drifted")
        prepared = _require_prepared_binding(
            prepared,
            request=request,
            manifest=manifest,
            entry=entry,
        )
    execute_authority = services.executor.load_execute_authority(
        request, prepared=prepared
    )
    if execute_authority.authority_class is not services.authority_class:
        _fail("execute authority class does not match this runtime")
    execute_prestate = services.executor.load_prestate_snapshot(
        request, phase="EXECUTE"
    )
    freeze_delta_audit = None
    if request.slot is FenceSlot.SOURCE_FAMILIES_FROZEN:
        loader = getattr(services.executor, "load_freeze_execute_delta_audit", None)
        if not callable(loader):
            _fail("freeze execute delta audit loader is absent")
        freeze_delta_audit = loader(request, prepared=prepared)
    return services.executor.execute_prepared_change_set(
        request=request,
        manifest=manifest,
        entry=entry,
        prepared=prepared,
        authority=execute_authority,
        prestate=execute_prestate,
        freeze_delta_audit=freeze_delta_audit,
    )


def bootstrap_publication_main(
    event: object,
    context: object,
    *,
    services: BootstrapPublicationHandlerServices | None = None,
) -> Mapping[str, object]:
    """Publish only the exact requested bootstrap phase through split clients."""

    if type(event) is not dict or type(event.get("phase")) is not str:
        _fail("bootstrap publication event is not exact")
    phase = event["phase"]
    if phase == "BRIDGE_SEED":
        if set(event) != {"phase", "request"}:
            _fail("bridge-seed publication event fields are not exact")
        parsed = parse_bridge_seed_publication_request_v2(event["request"])
        request = event["request"]
        checkpoint = None
    elif phase == "BOOTSTRAP_MANIFEST":
        if set(event) != {"phase", "request", "checkpoint"}:
            _fail("bootstrap-manifest publication event fields are not exact")
        checkpoint = parse_stack_migration_transfer_checkpoint_v2(event["checkpoint"])
        parsed = parse_bootstrap_publication_request_v2(
            event["request"],
            checkpoint=checkpoint,
        )
        request = event["request"]
    else:
        _fail("bootstrap publication phase event is not implemented")
    invoked_function_arn = getattr(context, "invoked_function_arn", None)
    if (
        type(invoked_function_arn) is not str
        or invoked_function_arn != parsed.materializer_function_version_arn
    ):
        _fail("bootstrap materializer function version identity drifted")
    if services is None:  # pragma: no cover - production Lambda boundary
        services = _build_bootstrap_publication_handler_services()
    if type(services) is not BootstrapPublicationHandlerServices:
        raise TypeError(
            "bootstrap handler services must be exact split-client services"
        )
    if phase == "BRIDGE_SEED":
        return publish_bridge_seed_policy_v2(
            request=request,
            services=services.fixed_artifacts,
        ).to_dict()
    if checkpoint is None:
        raise AssertionError("bootstrap checkpoint parser did not run")
    return publish_bootstrap_fence_artifacts_v2(
        request=request,
        checkpoint=checkpoint,
        services=services.fixed_artifacts,
    ).to_dict()


def source_settlement_main(
    event: object,
    context: object,
    *,
    services: SourceSettlementHandlerServices | None = None,
) -> Mapping[str, object]:
    """Reload immutable-coordinate evidence and settle source publication."""

    from .fence_source_settlement import (
        SourceSettlementResult,
        SourceSettlementServices,
        parse_source_settlement_request_v2,
        settle_source_publication,
    )

    if type(event) is not dict or set(event) != {"request"}:
        _fail("source settlement event must contain only request authority")
    request = parse_source_settlement_request_v2(event["request"])
    invoked_function_arn = getattr(context, "invoked_function_arn", None)
    if (
        type(invoked_function_arn) is not str
        or invoked_function_arn != request.materializer_function_version_arn
    ):
        _fail("source settlement function version identity drifted")
    if services is None:  # pragma: no cover - production Lambda boundary
        services = _build_source_settlement_handler_services(request)
    if type(services) is not SourceSettlementHandlerServices:
        raise TypeError("source settlement handler services are not exact")
    evidence = services.evidence_reader_factory(request)
    result = settle_source_publication(
        request=request,
        services=SourceSettlementServices(
            evidence=evidence,
            s3=services.s3,
            artifact_services=services.artifact_services,
            now_utc=services.now_utc,
            sleep=services.sleep,
        ),
    )
    if type(result) is not SourceSettlementResult:
        raise TypeError("source settlement returned a foreign result")
    unsigned = {
        "schema_version": 2,
        "record_type": "glm52_h1g_source_settlement_receipt_v2",
        "executed_function_version_arn": invoked_function_arn,
        "disposition": result.disposition.value,
        "classification": result.classification,
        "selected_entry": result.selected_entry.to_dict(),
        "source_settlement": (
            result.source_settlement.to_dict()
            if result.source_settlement is not None
            else None
        ),
        "source_settled_manifest": (
            result.source_settled_manifest.to_dict()
            if result.source_settled_manifest is not None
            else None
        ),
        "publication_coordinates": [
            coordinate.to_dict() for coordinate in result.publication_coordinates
        ],
        "no_launch_evidence": (
            dict(result.no_launch_evidence)
            if result.no_launch_evidence is not None
            else None
        ),
    }
    return {
        **unsigned,
        "canonical_identity_sha256": canonical_sha256(unsigned),
    }


def main(
    event: object,
    context: object,
    *,
    services: FenceHandlerServices | None = None,
) -> Mapping[str, object]:
    """Accept only ``{request_coordinate}``; all execution facts are reloaded."""

    if type(event) is not dict or set(event) != {"request_coordinate"}:
        _fail("handler event must contain only request_coordinate")
    coordinate = parse_artifact_coordinate(event["request_coordinate"])
    if services is None:  # pragma: no cover - production Lambda boundary
        from .task11_production import build_fence_handler_services

        remaining = getattr(context, "get_remaining_time_in_millis", None)
        if not callable(remaining):
            _fail("Lambda remaining-time boundary is absent")
        services = build_fence_handler_services(
            generation=1,
            remaining_time_millis=remaining(),
        )
    result = handle_request_coordinate(coordinate=coordinate, services=services)
    if type(result) is not FenceExecutionResult:
        _fail("executor returned a non-v2 result")
    return result.to_dict()


__all__ = [
    "BootstrapPublicationHandlerServices",
    "FenceHandlerServices",
    "SourceSettlementHandlerServices",
    "bootstrap_publication_main",
    "handle_request_coordinate",
    "main",
    "source_settlement_main",
]
