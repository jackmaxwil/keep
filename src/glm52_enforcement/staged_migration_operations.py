"""Strict production adapters for retained-to-fence ownership migration."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import parse_qs, urlparse

from .canonical import canonical_json_bytes
from .cloudformation_stacks import (
    ACCOUNT_ID,
    REGION,
    STACK_TAGS,
    BridgeSeedEstablishedV2,
    BridgeSeedOwnershipPlanV2,
    MigrationBootstrapResult,
    StackIdentity,
    StackKind,
    StackMigrationAuthorityV2,
    StackMigrationEvidenceV2,
    StackMigrationTransferBundleV2,
    StackMigrationTransferCheckpointV2,
    TemplateCoordinate,
    build_bridge_seed_ownership_plan_v2,
    build_stack_migration_v2,
    migration_bootstrap_result_from_state,
    parse_bridge_seed_established_v2,
    parse_migration_role_evidence_v2,
    stack_name,
)
from .cloudformation_stacks import (
    canonical_json_bytes as stack_canonical_json_bytes,
)
from .fence_artifacts import BridgeSeedArtifact, parse_bridge_seed_artifact
from .task13_migration_adapter import (
    SeededMigrationRuntime,
    bootstrap_seeded_migration,
    build_existing_seeded_migration_runtime,
    build_seeded_migration_runtime,
    validate_stack_migration_seed_projection,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ACTIVATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_COORDINATE_STAGES = (
    "bridge-seed-owner",
    "retention-only",
    "post-retain",
    "fence-import",
    "fence-transfer",
)
_BRIDGE_REQUEST_RECORD_TYPE = "glm52_h1g_bridge_seed_establishment_request_v2"
_TRANSFER_REQUEST_RECORD_TYPE = "glm52_h1g_stack_ownership_transfer_request_v2"


class StagedMigrationOperationError(ValueError):
    """A raw request or authenticated migration readback failed closed."""


@dataclass(frozen=True)
class MigrationEvidenceInputV2:
    retained_stack_id: str
    bucket_name: str
    current_policy_logical_id: str | None
    import_identifier: tuple[tuple[str, str], ...]
    preseed_policy: Mapping[str, object]
    preseed_template_body_sha256: str
    retained_resource_physical_ids: tuple[tuple[str, str], ...]
    retained_export_names: tuple[str, ...]
    support_export_names: tuple[str, ...]


@dataclass(frozen=True)
class StagedMigrationRequestV2:
    record_type: str
    activation_id: str
    activation_identity_sha256: str
    migration_seed: Mapping[str, object]
    authority: StackMigrationAuthorityV2
    evidence: MigrationEvidenceInputV2
    template_coordinates: tuple[Mapping[str, str], ...]


@dataclass(frozen=True)
class BridgeSeedEstablishmentRequestV2(StagedMigrationRequestV2):
    bridge_seed: Mapping[str, object]


@dataclass(frozen=True)
class _LiveBridgePlanResult:
    plan: BridgeSeedOwnershipPlanV2
    direct_preseed_policy_readbacks: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if (
            type(self.plan) is not BridgeSeedOwnershipPlanV2
            or type(self.direct_preseed_policy_readbacks) is not tuple
            or len(self.direct_preseed_policy_readbacks) != 2
            or any(
                type(readback) is not dict
                for readback in self.direct_preseed_policy_readbacks
            )
        ):
            raise TypeError("live bridge plan result is not exact")


class _MigrationClient:
    """Small explicit CFN/S3 composite used by the reviewed coordinator."""

    def __init__(self, *, cloudformation: object, s3: object) -> None:
        self._cloudformation = cloudformation
        self._s3 = s3

    def __getattr__(self, name: str) -> object:
        target = (
            self._s3
            if name in {"get_bucket_policy", "get_object"}
            else self._cloudformation
        )
        method = getattr(target, name, None)
        if not callable(method):
            raise AttributeError(name)
        return method


def _exact_mapping(value: object, fields: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise StagedMigrationOperationError(label + " fields are not exact")
    return dict(value)


def _detached_json_mapping(
    value: object,
    label: str,
) -> dict[str, object]:
    try:
        detached = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise StagedMigrationOperationError(
            label + " is not canonical JSON data"
        ) from exc
    if type(detached) is not dict:
        raise StagedMigrationOperationError(label + " is not one canonical JSON object")
    return detached


def _pairs(value: object, label: str) -> tuple[tuple[str, str], ...]:
    if type(value) is not list:
        raise StagedMigrationOperationError(label + " must be one list")
    rows: list[tuple[str, str]] = []
    for row in value:
        if (
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not str
            or not row[0]
            or type(row[1]) is not str
            or not row[1]
        ):
            raise StagedMigrationOperationError(label + " row is not exact")
        rows.append((row[0], row[1]))
    if len(set(rows)) != len(rows):
        raise StagedMigrationOperationError(label + " contains duplicates")
    return tuple(rows)


def _names(value: object, label: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise StagedMigrationOperationError(label + " is not exact")
    return tuple(value)


def _parse_evidence_input(value: object) -> MigrationEvidenceInputV2:
    item = _exact_mapping(
        value,
        {
            "retained_stack_id",
            "bucket_name",
            "current_policy_logical_id",
            "import_identifier",
            "preseed_policy",
            "preseed_template_body_sha256",
            "retained_resource_physical_ids",
            "retained_export_names",
            "support_export_names",
        },
        "migration evidence input",
    )
    logical_id = item["current_policy_logical_id"]
    if logical_id is not None and (type(logical_id) is not str or not logical_id):
        raise StagedMigrationOperationError("current policy logical ID is not exact")
    if type(item["preseed_policy"]) is not dict or not item["preseed_policy"]:
        raise StagedMigrationOperationError("preseed policy is not one object")
    preseed_policy = _detached_json_mapping(
        item["preseed_policy"],
        "preseed policy",
    )
    preseed_template_body_sha256 = item["preseed_template_body_sha256"]
    if (
        type(preseed_template_body_sha256) is not str
        or _SHA256.fullmatch(preseed_template_body_sha256) is None
    ):
        raise StagedMigrationOperationError("preseed template identity is not exact")
    return MigrationEvidenceInputV2(
        retained_stack_id=item["retained_stack_id"],
        bucket_name=item["bucket_name"],
        current_policy_logical_id=logical_id,
        import_identifier=_pairs(item["import_identifier"], "import identifier"),
        preseed_policy=preseed_policy,
        preseed_template_body_sha256=preseed_template_body_sha256,
        retained_resource_physical_ids=_pairs(
            item["retained_resource_physical_ids"], "retained resources"
        ),
        retained_export_names=_names(item["retained_export_names"], "retained exports"),
        support_export_names=_names(item["support_export_names"], "support exports"),
    )


def _parse_coordinates(value: object) -> tuple[Mapping[str, str], ...]:
    if type(value) is not list or len(value) != len(_COORDINATE_STAGES):
        raise StagedMigrationOperationError(
            "migration template coordinates are not exact"
        )
    rows: list[Mapping[str, str]] = []
    for expected_stage, raw in zip(_COORDINATE_STAGES, value, strict=True):
        item = _exact_mapping(
            raw, {"stage", "template_url", "version_id"}, "template coordinate"
        )
        if item["stage"] != expected_stage:
            raise StagedMigrationOperationError("template coordinate order drifted")
        TemplateCoordinate(
            stage=expected_stage,
            stack_id=(
                "arn:aws:cloudformation:us-west-2:246813579024:stack/"
                "keep-glm52-gpu/00000000-0000-0000-0000-000000000000"
            ),
            template_url=item["template_url"],
            version_id=item["version_id"],
        )
        rows.append(
            {key: str(item[key]) for key in ("stage", "template_url", "version_id")}
        )
    return tuple(rows)


def _parse_request(
    value: object,
    *,
    record_type: str,
) -> StagedMigrationRequestV2:
    fields = {
        "schema_version",
        "record_type",
        "activation_id",
        "activation_identity_sha256",
        "migration_seed",
        "api_caller",
        "migration_service_role",
        "fence_service_role",
        "evidence",
        "template_coordinates",
    }
    if record_type == _BRIDGE_REQUEST_RECORD_TYPE:
        fields.add("bridge_seed")
    item = _exact_mapping(value, fields, "staged migration request")
    if (
        item["schema_version"] != 2
        or item["record_type"] != record_type
        or type(item["activation_id"]) is not str
        or _ACTIVATION_ID.fullmatch(item["activation_id"]) is None
        or type(item["activation_identity_sha256"]) is not str
        or _SHA256.fullmatch(item["activation_identity_sha256"]) is None
        or type(item["migration_seed"]) is not dict
    ):
        raise StagedMigrationOperationError("staged migration identity is not exact")
    migration_seed = _detached_json_mapping(
        item["migration_seed"],
        "migration seed",
    )
    seed = validate_stack_migration_seed_projection(migration_seed)
    if seed.authority.action_identity_sha256 != item["activation_identity_sha256"]:
        raise StagedMigrationOperationError("activation and migration identity drifted")
    api_caller = parse_migration_role_evidence_v2(item["api_caller"])
    migration_role = parse_migration_role_evidence_v2(item["migration_service_role"])
    fence_role = parse_migration_role_evidence_v2(item["fence_service_role"])
    authority = StackMigrationAuthorityV2(
        api_caller=api_caller,
        migration_service_role=migration_role,
        fence_service_role=fence_role,
        fence_bootstrap=seed.authority.fence_bootstrap,
        support_bootstrap=seed.authority.support_bootstrap,
        import_change_set_name=seed.authority.import_change_set_name,
        action_identity_sha256=seed.authority.action_identity_sha256,
    )
    if (
        seed.authority.deployment_role_arn != migration_role.role_arn
        or seed.authority.deployment_role_id != migration_role.role_id
    ):
        raise StagedMigrationOperationError("seed migration service role drifted")
    common = {
        "record_type": record_type,
        "activation_id": item["activation_id"],
        "activation_identity_sha256": item["activation_identity_sha256"],
        "migration_seed": migration_seed,
        "authority": authority,
        "evidence": _parse_evidence_input(item["evidence"]),
        "template_coordinates": _parse_coordinates(item["template_coordinates"]),
    }
    if record_type == _BRIDGE_REQUEST_RECORD_TYPE:
        bridge_seed = _detached_json_mapping(
            item["bridge_seed"],
            "bridge seed",
        )
        parse_bridge_seed_artifact(bridge_seed)
        return BridgeSeedEstablishmentRequestV2(
            **common,
            bridge_seed=bridge_seed,
        )
    return StagedMigrationRequestV2(**common)


def parse_bridge_seed_establishment_request_v2(
    value: object,
) -> BridgeSeedEstablishmentRequestV2:
    parsed = _parse_request(value, record_type=_BRIDGE_REQUEST_RECORD_TYPE)
    if type(parsed) is not BridgeSeedEstablishmentRequestV2:
        raise StagedMigrationOperationError(
            "bridge seed establishment request is not exact"
        )
    return parsed


def parse_stack_ownership_transfer_request_v2(
    value: object,
) -> StagedMigrationRequestV2:
    return _parse_request(value, record_type=_TRANSFER_REQUEST_RECORD_TYPE)


def materialize_bridge_seed_establishment_request_v2(
    *,
    prototype: object,
    bridge_seed: object,
) -> dict[str, object]:
    if type(prototype) is not dict or "bridge_seed" in prototype:
        raise StagedMigrationOperationError(
            "bridge seed request prototype must omit bridge seed"
        )
    detached = _detached_json_mapping(
        prototype,
        "bridge seed request prototype",
    )
    artifact = (
        bridge_seed
        if type(bridge_seed) is BridgeSeedArtifact
        else parse_bridge_seed_artifact(bridge_seed)
    )
    materialized = {**detached, "bridge_seed": artifact.to_dict()}
    parse_bridge_seed_establishment_request_v2(materialized)
    return materialized


def _services(value: object) -> object:
    from .task13_production_operations import ProductionServices

    if type(value) is not ProductionServices or value.total_max_attempts != 1:
        raise StagedMigrationOperationError(
            "production services are not exact one-attempt clients"
        )
    return value


def _runtime(
    *,
    parsed: StagedMigrationRequestV2,
    services: object,
    existing: bool,
) -> SeededMigrationRuntime:
    client = _MigrationClient(
        cloudformation=services.cloudformation,
        s3=services.s3,
    )
    builder = (
        build_existing_seeded_migration_runtime
        if existing
        else build_seeded_migration_runtime
    )
    return builder(
        cloudformation=client,
        dynamodb=services.dynamodb,
        value=parsed.migration_seed,
    )


def _success(value: object, operation: str) -> Mapping[str, object]:
    if type(value) is not dict or type(value.get("ResponseMetadata")) is not dict:
        raise StagedMigrationOperationError(operation + " response is not exact")
    metadata = value["ResponseMetadata"]
    if (
        metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
    ):
        raise StagedMigrationOperationError(operation + " response is unauthenticated")
    return value


def _read_body(value: object, operation: str) -> bytes:
    body = value.get("Body") if type(value) is dict else None
    raw = body.read() if callable(getattr(body, "read", None)) else body
    if type(raw) is not bytes:
        raise StagedMigrationOperationError(operation + " body is not bytes")
    return raw


def _load_bridge_seed(
    parsed: BridgeSeedEstablishmentRequestV2,
    services: object,
) -> BridgeSeedArtifact:
    coordinate = parsed.bridge_seed
    response = _success(
        services.s3.get_object(
            Bucket=coordinate["bucket"],
            Key=coordinate["key"],
            VersionId=coordinate["version_id"],
            ExpectedBucketOwner=ACCOUNT_ID,
        ),
        "GetObject(BridgeSeed)",
    )
    return parse_bridge_seed_artifact(
        coordinate, raw_bytes=_read_body(response, "GetObject(BridgeSeed)")
    )


def _direct_policy(services: object, *, bucket_name: str) -> Mapping[str, object]:
    response = _success(
        services.s3.get_bucket_policy(
            Bucket=bucket_name, ExpectedBucketOwner=ACCOUNT_ID
        ),
        "GetBucketPolicy",
    )
    raw = response.get("Policy")
    if type(raw) is not str:
        raise StagedMigrationOperationError("GetBucketPolicy omitted canonical policy")
    try:
        policy = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StagedMigrationOperationError(
            "GetBucketPolicy returned invalid JSON"
        ) from exc
    if type(policy) is not dict:
        raise StagedMigrationOperationError("GetBucketPolicy policy is not one object")
    return policy


def _template(services: object, *, stack_id: str, stage: str) -> Mapping[str, object]:
    response = _success(
        services.cloudformation.get_template(StackName=stack_id, TemplateStage=stage),
        "GetTemplate(" + stage + ")",
    )
    body = response.get("TemplateBody")
    if type(body) is str:
        try:
            body = json.loads(body)
        except json.JSONDecodeError as exc:
            raise StagedMigrationOperationError(
                "template readback is invalid JSON"
            ) from exc
    if type(body) is not dict:
        raise StagedMigrationOperationError("template readback is not one object")
    return body


def _coordinates(
    parsed: StagedMigrationRequestV2, bootstrap: MigrationBootstrapResult
) -> tuple[TemplateCoordinate, ...]:
    stack_ids = {
        "bridge-seed-owner": parsed.evidence.retained_stack_id,
        "retention-only": parsed.evidence.retained_stack_id,
        "post-retain": parsed.evidence.retained_stack_id,
        "fence-import": bootstrap.fence.stack_id,
        "fence-transfer": bootstrap.fence.stack_id,
    }
    return tuple(
        TemplateCoordinate(
            stage=row["stage"],
            stack_id=stack_ids[row["stage"]],
            template_url=row["template_url"],
            version_id=row["version_id"],
        )
        for row in parsed.template_coordinates
    )


def _evidence(
    parsed: StagedMigrationRequestV2,
    bootstrap: MigrationBootstrapResult,
    *,
    direct_preseed_policy_readbacks: tuple[Mapping[str, object], ...],
) -> StackMigrationEvidenceV2:
    expected = parsed.evidence
    expected_policy_bytes = canonical_json_bytes(expected.preseed_policy)
    if (
        type(direct_preseed_policy_readbacks) is not tuple
        or len(direct_preseed_policy_readbacks) != 2
        or any(
            type(readback) is not dict for readback in direct_preseed_policy_readbacks
        )
        or tuple(
            canonical_json_bytes(readback)
            for readback in direct_preseed_policy_readbacks
        )
        != (expected_policy_bytes, expected_policy_bytes)
    ):
        raise StagedMigrationOperationError(
            "direct preseed policy readbacks differ from request projection"
        )
    retained = StackIdentity(
        kind=StackKind.RETAINED,
        name=stack_name(StackKind.RETAINED),
        stack_id=expected.retained_stack_id,
        account_id=ACCOUNT_ID,
        region=REGION,
        termination_protection=True,
        tags=STACK_TAGS,
    )
    return StackMigrationEvidenceV2(
        retained=retained,
        fence=bootstrap.fence,
        support=bootstrap.support,
        current_policy_logical_id=expected.current_policy_logical_id,
        current_policy_physical_id=expected.bucket_name,
        current_policy_stack_id=retained.stack_id,
        bucket_name=expected.bucket_name,
        import_identifier=expected.import_identifier,
        preseed_policy=dict(expected.preseed_policy),
        direct_preseed_policy_readbacks=tuple(
            dict(readback) for readback in direct_preseed_policy_readbacks
        ),
        retained_resource_physical_ids=expected.retained_resource_physical_ids,
        retained_export_names=expected.retained_export_names,
        support_export_names=expected.support_export_names,
    )


def _projected_evidence(
    parsed: StagedMigrationRequestV2,
    bootstrap: MigrationBootstrapResult,
) -> StackMigrationEvidenceV2:
    return _evidence(
        parsed,
        bootstrap,
        direct_preseed_policy_readbacks=(
            dict(parsed.evidence.preseed_policy),
            dict(parsed.evidence.preseed_policy),
        ),
    )


def _bridge_plan_from_live(
    parsed: StagedMigrationRequestV2,
    services: object,
    bridge_seed: BridgeSeedArtifact,
) -> _LiveBridgePlanResult:
    original = _template(
        services,
        stack_id=parsed.evidence.retained_stack_id,
        stage="Original",
    )
    processed = _template(
        services,
        stack_id=parsed.evidence.retained_stack_id,
        stage="Processed",
    )
    if stack_canonical_json_bytes(original) != stack_canonical_json_bytes(processed):
        raise StagedMigrationOperationError(
            "retained preseed original/processed templates differ"
        )
    if (
        hashlib.sha256(canonical_json_bytes(processed)).hexdigest()
        != parsed.evidence.preseed_template_body_sha256
    ):
        raise StagedMigrationOperationError(
            "retained preseed template identity drifted"
        )
    direct_readbacks = tuple(
        _direct_policy(services, bucket_name=parsed.evidence.bucket_name)
        for _readback in range(2)
    )
    expected_policy_bytes = canonical_json_bytes(parsed.evidence.preseed_policy)
    if tuple(canonical_json_bytes(readback) for readback in direct_readbacks) != (
        expected_policy_bytes,
        expected_policy_bytes,
    ):
        raise StagedMigrationOperationError(
            "live preseed policy differs from request projection"
        )
    plan = build_bridge_seed_ownership_plan_v2(
        retained_template=processed,
        bucket_name=parsed.evidence.bucket_name,
        bridge_seed=bridge_seed,
        current_policy_logical_id=parsed.evidence.current_policy_logical_id,
    )
    return _LiveBridgePlanResult(
        plan=plan,
        direct_preseed_policy_readbacks=direct_readbacks,
    )


def _load_bridge_template(parsed: StagedMigrationRequestV2, services: object) -> bytes:
    row = parsed.template_coordinates[0]
    parsed_url = urlparse(row["template_url"])
    query = parse_qs(parsed_url.query, keep_blank_values=True)
    if query != {"versionId": [row["version_id"]]}:
        raise StagedMigrationOperationError("bridge template URL version drifted")
    bucket = parsed_url.netloc.removesuffix(f".s3.{REGION}.amazonaws.com")
    key = parsed_url.path.removeprefix("/")
    response = _success(
        services.s3.get_object(
            Bucket=bucket,
            Key=key,
            VersionId=row["version_id"],
            ExpectedBucketOwner=ACCOUNT_ID,
        ),
        "GetObject(BridgeTemplate)",
    )
    raw = _read_body(response, "GetObject(BridgeTemplate)")
    try:
        template = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StagedMigrationOperationError(
            "bridge template is not canonical JSON"
        ) from exc
    if type(template) is not dict or raw != canonical_json_bytes(template) + b"\n":
        raise StagedMigrationOperationError("bridge template bytes are not canonical")
    return raw[:-1]


def _bridge_plan_from_projection(
    parsed: StagedMigrationRequestV2,
    services: object,
    *,
    seed_policy_sha256: str,
    established: BridgeSeedEstablishedV2 | None,
) -> BridgeSeedOwnershipPlanV2:
    body = _load_bridge_template(parsed, services)
    owner_branch = (
        "ADD" if parsed.evidence.current_policy_logical_id is None else "MODIFY"
    )
    owner_logical_id = (
        parsed.evidence.current_policy_logical_id or "H1gProductionFenceBucketPolicy"
    )
    fields = {
        "record_type": "glm52_h1g_bridge_seed_ownership_plan_v2",
        "owner_branch": owner_branch,
        "owner_logical_id": owner_logical_id,
        "preseed_template_body_sha256": (parsed.evidence.preseed_template_body_sha256),
        "expected_live_preseed_policy_sha256": hashlib.sha256(
            canonical_json_bytes(parsed.evidence.preseed_policy)
        ).hexdigest(),
        "template_body_sha256": hashlib.sha256(body).hexdigest(),
        "seed_policy_sha256": seed_policy_sha256,
    }
    plan = BridgeSeedOwnershipPlanV2(
        owner_branch=owner_branch,
        owner_logical_id=owner_logical_id,
        template_body=body,
        preseed_template_body_sha256=(parsed.evidence.preseed_template_body_sha256),
        expected_live_preseed_policy_sha256=fields[
            "expected_live_preseed_policy_sha256"
        ],
        template_body_sha256=fields["template_body_sha256"],
        seed_policy_sha256=seed_policy_sha256,
        canonical_identity_sha256=hashlib.sha256(
            canonical_json_bytes(fields)
        ).hexdigest(),
    )
    if established is not None and (
        established.owner_branch != plan.owner_branch
        or established.owner_logical_id != plan.owner_logical_id
        or established.preseed_template_body_sha256 != plan.preseed_template_body_sha256
        or established.bridge_template_body_sha256 != plan.template_body_sha256
        or established.seed_policy_sha256 != plan.seed_policy_sha256
    ):
        raise StagedMigrationOperationError("bridge result and immutable plan drifted")
    return plan


def _bundle(
    parsed: StagedMigrationRequestV2,
    services: object,
    bootstrap: MigrationBootstrapResult,
    evidence: StackMigrationEvidenceV2,
    plan: BridgeSeedOwnershipPlanV2,
) -> StackMigrationTransferBundleV2:
    bootstrap_template = _template(
        services, stack_id=bootstrap.fence.stack_id, stage="Original"
    )
    return build_stack_migration_v2(
        bootstrap_template=bootstrap_template,
        bootstrap_result=bootstrap,
        evidence=evidence,
        authority=parsed.authority,
        bridge_seed_plan=plan,
        template_coordinates=_coordinates(parsed, bootstrap),
    )


def _bootstrap_from_existing(
    runtime: SeededMigrationRuntime,
) -> MigrationBootstrapResult:
    state = runtime.coordinator._load_state(runtime.seed.authority)  # noqa: SLF001
    return migration_bootstrap_result_from_state(state)


def establish_bridge_seed_v2(
    *, request: object, services: object
) -> BridgeSeedEstablishedV2:
    parsed = parse_bridge_seed_establishment_request_v2(request)
    service_set = _services(services)
    runtime = _runtime(
        parsed=parsed,
        services=service_set,
        existing=False,
    )
    bootstrap = bootstrap_seeded_migration(runtime)
    seed = _load_bridge_seed(parsed, service_set)
    live = _bridge_plan_from_live(parsed, service_set, seed)
    evidence = _evidence(
        parsed,
        bootstrap,
        direct_preseed_policy_readbacks=live.direct_preseed_policy_readbacks,
    )
    bundle = _bundle(parsed, service_set, bootstrap, evidence, live.plan)
    return runtime.coordinator.establish_bridge_seed_v2(
        bundle=bundle,
        evidence=evidence,
        authority=parsed.authority,
        bootstrap_result=bootstrap,
    )


def read_bridge_seed_v2(
    *, request: object, services: object
) -> BridgeSeedEstablishedV2:
    parsed = parse_bridge_seed_establishment_request_v2(request)
    service_set = _services(services)
    runtime = _runtime(parsed=parsed, services=service_set, existing=True)
    bootstrap = _bootstrap_from_existing(runtime)
    evidence = _projected_evidence(parsed, bootstrap)
    seed = _load_bridge_seed(parsed, service_set)
    plan = _bridge_plan_from_projection(
        parsed,
        service_set,
        seed_policy_sha256=seed.policy_sha256,
        established=None,
    )
    bundle = _bundle(parsed, service_set, bootstrap, evidence, plan)
    return runtime.coordinator.read_bridge_seed_effect_v2(
        bundle=bundle,
        evidence=evidence,
        authority=parsed.authority,
        bootstrap_result=bootstrap,
    )


def complete_operations_1_to_6_v2(
    *, request: object, bridge_seed: object, services: object
) -> StackMigrationTransferCheckpointV2:
    parsed = parse_stack_ownership_transfer_request_v2(request)
    established = parse_bridge_seed_established_v2(
        bridge_seed.to_dict()
        if type(bridge_seed) is BridgeSeedEstablishedV2
        else bridge_seed
    )
    service_set = _services(services)
    runtime = _runtime(parsed=parsed, services=service_set, existing=True)
    bootstrap = _bootstrap_from_existing(runtime)
    evidence = _projected_evidence(parsed, bootstrap)
    plan = _bridge_plan_from_projection(
        parsed,
        service_set,
        seed_policy_sha256=established.seed_policy_sha256,
        established=established,
    )
    bundle = _bundle(parsed, service_set, bootstrap, evidence, plan)
    return runtime.coordinator.execute_operations_1_to_6_v2(
        bundle=bundle,
        evidence=evidence,
        authority=parsed.authority,
        bootstrap_result=bootstrap,
        bridge_seed=established,
    )


def read_operations_1_to_6_v2(
    *, request: object, bridge_seed: object, services: object
) -> StackMigrationTransferCheckpointV2:
    parsed = parse_stack_ownership_transfer_request_v2(request)
    established = parse_bridge_seed_established_v2(
        bridge_seed.to_dict()
        if type(bridge_seed) is BridgeSeedEstablishedV2
        else bridge_seed
    )
    service_set = _services(services)
    runtime = _runtime(parsed=parsed, services=service_set, existing=True)
    bootstrap = _bootstrap_from_existing(runtime)
    evidence = _projected_evidence(parsed, bootstrap)
    plan = _bridge_plan_from_projection(
        parsed,
        service_set,
        seed_policy_sha256=established.seed_policy_sha256,
        established=established,
    )
    bundle = _bundle(parsed, service_set, bootstrap, evidence, plan)
    return runtime.coordinator.read_operations_1_to_6_effect_v2(
        bundle=bundle,
        evidence=evidence,
        authority=parsed.authority,
        bootstrap_result=bootstrap,
        bridge_seed=established,
    )


__all__ = [
    "BridgeSeedEstablishmentRequestV2",
    "MigrationEvidenceInputV2",
    "StagedMigrationOperationError",
    "StagedMigrationRequestV2",
    "complete_operations_1_to_6_v2",
    "establish_bridge_seed_v2",
    "materialize_bridge_seed_establishment_request_v2",
    "parse_bridge_seed_establishment_request_v2",
    "parse_stack_ownership_transfer_request_v2",
    "read_bridge_seed_v2",
    "read_operations_1_to_6_v2",
]
