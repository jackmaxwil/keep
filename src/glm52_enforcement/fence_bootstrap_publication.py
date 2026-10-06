"""Strict v2 bootstrap fence-artifact derivation and publication.

The raw schema admits only authenticated renderer inputs and manifest authority
facts. Rendered policies, CloudFormation templates, entries, manifests, and all
of their identities are derived inside this boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .canonical import canonical_json_bytes, canonical_sha256
from .cloudformation_stacks import StackMigrationTransferCheckpointV2
from .fence_artifacts import (
    BOOTSTRAP_PUBLICATION_ORDER,
    ArtifactCoordinate,
    BridgeSeedArtifact,
    FenceArtifactEntry,
    FenceManifest,
    FenceSlot,
    ManifestStage,
    build_bridge_seed_artifact,
    build_fence_entry,
    build_fence_manifest,
    build_fence_template_bytes,
    parse_bridge_seed_artifact,
)
from .fence_policy_renderer import (
    BuildMode,
    FencePolicyInput,
    PolicyHead,
    PolicyLimits,
    PrincipalIdentity,
    ReservedFamily,
    WriterCohort,
    render_fence_policy,
)
from .task13_fixed_artifacts import (
    ACCOUNT_ID,
    CAMPAIGN_BUCKET,
    REGION,
    RUN_ID,
    FixedKeyPublicationDisposition,
    Task13FixedArtifactServices,
    adopt_fixed_key_bytes,
    publish_fixed_key_bytes_with_outcome,
)
from .task13_staged_deployment import BootstrapFencePublication

_ACTIVATION_ID = "glm52-v2-amber-quartz"
_GENERATION = 1
_STACK_NAME = "keep-glm52-h1g-fence"
_LOGICAL_ID = "H1gProductionFenceBucketPolicy"
_S3_POLICY_LIMIT = 20_480
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LAMBDA_VERSION_ARN = re.compile(
    r"arn:aws:lambda:us-west-2:246813579024:function:"
    r"[A-Za-z0-9_-]+:[1-9][0-9]*\Z"
)

BRIDGE_SEED_PUBLICATION_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "activation_id",
        "generation",
        "materializer_function_version_arn",
        "expected_live_preseed_policy_sha256",
        "renderer_input",
    }
)
BRIDGE_SEED_PUBLICATION_AUTHORITY_FIELDS = (
    BRIDGE_SEED_PUBLICATION_REQUEST_FIELDS - {"materializer_function_version_arn"}
)

BOOTSTRAP_PUBLICATION_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "activation_id",
        "generation",
        "materializer_function_version_arn",
        "checkpoint_identity_sha256",
        "renderer_inputs",
        "manifest_authority",
    }
)
BOOTSTRAP_PUBLICATION_AUTHORITY_FIELDS = (
    BOOTSTRAP_PUBLICATION_REQUEST_FIELDS
    - {"materializer_function_version_arn", "checkpoint_identity_sha256"}
)
FENCE_POLICY_INPUT_FIELDS = frozenset(
    {
        "build_mode",
        "policy_head",
        "bucket_name",
        "account_id",
        "kms_key_arn",
        "predecessor_policy_sha256",
        "freeze_denial_evidence_sha256",
        "source_publication_sealed_sha256",
        "all_version_inventory_sha256",
        "source_settlement_sha256",
        "publisher_deny_policy_sha256",
        "terminal_prerequisite_sha256",
        "legacy_statements",
        "permanent_enrolled_resources",
        "retired_publishers",
        "retired_publisher_resources",
        "reserved_families",
        "writer_cohorts",
        "writer_owned_resources",
        "source_validation_readers",
        "source_inventory_readers",
        "terminal_audit_readers",
        "selected_source_keys",
        "nonselected_source_keys",
        "policy_limits",
    }
)
MANIFEST_AUTHORITY_FIELDS = frozenset(
    {
        "mutation_authority_inventory",
        "member_account_authority",
        "bucket_control_plane",
        "kms_key_identity",
        "executor_inventory",
        "principal_inventory",
        "writer_inventory",
        "writer_policy_cohorts",
        "source_inventory",
    }
)

_PRINCIPAL_FIELDS = frozenset({"binding_id", "arn", "role_id"})
_RESERVED_FAMILY_FIELDS = frozenset({"family_id", "resources"})
_WRITER_COHORT_FIELDS = frozenset(
    {
        "cohort_id",
        "members",
        "guard_resources",
        "cross_member_denial_evidence_sha256",
    }
)
_POLICY_LIMIT_FIELDS = frozenset(
    {"max_policy_bytes", "unallocated_headroom_bytes", "component_max_bytes"}
)
_HEAD_ORDER = (
    PolicyHead.BRIDGE_SEED,
    PolicyHead.PREPARE,
    PolicyHead.RESERVATION,
    PolicyHead.SOURCE_FAMILIES_FROZEN,
    PolicyHead.CLOSED_SOURCE,
)
_HEAD_TO_SLOT = {
    PolicyHead.PREPARE: FenceSlot.PREPARE_GENESIS_LIVE_STATE,
    PolicyHead.RESERVATION: FenceSlot.RESERVATION_ONLY,
    PolicyHead.SOURCE_FAMILIES_FROZEN: FenceSlot.SOURCE_FAMILIES_FROZEN,
    PolicyHead.CLOSED_SOURCE: FenceSlot.CLOSED_SOURCE,
}
_SLOT_PUBLICATION_ORDER = (
    FenceSlot.PREPARE_GENESIS_LIVE_STATE,
    FenceSlot.RESERVATION_ONLY,
    FenceSlot.SOURCE_FAMILIES_FROZEN,
    FenceSlot.CLOSED_SOURCE,
)
_SLOT_MANIFEST_ORDER = (
    FenceSlot.PREPARE_GENESIS_LIVE_STATE,
    FenceSlot.RESERVATION_ONLY,
    FenceSlot.CLOSED_SOURCE,
    FenceSlot.SOURCE_FAMILIES_FROZEN,
)


class FenceBootstrapPublicationError(ValueError):
    """Bootstrap publication input or durable truth failed closed."""


def _fail(message: str) -> None:
    raise FenceBootstrapPublicationError(message)


def _exact_mapping(
    value: object, fields: frozenset[str], label: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        _fail(label + " schema fields are not exact")
    return value


def _array(value: object, label: str) -> list[object]:
    if type(value) is not list:
        _fail(label + " must be one exact array")
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    items = _array(value, label)
    if any(type(item) is not str for item in items):
        _fail(label + " must contain only strings")
    return tuple(items)  # type: ignore[arg-type]


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(label + " must be one lowercase SHA-256")
    return value


def _optional_sha(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _sha(value, label)


def _detached(value: object, label: str) -> object:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FenceBootstrapPublicationError(
            label + " is not canonical JSON data"
        ) from exc


def _principal(value: object, label: str) -> PrincipalIdentity:
    item = _exact_mapping(value, _PRINCIPAL_FIELDS, label)
    try:
        return PrincipalIdentity(
            binding_id=item["binding_id"],  # type: ignore[arg-type]
            arn=item["arn"],  # type: ignore[arg-type]
            role_id=item["role_id"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise FenceBootstrapPublicationError(label + " is not exact") from exc


def _principals(value: object, label: str) -> tuple[PrincipalIdentity, ...]:
    return tuple(
        _principal(item, "%s[%d]" % (label, index))
        for index, item in enumerate(_array(value, label))
    )


def _reserved_families(value: object) -> tuple[ReservedFamily, ...]:
    result = []
    for index, raw in enumerate(_array(value, "reserved_families")):
        item = _exact_mapping(
            raw, _RESERVED_FAMILY_FIELDS, "reserved_families[%d]" % index
        )
        try:
            result.append(
                ReservedFamily(
                    family_id=item["family_id"],  # type: ignore[arg-type]
                    resources=_strings(item["resources"], "reserved family resources"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise FenceBootstrapPublicationError(
                "reserved family is not exact"
            ) from exc
    return tuple(result)


def _writer_cohorts(value: object) -> tuple[WriterCohort, ...]:
    result = []
    for index, raw in enumerate(_array(value, "writer_cohorts")):
        item = _exact_mapping(raw, _WRITER_COHORT_FIELDS, "writer_cohorts[%d]" % index)
        try:
            result.append(
                WriterCohort(
                    cohort_id=item["cohort_id"],  # type: ignore[arg-type]
                    members=_principals(item["members"], "writer cohort members"),
                    guard_resources=_strings(
                        item["guard_resources"], "writer cohort resources"
                    ),
                    cross_member_denial_evidence_sha256=_sha(
                        item["cross_member_denial_evidence_sha256"],
                        "writer cohort denial evidence",
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise FenceBootstrapPublicationError("writer cohort is not exact") from exc
    return tuple(result)


def _policy_limits(value: object) -> PolicyLimits:
    item = _exact_mapping(value, _POLICY_LIMIT_FIELDS, "policy_limits")
    if type(item["component_max_bytes"]) is not dict:
        _fail("policy_limits component_max_bytes is not one exact object")
    try:
        return PolicyLimits(
            max_policy_bytes=item["max_policy_bytes"],  # type: ignore[arg-type]
            unallocated_headroom_bytes=item["unallocated_headroom_bytes"],  # type: ignore[arg-type]
            component_max_bytes=item["component_max_bytes"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise FenceBootstrapPublicationError("policy_limits is not exact") from exc


def _renderer_input(
    value: object,
    index: int,
    *,
    validate_renderable: bool = True,
) -> FencePolicyInput:
    item = _exact_mapping(
        value, FENCE_POLICY_INPUT_FIELDS, "renderer_inputs[%d]" % index
    )
    legacy = _array(item["legacy_statements"], "legacy_statements")
    if any(type(statement) is not dict for statement in legacy):
        _fail("legacy_statements contains a non-object")
    try:
        build_mode = BuildMode(item["build_mode"])
        policy_head = PolicyHead(item["policy_head"])
    except (TypeError, ValueError) as exc:
        raise FenceBootstrapPublicationError(
            "renderer enum discriminant is not exact"
        ) from exc
    try:
        rendered_input = FencePolicyInput(
            build_mode=build_mode,
            policy_head=policy_head,
            bucket_name=item["bucket_name"],  # type: ignore[arg-type]
            account_id=item["account_id"],  # type: ignore[arg-type]
            kms_key_arn=item["kms_key_arn"],  # type: ignore[arg-type]
            predecessor_policy_sha256=_optional_sha(
                item["predecessor_policy_sha256"], "predecessor policy"
            ),
            freeze_denial_evidence_sha256=_optional_sha(
                item["freeze_denial_evidence_sha256"], "freeze denial evidence"
            ),
            source_publication_sealed_sha256=_optional_sha(
                item["source_publication_sealed_sha256"], "source publication seal"
            ),
            all_version_inventory_sha256=_optional_sha(
                item["all_version_inventory_sha256"], "version inventory"
            ),
            source_settlement_sha256=_optional_sha(
                item["source_settlement_sha256"], "source settlement"
            ),
            publisher_deny_policy_sha256=_optional_sha(
                item["publisher_deny_policy_sha256"], "publisher deny policy"
            ),
            terminal_prerequisite_sha256=_optional_sha(
                item["terminal_prerequisite_sha256"], "terminal prerequisite"
            ),
            legacy_statements=tuple(legacy),
            permanent_enrolled_resources=_strings(
                item["permanent_enrolled_resources"], "permanent_enrolled_resources"
            ),
            retired_publishers=_principals(
                item["retired_publishers"], "retired_publishers"
            ),
            retired_publisher_resources=_strings(
                item["retired_publisher_resources"], "retired_publisher_resources"
            ),
            reserved_families=_reserved_families(item["reserved_families"]),
            writer_cohorts=_writer_cohorts(item["writer_cohorts"]),
            writer_owned_resources=_strings(
                item["writer_owned_resources"], "writer_owned_resources"
            ),
            source_validation_readers=_principals(
                item["source_validation_readers"], "source_validation_readers"
            ),
            source_inventory_readers=_principals(
                item["source_inventory_readers"], "source_inventory_readers"
            ),
            terminal_audit_readers=_principals(
                item["terminal_audit_readers"], "terminal_audit_readers"
            ),
            selected_source_keys=_strings(
                item["selected_source_keys"], "selected_source_keys"
            ),
            nonselected_source_keys=_strings(
                item["nonselected_source_keys"], "nonselected_source_keys"
            ),
            policy_limits=_policy_limits(item["policy_limits"]),
        )
        if validate_renderable:
            render_fence_policy(rendered_input)
        return rendered_input
    except FenceBootstrapPublicationError:
        raise
    except (TypeError, ValueError) as exc:
        raise FenceBootstrapPublicationError(
            "renderer_inputs[%d] is not approved" % index
        ) from exc


def _principal_projection(value: PrincipalIdentity) -> dict[str, str]:
    return {"binding_id": value.binding_id, "arn": value.arn, "role_id": value.role_id}


def _renderer_input_projection(value: FencePolicyInput) -> dict[str, object]:
    return {
        "build_mode": value.build_mode.value,
        "policy_head": value.policy_head.value,
        "bucket_name": value.bucket_name,
        "account_id": value.account_id,
        "kms_key_arn": value.kms_key_arn,
        "predecessor_policy_sha256": value.predecessor_policy_sha256,
        "freeze_denial_evidence_sha256": value.freeze_denial_evidence_sha256,
        "source_publication_sealed_sha256": value.source_publication_sealed_sha256,
        "all_version_inventory_sha256": value.all_version_inventory_sha256,
        "source_settlement_sha256": value.source_settlement_sha256,
        "publisher_deny_policy_sha256": value.publisher_deny_policy_sha256,
        "terminal_prerequisite_sha256": value.terminal_prerequisite_sha256,
        "legacy_statements": [dict(item) for item in value.legacy_statements],
        "permanent_enrolled_resources": list(value.permanent_enrolled_resources),
        "retired_publishers": [
            _principal_projection(item) for item in value.retired_publishers
        ],
        "retired_publisher_resources": list(value.retired_publisher_resources),
        "reserved_families": [
            {"family_id": item.family_id, "resources": list(item.resources)}
            for item in value.reserved_families
        ],
        "writer_cohorts": [
            {
                "cohort_id": item.cohort_id,
                "members": [_principal_projection(member) for member in item.members],
                "guard_resources": list(item.guard_resources),
                "cross_member_denial_evidence_sha256": item.cross_member_denial_evidence_sha256,
            }
            for item in value.writer_cohorts
        ],
        "writer_owned_resources": list(value.writer_owned_resources),
        "source_validation_readers": [
            _principal_projection(item) for item in value.source_validation_readers
        ],
        "source_inventory_readers": [
            _principal_projection(item) for item in value.source_inventory_readers
        ],
        "terminal_audit_readers": [
            _principal_projection(item) for item in value.terminal_audit_readers
        ],
        "selected_source_keys": list(value.selected_source_keys),
        "nonselected_source_keys": list(value.nonselected_source_keys),
        "policy_limits": {
            "max_policy_bytes": value.policy_limits.max_policy_bytes,
            "unallocated_headroom_bytes": value.policy_limits.unallocated_headroom_bytes,
            "component_max_bytes": dict(value.policy_limits.component_max_bytes),
        },
    }


def fence_policy_input_projection(value: FencePolicyInput) -> dict[str, object]:
    if type(value) is not FencePolicyInput:
        raise TypeError("fence policy projection input must be exact")
    render_fence_policy(value)
    return _renderer_input_projection(value)


def fence_policy_input_prototype_projection(
    value: FencePolicyInput,
) -> dict[str, object]:
    if type(value) is not FencePolicyInput:
        raise TypeError("fence policy prototype projection input must be exact")
    return _renderer_input_projection(value)


def parse_fence_policy_input_projection(value: object) -> FencePolicyInput:
    return _renderer_input(value, 0)


def parse_fence_policy_input_prototype_projection(
    value: object,
) -> FencePolicyInput:
    return _renderer_input(value, 0, validate_renderable=False)


@dataclass(frozen=True)
class BridgeSeedPublicationRequestV2:
    """Immutable authority for the sole pre-migration fence artifact."""

    activation_id: str
    generation: int
    materializer_function_version_arn: str
    expected_live_preseed_policy_sha256: str
    renderer_input: FencePolicyInput


@dataclass(frozen=True)
class BridgeSeedPublicationAuthorityV2:
    """Pre-effect bridge-seed authority with no future Lambda identity."""

    activation_id: str
    generation: int
    expected_live_preseed_policy_sha256: str
    renderer_input: FencePolicyInput


@dataclass(frozen=True)
class BridgeSeedPublicationV2:
    """Truthful seed publication evidence, including create/adopt disposition."""

    artifact: BridgeSeedArtifact
    materializer_function_version_arn: str
    disposition: FixedKeyPublicationDisposition
    canonical_identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "record_type": "glm52_h1g_bridge_seed_publication_v2",
            "artifact": self.artifact.to_dict(),
            "materializer_function_version_arn": (
                self.materializer_function_version_arn
            ),
            "disposition": self.disposition.value,
            "canonical_identity_sha256": self.canonical_identity_sha256,
        }


def parse_bridge_seed_publication_v2(
    value: object,
) -> BridgeSeedPublicationV2:
    item = _exact_mapping(
        value,
        frozenset(
            {
                "schema_version",
                "record_type",
                "artifact",
                "materializer_function_version_arn",
                "disposition",
                "canonical_identity_sha256",
            }
        ),
        "bridge seed publication evidence",
    )
    materializer_arn = item["materializer_function_version_arn"]
    if (
        item["schema_version"] != 2
        or item["record_type"] != "glm52_h1g_bridge_seed_publication_v2"
        or type(materializer_arn) is not str
        or _LAMBDA_VERSION_ARN.fullmatch(materializer_arn) is None
        or type(item["canonical_identity_sha256"]) is not str
        or _SHA256.fullmatch(item["canonical_identity_sha256"]) is None
    ):
        _fail("bridge seed publication evidence identity is not exact")
    try:
        disposition = FixedKeyPublicationDisposition(item["disposition"])
        artifact = parse_bridge_seed_artifact(item["artifact"])
    except (TypeError, ValueError) as exc:
        raise FenceBootstrapPublicationError(
            "bridge seed publication evidence is malformed"
        ) from exc
    unsigned = dict(item)
    identity = unsigned.pop("canonical_identity_sha256")
    if canonical_sha256(unsigned) != identity:
        _fail("bridge seed publication evidence canonical identity drifted")
    return BridgeSeedPublicationV2(
        artifact=artifact,
        materializer_function_version_arn=materializer_arn,
        disposition=disposition,
        canonical_identity_sha256=identity,
    )


def parse_bridge_seed_publication_request_v2(
    value: object,
) -> BridgeSeedPublicationRequestV2:
    item = _exact_mapping(
        value,
        BRIDGE_SEED_PUBLICATION_REQUEST_FIELDS,
        "bridge seed publication request",
    )
    materializer_arn = item["materializer_function_version_arn"]
    if (
        item["schema_version"] != 2
        or item["record_type"] != "glm52_h1g_bridge_seed_publication_request_v2"
        or item["activation_id"] != _ACTIVATION_ID
        or item["generation"] != _GENERATION
        or type(materializer_arn) is not str
        or _LAMBDA_VERSION_ARN.fullmatch(materializer_arn) is None
    ):
        _fail("bridge seed publication request identity is not exact")
    expected_preseed = _sha(
        item["expected_live_preseed_policy_sha256"],
        "expected live preseed policy",
    )
    renderer_input = _renderer_input(item["renderer_input"], 0)
    if renderer_input.policy_head is not PolicyHead.BRIDGE_SEED:
        _fail("bridge seed publication renderer head is not exact")
    return BridgeSeedPublicationRequestV2(
        activation_id=_ACTIVATION_ID,
        generation=_GENERATION,
        materializer_function_version_arn=materializer_arn,
        expected_live_preseed_policy_sha256=expected_preseed,
        renderer_input=renderer_input,
    )


def parse_bridge_seed_publication_authority_v2(
    value: object,
) -> BridgeSeedPublicationAuthorityV2:
    """Parse bridge-seed intent before the retained Lambda version exists."""

    item = _exact_mapping(
        value,
        BRIDGE_SEED_PUBLICATION_AUTHORITY_FIELDS,
        "bridge seed publication authority",
    )
    if (
        item["schema_version"] != 2
        or item["record_type"]
        != "glm52_h1g_bridge_seed_publication_authority_v2"
        or item["activation_id"] != _ACTIVATION_ID
        or item["generation"] != _GENERATION
    ):
        _fail("bridge seed publication authority identity is not exact")
    expected_preseed = _sha(
        item["expected_live_preseed_policy_sha256"],
        "expected live preseed policy",
    )
    renderer_input = _renderer_input(item["renderer_input"], 0)
    if renderer_input.policy_head is not PolicyHead.BRIDGE_SEED:
        _fail("bridge seed publication authority renderer head is not exact")
    return BridgeSeedPublicationAuthorityV2(
        activation_id=_ACTIVATION_ID,
        generation=_GENERATION,
        expected_live_preseed_policy_sha256=expected_preseed,
        renderer_input=renderer_input,
    )


def materialize_bridge_seed_publication_request_v2(
    *,
    authority: object,
    materializer_function_version_arn: str,
) -> dict[str, object]:
    """Bind one pre-effect authority to the exact live retained Lambda version."""

    parse_bridge_seed_publication_authority_v2(authority)
    if type(authority) is not dict:
        _fail("bridge seed publication authority is not one object")
    request = {
        **dict(authority),
        "record_type": "glm52_h1g_bridge_seed_publication_request_v2",
        "materializer_function_version_arn": materializer_function_version_arn,
    }
    parse_bridge_seed_publication_request_v2(request)
    return request


def _bridge_seed_publication(
    *,
    request: object,
    services: Task13FixedArtifactServices,
    read_only: bool,
) -> BridgeSeedPublicationV2:
    parsed = parse_bridge_seed_publication_request_v2(request)
    if (
        type(services) is not Task13FixedArtifactServices
        or services.total_max_attempts != 1
        or (
            not read_only
            and (services.publisher_s3 is None or services.publisher_s3 is services.s3)
        )
    ):
        _fail("bridge seed publisher client is not separately authorized")
    rendered = render_fence_policy(parsed.renderer_input)
    outcome = (
        adopt_fixed_key_bytes(
            services=services,
            bucket=CAMPAIGN_BUCKET,
            key=BOOTSTRAP_PUBLICATION_ORDER[0],
            raw=rendered.policy_bytes,
            record_type="glm52_h1g_bridge_seed_policy_v1",
            sse_kms_key_id=parsed.renderer_input.kms_key_arn,
        )
        if read_only
        else publish_fixed_key_bytes_with_outcome(
            services=services,
            bucket=CAMPAIGN_BUCKET,
            key=BOOTSTRAP_PUBLICATION_ORDER[0],
            raw=rendered.policy_bytes,
            record_type="glm52_h1g_bridge_seed_policy_v1",
            sse_kms_key_id=parsed.renderer_input.kms_key_arn,
        )
    )
    artifact = build_bridge_seed_artifact(
        rendered=rendered,
        expected_live_preseed_policy_sha256=(
            parsed.expected_live_preseed_policy_sha256
        ),
        version_id=outcome.version_id,
    )
    unsigned = {
        "schema_version": 2,
        "record_type": "glm52_h1g_bridge_seed_publication_v2",
        "artifact": artifact.to_dict(),
        "materializer_function_version_arn": (parsed.materializer_function_version_arn),
        "disposition": outcome.disposition.value,
    }
    return BridgeSeedPublicationV2(
        artifact=artifact,
        materializer_function_version_arn=(parsed.materializer_function_version_arn),
        disposition=outcome.disposition,
        canonical_identity_sha256=canonical_sha256(unsigned),
    )


def publish_bridge_seed_policy_v2(
    *,
    request: object,
    services: Task13FixedArtifactServices,
) -> BridgeSeedPublicationV2:
    """Create or adopt only BRIDGE_SEED through a distinct publisher client."""

    return _bridge_seed_publication(
        request=request,
        services=services,
        read_only=False,
    )


def read_bridge_seed_publication_v2(
    *,
    request: object,
    services: Task13FixedArtifactServices,
) -> BridgeSeedPublicationV2:
    """Adopt the exact BRIDGE_SEED using bounded reads only."""

    return _bridge_seed_publication(
        request=request,
        services=services,
        read_only=True,
    )


@dataclass(frozen=True)
class BootstrapPublicationRequestV2:
    """Parsed authority inputs from which every publication byte is derived."""

    activation_id: str
    generation: int
    materializer_function_version_arn: str
    checkpoint_identity_sha256: str
    renderer_inputs: tuple[FencePolicyInput, ...]
    manifest_authority: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            self.activation_id != _ACTIVATION_ID
            or self.generation != _GENERATION
            or tuple(item.policy_head for item in self.renderer_inputs) != _HEAD_ORDER
            or _LAMBDA_VERSION_ARN.fullmatch(self.materializer_function_version_arn)
            is None
        ):
            _fail("bootstrap publication activation/generation/renderer order drifted")
        _sha(self.checkpoint_identity_sha256, "checkpoint identity")
        object.__setattr__(
            self,
            "manifest_authority",
            MappingProxyType(
                dict(_detached(self.manifest_authority, "manifest authority"))
            ),
        )


@dataclass(frozen=True)
class BootstrapPublicationAuthorityV2:
    """Pre-effect five-head authority without checkpoint or Lambda outputs."""

    activation_id: str
    generation: int
    renderer_inputs: tuple[FencePolicyInput, ...]
    manifest_authority: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            self.activation_id != _ACTIVATION_ID
            or self.generation != _GENERATION
            or tuple(item.policy_head for item in self.renderer_inputs) != _HEAD_ORDER
        ):
            _fail("bootstrap publication authority renderer order drifted")
        object.__setattr__(
            self,
            "manifest_authority",
            MappingProxyType(
                dict(_detached(self.manifest_authority, "manifest authority"))
            ),
        )


def parse_bootstrap_publication_authority_v2(
    value: object,
) -> BootstrapPublicationAuthorityV2:
    """Parse complete pre-effect renderer intent without future effect IDs."""

    item = _exact_mapping(
        value,
        BOOTSTRAP_PUBLICATION_AUTHORITY_FIELDS,
        "bootstrap publication authority",
    )
    if (
        item["schema_version"] != 2
        or item["record_type"] != "glm52_fence_bootstrap_publication_authority_v2"
        or item["activation_id"] != _ACTIVATION_ID
        or item["generation"] != _GENERATION
    ):
        _fail("bootstrap publication authority identity is not exact")
    renderer_values = _array(item["renderer_inputs"], "renderer_inputs")
    if len(renderer_values) != len(_HEAD_ORDER):
        _fail("renderer_inputs must contain the exact five bootstrap inputs")
    renderer_inputs = tuple(
        _renderer_input(raw, index) for index, raw in enumerate(renderer_values)
    )
    manifest = _exact_mapping(
        item["manifest_authority"],
        MANIFEST_AUTHORITY_FIELDS,
        "manifest_authority",
    )
    for field in (
        "mutation_authority_inventory",
        "executor_inventory",
        "principal_inventory",
        "writer_inventory",
        "writer_policy_cohorts",
        "source_inventory",
    ):
        _array(manifest[field], "manifest_authority." + field)
    for field in (
        "member_account_authority",
        "bucket_control_plane",
        "kms_key_identity",
    ):
        if type(manifest[field]) is not dict or not manifest[field]:
            _fail("manifest_authority.%s is not one nonempty object" % field)
    if (
        manifest["member_account_authority"].get("account") != ACCOUNT_ID
        or manifest["member_account_authority"].get("region") != REGION
        or manifest["bucket_control_plane"].get("versioning") != "Enabled"
        or manifest["kms_key_identity"].get("key_arn")
        != renderer_inputs[0].kms_key_arn
    ):
        _fail("manifest authority does not bind account/bucket/KMS identity")
    return BootstrapPublicationAuthorityV2(
        activation_id=_ACTIVATION_ID,
        generation=_GENERATION,
        renderer_inputs=renderer_inputs,
        manifest_authority=manifest,
    )


def materialize_bootstrap_publication_request_v2(
    *,
    authority: object,
    materializer_function_version_arn: str,
    checkpoint: StackMigrationTransferCheckpointV2,
) -> dict[str, object]:
    """Bind pre-effect renderer intent to exact migration/runtime readbacks."""

    parse_bootstrap_publication_authority_v2(authority)
    if type(authority) is not dict:
        _fail("bootstrap publication authority is not one object")
    request = {
        **dict(authority),
        "record_type": "glm52_fence_bootstrap_publication_request_v2",
        "materializer_function_version_arn": materializer_function_version_arn,
        "checkpoint_identity_sha256": checkpoint.canonical_identity_sha256,
    }
    parse_bootstrap_publication_request_v2(request, checkpoint=checkpoint)
    return request


@dataclass(frozen=True)
class _DerivedBootstrapPublication:
    request: BootstrapPublicationRequestV2
    rendered_by_head: Mapping[PolicyHead, object]
    template_bytes: Mapping[FenceSlot, bytes]
    render_set_identity_sha256: str


def parse_bootstrap_publication_request_v2(
    value: object,
    *,
    checkpoint: StackMigrationTransferCheckpointV2,
) -> BootstrapPublicationRequestV2:
    """Reject non-authority data and bind the request to operations 1-6."""

    if type(checkpoint) is not StackMigrationTransferCheckpointV2:
        _fail("bootstrap publication checkpoint type is not exact")
    item = _exact_mapping(
        value, BOOTSTRAP_PUBLICATION_REQUEST_FIELDS, "bootstrap publication request"
    )
    if (
        item["schema_version"] != 2
        or item["record_type"] != "glm52_fence_bootstrap_publication_request_v2"
        or item["activation_id"] != _ACTIVATION_ID
        or item["generation"] != _GENERATION
        or type(item["materializer_function_version_arn"]) is not str
        or _LAMBDA_VERSION_ARN.fullmatch(item["materializer_function_version_arn"])
        is None
        or item["checkpoint_identity_sha256"] != checkpoint.canonical_identity_sha256
    ):
        _fail("bootstrap publication request checkpoint/activation identity drifted")
    renderer_values = _array(item["renderer_inputs"], "renderer_inputs")
    if len(renderer_values) != len(_HEAD_ORDER):
        _fail("renderer_inputs must contain the exact five bootstrap inputs")
    renderer_inputs = tuple(
        _renderer_input(raw, index) for index, raw in enumerate(renderer_values)
    )
    authority = _exact_mapping(
        item["manifest_authority"], MANIFEST_AUTHORITY_FIELDS, "manifest_authority"
    )
    for field in (
        "mutation_authority_inventory",
        "executor_inventory",
        "principal_inventory",
        "writer_inventory",
        "writer_policy_cohorts",
        "source_inventory",
    ):
        _array(authority[field], "manifest_authority." + field)
    for field in (
        "member_account_authority",
        "bucket_control_plane",
        "kms_key_identity",
    ):
        if type(authority[field]) is not dict or not authority[field]:
            _fail("manifest_authority.%s is not one nonempty object" % field)
    member = authority["member_account_authority"]
    kms = authority["kms_key_identity"]
    if (
        member.get("account") != ACCOUNT_ID  # type: ignore[union-attr]
        or member.get("region") != REGION  # type: ignore[union-attr]
        or member.get("arn") != checkpoint.api_caller.role_arn  # type: ignore[union-attr]
        or kms.get("key_arn") != renderer_inputs[0].kms_key_arn  # type: ignore[union-attr]
        or authority["bucket_control_plane"].get("versioning") != "Enabled"  # type: ignore[union-attr]
    ):
        _fail("manifest authority does not bind the checkpoint/bucket/KMS identity")
    return BootstrapPublicationRequestV2(
        activation_id=item["activation_id"],  # type: ignore[arg-type]
        generation=item["generation"],  # type: ignore[arg-type]
        materializer_function_version_arn=(
            item["materializer_function_version_arn"]  # type: ignore[arg-type]
        ),
        checkpoint_identity_sha256=item["checkpoint_identity_sha256"],  # type: ignore[arg-type]
        renderer_inputs=renderer_inputs,
        manifest_authority=authority,
    )


def _derive(
    request: BootstrapPublicationRequestV2,
    checkpoint: StackMigrationTransferCheckpointV2,
) -> _DerivedBootstrapPublication:
    if request.checkpoint_identity_sha256 != checkpoint.canonical_identity_sha256:
        _fail("bootstrap publication checkpoint identity drifted")
    rendered = {
        item.policy_head: render_fence_policy(item) for item in request.renderer_inputs
    }
    seed = rendered[PolicyHead.BRIDGE_SEED]
    if (
        seed.policy_sha256 != checkpoint.bridge_seed_policy_sha256  # type: ignore[attr-defined]
        or checkpoint.bucket_name != CAMPAIGN_BUCKET
    ):
        _fail("renderer bridge seed does not equal the migration checkpoint")
    for child, parent in (
        (PolicyHead.PREPARE, PolicyHead.BRIDGE_SEED),
        (PolicyHead.RESERVATION, PolicyHead.PREPARE),
        (PolicyHead.SOURCE_FAMILIES_FROZEN, PolicyHead.RESERVATION),
        (PolicyHead.CLOSED_SOURCE, PolicyHead.SOURCE_FAMILIES_FROZEN),
    ):
        source = next(
            item for item in request.renderer_inputs if item.policy_head is child
        )
        if source.predecessor_policy_sha256 != rendered[parent].policy_sha256:  # type: ignore[attr-defined]
            _fail("renderer bootstrap predecessor graph drifted")
    kms_keys = {item.kms_key_arn for item in request.renderer_inputs}
    build_modes = {item.build_mode for item in request.renderer_inputs}
    publisher_hashes = {
        item.publisher_deny_policy_sha256
        for item in request.renderer_inputs
        if item.policy_head is not PolicyHead.BRIDGE_SEED
    }
    if len(kms_keys) != 1 or len(build_modes) != 1 or len(publisher_hashes) != 1:
        _fail("renderer bootstrap common authority inputs drifted")
    template_bytes = {
        _HEAD_TO_SLOT[head]: build_fence_template_bytes(policy)  # type: ignore[arg-type]
        for head, policy in rendered.items()
        if head is not PolicyHead.BRIDGE_SEED
    }
    render_identity = canonical_sha256(
        [_renderer_input_projection(item) for item in request.renderer_inputs]
    )
    return _DerivedBootstrapPublication(
        request=request,
        rendered_by_head=MappingProxyType(rendered),
        template_bytes=MappingProxyType(template_bytes),
        render_set_identity_sha256=render_identity,
    )


def _role_projection(role: object) -> dict[str, object]:
    return {
        "arn": role.role_arn,
        "role_id": role.role_id,
        "trust_policy_sha256": role.trust_policy_sha256,
        "permission_policy_sha256": role.permission_policy_sha256,
    }


def _manifest_fields(
    derived: _DerivedBootstrapPublication,
    checkpoint: StackMigrationTransferCheckpointV2,
) -> dict[str, object]:
    first = derived.request.renderer_inputs[0]
    limits = first.policy_limits
    authority = derived.request.manifest_authority
    return {
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "mutation_authority_inventory": authority["mutation_authority_inventory"],
        "activation_id": derived.request.activation_id,
        "member_account_authority": authority["member_account_authority"],
        "generation": derived.request.generation,
        "legacy_mode": "ENABLED"
        if first.build_mode is BuildMode.LEGACY_ENABLED
        else "DISABLED",
        "bucket_name": CAMPAIGN_BUCKET,
        "bucket_arn": "arn:aws:s3:::" + CAMPAIGN_BUCKET,
        "bucket_control_plane": authority["bucket_control_plane"],
        "kms_key_identity": authority["kms_key_identity"],
        "stack_name": _STACK_NAME,
        "stack_id": checkpoint.fence_stack_id,
        "executor_inventory": authority["executor_inventory"],
        "logical_id": _LOGICAL_ID,
        "migration_service_role": _role_projection(checkpoint.migration_service_role),
        "fence_service_role": _role_projection(checkpoint.fence_service_role),
        "policy_limits": {
            "working_limit_bytes": limits.max_policy_bytes
            - limits.unallocated_headroom_bytes,
            "design_limit_bytes": limits.max_policy_bytes,
            "s3_limit_bytes": _S3_POLICY_LIMIT,
        },
        "render_input_identity_sha256": derived.render_set_identity_sha256,
        "legacy_fragment_sha256": canonical_sha256(
            [dict(item) for item in first.legacy_statements]
        ),
        "principal_inventory": authority["principal_inventory"],
        "writer_inventory": authority["writer_inventory"],
        "writer_policy_cohorts": authority["writer_policy_cohorts"],
        "source_inventory": authority["source_inventory"],
    }


def _coordinate(
    key: str, version_id: str, raw: bytes, body_identity: str
) -> ArtifactCoordinate:
    return ArtifactCoordinate(
        bucket=CAMPAIGN_BUCKET,
        key=key,
        version_id=version_id,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_identity_sha256=body_identity,
    )


def _build_entries(
    *,
    derived: _DerivedBootstrapPublication,
    checkpoint: StackMigrationTransferCheckpointV2,
    versions: Mapping[FenceSlot, str],
) -> tuple[FenceArtifactEntry, ...]:
    by_head = {item.policy_head: item for item in derived.request.renderer_inputs}
    by_slot_input = {
        _HEAD_TO_SLOT[head]: value
        for head, value in by_head.items()
        if head in _HEAD_TO_SLOT
    }
    bridge_seed_hash = derived.rendered_by_head[PolicyHead.BRIDGE_SEED].policy_sha256  # type: ignore[attr-defined]
    entries = []
    for slot in _SLOT_MANIFEST_ORDER:
        input_value = by_slot_input[slot]
        rendered = derived.rendered_by_head[input_value.policy_head]
        entries.append(
            build_fence_entry(
                slot=slot,
                rendered=rendered,  # type: ignore[arg-type]
                version_id=versions[slot],
                render_input_identity_sha256=derived.render_set_identity_sha256,
                stack_id=checkpoint.fence_stack_id,
                migration_service_role_arn=checkpoint.migration_service_role.role_arn,
                fence_service_role_arn=checkpoint.fence_service_role.role_arn,
                bridge_seed_policy_sha256=bridge_seed_hash,
                expected_prestate_policy_sha256=input_value.predecessor_policy_sha256,  # type: ignore[arg-type]
                publisher_deny_policy_sha256=input_value.publisher_deny_policy_sha256,  # type: ignore[arg-type]
                batch_projection_contract=None,
                successor_contract_sha256=(
                    None
                    if slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE
                    else input_value.render_input_identity_sha256
                ),
            )
        )
    return tuple(entries)


def _publication_result(
    *,
    coordinates: tuple[ArtifactCoordinate, ...],
    manifest: FenceManifest,
    checkpoint: StackMigrationTransferCheckpointV2,
    seed_disposition: FixedKeyPublicationDisposition,
) -> BootstrapFencePublication:
    prepare_identity = manifest.entry(
        FenceSlot.PREPARE_GENESIS_LIVE_STATE
    ).entry_identity_sha256
    unsigned = {
        "schema_version": 2,
        "record_type": "glm52_h1g_bootstrap_fence_publication_v2",
        "coordinates": [item.to_dict() for item in coordinates],
        "prepare_entry_identity_sha256": prepare_identity,
        "checkpoint_identity_sha256": checkpoint.canonical_identity_sha256,
        "seed_adopted": seed_disposition is FixedKeyPublicationDisposition.ADOPTED,
    }
    return BootstrapFencePublication(
        coordinates=coordinates,
        prepare_entry_identity_sha256=prepare_identity,
        checkpoint_identity_sha256=checkpoint.canonical_identity_sha256,
        seed_adopted=unsigned["seed_adopted"],  # type: ignore[arg-type]
        canonical_identity_sha256=canonical_sha256(unsigned),
    )


def _materialize(
    *,
    request: object,
    checkpoint: StackMigrationTransferCheckpointV2,
    services: object,
    read_only: bool,
) -> BootstrapFencePublication:
    from .task13_production_operations import ProductionServices

    if type(services) is ProductionServices:
        fixed_services = Task13FixedArtifactServices(
            sts=services.sts,
            s3=services.s3,
            total_max_attempts=services.total_max_attempts,
            publisher_s3=services.fixed_artifact_publisher_s3,
        )
    elif type(services) is Task13FixedArtifactServices:
        fixed_services = services
    else:
        _fail("bootstrap publication services are not exact zero-retry services")
    if fixed_services.total_max_attempts != 1:
        _fail("bootstrap publication services are not exact zero-retry services")
    if not read_only and (
        fixed_services.publisher_s3 is None
        or fixed_services.publisher_s3 is fixed_services.s3
    ):
        _fail("bootstrap publisher client is not separately authorized")
    parsed = parse_bootstrap_publication_request_v2(
        request,
        checkpoint=checkpoint,
    )
    derived = _derive(parsed, checkpoint)
    kms_key_arn = parsed.renderer_inputs[0].kms_key_arn
    seed_raw = derived.rendered_by_head[PolicyHead.BRIDGE_SEED].policy_bytes  # type: ignore[attr-defined]
    seed_outcome = adopt_fixed_key_bytes(
        services=fixed_services,
        bucket=CAMPAIGN_BUCKET,
        key=BOOTSTRAP_PUBLICATION_ORDER[0],
        raw=seed_raw,
        record_type="glm52_h1g_bridge_seed_policy_v1",
        sse_kms_key_id=kms_key_arn,
    )
    seed_coordinate = _coordinate(
        BOOTSTRAP_PUBLICATION_ORDER[0],
        seed_outcome.version_id,
        seed_raw,
        hashlib.sha256(seed_raw).hexdigest(),
    )
    coordinates = [seed_coordinate]
    versions: dict[FenceSlot, str] = {}
    for slot, key in zip(_SLOT_PUBLICATION_ORDER, BOOTSTRAP_PUBLICATION_ORDER[1:-1]):
        raw = derived.template_bytes[slot]
        outcome = (
            adopt_fixed_key_bytes(
                services=fixed_services,
                bucket=CAMPAIGN_BUCKET,
                key=key,
                raw=raw,
                record_type="aws_cloudformation_template",
                sse_kms_key_id=kms_key_arn,
            )
            if read_only
            else publish_fixed_key_bytes_with_outcome(
                services=fixed_services,
                bucket=CAMPAIGN_BUCKET,
                key=key,
                raw=raw,
                record_type="aws_cloudformation_template",
                sse_kms_key_id=kms_key_arn,
            )
        )
        versions[slot] = outcome.version_id
        coordinates.append(
            _coordinate(
                key, outcome.version_id, raw, hashlib.sha256(raw[:-1]).hexdigest()
            )
        )
    seed_artifact = build_bridge_seed_artifact(
        rendered=derived.rendered_by_head[PolicyHead.BRIDGE_SEED],  # type: ignore[arg-type]
        expected_live_preseed_policy_sha256=checkpoint.expected_live_preseed_policy_sha256,
        version_id=seed_outcome.version_id,
    )
    entries = _build_entries(
        derived=derived,
        checkpoint=checkpoint,
        versions=versions,
    )
    manifest = build_fence_manifest(
        stage=ManifestStage.BOOTSTRAP,
        fields=_manifest_fields(derived, checkpoint),
        bridge_seed=seed_artifact,
        entries=entries,
    )
    manifest_raw = canonical_json_bytes(manifest.to_dict()) + b"\n"
    manifest_key = BOOTSTRAP_PUBLICATION_ORDER[-1]
    manifest_outcome = (
        adopt_fixed_key_bytes(
            services=fixed_services,
            bucket=CAMPAIGN_BUCKET,
            key=manifest_key,
            raw=manifest_raw,
            record_type="glm52_fence_bootstrap_manifest_v2",
            sse_kms_key_id=kms_key_arn,
        )
        if read_only
        else publish_fixed_key_bytes_with_outcome(
            services=fixed_services,
            bucket=CAMPAIGN_BUCKET,
            key=manifest_key,
            raw=manifest_raw,
            record_type="glm52_fence_bootstrap_manifest_v2",
            sse_kms_key_id=kms_key_arn,
        )
    )
    coordinates.append(
        _coordinate(
            manifest_key,
            manifest_outcome.version_id,
            manifest_raw,
            manifest.canonical_identity_sha256,
        )
    )
    result_coordinates = tuple(coordinates)
    if tuple(item.key for item in result_coordinates) != BOOTSTRAP_PUBLICATION_ORDER:
        raise AssertionError("bootstrap publication order implementation drifted")
    return _publication_result(
        coordinates=result_coordinates,
        manifest=manifest,
        checkpoint=checkpoint,
        seed_disposition=seed_outcome.disposition,
    )


def publish_bootstrap_fence_artifacts_v2(
    *,
    request: object,
    checkpoint: StackMigrationTransferCheckpointV2,
    services: object,
) -> BootstrapFencePublication:
    """Adopt the exact seed, then create/adopt four templates and manifest last."""

    return _materialize(
        request=request,
        checkpoint=checkpoint,
        services=services,
        read_only=False,
    )


def read_bootstrap_fence_publication_v2(
    *,
    request: object,
    checkpoint: StackMigrationTransferCheckpointV2,
    services: object,
) -> BootstrapFencePublication:
    """Rebuild and adopt all six expected objects using bounded reads only."""

    return _materialize(
        request=request,
        checkpoint=checkpoint,
        services=services,
        read_only=True,
    )


__all__ = [
    "BOOTSTRAP_PUBLICATION_AUTHORITY_FIELDS",
    "BOOTSTRAP_PUBLICATION_REQUEST_FIELDS",
    "BRIDGE_SEED_PUBLICATION_AUTHORITY_FIELDS",
    "BRIDGE_SEED_PUBLICATION_REQUEST_FIELDS",
    "FENCE_POLICY_INPUT_FIELDS",
    "MANIFEST_AUTHORITY_FIELDS",
    "BootstrapPublicationAuthorityV2",
    "BootstrapPublicationRequestV2",
    "BridgeSeedPublicationAuthorityV2",
    "BridgeSeedPublicationRequestV2",
    "BridgeSeedPublicationV2",
    "FenceBootstrapPublicationError",
    "fence_policy_input_projection",
    "fence_policy_input_prototype_projection",
    "materialize_bootstrap_publication_request_v2",
    "materialize_bridge_seed_publication_request_v2",
    "parse_bootstrap_publication_authority_v2",
    "parse_bootstrap_publication_request_v2",
    "parse_bridge_seed_publication_authority_v2",
    "parse_bridge_seed_publication_request_v2",
    "parse_bridge_seed_publication_v2",
    "parse_fence_policy_input_projection",
    "parse_fence_policy_input_prototype_projection",
    "publish_bootstrap_fence_artifacts_v2",
    "publish_bridge_seed_policy_v2",
    "read_bootstrap_fence_publication_v2",
    "read_bridge_seed_publication_v2",
]
