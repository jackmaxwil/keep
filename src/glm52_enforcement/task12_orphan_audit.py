"""Exact retained-resource, KMS-grant, and marker-last Task 12 audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import re
from typing import Optional, Tuple

from .canonical import canonical_sha256


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
_SHA = re.compile(r"^[0-9a-f]{64}$")

# This is the closed campaign-wide resource-family scan, including families
# expected to be empty after support deletion.
AUDITED_RESOURCE_TYPES = (
    "EC2_INSTANCE",
    "EBS_VOLUME",
    "EBS_SNAPSHOT",
    "ELASTIC_IP",
    "NAT_GATEWAY",
    "NETWORK_INTERFACE",
    "SECURITY_GROUP",
    "SUBNET",
    "ROUTE_TABLE",
    "ROUTE_ASSOCIATION",
    "ROUTE",
    "GATEWAY_ENDPOINT",
    "INTERFACE_ENDPOINT",
    "LAMBDA_FUNCTION",
    "LAMBDA_VERSION",
    "LAMBDA_POLICY",
    "IAM_ROLE",
    "IAM_INSTANCE_PROFILE",
    "IAM_POLICY",
    "STEP_FUNCTION_VERSION",
    "STEP_FUNCTION_EXECUTION",
    "SCHEDULE",
    "EVENT_RULE",
    "ALARM",
    "LOG_GROUP",
    "QUEUE",
    "DLQ",
    "REHEARSAL_BUCKET",
    "SECRET",
    "KMS_GRANT",
    "LEDGER",
    "KMS_KEY",
    "EVIDENCE_BUCKET",
    "PRODUCTION_FENCE_STACK",
    "LIFECYCLE_RESOURCE",
    "SOURCE_PUBLISHER_IDENTITY",
)
INTENTIONAL_RETAINED_RESOURCE_TYPES = (
    "LEDGER",
    "KMS_KEY",
    "EVIDENCE_BUCKET",
    "PRODUCTION_FENCE_STACK",
    "LIFECYCLE_RESOURCE",
    "SOURCE_PUBLISHER_IDENTITY",
)
_ACTIVATION_GRANT_OPERATIONS = frozenset(
    {
        "Decrypt",
        "Encrypt",
        "GenerateDataKey",
        "GenerateDataKeyWithoutPlaintext",
        "DescribeKey",
        "ReEncryptFrom",
        "ReEncryptTo",
    }
)


class Task12OrphanAuditError(ValueError):
    """Finalization evidence contains an orphan or incomplete audit."""


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise Task12OrphanAuditError(label + " must be a lowercase SHA-256")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise Task12OrphanAuditError(label + " must be a nonempty exact string")
    return value


def _utc(value: object, label: str) -> datetime:
    if type(value) is not str:
        raise Task12OrphanAuditError(label + " must be canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise Task12OrphanAuditError(label + " must be canonical UTC") from exc
    return parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class RetainedResource:
    resource_type: str
    resource_id: str
    cost_class: str

    def __post_init__(self) -> None:
        if self.resource_type not in AUDITED_RESOURCE_TYPES:
            raise Task12OrphanAuditError("retained resource type is unknown")
        _text(self.resource_id, "retained resource ID")
        _text(self.cost_class, "retained resource cost class")


@dataclass(frozen=True)
class ResourceInventoryScan:
    covered_resource_types: Tuple[str, ...]
    resources: Tuple[RetainedResource, ...]
    unreadable_resource_ids: Tuple[str, ...]
    pagination_complete: bool
    observed_at: str
    evidence_identity_sha256: str


@dataclass(frozen=True)
class KmsGrantIdentity:
    grant_id: str
    grant_name: str
    grantee_principal: str
    retiring_principal: str
    operations: Tuple[str, ...]
    encryption_context_identity_sha256: str

    def __post_init__(self) -> None:
        _text(self.grant_id, "KMS grant ID")
        _text(self.grant_name, "KMS grant name")
        _text(self.grantee_principal, "KMS grant grantee")
        _text(self.retiring_principal, "KMS grant retiring principal")
        if (
            type(self.operations) is not tuple
            or not self.operations
            or tuple(sorted(self.operations)) != self.operations
            or len(self.operations) != len(set(self.operations))
        ):
            raise Task12OrphanAuditError(
                "KMS grant operations must be sorted and unique"
            )
        for operation in self.operations:
            _text(operation, "KMS grant operation")
        _sha(
            self.encryption_context_identity_sha256,
            "KMS grant encryption context",
        )


@dataclass(frozen=True)
class DirectGrantAttribution:
    grant: KmsGrantIdentity
    provenance: str
    create_request_identity_sha256: str
    create_response_identity_sha256: Optional[str]
    reconciliation_list_grants_identity_sha256: Optional[str]
    revoke_or_retire_identity_sha256: str


@dataclass(frozen=True)
class ServiceGrantAttribution:
    grant: KmsGrantIdentity
    baseline_diff_identity_sha256: str
    list_grants_identity_sha256: str
    cloudtrail_request_identity_sha256: str
    cloudtrail_response_identity_sha256: str
    originating_resource_identity: str
    originating_service: str
    encryption_context_correlation_identity_sha256: str
    revoke_or_retire_identity_sha256: str


@dataclass(frozen=True)
class KmsGrantScan:
    grants: Tuple[KmsGrantIdentity, ...]
    pagination_complete: bool
    observed_at: str
    list_grants_identity_sha256: str


@dataclass(frozen=True)
class OrphanAuditProof:
    retained_resources: Tuple[RetainedResource, ...]
    retained_cost_classes: Tuple[str, ...]
    orphan_resource_ids: Tuple[str, ...]
    retained_grant_baseline_identity_sha256: str
    final_grants: Tuple[KmsGrantIdentity, ...]
    final_list_grants_identity_sha256: str
    settling_deadline: str
    observed_at: str
    canonical_identity_sha256: str


def _resource_order(resources: Tuple[RetainedResource, ...]) -> Tuple[
    Tuple[str, str], ...
]:
    rank = {
        item: index
        for index, item in enumerate(INTENTIONAL_RETAINED_RESOURCE_TYPES)
    }
    return tuple(
        (item.resource_type, item.resource_id)
        for item in sorted(
            resources,
            key=lambda item: (
                rank.get(item.resource_type, len(rank)),
                item.resource_type,
                item.resource_id,
            ),
        )
    )


def _validate_expected_resources(
    value: object,
) -> Tuple[RetainedResource, ...]:
    if type(value) is not tuple or not value:
        raise Task12OrphanAuditError(
            "expected retained inventory must be a nonempty exact tuple"
        )
    if any(not isinstance(item, RetainedResource) for item in value):
        raise Task12OrphanAuditError("retained inventory members must be typed")
    identities = tuple((item.resource_type, item.resource_id) for item in value)
    if (
        identities != _resource_order(value)
        or len(identities) != len(set(identities))
        or set(item.resource_type for item in value)
        != set(INTENTIONAL_RETAINED_RESOURCE_TYPES)
    ):
        raise Task12OrphanAuditError(
            "expected retained inventory is missing, duplicated, or unordered"
        )
    return value


def _validate_inventory(
    value: object,
    *,
    expected: Tuple[RetainedResource, ...],
) -> ResourceInventoryScan:
    if not isinstance(value, ResourceInventoryScan):
        raise Task12OrphanAuditError("resource inventory scan must be typed")
    if value.covered_resource_types != AUDITED_RESOURCE_TYPES:
        raise Task12OrphanAuditError(
            "orphan audit resource-family coverage is incomplete"
        )
    if value.pagination_complete is not True:
        raise Task12OrphanAuditError("resource inventory scan is not complete")
    _utc(value.observed_at, "resource inventory observed_at")
    _sha(value.evidence_identity_sha256, "resource inventory evidence")
    if (
        type(value.resources) is not tuple
        or type(value.unreadable_resource_ids) is not tuple
        or any(
            not isinstance(item, RetainedResource)
            for item in value.resources
        )
        or any(
            type(item) is not str or not item
            for item in value.unreadable_resource_ids
        )
        or value.resources != expected
        or value.unreadable_resource_ids
    ):
        raise Task12OrphanAuditError(
            "unknown, unreadable, missing, or billable resource remains"
        )
    return value


def _validate_grants(
    grants: object,
    *,
    label: str,
) -> Tuple[KmsGrantIdentity, ...]:
    if type(grants) is not tuple or any(
        not isinstance(item, KmsGrantIdentity) for item in grants
    ):
        raise Task12OrphanAuditError(label + " grants must be an exact tuple")
    ids = tuple(item.grant_id for item in grants)
    if len(ids) != len(set(ids)):
        raise Task12OrphanAuditError(label + " grants contain a duplicate")
    return grants


def _validate_grant_scan(
    value: object,
    *,
    label: str,
) -> KmsGrantScan:
    if not isinstance(value, KmsGrantScan):
        raise Task12OrphanAuditError(label + " ListGrants scan must be typed")
    _validate_grants(value.grants, label=label)
    if value.pagination_complete is not True:
        raise Task12OrphanAuditError(label + " ListGrants scan is not complete")
    _utc(value.observed_at, label + " ListGrants observed_at")
    _sha(value.list_grants_identity_sha256, label + " ListGrants identity")
    return value


def _validate_activation_grant(
    value: KmsGrantIdentity,
) -> None:
    if not set(value.operations).issubset(_ACTIVATION_GRANT_OPERATIONS):
        raise Task12OrphanAuditError(
            "activation KMS grant is unknown or overbroad"
        )


def _validate_direct(
    value: object,
) -> DirectGrantAttribution:
    if not isinstance(value, DirectGrantAttribution):
        raise Task12OrphanAuditError("direct grant attribution must be typed")
    _validate_activation_grant(value.grant)
    _sha(value.create_request_identity_sha256, "direct grant create request")
    if value.provenance == "DIRECT_RESPONSE":
        _sha(
            value.create_response_identity_sha256,
            "direct grant create response",
        )
        if value.reconciliation_list_grants_identity_sha256 is not None:
            raise Task12OrphanAuditError(
                "direct grant provenance contains foreign reconciliation"
            )
    elif value.provenance == "LIST_GRANTS_RECONCILIATION":
        if value.create_response_identity_sha256 is not None:
            raise Task12OrphanAuditError(
                "reconciled direct grant fabricates a create response"
            )
        _sha(
            value.reconciliation_list_grants_identity_sha256,
            "direct grant reconciliation ListGrants",
        )
    else:
        raise Task12OrphanAuditError(
            "direct grant provenance is unknown"
        )
    _sha(
        value.revoke_or_retire_identity_sha256,
        "direct grant revocation",
    )
    return value


def _validate_service(
    value: object,
) -> ServiceGrantAttribution:
    if not isinstance(value, ServiceGrantAttribution):
        raise Task12OrphanAuditError("service grant attribution must be typed")
    _validate_activation_grant(value.grant)
    try:
        _sha(value.baseline_diff_identity_sha256, "service grant baseline diff")
        _sha(value.list_grants_identity_sha256, "service grant ListGrants")
        _sha(
            value.cloudtrail_request_identity_sha256,
            "service grant CloudTrail request",
        )
        _sha(
            value.cloudtrail_response_identity_sha256,
            "service grant CloudTrail response",
        )
        _text(
            value.originating_resource_identity,
            "service grant originating resource",
        )
        if (
            type(value.originating_service) is not str
            or re.fullmatch(r"[a-z0-9-]+\.amazonaws\.com", value.originating_service)
            is None
        ):
            raise Task12OrphanAuditError(
                "service grant originating service is invalid"
            )
        _sha(
            value.encryption_context_correlation_identity_sha256,
            "service grant encryption-context correlation",
        )
        _sha(
            value.revoke_or_retire_identity_sha256,
            "service grant revocation",
        )
    except Task12OrphanAuditError as exc:
        raise Task12OrphanAuditError(
            "service attribution is incomplete"
        ) from exc
    return value


def audit_orphans(
    *,
    expected_retained: Tuple[RetainedResource, ...],
    inventory: ResourceInventoryScan,
    retained_grant_baseline: Tuple[KmsGrantIdentity, ...],
    pre_cleanup_grants: KmsGrantScan,
    final_grants: KmsGrantScan,
    direct_grants: Tuple[DirectGrantAttribution, ...],
    service_grants: Tuple[ServiceGrantAttribution, ...],
    settling_deadline: str,
) -> OrphanAuditProof:
    """Prove exact retained inventory and restore the frozen KMS baseline."""

    expected = _validate_expected_resources(expected_retained)
    inventory = _validate_inventory(inventory, expected=expected)
    baseline = _validate_grants(
        retained_grant_baseline,
        label="retained baseline",
    )
    pre = _validate_grant_scan(pre_cleanup_grants, label="pre-cleanup")
    final = _validate_grant_scan(final_grants, label="final")
    if type(direct_grants) is not tuple or type(service_grants) is not tuple:
        raise Task12OrphanAuditError("grant attribution arrays must be tuples")
    direct = tuple(_validate_direct(item) for item in direct_grants)
    service = tuple(_validate_service(item) for item in service_grants)
    attributed = tuple(item.grant for item in direct + service)
    attributed_ids = tuple(item.grant_id for item in attributed)
    baseline_ids = tuple(item.grant_id for item in baseline)
    if (
        len(attributed_ids) != len(set(attributed_ids))
        or set(attributed_ids) & set(baseline_ids)
    ):
        raise Task12OrphanAuditError(
            "KMS grant attribution is missing or duplicated"
        )
    expected_pre = set(baseline) | set(attributed)
    if set(pre.grants) != expected_pre:
        raise Task12OrphanAuditError(
            "activation KMS grant is unknown or overbroad"
        )
    deadline = _utc(settling_deadline, "KMS grant settling deadline")
    if _utc(final.observed_at, "final ListGrants observed_at") < deadline:
        raise Task12OrphanAuditError(
            "final KMS ListGrants scan predates settling deadline"
        )
    if set(final.grants) != set(baseline):
        raise Task12OrphanAuditError(
            "final KMS grants do not equal the frozen retained baseline"
        )
    baseline_identity = canonical_sha256(
        [asdict(item) for item in baseline]
    )
    body = {
        "retained_resources": [asdict(item) for item in expected],
        "retained_cost_classes": tuple(
            item.cost_class for item in expected
        ),
        "orphan_resource_ids": (),
        "retained_grant_baseline_identity_sha256": baseline_identity,
        "final_grants": [asdict(item) for item in final.grants],
        "final_list_grants_identity_sha256": (
            final.list_grants_identity_sha256
        ),
        "settling_deadline": settling_deadline,
        "observed_at": final.observed_at,
    }
    return OrphanAuditProof(
        retained_resources=expected,
        retained_cost_classes=tuple(
            item.cost_class for item in expected
        ),
        orphan_resource_ids=(),
        retained_grant_baseline_identity_sha256=baseline_identity,
        final_grants=final.grants,
        final_list_grants_identity_sha256=(
            final.list_grants_identity_sha256
        ),
        settling_deadline=settling_deadline,
        observed_at=final.observed_at,
        canonical_identity_sha256=canonical_sha256(body),
    )


@dataclass(frozen=True)
class MarkerLastPrerequisites:
    terminal_v2_identity_sha256: str
    finalization_identity_sha256: str
    snapshot_cleanup_control_identity_sha256: str
    controller_quiesced_identity_sha256: str
    spend_ledger_head_identity_sha256: str
    orphan_audit: OrphanAuditProof
    support_stack_absent: bool
    snapshot_cleanup_armed: bool
    all_workers_terminal: bool
    all_allocations_closed: bool
    liability_state: str
    liability_identity_sha256: Optional[str]


@dataclass(frozen=True)
class H1gDrainedPrerequisites:
    terminal_v2_identity_sha256: str
    finalization_identity_sha256: str
    snapshot_cleanup_control_identity_sha256: str
    controller_quiesced_identity_sha256: str
    spend_ledger_head_identity_sha256: str
    orphan_audit_identity_sha256: str
    liability_state: str
    liability_identity_sha256: str
    marker_write_order: str
    canonical_identity_sha256: str


def _validate_orphan_audit_proof(
    value: object,
) -> OrphanAuditProof:
    if not isinstance(value, OrphanAuditProof):
        raise Task12OrphanAuditError("exact orphan audit is absent")
    body = {
        "retained_resources": [
            asdict(item) for item in value.retained_resources
        ],
        "retained_cost_classes": value.retained_cost_classes,
        "orphan_resource_ids": value.orphan_resource_ids,
        "retained_grant_baseline_identity_sha256": (
            value.retained_grant_baseline_identity_sha256
        ),
        "final_grants": [asdict(item) for item in value.final_grants],
        "final_list_grants_identity_sha256": (
            value.final_list_grants_identity_sha256
        ),
        "settling_deadline": value.settling_deadline,
        "observed_at": value.observed_at,
    }
    if value.canonical_identity_sha256 != canonical_sha256(body):
        raise Task12OrphanAuditError("orphan audit identity drifted")
    return value


def build_h1g_drained_prerequisites(
    value: MarkerLastPrerequisites,
) -> H1gDrainedPrerequisites:
    """Validate all prior evidence; the H1G_DRAINED write remains last."""

    if not isinstance(value, MarkerLastPrerequisites):
        raise Task12OrphanAuditError("marker-last prerequisites must be typed")
    for field in (
        "terminal_v2_identity_sha256",
        "finalization_identity_sha256",
        "snapshot_cleanup_control_identity_sha256",
        "controller_quiesced_identity_sha256",
        "spend_ledger_head_identity_sha256",
    ):
        _sha(getattr(value, field), field)
    orphan_audit = _validate_orphan_audit_proof(value.orphan_audit)
    if (
        value.support_stack_absent is not True
        or value.all_workers_terminal is not True
        or value.all_allocations_closed is not True
    ):
        raise Task12OrphanAuditError(
            "support, worker, or allocation finalization is incomplete"
        )
    if value.snapshot_cleanup_armed is not True:
        raise Task12OrphanAuditError(
            "snapshot cleanup control is not armed"
        )
    if value.liability_state not in {
        "SETTLED",
        "RETAINED_TERMINATION_ONLY",
    } or value.liability_identity_sha256 is None:
        raise Task12OrphanAuditError(
            "liability is neither settled nor durably transferred"
        )
    liability = _sha(
        value.liability_identity_sha256,
        "liability identity",
    )
    body = {
        "terminal_v2_identity_sha256": value.terminal_v2_identity_sha256,
        "finalization_identity_sha256": value.finalization_identity_sha256,
        "snapshot_cleanup_control_identity_sha256": (
            value.snapshot_cleanup_control_identity_sha256
        ),
        "controller_quiesced_identity_sha256": (
            value.controller_quiesced_identity_sha256
        ),
        "spend_ledger_head_identity_sha256": (
            value.spend_ledger_head_identity_sha256
        ),
        "orphan_audit_identity_sha256": (
            orphan_audit.canonical_identity_sha256
        ),
        "liability_state": value.liability_state,
        "liability_identity_sha256": liability,
        "marker_write_order": "LAST_CONDITIONAL_CREATE",
    }
    return H1gDrainedPrerequisites(
        **body,
        canonical_identity_sha256=canonical_sha256(body),
    )


__all__ = [
    "ACCOUNT_ID",
    "AUDITED_RESOURCE_TYPES",
    "H1gDrainedPrerequisites",
    "INTENTIONAL_RETAINED_RESOURCE_TYPES",
    "KmsGrantIdentity",
    "KmsGrantScan",
    "MarkerLastPrerequisites",
    "OrphanAuditProof",
    "REGION",
    "RUN_ID",
    "ResourceInventoryScan",
    "RetainedResource",
    "ServiceGrantAttribution",
    "DirectGrantAttribution",
    "Task12OrphanAuditError",
    "audit_orphans",
    "build_h1g_drained_prerequisites",
]
