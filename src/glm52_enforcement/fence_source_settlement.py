"""Strict injected-I/O terminal settlement for the five H.1g source families.

This module owns no AWS client construction and no transition authority.  It
reconciles already-frozen source evidence, constructs the canonical Task-1
seal, and only then renders and singularly publishes the two late templates
and their manifest.  Every ineligible post-freeze observation selects the
bootstrap-rendered ``CLOSED_SOURCE`` entry without rendering or publishing it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import Enum
from types import MappingProxyType

from .canonical import canonical_json_bytes, canonical_sha256
from .fence_artifacts import (
    ArtifactCoordinate,
    FenceArtifactEntry,
    FenceArtifactError,
    FenceManifest,
    FenceSlot,
    ManifestStage,
    SourceSettlementSeal,
    SupportRuntimeIdentityCoordinate,
    build_fence_entry,
    build_fence_manifest,
    build_fence_template_bytes,
    build_source_settlement_seal,
    publish_fence_stage,
)
from .fence_policy_renderer import (
    FIXED_BUCKET_ARN,
    FULL_MUTATION,
    FencePolicyInput,
    PolicyComponentBudgetExhausted,
    PolicyHead,
    RenderedPolicy,
    render_fence_policy,
)
from .task13_fixed_artifacts import (
    ACCOUNT_ID,
    CAMPAIGN_BUCKET,
    Task13FixedArtifactError,
    Task13FixedArtifactServices,
)

_ACTIVATION_ID = "glm52-v2-amber-quartz"
_GENERATION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROLE_ARN = re.compile(r"arn:aws:iam::246813579024:role/[A-Za-z0-9+=,.@_/-]+\Z")
_VERSIONED_LAMBDA_ARN = re.compile(
    r"arn:aws:lambda:us-west-2:246813579024:function:"
    r"[A-Za-z0-9-_]+:[1-9][0-9]*\Z"
)
_STATE_MACHINE_VERSION_ARN = re.compile(
    r"arn:aws:states:us-west-2:246813579024:"
    r"stateMachine:[A-Za-z0-9-_]+:[1-9][0-9]*\Z"
)
_CHANGE_SET_ARN = re.compile(
    r"arn:aws:cloudformation:us-west-2:246813579024:"
    r"changeSet/[A-Za-z0-9][-A-Za-z0-9]*/[A-Za-z0-9-]+\Z"
)
_BOOTSTRAP_MANIFEST_KEY = (
    "campaigns/glm52-sky-20260724/authorities/fence/manifests/"
    "glm52-v2-amber-quartz/00000001/FENCE_BOOTSTRAP_MANIFEST.json"
)
_SOURCE_SETTLED_MANIFEST_KEY = (
    "campaigns/glm52-sky-20260724/authorities/fence/manifests/"
    "glm52-v2-amber-quartz/00000001/FENCE_SOURCE_SETTLED_MANIFEST.json"
)

SOURCE_FAMILY_ORDER: tuple[str, ...] = (
    "gpu-spend-snapshot",
    "production-submission-intent",
    "production-controller-baseline",
    "production-control-plane-readiness",
    "production-submission-acquisition",
)
SOURCE_FAMILY_PREFIXES: Mapping[str, str] = MappingProxyType(
    {
        "gpu-spend-snapshot": ("campaigns/glm52-sky-20260724/spend-snapshots/"),
        "production-submission-intent": (
            "campaigns/glm52-sky-20260724/submissions/production/intents/"
        ),
        "production-controller-baseline": (
            "campaigns/glm52-sky-20260724/production/controller-baselines/"
        ),
        "production-control-plane-readiness": (
            "campaigns/glm52-sky-20260724/monitor/must-start/production/"
        ),
        "production-submission-acquisition": (
            "campaigns/glm52-sky-20260724/submissions/production/acquisitions/"
        ),
    }
)


SOURCE_SETTLEMENT_EVIDENCE_READER_NAMES: tuple[str, ...] = (
    "read_freeze_execution_evidence",
    "read_selected_source_rows",
    "read_freeze_denial_probe_rows",
    "read_source_action_terminal_rows",
    "read_source_lambda_execution_terminal_rows",
    "read_workflow_post_source_state",
    "read_publisher_reachability_proof",
)
SOURCE_DENIAL_OPERATIONS: tuple[str, ...] = FULL_MUTATION
_SOURCE_KEY_PATTERNS: Mapping[str, re.Pattern[str]] = MappingProxyType(
    {
        "gpu-spend-snapshot": re.compile(
            re.escape(SOURCE_FAMILY_PREFIXES["gpu-spend-snapshot"])
            + r"[0-9a-f]{64}/GPU_SPEND_SNAPSHOT\.json\Z"
        ),
        "production-submission-intent": re.compile(
            re.escape(SOURCE_FAMILY_PREFIXES["production-submission-intent"])
            + r"[0-9a-f]{64}/SKYPILOT_SUBMISSION_INTENT\.json\Z"
        ),
        "production-controller-baseline": re.compile(
            re.escape(SOURCE_FAMILY_PREFIXES["production-controller-baseline"])
            + r"[0-9a-f]{64}/CONTROLLER_BASELINE\.json\Z"
        ),
        "production-control-plane-readiness": re.compile(
            re.escape(SOURCE_FAMILY_PREFIXES["production-control-plane-readiness"])
            + r"[0-9a-f]{64}/control-plane-ready/[0-9a-f]{64}/"
            r"CONTROL_PLANE_READY\.json\Z"
        ),
        "production-submission-acquisition": re.compile(
            re.escape(SOURCE_FAMILY_PREFIXES["production-submission-acquisition"])
            + r"[0-9a-f]{64}/SUBMISSION_ACQUIRED\.json\Z"
        ),
    }
)
_EXPECTED_FREEZE_SID = "DenyAllReservedFamilyMutation_SOURCE_FAMILIES_FROZEN"
_TERMINAL_STATUSES = frozenset(
    {
        "SUCCEEDED",
        "FAILED",
        "TIMED_OUT",
        "CANCELLED",
        "UNKNOWN_PRE_FREEZE_INVOCATION",
    }
)
_STATUS_CLASSIFICATIONS = {
    "FAILED": "SOURCE_ACTION_FAILED",
    "TIMED_OUT": "SOURCE_ACTION_TIMED_OUT",
    "CANCELLED": "SOURCE_ACTION_CANCELLED",
    "UNKNOWN_PRE_FREEZE_INVOCATION": "UNKNOWN_PRE_FREEZE_INVOCATION",
}

_FREEZE_FIELDS = frozenset(
    {
        "slot",
        "outcome",
        "bootstrap_manifest_identity_sha256",
        "freeze_entry_identity_sha256",
        "deployed_policy_sha256",
        "stack_id",
        "stack_role_arn",
        "change_set_arn",
        "execute_request_id",
        "executed_at",
        "canonical_identity_sha256",
    }
)
_PROBE_FIELDS = frozenset(
    {
        "round",
        "observed_at",
        "policy_sha256",
        "direct_policy_sha256",
        "attribution_rows",
        "canonical_identity_sha256",
    }
)
_ATTRIBUTION_FIELDS = frozenset(
    {
        "source_family",
        "expected_sid",
        "publisher_role_arn",
        "publisher_role_id",
        "existing_credentials_denied",
        "fresh_session_denied",
        "alternate_principal_denied",
        "denied_operations",
        "service_request_ids",
        "canonical_identity_sha256",
    }
)
_ACTION_FIELDS = frozenset(
    {
        "source_family",
        "action_identity_sha256",
        "invocation_identity_sha256",
        "terminal_status",
        "terminal_observed_at",
        "canonical_identity_sha256",
    }
)
_INVOCATION_FIELDS = frozenset(
    {
        "source_family",
        "action_identity_sha256",
        "invocation_identity_sha256",
        "function_version_arn",
        "executed_version",
        "lambda_request_id",
        "request_identity_sha256",
        "response_identity_sha256",
        "s3_request_id",
        "version_id",
        "terminal_status",
        "terminal_observed_at",
        "canonical_identity_sha256",
    }
)
_SELECTED_FIELDS = frozenset(
    {
        "source_family",
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
        "publisher_role_arn",
        "publisher_role_id",
        "s3_request_id",
        "canonical_identity_sha256",
    }
)
_WORKFLOW_FIELDS = frozenset(
    {
        "state_machine_version_arn",
        "post_source_state",
        "source_task_families",
        "reachable_source_task_families",
        "graph_sha256",
        "observed_at",
        "canonical_identity_sha256",
    }
)
_REACHABILITY_FIELDS = frozenset(
    {
        "source_families",
        "publisher_function_version_arns",
        "reachable_source_families",
        "closed",
        "no_launch",
        "observed_at",
        "canonical_identity_sha256",
    }
)
_MANIFEST_COMMON_FIELDS = (
    "account_id",
    "region",
    "run_id",
    "mutation_authority_inventory",
    "activation_id",
    "member_account_authority",
    "generation",
    "legacy_mode",
    "bucket_name",
    "bucket_arn",
    "bucket_control_plane",
    "kms_key_identity",
    "stack_name",
    "stack_id",
    "executor_inventory",
    "logical_id",
    "migration_service_role",
    "fence_service_role",
    "policy_limits",
    "render_input_identity_sha256",
    "legacy_fragment_sha256",
    "principal_inventory",
    "writer_inventory",
    "writer_policy_cohorts",
    "source_inventory",
)


class SourceSettlementError(ValueError):
    """The injected settlement boundary or one of its records is malformed."""


class SourceSettlementDisposition(str, Enum):
    """The only two post-freeze source dispositions."""

    SEALED = "SOURCE_PUBLICATION_SEALED"
    CLOSED_SOURCE = "CLOSED_SOURCE"


@dataclass(frozen=True)
class SourceSettlementRequest:
    """Authenticated fixed inputs; all mutable observations come from ports."""

    bootstrap_manifest: FenceManifest
    bootstrap_manifest_coordinate: ArtifactCoordinate
    support_runtime_identity_coordinate: SupportRuntimeIdentityCoordinate
    reservation_coordinate: ArtifactCoordinate
    freeze_entry_coordinate: ArtifactCoordinate
    batch_policy_input: FencePolicyInput
    terminal_policy_input: FencePolicyInput
    batch_successor_contract_sha256: str
    terminal_successor_contract_sha256: str
    kms_key_arn: str
    materializer_function_version_arn: str
    evidence_coordinates: Mapping[str, ArtifactCoordinate]

    def __post_init__(self) -> None:
        if type(self.bootstrap_manifest) is not FenceManifest:
            raise TypeError("bootstrap_manifest must be an authenticated FenceManifest")
        if type(self.bootstrap_manifest_coordinate) is not ArtifactCoordinate:
            raise TypeError("bootstrap_manifest_coordinate must be exact")
        if (
            type(self.support_runtime_identity_coordinate)
            is not SupportRuntimeIdentityCoordinate
        ):
            raise TypeError("support_runtime_identity_coordinate must be exact")
        if type(self.reservation_coordinate) is not ArtifactCoordinate:
            raise TypeError("reservation_coordinate must be exact")
        if type(self.freeze_entry_coordinate) is not ArtifactCoordinate:
            raise TypeError("freeze_entry_coordinate must be exact")
        if type(self.batch_policy_input) is not FencePolicyInput:
            raise TypeError("batch_policy_input must be exact")
        if type(self.terminal_policy_input) is not FencePolicyInput:
            raise TypeError("terminal_policy_input must be exact")
        _sha(self.batch_successor_contract_sha256, "batch successor contract")
        _sha(self.terminal_successor_contract_sha256, "terminal successor contract")
        if type(self.kms_key_arn) is not str or self.kms_key_arn == "":
            raise SourceSettlementError("kms_key_arn must be exact")
        if (
            _VERSIONED_LAMBDA_ARN.fullmatch(self.materializer_function_version_arn)
            is None
            or self.materializer_function_version_arn.split(":function:", 1)[1].rsplit(
                ":", 1
            )[0]
            != "keep-glm52-h1g-fence-source-settlement"
            or not isinstance(self.evidence_coordinates, Mapping)
            or set(self.evidence_coordinates)
            != set(SOURCE_SETTLEMENT_EVIDENCE_READER_NAMES)
        ):
            raise SourceSettlementError(
                "source settlement materializer/evidence authority is not exact"
            )
        evidence_prefix = (
            "campaigns/glm52-sky-20260724/authorities/fence/evidence/"
            f"{_ACTIVATION_ID}/00000001/"
        )
        for name, coordinate in self.evidence_coordinates.items():
            if (
                type(coordinate) is not ArtifactCoordinate
                or coordinate.bucket != CAMPAIGN_BUCKET
                or coordinate.key != evidence_prefix + name + ".json"
            ):
                raise SourceSettlementError(
                    "source settlement evidence coordinate is not exact"
                )
        object.__setattr__(
            self,
            "evidence_coordinates",
            MappingProxyType(
                {
                    name: self.evidence_coordinates[name]
                    for name in SOURCE_SETTLEMENT_EVIDENCE_READER_NAMES
                }
            ),
        )


_SOURCE_SETTLEMENT_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "bootstrap_manifest",
        "bootstrap_manifest_coordinate",
        "support_runtime_identity_coordinate",
        "reservation_coordinate",
        "freeze_entry_coordinate",
        "batch_policy_input",
        "terminal_policy_input",
        "batch_successor_contract_sha256",
        "materializer_function_version_arn",
        "evidence_coordinates",
        "terminal_successor_contract_sha256",
        "kms_key_arn",
    }
)


def source_settlement_request_projection(
    request: SourceSettlementRequest,
) -> dict[str, object]:
    from .fence_bootstrap_publication import (
        fence_policy_input_prototype_projection,
    )

    if type(request) is not SourceSettlementRequest:
        raise TypeError("source settlement request must be exact")
    return {
        "schema_version": 2,
        "record_type": "glm52_h1g_source_settlement_request_v2",
        "bootstrap_manifest": request.bootstrap_manifest.to_dict(),
        "bootstrap_manifest_coordinate": (
            request.bootstrap_manifest_coordinate.to_dict()
        ),
        "support_runtime_identity_coordinate": (
            request.support_runtime_identity_coordinate.to_dict()
        ),
        "reservation_coordinate": request.reservation_coordinate.to_dict(),
        "freeze_entry_coordinate": request.freeze_entry_coordinate.to_dict(),
        "batch_policy_input": fence_policy_input_prototype_projection(
            request.batch_policy_input
        ),
        "terminal_policy_input": fence_policy_input_prototype_projection(
            request.terminal_policy_input
        ),
        "batch_successor_contract_sha256": (request.batch_successor_contract_sha256),
        "terminal_successor_contract_sha256": (
            request.terminal_successor_contract_sha256
        ),
        "kms_key_arn": request.kms_key_arn,
        "materializer_function_version_arn": (
            request.materializer_function_version_arn
        ),
        "evidence_coordinates": {
            name: coordinate.to_dict()
            for name, coordinate in request.evidence_coordinates.items()
        },
    }


def parse_source_settlement_request_v2(
    value: object,
) -> SourceSettlementRequest:
    from .fence_artifacts import (
        parse_artifact_coordinate,
        parse_fence_manifest,
        parse_support_runtime_identity_coordinate,
    )
    from .fence_bootstrap_publication import (
        parse_fence_policy_input_prototype_projection,
    )

    if (
        type(value) is not dict
        or set(value) != _SOURCE_SETTLEMENT_REQUEST_FIELDS
        or value["schema_version"] != 2
        or value["record_type"] != "glm52_h1g_source_settlement_request_v2"
    ):
        raise SourceSettlementError("source settlement request fields are not exact v2")
    try:
        return SourceSettlementRequest(
            bootstrap_manifest=parse_fence_manifest(value["bootstrap_manifest"]),
            bootstrap_manifest_coordinate=parse_artifact_coordinate(
                value["bootstrap_manifest_coordinate"]
            ),
            support_runtime_identity_coordinate=(
                parse_support_runtime_identity_coordinate(
                    value["support_runtime_identity_coordinate"]
                )
            ),
            reservation_coordinate=parse_artifact_coordinate(
                value["reservation_coordinate"]
            ),
            freeze_entry_coordinate=parse_artifact_coordinate(
                value["freeze_entry_coordinate"]
            ),
            batch_policy_input=parse_fence_policy_input_prototype_projection(
                value["batch_policy_input"]
            ),
            terminal_policy_input=(
                parse_fence_policy_input_prototype_projection(
                    value["terminal_policy_input"]
                )
            ),
            batch_successor_contract_sha256=value["batch_successor_contract_sha256"],
            terminal_successor_contract_sha256=value[
                "terminal_successor_contract_sha256"
            ],
            materializer_function_version_arn=value[
                "materializer_function_version_arn"
            ],
            evidence_coordinates={
                name: parse_artifact_coordinate(coordinate)
                for name, coordinate in value["evidence_coordinates"].items()
            }
            if type(value["evidence_coordinates"]) is dict
            else {},
            kms_key_arn=value["kms_key_arn"],
        )
    except (TypeError, ValueError, FenceArtifactError) as exc:
        raise SourceSettlementError("source settlement request is malformed") from exc


@dataclass(frozen=True)
class SourceSettlementServices:
    """All mutable reads, clock access, waiting, and publication are injected."""

    evidence: object
    s3: object
    artifact_services: Task13FixedArtifactServices
    now_utc: Callable[[], datetime]
    sleep: Callable[[float], None]

    def __post_init__(self) -> None:
        if self.evidence is None or self.s3 is None or self.artifact_services is None:
            raise TypeError("source-settlement services must be present")
        if not callable(self.now_utc):
            raise TypeError("source-settlement clock must be callable")
        if not callable(self.sleep):
            raise TypeError("source-settlement sleep must be callable")


@dataclass(frozen=True)
class SourceSettlementResult:
    """Sealed late artifacts or an explicit bootstrap CLOSED_SOURCE choice."""

    disposition: SourceSettlementDisposition
    classification: str
    selected_entry: FenceArtifactEntry
    source_settlement: SourceSettlementSeal | None
    source_settled_manifest: FenceManifest | None
    publication_coordinates: tuple[ArtifactCoordinate, ...]
    no_launch_evidence: Mapping[str, object] | None


class _Ineligible(Exception):
    def __init__(
        self,
        classification: str,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.classification = classification
        self.details = dict(details or {})


@dataclass(frozen=True)
class _EvidenceSnapshot:
    freeze_execution: Mapping[str, object]
    probes: tuple[Mapping[str, object], ...]
    actions: tuple[Mapping[str, object], ...]
    invocations: tuple[Mapping[str, object], ...]
    workflow: Mapping[str, object]
    reachability: Mapping[str, object]
    selected: tuple[Mapping[str, object], ...]

    def canonical_value(self) -> Mapping[str, object]:
        return {
            "freeze_execution": self.freeze_execution,
            "probes": self.probes,
            "actions": self.actions,
            "invocations": self.invocations,
            "workflow": self.workflow,
            "reachability": self.reachability,
            "selected": self.selected,
        }


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise SourceSettlementError(label + " must be one lowercase SHA-256")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or value == "" or "\x00" in value:
        raise SourceSettlementError(label + " must be one exact nonempty string")
    return value


def _utc(value: object, label: str) -> datetime:
    if type(value) is datetime:
        parsed = value
    elif type(value) is str and value.endswith("Z"):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise SourceSettlementError(label + " is not canonical UTC") from exc
    else:
        raise SourceSettlementError(label + " is not canonical UTC")
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.microsecond != 0:
        raise SourceSettlementError(label + " must be a whole-second UTC time")
    return parsed.astimezone(UTC)


def _utc_text(value: object, label: str) -> str:
    return _utc(value, label).strftime("%Y-%m-%dT%H:%M:%SZ")


def _signed_mapping(
    value: object, fields: frozenset[str], label: str
) -> Mapping[str, object]:
    if type(value) is not dict or set(value) != set(fields):
        raise SourceSettlementError(label + " field set is not exact")
    identity = _sha(value["canonical_identity_sha256"], label + " identity")
    unsigned = dict(value)
    unsigned.pop("canonical_identity_sha256")
    if canonical_sha256(unsigned) != identity:
        raise SourceSettlementError(label + " canonical identity drifted")
    return dict(value)


def _exact_tuple(value: object, label: str) -> tuple:
    if type(value) is not tuple:
        raise SourceSettlementError(label + " must be one exact tuple")
    return value


def _method(value: object, name: str) -> Callable[[], object]:
    try:
        method = getattr(value, name)
    except Exception as exc:
        raise SourceSettlementError("evidence reader lacks " + name) from exc
    if not callable(method):
        raise SourceSettlementError("evidence reader lacks " + name)
    return method


def _read(value: object, name: str, classification: str) -> object:
    try:
        return _method(value, name)()
    except SourceSettlementError:
        raise
    except Exception as exc:
        raise _Ineligible(classification, name + " failed") from exc


def _validate_request_bindings(request: SourceSettlementRequest) -> None:
    manifest = request.bootstrap_manifest
    if manifest.manifest_stage is not ManifestStage.BOOTSTRAP:
        raise SourceSettlementError("source settlement requires BOOTSTRAP manifest")
    manifest_coordinate = request.bootstrap_manifest_coordinate
    if (
        manifest_coordinate.key != _BOOTSTRAP_MANIFEST_KEY
        or manifest_coordinate.canonical_identity_sha256
        != manifest.canonical_identity_sha256
    ):
        raise SourceSettlementError("bootstrap manifest coordinate identity drifted")
    for slot, coordinate in (
        (FenceSlot.RESERVATION_ONLY, request.reservation_coordinate),
        (FenceSlot.SOURCE_FAMILIES_FROZEN, request.freeze_entry_coordinate),
    ):
        entry = manifest.entry(slot).to_dict()
        if (
            coordinate.key != entry["template_key"]
            or coordinate.version_id != entry["version_id"]
            or coordinate.file_sha256 != entry["template_sha256"]
            or coordinate.canonical_identity_sha256 != entry["template_body_sha256"]
        ):
            raise SourceSettlementError(slot.value + " coordinate drifted")
    manifest_value = manifest.to_dict()
    if (
        type(manifest_value["kms_key_identity"]) is not dict
        or manifest_value["kms_key_identity"].get("key_arn") != request.kms_key_arn
    ):
        raise SourceSettlementError("KMS key identity drifted")
    for prototype in (
        request.batch_policy_input,
        request.terminal_policy_input,
    ):
        if (
            prototype.kms_key_arn != request.kms_key_arn
            or prototype.build_mode.value
            != "LEGACY_" + str(manifest_value["legacy_mode"])
        ):
            raise SourceSettlementError("late policy prototype binding drifted")
    if request.batch_policy_input.policy_head is not PolicyHead.BATCH:
        raise SourceSettlementError("batch_policy_input head is not BATCH")
    if request.terminal_policy_input.policy_head is not PolicyHead.TERMINAL:
        raise SourceSettlementError("terminal_policy_input head is not TERMINAL")


def _validate_freeze(
    value: object, *, request: SourceSettlementRequest
) -> Mapping[str, object]:
    row = _signed_mapping(value, _FREEZE_FIELDS, "freeze execution")
    freeze_entry = request.bootstrap_manifest.entry(FenceSlot.SOURCE_FAMILIES_FROZEN)
    manifest_value = request.bootstrap_manifest.to_dict()
    if (
        row["slot"] != FenceSlot.SOURCE_FAMILIES_FROZEN.value
        or row["outcome"] != "APPLIED"
        or row["bootstrap_manifest_identity_sha256"]
        != request.bootstrap_manifest.canonical_identity_sha256
        or row["freeze_entry_identity_sha256"] != freeze_entry.entry_identity_sha256
        or row["deployed_policy_sha256"] != freeze_entry.policy_sha256
        or row["stack_id"] != manifest_value["stack_id"]
        or row["stack_role_arn"] != manifest_value["fence_service_role"]["arn"]
    ):
        raise _Ineligible("FREEZE_DEPLOYMENT_UNPROVED", "freeze deployment drifted")
    change_set_arn = _text(row["change_set_arn"], "freeze change_set_arn")
    if _CHANGE_SET_ARN.fullmatch(change_set_arn) is None:
        raise _Ineligible("FREEZE_DEPLOYMENT_UNPROVED", "freeze change-set ARN drifted")
    _text(row["execute_request_id"], "freeze execute_request_id")
    _utc(row["executed_at"], "freeze executed_at")
    return row


def _validate_selected(value: object) -> tuple[Mapping[str, object], ...]:
    rows = _exact_tuple(value, "selected source rows")
    if len(rows) != len(SOURCE_FAMILY_ORDER):
        raise _Ineligible(
            "SOURCE_PUBLICATION_UNPROVED", "selected source cardinality drifted"
        )
    selected = tuple(
        _signed_mapping(row, _SELECTED_FIELDS, "selected source row") for row in rows
    )
    if tuple(row["source_family"] for row in selected) != SOURCE_FAMILY_ORDER:
        raise _Ineligible(
            "SOURCE_PUBLICATION_UNPROVED", "selected source order drifted"
        )
    keys = []
    requests = []
    roles = []
    for row in selected:
        family = str(row["source_family"])
        key = _text(row["key"], family + " key")
        if _SOURCE_KEY_PATTERNS[family].fullmatch(key) is None:
            raise _Ineligible(
                "SOURCE_PUBLICATION_UNPROVED", family + " selected key drifted"
            )
        _text(row["version_id"], family + " VersionId")
        _sha(row["file_sha256"], family + " file sha")
        _sha(row["body_sha256"], family + " body sha")
        arn = _text(row["publisher_role_arn"], family + " publisher ARN")
        role_id = _text(row["publisher_role_id"], family + " publisher RoleId")
        if _ROLE_ARN.fullmatch(arn) is None or ":" in role_id:
            raise _Ineligible(
                "SOURCE_PUBLICATION_UNPROVED",
                family + " publisher identity drifted",
            )
        _text(row["s3_request_id"], family + " S3 request ID")
        keys.append(key)
        requests.append(row["s3_request_id"])
        roles.append((arn, role_id))
    if any(len(set(values)) != len(values) for values in (keys, requests, roles)):
        raise _Ineligible(
            "SOURCE_PUBLICATION_UNPROVED",
            "selected source identity is not unique",
        )
    return selected


def _validate_probes(
    value: object,
    *,
    freeze: Mapping[str, object],
    selected: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    raw_rows = _exact_tuple(value, "freeze denial probe rows")
    if len(raw_rows) != 2:
        raise _Ineligible("FREEZE_DENIAL_UNPROVED", "two probes are required")
    probes = tuple(
        _signed_mapping(row, _PROBE_FIELDS, "freeze denial probe") for row in raw_rows
    )
    if tuple(row["round"] for row in probes) != (1, 2):
        raise _Ineligible("FREEZE_DENIAL_UNPROVED", "probe order drifted")
    observed = tuple(
        _utc(row["observed_at"], "freeze denial probe observed_at") for row in probes
    )
    if observed[1] - observed[0] < timedelta(seconds=10):
        raise _Ineligible("FREEZE_DENIAL_UNPROVED", "probe interval is too short")
    if observed[0] < _utc(freeze["executed_at"], "freeze executed_at"):
        raise _Ineligible(
            "FREEZE_DENIAL_UNPROVED",
            "denial probes predate freeze execution",
        )
    expected_bindings = tuple(
        (
            row["source_family"],
            row["publisher_role_arn"],
            row["publisher_role_id"],
        )
        for row in selected
    )
    all_request_ids = []
    for probe in probes:
        if (
            probe["policy_sha256"] != freeze["deployed_policy_sha256"]
            or probe["direct_policy_sha256"] != freeze["deployed_policy_sha256"]
        ):
            raise _Ineligible("FREEZE_DENIAL_UNPROVED", "probe policy drifted")
        raw_attributions = _exact_tuple(
            probe["attribution_rows"], "freeze probe attributions"
        )
        attributions = tuple(
            _signed_mapping(row, _ATTRIBUTION_FIELDS, "freeze probe attribution")
            for row in raw_attributions
        )
        observed_bindings = tuple(
            (
                row["source_family"],
                row["publisher_role_arn"],
                row["publisher_role_id"],
            )
            for row in attributions
        )
        if observed_bindings != expected_bindings:
            raise _Ineligible(
                "FREEZE_DENIAL_UNPROVED", "probe publisher attribution drifted"
            )
        for attribution in attributions:
            operations = _exact_tuple(
                attribution["denied_operations"], "denied operations"
            )
            request_ids = _exact_tuple(
                attribution["service_request_ids"], "denial request IDs"
            )
            if (
                attribution["expected_sid"] != _EXPECTED_FREEZE_SID
                or attribution["existing_credentials_denied"] is not True
                or attribution["fresh_session_denied"] is not True
                or attribution["alternate_principal_denied"] is not True
                or operations != SOURCE_DENIAL_OPERATIONS
                or len(request_ids) != len(SOURCE_DENIAL_OPERATIONS)
                or any(type(item) is not str or item == "" for item in request_ids)
                or len(set(request_ids)) != len(request_ids)
            ):
                raise _Ineligible(
                    "FREEZE_DENIAL_UNPROVED", "probe denial attribution is incomplete"
                )
            all_request_ids.extend(request_ids)
    if len(set(all_request_ids)) != len(all_request_ids):
        raise _Ineligible("FREEZE_DENIAL_UNPROVED", "denial request IDs are not unique")
    return probes


def _validate_terminal_rows(
    actions_value: object,
    invocations_value: object,
    *,
    selected: Sequence[Mapping[str, object]],
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    actions_raw = _exact_tuple(actions_value, "source action terminal rows")
    invocations_raw = _exact_tuple(invocations_value, "source Lambda terminal rows")
    if len(actions_raw) != 5 or len(invocations_raw) != 5:
        raise _Ineligible(
            "SOURCE_TERMINALITY_UNPROVED", "source terminal row cardinality drifted"
        )
    actions = tuple(
        _signed_mapping(row, _ACTION_FIELDS, "source action terminal row")
        for row in actions_raw
    )
    invocations = tuple(
        _signed_mapping(row, _INVOCATION_FIELDS, "source Lambda execution terminal row")
        for row in invocations_raw
    )
    if (
        tuple(row["source_family"] for row in actions) != SOURCE_FAMILY_ORDER
        or tuple(row["source_family"] for row in invocations) != SOURCE_FAMILY_ORDER
    ):
        raise _Ineligible(
            "SOURCE_TERMINALITY_UNPROVED", "source terminal order drifted"
        )
    for action, invocation, source in zip(actions, invocations, selected):
        status = action["terminal_status"]
        invocation_status = invocation["terminal_status"]
        if (
            status not in _TERMINAL_STATUSES
            or invocation_status not in _TERMINAL_STATUSES
        ):
            raise _Ineligible(
                "SOURCE_TERMINALITY_UNPROVED", "source status is not terminal"
            )
        for row, label in ((action, "action"), (invocation, "invocation")):
            _utc(row["terminal_observed_at"], label + " terminal_observed_at")
        if (
            action["action_identity_sha256"] != invocation["action_identity_sha256"]
            or action["invocation_identity_sha256"]
            != invocation["invocation_identity_sha256"]
        ):
            raise _Ineligible(
                "SOURCE_TERMINALITY_UNPROVED", "action/invocation binding drifted"
            )
        _sha(action["action_identity_sha256"], "action identity")
        _sha(action["invocation_identity_sha256"], "invocation identity")
        _sha(invocation["request_identity_sha256"], "Lambda request identity")
        version_arn = _text(
            invocation["function_version_arn"], "publisher function version ARN"
        )
        if (
            _VERSIONED_LAMBDA_ARN.fullmatch(version_arn) is None
            or version_arn.rsplit(":", 1)[-1] != invocation["executed_version"]
            or type(invocation["lambda_request_id"]) is not str
            or invocation["lambda_request_id"] == ""
        ):
            raise _Ineligible(
                "SOURCE_TERMINALITY_UNPROVED",
                "publisher invocation identity drifted",
            )
        if status != "SUCCEEDED" or invocation_status != "SUCCEEDED":
            if invocation["response_identity_sha256"] is not None:
                _sha(
                    invocation["response_identity_sha256"],
                    "Lambda response identity",
                )
            classification = _STATUS_CLASSIFICATIONS.get(
                "UNKNOWN_PRE_FREEZE_INVOCATION"
                if "UNKNOWN_PRE_FREEZE_INVOCATION" in (status, invocation_status)
                else (status if status != "SUCCEEDED" else invocation_status),
                "SOURCE_TERMINALITY_UNPROVED",
            )
            raise _Ineligible(
                classification,
                "source phase did not succeed",
                details={
                    "source_action_terminal_rows": actions,
                    "source_lambda_execution_terminal_rows": invocations,
                },
            )
        if (
            invocation["s3_request_id"] != source["s3_request_id"]
            or invocation["version_id"] != source["version_id"]
        ):
            raise _Ineligible(
                "SOURCE_PUBLICATION_UNPROVED",
                "publisher terminal response drifted",
            )
        _sha(invocation["response_identity_sha256"], "Lambda response identity")
    if (
        len({row["action_identity_sha256"] for row in actions}) != 5
        or len({row["invocation_identity_sha256"] for row in invocations}) != 5
    ):
        raise _Ineligible(
            "SOURCE_TERMINALITY_UNPROVED",
            "source action or invocation identity is not unique",
        )
    if len({row["lambda_request_id"] for row in invocations}) != 5:
        raise _Ineligible(
            "SOURCE_PUBLICATION_UNPROVED", "Lambda request IDs are not unique"
        )
    return actions, invocations


def _validate_workflow(value: object) -> Mapping[str, object]:
    row = _signed_mapping(value, _WORKFLOW_FIELDS, "workflow post-source state")
    if (
        type(row["state_machine_version_arn"]) is not str
        or _STATE_MACHINE_VERSION_ARN.fullmatch(row["state_machine_version_arn"])
        is None
        or row["post_source_state"] != "SOURCE_PHASE_CLOSED"
        or _exact_tuple(row["source_task_families"], "workflow source families")
        != SOURCE_FAMILY_ORDER
        or _exact_tuple(
            row["reachable_source_task_families"],
            "workflow reachable source families",
        )
        != ()
    ):
        raise _Ineligible(
            "WORKFLOW_REACHABILITY_UNPROVED", "workflow can reach a source task"
        )
    _sha(row["graph_sha256"], "workflow graph")
    _utc(row["observed_at"], "workflow observed_at")
    return row


def _validate_reachability(
    value: object, *, invocations: Sequence[Mapping[str, object]]
) -> Mapping[str, object]:
    if type(value) is not dict:
        raise _Ineligible(
            "PUBLISHER_REACHABILITY_UNPROVED",
            "legacy BatchSuccessor is not source authority",
        )
    row = _signed_mapping(value, _REACHABILITY_FIELDS, "publisher reachability proof")
    if (
        _exact_tuple(row["source_families"], "publisher source families")
        != SOURCE_FAMILY_ORDER
        or _exact_tuple(
            row["publisher_function_version_arns"],
            "publisher function version ARNs",
        )
        != tuple(item["function_version_arn"] for item in invocations)
        or _exact_tuple(
            row["reachable_source_families"],
            "publisher reachable source families",
        )
        != ()
        or row["closed"] is not True
        or row["no_launch"] is not True
    ):
        raise _Ineligible(
            "PUBLISHER_REACHABILITY_UNPROVED", "publisher path is still reachable"
        )
    _utc(row["observed_at"], "publisher reachability observed_at")
    return row


def _load_snapshot(
    request: SourceSettlementRequest, services: SourceSettlementServices
) -> _EvidenceSnapshot:
    evidence = services.evidence
    freeze = _validate_freeze(
        _read(
            evidence,
            "read_freeze_execution_evidence",
            "FREEZE_DEPLOYMENT_UNPROVED",
        ),
        request=request,
    )
    context: dict[str, object] = {
        "freeze_execution_evidence": freeze,
    }
    try:
        selected = _validate_selected(
            _read(
                evidence,
                "read_selected_source_rows",
                "SOURCE_PUBLICATION_UNPROVED",
            )
        )
        context["selected_source_rows"] = selected
        probes = _validate_probes(
            _read(
                evidence,
                "read_freeze_denial_probe_rows",
                "FREEZE_DENIAL_UNPROVED",
            ),
            freeze=freeze,
            selected=selected,
        )
        context["freeze_denial_probe_rows"] = probes
        actions, invocations = _validate_terminal_rows(
            _read(
                evidence,
                "read_source_action_terminal_rows",
                "SOURCE_TERMINALITY_UNPROVED",
            ),
            _read(
                evidence,
                "read_source_lambda_execution_terminal_rows",
                "SOURCE_TERMINALITY_UNPROVED",
            ),
            selected=selected,
        )
        context["source_action_terminal_rows"] = actions
        context["source_lambda_execution_terminal_rows"] = invocations
        workflow = _validate_workflow(
            _read(
                evidence,
                "read_workflow_post_source_state",
                "WORKFLOW_REACHABILITY_UNPROVED",
            )
        )
        context["workflow_post_source_state"] = workflow
        reachability = _validate_reachability(
            _read(
                evidence,
                "read_publisher_reachability_proof",
                "PUBLISHER_REACHABILITY_UNPROVED",
            ),
            invocations=invocations,
        )
        context["publisher_reachability_proof"] = reachability
    except _Ineligible as exc:
        raise _Ineligible(
            exc.classification,
            str(exc),
            details={**context, **dict(exc.details)},
        ) from exc
    except SourceSettlementError as exc:
        raise _Ineligible(
            "SOURCE_SETTLEMENT_DRIFT",
            str(exc),
            details=context,
        ) from exc
    return _EvidenceSnapshot(
        freeze_execution=freeze,
        probes=probes,
        actions=actions,
        invocations=invocations,
        workflow=workflow,
        reachability=reachability,
        selected=selected,
    )


def _s3_call(s3: object, request: Mapping[str, object]) -> Mapping[str, object]:
    try:
        method = s3.list_object_versions
    except Exception as exc:
        raise SourceSettlementError("S3 lacks list_object_versions") from exc
    if not callable(method):
        raise SourceSettlementError("S3 lacks list_object_versions")
    try:
        response = method(**dict(request))
    except Exception as exc:
        raise _Ineligible(
            "SOURCE_INVENTORY_UNPROVED", "ListObjectVersions failed"
        ) from exc
    if type(response) is not dict:
        raise _Ineligible(
            "SOURCE_INVENTORY_UNPROVED", "ListObjectVersions response is not exact"
        )
    return response


def _inventory_pass(
    services: SourceSettlementServices,
) -> tuple[tuple[Mapping[str, object], ...], Mapping[str, object]]:
    rows = []
    page_request_ids = []
    seen_inventory_rows = set()
    for family in SOURCE_FAMILY_ORDER:
        prefix = SOURCE_FAMILY_PREFIXES[family]
        markers: tuple[str, str] | None = None
        seen_markers = set()
        pages = 0
        while True:
            pages += 1
            if pages > 1_000:
                raise _Ineligible(
                    "SOURCE_INVENTORY_UNPROVED", "inventory page bound exhausted"
                )
            request = {
                "Bucket": CAMPAIGN_BUCKET,
                "Prefix": prefix,
                "MaxKeys": 1000,
                "ExpectedBucketOwner": ACCOUNT_ID,
            }
            if markers is not None:
                request["KeyMarker"] = markers[0]
                request["VersionIdMarker"] = markers[1]
            response = _s3_call(services.s3, request)
            metadata = response.get("ResponseMetadata")
            if (
                type(metadata) is not dict
                or metadata.get("HTTPStatusCode") != 200
                or type(metadata.get("RequestId")) is not str
                or metadata.get("RequestId") == ""
                or metadata.get("RetryAttempts") != 0
                or response.get("Name", CAMPAIGN_BUCKET) != CAMPAIGN_BUCKET
                or response.get("Prefix", prefix) != prefix
                or type(response.get("IsTruncated")) is not bool
            ):
                raise _Ineligible(
                    "SOURCE_INVENTORY_UNPROVED", "inventory transport drifted"
                )
            page_request_ids.append(metadata["RequestId"])
            for collection in ("Versions", "DeleteMarkers", "CommonPrefixes"):
                if collection in response and type(response[collection]) is not list:
                    raise _Ineligible(
                        "SOURCE_INVENTORY_UNPROVED",
                        "inventory collection is not exact",
                    )
            if response.get("CommonPrefixes", []):
                raise _Ineligible(
                    "SOURCE_INVENTORY_UNPROVED", "inventory returned a common prefix"
                )
            has_key = "KeyMarker" in response
            has_version = "VersionIdMarker" in response
            if has_key != has_version:
                raise _Ineligible(
                    "SOURCE_INVENTORY_UNPROVED", "inventory marker echo is partial"
                )
            expected_echo = ("", "") if markers is None else markers
            if (
                has_key
                and (
                    response["KeyMarker"],
                    response["VersionIdMarker"],
                )
                != expected_echo
            ):
                raise _Ineligible(
                    "SOURCE_INVENTORY_UNPROVED", "inventory marker echo drifted"
                )
            if markers is not None and not has_key:
                raise _Ineligible(
                    "SOURCE_INVENTORY_UNPROVED", "inventory omitted marker echo"
                )
            for collection, delete_marker in (
                ("Versions", False),
                ("DeleteMarkers", True),
            ):
                for raw in response.get(collection, []):
                    if type(raw) is not dict:
                        raise _Ineligible(
                            "SOURCE_INVENTORY_UNPROVED", "inventory row is not exact"
                        )
                    key = _text(raw.get("Key"), "inventory key")
                    version_id = _text(raw.get("VersionId"), "inventory VersionId")
                    if _SOURCE_KEY_PATTERNS[family].fullmatch(key) is None:
                        raise _Ineligible(
                            "SOURCE_INVENTORY_UNPROVED", "inventory key is foreign"
                        )
                    if type(raw.get("IsLatest")) is not bool:
                        raise _Ineligible(
                            "SOURCE_INVENTORY_UNPROVED", "IsLatest is not exact"
                        )
                    identity = (family, key, version_id, delete_marker)
                    if identity in seen_inventory_rows:
                        raise _Ineligible(
                            "SOURCE_INVENTORY_UNPROVED", "inventory row duplicated"
                        )
                    seen_inventory_rows.add(identity)
                    if delete_marker:
                        if "ETag" in raw or "Size" in raw:
                            raise _Ineligible(
                                "SOURCE_INVENTORY_UNPROVED",
                                "delete marker carries object fields",
                            )
                        etag = None
                        size = None
                    else:
                        etag = _text(raw.get("ETag"), "inventory ETag")
                        size = raw.get("Size")
                        if type(size) is not int or size < 0:
                            raise _Ineligible(
                                "SOURCE_INVENTORY_UNPROVED",
                                "inventory size is not exact",
                            )
                    rows.append(
                        {
                            "source_family": family,
                            "key": key,
                            "version_id": version_id,
                            "is_latest": raw["IsLatest"],
                            "is_delete_marker": delete_marker,
                            "etag": etag,
                            "size": size,
                            "last_modified": _utc_text(
                                raw.get("LastModified"),
                                "inventory LastModified",
                            ),
                        }
                    )
            next_key = "NextKeyMarker" in response
            next_version = "NextVersionIdMarker" in response
            if next_key != next_version:
                raise _Ineligible(
                    "SOURCE_INVENTORY_UNPROVED", "inventory next marker is partial"
                )
            if response["IsTruncated"]:
                if not next_key:
                    raise _Ineligible(
                        "SOURCE_INVENTORY_UNPROVED",
                        "truncated inventory lacks markers",
                    )
                pair = (
                    _text(response["NextKeyMarker"], "next key marker"),
                    _text(
                        response["NextVersionIdMarker"],
                        "next VersionId marker",
                    ),
                )
                if (
                    not pair[0].startswith(prefix)
                    or pair == markers
                    or pair in seen_markers
                ):
                    raise _Ineligible(
                        "SOURCE_INVENTORY_UNPROVED", "inventory marker cycled"
                    )
                seen_markers.add(pair)
                markers = pair
                continue
            if next_key and (
                response["NextKeyMarker"] != "" or response["NextVersionIdMarker"] != ""
            ):
                raise _Ineligible(
                    "SOURCE_INVENTORY_UNPROVED",
                    "terminal inventory has leftover markers",
                )
            break
    if len(set(page_request_ids)) != len(page_request_ids):
        raise _Ineligible(
            "SOURCE_INVENTORY_UNPROVED",
            "inventory page request IDs are not unique",
        )
    try:
        observed_at = _utc_text(services.now_utc(), "inventory observed_at")
    except SourceSettlementError:
        raise
    except Exception as exc:
        raise _Ineligible(
            "SOURCE_INVENTORY_UNPROVED", "inventory clock failed"
        ) from exc
    inventory = tuple(rows)
    identity = canonical_sha256(inventory)
    observation_body = {
        "observed_at": observed_at,
        "inventory_sha256": identity,
        "page_request_ids": tuple(page_request_ids),
    }
    observation = {
        **observation_body,
        "canonical_identity_sha256": canonical_sha256(observation_body),
    }
    return inventory, observation


def _validate_inventory_binding(
    inventory: Sequence[Mapping[str, object]],
    selected: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    selected_identities = {
        (row["source_family"], row["key"], row["version_id"]) for row in selected
    }
    inventory_objects = {
        (row["source_family"], row["key"], row["version_id"])
        for row in inventory
        if row["is_delete_marker"] is False
    }
    if not selected_identities.issubset(inventory_objects):
        raise _Ineligible(
            "SOURCE_INVENTORY_UNPROVED", "selected source is absent from inventory"
        )
    for source in selected:
        matches = tuple(
            row
            for row in inventory
            if (
                row["source_family"],
                row["key"],
                row["version_id"],
            )
            == (
                source["source_family"],
                source["key"],
                source["version_id"],
            )
            and row["is_delete_marker"] is False
        )
        if len(matches) != 1:
            raise _Ineligible(
                "SOURCE_INVENTORY_UNPROVED", "selected inventory row is not singular"
            )
    return tuple(
        row
        for row in inventory
        if (
            row["source_family"],
            row["key"],
            row["version_id"],
        )
        not in selected_identities
        or row["is_delete_marker"] is True
    )


def _manifest_fields(
    bootstrap: FenceManifest, *, render_identity: str
) -> dict[str, object]:
    value = bootstrap.to_dict()
    try:
        fields = {field: value[field] for field in _MANIFEST_COMMON_FIELDS}
    except KeyError as exc:
        raise SourceSettlementError(
            "bootstrap manifest common projection is incomplete"
        ) from exc
    fields["render_input_identity_sha256"] = render_identity
    return fields


def _bound_policy_inputs(
    request: SourceSettlementRequest,
    *,
    snapshot: _EvidenceSnapshot,
    seal: SourceSettlementSeal,
    inventory_sha256: str,
    nonselected: Sequence[Mapping[str, object]],
) -> tuple[FencePolicyInput, FencePolicyInput]:
    selected_keys = tuple(row["key"] for row in snapshot.selected)
    selected_resources = tuple(FIXED_BUCKET_ARN + "/" + key for key in selected_keys)
    selected_publishers = tuple(
        (
            row["publisher_role_arn"],
            row["publisher_role_id"],
        )
        for row in snapshot.selected
    )
    nonselected_keys = tuple(
        dict.fromkeys(
            str(row["key"]) for row in nonselected if row["key"] not in selected_keys
        )
    )
    for prototype, label in (
        (request.batch_policy_input, "BATCH"),
        (request.terminal_policy_input, "TERMINAL"),
    ):
        if (
            tuple(
                (publisher.arn, publisher.role_id)
                for publisher in prototype.retired_publishers[-5:]
            )
            != selected_publishers
            or tuple(prototype.retired_publisher_resources[-5:]) != selected_resources
            or prototype.selected_source_keys != selected_keys
            or prototype.nonselected_source_keys != nonselected_keys
        ):
            raise _Ineligible(
                "SOURCE_SETTLEMENT_DRIFT", label + " source projection drifted"
            )
    freeze_hash = str(snapshot.freeze_execution["deployed_policy_sha256"])
    probe_hash = canonical_sha256(snapshot.probes)
    seal_hash = seal.canonical_identity_sha256
    common = {
        "freeze_denial_evidence_sha256": probe_hash,
        "source_publication_sealed_sha256": seal_hash,
        "all_version_inventory_sha256": inventory_sha256,
        "source_settlement_sha256": seal_hash,
    }
    batch = replace(
        request.batch_policy_input,
        predecessor_policy_sha256=freeze_hash,
        **common,
    )
    terminal = replace(
        request.terminal_policy_input,
        predecessor_policy_sha256=None,
        **common,
    )
    return batch, terminal


def _closed_result(
    request: SourceSettlementRequest,
    *,
    classification: str,
    reason: str,
    failure_evidence: Mapping[str, object] | None = None,
) -> SourceSettlementResult:
    closed = request.bootstrap_manifest.entry(FenceSlot.CLOSED_SOURCE)
    body = {
        "schema_version": 1,
        "record_type": "glm52_h1g_source_settlement_no_launch_v1",
        "activation_id": _ACTIVATION_ID,
        "generation": _GENERATION,
        "classification": classification,
        "reason": reason,
        "bootstrap_manifest_coordinate": (
            request.bootstrap_manifest_coordinate.to_dict()
        ),
        "freeze_entry_coordinate": request.freeze_entry_coordinate.to_dict(),
        "selected_closed_source_entry_identity_sha256": (closed.entry_identity_sha256),
        "no_batch": True,
        "no_launch": True,
        "failure_evidence": (
            dict(failure_evidence) if failure_evidence is not None else None
        ),
    }
    evidence = MappingProxyType(
        {**body, "canonical_identity_sha256": canonical_sha256(body)}
    )
    return SourceSettlementResult(
        disposition=SourceSettlementDisposition.CLOSED_SOURCE,
        classification=classification,
        selected_entry=closed,
        source_settlement=None,
        source_settled_manifest=None,
        publication_coordinates=(),
        no_launch_evidence=evidence,
    )


def settle_source_publication(
    *,
    request: SourceSettlementRequest,
    services: SourceSettlementServices,
) -> SourceSettlementResult:
    """Reconcile, seal, render, and publish; otherwise choose CLOSED_SOURCE.

    Business ineligibility is an explicit result rather than an exception.  No
    code path renders or republishes ``CLOSED_SOURCE`` and no legacy direct
    ``BatchSuccessor`` result is accepted as authority.
    """

    if type(request) is not SourceSettlementRequest:
        raise TypeError("request must be exact SourceSettlementRequest")
    if type(services) is not SourceSettlementServices:
        raise TypeError("services must be exact SourceSettlementServices")
    observed_snapshot: _EvidenceSnapshot | None = None
    try:
        _validate_request_bindings(request)
        observed_snapshot = _load_snapshot(request, services)
        initial = observed_snapshot
        try:
            first_inventory, first_observation = _inventory_pass(services)
            services.sleep(10)
            second_inventory, second_observation = _inventory_pass(services)
        except SourceSettlementError as exc:
            raise _Ineligible(
                "SOURCE_INVENTORY_UNPROVED",
                "inventory evidence is malformed",
            ) from exc
        if (
            _utc(second_observation["observed_at"], "second inventory")
            - _utc(first_observation["observed_at"], "first inventory")
            < timedelta(seconds=10)
            or canonical_json_bytes(first_inventory)
            != canonical_json_bytes(second_inventory)
            or set(first_observation["page_request_ids"])
            & set(second_observation["page_request_ids"])
        ):
            raise _Ineligible(
                "SOURCE_INVENTORY_UNPROVED",
                "two inventory observations are not stable",
            )
        nonselected = _validate_inventory_binding(first_inventory, initial.selected)
        inventory_sha256 = canonical_sha256(first_inventory)
        if (
            first_observation["inventory_sha256"] != inventory_sha256
            or second_observation["inventory_sha256"] != inventory_sha256
        ):
            raise _Ineligible("SOURCE_INVENTORY_UNPROVED", "inventory digest drifted")
        try:
            seal = build_source_settlement_seal(
                {
                    "generation": _GENERATION,
                    "reservation_coordinate": request.reservation_coordinate.to_dict(),
                    "freeze_entry_coordinate": request.freeze_entry_coordinate.to_dict(),
                    "freeze_execution_evidence": initial.freeze_execution,
                    "freeze_denial_probe_rows": initial.probes,
                    "source_action_terminal_rows": initial.actions,
                    "source_lambda_execution_terminal_rows": initial.invocations,
                    "workflow_post_source_state_identity": initial.workflow[
                        "canonical_identity_sha256"
                    ],
                    "publisher_reachability_proof": initial.reachability,
                    "selected_source_rows": initial.selected,
                    "nonselected_provisional_rows": nonselected,
                    "complete_family_version_inventory": first_inventory,
                    "first_inventory_observation": first_observation,
                    "second_inventory_observation": second_observation,
                    "inventory_sha256": inventory_sha256,
                    "sealed_at": second_observation["observed_at"],
                }
            )
        except FenceArtifactError as exc:
            raise _Ineligible(
                "SOURCE_SETTLEMENT_DRIFT",
                "canonical source-settlement seal was rejected",
            ) from exc
        try:
            revalidated = _load_snapshot(request, services)
            post_seal_inventory, _ = _inventory_pass(services)
        except (SourceSettlementError, _Ineligible) as exc:
            raise _Ineligible(
                "SOURCE_SETTLEMENT_DRIFT", "post-seal evidence is not stable"
            ) from exc
        if canonical_json_bytes(initial.canonical_value()) != canonical_json_bytes(
            revalidated.canonical_value()
        ) or canonical_json_bytes(first_inventory) != canonical_json_bytes(
            post_seal_inventory
        ):
            raise _Ineligible("SOURCE_SETTLEMENT_DRIFT", "post-seal evidence drifted")
        batch_input, terminal_prototype = _bound_policy_inputs(
            request,
            snapshot=initial,
            seal=seal,
            inventory_sha256=inventory_sha256,
            nonselected=nonselected,
        )
        try:
            batch_rendered = render_fence_policy(batch_input)
            terminal_input = replace(
                terminal_prototype,
                predecessor_policy_sha256=batch_rendered.policy_sha256,
            )
            terminal_rendered = render_fence_policy(terminal_input)
        except PolicyComponentBudgetExhausted as exc:
            raise _Ineligible(
                "LATE_RENDER_UNPROVED", "late policy rendering failed"
            ) from exc
        if (
            type(batch_rendered) is not RenderedPolicy
            or type(terminal_rendered) is not RenderedPolicy
            or batch_rendered.policy_sha256
            == initial.freeze_execution["deployed_policy_sha256"]
            or terminal_rendered.policy_sha256 == batch_rendered.policy_sha256
        ):
            raise _Ineligible(
                "LATE_RENDER_UNPROVED", "late policy render is not monotonic"
            )
        try:
            batch_template_bytes = build_fence_template_bytes(batch_rendered)
            terminal_template_bytes = build_fence_template_bytes(terminal_rendered)
        except FenceArtifactError as exc:
            raise _Ineligible(
                "LATE_RENDER_UNPROVED",
                "late template serialization failed",
            ) from exc
        render_identity = canonical_sha256(
            (
                batch_input.render_input_identity_sha256,
                terminal_input.render_input_identity_sha256,
            )
        )
        source_manifest: FenceManifest | None = None

        def manifest_builder(
            coordinates: Mapping[FenceSlot, ArtifactCoordinate],
            seed_coordinate: ArtifactCoordinate | None,
        ) -> bytes:
            nonlocal source_manifest
            if seed_coordinate is not None or tuple(coordinates) != (
                FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION,
                FenceSlot.TERMINAL,
            ):
                raise SourceSettlementError("late publication coordinate order drifted")
            batch_entry = build_fence_entry(
                slot=FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION,
                rendered=batch_rendered,
                version_id=coordinates[
                    FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION
                ].version_id,
                render_input_identity_sha256=render_identity,
                stack_id=str(request.bootstrap_manifest.to_dict()["stack_id"]),
                migration_service_role_arn=str(
                    request.bootstrap_manifest.to_dict()["migration_service_role"][
                        "arn"
                    ]
                ),
                fence_service_role_arn=str(
                    request.bootstrap_manifest.to_dict()["fence_service_role"]["arn"]
                ),
                bridge_seed_policy_sha256=(
                    request.bootstrap_manifest.bridge_seed.policy_sha256
                ),
                expected_prestate_policy_sha256=str(
                    initial.freeze_execution["deployed_policy_sha256"]
                ),
                publisher_deny_policy_sha256=str(
                    batch_input.publisher_deny_policy_sha256
                ),
                batch_projection_contract={
                    "generation": _GENERATION,
                    "selected_source_rows": initial.selected,
                    "nonselected_inventory_sha256": canonical_sha256(nonselected),
                    "publisher_identities": tuple(
                        {
                            "arn": row["publisher_role_arn"],
                            "role_id": row["publisher_role_id"],
                        }
                        for row in initial.selected
                    ),
                },
                successor_contract_sha256=(request.batch_successor_contract_sha256),
                bootstrap_manifest_coordinate=(request.bootstrap_manifest_coordinate),
            )
            terminal_entry = build_fence_entry(
                slot=FenceSlot.TERMINAL,
                rendered=terminal_rendered,
                version_id=coordinates[FenceSlot.TERMINAL].version_id,
                render_input_identity_sha256=render_identity,
                stack_id=str(request.bootstrap_manifest.to_dict()["stack_id"]),
                migration_service_role_arn=str(
                    request.bootstrap_manifest.to_dict()["migration_service_role"][
                        "arn"
                    ]
                ),
                fence_service_role_arn=str(
                    request.bootstrap_manifest.to_dict()["fence_service_role"]["arn"]
                ),
                bridge_seed_policy_sha256=(
                    request.bootstrap_manifest.bridge_seed.policy_sha256
                ),
                expected_prestate_policy_sha256=batch_rendered.policy_sha256,
                publisher_deny_policy_sha256=str(
                    terminal_input.publisher_deny_policy_sha256
                ),
                batch_projection_contract=None,
                successor_contract_sha256=(request.terminal_successor_contract_sha256),
            )
            source_manifest = build_fence_manifest(
                stage=ManifestStage.SOURCE_SETTLED,
                fields=_manifest_fields(
                    request.bootstrap_manifest,
                    render_identity=render_identity,
                ),
                bridge_seed=request.bootstrap_manifest.bridge_seed,
                entries=(batch_entry, terminal_entry),
                bootstrap_manifest_coordinate=(request.bootstrap_manifest_coordinate),
                support_runtime_identity_coordinate=(
                    request.support_runtime_identity_coordinate
                ),
                source_settlement=seal,
            )
            return canonical_json_bytes(source_manifest.to_dict()) + b"\n"

        coordinates = publish_fence_stage(
            stage=ManifestStage.SOURCE_SETTLED,
            template_bytes={
                FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION: batch_template_bytes,
                FenceSlot.TERMINAL: terminal_template_bytes,
            },
            manifest_builder=manifest_builder,
            kms_key_arn=request.kms_key_arn,
            services=services.artifact_services,
        )
        if source_manifest is None:
            raise _Ineligible(
                "LATE_PUBLICATION_UNPROVED",
                "manifest callback was not invoked",
            )
        if type(coordinates) is not tuple or len(coordinates) != 3:
            raise _Ineligible(
                "LATE_PUBLICATION_UNPROVED",
                "late publication coordinate set drifted",
            )
        for coordinate, slot in zip(
            coordinates[:2],
            (
                FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION,
                FenceSlot.TERMINAL,
            ),
        ):
            if type(coordinate) is not ArtifactCoordinate:
                raise _Ineligible(
                    "LATE_PUBLICATION_UNPROVED",
                    "late template coordinate type drifted",
                )
            entry = source_manifest.entry(slot).to_dict()
            if (
                coordinate.key != entry["template_key"]
                or coordinate.version_id != entry["version_id"]
                or coordinate.file_sha256 != entry["template_sha256"]
                or coordinate.canonical_identity_sha256 != entry["template_body_sha256"]
            ):
                raise _Ineligible(
                    "LATE_PUBLICATION_UNPROVED",
                    slot.value + " published coordinate drifted",
                )
        manifest_coordinate = coordinates[2]
        if (
            type(manifest_coordinate) is not ArtifactCoordinate
            or manifest_coordinate.key != _SOURCE_SETTLED_MANIFEST_KEY
            or manifest_coordinate.canonical_identity_sha256
            != source_manifest.canonical_identity_sha256
        ):
            raise _Ineligible(
                "LATE_PUBLICATION_UNPROVED",
                "source-settled manifest coordinate drifted",
            )
        batch_entry = source_manifest.entry(FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION)
        return SourceSettlementResult(
            disposition=SourceSettlementDisposition.SEALED,
            classification="SOURCE_PUBLICATION_SEALED",
            selected_entry=batch_entry,
            source_settlement=seal,
            source_settled_manifest=source_manifest,
            publication_coordinates=coordinates,
            no_launch_evidence=None,
        )
    except _Ineligible as exc:
        return _closed_result(
            request,
            classification=exc.classification,
            reason=str(exc),
            failure_evidence=(
                exc.details
                if exc.details
                else (
                    observed_snapshot.canonical_value()
                    if observed_snapshot is not None
                    else None
                )
            ),
        )
    except SourceSettlementError as exc:
        return _closed_result(
            request,
            classification="SOURCE_SETTLEMENT_DRIFT",
            reason=str(exc),
            failure_evidence=(
                observed_snapshot.canonical_value()
                if observed_snapshot is not None
                else None
            ),
        )
    except (FenceArtifactError, Task13FixedArtifactError) as exc:
        return _closed_result(
            request,
            classification="LATE_PUBLICATION_UNPROVED",
            reason=type(exc).__name__,
            failure_evidence=(
                observed_snapshot.canonical_value()
                if observed_snapshot is not None
                else None
            ),
        )


__all__ = [
    "SOURCE_DENIAL_OPERATIONS",
    "SOURCE_FAMILY_ORDER",
    "SOURCE_FAMILY_PREFIXES",
    "SOURCE_SETTLEMENT_EVIDENCE_READER_NAMES",
    "SourceSettlementDisposition",
    "SourceSettlementError",
    "SourceSettlementRequest",
    "SourceSettlementResult",
    "SourceSettlementServices",
    "parse_source_settlement_request_v2",
    "settle_source_publication",
    "source_settlement_request_projection",
]
