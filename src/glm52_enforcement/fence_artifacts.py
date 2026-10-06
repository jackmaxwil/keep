"""Closed immutable H.1g fence artifacts and execution requests.

This module is pure apart from the explicitly named staged publication helper.
It owns canonical projections, not renderer policy construction or AWS clients.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote

from .canonical import canonical_json_bytes, canonical_sha256
from .fence_policy_renderer import RenderedPolicy, StatementLedgerEntry
from .task13_fixed_artifacts import (
    ACCOUNT_ID,
    CAMPAIGN_BUCKET,
    REGION,
    RUN_ID,
    Task13FixedArtifactServices,
    adopt_fixed_key_bytes,
    publish_fixed_key_bytes,
)


_GENERATION_TEXT = "00000001"
_ACTIVATION_ID = "glm52-v2-amber-quartz"
_PREFIX = "campaigns/%s/authorities/fence" % RUN_ID
_SEED_KEY = "%s/seeds/%s/%s/BRIDGE_SEED_POLICY.json" % (
    _PREFIX,
    _ACTIVATION_ID,
    _GENERATION_TEXT,
)
_TEMPLATE_PREFIX = "%s/templates/%s/%s" % (
    _PREFIX,
    _ACTIVATION_ID,
    _GENERATION_TEXT,
)
_MANIFEST_PREFIX = "%s/manifests/%s/%s" % (
    _PREFIX,
    _ACTIVATION_ID,
    _GENERATION_TEXT,
)


class FenceArtifactError(ValueError):
    """A fence artifact is not the one closed v2 record."""


class ManifestStage(str, Enum):
    """The two independently publishable manifest stages."""

    BOOTSTRAP = "BOOTSTRAP"
    SOURCE_SETTLED = "SOURCE_SETTLED"


class FenceSlot(str, Enum):
    """The six executable H.1g fence transitions."""

    PREPARE_GENESIS_LIVE_STATE = "PREPARE_GENESIS_LIVE_STATE"
    BATCH_FIVE_SOURCE_ACTIVATION = "BATCH_FIVE_SOURCE_ACTIVATION"
    RESERVATION_ONLY = "RESERVATION_ONLY"
    SOURCE_FAMILIES_FROZEN = "SOURCE_FAMILIES_FROZEN"
    CLOSED_SOURCE = "CLOSED_SOURCE"
    TERMINAL = "TERMINAL"


class ExecutorAuthorityClass(str, Enum):
    """The two non-interchangeable transition executor classes."""

    RETAINED_PRE_SUPPORT = "RETAINED_PRE_SUPPORT"
    SUPPORT_RUNTIME = "SUPPORT_RUNTIME"


_TEMPLATE_KEYS = {
    FenceSlot.PREPARE_GENESIS_LIVE_STATE: (
        _TEMPLATE_PREFIX + "/PREPARE_GENESIS_LIVE_STATE.json"
    ),
    FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION: (
        _TEMPLATE_PREFIX + "/BATCH_FIVE_SOURCE_ACTIVATION.json"
    ),
    FenceSlot.RESERVATION_ONLY: _TEMPLATE_PREFIX + "/RESERVATION_ONLY.json",
    FenceSlot.SOURCE_FAMILIES_FROZEN: (
        _TEMPLATE_PREFIX + "/SOURCE_FAMILIES_FROZEN.json"
    ),
    FenceSlot.CLOSED_SOURCE: _TEMPLATE_PREFIX + "/CLOSED_SOURCE.json",
    FenceSlot.TERMINAL: _TEMPLATE_PREFIX + "/TERMINAL.json",
}
_BOOTSTRAP_MANIFEST_KEY = _MANIFEST_PREFIX + "/FENCE_BOOTSTRAP_MANIFEST.json"
_SOURCE_SETTLED_MANIFEST_KEY = (
    _MANIFEST_PREFIX + "/FENCE_SOURCE_SETTLED_MANIFEST.json"
)

FENCE_ARTIFACT_KEYS = (
    _SEED_KEY,
    _TEMPLATE_KEYS[FenceSlot.PREPARE_GENESIS_LIVE_STATE],
    _TEMPLATE_KEYS[FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION],
    _TEMPLATE_KEYS[FenceSlot.RESERVATION_ONLY],
    _TEMPLATE_KEYS[FenceSlot.SOURCE_FAMILIES_FROZEN],
    _TEMPLATE_KEYS[FenceSlot.CLOSED_SOURCE],
    _TEMPLATE_KEYS[FenceSlot.TERMINAL],
    _BOOTSTRAP_MANIFEST_KEY,
    _SOURCE_SETTLED_MANIFEST_KEY,
)

BOOTSTRAP_ENTRY_ORDER = (
    FenceSlot.PREPARE_GENESIS_LIVE_STATE.value,
    FenceSlot.RESERVATION_ONLY.value,
    FenceSlot.CLOSED_SOURCE.value,
    FenceSlot.SOURCE_FAMILIES_FROZEN.value,
)
SOURCE_SETTLED_ENTRY_ORDER = (
    FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION.value,
    FenceSlot.TERMINAL.value,
)
BOOTSTRAP_PUBLICATION_ORDER = (
    _SEED_KEY,
    _TEMPLATE_KEYS[FenceSlot.PREPARE_GENESIS_LIVE_STATE],
    _TEMPLATE_KEYS[FenceSlot.RESERVATION_ONLY],
    _TEMPLATE_KEYS[FenceSlot.SOURCE_FAMILIES_FROZEN],
    _TEMPLATE_KEYS[FenceSlot.CLOSED_SOURCE],
    _BOOTSTRAP_MANIFEST_KEY,
)
SOURCE_SETTLED_PUBLICATION_ORDER = (
    _TEMPLATE_KEYS[FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION],
    _TEMPLATE_KEYS[FenceSlot.TERMINAL],
    _SOURCE_SETTLED_MANIFEST_KEY,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STACK_ID = re.compile(
    r"arn:aws:cloudformation:%s:%s:stack/keep-glm52-h1g-fence/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
    % (REGION, ACCOUNT_ID)
)
_ROLE_ARN = re.compile(
    r"arn:aws:iam::%s:role/[A-Za-z0-9+=,.@_-]+"
    r"(?:/[A-Za-z0-9+=,.@_-]+)*\Z" % ACCOUNT_ID
)
_CHANGE_SET_NAME = re.compile(r"[A-Za-z][-A-Za-z0-9]*\Z")

_ENTRY_FIELDS = frozenset(
    {
        "slot",
        "transition_class",
        "allowed_predecessor_heads",
        "terminal_head",
        "template_key",
        "template_url",
        "version_id",
        "template_sha256",
        "template_body_sha256",
        "policy_sha256",
        "expected_prestate_policy_sha256",
        "expected_prestate_stack_role_arn",
        "expected_poststate_stack_role_arn",
        "publisher_deny_policy_sha256",
        "rendered_policy_bytes",
        "entry_identity_sha256",
        "statement_ledger",
        "change_set_name",
        "request_skeleton",
        "request_skeleton_sha256",
        "batch_projection_contract",
        "successor_contract_sha256",
        "allowed_create_authority_classes",
        "allowed_execute_authority_classes",
    }
)
_REQUEST_SKELETON_FIELDS = frozenset(
    {
        "StackName",
        "ChangeSetName",
        "ChangeSetType",
        "TemplateURL",
        "RoleARN",
        "ResourceTypes",
        "IncludeNestedStacks",
        "ImportExistingResources",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "manifest_stage",
        "account_id",
        "region",
        "run_id",
        "mutation_authority_inventory",
        "mutation_authority_inventory_sha256",
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
        "executor_inventory_sha256",
        "support_runtime_identity_coordinate",
        "logical_id",
        "migration_service_role",
        "fence_service_role",
        "policy_limits",
        "render_input_identity_sha256",
        "legacy_fragment_sha256",
        "bridge_seed",
        "transition_graph",
        "principal_inventory",
        "writer_inventory",
        "writer_inventory_sha256",
        "writer_policy_cohorts",
        "writer_policy_cohorts_sha256",
        "source_inventory",
        "source_inventory_sha256",
        "bootstrap_manifest_coordinate",
        "source_settlement",
        "entries",
        "canonical_identity_sha256",
    }
)
_MANIFEST_COMMON_INPUT_FIELDS = frozenset(
    {
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
    }
)
_SOURCE_SEAL_FIELDS = frozenset(
    {
        "generation",
        "reservation_coordinate",
        "freeze_entry_coordinate",
        "freeze_execution_evidence",
        "freeze_denial_probe_rows",
        "source_action_terminal_rows",
        "source_lambda_execution_terminal_rows",
        "workflow_post_source_state_identity",
        "publisher_reachability_proof",
        "selected_source_rows",
        "nonselected_provisional_rows",
        "complete_family_version_inventory",
        "first_inventory_observation",
        "second_inventory_observation",
        "inventory_sha256",
        "sealed_at",
        "canonical_identity_sha256",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "manifest_coordinate",
        "manifest_stage",
        "selected_entry_identity_sha256",
        "slot",
        "allowed_create_authority_classes",
        "allowed_execute_authority_classes",
        "support_runtime_identity_coordinate",
        "predecessor_head",
        "observed_h1f_active_head_coordinate",
        "h1f_successor_coordinate",
        "batch_projection",
        "source_settled_manifest_coordinate",
        "request_skeleton_sha256",
        "canonical_identity_sha256",
    }
)
_COORDINATE_FIELDS = frozenset(
    {
        "bucket",
        "key",
        "version_id",
        "file_sha256",
        "canonical_identity_sha256",
    }
)
_SOURCE_MANIFEST_COORDINATE_FIELDS = frozenset(
    set(_COORDINATE_FIELDS) | {"source_settlement_identity_sha256"}
)
_BRIDGE_SEED_FIELDS = frozenset(
    {
        "artifact_kind",
        "record_type",
        "bucket",
        "expected_live_preseed_policy_sha256",
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
        "policy_sha256",
        "rendered_policy_bytes",
        "statement_ledger",
        "publisher_deny_policy_sha256",
    }
)

_PREDECESSORS = {
    FenceSlot.RESERVATION_ONLY: FenceSlot.PREPARE_GENESIS_LIVE_STATE,
    FenceSlot.SOURCE_FAMILIES_FROZEN: FenceSlot.RESERVATION_ONLY,
    FenceSlot.CLOSED_SOURCE: FenceSlot.SOURCE_FAMILIES_FROZEN,
    FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION: (
        FenceSlot.SOURCE_FAMILIES_FROZEN
    ),
    FenceSlot.TERMINAL: FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION,
}
_AUTHORITIES = {
    FenceSlot.PREPARE_GENESIS_LIVE_STATE: (
        (ExecutorAuthorityClass.RETAINED_PRE_SUPPORT.value,),
        (ExecutorAuthorityClass.RETAINED_PRE_SUPPORT.value,),
    ),
    FenceSlot.RESERVATION_ONLY: (
        (ExecutorAuthorityClass.SUPPORT_RUNTIME.value,),
        (ExecutorAuthorityClass.SUPPORT_RUNTIME.value,),
    ),
    FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION: (
        (ExecutorAuthorityClass.SUPPORT_RUNTIME.value,),
        (ExecutorAuthorityClass.SUPPORT_RUNTIME.value,),
    ),
    FenceSlot.TERMINAL: (
        (ExecutorAuthorityClass.SUPPORT_RUNTIME.value,),
        (ExecutorAuthorityClass.SUPPORT_RUNTIME.value,),
    ),
    FenceSlot.SOURCE_FAMILIES_FROZEN: (
        (ExecutorAuthorityClass.SUPPORT_RUNTIME.value,),
        (
            ExecutorAuthorityClass.SUPPORT_RUNTIME.value,
            ExecutorAuthorityClass.RETAINED_PRE_SUPPORT.value,
        ),
    ),
    FenceSlot.CLOSED_SOURCE: (
        (
            ExecutorAuthorityClass.SUPPORT_RUNTIME.value,
            ExecutorAuthorityClass.RETAINED_PRE_SUPPORT.value,
        ),
        (
            ExecutorAuthorityClass.SUPPORT_RUNTIME.value,
            ExecutorAuthorityClass.RETAINED_PRE_SUPPORT.value,
        ),
    ),
}


def _fail(message: str) -> None:
    raise FenceArtifactError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(label + " must be one lowercase SHA-256")
    return value


def _require_text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(ord(character) < 0x20 for character in value)
    ):
        _fail(label + " must be one nonempty exact string")
    return value


def _require_version(value: object, label: str = "version_id") -> str:
    text = _require_text(value, label)
    if text == "null" or len(text.encode("utf-8")) > 1024:
        _fail(label + " must be one opaque service VersionId")
    return text


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if type(value) is not dict:
        _fail(label + " must be one exact object")
    return value


def _require_exact_fields(
    value: object, fields: frozenset, label: str
) -> Mapping[str, object]:
    mapping = _require_mapping(value, label)
    if frozenset(mapping) != fields or any(type(key) is not str for key in mapping):
        _fail(label + " field set is not exact")
    return mapping


def _freeze(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if type(value) in (list, tuple):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, int, bool):
        return value
    _fail("artifact projection contains a non-canonical value")


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw(item) for item in value]
    return value


def _detached_mapping(value: object, label: str) -> Mapping[str, object]:
    mapping = _require_mapping(value, label)
    return _freeze(json.loads(canonical_json_bytes(mapping)))  # type: ignore[return-value]


def _detached_sequence(value: object, label: str) -> Tuple[object, ...]:
    if type(value) not in (list, tuple):
        _fail(label + " must be one exact array")
    return _freeze(json.loads(canonical_json_bytes(value)))  # type: ignore[return-value]


def _slot(value: object) -> FenceSlot:
    if type(value) is not str:
        _fail("slot is not one closed enum value")
    try:
        return FenceSlot(value)
    except ValueError:
        _fail("slot is not one closed enum value")
    raise AssertionError("unreachable")


def _stage(value: object) -> ManifestStage:
    if type(value) is not str:
        _fail("manifest_stage is not one closed enum value")
    try:
        return ManifestStage(value)
    except ValueError:
        _fail("manifest_stage is not one closed enum value")
    raise AssertionError("unreachable")


def _identity_projection(value: Mapping[str, object], label: str) -> None:
    unsigned = dict(value)
    identity = unsigned.pop("canonical_identity_sha256", None)
    if _require_sha(identity, label + ".canonical_identity_sha256") != canonical_sha256(
        unsigned
    ):
        _fail(label + " canonical identity drifted")


def _ledger_projection(
    ledger: Sequence[StatementLedgerEntry],
) -> Tuple[Mapping[str, object], ...]:
    rows = []
    for entry in ledger:
        if type(entry) is not StatementLedgerEntry:
            _fail("renderer statement ledger contains a foreign row")
        rows.append(
            {
                "sequence": entry.sequence,
                "sid": entry.sid,
                "component": entry.component,
                "leading_comma_bytes": entry.leading_comma_bytes,
                "statement_bytes": entry.statement_bytes,
                "start_offset": entry.start_offset,
                "end_offset": entry.end_offset,
                "statement_sha256": entry.statement_sha256,
            }
        )
    return tuple(rows)


def _validate_ledger(value: object) -> Tuple[Mapping[str, object], ...]:
    rows = _detached_sequence(value, "statement_ledger")
    expected_fields = {
        "sequence",
        "sid",
        "component",
        "leading_comma_bytes",
        "statement_bytes",
        "start_offset",
        "end_offset",
        "statement_sha256",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            _fail("statement_ledger row field set is not exact")
        if row["sequence"] != index:
            _fail("statement_ledger order is not exact")
        _require_sha(row["statement_sha256"], "statement_ledger hash")
    return rows  # type: ignore[return-value]


@dataclass(frozen=True)
class ArtifactCoordinate:
    """One immutable canonical artifact coordinate."""

    bucket: str
    key: str
    version_id: str
    file_sha256: str
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        if self.bucket != CAMPAIGN_BUCKET:
            _fail("artifact coordinate bucket is foreign")
        key = _require_text(self.key, "artifact coordinate key")
        if not key.startswith("campaigns/%s/" % RUN_ID) or "\\" in key:
            _fail("artifact coordinate key is foreign")
        _require_version(self.version_id)
        _require_sha(self.file_sha256, "artifact coordinate file_sha256")
        _require_sha(
            self.canonical_identity_sha256,
            "artifact coordinate canonical_identity_sha256",
        )

    def to_dict(self) -> dict:
        return {
            "bucket": self.bucket,
            "key": self.key,
            "version_id": self.version_id,
            "file_sha256": self.file_sha256,
            "canonical_identity_sha256": self.canonical_identity_sha256,
        }


@dataclass(frozen=True)
class SupportRuntimeIdentityCoordinate(ArtifactCoordinate):
    """The singular post-operation-7 support-runtime identity coordinate."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if "/authorities/fence/runtime/" not in self.key:
            _fail("support_runtime_identity_coordinate key is foreign")


@dataclass(frozen=True)
class SourceSettledManifestCoordinate(ArtifactCoordinate):
    """A source-settled manifest coordinate bound to its seal identity."""

    source_settlement_identity_sha256: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.key != _SOURCE_SETTLED_MANIFEST_KEY:
            _fail("source-settled manifest coordinate key is foreign")
        _require_sha(
            self.source_settlement_identity_sha256,
            "source_settlement_identity_sha256",
        )

    @property
    def artifact_coordinate(self) -> ArtifactCoordinate:
        return ArtifactCoordinate(
            bucket=self.bucket,
            key=self.key,
            version_id=self.version_id,
            file_sha256=self.file_sha256,
            canonical_identity_sha256=self.canonical_identity_sha256,
        )

    def to_dict(self) -> dict:
        value = super().to_dict()
        value["source_settlement_identity_sha256"] = (
            self.source_settlement_identity_sha256
        )
        return value


@dataclass(frozen=True)
class SourceSettlementSeal:
    """The closed canonical source-publication seal."""

    _value: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_value",
            _validate_source_settlement_seal_value(self._value),
        )

    @property
    def canonical_identity_sha256(self) -> str:
        return str(self._value["canonical_identity_sha256"])

    def to_dict(self) -> dict:
        return _thaw(self._value)  # type: ignore[return-value]


@dataclass(frozen=True)
class BridgeSeedArtifact:
    """The raw no-LF bridge-seed policy and manifest projection."""

    _value: Mapping[str, object]
    raw_bytes: bytes

    def __post_init__(self) -> None:
        value, raw = _validate_bridge_seed_artifact_value(
            self._value,
            self.raw_bytes,
        )
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "raw_bytes", raw)

    @property
    def policy_sha256(self) -> str:
        return str(self._value["policy_sha256"])

    @property
    def version_id(self) -> str:
        return str(self._value["version_id"])

    def to_dict(self) -> dict:
        return _thaw(self._value)  # type: ignore[return-value]


@dataclass(frozen=True)
class FenceArtifactEntry:
    """One authenticated template coordinate and transition contract."""

    _value: Mapping[str, object]
    template_bytes: bytes = b""

    def __post_init__(self) -> None:
        value, raw = _validate_fence_entry_value(
            self._value,
            self.template_bytes,
        )
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "template_bytes", raw)

    @property
    def slot(self) -> FenceSlot:
        return FenceSlot(str(self._value["slot"]))

    @property
    def entry_identity_sha256(self) -> str:
        return str(self._value["entry_identity_sha256"])

    @property
    def policy_sha256(self) -> str:
        return str(self._value["policy_sha256"])

    @property
    def request_skeleton_sha256(self) -> str:
        return str(self._value["request_skeleton_sha256"])

    @property
    def allowed_create_authority_classes(self) -> Tuple[str, ...]:
        return tuple(self._value["allowed_create_authority_classes"])  # type: ignore[arg-type]

    @property
    def allowed_execute_authority_classes(self) -> Tuple[str, ...]:
        return tuple(self._value["allowed_execute_authority_classes"])  # type: ignore[arg-type]

    @property
    def allowed_predecessor_heads(self) -> Tuple[Mapping[str, object], ...]:
        return tuple(self._value["allowed_predecessor_heads"])  # type: ignore[arg-type,return-value]

    @property
    def batch_projection_contract(self) -> Optional[Mapping[str, object]]:
        value = self._value["batch_projection_contract"]
        return value if isinstance(value, Mapping) else None

    @property
    def request_skeleton(self) -> Mapping[str, object]:
        return self._value["request_skeleton"]  # type: ignore[return-value]

    def to_dict(self) -> dict:
        return _thaw(self._value)  # type: ignore[return-value]


@dataclass(frozen=True)
class FenceManifest:
    """One closed self-hashed manifest-v2 stage."""

    _value: Mapping[str, object]
    entries: Tuple[FenceArtifactEntry, ...]
    bridge_seed: BridgeSeedArtifact
    source_settlement: Optional[SourceSettlementSeal]
    support_runtime_identity_coordinate: Optional[SupportRuntimeIdentityCoordinate]
    bootstrap_manifest_coordinate: Optional[ArtifactCoordinate]

    def __post_init__(self) -> None:
        (
            value,
            expected_entries,
            expected_bridge_seed,
            expected_source_settlement,
            expected_runtime_coordinate,
            expected_bootstrap_coordinate,
        ) = _validate_fence_manifest_value(self._value)
        if (
            type(self.entries) is not tuple
            or any(type(entry) is not FenceArtifactEntry for entry in self.entries)
            or tuple(entry.to_dict() for entry in self.entries)
            != tuple(entry.to_dict() for entry in expected_entries)
        ):
            _fail("manifest entries do not match _value")
        if (
            type(self.bridge_seed) is not BridgeSeedArtifact
            or self.bridge_seed.to_dict() != expected_bridge_seed.to_dict()
        ):
            _fail("manifest bridge_seed does not match _value")
        if (
            (self.source_settlement is None)
            != (expected_source_settlement is None)
            or self.source_settlement is not None
            and (
                type(self.source_settlement) is not SourceSettlementSeal
                or self.source_settlement.to_dict()
                != expected_source_settlement.to_dict()  # type: ignore[union-attr]
            )
        ):
            _fail("manifest source_settlement does not match _value")
        if (
            (self.support_runtime_identity_coordinate is None)
            != (expected_runtime_coordinate is None)
            or self.support_runtime_identity_coordinate is not None
            and (
                type(self.support_runtime_identity_coordinate)
                is not SupportRuntimeIdentityCoordinate
                or self.support_runtime_identity_coordinate.to_dict()
                != expected_runtime_coordinate.to_dict()  # type: ignore[union-attr]
            )
        ):
            _fail(
                "manifest support_runtime_identity_coordinate does not match _value"
            )
        if (
            (self.bootstrap_manifest_coordinate is None)
            != (expected_bootstrap_coordinate is None)
            or self.bootstrap_manifest_coordinate is not None
            and (
                type(self.bootstrap_manifest_coordinate) is not ArtifactCoordinate
                or self.bootstrap_manifest_coordinate.to_dict()
                != expected_bootstrap_coordinate.to_dict()  # type: ignore[union-attr]
            )
        ):
            _fail("manifest bootstrap_manifest_coordinate does not match _value")
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "entries", expected_entries)
        object.__setattr__(self, "bridge_seed", expected_bridge_seed)
        object.__setattr__(
            self,
            "source_settlement",
            expected_source_settlement,
        )
        object.__setattr__(
            self,
            "support_runtime_identity_coordinate",
            expected_runtime_coordinate,
        )
        object.__setattr__(
            self,
            "bootstrap_manifest_coordinate",
            expected_bootstrap_coordinate,
        )

    @property
    def manifest_stage(self) -> ManifestStage:
        return ManifestStage(str(self._value["manifest_stage"]))

    @property
    def canonical_identity_sha256(self) -> str:
        return str(self._value["canonical_identity_sha256"])

    def entry(self, slot: FenceSlot) -> FenceArtifactEntry:
        if type(slot) is not FenceSlot:
            _fail("manifest entry lookup requires one FenceSlot")
        matches = tuple(entry for entry in self.entries if entry.slot is slot)
        if len(matches) != 1:
            _fail("manifest entry lookup is not singular")
        return matches[0]

    def to_dict(self) -> dict:
        return _thaw(self._value)  # type: ignore[return-value]


@dataclass(frozen=True)
class FenceTransitionRequest:
    """One immutable manifest-derived transition request-v2."""

    _value: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_value",
            _validate_fence_transition_request_value(self._value),
        )

    @property
    def canonical_identity_sha256(self) -> str:
        return str(self._value["canonical_identity_sha256"])

    @property
    def slot(self) -> FenceSlot:
        return FenceSlot(str(self._value["slot"]))

    def to_dict(self) -> dict:
        return _thaw(self._value)  # type: ignore[return-value]


def parse_artifact_coordinate(value: object) -> ArtifactCoordinate:
    mapping = _require_exact_fields(value, _COORDINATE_FIELDS, "artifact coordinate")
    return ArtifactCoordinate(
        bucket=mapping["bucket"],  # type: ignore[arg-type]
        key=mapping["key"],  # type: ignore[arg-type]
        version_id=mapping["version_id"],  # type: ignore[arg-type]
        file_sha256=mapping["file_sha256"],  # type: ignore[arg-type]
        canonical_identity_sha256=mapping["canonical_identity_sha256"],  # type: ignore[arg-type]
    )


def parse_support_runtime_identity_coordinate(
    value: object,
) -> SupportRuntimeIdentityCoordinate:
    coordinate = parse_artifact_coordinate(value)
    return SupportRuntimeIdentityCoordinate(**coordinate.to_dict())


def parse_source_settled_manifest_coordinate(
    value: object,
) -> SourceSettledManifestCoordinate:
    mapping = _require_exact_fields(
        value,
        _SOURCE_MANIFEST_COORDINATE_FIELDS,
        "source_settled_manifest_coordinate",
    )
    return SourceSettledManifestCoordinate(
        bucket=mapping["bucket"],  # type: ignore[arg-type]
        key=mapping["key"],  # type: ignore[arg-type]
        version_id=mapping["version_id"],  # type: ignore[arg-type]
        file_sha256=mapping["file_sha256"],  # type: ignore[arg-type]
        canonical_identity_sha256=mapping["canonical_identity_sha256"],  # type: ignore[arg-type]
        source_settlement_identity_sha256=mapping[
            "source_settlement_identity_sha256"
        ],  # type: ignore[arg-type]
    )


def build_source_settlement_seal(fields: Mapping[str, object]) -> SourceSettlementSeal:
    if type(fields) is not dict or set(fields) != set(_SOURCE_SEAL_FIELDS) - {
        "canonical_identity_sha256"
    }:
        _fail("source settlement seal input field set is not exact")
    value = json.loads(canonical_json_bytes(fields))
    value["canonical_identity_sha256"] = canonical_sha256(value)
    return parse_source_settlement_seal(value)


def _validate_source_settlement_seal_value(
    value: object,
) -> Mapping[str, object]:
    mapping = _require_exact_fields(value, _SOURCE_SEAL_FIELDS, "source_settlement")
    if mapping["generation"] != 1:
        _fail("source_settlement generation is not exact")
    for field in (
        "freeze_denial_probe_rows",
        "source_action_terminal_rows",
        "source_lambda_execution_terminal_rows",
        "selected_source_rows",
        "nonselected_provisional_rows",
        "complete_family_version_inventory",
    ):
        _detached_sequence(mapping[field], "source_settlement." + field)
    if len(mapping["freeze_denial_probe_rows"]) != 2:  # type: ignore[arg-type]
        _fail("source_settlement denial probes require two ordered rounds")
    if len(mapping["selected_source_rows"]) != 5:  # type: ignore[arg-type]
        _fail("source_settlement requires five selected source rows")
    inventory_sha256 = _require_sha(
        mapping["inventory_sha256"], "source_settlement.inventory_sha256"
    )
    inventory = mapping["complete_family_version_inventory"]
    if canonical_sha256(inventory) != inventory_sha256:
        _fail("source_settlement inventory hash drifted")

    observation_fields = frozenset(
        {
            "observed_at",
            "inventory_sha256",
            "page_request_ids",
            "canonical_identity_sha256",
        }
    )

    def validate_observation(
        raw: object, label: str
    ) -> tuple[datetime, tuple[str, ...], str]:
        observation = _require_exact_fields(raw, observation_fields, label)
        _identity_projection(observation, label)
        if (
            _require_sha(
                observation["inventory_sha256"], label + ".inventory_sha256"
            )
            != inventory_sha256
        ):
            _fail(label + " inventory hash drifted")

        observed_at = observation["observed_at"]
        if (
            type(observed_at) is not str
            or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", observed_at)
            is None
        ):
            _fail(label + ".observed_at is not canonical UTC")
        try:
            observed_time = datetime.fromisoformat(observed_at)
        except ValueError as exc:
            raise FenceArtifactError(
                label + ".observed_at is not canonical UTC"
            ) from exc

        raw_request_ids = observation["page_request_ids"]
        if type(raw_request_ids) not in (list, tuple) or not raw_request_ids:
            _fail(label + ".page_request_ids must be one nonempty exact array")
        request_ids = tuple(raw_request_ids)
        if any(
            type(request_id) is not str or request_id == ""
            for request_id in request_ids
        ):
            _fail(label + ".page_request_ids contains an invalid request ID")
        if len(set(request_ids)) != len(request_ids):
            _fail(label + ".page_request_ids contains duplicate request IDs")
        identity = _require_sha(
            observation["canonical_identity_sha256"],
            label + ".canonical_identity_sha256",
        )
        return observed_time, request_ids, identity

    first_time, first_request_ids, first_identity = validate_observation(
        mapping["first_inventory_observation"],
        "source_settlement.first_inventory_observation",
    )
    second_time, second_request_ids, second_identity = validate_observation(
        mapping["second_inventory_observation"],
        "source_settlement.second_inventory_observation",
    )
    if first_time >= second_time:
        _fail("source_settlement inventory observations are not time ordered")
    if set(first_request_ids) & set(second_request_ids):
        _fail("source_settlement inventory observation request IDs overlap")
    if first_identity == second_identity:
        _fail("source_settlement inventory observation identities are not distinct")
    _identity_projection(mapping, "source_settlement")
    return _detached_mapping(dict(mapping), "source_settlement")


def parse_source_settlement_seal(value: object) -> SourceSettlementSeal:
    return SourceSettlementSeal(_value=value)  # type: ignore[arg-type]


def build_bridge_seed_artifact(
    *,
    rendered: RenderedPolicy,
    expected_live_preseed_policy_sha256: str,
    version_id: str,
) -> BridgeSeedArtifact:
    if type(rendered) is not RenderedPolicy:
        _fail("bridge seed must consume one RenderedPolicy")
    _require_sha(
        expected_live_preseed_policy_sha256,
        "expected_live_preseed_policy_sha256",
    )
    _require_version(version_id)
    digest = _sha(rendered.policy_bytes)
    if digest != rendered.policy_sha256:
        _fail("bridge seed renderer hash drifted")
    value = {
        "artifact_kind": "BRIDGE_SEED_POLICY",
        "record_type": "glm52_h1g_bridge_seed_policy_v1",
        "bucket": CAMPAIGN_BUCKET,
        "expected_live_preseed_policy_sha256": (
            expected_live_preseed_policy_sha256
        ),
        "key": _SEED_KEY,
        "version_id": version_id,
        "file_sha256": digest,
        "body_sha256": digest,
        "policy_sha256": digest,
        "rendered_policy_bytes": len(rendered.policy_bytes),
        "statement_ledger": _ledger_projection(rendered.statement_ledger),
        "publisher_deny_policy_sha256": None,
    }
    return parse_bridge_seed_artifact(value, raw_bytes=rendered.policy_bytes)


def _validate_bridge_seed_artifact_value(
    value: object,
    raw_bytes: Optional[bytes] = None,
) -> Tuple[Mapping[str, object], bytes]:
    mapping = _require_exact_fields(value, _BRIDGE_SEED_FIELDS, "bridge_seed")
    if (
        mapping["artifact_kind"] != "BRIDGE_SEED_POLICY"
        or mapping["record_type"] != "glm52_h1g_bridge_seed_policy_v1"
        or mapping["bucket"] != CAMPAIGN_BUCKET
        or mapping["key"] != _SEED_KEY
        or mapping["publisher_deny_policy_sha256"] is not None
    ):
        _fail("bridge_seed discriminants are not exact")
    _require_version(mapping["version_id"])
    expected = _require_sha(mapping["file_sha256"], "bridge_seed.file_sha256")
    if (
        _require_sha(mapping["body_sha256"], "bridge_seed.body_sha256") != expected
        or _require_sha(mapping["policy_sha256"], "bridge_seed.policy_sha256")
        != expected
        or type(mapping["rendered_policy_bytes"]) is not int
        or mapping["rendered_policy_bytes"] <= 0
    ):
        _fail("bridge_seed raw-policy hashes are not equal")
    _require_sha(
        mapping["expected_live_preseed_policy_sha256"],
        "bridge_seed.expected_live_preseed_policy_sha256",
    )
    _validate_ledger(mapping["statement_ledger"])
    raw = b"" if raw_bytes is None else raw_bytes
    if type(raw) is not bytes:
        _fail("bridge_seed raw bytes are not immutable")
    if raw and (_sha(raw) != expected or len(raw) != mapping["rendered_policy_bytes"]):
        _fail("bridge_seed raw bytes drifted")
    return _detached_mapping(dict(mapping), "bridge_seed"), raw


def parse_bridge_seed_artifact(
    value: object, *, raw_bytes: Optional[bytes] = None
) -> BridgeSeedArtifact:
    return BridgeSeedArtifact(
        _value=value,  # type: ignore[arg-type]
        raw_bytes=b"" if raw_bytes is None else raw_bytes,
    )


def versioned_artifact_url(key: str, version_id: str) -> str:
    """Return the sole accepted virtual-hosted version-pinned S3 URL."""

    if key not in _TEMPLATE_KEYS.values():
        _fail("template URL key is not one exact artifact key")
    version = _require_version(version_id)
    return (
        "https://%s.s3.%s.amazonaws.com/%s?versionId=%s"
        % (
            CAMPAIGN_BUCKET,
            REGION,
            quote(key, safe="/", encoding="utf-8", errors="strict"),
            quote(version, safe="", encoding="utf-8", errors="strict"),
        )
    )


def _template_bytes(rendered: RenderedPolicy) -> bytes:
    if type(rendered) is not RenderedPolicy:
        _fail("template builder must consume one RenderedPolicy")
    try:
        policy = json.loads(rendered.policy_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FenceArtifactError("renderer policy bytes are not canonical JSON") from exc
    if type(policy) is not dict or canonical_json_bytes(policy) != rendered.policy_bytes:
        _fail("renderer policy bytes are not one canonical object")
    if (
        set(policy) != {"Version", "Statement"}
        or policy["Version"] != "2012-10-17"
        or type(policy["Statement"]) is not list
    ):
        _fail("renderer PolicyDocument shape is not exact")
    _reject_template_intrinsics(policy)
    template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "H1gProductionFenceBucketPolicy": {
                "DeletionPolicy": "Retain",
                "Properties": {
                    "Bucket": CAMPAIGN_BUCKET,
                    "PolicyDocument": policy,
                },
                "Type": "AWS::S3::BucketPolicy",
                "UpdateReplacePolicy": "Retain",
            }
        },
    }
    return canonical_json_bytes(template) + b"\n"


def build_fence_template_bytes(rendered: RenderedPolicy) -> bytes:
    """Serialize one renderer policy as the exact immutable CFN template."""

    return _template_bytes(rendered)


def _predecessor_projection(
    slot: FenceSlot,
    bridge_seed_policy_sha256: str,
    expected_prestate_policy_sha256: str,
    bootstrap_manifest_coordinate: Optional[ArtifactCoordinate],
) -> dict:
    if slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE:
        if expected_prestate_policy_sha256 != bridge_seed_policy_sha256:
            _fail("PREPARE predecessor is not the bridge seed")
        return {
            "kind": "BRIDGE_SEED",
            "policy_sha256": bridge_seed_policy_sha256,
        }
    predecessor = {
        "kind": "FENCE_SLOT",
        "slot": _PREDECESSORS[slot].value,
        "policy_sha256": expected_prestate_policy_sha256,
    }
    if slot is FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION:
        if bootstrap_manifest_coordinate is None:
            _fail("BATCH predecessor requires bootstrap manifest coordinate")
        if bootstrap_manifest_coordinate.key != _BOOTSTRAP_MANIFEST_KEY:
            _fail("BATCH predecessor bootstrap coordinate is foreign")
        predecessor["manifest_coordinate"] = (
            bootstrap_manifest_coordinate.to_dict()
        )
    elif bootstrap_manifest_coordinate is not None:
        _fail("only BATCH accepts a cross-manifest predecessor coordinate")
    return predecessor


def build_fence_entry(
    *,
    slot: FenceSlot,
    rendered: RenderedPolicy,
    version_id: str,
    render_input_identity_sha256: str,
    stack_id: str,
    migration_service_role_arn: str,
    fence_service_role_arn: str,
    bridge_seed_policy_sha256: str,
    expected_prestate_policy_sha256: str,
    publisher_deny_policy_sha256: str,
    batch_projection_contract: Optional[Mapping[str, object]],
    successor_contract_sha256: Optional[str],
    bootstrap_manifest_coordinate: Optional[ArtifactCoordinate] = None,
) -> FenceArtifactEntry:
    if type(slot) is not FenceSlot or type(rendered) is not RenderedPolicy:
        _fail("entry requires one closed slot and RenderedPolicy")
    version = _require_version(version_id)
    render_identity = _require_sha(
        render_input_identity_sha256, "render_input_identity_sha256"
    )
    seed_hash = _require_sha(
        bridge_seed_policy_sha256, "bridge_seed_policy_sha256"
    )
    prestate_hash = _require_sha(
        expected_prestate_policy_sha256,
        "expected_prestate_policy_sha256",
    )
    publisher_hash = _require_sha(
        publisher_deny_policy_sha256, "publisher_deny_policy_sha256"
    )
    if type(stack_id) is not str or _STACK_ID.fullmatch(stack_id) is None:
        _fail("StackName is not the service-assigned fence stack ID")
    if (
        type(migration_service_role_arn) is not str
        or _ROLE_ARN.fullmatch(migration_service_role_arn) is None
        or type(fence_service_role_arn) is not str
        or _ROLE_ARN.fullmatch(fence_service_role_arn) is None
        or migration_service_role_arn == fence_service_role_arn
    ):
        _fail("entry service-role identities are not exact")
    if slot is FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION:
        if type(batch_projection_contract) is not dict:
            _fail("BATCH requires one batch_projection_contract")
    elif batch_projection_contract is not None:
        _fail("batch_projection_contract is non-null only for BATCH")
    if slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE:
        if successor_contract_sha256 is not None:
            _fail("PREPARE successor_contract_sha256 must be null")
    else:
        _require_sha(successor_contract_sha256, "successor_contract_sha256")

    template_raw = _template_bytes(rendered)
    key = _TEMPLATE_KEYS[slot]
    url = versioned_artifact_url(key, version)
    change_set_name = "keep-glm52-h1g-%s-%s-%s" % (
        _GENERATION_TEXT,
        slot.value.lower().replace("_", "-"),
        render_identity[:16],
    )
    predecessor = _predecessor_projection(
        slot, seed_hash, prestate_hash, bootstrap_manifest_coordinate
    )
    expected_pre_role = (
        migration_service_role_arn
        if slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE
        else fence_service_role_arn
    )
    request_skeleton = {
        "StackName": stack_id,
        "ChangeSetName": change_set_name,
        "ChangeSetType": "UPDATE",
        "TemplateURL": url,
        "RoleARN": fence_service_role_arn,
        "ResourceTypes": ["AWS::S3::BucketPolicy"],
        "IncludeNestedStacks": False,
        "ImportExistingResources": False,
    }
    create_classes, execute_classes = _AUTHORITIES[slot]
    value = {
        "slot": slot.value,
        "transition_class": slot.value,
        "allowed_predecessor_heads": [predecessor],
        "terminal_head": {
            "kind": "FENCE_SLOT",
            "slot": slot.value,
            "policy_sha256": rendered.policy_sha256,
        },
        "template_key": key,
        "template_url": url,
        "version_id": version,
        "template_sha256": _sha(template_raw),
        "template_body_sha256": _sha(template_raw[:-1]),
        "policy_sha256": rendered.policy_sha256,
        "expected_prestate_policy_sha256": prestate_hash,
        "expected_prestate_stack_role_arn": expected_pre_role,
        "expected_poststate_stack_role_arn": fence_service_role_arn,
        "publisher_deny_policy_sha256": publisher_hash,
        "rendered_policy_bytes": rendered.rendered_policy_bytes,
        "entry_identity_sha256": "",
        "statement_ledger": _ledger_projection(rendered.statement_ledger),
        "change_set_name": change_set_name,
        "request_skeleton": request_skeleton,
        "request_skeleton_sha256": canonical_sha256(request_skeleton),
        "batch_projection_contract": batch_projection_contract,
        "successor_contract_sha256": successor_contract_sha256,
        "allowed_create_authority_classes": create_classes,
        "allowed_execute_authority_classes": execute_classes,
    }
    unsigned = dict(value)
    unsigned.pop("entry_identity_sha256")
    value["entry_identity_sha256"] = canonical_sha256(unsigned)
    return parse_fence_entry(value, template_bytes=template_raw)


def _validate_fence_entry_value(
    value: object,
    template_bytes: bytes = b"",
) -> Tuple[Mapping[str, object], bytes]:
    mapping = _require_exact_fields(value, _ENTRY_FIELDS, "manifest entry")
    slot = _slot(mapping["slot"])
    if mapping["transition_class"] != slot.value:
        _fail("entry transition_class does not match slot")
    key = mapping["template_key"]
    if key != _TEMPLATE_KEYS[slot]:
        _fail("entry template_key does not match slot")
    version = _require_version(mapping["version_id"])
    if mapping["template_url"] != versioned_artifact_url(key, version):  # type: ignore[arg-type]
        _fail("entry template_url is not the exact versioned URL")
    for field in (
        "template_sha256",
        "template_body_sha256",
        "policy_sha256",
        "expected_prestate_policy_sha256",
        "publisher_deny_policy_sha256",
        "request_skeleton_sha256",
        "entry_identity_sha256",
    ):
        _require_sha(mapping[field], "entry." + field)
    successor = mapping["successor_contract_sha256"]
    if slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE:
        if successor is not None:
            _fail("PREPARE successor_contract_sha256 must be null")
    else:
        _require_sha(successor, "entry.successor_contract_sha256")
    if (
        type(mapping["rendered_policy_bytes"]) is not int
        or not 0 < mapping["rendered_policy_bytes"] <= 17_920
    ):
        _fail("entry rendered_policy_bytes is outside the design limit")
    _validate_ledger(mapping["statement_ledger"])
    create_classes, execute_classes = _AUTHORITIES[slot]
    if tuple(mapping["allowed_create_authority_classes"]) != create_classes or tuple(  # type: ignore[arg-type]
        mapping["allowed_execute_authority_classes"]  # type: ignore[arg-type]
    ) != execute_classes:
        _fail("entry authority arrays do not match slot")
    predecessors = mapping["allowed_predecessor_heads"]
    if type(predecessors) not in (list, tuple) or len(predecessors) != 1:
        _fail("entry allowed_predecessor_heads is not singular")
    predecessor = predecessors[0]
    if type(predecessor) is not dict:
        _fail("entry predecessor is not a closed tagged union")
    prestate_hash = mapping["expected_prestate_policy_sha256"]
    if slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE:
        if set(predecessor) != {"kind", "policy_sha256"} or predecessor != {
            "kind": "BRIDGE_SEED",
            "policy_sha256": prestate_hash,
        }:
            _fail("PREPARE predecessor graph is not exact")
    else:
        expected_fields = {"kind", "slot", "policy_sha256"}
        if slot is FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION:
            expected_fields.add("manifest_coordinate")
        if (
            set(predecessor) != expected_fields
            or predecessor.get("kind") != "FENCE_SLOT"
            or predecessor.get("slot") != _PREDECESSORS[slot].value
            or predecessor.get("policy_sha256") != prestate_hash
        ):
            _fail("entry predecessor graph is not exact")
        if slot is FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION:
            coordinate = parse_artifact_coordinate(
                predecessor["manifest_coordinate"]
            )
            if coordinate.key != _BOOTSTRAP_MANIFEST_KEY:
                _fail("BATCH predecessor manifest coordinate is foreign")
    terminal = mapping["terminal_head"]
    if terminal != {
        "kind": "FENCE_SLOT",
        "slot": slot.value,
        "policy_sha256": mapping["policy_sha256"],
    }:
        _fail("entry terminal_head is not exact")
    if slot is FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION:
        if type(mapping["batch_projection_contract"]) is not dict:
            _fail("BATCH batch_projection_contract is absent")
    elif mapping["batch_projection_contract"] is not None:
        _fail("batch_projection_contract is non-null only for BATCH")
    change_set_name = mapping["change_set_name"]
    if (
        type(change_set_name) is not str
        or _CHANGE_SET_NAME.fullmatch(change_set_name) is None
        or re.fullmatch(
            r"keep-glm52-h1g-%s-%s-[0-9a-f]{16}"
            % (
                _GENERATION_TEXT,
                slot.value.lower().replace("_", "-"),
            ),
            change_set_name,
        )
        is None
    ):
        _fail("entry change_set_name is not deterministic")
    skeleton = _require_exact_fields(
        mapping["request_skeleton"],
        _REQUEST_SKELETON_FIELDS,
        "request_skeleton",
    )
    if (
        type(skeleton["StackName"]) is not str
        or _STACK_ID.fullmatch(skeleton["StackName"]) is None  # type: ignore[arg-type]
        or skeleton["ChangeSetName"] != change_set_name
        or skeleton["ChangeSetType"] != "UPDATE"
        or skeleton["TemplateURL"] != mapping["template_url"]
        or type(skeleton["RoleARN"]) is not str
        or _ROLE_ARN.fullmatch(skeleton["RoleARN"]) is None  # type: ignore[arg-type]
        or skeleton["ResourceTypes"] != ["AWS::S3::BucketPolicy"]
        and skeleton["ResourceTypes"] != ("AWS::S3::BucketPolicy",)
        or skeleton["IncludeNestedStacks"] is not False
        or skeleton["ImportExistingResources"] is not False
    ):
        _fail("request_skeleton is not exact")
    expected_pre_role = mapping["expected_prestate_stack_role_arn"]
    expected_post_role = mapping["expected_poststate_stack_role_arn"]
    if (
        type(expected_pre_role) is not str
        or _ROLE_ARN.fullmatch(expected_pre_role) is None
        or type(expected_post_role) is not str
        or _ROLE_ARN.fullmatch(expected_post_role) is None
        or expected_post_role != skeleton["RoleARN"]
        or (
            slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE
            and expected_pre_role == expected_post_role
        )
        or (
            slot is not FenceSlot.PREPARE_GENESIS_LIVE_STATE
            and expected_pre_role != expected_post_role
        )
    ):
        _fail("entry stack service-role transition is not exact")
    if canonical_sha256(dict(skeleton)) != mapping["request_skeleton_sha256"]:
        _fail("request_skeleton_sha256 drifted")
    unsigned = dict(mapping)
    identity = unsigned.pop("entry_identity_sha256")
    if canonical_sha256(unsigned) != identity:
        _fail("entry_identity_sha256 drifted")
    raw = template_bytes
    if type(raw) is not bytes:
        _fail("template_bytes are not immutable bytes")
    if raw and (
        not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
        or _sha(raw) != mapping["template_sha256"]
        or _sha(raw[:-1]) != mapping["template_body_sha256"]
    ):
        _fail("entry template bytes drifted")
    return _detached_mapping(dict(mapping), "manifest entry"), raw


def parse_fence_entry(
    value: object, *, template_bytes: bytes = b""
) -> FenceArtifactEntry:
    return FenceArtifactEntry(
        _value=value,  # type: ignore[arg-type]
        template_bytes=template_bytes,
    )


def _manifest_record_type(stage: ManifestStage) -> str:
    return (
        "glm52_fence_bootstrap_manifest_v2"
        if stage is ManifestStage.BOOTSTRAP
        else "glm52_fence_source_settled_manifest_v2"
    )


def build_fence_manifest(
    *,
    stage: ManifestStage,
    fields: Mapping[str, object],
    bridge_seed: BridgeSeedArtifact,
    entries: Sequence[FenceArtifactEntry],
    bootstrap_manifest_coordinate: Optional[ArtifactCoordinate] = None,
    support_runtime_identity_coordinate: Optional[
        SupportRuntimeIdentityCoordinate
    ] = None,
    source_settlement: Optional[SourceSettlementSeal] = None,
) -> FenceManifest:
    if type(stage) is not ManifestStage:
        _fail("manifest stage is not closed")
    if type(fields) is not dict or frozenset(fields) != _MANIFEST_COMMON_INPUT_FIELDS:
        _fail("manifest builder common field set is not exact")
    if type(bridge_seed) is not BridgeSeedArtifact:
        _fail("manifest bridge_seed is not authenticated")
    if type(entries) not in (list, tuple) or any(
        type(entry) is not FenceArtifactEntry for entry in entries
    ):
        _fail("manifest entries are not authenticated")
    value = json.loads(canonical_json_bytes(fields))
    value.update(
        {
            "schema_version": 2,
            "record_type": _manifest_record_type(stage),
            "manifest_stage": stage.value,
            "mutation_authority_inventory_sha256": canonical_sha256(
                value["mutation_authority_inventory"]
            ),
            "executor_inventory_sha256": canonical_sha256(
                value["executor_inventory"]
            ),
            "support_runtime_identity_coordinate": (
                None
                if support_runtime_identity_coordinate is None
                else support_runtime_identity_coordinate.to_dict()
            ),
            "bridge_seed": bridge_seed.to_dict(),
            "transition_graph": [
                {
                    "slot": entry.slot.value,
                    "allowed_predecessor_heads": entry.to_dict()[
                        "allowed_predecessor_heads"
                    ],
                    "terminal_head": entry.to_dict()["terminal_head"],
                }
                for entry in entries
            ],
            "writer_inventory_sha256": canonical_sha256(
                value["writer_inventory"]
            ),
            "writer_policy_cohorts_sha256": canonical_sha256(
                value["writer_policy_cohorts"]
            ),
            "source_inventory_sha256": canonical_sha256(
                value["source_inventory"]
            ),
            "bootstrap_manifest_coordinate": (
                None
                if bootstrap_manifest_coordinate is None
                else bootstrap_manifest_coordinate.to_dict()
            ),
            "source_settlement": (
                None if source_settlement is None else source_settlement.to_dict()
            ),
            "entries": [entry.to_dict() for entry in entries],
        }
    )
    value["canonical_identity_sha256"] = canonical_sha256(value)
    return parse_fence_manifest(value)


def _validate_fence_manifest_value(
    value: object,
) -> Tuple[
    Mapping[str, object],
    Tuple[FenceArtifactEntry, ...],
    BridgeSeedArtifact,
    Optional[SourceSettlementSeal],
    Optional[SupportRuntimeIdentityCoordinate],
    Optional[ArtifactCoordinate],
]:
    mapping = _require_exact_fields(value, _MANIFEST_FIELDS, "fence manifest v2")
    stage = _stage(mapping["manifest_stage"])
    if mapping["schema_version"] != 2 or mapping["record_type"] != _manifest_record_type(
        stage
    ):
        _fail("manifest record_type is not the exact v2 stage discriminant")
    if (
        mapping["account_id"] != ACCOUNT_ID
        or mapping["region"] != REGION
        or mapping["run_id"] != RUN_ID
        or mapping["activation_id"] != _ACTIVATION_ID
        or mapping["generation"] != 1
        or mapping["bucket_name"] != CAMPAIGN_BUCKET
        or mapping["bucket_arn"] != "arn:aws:s3:::%s" % CAMPAIGN_BUCKET
        or mapping["stack_name"] != "keep-glm52-h1g-fence"
        or type(mapping["stack_id"]) is not str
        or _STACK_ID.fullmatch(mapping["stack_id"]) is None  # type: ignore[arg-type]
        or mapping["logical_id"] != "H1gProductionFenceBucketPolicy"
    ):
        _fail("manifest fixed identity fields drifted")
    for field in (
        "member_account_authority",
        "bucket_control_plane",
        "kms_key_identity",
        "migration_service_role",
        "fence_service_role",
        "policy_limits",
    ):
        if type(mapping[field]) is not dict or not mapping[field]:
            _fail("manifest %s is not one closed object" % field)
    if mapping["legacy_mode"] not in ("ENABLED", "DISABLED"):
        _fail("manifest legacy_mode is not closed")
    _require_sha(
        mapping["render_input_identity_sha256"],
        "manifest.render_input_identity_sha256",
    )
    _require_sha(
        mapping["legacy_fragment_sha256"], "manifest.legacy_fragment_sha256"
    )
    for field, hash_field in (
        ("mutation_authority_inventory", "mutation_authority_inventory_sha256"),
        ("executor_inventory", "executor_inventory_sha256"),
        ("writer_inventory", "writer_inventory_sha256"),
        ("writer_policy_cohorts", "writer_policy_cohorts_sha256"),
        ("source_inventory", "source_inventory_sha256"),
    ):
        rows = mapping[field]
        if type(rows) not in (list, tuple):
            _fail("manifest %s is not an ordered array" % field)
        if canonical_sha256(rows) != mapping[hash_field]:
            _fail("manifest %s drifted" % hash_field)
        identities = tuple(
            canonical_sha256(row)
            for row in rows  # type: ignore[union-attr]
        )
        if len(set(identities)) != len(identities):
            _fail("manifest %s contains duplicate rows" % field)
    executor_rows = mapping["executor_inventory"]
    if (
        len(executor_rows) != 2  # type: ignore[arg-type]
        or any(type(row) is not dict for row in executor_rows)  # type: ignore[union-attr]
        or tuple(
            row.get("authority_class")  # type: ignore[union-attr]
            for row in executor_rows  # type: ignore[union-attr]
        )
        != (
            ExecutorAuthorityClass.RETAINED_PRE_SUPPORT.value,
            ExecutorAuthorityClass.SUPPORT_RUNTIME.value,
        )
    ):
        _fail("manifest executor_inventory order is not exact")
    if len(mapping["source_inventory"]) != 7:  # type: ignore[arg-type]
        _fail("manifest source_inventory does not contain seven genesis rows")
    bridge_seed = parse_bridge_seed_artifact(mapping["bridge_seed"])
    entries_value = mapping["entries"]
    if type(entries_value) not in (list, tuple):
        _fail("manifest entries are not one ordered array")
    entries = tuple(parse_fence_entry(entry) for entry in entries_value)
    observed_order = tuple(entry.slot.value for entry in entries)
    expected_order = (
        BOOTSTRAP_ENTRY_ORDER
        if stage is ManifestStage.BOOTSTRAP
        else SOURCE_SETTLED_ENTRY_ORDER
    )
    if observed_order != expected_order or len(set(observed_order)) != len(
        observed_order
    ):
        _fail("manifest entries are not the exact stage order")
    by_slot = {entry.slot: entry for entry in entries}
    render_identity = str(mapping["render_input_identity_sha256"])
    migration_role = mapping["migration_service_role"].get("arn")  # type: ignore[union-attr]
    fence_role = mapping["fence_service_role"].get("arn")  # type: ignore[union-attr]
    if (
        type(migration_role) is not str
        or _ROLE_ARN.fullmatch(migration_role) is None
        or type(fence_role) is not str
        or _ROLE_ARN.fullmatch(fence_role) is None
        or migration_role == fence_role
    ):
        _fail("manifest service-role identities are not exact")
    for entry in entries:
        entry_value = entry.to_dict()
        expected_change_set_name = "keep-glm52-h1g-%s-%s-%s" % (
            _GENERATION_TEXT,
            entry.slot.value.lower().replace("_", "-"),
            render_identity[:16],
        )
        if (
            entry_value["change_set_name"] != expected_change_set_name
            or entry_value["request_skeleton"]["StackName"]
            != mapping["stack_id"]
            or entry_value["request_skeleton"]["RoleARN"] != fence_role
            or entry_value["expected_poststate_stack_role_arn"] != fence_role
            or (
                entry.slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE
                and entry_value["expected_prestate_stack_role_arn"]
                != migration_role
            )
            or (
                entry.slot is not FenceSlot.PREPARE_GENESIS_LIVE_STATE
                and entry_value["expected_prestate_stack_role_arn"] != fence_role
            )
        ):
            _fail("manifest entry derivation from roles/input identity drifted")
    if stage is ManifestStage.BOOTSTRAP:
        if (
            mapping["bootstrap_manifest_coordinate"] is not None
            or mapping["support_runtime_identity_coordinate"] is not None
            or mapping["source_settlement"] is not None
        ):
            _fail("BOOTSTRAP stage coordinates and source_settlement must be null")
        bootstrap_coordinate = None
        runtime_coordinate = None
        source_seal = None
        if (
            by_slot[FenceSlot.PREPARE_GENESIS_LIVE_STATE]
            .allowed_predecessor_heads[0]["policy_sha256"]
            != bridge_seed.policy_sha256
        ):
            _fail("BOOTSTRAP PREPARE does not bind bridge seed")
        for child, parent in (
            (
                FenceSlot.RESERVATION_ONLY,
                FenceSlot.PREPARE_GENESIS_LIVE_STATE,
            ),
            (
                FenceSlot.SOURCE_FAMILIES_FROZEN,
                FenceSlot.RESERVATION_ONLY,
            ),
            (FenceSlot.CLOSED_SOURCE, FenceSlot.SOURCE_FAMILIES_FROZEN),
        ):
            if (
                by_slot[child].allowed_predecessor_heads[0]["policy_sha256"]
                != by_slot[parent].policy_sha256
            ):
                _fail("BOOTSTRAP predecessor graph hash drifted")
        if len(
            {
                entry.to_dict()["publisher_deny_policy_sha256"]
                for entry in entries
            }
        ) != 1:
            _fail("BOOTSTRAP publisher deny policy changes before BATCH")
    else:
        if (
            mapping["bootstrap_manifest_coordinate"] is None
            or mapping["support_runtime_identity_coordinate"] is None
            or mapping["source_settlement"] is None
        ):
            _fail("SOURCE_SETTLED stage coordinates and source_settlement are required")
        bootstrap_coordinate = parse_artifact_coordinate(
            mapping["bootstrap_manifest_coordinate"]
        )
        if bootstrap_coordinate.key != _BOOTSTRAP_MANIFEST_KEY:
            _fail("SOURCE_SETTLED bootstrap_manifest_coordinate is foreign")
        runtime_coordinate = parse_support_runtime_identity_coordinate(
            mapping["support_runtime_identity_coordinate"]
        )
        source_seal = parse_source_settlement_seal(mapping["source_settlement"])
        batch_predecessor = by_slot[
            FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION
        ].allowed_predecessor_heads[0]
        if batch_predecessor["manifest_coordinate"] != bootstrap_coordinate.to_dict():
            _fail("SOURCE_SETTLED cross-manifest predecessor coordinate drifted")
        if (
            by_slot[FenceSlot.TERMINAL]
            .allowed_predecessor_heads[0]["policy_sha256"]
            != by_slot[FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION].policy_sha256
        ):
            _fail("SOURCE_SETTLED TERMINAL predecessor hash drifted")
        if (
            by_slot[FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION].to_dict()[
                "publisher_deny_policy_sha256"
            ]
            != by_slot[FenceSlot.TERMINAL].to_dict()[
                "publisher_deny_policy_sha256"
            ]
        ):
            _fail("publisher deny policy changes after BATCH")
    expected_graph = [
        {
            "slot": entry.slot.value,
            "allowed_predecessor_heads": entry.to_dict()[
                "allowed_predecessor_heads"
            ],
            "terminal_head": entry.to_dict()["terminal_head"],
        }
        for entry in entries
    ]
    if mapping["transition_graph"] != expected_graph:
        _fail("manifest transition_graph drifted")
    _identity_projection(mapping, "fence manifest v2")
    return (
        _detached_mapping(dict(mapping), "fence manifest v2"),
        entries,
        bridge_seed,
        source_seal,
        runtime_coordinate,
        bootstrap_coordinate,
    )


def parse_fence_manifest(value: object) -> FenceManifest:
    (
        detached,
        entries,
        bridge_seed,
        source_seal,
        runtime_coordinate,
        bootstrap_coordinate,
    ) = _validate_fence_manifest_value(value)
    manifest = object.__new__(FenceManifest)
    object.__setattr__(manifest, "_value", detached)
    object.__setattr__(manifest, "entries", entries)
    object.__setattr__(manifest, "bridge_seed", bridge_seed)
    object.__setattr__(manifest, "source_settlement", source_seal)
    object.__setattr__(
        manifest,
        "support_runtime_identity_coordinate",
        runtime_coordinate,
    )
    object.__setattr__(
        manifest,
        "bootstrap_manifest_coordinate",
        bootstrap_coordinate,
    )
    return manifest


def build_fence_transition_request(
    *,
    manifest_coordinate: ArtifactCoordinate,
    manifest_stage: ManifestStage,
    selected_entry: FenceArtifactEntry,
    support_runtime_identity_coordinate: Optional[
        SupportRuntimeIdentityCoordinate
    ],
    observed_h1f_active_head_coordinate: Mapping[str, object],
    h1f_successor_coordinate: Optional[Mapping[str, object]],
    source_settled_manifest_coordinate: Optional[
        SourceSettledManifestCoordinate
    ],
) -> FenceTransitionRequest:
    if (
        type(manifest_coordinate) is not ArtifactCoordinate
        or type(manifest_stage) is not ManifestStage
        or type(selected_entry) is not FenceArtifactEntry
    ):
        _fail("transition request inputs are not authenticated")
    value = {
        "manifest_coordinate": manifest_coordinate.to_dict(),
        "manifest_stage": manifest_stage.value,
        "selected_entry_identity_sha256": (
            selected_entry.entry_identity_sha256
        ),
        "slot": selected_entry.slot.value,
        "allowed_create_authority_classes": (
            selected_entry.allowed_create_authority_classes
        ),
        "allowed_execute_authority_classes": (
            selected_entry.allowed_execute_authority_classes
        ),
        "support_runtime_identity_coordinate": (
            None
            if support_runtime_identity_coordinate is None
            else support_runtime_identity_coordinate.to_dict()
        ),
        "predecessor_head": _thaw(
            selected_entry.allowed_predecessor_heads[0]
        ),
        "observed_h1f_active_head_coordinate": (
            observed_h1f_active_head_coordinate
        ),
        "h1f_successor_coordinate": h1f_successor_coordinate,
        "batch_projection": (
            _thaw(selected_entry.batch_projection_contract)
            if selected_entry.slot is FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION
            else None
        ),
        "source_settled_manifest_coordinate": (
            None
            if source_settled_manifest_coordinate is None
            else source_settled_manifest_coordinate.to_dict()
        ),
        "request_skeleton_sha256": selected_entry.request_skeleton_sha256,
    }
    value = json.loads(canonical_json_bytes(value))
    value["canonical_identity_sha256"] = canonical_sha256(value)
    return parse_fence_transition_request(value)


def _validate_fence_transition_request_value(
    value: object,
) -> Mapping[str, object]:
    mapping = _require_exact_fields(
        value, _REQUEST_FIELDS, "glm52_fence_transition_request_v2"
    )
    stage = _stage(mapping["manifest_stage"])
    slot = _slot(mapping["slot"])
    manifest_coordinate = parse_artifact_coordinate(mapping["manifest_coordinate"])
    expected_manifest_key = (
        _BOOTSTRAP_MANIFEST_KEY
        if stage is ManifestStage.BOOTSTRAP
        else _SOURCE_SETTLED_MANIFEST_KEY
    )
    if manifest_coordinate.key != expected_manifest_key:
        _fail("transition request manifest_coordinate does not match stage")
    _require_sha(
        mapping["selected_entry_identity_sha256"],
        "selected_entry_identity_sha256",
    )
    _require_sha(mapping["request_skeleton_sha256"], "request_skeleton_sha256")
    create_classes, execute_classes = _AUTHORITIES[slot]
    if tuple(mapping["allowed_create_authority_classes"]) != create_classes or tuple(  # type: ignore[arg-type]
        mapping["allowed_execute_authority_classes"]  # type: ignore[arg-type]
    ) != execute_classes:
        _fail("transition request authority arrays drifted")
    predecessor = mapping["predecessor_head"]
    if type(predecessor) is not dict:
        _fail("transition request predecessor_head is not exact")
    if slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE:
        if set(predecessor) != {"kind", "policy_sha256"} or predecessor.get(
            "kind"
        ) != "BRIDGE_SEED":
            _fail("PREPARE predecessor_head is not bridge seed")
    else:
        expected_fields = {"kind", "slot", "policy_sha256"}
        if slot is FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION:
            expected_fields.add("manifest_coordinate")
        if (
            set(predecessor) != expected_fields
            or predecessor.get("kind") != "FENCE_SLOT"
            or predecessor.get("slot") != _PREDECESSORS[slot].value
        ):
            _fail("transition request predecessor_head graph drifted")
    _require_sha(predecessor.get("policy_sha256"), "predecessor_head.policy_sha256")
    observed = mapping["observed_h1f_active_head_coordinate"]
    if type(observed) is not dict or not observed:
        _fail("observed_h1f_active_head_coordinate is required")
    runtime = mapping["support_runtime_identity_coordinate"]
    successor = mapping["h1f_successor_coordinate"]
    batch = mapping["batch_projection"]
    source_coordinate = mapping["source_settled_manifest_coordinate"]
    if slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE:
        if (
            runtime is not None
            or successor is not None
            or batch is not None
            or source_coordinate is not None
        ):
            _fail("PREPARE stage-specific request fields must be null")
    else:
        if runtime is None:
            _fail("post-PREPARE support_runtime_identity_coordinate is required")
        parse_support_runtime_identity_coordinate(runtime)
        if slot in (
            FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION,
            FenceSlot.TERMINAL,
        ):
            if (
                stage is not ManifestStage.SOURCE_SETTLED
                or type(successor) is not dict
                or not successor
                or source_coordinate is None
            ):
                _fail("late transition source-settled coordinates are required")
            parsed_source_coordinate = parse_source_settled_manifest_coordinate(
                source_coordinate
            )
            if (
                parsed_source_coordinate.artifact_coordinate
                != manifest_coordinate
            ):
                _fail("source_settled_manifest_coordinate does not bind manifest")
        else:
            if (
                stage is not ManifestStage.BOOTSTRAP
                or successor is not None
                or source_coordinate is not None
            ):
                _fail("bootstrap transition late coordinates must be null")
        if slot is FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION:
            if type(batch) is not dict:
                _fail("BATCH batch_projection is required")
        elif batch is not None:
            _fail("batch_projection is non-null only for BATCH")
    _identity_projection(mapping, "glm52_fence_transition_request_v2")
    return _detached_mapping(
        dict(mapping),
        "glm52_fence_transition_request_v2",
    )


def parse_fence_transition_request(value: object) -> FenceTransitionRequest:
    return FenceTransitionRequest(_value=value)  # type: ignore[arg-type]


def _reject_template_intrinsics(value: object) -> None:
    if type(value) is dict:
        for key, child in value.items():
            if (
                type(key) is not str
                or key == "Ref"
                or key.startswith("Fn::")
            ):
                _fail("published template contains a forbidden intrinsic")
            _reject_template_intrinsics(child)
    elif type(value) in (list, tuple):
        for child in value:
            _reject_template_intrinsics(child)
    elif type(value) is str and "{{resolve:" in value:
        _fail("published template contains a dynamic reference")


def _publication_identity(
    key: str, raw: bytes, stage: ManifestStage
) -> str:
    if key == _SEED_KEY:
        if raw.endswith(b"\n"):
            _fail("bridge-seed publication bytes must not end in LF")
        body = raw
    else:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            _fail("template and manifest publication bytes require one LF")
        body = raw[:-1]
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FenceArtifactError("publication artifact is not JSON") from exc
    if canonical_json_bytes(value) != body:
        _fail("publication artifact is not canonical JSON")
    if key in _TEMPLATE_KEYS.values():
        if type(value) is not dict or set(value) != {
            "AWSTemplateFormatVersion",
            "Resources",
        }:
            _fail("published template top-level shape is not exact")
        resources = value.get("Resources")
        if type(resources) is not dict or set(resources) != {
            "H1gProductionFenceBucketPolicy"
        }:
            _fail("published template resource set is not exact")
        resource = resources["H1gProductionFenceBucketPolicy"]
        if type(resource) is not dict or set(resource) != {
            "DeletionPolicy",
            "Properties",
            "Type",
            "UpdateReplacePolicy",
        }:
            _fail("published template resource shape is not exact")
        properties = resource.get("Properties")
        if (
            value["AWSTemplateFormatVersion"] != "2010-09-09"
            or resource.get("DeletionPolicy") != "Retain"
            or resource.get("Type") != "AWS::S3::BucketPolicy"
            or resource.get("UpdateReplacePolicy") != "Retain"
            or type(properties) is not dict
            or set(properties) != {"Bucket", "PolicyDocument"}
            or properties.get("Bucket") != CAMPAIGN_BUCKET
            or type(properties.get("PolicyDocument")) is not dict
        ):
            _fail("published template body is not exact")
        policy_document = properties["PolicyDocument"]
        if (
            set(policy_document) != {"Version", "Statement"}
            or policy_document["Version"] != "2012-10-17"
            or type(policy_document["Statement"]) is not list
        ):
            _fail("published PolicyDocument shape is not exact")
        _reject_template_intrinsics(value)
    elif key in (_BOOTSTRAP_MANIFEST_KEY, _SOURCE_SETTLED_MANIFEST_KEY):
        manifest = parse_fence_manifest(value)
        if manifest.manifest_stage is not stage:
            _fail("published manifest does not match publication stage")
        return manifest.canonical_identity_sha256
    if type(value) is not dict:
        _fail("publication artifact body is not one object")
    return canonical_sha256(value)


FenceManifestBytesBuilder = Callable[
    [
        Mapping[FenceSlot, ArtifactCoordinate],
        Optional[ArtifactCoordinate],
    ],
    bytes,
]


def _publish_stage_object(
    *,
    key: str,
    raw: bytes,
    stage: ManifestStage,
    record_type: str,
    kms_key_arn: str,
    services: Task13FixedArtifactServices,
) -> ArtifactCoordinate:
    identity = _publication_identity(key, raw, stage)
    version_id = publish_fixed_key_bytes(
        services=services,
        bucket=CAMPAIGN_BUCKET,
        key=key,
        raw=raw,
        record_type=record_type,
        sse_kms_key_id=kms_key_arn,
    )
    return ArtifactCoordinate(
        bucket=CAMPAIGN_BUCKET,
        key=key,
        version_id=version_id,
        file_sha256=_sha(raw),
        canonical_identity_sha256=identity,
    )


def _adopt_stage_object(
    *,
    key: str,
    raw: bytes,
    stage: ManifestStage,
    record_type: str,
    kms_key_arn: str,
    services: Task13FixedArtifactServices,
) -> ArtifactCoordinate:
    identity = _publication_identity(key, raw, stage)
    outcome = adopt_fixed_key_bytes(
        services=services,
        bucket=CAMPAIGN_BUCKET,
        key=key,
        raw=raw,
        record_type=record_type,
        sse_kms_key_id=kms_key_arn,
    )
    return ArtifactCoordinate(
        bucket=CAMPAIGN_BUCKET,
        key=key,
        version_id=outcome.version_id,
        file_sha256=_sha(raw),
        canonical_identity_sha256=identity,
    )


def _validate_stage_manifest_bindings(
    *,
    raw: bytes,
    stage: ManifestStage,
    slot_coordinates: Mapping[FenceSlot, ArtifactCoordinate],
    seed_coordinate: Optional[ArtifactCoordinate],
) -> None:
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail("published manifest bytes require exactly one LF")
    try:
        value = json.loads(raw[:-1].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FenceArtifactError("published manifest is not JSON") from exc
    manifest = parse_fence_manifest(value)
    if manifest.manifest_stage is not stage:
        _fail("published manifest stage drifted")
    for slot, coordinate in slot_coordinates.items():
        entry = manifest.entry(slot).to_dict()
        if (
            entry["template_key"] != coordinate.key
            or entry["version_id"] != coordinate.version_id
            or entry["template_sha256"] != coordinate.file_sha256
            or entry["template_body_sha256"]
            != coordinate.canonical_identity_sha256
        ):
            _fail("published manifest entry does not bind returned coordinate")
    if stage is ManifestStage.BOOTSTRAP:
        if seed_coordinate is None:
            _fail("bootstrap seed coordinate is absent")
        seed = manifest.bridge_seed.to_dict()
        if (
            seed["key"] != seed_coordinate.key
            or seed["version_id"] != seed_coordinate.version_id
            or seed["file_sha256"] != seed_coordinate.file_sha256
            or seed["body_sha256"]
            != seed_coordinate.canonical_identity_sha256
        ):
            _fail("published manifest bridge seed does not bind returned coordinate")
    elif seed_coordinate is not None:
        _fail("source-settled manifest received a bridge-seed publication")


def publish_fence_stage(
    *,
    stage: ManifestStage,
    template_bytes: Mapping[FenceSlot, bytes],
    manifest_builder: FenceManifestBytesBuilder,
    kms_key_arn: str,
    services: Task13FixedArtifactServices,
    bridge_seed_bytes: Optional[bytes] = None,
) -> Tuple[ArtifactCoordinate, ...]:
    """Publish fixed stage objects, then build and publish its manifest last."""

    if type(stage) is not ManifestStage or type(template_bytes) is not dict:
        _fail("fence publication stage input is not exact")
    slot_order = (
        (
            FenceSlot.PREPARE_GENESIS_LIVE_STATE,
            FenceSlot.RESERVATION_ONLY,
            FenceSlot.SOURCE_FAMILIES_FROZEN,
            FenceSlot.CLOSED_SOURCE,
        )
        if stage is ManifestStage.BOOTSTRAP
        else (
            FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION,
            FenceSlot.TERMINAL,
        )
    )
    if (
        set(template_bytes) != set(slot_order)
        or any(type(template_bytes[slot]) is not bytes for slot in slot_order)
        or not callable(manifest_builder)
        or (
            stage is ManifestStage.BOOTSTRAP
            and type(bridge_seed_bytes) is not bytes
        )
        or (
            stage is ManifestStage.SOURCE_SETTLED
            and bridge_seed_bytes is not None
        )
    ):
        _fail("fence publication artifact set is not exact")

    coordinates = []
    seed_coordinate = None
    if stage is ManifestStage.BOOTSTRAP:
        seed_coordinate = _adopt_stage_object(
            key=_SEED_KEY,
            raw=bridge_seed_bytes,  # type: ignore[arg-type]
            stage=stage,
            record_type="glm52_h1g_bridge_seed_policy_v1",
            kms_key_arn=kms_key_arn,
            services=services,
        )
        coordinates.append(seed_coordinate)

    slot_coordinates = {}
    for slot in slot_order:
        coordinate = _publish_stage_object(
            key=_TEMPLATE_KEYS[slot],
            raw=template_bytes[slot],
            stage=stage,
            record_type="aws_cloudformation_template",
            kms_key_arn=kms_key_arn,
            services=services,
        )
        slot_coordinates[slot] = coordinate
        coordinates.append(coordinate)

    manifest_raw = manifest_builder(
        MappingProxyType(dict(slot_coordinates)),
        seed_coordinate,
    )
    if type(manifest_raw) is not bytes:
        _fail("fence manifest builder did not return immutable bytes")
    _validate_stage_manifest_bindings(
        raw=manifest_raw,
        stage=stage,
        slot_coordinates=slot_coordinates,
        seed_coordinate=seed_coordinate,
    )
    manifest_key = (
        _BOOTSTRAP_MANIFEST_KEY
        if stage is ManifestStage.BOOTSTRAP
        else _SOURCE_SETTLED_MANIFEST_KEY
    )
    coordinates.append(
        _publish_stage_object(
            key=manifest_key,
            raw=manifest_raw,
            stage=stage,
            record_type=_manifest_record_type(stage),
            kms_key_arn=kms_key_arn,
            services=services,
        )
    )
    return tuple(coordinates)


__all__ = [
    "ArtifactCoordinate",
    "BOOTSTRAP_ENTRY_ORDER",
    "BOOTSTRAP_PUBLICATION_ORDER",
    "BridgeSeedArtifact",
    "ExecutorAuthorityClass",
    "FENCE_ARTIFACT_KEYS",
    "FenceArtifactEntry",
    "FenceArtifactError",
    "FenceManifest",
    "FenceSlot",
    "FenceTransitionRequest",
    "ManifestStage",
    "SOURCE_SETTLED_ENTRY_ORDER",
    "SOURCE_SETTLED_PUBLICATION_ORDER",
    "SourceSettledManifestCoordinate",
    "SourceSettlementSeal",
    "SupportRuntimeIdentityCoordinate",
    "FenceManifestBytesBuilder",
    "build_bridge_seed_artifact",
    "build_fence_entry",
    "build_fence_template_bytes",
    "build_fence_manifest",
    "build_fence_transition_request",
    "build_source_settlement_seal",
    "parse_artifact_coordinate",
    "parse_bridge_seed_artifact",
    "parse_fence_entry",
    "parse_fence_manifest",
    "parse_fence_transition_request",
    "parse_source_settled_manifest_coordinate",
    "parse_source_settlement_seal",
    "parse_support_runtime_identity_coordinate",
    "publish_fence_stage",
    "versioned_artifact_url",
]
