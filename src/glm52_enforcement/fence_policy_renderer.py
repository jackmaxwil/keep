"""Deterministic, deny-only S3 bucket-policy renderer for the H.1g fence.

The renderer accepts already-authenticated identities and resources.  It does not
look up AWS state, infer principals, broaden resources, or repair incomplete
inputs.  Its output is canonical JSON plus a byte-exact statement ledger.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from .s3_keys import validate_frozen_key
from typing import Iterable, Mapping, Optional, Sequence, Tuple



CREATE: Tuple[str, ...] = ("s3:PutObject",)
NON_CREATE_MUTATION: Tuple[str, ...] = (
    "s3:AbortMultipartUpload",
    "s3:DeleteObject",
    "s3:DeleteObjectVersion",
    "s3:PutObjectAcl",
    "s3:PutObjectVersionAcl",
    "s3:PutObjectTagging",
    "s3:PutObjectVersionTagging",
    "s3:DeleteObjectTagging",
    "s3:DeleteObjectVersionTagging",
    "s3:PutObjectAnnotation",
    "s3:PutObjectVersionAnnotation",
    "s3:DeleteObjectAnnotation",
    "s3:DeleteObjectVersionAnnotation",
    "s3:UpdateObjectEncryption",
    "s3:PutObjectLegalHold",
    "s3:PutObjectRetention",
    "s3:BypassGovernanceRetention",
    "s3:RestoreObject",
    "s3:ObjectOwnerOverrideToBucketOwner",
    "s3:InitiateReplication",
    "s3:ReplicateObject",
    "s3:ReplicateObjectAnnotation",
    "s3:ReplicateDelete",
    "s3:ReplicateTags",
)
FULL_MUTATION: Tuple[str, ...] = CREATE + NON_CREATE_MUTATION
SOURCE_OBJECT_READ: Tuple[str, ...] = (
    "s3:GetObject",
    "s3:GetObjectAcl",
    "s3:GetObjectAnnotation",
    "s3:GetObjectAttributes",
    "s3:GetObjectLegalHold",
    "s3:GetObjectRetention",
    "s3:GetObjectTagging",
    "s3:GetObjectTorrent",
    "s3:GetObjectVersion",
    "s3:GetObjectVersionAcl",
    "s3:GetObjectVersionAnnotation",
    "s3:GetObjectVersionAnnotationForReplication",
    "s3:GetObjectVersionAttributes",
    "s3:GetObjectVersionForReplication",
    "s3:GetObjectVersionTagging",
    "s3:GetObjectVersionTorrent",
    "s3:ListObjectAnnotations",
    "s3:ListObjectVersionAnnotations",
    "s3:SelectObjectContent",
)
SOURCE_LIST: Tuple[str, ...] = ("s3:ListBucket", "s3:ListBucketVersions")

FIXED_ACCOUNT_ID = "246813579024"
FIXED_BUCKET_NAME = "keep-glm52-models-246813579024-us-west-2"
FIXED_RUN_ID = "glm52-sky-20260724"
FIXED_BUCKET_ARN = "arn:aws:s3:::" + FIXED_BUCKET_NAME
RUN_GUARD_RESOURCE = (
    FIXED_BUCKET_ARN + "/campaigns/glm52-sky-20260724/*"
)
SOURCE_FAMILY_SPECS: Tuple[Tuple[str, str], ...] = (
    (
        "gpu-spend-snapshot",
        FIXED_BUCKET_ARN
        + "/campaigns/glm52-sky-20260724/spend-snapshots/*/"
        "GPU_SPEND_SNAPSHOT.json",
    ),
    (
        "production-submission-intent",
        FIXED_BUCKET_ARN
        + "/campaigns/glm52-sky-20260724/submissions/production/intents/*/"
        "SKYPILOT_SUBMISSION_INTENT.json",
    ),
    (
        "production-controller-baseline",
        FIXED_BUCKET_ARN
        + "/campaigns/glm52-sky-20260724/production/controller-baselines/*/"
        "CONTROLLER_BASELINE.json",
    ),
    (
        "production-control-plane-readiness",
        FIXED_BUCKET_ARN
        + "/campaigns/glm52-sky-20260724/monitor/must-start/production/*/"
        "control-plane-ready/*/CONTROL_PLANE_READY.json",
    ),
    (
        "production-submission-acquisition",
        FIXED_BUCKET_ARN
        + "/campaigns/glm52-sky-20260724/submissions/production/acquisitions/*/"
        "SUBMISSION_ACQUIRED.json",
    ),
)
CORE_WRITER_RESOURCES: Tuple[str, ...] = (
    FIXED_BUCKET_ARN
    + "/campaigns/glm52-sky-20260724/authorities/fence/FENCE_GENESIS.json",
    FIXED_BUCKET_ARN
    + "/campaigns/glm52-sky-20260724/authorities/fence/successors/*/"
    "FENCE_SUCCESSOR.json",
    FIXED_BUCKET_ARN
    + "/campaigns/glm52-sky-20260724/submissions/production/generations/"
    "00000001/GENERATION_CLAIM.json",
    FIXED_BUCKET_ARN
    + "/campaigns/glm52-sky-20260724/submissions/production/generations/"
    "00000001/START_DECISION.json",
    FIXED_BUCKET_ARN
    + "/campaigns/glm52-sky-20260724/submissions/production/generations/"
    "00000001/GENERATION_TERMINAL.json",
    FIXED_BUCKET_ARN
    + "/campaigns/glm52-sky-20260724/submissions/production/generations/"
    "00000001/handoff/SKY_POST_HANDOFF.json",
)
SOURCE_FAMILY_IDS = tuple(item[0] for item in SOURCE_FAMILY_SPECS)

_SOURCE_CLOSED_HEADS = frozenset(
    {
        "BRIDGE_SEED",
        "PREPARE",
        "SOURCE_FAMILIES_FROZEN",
        "BATCH",
        "CLOSED_SOURCE",
        "TERMINAL",
    }
)
_TERMINAL_HEADS = frozenset({"CLOSED_SOURCE", "TERMINAL"})

_COMPONENT_ORDER: Tuple[str, ...] = (
    "global_guards",
    "legacy_fragment",
    "permanent_lineage",
    "source_family_closures",
    "active_writer_cohort",
    "object_reader_guards",
    "bucket_list_reader_guards",
    "terminal_closure",
)
_POLICY_PREFIX_BYTES = b'{"Statement":['
_POLICY_SUFFIX_BYTES = b'],"Version":"2012-10-17"}'
_S3_MAX_POLICY_BYTES = 20480

_COMPONENT_CAP_PAIRS: Tuple[Tuple[str, int], ...] = (
    ("global_guards", 1536),
    ("legacy_fragment", 2048),
    ("permanent_lineage", 5632),
    ("source_family_closures", 2048),
    ("active_writer_cohort", 5632),
    ("object_reader_guards", 6144),
    ("bucket_list_reader_guards", 1024),
    ("terminal_closure", 2048),
)
_DEFAULT_COMPONENT_MAX_BYTES: Mapping[str, int] = MappingProxyType(
    dict(_COMPONENT_CAP_PAIRS)
)

_ACCOUNT_RE = re.compile(r"^[0-9]{12}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_ROLE_ID_RE = re.compile(r"^[A-Z0-9]{8,128}$")
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_ROLE_ARN_RE = re.compile(
    r"^arn:aws:iam::246813579024:role/"
    r"(?:[A-Za-z0-9+=,.@_-]+/)*[A-Za-z0-9+=,.@_-]{1,64}$"
)
_KMS_KEY_ARN_RE = re.compile(
    r"^arn:aws:kms:us-west-2:246813579024:key/"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _policy_json_bytes(value: object) -> bytes:
    """Encode the exact ASCII canonical form used as policy authority."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _require_exact_tuple(label: str, value: object) -> None:
    if type(value) is not tuple:
        raise TypeError("%s must be an immutable tuple" % label)

def _require_nonnegative_exact_int(label: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError("%s must be an integer" % label)
    if value < 0:
        raise ValueError("%s must be nonnegative" % label)


def _snapshot_component_mapping(
    label: str, value: object
) -> dict[str, int]:
    if type(value) not in (dict, MappingProxyType):
        raise TypeError("%s must be a plain mapping" % label)
    snapshot = dict(value)
    for component, count in snapshot.items():
        if type(component) is not str:
            raise TypeError("%s keys must be strings" % label)
        _require_nonnegative_exact_int(
            "%s[%s]" % (label, component), count
        )
    if set(snapshot) != set(_COMPONENT_ORDER):
        raise ValueError("%s must have the exact closed component set" % label)
    return {component: snapshot[component] for component in _COMPONENT_ORDER}



def _require_plain_json(value: object, path: str) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("%s JSON object keys must be strings" % path)
            _require_plain_json(item, path + "." + key)
        return
    if type(value) in (list, tuple):
        for index, item in enumerate(value):
            _require_plain_json(item, "%s[%d]" % (path, index))
        return
    raise TypeError("%s contains a non-plain JSON value" % path)


def _freeze_json(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if type(value) in (dict, MappingProxyType):
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("JSON object keys must be exact strings")
            result[key] = _thaw_json(item)
        return result
    if type(value) in (list, tuple):
        return [_thaw_json(item) for item in value]
    if value is None or type(value) in (bool, int, str):
        return value
    raise TypeError("legacy statement contains a non-exact JSON value")


def _freeze_legacy_statement(
    statement: Mapping[str, object],
) -> Mapping[str, object]:
    if type(statement) not in (dict, MappingProxyType):
        raise TypeError("legacy statement must be a plain object or exact frozen object")
    plain_statement = _thaw_json(statement)
    _require_plain_json(plain_statement, "legacy statement")
    try:
        snapshot = json.loads(_policy_json_bytes(plain_statement))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("legacy statement could not be snapshotted") from exc
    frozen = _freeze_json(snapshot)
    assert isinstance(frozen, Mapping)
    return frozen


def _authority_json(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _authority_json(getattr(value, item.name))
            for item in fields(value)
            if item.name != "render_input_identity_sha256"
        }
    if isinstance(value, Mapping):
        return {
            str(key): _authority_json(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_authority_json(item) for item in value]
    return value


class BuildMode(str, Enum):
    """Authenticated retained-policy state before H.1g migration."""

    LEGACY_ENABLED = "LEGACY_ENABLED"
    LEGACY_DISABLED = "LEGACY_DISABLED"


class PolicyHead(str, Enum):
    """Closed bridge-seed and six-slot policy topology."""

    BRIDGE_SEED = "BRIDGE_SEED"
    PREPARE = "PREPARE"
    RESERVATION = "RESERVATION"
    SOURCE_FAMILIES_FROZEN = "SOURCE_FAMILIES_FROZEN"
    BATCH = "BATCH"
    CLOSED_SOURCE = "CLOSED_SOURCE"
    TERMINAL = "TERMINAL"


class PolicyComponentBudgetExhausted(ValueError):
    """Raised when canonical output exceeds a closed policy-size budget."""


@dataclass(frozen=True)
class PrincipalIdentity:
    """One live-proved IAM role binding."""

    binding_id: str
    arn: str
    role_id: str

    def __post_init__(self) -> None:
        for label, value in (
            ("binding_id", self.binding_id),
            ("arn", self.arn),
            ("role_id", self.role_id),
        ):
            if type(value) is not str:
                raise TypeError("%s must be a string" % label)


@dataclass(frozen=True)
class WriterCohort:
    """One active writer cohort and its run-scoped resource union."""

    cohort_id: str
    members: Tuple[PrincipalIdentity, ...]
    guard_resources: Tuple[str, ...]
    cross_member_denial_evidence_sha256: str

    def __post_init__(self) -> None:
        _require_exact_tuple("writer cohort members", self.members)
        _require_exact_tuple(
            "writer cohort guard_resources", self.guard_resources
        )


@dataclass(frozen=True)
class ReservedFamily:
    """A source or record family that is closed to every principal."""

    family_id: str
    resources: Tuple[str, ...]

    def __post_init__(self) -> None:
        _require_exact_tuple("reserved family resources", self.resources)


@dataclass(frozen=True)
class ObjectReaderScope:
    """Exact object-read scope and its accepted readers."""

    scope_id: str
    readers: Tuple[PrincipalIdentity, ...]
    resources: Tuple[str, ...]

    def __post_init__(self) -> None:
        _require_exact_tuple("object reader scope readers", self.readers)
        _require_exact_tuple("object reader scope resources", self.resources)


@dataclass(frozen=True)
class ListReaderScope:
    """Bucket-wide source-list guard and its accepted readers."""

    scope_id: str
    readers: Tuple[PrincipalIdentity, ...]

    def __post_init__(self) -> None:
        _require_exact_tuple("list reader scope readers", self.readers)


@dataclass(frozen=True)
class PolicyLimits:
    """Closed canonical-byte limits for one rendered policy."""

    max_policy_bytes: int = 18432
    unallocated_headroom_bytes: int = 512
    component_max_bytes: Mapping[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_COMPONENT_MAX_BYTES)
    )

    def __post_init__(self) -> None:
        if type(self.component_max_bytes) not in (dict, MappingProxyType):
            raise TypeError("component_max_bytes must be a plain mapping")
        object.__setattr__(
            self,
            "component_max_bytes",
            MappingProxyType(dict(self.component_max_bytes)),
        )


@dataclass(frozen=True)
class FencePolicyInput:
    """Complete authenticated input for one seed or slot policy."""

    build_mode: BuildMode
    policy_head: PolicyHead
    bucket_name: str
    account_id: str
    kms_key_arn: str
    predecessor_policy_sha256: Optional[str] = None
    freeze_denial_evidence_sha256: Optional[str] = None
    source_publication_sealed_sha256: Optional[str] = None
    all_version_inventory_sha256: Optional[str] = None
    source_settlement_sha256: Optional[str] = None
    publisher_deny_policy_sha256: Optional[str] = None
    terminal_prerequisite_sha256: Optional[str] = None
    legacy_statements: Tuple[Mapping[str, object], ...] = ()
    permanent_enrolled_resources: Tuple[str, ...] = ()
    retired_publishers: Tuple[PrincipalIdentity, ...] = ()
    retired_publisher_resources: Tuple[str, ...] = ()
    reserved_families: Tuple[ReservedFamily, ...] = ()
    writer_cohorts: Tuple[WriterCohort, ...] = ()
    writer_owned_resources: Tuple[str, ...] = ()
    source_validation_readers: Tuple[PrincipalIdentity, ...] = ()
    source_inventory_readers: Tuple[PrincipalIdentity, ...] = ()
    terminal_audit_readers: Tuple[PrincipalIdentity, ...] = ()
    selected_source_keys: Tuple[str, ...] = ()
    nonselected_source_keys: Tuple[str, ...] = ()
    policy_limits: PolicyLimits = field(default_factory=PolicyLimits)
    render_input_identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        tuple_fields = (
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
        )
        for field_name in tuple_fields:
            _require_exact_tuple(field_name, getattr(self, field_name))
        object.__setattr__(
            self,
            "legacy_statements",
            tuple(
                _freeze_legacy_statement(statement)
                for statement in self.legacy_statements
            ),
        )
        object.__setattr__(
            self,
            "render_input_identity_sha256",
            hashlib.sha256(
                _policy_json_bytes(_authority_json(self))
            ).hexdigest(),
        )

@dataclass(frozen=True)
class StatementLedgerEntry:
    """Byte offsets and digest for one canonical statement."""

    sequence: int
    sid: str
    component: str
    leading_comma_bytes: int
    statement_bytes: int
    start_offset: int
    end_offset: int
    statement_sha256: str


def _decode_canonical_policy(policy_bytes: bytes) -> list[dict[str, object]]:
    try:
        policy = json.loads(policy_bytes)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("policy_bytes must contain valid JSON") from exc
    _require_plain_json(policy, "policy")
    _validate_json_value(policy, "policy")
    if type(policy) is not dict:
        raise TypeError("policy JSON must be an object")
    if set(policy) != {"Statement", "Version"}:
        raise ValueError("policy JSON must have the exact canonical shape")
    version = policy["Version"]
    if type(version) is not str:
        raise TypeError("policy Version must be a string")
    if version != "2012-10-17":
        raise ValueError("policy Version must be 2012-10-17")
    statements = policy["Statement"]
    if type(statements) is not list:
        raise TypeError("policy Statement must be an array")

    required_statement_keys = {
        "Action",
        "Effect",
        "Principal",
        "Resource",
        "Sid",
    }
    allowed_statement_keys = required_statement_keys | {"Condition"}
    seen_sids = set()
    for index, statement in enumerate(statements):
        if type(statement) is not dict:
            raise TypeError("policy Statement[%d] must be an object" % index)
        if not required_statement_keys.issubset(statement) or not set(
            statement
        ).issubset(allowed_statement_keys):
            raise ValueError(
                "policy Statement[%d] has a noncanonical shape" % index
            )
        sid = statement["Sid"]
        if type(sid) is not str:
            raise TypeError("policy Statement[%d] Sid must be a string" % index)
        _validate_identifier("policy Statement[%d] Sid" % index, sid)
        if sid in seen_sids:
            raise ValueError("policy statement Sids must be unique")
        seen_sids.add(sid)

        for field_name, required_value in (
            ("Effect", "Deny"),
            ("Principal", "*"),
        ):
            field_value = statement[field_name]
            if type(field_value) is not str:
                raise TypeError(
                    "policy Statement[%d] %s must be a string"
                    % (index, field_name)
                )
            if field_value != required_value:
                raise ValueError(
                    "policy Statement[%d] %s is not canonical"
                    % (index, field_name)
                )
        for field_name in ("Action", "Resource"):
            values = statement[field_name]
            if type(values) is not list:
                raise TypeError(
                    "policy Statement[%d] %s must be an array"
                    % (index, field_name)
                )
            if not values:
                raise ValueError(
                    "policy Statement[%d] %s must not be empty"
                    % (index, field_name)
                )
            for value in values:
                if type(value) is not str:
                    raise TypeError(
                        "policy Statement[%d] %s elements must be strings"
                        % (index, field_name)
                    )
                _require_ascii(
                    "policy Statement[%d] %s" % (index, field_name), value
                )
                if not value:
                    raise ValueError(
                        "policy Statement[%d] %s must not contain empty strings"
                        % (index, field_name)
                    )
        if "Condition" in statement:
            condition = statement["Condition"]
            if type(condition) is not dict:
                raise TypeError(
                    "policy Statement[%d] Condition must be an object" % index
                )
            if not condition:
                raise ValueError(
                    "policy Statement[%d] Condition must not be empty" % index
                )

    if _policy_json_bytes(policy) != policy_bytes:
        raise ValueError("policy_bytes is not canonical policy JSON")
    return statements


def _validate_statement_ledger_entry_types(
    entry: object, index: int
) -> StatementLedgerEntry:
    if type(entry) is not StatementLedgerEntry:
        raise TypeError(
            "statement_ledger[%d] must be a StatementLedgerEntry" % index
        )
    for field_name in ("sid", "component", "statement_sha256"):
        if type(getattr(entry, field_name)) is not str:
            raise TypeError(
                "statement_ledger[%d].%s must be a string"
                % (index, field_name)
            )
    for field_name in (
        "sequence",
        "leading_comma_bytes",
        "statement_bytes",
        "start_offset",
        "end_offset",
    ):
        _require_nonnegative_exact_int(
            "statement_ledger[%d].%s" % (index, field_name),
            getattr(entry, field_name),
        )
    if entry.component not in _COMPONENT_ORDER:
        raise ValueError(
            "statement_ledger[%d] has an unknown component" % index
        )
    return entry


@dataclass(frozen=True)
class PolicyLedgerSummary:
    """Whole-policy byte limits, headrooms, and component counts."""

    policy_prefix_bytes: int
    policy_suffix_bytes: int
    statement_count: int
    max_policy_bytes: int
    working_limit_bytes: int
    reserved_design_headroom_bytes: int
    working_headroom_bytes: int
    design_headroom_bytes: int
    s3_headroom_bytes: int
    component_statement_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        integer_fields = (
            "policy_prefix_bytes",
            "policy_suffix_bytes",
            "statement_count",
            "max_policy_bytes",
            "working_limit_bytes",
            "reserved_design_headroom_bytes",
            "working_headroom_bytes",
            "design_headroom_bytes",
            "s3_headroom_bytes",
        )
        for field_name in integer_fields:
            _require_nonnegative_exact_int(
                field_name, getattr(self, field_name)
            )
        counts = _snapshot_component_mapping(
            "component_statement_counts",
            self.component_statement_counts,
        )

        if self.policy_prefix_bytes != len(_POLICY_PREFIX_BYTES):
            raise ValueError("policy_prefix_bytes is not canonical")
        if self.policy_suffix_bytes != len(_POLICY_SUFFIX_BYTES):
            raise ValueError("policy_suffix_bytes is not canonical")
        if sum(counts.values()) != self.statement_count:
            raise ValueError(
                "component_statement_counts does not match statement_count"
            )
        if self.max_policy_bytes > _S3_MAX_POLICY_BYTES:
            raise ValueError("max_policy_bytes exceeds the S3 policy limit")
        if (
            self.working_limit_bytes
            != self.max_policy_bytes
            - self.reserved_design_headroom_bytes
        ):
            raise ValueError(
                "working_limit_bytes does not reconcile with design reserve"
            )
        rendered_policy_bytes = (
            self.working_limit_bytes - self.working_headroom_bytes
        )
        if rendered_policy_bytes < (
            self.policy_prefix_bytes + self.policy_suffix_bytes
        ):
            raise ValueError("headrooms imply an invalid rendered policy size")
        if (
            self.max_policy_bytes - self.design_headroom_bytes
            != rendered_policy_bytes
        ):
            raise ValueError("design_headroom_bytes does not reconcile")
        if (
            _S3_MAX_POLICY_BYTES - self.s3_headroom_bytes
            != rendered_policy_bytes
        ):
            raise ValueError("s3_headroom_bytes does not reconcile")

        object.__setattr__(
            self,
            "component_statement_counts",
            MappingProxyType(counts),
        )


@dataclass(frozen=True)
class RenderedPolicySet:
    """The complete seven-template topology and its canonical set digest."""

    policies: Mapping[PolicyHead, "RenderedPolicy"]
    render_input_identity_sha256_by_head: Mapping[PolicyHead, str]
    topology_predecessors: Mapping[PolicyHead, Optional[PolicyHead]]
    policy_set_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("policies", self.policies),
            (
                "render_input_identity_sha256_by_head",
                self.render_input_identity_sha256_by_head,
            ),
            ("topology_predecessors", self.topology_predecessors),
        ):
            if type(value) not in (dict, MappingProxyType):
                raise TypeError("%s must be a plain mapping" % label)

        policies = dict(self.policies)
        input_identities = dict(
            self.render_input_identity_sha256_by_head
        )
        topology_predecessors = dict(self.topology_predecessors)
        if tuple(policies) != _POLICY_SET_ORDER:
            raise ValueError("policies must use the exact seven-head order")
        if tuple(input_identities) != _POLICY_SET_ORDER:
            raise ValueError(
                "render_input_identity_sha256_by_head must use the exact "
                "seven-head order"
            )
        if (
            tuple(topology_predecessors) != _POLICY_SET_ORDER
            or topology_predecessors != dict(_TOPOLOGY_PREDECESSORS)
        ):
            raise ValueError(
                "topology_predecessors must equal the exact topology "
                "predecessor map"
            )

        for head in _POLICY_SET_ORDER:
            policy = policies[head]
            if type(policy) is not RenderedPolicy:
                raise TypeError(
                    "policies[%s] must be a RenderedPolicy" % head.value
                )
            expected_head_sids = (
                ()
                if head is PolicyHead.RESERVATION
                else ("DenyAllReservedFamilyMutation_" + head.value,)
            )
            head_sids = tuple(
                entry.sid
                for entry in policy.statement_ledger
                if entry.component == "source_family_closures"
            )
            if head_sids != expected_head_sids:
                raise ValueError(
                    "policies[%s] does not align with policy head"
                    % head.value
                )

            input_identity = input_identities[head]
            if type(input_identity) is not str:
                raise TypeError(
                    "render_input_identity_sha256_by_head[%s] must be a "
                    "string" % head.value
                )
            if _SHA256_RE.fullmatch(input_identity) is None:
                raise ValueError(
                    "render_input_identity_sha256_by_head[%s] must be a "
                    "lowercase SHA-256" % head.value
                )

        if type(self.policy_set_sha256) is not str:
            raise TypeError("policy_set_sha256 must be a string")
        if _SHA256_RE.fullmatch(self.policy_set_sha256) is None:
            raise ValueError("policy_set_sha256 must be a lowercase SHA-256")
        if (
            _policy_set_sha256(policies, input_identities)
            != self.policy_set_sha256
        ):
            raise ValueError("policy_set_sha256 does not reconcile")

        object.__setattr__(
            self, "policies", MappingProxyType(policies)
        )
        object.__setattr__(
            self,
            "render_input_identity_sha256_by_head",
            MappingProxyType(input_identities),
        )
        object.__setattr__(
            self,
            "topology_predecessors",
            MappingProxyType(topology_predecessors),
        )


@dataclass(frozen=True)
class RenderedPolicy:
    """Canonical policy and its fully reconciling measurement evidence."""

    policy_bytes: bytes
    policy_sha256: str
    rendered_policy_bytes: int
    statement_ledger: Tuple[StatementLedgerEntry, ...]
    component_bytes: Mapping[str, int]
    ledger_summary: PolicyLedgerSummary

    def __post_init__(self) -> None:
        if type(self.policy_bytes) is not bytes:
            raise TypeError("policy_bytes must be immutable bytes")
        if type(self.policy_sha256) is not str:
            raise TypeError("policy_sha256 must be a string")
        _require_nonnegative_exact_int(
            "rendered_policy_bytes", self.rendered_policy_bytes
        )
        _require_exact_tuple("statement_ledger", self.statement_ledger)
        for index, entry in enumerate(self.statement_ledger):
            _validate_statement_ledger_entry_types(entry, index)
        component_bytes = _snapshot_component_mapping(
            "component_bytes", self.component_bytes
        )
        if type(self.ledger_summary) is not PolicyLedgerSummary:
            raise TypeError("ledger_summary must be a PolicyLedgerSummary")

        if self.rendered_policy_bytes != len(self.policy_bytes):
            raise ValueError("rendered_policy_bytes does not match policy_bytes")
        if hashlib.sha256(self.policy_bytes).hexdigest() != self.policy_sha256:
            raise ValueError("policy_sha256 does not match policy_bytes")
        statements = _decode_canonical_policy(self.policy_bytes)
        if len(statements) != len(self.statement_ledger):
            raise ValueError(
                "statement_ledger does not have one entry per statement"
            )

        expected_component_bytes = {
            component: 0 for component in _COMPONENT_ORDER
        }
        expected_component_bytes["global_guards"] = len(
            _POLICY_PREFIX_BYTES
        ) + len(_POLICY_SUFFIX_BYTES)
        expected_component_counts = {
            component: 0 for component in _COMPONENT_ORDER
        }
        offset = len(_POLICY_PREFIX_BYTES)
        for sequence, statement in enumerate(statements):
            entry = self.statement_ledger[sequence]
            canonical_statement_bytes = _policy_json_bytes(statement)
            leading_comma_bytes = 0 if sequence == 0 else 1
            start_offset = offset + leading_comma_bytes
            end_offset = start_offset + len(canonical_statement_bytes)
            expected_fields = (
                ("sequence", sequence),
                ("sid", statement["Sid"]),
                ("leading_comma_bytes", leading_comma_bytes),
                ("statement_bytes", len(canonical_statement_bytes)),
                ("start_offset", start_offset),
                ("end_offset", end_offset),
                (
                    "statement_sha256",
                    hashlib.sha256(canonical_statement_bytes).hexdigest(),
                ),
            )
            for field_name, expected_value in expected_fields:
                if getattr(entry, field_name) != expected_value:
                    raise ValueError(
                        "statement_ledger[%d].%s does not reconcile"
                        % (sequence, field_name)
                    )
            if (
                self.policy_bytes[start_offset:end_offset]
                != canonical_statement_bytes
            ):
                raise ValueError(
                    "statement_ledger[%d] does not select canonical bytes"
                    % sequence
                )
            expected_component_bytes[entry.component] += (
                leading_comma_bytes + len(canonical_statement_bytes)
            )
            expected_component_counts[entry.component] += 1
            offset = end_offset
        if offset + len(_POLICY_SUFFIX_BYTES) != len(self.policy_bytes):
            raise ValueError("statement ledger offsets do not span policy_bytes")
        if component_bytes != expected_component_bytes:
            raise ValueError("component_bytes does not reconcile with the ledger")

        summary = self.ledger_summary
        if summary.policy_prefix_bytes != len(_POLICY_PREFIX_BYTES):
            raise ValueError("ledger summary policy_prefix_bytes does not reconcile")
        if summary.policy_suffix_bytes != len(_POLICY_SUFFIX_BYTES):
            raise ValueError("ledger summary policy_suffix_bytes does not reconcile")
        if summary.statement_count != len(self.statement_ledger):
            raise ValueError("ledger summary statement_count does not reconcile")
        if (
            dict(summary.component_statement_counts)
            != expected_component_counts
        ):
            raise ValueError(
                "ledger summary component_statement_counts does not reconcile"
            )
        if (
            summary.working_headroom_bytes
            != summary.working_limit_bytes - self.rendered_policy_bytes
        ):
            raise ValueError(
                "ledger summary working_headroom_bytes does not reconcile"
            )
        if (
            summary.design_headroom_bytes
            != summary.max_policy_bytes - self.rendered_policy_bytes
        ):
            raise ValueError(
                "ledger summary design_headroom_bytes does not reconcile"
            )
        if (
            summary.s3_headroom_bytes
            != _S3_MAX_POLICY_BYTES - self.rendered_policy_bytes
        ):
            raise ValueError(
                "ledger summary s3_headroom_bytes does not reconcile"
            )

        object.__setattr__(
            self,
            "component_bytes",
            MappingProxyType(component_bytes),
        )

    @property
    def policy(self) -> Mapping[str, object]:
        """Return a fresh mutable projection without exposing hashed state."""

        value = json.loads(self.policy_bytes)
        assert isinstance(value, dict)
        return value


@dataclass(frozen=True)
class ProductionPolicyCheck:
    """Offline proof that a rendered policy is the sole planned stack modify."""

    policy_sha256: str
    rendered_policy_bytes: int
    s3_headroom_bytes: int
    template_body_sha256: str
    logical_resource_id: str
    replacement: str


@dataclass(frozen=True)
class _PendingStatement:
    component: str
    value: Mapping[str, object]


class _StatementBuilder:
    def __init__(self) -> None:
        self.pending = []  # type: list[_PendingStatement]
        self.sids = set()  # type: set[str]

    def add(
        self,
        *,
        component: str,
        sid: str,
        actions: Sequence[str],
        resources: Sequence[str],
        condition: Optional[Mapping[str, object]] = None,
    ) -> None:
        if component not in _COMPONENT_ORDER:
            raise ValueError("unknown policy component: %s" % component)
        _validate_identifier("Sid", sid)
        if sid in self.sids:
            raise ValueError("duplicate Sid: %s" % sid)
        self.sids.add(sid)
        action_values = _dedupe_strings(actions, "actions")
        resource_values = _dedupe_strings(resources, "resources")
        if not action_values:
            raise ValueError("statement actions must not be empty")
        if not resource_values:
            raise ValueError("statement resources must not be empty")
        statement = {
            "Sid": sid,
            "Effect": "Deny",
            "Principal": "*",
            "Action": list(action_values),
            "Resource": list(resource_values),
        }  # type: Dict[str, object]
        if condition is not None:
            if not condition:
                raise ValueError("statement condition must not be empty")
            _validate_json_value(condition, "Condition")
            statement["Condition"] = condition
        self.pending.append(_PendingStatement(component=component, value=statement))

    def add_legacy(self, statement: Mapping[str, object]) -> None:
        copied = _validate_legacy_statement(statement)
        sid = copied["Sid"]
        assert isinstance(sid, str)
        if sid in self.sids:
            raise ValueError("duplicate Sid: %s" % sid)
        self.sids.add(sid)
        self.pending.append(
            _PendingStatement(component="legacy_fragment", value=copied)
        )


def _require_ascii(label: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("%s must be a string" % label)
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("%s must be ASCII" % label) from exc


def _validate_identifier(label: str, value: str) -> None:
    _require_ascii(label, value)
    if _ID_RE.fullmatch(value) is None:
        raise ValueError("%s has invalid characters or length" % label)


def _validate_json_value(value: object, path: str) -> None:
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return
    if isinstance(value, float):
        raise ValueError("%s must not contain floats" % path)
    if isinstance(value, str):
        _require_ascii(path, value)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("%s JSON object keys must be strings" % path)
            _require_ascii(path + " key", key)
            _validate_json_value(item, path + "." + key)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, "%s[%d]" % (path, index))
        return
    raise TypeError("%s contains a non-JSON value" % path)


def _require_tuple(label: str, value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("%s must be an immutable tuple" % label)


def _dedupe_strings(values: Iterable[str], label: str) -> Tuple[str, ...]:
    result = []
    seen = set()
    for value in values:
        _require_ascii(label, value)
        if not value:
            raise ValueError("%s must not contain an empty string" % label)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _dedupe_identities(
    identities: Iterable[PrincipalIdentity],
) -> Tuple[PrincipalIdentity, ...]:
    result = []
    by_arn = {}  # type: Dict[str, PrincipalIdentity]
    by_binding_id = {}  # type: Dict[str, PrincipalIdentity]
    by_role_id = {}  # type: Dict[str, PrincipalIdentity]
    for identity in identities:
        _validate_identity(identity)
        previous_binding = by_binding_id.get(identity.binding_id)
        previous_arn = by_arn.get(identity.arn)
        previous_role_id = by_role_id.get(identity.role_id)
        if previous_binding is not None and previous_binding != identity:
            raise ValueError(
                "binding_id %s names multiple identities"
                % identity.binding_id
            )
        if previous_arn is not None and previous_arn != identity:
            raise ValueError("%s appears with another RoleId" % identity.arn)
        if previous_role_id is not None and previous_role_id != identity:
            raise ValueError(
                "RoleId %s names another principal ARN" % identity.role_id
            )
        if previous_binding is not None:
            continue
        by_binding_id[identity.binding_id] = identity
        by_arn[identity.arn] = identity
        by_role_id[identity.role_id] = identity
        result.append(identity)
    return tuple(result)


def _validate_identity(identity: PrincipalIdentity) -> None:
    if not isinstance(identity, PrincipalIdentity):
        raise TypeError("principal identity has wrong type")
    _validate_identifier("binding_id", identity.binding_id)
    _require_ascii("principal ARN", identity.arn)
    if _ROLE_ARN_RE.fullmatch(identity.arn) is None or len(identity.arn) > 512:
        raise ValueError("principal ARN must be the exact fixed-account IAM role ARN")
    _require_ascii("RoleId", identity.role_id)
    if _ROLE_ID_RE.fullmatch(identity.role_id) is None:
        raise ValueError("RoleId has invalid characters or length")


def _validate_legacy_statement(
    statement: Mapping[str, object],
) -> Mapping[str, object]:
    if type(statement) is not MappingProxyType:
        raise TypeError("legacy statement must be an owned frozen object")
    snapshot = _thaw_json(statement)
    if type(snapshot) is not dict:
        raise TypeError("legacy statement snapshot must be an object")
    required = {"Sid", "Effect", "Principal", "Action", "Resource"}
    allowed = required | {"Condition"}
    keys = set(snapshot)
    if keys - allowed:
        raise ValueError(
            "legacy statement has unknown fields: %s"
            % sorted(keys - allowed)
        )
    if required - keys:
        raise ValueError(
            "legacy statement is missing fields: %s"
            % sorted(required - keys)
        )
    _validate_json_value(snapshot, "legacy statement")
    if snapshot["Effect"] != "Deny" or snapshot["Principal"] != "*":
        raise ValueError("legacy statement must be deny-only with Principal '*'")
    sid = snapshot["Sid"]
    if not isinstance(sid, str):
        raise TypeError("legacy Sid must be a string")
    _validate_identifier("legacy Sid", sid)
    for key in ("Action", "Resource"):
        values = snapshot[key]
        if type(values) is not list or not values:
            raise ValueError("legacy %s must be a non-empty array" % key)
        _dedupe_strings(values, "legacy %s" % key)
    if "Condition" in snapshot and (
        type(snapshot["Condition"]) is not dict or not snapshot["Condition"]
    ):
        raise ValueError("legacy Condition must be a non-empty object")
    return snapshot


def _validate_object_resources(
    resources: Iterable[str], bucket_arn: str, label: str
) -> Tuple[str, ...]:
    values = _dedupe_strings(resources, label)
    for resource in values:
        if not resource.startswith(bucket_arn + "/"):
            raise ValueError("%s must remain inside the exact bucket" % label)
        if resource == bucket_arn + "/*":
            raise ValueError("%s must not broaden to the whole bucket" % label)
        if any(token in resource for token in ("\\", "?", " ")):
            raise ValueError("%s contains an unsafe resource spelling" % label)
    return values


def _validate_limits(limits: PolicyLimits) -> None:
    if not isinstance(limits, PolicyLimits):
        raise TypeError("policy_limits has wrong type")
    if isinstance(limits.max_policy_bytes, bool) or not isinstance(
        limits.max_policy_bytes, int
    ):
        raise TypeError("max_policy_bytes must be an integer")
    if isinstance(limits.unallocated_headroom_bytes, bool) or not isinstance(
        limits.unallocated_headroom_bytes, int
    ):
        raise TypeError("unallocated_headroom_bytes must be an integer")
    if (
        limits.max_policy_bytes < 1
        or limits.max_policy_bytes > _S3_MAX_POLICY_BYTES
    ):
        raise ValueError("max_policy_bytes must be in 1..20480")
    if limits.unallocated_headroom_bytes < 0:
        raise ValueError("unallocated_headroom_bytes must be nonnegative")
    if limits.unallocated_headroom_bytes >= limits.max_policy_bytes:
        raise ValueError(
            "unallocated_headroom_bytes must be smaller than max_policy_bytes"
        )
    if (
        limits.max_policy_bytes != 18432
        or limits.unallocated_headroom_bytes != 512
        or dict(limits.component_max_bytes) != dict(_DEFAULT_COMPONENT_MAX_BYTES)
    ):
        raise ValueError("policy_limits must equal the frozen H.1g budget")
    if set(limits.component_max_bytes) != set(_COMPONENT_ORDER):
        raise ValueError("component_max_bytes must have the exact closed component set")
    for component, value in limits.component_max_bytes.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("component budget for %s must be nonnegative integer" % component)


_STAGE_EVIDENCE_FIELDS: Tuple[str, ...] = (
    "predecessor_policy_sha256",
    "freeze_denial_evidence_sha256",
    "source_publication_sealed_sha256",
    "all_version_inventory_sha256",
    "source_settlement_sha256",
    "publisher_deny_policy_sha256",
    "terminal_prerequisite_sha256",
)

_REQUIRED_STAGE_EVIDENCE: Mapping[PolicyHead, frozenset[str]] = MappingProxyType(
    {
        PolicyHead.BRIDGE_SEED: frozenset(),
        PolicyHead.PREPARE: frozenset(
            {"predecessor_policy_sha256", "publisher_deny_policy_sha256"}
        ),
        PolicyHead.RESERVATION: frozenset(
            {"predecessor_policy_sha256", "publisher_deny_policy_sha256"}
        ),
        PolicyHead.SOURCE_FAMILIES_FROZEN: frozenset(
            {
                "predecessor_policy_sha256",
                "freeze_denial_evidence_sha256",
                "publisher_deny_policy_sha256",
            }
        ),
        PolicyHead.BATCH: frozenset(
            {
                "predecessor_policy_sha256",
                "freeze_denial_evidence_sha256",
                "source_publication_sealed_sha256",
                "all_version_inventory_sha256",
                "source_settlement_sha256",
                "publisher_deny_policy_sha256",
            }
        ),
        PolicyHead.CLOSED_SOURCE: frozenset(
            {
                "predecessor_policy_sha256",
                "freeze_denial_evidence_sha256",
                "publisher_deny_policy_sha256",
                "terminal_prerequisite_sha256",
            }
        ),
        PolicyHead.TERMINAL: frozenset(_STAGE_EVIDENCE_FIELDS),
    }
)


def _validate_stage_evidence(value: FencePolicyInput) -> None:
    required = _REQUIRED_STAGE_EVIDENCE[value.policy_head]
    for field_name in _STAGE_EVIDENCE_FIELDS:
        field_value = getattr(value, field_name)
        if field_name not in required:
            if field_value is not None:
                raise ValueError(
                    "%s forbids stage evidence %s"
                    % (value.policy_head.value, field_name)
                )
            continue
        if not isinstance(field_value, str):
            raise TypeError(
                "%s requires string stage evidence %s"
                % (value.policy_head.value, field_name)
            )
        if _SHA256_RE.fullmatch(field_value) is None:
            raise ValueError(
                "%s requires lowercase SHA-256 stage evidence %s"
                % (value.policy_head.value, field_name)
            )


def _validate_input(value: FencePolicyInput) -> str:
    if not isinstance(value, FencePolicyInput):
        raise TypeError("render input must be FencePolicyInput")
    if not isinstance(value.build_mode, BuildMode):
        raise TypeError("build_mode must be BuildMode")
    if not isinstance(value.policy_head, PolicyHead):
        raise TypeError("policy_head must be PolicyHead")
    _validate_stage_evidence(value)
    _require_ascii("bucket_name", value.bucket_name)
    if _BUCKET_RE.fullmatch(value.bucket_name) is None:
        raise ValueError("bucket_name is not a valid exact bucket name")
    if value.bucket_name != FIXED_BUCKET_NAME:
        raise ValueError("bucket_name must equal the fixed H.1g model bucket")
    _require_ascii("account_id", value.account_id)
    if _ACCOUNT_RE.fullmatch(value.account_id) is None:
        raise ValueError("account_id must be exactly 12 decimal digits")
    if value.account_id != FIXED_ACCOUNT_ID:
        raise ValueError("account_id must equal the fixed H.1g member account")
    _require_ascii("kms_key_arn", value.kms_key_arn)
    if _KMS_KEY_ARN_RE.fullmatch(value.kms_key_arn) is None:
        raise ValueError(
            "kms_key_arn must be an exact us-west-2 fixed-account KMS key ARN"
        )
    _validate_limits(value.policy_limits)
    for tuple_field in (
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
    ):
        _require_tuple(tuple_field, getattr(value, tuple_field))

    if value.build_mode is BuildMode.LEGACY_ENABLED and not value.legacy_statements:
        raise ValueError("LEGACY_ENABLED requires authenticated legacy statements")
    if value.build_mode is BuildMode.LEGACY_DISABLED and value.legacy_statements:
        raise ValueError("LEGACY_DISABLED forbids legacy statements")

    bucket_arn = "arn:aws:s3:::%s" % value.bucket_name
    if bool(value.retired_publishers) != bool(value.retired_publisher_resources):
        raise ValueError("retired publishers and resources must both be present or absent")
    _validate_object_resources(
        value.permanent_enrolled_resources,
        bucket_arn,
        "permanent_enrolled_resources",
    )
    _validate_object_resources(
        value.retired_publisher_resources,
        bucket_arn,
        "retired_publisher_resources",
    )
    _validate_object_resources(
        value.writer_owned_resources,
        bucket_arn,
        "writer_owned_resources",
    )

    all_identities = list(value.retired_publishers)
    family_ids = set()
    for family in value.reserved_families:
        if not isinstance(family, ReservedFamily):
            raise TypeError("reserved family has wrong type")
        _require_tuple("reserved family resources", family.resources)
        _validate_identifier("family_id", family.family_id)
        if family.family_id in family_ids:
            raise ValueError("duplicate reserved family: %s" % family.family_id)
        family_ids.add(family.family_id)
        if not family.resources:
            raise ValueError("reserved family resources must not be empty")
        _validate_object_resources(
            family.resources, bucket_arn, "reserved family resources"
        )
    expected_source_families = tuple(
        (family_id, (resource,))
        for family_id, resource in SOURCE_FAMILY_SPECS
    )
    if value.policy_head is not PolicyHead.BRIDGE_SEED and (
        value.writer_owned_resources[: len(CORE_WRITER_RESOURCES)]
        != CORE_WRITER_RESOURCES
    ):
        raise ValueError(
            "writer_owned_resources must begin with the exact core writer "
            "resource order"
        )
    observed_source_prefix = tuple(
        (family.family_id, tuple(family.resources))
        for family in value.reserved_families[: len(SOURCE_FAMILY_SPECS)]
    )
    if value.policy_head.value in _SOURCE_CLOSED_HEADS:
        if observed_source_prefix != expected_source_families:
            raise ValueError(
                "%s requires the exact five source-family closures in frozen order"
                % value.policy_head.value
            )
    elif any(
        family.family_id in SOURCE_FAMILY_IDS
        for family in value.reserved_families
    ):
        raise ValueError(
            "RESERVATION must not contain all-principal source-family closures"
        )
    if value.policy_head in {PolicyHead.BATCH, PolicyHead.TERMINAL}:
        if len(value.selected_source_keys) != len(SOURCE_FAMILY_IDS):
            raise ValueError(
                "%s requires exactly five selected source keys"
                % value.policy_head.value
            )
        for family_id, key in zip(
            SOURCE_FAMILY_IDS, value.selected_source_keys
        ):
            validate_frozen_key(
                key=key,
                run_id=FIXED_RUN_ID,
                allowed_families=(family_id,),
            )
        for key in value.nonselected_source_keys:
            validate_frozen_key(
                key=key,
                run_id=FIXED_RUN_ID,
                allowed_families=SOURCE_FAMILY_IDS,
            )
        all_source_keys = (
            value.selected_source_keys + value.nonselected_source_keys
        )
        if len(set(all_source_keys)) != len(all_source_keys):
            raise ValueError(
                "selected/nonselected source keys must be pairwise distinct"
            )
    elif value.selected_source_keys or value.nonselected_source_keys:
        raise ValueError(
            "%s forbids selected/nonselected source keys"
            % value.policy_head.value
        )

    cohort_ids = set()
    for cohort in value.writer_cohorts:
        if not isinstance(cohort, WriterCohort):
            raise TypeError("writer cohort has wrong type")
        _require_tuple("writer cohort members", cohort.members)
        _require_tuple("writer cohort guard_resources", cohort.guard_resources)
        _validate_identifier("cohort_id", cohort.cohort_id)
        if cohort.cohort_id in cohort_ids:
            raise ValueError("duplicate writer cohort: %s" % cohort.cohort_id)
        cohort_ids.add(cohort.cohort_id)
        if not cohort.members:
            raise ValueError("writer cohort members must not be empty")
        if not cohort.guard_resources:
            raise ValueError("writer cohort guard_resources must not be empty")
        if _SHA256_RE.fullmatch(cohort.cross_member_denial_evidence_sha256) is None:
            raise ValueError("cross_member_denial_evidence_sha256 must be lowercase SHA-256")
        _validate_object_resources(
            cohort.guard_resources, bucket_arn, "writer cohort guard_resources"
        )
        if tuple(cohort.guard_resources) != (RUN_GUARD_RESOURCE,):
            raise ValueError(
                "writer cohort guard_resources must equal the frozen run guard"
            )
        all_identities.extend(cohort.members)

    for label, readers in (
        ("source_validation_readers", value.source_validation_readers),
        ("source_inventory_readers", value.source_inventory_readers),
        ("terminal_audit_readers", value.terminal_audit_readers),
    ):
        for reader in readers:
            if not isinstance(reader, PrincipalIdentity):
                raise TypeError("%s has a reader with wrong type" % label)
        if len(_dedupe_identities(readers)) != len(readers):
            raise ValueError("%s must not contain duplicates" % label)
        all_identities.extend(readers)

    if value.policy_head is PolicyHead.BRIDGE_SEED:
        if (
            value.permanent_enrolled_resources
            or value.retired_publishers
            or value.retired_publisher_resources
            or value.writer_owned_resources
            or value.source_validation_readers
            or value.source_inventory_readers
            or value.terminal_audit_readers
            or value.selected_source_keys
            or value.nonselected_source_keys
        ):
            raise ValueError(
                "BRIDGE_SEED forbids post-seed lineage, reader, and source fields"
            )
        if (
            len(value.writer_cohorts) == 1
            and len(value.writer_cohorts[0].members) != 2
        ):
            raise ValueError(
                "BRIDGE_SEED requires the exact two-member artifact-writer cohort"
            )
    else:
        if not value.permanent_enrolled_resources:
            raise ValueError(
                "%s requires permanent enrolled-source resources"
                % value.policy_head.value
            )
        if not value.retired_publishers:
            raise ValueError(
                "%s requires retired publisher bindings and resources"
                % value.policy_head.value
            )
        if not value.writer_owned_resources:
            raise ValueError(
                "%s requires the authenticated writer-owned resource union"
                % value.policy_head.value
            )
        if (
            not value.source_validation_readers
            or not value.source_inventory_readers
            or not value.terminal_audit_readers
        ):
            raise ValueError(
                "%s requires all three closed reader inventories"
                % value.policy_head.value
            )

    if value.policy_head.value in _TERMINAL_HEADS:
        if value.writer_cohorts:
            raise ValueError(
                "%s forbids an active writer cohort" % value.policy_head.value
            )
    elif len(value.writer_cohorts) != 1:
        raise ValueError(
            "%s requires exactly one active writer cohort"
            % value.policy_head.value
        )

    identities = _dedupe_identities(all_identities)
    for identity in identities:
        if identity.arn.split(":", 5)[4] != value.account_id:
            raise ValueError("principal ARN belongs to another account")
    return bucket_arn


def _identity_values(
    identities: Iterable[PrincipalIdentity],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    deduped = _dedupe_identities(identities)
    if not deduped:
        raise ValueError("principal identity array must not be empty")
    return (
        tuple(identity.arn for identity in deduped),
        tuple(identity.role_id + ":*" for identity in deduped),
    )


def _add_global_guards(
    builder: _StatementBuilder, bucket_arn: str, account_id: str
) -> None:
    resources = (bucket_arn, bucket_arn + "/*")
    builder.add(
        component="global_guards",
        sid="DenyInsecureTransport",
        actions=("s3:*",),
        resources=resources,
        condition={"Bool": {"aws:SecureTransport": "false"}},
    )
    builder.add(
        component="global_guards",
        sid="DenyWrongPrincipalAccount",
        actions=("s3:*",),
        resources=resources,
        condition={
            "StringNotEquals": {"aws:PrincipalAccount": account_id},
            "Null": {"aws:PrincipalAccount": "false"},
            "BoolIfExists": {"aws:PrincipalIsAWSService": "false"},
        },
    )
    builder.add(
        component="global_guards",
        sid="DenyMissingPrincipalAccount",
        actions=("s3:*",),
        resources=resources,
        condition={
            "Null": {"aws:PrincipalAccount": "true"},
            "BoolIfExists": {"aws:PrincipalIsAWSService": "false"},
        },
    )


def _add_writer_cohort(
    builder: _StatementBuilder, cohort: WriterCohort, kms_key_arn: str
) -> None:
    arns, role_ids = _identity_values(cohort.members)
    suffix = cohort.cohort_id
    resources = _dedupe_strings(cohort.guard_resources, "guard_resources")
    builder.add(
        component="active_writer_cohort",
        sid="DenyCreateWrongWriterArn_" + suffix,
        actions=CREATE,
        resources=resources,
        condition={"ArnNotEqualsIfExists": {"aws:PrincipalArn": list(arns)}},
    )
    builder.add(
        component="active_writer_cohort",
        sid="DenyCreateWrongWriterId_" + suffix,
        actions=CREATE,
        resources=resources,
        condition={"StringNotLikeIfExists": {"aws:userid": list(role_ids)}},
    )
    builder.add(
        component="active_writer_cohort",
        sid="DenyMissingIfNoneMatch_" + suffix,
        actions=CREATE,
        resources=resources,
        condition={"Null": {"s3:if-none-match": "true"}},
    )
    builder.add(
        component="active_writer_cohort",
        sid="DenyWrongIfNoneMatch_" + suffix,
        actions=CREATE,
        resources=resources,
        condition={"StringNotEquals": {"s3:if-none-match": "*"}},
    )
    builder.add(
        component="active_writer_cohort",
        sid="DenyCopySourceHeader_" + suffix,
        actions=CREATE,
        resources=resources,
        condition={"Null": {"s3:x-amz-copy-source": "false"}},
    )
    builder.add(
        component="active_writer_cohort",
        sid="DenyWrongSseAlgorithm_" + suffix,
        actions=CREATE,
        resources=resources,
        condition={
            "StringNotEqualsIfExists": {
                "s3:x-amz-server-side-encryption": "aws:kms"
            }
        },
    )
    builder.add(
        component="active_writer_cohort",
        sid="DenyWrongSseKmsKey_" + suffix,
        actions=CREATE,
        resources=resources,
        condition={
            "StringNotEqualsIfExists": {
                "s3:x-amz-server-side-encryption-aws-kms-key-id": kms_key_arn
            }
        },
    )
    builder.add(
        component="active_writer_cohort",
        sid="DenyNonCreateMutation_" + suffix,
        actions=NON_CREATE_MUTATION,
        resources=resources,
    )


def _add_reader_scope(
    builder: _StatementBuilder, scope: ObjectReaderScope
) -> None:
    arns, role_ids = _identity_values(scope.readers)
    builder.add(
        component="object_reader_guards",
        sid="DenySourceReadWrongReaderArn_" + scope.scope_id,
        actions=SOURCE_OBJECT_READ,
        resources=scope.resources,
        condition={"ArnNotEqualsIfExists": {"aws:PrincipalArn": list(arns)}},
    )
    builder.add(
        component="object_reader_guards",
        sid="DenySourceReadWrongReaderId_" + scope.scope_id,
        actions=SOURCE_OBJECT_READ,
        resources=scope.resources,
        condition={"StringNotLikeIfExists": {"aws:userid": list(role_ids)}},
    )


def _add_list_scope(
    builder: _StatementBuilder, scope: ListReaderScope, bucket_arn: str
) -> None:
    arns, role_ids = _identity_values(scope.readers)
    builder.add(
        component="bucket_list_reader_guards",
        sid="DenySourceListWrongReaderArn_" + scope.scope_id,
        actions=SOURCE_LIST,
        resources=(bucket_arn,),
        condition={"ArnNotEqualsIfExists": {"aws:PrincipalArn": list(arns)}},
    )
    builder.add(
        component="bucket_list_reader_guards",
        sid="DenySourceListWrongReaderId_" + scope.scope_id,
        actions=SOURCE_LIST,
        resources=(bucket_arn,),
        condition={"StringNotLikeIfExists": {"aws:userid": list(role_ids)}},
    )


def _reader_union(
    *inventories: Tuple[PrincipalIdentity, ...],
) -> Tuple[PrincipalIdentity, ...]:
    return _dedupe_identities(
        identity
        for inventory in inventories
        for identity in inventory
    )


def _derived_reader_scopes(
    value: FencePolicyInput,
) -> Tuple[Tuple[ObjectReaderScope, ...], Tuple[ListReaderScope, ...]]:
    if value.policy_head is PolicyHead.BRIDGE_SEED:
        return (), ()

    active_readers = _reader_union(
        value.source_validation_readers,
        value.source_inventory_readers,
    )
    activated_readers = _reader_union(
        active_readers,
        value.terminal_audit_readers,
    )
    inventory_audit_readers = _reader_union(
        value.source_inventory_readers,
        value.terminal_audit_readers,
    )
    source_patterns = tuple(
        resource for _, resource in SOURCE_FAMILY_SPECS
    )
    exact_resources = tuple(
        FIXED_BUCKET_ARN + "/" + key
        for key in (
            value.selected_source_keys + value.nonselected_source_keys
        )
    )

    if value.policy_head in {
        PolicyHead.PREPARE,
        PolicyHead.RESERVATION,
        PolicyHead.SOURCE_FAMILIES_FROZEN,
    }:
        object_scopes = (
            ObjectReaderScope(
                scope_id="source-active",
                readers=active_readers,
                resources=source_patterns,
            ),
        )
        list_scopes = (
            ListReaderScope(
                scope_id="source-inventory",
                readers=value.source_inventory_readers,
            ),
        )
    elif value.policy_head in {PolicyHead.BATCH, PolicyHead.TERMINAL}:
        selected_resources = exact_resources[
            : len(value.selected_source_keys)
        ]
        object_scopes = (
            ObjectReaderScope(
                scope_id="selected-source",
                readers=activated_readers,
                resources=selected_resources,
            ),
        )
        if value.nonselected_source_keys:
            object_scopes += (
                ObjectReaderScope(
                    scope_id="nonselected-source",
                    readers=value.terminal_audit_readers,
                    resources=exact_resources[
                        len(value.selected_source_keys) :
                    ],
                ),
            )
        list_scopes = (
            ListReaderScope(
                scope_id="inventory-audit",
                readers=inventory_audit_readers,
            ),
        )
    elif value.policy_head is PolicyHead.CLOSED_SOURCE:
        object_scopes = (
            ObjectReaderScope(
                scope_id="terminal-audit",
                readers=value.terminal_audit_readers,
                resources=source_patterns,
            ),
        )
        list_scopes = (
            ListReaderScope(
                scope_id="terminal-audit",
                readers=value.terminal_audit_readers,
            ),
        )
    return object_scopes, list_scopes


def _assemble(
    builder: _StatementBuilder, limits: PolicyLimits
) -> RenderedPolicy:
    prefix = _POLICY_PREFIX_BYTES
    suffix = _POLICY_SUFFIX_BYTES
    chunks = [prefix]
    ledger = []
    component_bytes = {component: 0 for component in _COMPONENT_ORDER}
    component_statement_counts = {
        component: 0 for component in _COMPONENT_ORDER
    }
    component_bytes["global_guards"] += len(prefix) + len(suffix)
    offset = len(prefix)

    for sequence, pending in enumerate(builder.pending):
        statement_bytes = _policy_json_bytes(pending.value)
        leading_comma_bytes = 0 if sequence == 0 else 1
        if leading_comma_bytes:
            chunks.append(b",")
            offset += 1
        start_offset = offset
        chunks.append(statement_bytes)
        offset += len(statement_bytes)
        component_bytes[pending.component] += (
            leading_comma_bytes + len(statement_bytes)
        )
        component_statement_counts[pending.component] += 1
        ledger.append(
            StatementLedgerEntry(
                sequence=sequence,
                sid=str(pending.value["Sid"]),
                component=pending.component,
                leading_comma_bytes=leading_comma_bytes,
                statement_bytes=len(statement_bytes),
                start_offset=start_offset,
                end_offset=offset,
                statement_sha256=hashlib.sha256(statement_bytes).hexdigest(),
            )
        )

    chunks.append(suffix)
    policy_bytes = b"".join(chunks)
    policy = {
        "Version": "2012-10-17",
        "Statement": [pending.value for pending in builder.pending],
    }
    if _policy_json_bytes(policy) != policy_bytes:
        raise AssertionError("manual policy assembly diverged from canonical JSON")
    if sum(component_bytes.values()) != len(policy_bytes):
        raise AssertionError("component byte ledger does not reconcile")

    working_limit = (
        limits.max_policy_bytes - limits.unallocated_headroom_bytes
    )
    if len(policy_bytes) > working_limit:
        raise PolicyComponentBudgetExhausted(
            "POLICY_COMPONENT_BUDGET_EXHAUSTED: policy uses %d bytes, "
            "working limit %d (%d-byte headroom)"
            % (
                len(policy_bytes),
                working_limit,
                limits.unallocated_headroom_bytes,
            )
        )
    offenders = [
        "%s=%d>%d"
        % (
            component,
            component_bytes[component],
            limits.component_max_bytes[component],
        )
        for component in _COMPONENT_ORDER
        if component_bytes[component] > limits.component_max_bytes[component]
    ]
    if offenders:
        raise PolicyComponentBudgetExhausted(
            "POLICY_COMPONENT_BUDGET_EXHAUSTED: " + ", ".join(offenders)
        )

    summary = PolicyLedgerSummary(
        policy_prefix_bytes=len(prefix),
        policy_suffix_bytes=len(suffix),
        statement_count=len(ledger),
        max_policy_bytes=limits.max_policy_bytes,
        working_limit_bytes=working_limit,
        reserved_design_headroom_bytes=limits.unallocated_headroom_bytes,
        working_headroom_bytes=working_limit - len(policy_bytes),
        design_headroom_bytes=limits.max_policy_bytes - len(policy_bytes),
        s3_headroom_bytes=_S3_MAX_POLICY_BYTES - len(policy_bytes),
        component_statement_counts=MappingProxyType(
            dict(component_statement_counts)
        ),
    )
    return RenderedPolicy(
        policy_bytes=policy_bytes,
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        rendered_policy_bytes=len(policy_bytes),
        statement_ledger=tuple(ledger),
        component_bytes=MappingProxyType(dict(component_bytes)),
        ledger_summary=summary,
    )


def render_fence_policy(value: FencePolicyInput) -> RenderedPolicy:
    """Render one canonical deny-only fence policy or fail closed."""

    bucket_arn = _validate_input(value)
    builder = _StatementBuilder()
    _add_global_guards(builder, bucket_arn, value.account_id)

    for statement in value.legacy_statements:
        builder.add_legacy(statement)

    if value.permanent_enrolled_resources:
        builder.add(
            component="permanent_lineage",
            sid="DenyPermanentEnrolledSourceMutation",
            actions=FULL_MUTATION,
            resources=value.permanent_enrolled_resources,
        )

    if value.retired_publishers:
        arns, role_ids = _identity_values(value.retired_publishers)
        builder.add(
            component="permanent_lineage",
            sid="DenyRetiredSourcePublisherArns",
            actions=FULL_MUTATION,
            resources=value.retired_publisher_resources,
            condition={"ArnEquals": {"aws:PrincipalArn": list(arns)}},
        )
        builder.add(
            component="permanent_lineage",
            sid="DenyRetiredSourcePublisherIds",
            actions=FULL_MUTATION,
            resources=value.retired_publisher_resources,
            condition={"StringLike": {"aws:userid": list(role_ids)}},
        )

    if value.reserved_families:
        builder.add(
            component="source_family_closures",
            sid="DenyAllReservedFamilyMutation_" + value.policy_head.value,
            actions=FULL_MUTATION,
            resources=tuple(
                resource
                for family in value.reserved_families
                for resource in family.resources
            ),
        )

    for cohort in value.writer_cohorts:
        _add_writer_cohort(builder, cohort, value.kms_key_arn)

    object_reader_scopes, list_reader_scopes = _derived_reader_scopes(value)
    for scope in object_reader_scopes:
        _add_reader_scope(builder, scope)
    for scope in list_reader_scopes:
        _add_list_scope(builder, scope, bucket_arn)

    if value.policy_head.value in _TERMINAL_HEADS:
        builder.add(
            component="terminal_closure",
            sid="DenyTerminalClosedMutation",
            actions=FULL_MUTATION,
            resources=value.writer_owned_resources,
        )

    return _assemble(builder, value.policy_limits)


_POLICY_SET_ORDER: Tuple[PolicyHead, ...] = (
    PolicyHead.BRIDGE_SEED,
    PolicyHead.PREPARE,
    PolicyHead.RESERVATION,
    PolicyHead.SOURCE_FAMILIES_FROZEN,
    PolicyHead.BATCH,
    PolicyHead.CLOSED_SOURCE,
    PolicyHead.TERMINAL,
)
_TOPOLOGY_PREDECESSORS: Mapping[
    PolicyHead, Optional[PolicyHead]
] = MappingProxyType(
    {
        PolicyHead.BRIDGE_SEED: None,
        PolicyHead.PREPARE: PolicyHead.BRIDGE_SEED,
        PolicyHead.RESERVATION: PolicyHead.PREPARE,
        PolicyHead.SOURCE_FAMILIES_FROZEN: PolicyHead.RESERVATION,
        PolicyHead.BATCH: PolicyHead.SOURCE_FAMILIES_FROZEN,
        PolicyHead.CLOSED_SOURCE: PolicyHead.SOURCE_FAMILIES_FROZEN,
        PolicyHead.TERMINAL: PolicyHead.BATCH,
    }
)


def _policy_set_sha256(
    policies: Mapping[PolicyHead, RenderedPolicy],
    input_identities: Mapping[PolicyHead, str],
) -> str:
    digest_input = [
        {
            "policy_head": head.value,
            "policy_sha256": policies[head].policy_sha256,
            "render_input_identity_sha256": input_identities[head],
            "rendered_policy_bytes": policies[head].rendered_policy_bytes,
        }
        for head in _POLICY_SET_ORDER
    ]
    return hashlib.sha256(_policy_json_bytes(digest_input)).hexdigest()

_ALLOWED_REMOVED_COMPONENTS: Mapping[
    PolicyHead, frozenset[str]
] = MappingProxyType(
    {
        PolicyHead.PREPARE: frozenset({"active_writer_cohort"}),
        PolicyHead.RESERVATION: frozenset(
            {"active_writer_cohort", "source_family_closures"}
        ),
        PolicyHead.SOURCE_FAMILIES_FROZEN: frozenset(
            {"active_writer_cohort"}
        ),
        PolicyHead.BATCH: frozenset(
            {
                "active_writer_cohort",
                "permanent_lineage",
                "object_reader_guards",
                "bucket_list_reader_guards",
            }
        ),
        PolicyHead.CLOSED_SOURCE: frozenset(
            {
                "active_writer_cohort",
                "object_reader_guards",
                "bucket_list_reader_guards",
            }
        ),
        PolicyHead.TERMINAL: frozenset({"active_writer_cohort"}),
    }
)


def _validate_policy_set_projections(
    values: Mapping[PolicyHead, FencePolicyInput],
) -> None:
    prepare = values[PolicyHead.PREPARE]
    reservation = values[PolicyHead.RESERVATION]
    frozen = values[PolicyHead.SOURCE_FAMILIES_FROZEN]
    batch = values[PolicyHead.BATCH]
    closed = values[PolicyHead.CLOSED_SOURCE]
    terminal = values[PolicyHead.TERMINAL]

    for value in values.values():
        if (
            value.build_mode is not prepare.build_mode
            or value.legacy_statements != prepare.legacy_statements
        ):
            raise ValueError(
                "POLICY_SEMANTIC_ROLLBACK: global/legacy projection drifted"
            )
    for value in (reservation, frozen, batch, closed, terminal):
        if value.writer_owned_resources != prepare.writer_owned_resources:
            raise ValueError(
                "POLICY_SEMANTIC_ROLLBACK: %s writer-owned terminal "
                "resource projection drifted" % value.policy_head.value
            )

    prepare_members = prepare.writer_cohorts[0].members
    reservation_members = reservation.writer_cohorts[0].members
    if (
        len(reservation_members) != len(prepare_members) + 5
        or reservation_members[: len(prepare_members)] != prepare_members
    ):
        raise ValueError(
            "POLICY_SEMANTIC_ROLLBACK: RESERVATION writer projection "
            "must preserve PREPARE and add exactly five source identities"
        )
    source_publishers = reservation_members[len(prepare_members) :]
    if frozen.writer_cohorts[0].members != prepare_members:
        raise ValueError(
            "POLICY_SEMANTIC_ROLLBACK: SOURCE_FAMILIES_FROZEN writer "
            "projection must remove exactly the five source identities"
        )
    if batch.writer_cohorts[0].members != frozen.writer_cohorts[0].members:
        raise ValueError(
            "POLICY_SEMANTIC_ROLLBACK: BATCH contains an unrelated writer "
            "identity or drops the frozen writer projection"
        )

    source_families = tuple(
        ReservedFamily(family_id=family_id, resources=(resource,))
        for family_id, resource in SOURCE_FAMILY_SPECS
    )
    prepare_non_source = prepare.reserved_families[
        len(SOURCE_FAMILY_SPECS) :
    ]
    if reservation.reserved_families != prepare_non_source:
        raise ValueError(
            "POLICY_SEMANTIC_ROLLBACK: RESERVATION family replacement "
            "is not the exact non-source projection"
        )
    expected_frozen_families = source_families + reservation.reserved_families
    for value in (frozen, batch, closed, terminal):
        if value.reserved_families != expected_frozen_families:
            raise ValueError(
                "POLICY_SEMANTIC_ROLLBACK: source-family closure projection "
                "drifted at %s" % value.policy_head.value
            )

    for value in (reservation, frozen, batch, closed, terminal):
        if (
            value.source_validation_readers
            != prepare.source_validation_readers
            or value.source_inventory_readers
            != prepare.source_inventory_readers
            or value.terminal_audit_readers
            != prepare.terminal_audit_readers
        ):
            raise ValueError(
                "POLICY_SEMANTIC_ROLLBACK: %s contains an unrelated reader "
                "identity or drops a closed reader inventory"
                % value.policy_head.value
            )
    if (
        terminal.selected_source_keys != batch.selected_source_keys
        or terminal.nonselected_source_keys != batch.nonselected_source_keys
    ):
        raise ValueError(
            "POLICY_SEMANTIC_ROLLBACK: TERMINAL source projection drifted"
        )

    baseline_publishers = prepare.retired_publishers
    baseline_resources = prepare.retired_publisher_resources
    for value in (reservation, frozen, closed):
        if (
            value.retired_publishers != baseline_publishers
            or value.retired_publisher_resources != baseline_resources
        ):
            raise ValueError(
                "POLICY_SEMANTIC_ROLLBACK: bootstrap publisher retirement "
                "projection drifted at %s" % value.policy_head.value
            )
    if batch.retired_publishers != baseline_publishers + source_publishers:
        raise ValueError(
            "POLICY_SEMANTIC_ROLLBACK: BATCH must retire the exact five "
            "source identities"
        )
    expected_retired_resources = baseline_resources + tuple(
        FIXED_BUCKET_ARN + "/" + key for key in batch.selected_source_keys
    )
    if (
        batch.retired_publisher_resources != expected_retired_resources
        or terminal.retired_publishers != batch.retired_publishers
        or terminal.retired_publisher_resources
        != batch.retired_publisher_resources
    ):
        raise ValueError(
            "POLICY_SEMANTIC_ROLLBACK: BATCH/TERMINAL retired-publisher "
            "projection is not exact"
        )




def _semantic_statements_by_component(
    policy: RenderedPolicy,
) -> Mapping[str, frozenset[bytes]]:
    result = {}  # type: Dict[str, set[bytes]]
    statements = policy.policy["Statement"]
    if len(statements) != len(policy.statement_ledger):
        raise AssertionError("policy statement ledger length drifted")
    for statement, ledger_entry in zip(
        statements, policy.statement_ledger
    ):
        semantic = _policy_json_bytes(
            {
                key: item
                for key, item in statement.items()
                if key != "Sid"
            }
        )
        result.setdefault(ledger_entry.component, set()).add(semantic)
    return MappingProxyType(
        {
            component: frozenset(statements)
            for component, statements in result.items()
        }
    )


def render_fence_policy_set(
    values: Tuple[FencePolicyInput, ...],
) -> RenderedPolicySet:
    """Render and cross-bind the complete seven-template policy topology."""

    _require_tuple("policy set inputs", values)
    actual_order = tuple(value.policy_head for value in values)
    if actual_order != _POLICY_SET_ORDER:
        raise ValueError("policy set inputs must use the exact seven-head order")

    rendered = {
        value.policy_head: render_fence_policy(value) for value in values
    }
    values_by_head = {value.policy_head: value for value in values}
    _validate_policy_set_projections(values_by_head)
    for value in values:
        predecessor_head = _TOPOLOGY_PREDECESSORS[value.policy_head]
        expected_hash = (
            None
            if predecessor_head is None
            else rendered[predecessor_head].policy_sha256
        )
        if value.predecessor_policy_sha256 != expected_hash:
            raise ValueError(
                "%s predecessor policy hash does not match %s"
                % (
                    value.policy_head.value,
                    "the bridge origin"
                    if predecessor_head is None
                    else predecessor_head.value,
                )
            )

    baseline_digest = values_by_head[
        PolicyHead.PREPARE
    ].publisher_deny_policy_sha256
    if any(
        values_by_head[head].publisher_deny_policy_sha256 != baseline_digest
        for head in (
            PolicyHead.RESERVATION,
            PolicyHead.SOURCE_FAMILIES_FROZEN,
            PolicyHead.CLOSED_SOURCE,
        )
    ):
        raise ValueError("bootstrap publisher-deny digest is not monotonic")
    augmented_digest = values_by_head[
        PolicyHead.BATCH
    ].publisher_deny_policy_sha256
    if (
        augmented_digest == baseline_digest
        or values_by_head[PolicyHead.TERMINAL].publisher_deny_policy_sha256
        != augmented_digest
    ):
        raise ValueError(
            "BATCH/TERMINAL publisher-deny digest transition is invalid"
        )

    for child, parent in _TOPOLOGY_PREDECESSORS.items():
        if parent is None:
            continue
        child_value = values_by_head[child]
        parent_value = values_by_head[parent]
        if not set(parent_value.permanent_enrolled_resources).issubset(
            child_value.permanent_enrolled_resources
        ):
            raise ValueError(
                "%s removes a permanent enrolled resource" % child.value
            )
        if not set(parent_value.retired_publisher_resources).issubset(
            child_value.retired_publisher_resources
        ):
            raise ValueError(
                "%s removes a retired publisher resource" % child.value
            )

    semantic_by_component = {
        head: _semantic_statements_by_component(policy)
        for head, policy in rendered.items()
    }
    semantic_sets = {
        head: frozenset().union(*by_component.values())
        for head, by_component in semantic_by_component.items()
    }
    for child, parent in _TOPOLOGY_PREDECESSORS.items():
        if parent is None:
            continue
        if semantic_sets[child] == semantic_sets[parent]:
            raise ValueError(
                "%s has no semantic delta from %s"
                % (child.value, parent.value)
            )
        removed = semantic_sets[parent] - semantic_sets[child]
        removed_components = {
            component
            for component, statements in semantic_by_component[
                parent
            ].items()
            if statements & removed
        }
        unexpected = (
            removed_components - _ALLOWED_REMOVED_COMPONENTS[child]
        )
        if unexpected:
            raise ValueError(
                "POLICY_SEMANTIC_ROLLBACK: %s removes deny component(s) %s"
                % (child.value, ",".join(sorted(unexpected)))
            )

    hashes = tuple(rendered[head].policy_sha256 for head in _POLICY_SET_ORDER)
    if len(set(hashes)) != len(_POLICY_SET_ORDER):
        raise AssertionError("policy heads did not render pairwise-distinct bytes")
    input_identities = {
        head: values_by_head[head].render_input_identity_sha256
        for head in _POLICY_SET_ORDER
    }
    return RenderedPolicySet(
        policies=MappingProxyType(dict(rendered)),
        render_input_identity_sha256_by_head=MappingProxyType(
            input_identities
        ),
        topology_predecessors=_TOPOLOGY_PREDECESSORS,
        policy_set_sha256=_policy_set_sha256(rendered, input_identities),
    )


def check_production_policy_plan(
    *,
    rendered: RenderedPolicy,
    template_body: object,
    changes: object,
) -> ProductionPolicyCheck:
    """Reconcile exact policy bytes with the closed template and plan shape."""

    if not isinstance(rendered, RenderedPolicy):
        raise TypeError("rendered must be a RenderedPolicy")
    if type(template_body) is not dict or set(template_body) != {
        "AWSTemplateFormatVersion",
        "Resources",
    }:
        raise ValueError("production policy template shape is not exact")
    resources = template_body.get("Resources")
    if type(resources) is not dict or set(resources) != {
        "H1gProductionFenceBucketPolicy"
    }:
        raise ValueError("production policy template shape is not exact")
    resource = resources["H1gProductionFenceBucketPolicy"]
    if type(resource) is not dict or set(resource) != {
        "DeletionPolicy",
        "Properties",
        "Type",
        "UpdateReplacePolicy",
    }:
        raise ValueError("production policy template shape is not exact")
    properties = resource.get("Properties")
    if (
        template_body["AWSTemplateFormatVersion"] != "2010-09-09"
        or resource.get("DeletionPolicy") != "Retain"
        or resource.get("Type") != "AWS::S3::BucketPolicy"
        or resource.get("UpdateReplacePolicy") != "Retain"
        or type(properties) is not dict
        or set(properties) != {"Bucket", "PolicyDocument"}
        or properties.get("Bucket") != FIXED_BUCKET_NAME
    ):
        raise ValueError("production policy template shape is not exact")
    policy_document = properties["PolicyDocument"]
    if type(policy_document) is not dict:
        raise ValueError("production PolicyDocument must be an exact object")
    policy_bytes = _policy_json_bytes(policy_document)
    if (
        policy_bytes != rendered.policy_bytes
        or len(policy_bytes) != rendered.rendered_policy_bytes
        or hashlib.sha256(policy_bytes).hexdigest()
        != rendered.policy_sha256
    ):
        raise ValueError(
            "production PolicyDocument bytes do not match rendered policy"
        )

    expected_changes = [
        {
            "Type": "Resource",
            "ResourceChange": {
                "Action": "Modify",
                "LogicalResourceId": "H1gProductionFenceBucketPolicy",
                "PhysicalResourceId": FIXED_BUCKET_NAME,
                "ResourceType": "AWS::S3::BucketPolicy",
                "Replacement": "False",
                "Scope": ["Properties"],
                "Details": [],
            },
        }
    ]
    if changes != expected_changes:
        raise ValueError(
            "production change set is not one nonreplacement policy modify"
        )
    if rendered.rendered_policy_bytes > 20_480:
        raise ValueError("production policy exceeds the S3 hard limit")

    return ProductionPolicyCheck(
        policy_sha256=rendered.policy_sha256,
        rendered_policy_bytes=rendered.rendered_policy_bytes,
        s3_headroom_bytes=20_480 - rendered.rendered_policy_bytes,
        template_body_sha256=hashlib.sha256(
            _policy_json_bytes(template_body)
        ).hexdigest(),
        logical_resource_id="H1gProductionFenceBucketPolicy",
        replacement="False",
    )
