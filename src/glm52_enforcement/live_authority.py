"""Fresh, complete, no-retry H.1d live authority inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import re
from typing import Mapping, Optional, Tuple

from .canonical import canonical_json_bytes, canonical_sha256
from .spend_authority import validate_spend_authority_result


PROFILE = "keep-gpu"
ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
REQUIRED_LIVE_FAMILIES = (
    "cloudformation",
    "lambda",
    "iam",
    "eventbridge",
    "scheduler",
    "sqs",
    "sns",
    "ec2",
    "s3",
    "dynamodb",
    "ssm",
    "cloudwatch",
    "logs",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INSTANCE_ID = re.compile(r"i-(?:[0-9a-f]{8}|[0-9a-f]{17})\Z")
_VERSION_ID = re.compile(r"[\x21-\x7e]{1,1024}\Z")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]\Z")
RUNTIME_REVALIDATION_SOURCE_KINDS = (
    "TASK6_MIGRATION_MANIFEST",
    "TASK6_TEMPLATE_INVENTORY",
    "TASK7_POSTCREATE_MANIFEST",
    "TASK7_SUPPORT_INVENTORY",
    "RUNTIME_CUTOFF_AUTHORITY",
    "RUNTIME_ATTACHMENT_INVENTORY",
    "CREDENTIAL_PROBE_INVENTORY",
)


class H1dLiveAuthorityError(ValueError):
    """The H.1d live walk is incomplete, stale, divergent, or foreign."""


@dataclass(frozen=True)
class RuntimeRevalidationSourceCoordinate:
    input_kind: str
    bucket: str
    key: str
    version_id: str
    file_sha256: str
    body_sha256: str
    canonical_identity_sha256: str


@dataclass(frozen=True)
class CurrentClosureSessionBinding:
    activation_id: str
    decision_role_arn: str
    closure_role_arn: str
    closure_role_id: str
    cutoff_at: str
    credential_issue_time: str
    credential_expiration: str
    observed_at: str
    assume_role_request_id: str
    source_request_id: str
    closure_request_id: str
    current_session_proven: bool
    public_post_cutoff_expiration: bool
    absence_observed: bool
    canonical_identity_sha256: str


@dataclass(frozen=True)
class RuntimeRevalidationSemanticAuthority:
    runtime_credential_cutoff_at: str
    role_ids_by_arn: Mapping[str, str]
    attachment_cardinality: int
    attachment_cardinality_by_kind: Mapping[str, int]
    current_positive_role_ids: Tuple[str, ...]
    absence_observed: bool
    source_identities: Mapping[str, str]
    canonical_identity_sha256: str


def bind_current_closure_session_to_cutoff(
    *,
    activation_id: str,
    decision_role_arn: str,
    closure_role_arn: str,
    closure_role_id: str,
    cutoff_at: str,
    credential_issue_time: str,
    credential_expiration: str,
    observed_at: str,
    assume_role_request_id: str,
    source_caller_identity: object,
    closure_caller_identity: object,
) -> CurrentClosureSessionBinding:
    """Authenticate the live DecisionRole-to-ClosureSessionRole custody."""

    if type(activation_id) is not str or not activation_id.isascii():
        raise H1dLiveAuthorityError(
            "current closure session activation is invalid"
        )
    activation_sha = hashlib.sha256(
        activation_id.encode("ascii")
    ).hexdigest()[:16]
    expected_decision_role = (
        "arn:aws:iam::246813579024:role/"
        "keep-glm52-h1g-support-decision"
    )
    expected_closure_role = (
        "arn:aws:iam::246813579024:role/"
        "keep-glm52-h1g-closure-session-"
        + activation_sha
    )
    if (
        decision_role_arn != expected_decision_role
        or closure_role_arn != expected_closure_role
        or type(closure_role_id) is not str
        or not closure_role_id
        or type(assume_role_request_id) is not str
        or not assume_role_request_id
    ):
        raise H1dLiveAuthorityError(
            "current closure session role binding drifted"
        )
    cutoff = _time(cutoff_at, field="runtime credential cutoff")
    issued = _time(
        credential_issue_time,
        field="closure credential issue time",
    )
    expiration = _time(
        credential_expiration,
        field="closure credential expiration",
    )
    observed = _time(observed_at, field="closure session observation")
    lifetime = (expiration - issued).total_seconds()
    if (
        issued < cutoff
        or observed < issued
        or observed >= expiration
        or lifetime < 780
        or lifetime > 905
    ):
        raise H1dLiveAuthorityError(
            "current closure session is stale or outside cutoff"
        )

    def caller(
        value: object,
        *,
        role_name: str,
        role_id: Optional[str],
        label: str,
    ) -> tuple[str, str]:
        metadata = (
            value.get("ResponseMetadata")
            if type(value) is dict
            else None
        )
        arn_prefix = (
            "arn:aws:sts::246813579024:assumed-role/"
            + role_name
            + "/"
        )
        if (
            type(value) is not dict
            or value.get("Account") != ACCOUNT_ID
            or type(value.get("Arn")) is not str
            or not value["Arn"].startswith(arn_prefix)
            or type(value.get("UserId")) is not str
            or not value["UserId"]
            or (
                role_id is not None
                and not value["UserId"].startswith(role_id + ":")
            )
            or type(metadata) is not dict
            or type(metadata.get("HTTPStatusCode")) is not int
            or metadata["HTTPStatusCode"] != 200
            or type(metadata.get("RequestId")) is not str
            or not metadata["RequestId"]
        ):
            raise H1dLiveAuthorityError(
                label + " caller identity is unauthenticated"
            )
        return value["Arn"], metadata["RequestId"]

    _source_arn, source_request_id = caller(
        source_caller_identity,
        role_name="keep-glm52-h1g-support-decision",
        role_id=None,
        label="DecisionRole",
    )
    _closure_arn, closure_request_id = caller(
        closure_caller_identity,
        role_name=expected_closure_role.rsplit("/", 1)[1],
        role_id=closure_role_id,
        label="ClosureSessionRole",
    )
    if len(
        {
            assume_role_request_id,
            source_request_id,
            closure_request_id,
        }
    ) != 3:
        raise H1dLiveAuthorityError(
            "current closure session request evidence was reused"
        )
    body = {
        "activation_id": activation_id,
        "decision_role_arn": decision_role_arn,
        "closure_role_arn": closure_role_arn,
        "closure_role_id": closure_role_id,
        "cutoff_at": cutoff_at,
        "credential_issue_time": credential_issue_time,
        "credential_expiration": credential_expiration,
        "observed_at": observed_at,
        "assume_role_request_id": assume_role_request_id,
        "source_request_id": source_request_id,
        "closure_request_id": closure_request_id,
        "current_session_proven": True,
        "public_post_cutoff_expiration": True,
        "absence_observed": False,
    }
    return CurrentClosureSessionBinding(
        **body,
        canonical_identity_sha256=canonical_sha256(body),
    )


_SEMANTIC_RUNTIME_KINDS = (
    "RUNTIME_CUTOFF_AUTHORITY",
    "RUNTIME_ATTACHMENT_INVENTORY",
    "CREDENTIAL_PROBE_INVENTORY",
)
_SEMANTIC_SOURCE_FIELDS = {
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "run_id",
    "activation_id",
    "generation",
    "input_kind",
    "authority",
    "canonical_identity_sha256",
}
_ATTACHMENT_KINDS = {
    "LAMBDA_VERSION",
    "STEP_FUNCTIONS_VERSION",
    "INSTANCE_PROFILE_ASSOCIATION",
    "EC2_INSTANCE_PROFILE",
    "CLOUDFORMATION_ROLE",
    "SERVICE_LINKED_ROLE_REACHABILITY",
}
_ROLE_ID = re.compile(r"AROA[A-Z0-9]{8,128}\Z")


def _runtime_semantic_document(
    value: object,
    *,
    kind: str,
    activation_id: str,
    generation: int,
) -> tuple[Mapping[str, object], str]:
    if (
        type(value) is not dict
        or set(value) != _SEMANTIC_SOURCE_FIELDS
        or value["schema_version"] != 1
        or value["record_type"]
        != "glm52_task11_" + kind.lower() + "_v1"
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["run_id"] != RUN_ID
        or value["activation_id"] != activation_id
        or value["generation"] != generation
        or value["input_kind"] != kind
        or type(value["authority"]) is not dict
    ):
        raise H1dLiveAuthorityError(
            kind + " semantic source schema drifted"
        )
    body = dict(value)
    identity = body.pop("canonical_identity_sha256")
    if (
        type(identity) is not str
        or _SHA256.fullmatch(identity) is None
        or identity != canonical_sha256(body)
    ):
        raise H1dLiveAuthorityError(
            kind + " semantic source identity drifted"
        )
    return value["authority"], identity


def validate_runtime_revalidation_source_documents(
    value: object,
    *,
    activation_id: str,
    generation: int,
    task7_inventory_identity_sha256: str,
) -> RuntimeRevalidationSemanticAuthority:
    """Validate runtime attachment and credential source cross-bindings."""

    if (
        type(value) is not dict
        or set(value) != set(_SEMANTIC_RUNTIME_KINDS)
        or type(activation_id) is not str
        or not activation_id
        or type(generation) is not int
        or generation <= 0
        or type(task7_inventory_identity_sha256) is not str
        or _SHA256.fullmatch(task7_inventory_identity_sha256) is None
    ):
        raise H1dLiveAuthorityError(
            "runtime semantic source inventory drifted"
        )
    authorities: dict[str, Mapping[str, object]] = {}
    identities: dict[str, str] = {}
    for kind in _SEMANTIC_RUNTIME_KINDS:
        authority, identity = _runtime_semantic_document(
            value[kind],
            kind=kind,
            activation_id=activation_id,
            generation=generation,
        )
        authorities[kind] = authority
        identities[kind] = identity
    cutoff = authorities["RUNTIME_CUTOFF_AUTHORITY"]
    if set(cutoff) != {
        "runtime_credential_cutoff_at",
        "roles",
        "roles_identity_sha256",
    }:
        raise H1dLiveAuthorityError("runtime cutoff authority is opaque")
    cutoff_at = cutoff["runtime_credential_cutoff_at"]
    cutoff_time = _time(cutoff_at, field="runtime credential cutoff")
    roles = cutoff["roles"]
    role_fields = {
        "role_arn",
        "role_id",
        "trust_policy_sha256",
        "inline_policy_identities",
        "managed_policy_versions",
    }
    role_ids_by_arn: dict[str, str] = {}
    for role in roles if type(roles) is list else ():
        if (
            type(role) is not dict
            or set(role) != role_fields
            or type(role["role_arn"]) is not str
            or not role["role_arn"].startswith(
                "arn:aws:iam::" + ACCOUNT_ID + ":role/keep-glm52-h1g-"
            )
            or type(role["role_id"]) is not str
            or _ROLE_ID.fullmatch(role["role_id"]) is None
            or _SHA256.fullmatch(role["trust_policy_sha256"]) is None
            or type(role["inline_policy_identities"]) is not dict
            or not role["inline_policy_identities"]
            or any(
                type(name) is not str
                or not name
                or type(identity) is not str
                or _SHA256.fullmatch(identity) is None
                for name, identity in role[
                    "inline_policy_identities"
                ].items()
            )
            or type(role["managed_policy_versions"]) is not list
            or any(
                type(item) is not dict
                or set(item)
                != {
                    "policy_arn",
                    "default_version_id",
                    "document_sha256",
                }
                or type(item["policy_arn"]) is not str
                or not item["policy_arn"].startswith(
                    "arn:aws:iam::" + ACCOUNT_ID + ":policy/"
                )
                or type(item["default_version_id"]) is not str
                or not item["default_version_id"]
                or _SHA256.fullmatch(item["document_sha256"]) is None
                for item in role["managed_policy_versions"]
            )
            or role["role_arn"] in role_ids_by_arn
            or role["role_id"] in role_ids_by_arn.values()
        ):
            raise H1dLiveAuthorityError(
                "runtime cutoff role semantics drifted"
            )
        role_ids_by_arn[role["role_arn"]] = role["role_id"]
    activation_sha = hashlib.sha256(
        activation_id.encode("ascii")
    ).hexdigest()[:16]
    required_roles = {
        (
            "arn:aws:iam::"
            + ACCOUNT_ID
            + ":role/keep-glm52-h1g-support-decision"
        ),
        (
            "arn:aws:iam::"
            + ACCOUNT_ID
            + ":role/keep-glm52-h1g-closure-session-"
            + activation_sha
        ),
    }
    if (
        type(roles) is not list
        or set(role_ids_by_arn) != required_roles
        or cutoff["roles_identity_sha256"] != canonical_sha256(roles)
    ):
        raise H1dLiveAuthorityError(
            "runtime cutoff role inventory drifted"
        )

    runtime_attachments = authorities["RUNTIME_ATTACHMENT_INVENTORY"]
    if set(runtime_attachments) != {
        "task7_inventory_identity_sha256",
        "runtime_cutoff_identity_sha256",
        "attachments",
        "cardinality_by_kind",
        "attachments_identity_sha256",
    }:
        raise H1dLiveAuthorityError(
            "runtime attachment inventory is opaque"
        )
    if (
        runtime_attachments["task7_inventory_identity_sha256"]
        != task7_inventory_identity_sha256
        or runtime_attachments["runtime_cutoff_identity_sha256"]
        != identities["RUNTIME_CUTOFF_AUTHORITY"]
    ):
        raise H1dLiveAuthorityError(
            "runtime attachment source binding drifted"
        )
    attachments = runtime_attachments["attachments"]
    cardinality = runtime_attachments["cardinality_by_kind"]
    attachment_fields = {
        "attachment_kind",
        "resource_arn",
        "role_arn",
        "role_id",
        "association_id",
    }
    observed_cardinality: dict[str, int] = {}
    seen_resources: set[str] = set()
    if type(attachments) is not list or not attachments:
        raise H1dLiveAuthorityError(
            "runtime attachment inventory is empty"
        )
    for attachment in attachments:
        if (
            type(attachment) is not dict
            or set(attachment) != attachment_fields
            or attachment["attachment_kind"] not in _ATTACHMENT_KINDS
            or type(attachment["resource_arn"]) is not str
            or not attachment["resource_arn"]
            or attachment["resource_arn"] in seen_resources
            or attachment["role_arn"] not in role_ids_by_arn
            or attachment["role_id"]
            != role_ids_by_arn[attachment["role_arn"]]
            or (
                attachment["association_id"] is not None
                and (
                    type(attachment["association_id"]) is not str
                    or not attachment["association_id"]
                )
            )
        ):
            raise H1dLiveAuthorityError(
                "runtime attachment semantics drifted"
            )
        seen_resources.add(attachment["resource_arn"])
        kind = attachment["attachment_kind"]
        observed_cardinality[kind] = (
            observed_cardinality.get(kind, 0) + 1
        )
    if (
        type(cardinality) is not dict
        or cardinality != observed_cardinality
        or runtime_attachments["attachments_identity_sha256"]
        != canonical_sha256(attachments)
    ):
        raise H1dLiveAuthorityError(
            "runtime attachment cardinality drifted"
        )

    credentials = authorities["CREDENTIAL_PROBE_INVENTORY"]
    if set(credentials) != {
        "runtime_cutoff_identity_sha256",
        "runtime_attachment_inventory_identity_sha256",
        "runtime_credential_cutoff_at",
        "old_session_denials",
        "missing_issue_time_denials",
        "post_cutoff_positive_probes",
        "absence_observed",
    }:
        raise H1dLiveAuthorityError(
            "credential probe inventory is opaque"
        )
    if (
        credentials["runtime_cutoff_identity_sha256"]
        != identities["RUNTIME_CUTOFF_AUTHORITY"]
        or credentials["runtime_attachment_inventory_identity_sha256"]
        != identities["RUNTIME_ATTACHMENT_INVENTORY"]
        or credentials["runtime_credential_cutoff_at"] != cutoff_at
        or credentials["absence_observed"] is not False
    ):
        raise H1dLiveAuthorityError(
            "credential probe source binding drifted"
        )
    request_ids: set[str] = set()
    for probe_kind, fields, issued_required in (
        (
            "old_session_denials",
            {
                "role_id",
                "credential_issue_time",
                "observed_at",
                "error_code",
                "request_id",
            },
            True,
        ),
        (
            "missing_issue_time_denials",
            {"role_id", "observed_at", "error_code", "request_id"},
            False,
        ),
    ):
        probes = credentials[probe_kind]
        if type(probes) is not list or not probes:
            raise H1dLiveAuthorityError(
                "credential denial probe inventory is incomplete"
            )
        for probe in probes:
            observed = (
                _time(
                    probe.get("observed_at"),
                    field=probe_kind + " observed_at",
                )
                if type(probe) is dict
                else cutoff_time
            )
            issued = (
                _time(
                    probe.get("credential_issue_time"),
                    field=probe_kind + " credential_issue_time",
                )
                if issued_required and type(probe) is dict
                else None
            )
            if (
                type(probe) is not dict
                or set(probe) != fields
                or probe["role_id"] not in role_ids_by_arn.values()
                or probe["error_code"] != "AccessDenied"
                or observed < cutoff_time
                or (issued is not None and issued >= cutoff_time)
                or type(probe["request_id"]) is not str
                or not probe["request_id"]
                or probe["request_id"] in request_ids
            ):
                raise H1dLiveAuthorityError(
                    "credential denial probe drifted"
                )
            request_ids.add(probe["request_id"])
    positives = credentials["post_cutoff_positive_probes"]
    positive_role_ids: list[str] = []
    if type(positives) is not list or not positives:
        raise H1dLiveAuthorityError(
            "post-cutoff positive probes are absent"
        )
    for probe in positives:
        if type(probe) is not dict or set(probe) != {
            "role_id",
            "credential_issue_time",
            "credential_expiration",
            "observed_at",
            "request_id",
        }:
            raise H1dLiveAuthorityError(
                "post-cutoff positive probe is opaque"
            )
        issued = _time(
            probe["credential_issue_time"],
            field="positive credential issue time",
        )
        expiration = _time(
            probe["credential_expiration"],
            field="positive credential expiration",
        )
        observed = _time(
            probe["observed_at"],
            field="positive credential observation",
        )
        if (
            probe["role_id"] not in role_ids_by_arn.values()
            or issued < cutoff_time
            or observed < issued
            or observed >= expiration
            or type(probe["request_id"]) is not str
            or not probe["request_id"]
            or probe["request_id"] in request_ids
            or probe["role_id"] in positive_role_ids
        ):
            raise H1dLiveAuthorityError(
                "post-cutoff positive probe drifted"
            )
        request_ids.add(probe["request_id"])
        positive_role_ids.append(probe["role_id"])
    closure_role_arn = next(
        arn for arn in required_roles if "closure-session-" in arn
    )
    if role_ids_by_arn[closure_role_arn] not in positive_role_ids:
        raise H1dLiveAuthorityError(
            "current closure credential positive proof is absent"
        )
    result_body = {
        "runtime_credential_cutoff_at": cutoff_at,
        "role_ids_by_arn": role_ids_by_arn,
        "attachment_cardinality": len(attachments),
        "attachment_cardinality_by_kind": observed_cardinality,
        "current_positive_role_ids": tuple(sorted(positive_role_ids)),
        "absence_observed": False,
        "source_identities": identities,
    }
    return RuntimeRevalidationSemanticAuthority(
        **result_body,
        canonical_identity_sha256=canonical_sha256(result_body),
    )


def runtime_revalidation_sources_from_mapping(
    value: object,
    *,
    activation_id: str,
    generation: int,
) -> Tuple[RuntimeRevalidationSourceCoordinate, ...]:
    """Parse exact immutable Task 6/7 and runtime source coordinates."""

    if (
        type(value) is not dict
        or set(value) != set(RUNTIME_REVALIDATION_SOURCE_KINDS)
        or type(activation_id) is not str
        or not activation_id
        or type(generation) is not int
        or generation <= 0
    ):
        raise H1dLiveAuthorityError(
            "runtime revalidation source inventory drifted"
        )
    generation_text = f"{generation:08d}"
    result = []
    for index, kind in enumerate(RUNTIME_REVALIDATION_SOURCE_KINDS, 1):
        item = value[kind]
        if (
            type(item) is not dict
            or set(item)
            != set(RuntimeRevalidationSourceCoordinate.__dataclass_fields__)
        ):
            raise H1dLiveAuthorityError(
                "runtime revalidation source coordinate is not closed"
            )
        coordinate = RuntimeRevalidationSourceCoordinate(**item)
        body = asdict(coordinate)
        body.pop("canonical_identity_sha256")
        expected_key = (
            "campaigns/"
            + RUN_ID
            + "/authorities/task11/"
            + activation_id
            + "/"
            + generation_text
            + "/runtime-revalidation/"
            + f"{index:02d}-"
            + kind.lower().replace("_", "-")
            + ".json"
        )
        if (
            coordinate.input_kind != kind
            or _BUCKET.fullmatch(coordinate.bucket) is None
            or coordinate.key != expected_key
            or _VERSION_ID.fullmatch(coordinate.version_id) is None
            or coordinate.version_id == "null"
            or _SHA256.fullmatch(coordinate.file_sha256) is None
            or _SHA256.fullmatch(coordinate.body_sha256) is None
            or coordinate.canonical_identity_sha256
            != canonical_sha256(body)
        ):
            raise H1dLiveAuthorityError(
                "runtime revalidation source coordinate drifted"
            )
        result.append(coordinate)
    return tuple(result)


def _digest(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise H1dLiveAuthorityError(f"{field} must be a lowercase SHA-256")
    return value


def _time(value: object, *, field: str) -> datetime:
    if type(value) is not str:
        raise H1dLiveAuthorityError(f"{field} must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError) as exc:
        raise H1dLiveAuthorityError(f"{field} must be canonical UTC") from exc
    if parsed.tzinfo is None:
        raise H1dLiveAuthorityError(f"{field} must be timezone-aware")
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if canonical != value:
        raise H1dLiveAuthorityError(f"{field} must be canonical UTC")
    return parsed.astimezone(timezone.utc)


def _canonical_mapping(value: Mapping[str, object]) -> dict[str, object]:
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise H1dLiveAuthorityError("live authority value is not canonical") from exc
    return dict(value)


@dataclass(frozen=True)
class LiveReadSpec:
    family: str
    operation: str
    parameters: Mapping[str, object]
    identity_field: str
    expected_items: Tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class H1dExpectedState:
    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    activation_id: str
    manifest_identity_sha256: str
    support_host_instance_id: str
    must_start_by: str
    execution_deadline: str
    specs: Tuple[LiveReadSpec, ...]
    canonical_identity_sha256: str


def _spec_mapping(spec: LiveReadSpec) -> dict[str, object]:
    return {
        "family": spec.family,
        "operation": spec.operation,
        "parameters": dict(spec.parameters),
        "identity_field": spec.identity_field,
        "expected_items": tuple(dict(item) for item in spec.expected_items),
    }


def _expected_state_body(value: H1dExpectedState) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "record_type": value.record_type,
        "account_id": value.account_id,
        "region": value.region,
        "run_id": value.run_id,
        "activation_id": value.activation_id,
        "manifest_identity_sha256": value.manifest_identity_sha256,
        "support_host_instance_id": value.support_host_instance_id,
        "must_start_by": value.must_start_by,
        "execution_deadline": value.execution_deadline,
        "specs": tuple(_spec_mapping(spec) for spec in value.specs),
    }


def build_h1d_expected_state(
    *,
    account_id: str,
    region: str,
    run_id: str,
    activation_id: str,
    manifest_identity_sha256: str,
    support_host_instance_id: str,
    must_start_by: str,
    execution_deadline: str,
    specs: Tuple[LiveReadSpec, ...],
) -> H1dExpectedState:
    """Build the canonical exact-state contract consumed by one live walk."""

    if (
        account_id != ACCOUNT_ID
        or region != REGION
        or run_id != RUN_ID
        or type(activation_id) is not str
        or not activation_id
        or type(support_host_instance_id) is not str
        or _INSTANCE_ID.fullmatch(support_host_instance_id) is None
    ):
        raise H1dLiveAuthorityError("expected-state identity is not exact")
    _digest(manifest_identity_sha256, field="manifest identity")
    must_start = _time(must_start_by, field="must_start_by")
    deadline = _time(execution_deadline, field="execution_deadline")
    if deadline <= must_start:
        raise H1dLiveAuthorityError("execution deadline must follow must-start")
    if type(specs) is not tuple or tuple(spec.family for spec in specs) != (
        REQUIRED_LIVE_FAMILIES
    ):
        raise H1dLiveAuthorityError(
            "expected state must contain every live family exactly once"
        )
    for spec in specs:
        if (
            type(spec) is not LiveReadSpec
            or spec.operation != f"{spec.family}.inspect_complete"
            or type(spec.parameters) is not dict
            or type(spec.identity_field) is not str
            or not spec.identity_field
            or type(spec.expected_items) is not tuple
        ):
            raise H1dLiveAuthorityError("live read specification is not closed")
        seen: set[str] = set()
        for item in spec.expected_items:
            if type(item) is not dict:
                raise H1dLiveAuthorityError("expected live item is not an object")
            _canonical_mapping(item)
            identity = item.get(spec.identity_field)
            if type(identity) is not str or not identity or identity in seen:
                raise H1dLiveAuthorityError(
                    "expected live identity is absent or duplicated"
                )
            seen.add(identity)
    provisional = H1dExpectedState(
        schema_version=1,
        record_type="glm52_h1d_expected_live_state_v1",
        account_id=account_id,
        region=region,
        run_id=run_id,
        activation_id=activation_id,
        manifest_identity_sha256=manifest_identity_sha256,
        support_host_instance_id=support_host_instance_id,
        must_start_by=must_start_by,
        execution_deadline=execution_deadline,
        specs=specs,
        canonical_identity_sha256="",
    )
    return H1dExpectedState(
        **{
            **asdict(provisional),
            "specs": specs,
            "canonical_identity_sha256": canonical_sha256(
                _expected_state_body(provisional)
            ),
        }
    )


def validate_h1d_expected_state(value: object) -> H1dExpectedState:
    if not isinstance(value, H1dExpectedState):
        raise H1dLiveAuthorityError("expected live state is not typed")
    rebuilt = build_h1d_expected_state(
        account_id=value.account_id,
        region=value.region,
        run_id=value.run_id,
        activation_id=value.activation_id,
        manifest_identity_sha256=value.manifest_identity_sha256,
        support_host_instance_id=value.support_host_instance_id,
        must_start_by=value.must_start_by,
        execution_deadline=value.execution_deadline,
        specs=value.specs,
    )
    if value != rebuilt:
        raise H1dLiveAuthorityError("expected live state identity drifted")
    return value


def h1d_expected_state_to_mapping(
    value: H1dExpectedState,
) -> dict[str, object]:
    """Serialize the closed expected-state contract, including its identity."""

    validate_h1d_expected_state(value)
    return {
        **_expected_state_body(value),
        "canonical_identity_sha256": value.canonical_identity_sha256,
    }


def h1d_expected_state_from_mapping(
    value: Mapping[str, object],
) -> H1dExpectedState:
    """Parse one canonical authenticated expected-state JSON object."""

    fields = {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "manifest_identity_sha256",
        "support_host_instance_id",
        "must_start_by",
        "execution_deadline",
        "specs",
        "canonical_identity_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise H1dLiveAuthorityError("expected-state schema mismatch")
    specs_raw = value["specs"]
    if type(specs_raw) is not list:
        raise H1dLiveAuthorityError("expected-state specs must be an array")
    specs: list[LiveReadSpec] = []
    spec_fields = {
        "family",
        "operation",
        "parameters",
        "identity_field",
        "expected_items",
    }
    for raw in specs_raw:
        if type(raw) is not dict or set(raw) != spec_fields:
            raise H1dLiveAuthorityError("live-read spec schema mismatch")
        expected_items = raw["expected_items"]
        if type(expected_items) is not list or any(
            type(item) is not dict for item in expected_items
        ):
            raise H1dLiveAuthorityError("live-read expected items are invalid")
        specs.append(
            LiveReadSpec(
                family=str(raw["family"]),
                operation=str(raw["operation"]),
                parameters=dict(raw["parameters"]),
                identity_field=str(raw["identity_field"]),
                expected_items=tuple(dict(item) for item in expected_items),
            )
        )
    rebuilt = build_h1d_expected_state(
        account_id=str(value["account_id"]),
        region=str(value["region"]),
        run_id=str(value["run_id"]),
        activation_id=str(value["activation_id"]),
        manifest_identity_sha256=str(value["manifest_identity_sha256"]),
        support_host_instance_id=str(value["support_host_instance_id"]),
        must_start_by=str(value["must_start_by"]),
        execution_deadline=str(value["execution_deadline"]),
        specs=tuple(specs),
    )
    if (
        value["schema_version"] != 1
        or value["record_type"] != "glm52_h1d_expected_live_state_v1"
        or value["canonical_identity_sha256"]
        != rebuilt.canonical_identity_sha256
    ):
        raise H1dLiveAuthorityError("expected-state identity mismatch")
    return rebuilt


@dataclass(frozen=True)
class H1dLiveAuthorityRequest:
    profile: str
    account_id: str
    region: str
    run_id: str
    activation_id: str
    expected_state_identity_sha256: str
    spend_request: object
    sky_probe_request: object


@dataclass(frozen=True)
class CallerIdentityObservation:
    account_id: str
    arn: str
    user_id: str
    credential_expiration: str
    request_id: str
    observed_at: str


@dataclass(frozen=True)
class LiveReadPage:
    family: str
    operation: str
    request_token: Optional[str]
    page_index: int
    items: Tuple[Mapping[str, object], ...]
    next_token: Optional[str]
    request_id: str
    observed_at: str
    service_request_ids: Tuple[str, ...] = ()
    service_response_identities: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SkyRelayProbeResult:
    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    request_identity_sha256: str
    direct_response_request_id: str
    tls_peer_certificate_sha256: str
    attestation_identity_sha256: str
    admission_identity_sha256: str
    sky_user_identity: str
    sky_roles: Tuple[str, ...]
    token_expires_at: str
    effective_controller_identity_sha256: str
    observed_at: str
    canonical_identity_sha256: str


@dataclass(frozen=True)
class ExpectedStateAuthentication:
    schema_version: int
    record_type: str
    expected_state_identity_sha256: str
    task6_manifest_identity_sha256: str
    task6_templates_identity_sha256: str
    task7_postcreate_manifest_identity_sha256: str
    task7_inventory_identity_sha256: str
    direct_read_request_ids: Tuple[str, ...]
    observed_at: str
    canonical_identity_sha256: str


def _expected_auth_body(value: ExpectedStateAuthentication) -> dict[str, object]:
    body = asdict(value)
    body.pop("canonical_identity_sha256")
    return body


def build_expected_state_authentication(
    *,
    expected_state_identity_sha256: str,
    task6_manifest_identity_sha256: str,
    task6_templates_identity_sha256: str,
    task7_postcreate_manifest_identity_sha256: str,
    task7_inventory_identity_sha256: str,
    direct_read_request_ids: Tuple[str, ...],
    observed_at: str,
) -> ExpectedStateAuthentication:
    """Bind expected state to exact-read Task 6/7 manifests and templates."""

    for field, value in (
        ("expected state", expected_state_identity_sha256),
        ("Task 6 manifest", task6_manifest_identity_sha256),
        ("Task 6 templates", task6_templates_identity_sha256),
        ("Task 7 postcreate manifest", task7_postcreate_manifest_identity_sha256),
        ("Task 7 inventory", task7_inventory_identity_sha256),
    ):
        _digest(value, field=field)
    if (
        type(direct_read_request_ids) is not tuple
        or len(direct_read_request_ids) < 4
        or len(set(direct_read_request_ids)) != len(direct_read_request_ids)
        or any(type(item) is not str or not item for item in direct_read_request_ids)
    ):
        raise H1dLiveAuthorityError(
            "expected-state source reads are incomplete or duplicated"
        )
    _time(observed_at, field="expected-state authentication observed_at")
    provisional = ExpectedStateAuthentication(
        schema_version=1,
        record_type="glm52_h1d_expected_state_authentication_v1",
        expected_state_identity_sha256=expected_state_identity_sha256,
        task6_manifest_identity_sha256=task6_manifest_identity_sha256,
        task6_templates_identity_sha256=task6_templates_identity_sha256,
        task7_postcreate_manifest_identity_sha256=(
            task7_postcreate_manifest_identity_sha256
        ),
        task7_inventory_identity_sha256=task7_inventory_identity_sha256,
        direct_read_request_ids=direct_read_request_ids,
        observed_at=observed_at,
        canonical_identity_sha256="",
    )
    return ExpectedStateAuthentication(
        **{
            **asdict(provisional),
            "direct_read_request_ids": direct_read_request_ids,
            "canonical_identity_sha256": canonical_sha256(
                _expected_auth_body(provisional)
            ),
        }
    )


def _validate_expected_state_authentication(
    value: object, expected: H1dExpectedState
) -> ExpectedStateAuthentication:
    if not isinstance(value, ExpectedStateAuthentication):
        raise H1dLiveAuthorityError(
            "expected state lacks authenticated Task 6/7 source reads"
        )
    rebuilt = build_expected_state_authentication(
        expected_state_identity_sha256=value.expected_state_identity_sha256,
        task6_manifest_identity_sha256=value.task6_manifest_identity_sha256,
        task6_templates_identity_sha256=value.task6_templates_identity_sha256,
        task7_postcreate_manifest_identity_sha256=(
            value.task7_postcreate_manifest_identity_sha256
        ),
        task7_inventory_identity_sha256=value.task7_inventory_identity_sha256,
        direct_read_request_ids=value.direct_read_request_ids,
        observed_at=value.observed_at,
    )
    if (
        value != rebuilt
        or value.expected_state_identity_sha256
        != expected.canonical_identity_sha256
        or value.task7_postcreate_manifest_identity_sha256
        != expected.manifest_identity_sha256
    ):
        raise H1dLiveAuthorityError(
            "expected resources are not bound to authenticated manifests"
        )
    return value


def _probe_body(value: SkyRelayProbeResult) -> dict[str, object]:
    body = asdict(value)
    body.pop("canonical_identity_sha256")
    return body


def build_sky_relay_probe_result(
    *,
    account_id: str,
    region: str,
    run_id: str,
    request_identity_sha256: str,
    direct_response_request_id: str,
    tls_peer_certificate_sha256: str,
    attestation_identity_sha256: str,
    admission_identity_sha256: str,
    sky_user_identity: str,
    sky_roles: Tuple[str, ...],
    token_expires_at: str,
    effective_controller_identity_sha256: str,
    observed_at: str,
) -> SkyRelayProbeResult:
    """Materialize Task 9's exact direct mTLS/identity probe result."""

    if (
        account_id != ACCOUNT_ID
        or region != REGION
        or run_id != RUN_ID
        or type(direct_response_request_id) is not str
        or not direct_response_request_id
        or type(sky_user_identity) is not str
        or not sky_user_identity
        or type(sky_roles) is not tuple
        or not sky_roles
        or len(set(sky_roles)) != len(sky_roles)
        or any(type(role) is not str or not role for role in sky_roles)
    ):
        raise H1dLiveAuthorityError("Sky relay probe identity is invalid")
    for field, value in (
        ("request identity", request_identity_sha256),
        ("TLS peer certificate", tls_peer_certificate_sha256),
        ("attestation identity", attestation_identity_sha256),
        ("admission identity", admission_identity_sha256),
        ("effective controller identity", effective_controller_identity_sha256),
    ):
        _digest(value, field=field)
    _time(token_expires_at, field="token_expires_at")
    _time(observed_at, field="probe observed_at")
    provisional = SkyRelayProbeResult(
        schema_version=1,
        record_type="glm52_h1d_sky_relay_probe_v1",
        account_id=account_id,
        region=region,
        run_id=run_id,
        request_identity_sha256=request_identity_sha256,
        direct_response_request_id=direct_response_request_id,
        tls_peer_certificate_sha256=tls_peer_certificate_sha256,
        attestation_identity_sha256=attestation_identity_sha256,
        admission_identity_sha256=admission_identity_sha256,
        sky_user_identity=sky_user_identity,
        sky_roles=sky_roles,
        token_expires_at=token_expires_at,
        effective_controller_identity_sha256=(
            effective_controller_identity_sha256
        ),
        observed_at=observed_at,
        canonical_identity_sha256="",
    )
    return SkyRelayProbeResult(
        **{
            **asdict(provisional),
            "sky_roles": sky_roles,
            "canonical_identity_sha256": canonical_sha256(
                _probe_body(provisional)
            ),
        }
    )


def _validate_probe(
    value: object,
    *,
    request: object,
    phase_started_at: datetime,
) -> SkyRelayProbeResult:
    if not isinstance(value, SkyRelayProbeResult):
        raise H1dLiveAuthorityError(
            "caller-asserted Sky success is not a probe result"
        )
    rebuilt = build_sky_relay_probe_result(
        account_id=value.account_id,
        region=value.region,
        run_id=value.run_id,
        request_identity_sha256=value.request_identity_sha256,
        direct_response_request_id=value.direct_response_request_id,
        tls_peer_certificate_sha256=value.tls_peer_certificate_sha256,
        attestation_identity_sha256=value.attestation_identity_sha256,
        admission_identity_sha256=value.admission_identity_sha256,
        sky_user_identity=value.sky_user_identity,
        sky_roles=value.sky_roles,
        token_expires_at=value.token_expires_at,
        effective_controller_identity_sha256=(
            value.effective_controller_identity_sha256
        ),
        observed_at=value.observed_at,
    )
    if value != rebuilt:
        raise H1dLiveAuthorityError("Sky relay probe identity drifted")
    probe_observed_at = _time(value.observed_at, field="probe observed_at")
    if (
        value.request_identity_sha256 != canonical_sha256(request)
        or probe_observed_at < phase_started_at
        or _time(value.token_expires_at, field="token_expires_at")
        <= phase_started_at
    ):
        raise H1dLiveAuthorityError(
            "Sky relay probe is stale or request-substituted"
        )
    return value


@dataclass(frozen=True)
class H1dLiveServices:
    identity: object
    reader: object
    spend: object
    sky_relay_probe: object
    clock: object
    expected_state_authority: object = None


@dataclass(frozen=True)
class H1dLiveAuthorityResult:
    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    activation_id: str
    caller_identity_sha256: str
    expected_state_identity_sha256: str
    expected_state_authentication_identity_sha256: str
    page_identities: Tuple[str, ...]
    family_identities: Mapping[str, str]
    spend_authority_identity_sha256: str
    sky_probe_identity_sha256: str
    phase_started_at: str
    phase_ended_at: str
    phase_elapsed_seconds: Decimal
    canonical_identity_sha256: str


def _result_body(result: H1dLiveAuthorityResult) -> dict[str, object]:
    _digest(
        result.expected_state_authentication_identity_sha256,
        field="expected-state authentication identity",
    )
    body = asdict(result)
    body.pop("canonical_identity_sha256")
    body["phase_elapsed_seconds"] = format(
        result.phase_elapsed_seconds, "f"
    )
    return body


def _validate_request(
    request: H1dLiveAuthorityRequest, expected: H1dExpectedState
) -> None:
    if not isinstance(request, H1dLiveAuthorityRequest):
        raise H1dLiveAuthorityError("live authority request is not typed")
    if (
        request.profile != PROFILE
        or request.account_id != ACCOUNT_ID
        or request.region != REGION
        or request.run_id != RUN_ID
        or request.activation_id != expected.activation_id
        or request.expected_state_identity_sha256
        != expected.canonical_identity_sha256
    ):
        raise H1dLiveAuthorityError(
            "profile/account/region/run/activation/manifest guard failed"
        )
    if (
        type(request.sky_probe_request) is not dict
        or set(request.sky_probe_request)
        != {"account_id", "region", "run_id", "activation_id"}
        or request.sky_probe_request["account_id"] != ACCOUNT_ID
        or request.sky_probe_request["region"] != REGION
        or request.sky_probe_request["run_id"] != RUN_ID
        or request.sky_probe_request["activation_id"] != expected.activation_id
    ):
        raise H1dLiveAuthorityError("Sky probe request identity is not exact")


def _caller_identity(
    services: H1dLiveServices, *, phase_started_at: datetime
) -> tuple[CallerIdentityObservation, str]:
    method = getattr(services.identity, "get_caller_identity", None)
    if not callable(method):
        raise H1dLiveAuthorityError("STS identity boundary is absent")
    value = method()
    if not isinstance(value, CallerIdentityObservation):
        raise H1dLiveAuthorityError("STS identity observation is not typed")
    if (
        value.account_id != ACCOUNT_ID
        or type(value.arn) is not str
        or f"::{ACCOUNT_ID}:" not in value.arn
        or type(value.user_id) is not str
        or not value.user_id
        or type(value.request_id) is not str
        or not value.request_id
        or _time(
            value.credential_expiration, field="credential_expiration"
        )
        <= phase_started_at
    ):
        raise H1dLiveAuthorityError("STS caller/session identity is not exact")
    _time(value.observed_at, field="STS observed_at")
    return value, canonical_sha256(asdict(value))


def _page_identity(spec: LiveReadSpec, page: LiveReadPage) -> str:
    request_digest = canonical_sha256(
        {
            "family": spec.family,
            "operation": spec.operation,
            "parameters": dict(spec.parameters),
            "continuation_token": page.request_token,
        }
    )
    return canonical_sha256(
        {
            "request_digest_sha256": request_digest,
            "family": page.family,
            "operation": page.operation,
            "request_token": page.request_token,
            "page_index": page.page_index,
            "items": tuple(dict(item) for item in page.items),
            "next_token": page.next_token,
            "request_id": page.request_id,
            "service_request_ids": page.service_request_ids,
            "service_response_identities": (
                page.service_response_identities
            ),
            "observed_at": page.observed_at,
        }
    )


def _read_family(
    spec: LiveReadSpec,
    services: H1dLiveServices,
) -> tuple[Tuple[str, ...], str, Tuple[Mapping[str, object], ...]]:
    method = getattr(services.reader, "read_page", None)
    if not callable(method):
        raise H1dLiveAuthorityError("live read boundary is absent")
    token: Optional[str] = None
    seen_tokens: set[str] = set()
    identities: list[str] = []
    items: list[Mapping[str, object]] = []
    item_ids: set[str] = set()
    page_index = 0
    while True:
        page = method(spec=spec, continuation_token=token)
        if not isinstance(page, LiveReadPage):
            raise H1dLiveAuthorityError("live read page is not typed")
        if (
            page.family != spec.family
            or page.operation != spec.operation
            or page.request_token != token
            or page.page_index != page_index
            or type(page.items) is not tuple
            or type(page.request_id) is not str
            or not page.request_id
            or type(page.service_request_ids) is not tuple
            or any(
                type(request_id) is not str or not request_id
                for request_id in page.service_request_ids
            )
            or len(page.service_request_ids)
            != len(set(page.service_request_ids))
            or type(page.service_response_identities) is not tuple
            or any(
                _SHA256.fullmatch(identity) is None
                for identity in page.service_response_identities
            )
            or len(page.service_response_identities)
            != len(set(page.service_response_identities))
            or (
                page.service_response_identities
                and len(page.service_response_identities)
                != len(page.service_request_ids)
            )
        ):
            raise H1dLiveAuthorityError("live read page identity is incomplete")
        _time(page.observed_at, field=f"{spec.family} observed_at")
        for item in page.items:
            if type(item) is not dict:
                raise H1dLiveAuthorityError("live resource is not an object")
            _canonical_mapping(item)
            item_identity = item.get(spec.identity_field)
            if (
                type(item_identity) is not str
                or not item_identity
                or item_identity in item_ids
            ):
                raise H1dLiveAuthorityError(
                    "live resource identity is missing or duplicated"
                )
            item_ids.add(item_identity)
            items.append(item)
        identity = _page_identity(spec, page)
        if identity in identities:
            raise H1dLiveAuthorityError("live page identity repeated")
        identities.append(identity)
        if page.next_token is None:
            break
        if (
            type(page.next_token) is not str
            or not page.next_token
            or page.next_token in seen_tokens
        ):
            raise H1dLiveAuthorityError("live pagination token repeated or cycled")
        seen_tokens.add(page.next_token)
        token = page.next_token
        page_index += 1
    sorted_actual = tuple(
        sorted(items, key=lambda item: str(item[spec.identity_field]))
    )
    sorted_expected = tuple(
        sorted(
            spec.expected_items,
            key=lambda item: str(item[spec.identity_field]),
        )
    )
    if sorted_actual != sorted_expected:
        raise H1dLiveAuthorityError(f"{spec.family} exact live state drifted")
    family_identity = canonical_sha256(
        {
            "family": spec.family,
            "operation": spec.operation,
            "page_identities": tuple(identities),
            "items": sorted_actual,
        }
    )
    return tuple(identities), family_identity, sorted_actual


def _fail_closed_observations(
    family_items: Mapping[str, Tuple[Mapping[str, object], ...]],
    expected: H1dExpectedState,
) -> None:
    for alarm in family_items["cloudwatch"]:
        if (
            alarm.get("state") != "OK"
            or alarm.get("metric_status") != "COMPLETE"
        ):
            raise H1dLiveAuthorityError("alarm or required metric is not healthy")
    for queue in family_items["sqs"]:
        if (
            queue.get("approximate_messages") != 0
            or not queue.get("redrive_policy_sha256")
        ):
            raise H1dLiveAuthorityError("DLQ is nonzero or redrive is absent")
    for subscription in family_items["sns"]:
        if (
            subscription.get("confirmed") is not True
            or subscription.get("confirmation_authenticated") is not True
        ):
            raise H1dLiveAuthorityError("SNS subscription is not authenticated")
    ssm = family_items["ssm"]
    if (
        len(ssm) != 1
        or ssm[0].get("instance_id") != expected.support_host_instance_id
        or ssm[0].get("ping_status") != "Online"
    ):
        raise H1dLiveAuthorityError("SSM host identity is wrong or offline")
    for instance in family_items["ec2"]:
        if (
            instance.get("instance_type") == "p5.48xlarge"
            or instance.get("role") != "support-host"
            or instance.get("instance_id") != expected.support_host_instance_id
        ):
            raise H1dLiveAuthorityError(
                "unexpected P5, exact-name job, or unauthorized instance"
            )
    retained_buckets = 0
    rehearsal_buckets = 0
    for bucket in family_items["s3"]:
        bucket_class = bucket.get("bucket_class")
        if bucket_class == "retained_model_evidence":
            retained_buckets += 1
            invalid = (
                bucket.get("versioning") != "Enabled"
                or bucket.get("lifecycle") is not None
                or bucket.get("replication") is not None
            )
        elif bucket_class == "support_rehearsal":
            rehearsal_buckets += 1
            invalid = (
                bucket.get("versioning") is not None
                or bucket.get("lifecycle") is not None
                or bucket.get("replication") is not None
            )
        else:
            invalid = True
        if invalid:
            raise H1dLiveAuthorityError("protected bucket configuration drifted")
    if retained_buckets != 1 or rehearsal_buckets != 1:
        raise H1dLiveAuthorityError("protected bucket inventory is incomplete")


def inspect_h1d_live_authority(
    request: H1dLiveAuthorityRequest,
    expected: H1dExpectedState,
    services: H1dLiveServices,
) -> H1dLiveAuthorityResult:
    """Perform one in-memory complete H.1d inspection in at most seven seconds."""

    validate_h1d_expected_state(expected)
    _validate_request(request, expected)
    clock = services.clock
    if not callable(clock):
        raise H1dLiveAuthorityError("live authority clock is absent")
    started = clock()
    if not isinstance(started, datetime) or started.tzinfo is None:
        raise H1dLiveAuthorityError("live authority clock must be timezone-aware")
    started = started.astimezone(timezone.utc)
    if (
        started > _time(expected.must_start_by, field="must_start_by")
        or started >= _time(expected.execution_deadline, field="execution_deadline")
    ):
        raise H1dLiveAuthorityError("must-start or execution deadline is stale")
    authenticate_expected = getattr(
        services.expected_state_authority, "authenticate", None
    )
    if not callable(authenticate_expected):
        raise H1dLiveAuthorityError(
            "authenticated Task 6/7 expected-state boundary is absent"
        )
    expected_state_authentication = _validate_expected_state_authentication(
        authenticate_expected(expected),
        expected,
    )
    _caller, caller_identity = _caller_identity(
        services, phase_started_at=started
    )
    prepare_reader = getattr(services.reader, "prepare", None)
    if callable(prepare_reader):
        try:
            prepare_reader(
                expected.specs,
                deadline=started + timedelta(seconds=7),
            )
        except (TypeError, ValueError) as exc:
            raise H1dLiveAuthorityError(
                "concurrent live read preparation failed"
            ) from exc
    all_page_identities: list[str] = []
    family_identities: dict[str, str] = {}
    family_items: dict[str, Tuple[Mapping[str, object], ...]] = {}
    for spec in expected.specs:
        page_ids, family_identity, items = _read_family(spec, services)
        all_page_identities.extend(page_ids)
        family_identities[spec.family] = family_identity
        family_items[spec.family] = items
    _fail_closed_observations(family_items, expected)
    spend_method = getattr(services.spend, "inspect", None)
    if not callable(spend_method):
        raise H1dLiveAuthorityError("live spend boundary is absent")
    try:
        spend = validate_spend_authority_result(
            spend_method(request.spend_request)
        )
    except (TypeError, ValueError) as exc:
        raise H1dLiveAuthorityError("live spend authority failed") from exc
    spend_observed_at = _time(
        spend.observed_at,
        field="spend observed_at",
    )
    if (
        spend.open_gpu_seconds != 0
        or spend.open_gpu_cost_usd != Decimal("0.00")
    ):
        raise H1dLiveAuthorityError("foreign/open GPU allocation is present")
    if spend_observed_at < started:
        raise H1dLiveAuthorityError(
            "spend observation is outside the current H.1d phase"
        )
    probe_method = getattr(services.sky_relay_probe, "inspect", None)
    if not callable(probe_method):
        raise H1dLiveAuthorityError("Task 9 Sky probe boundary is absent")
    probe = _validate_probe(
        probe_method(request.sky_probe_request),
        request=request.sky_probe_request,
        phase_started_at=started,
    )
    ended = clock()
    if not isinstance(ended, datetime) or ended.tzinfo is None:
        raise H1dLiveAuthorityError("live authority clock must be timezone-aware")
    ended = ended.astimezone(timezone.utc)
    elapsed = Decimal(str((ended - started).total_seconds())).quantize(
        Decimal("0.000001")
    )
    probe_observed_at = _time(probe.observed_at, field="probe observed_at")
    if elapsed < 0 or elapsed > Decimal("7.000000"):
        raise H1dLiveAuthorityError("H.1d live reinspection exceeded seven seconds")
    if ended >= _time(expected.execution_deadline, field="execution_deadline"):
        raise H1dLiveAuthorityError(
            "execution deadline elapsed during H.1d reinspection"
        )
    if probe_observed_at > ended:
        raise H1dLiveAuthorityError(
            "Sky relay probe observation is outside the measured phase"
        )
    if spend_observed_at > ended:
        raise H1dLiveAuthorityError(
            "spend observation is outside the current H.1d phase"
        )
    if _time(probe.token_expires_at, field="token_expires_at") <= ended:
        raise H1dLiveAuthorityError(
            "Sky token expires before H.1d phase completion"
        )
    provisional = H1dLiveAuthorityResult(
        schema_version=1,
        record_type="glm52_h1d_live_authority_result_v1",
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        activation_id=expected.activation_id,
        caller_identity_sha256=caller_identity,
        expected_state_identity_sha256=expected.canonical_identity_sha256,
        expected_state_authentication_identity_sha256=(
            expected_state_authentication.canonical_identity_sha256
        ),
        page_identities=tuple(all_page_identities),
        family_identities=dict(sorted(family_identities.items())),
        spend_authority_identity_sha256=spend.canonical_identity_sha256,
        sky_probe_identity_sha256=probe.canonical_identity_sha256,
        phase_started_at=started.isoformat().replace("+00:00", "Z"),
        phase_ended_at=ended.isoformat().replace("+00:00", "Z"),
        phase_elapsed_seconds=elapsed,
        canonical_identity_sha256="",
    )
    return H1dLiveAuthorityResult(
        **{
            **asdict(provisional),
            "page_identities": provisional.page_identities,
            "family_identities": provisional.family_identities,
            "phase_elapsed_seconds": elapsed,
            "canonical_identity_sha256": canonical_sha256(
                _result_body(provisional)
            ),
        }
    )


def serialize_non_authoritative_evidence(
    result: H1dLiveAuthorityResult,
) -> dict[str, object]:
    """Serialize evidence that is explicitly impossible to reuse as authority."""

    if not isinstance(result, H1dLiveAuthorityResult):
        raise H1dLiveAuthorityError("H.1d result is not typed")
    body = _result_body(result)
    if result.canonical_identity_sha256 != canonical_sha256(body):
        raise H1dLiveAuthorityError("H.1d result identity mismatch")
    evidence_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1d_live_evidence_v1",
        "authority_classification": "NON_AUTHORITATIVE_EVIDENCE",
        "live_result_identity_sha256": result.canonical_identity_sha256,
        "phase_started_at": result.phase_started_at,
        "phase_ended_at": result.phase_ended_at,
        "page_identities": result.page_identities,
    }
    return {
        **evidence_body,
        "canonical_body_sha256": canonical_sha256(evidence_body),
    }


def reject_serialized_authority_input(value: object) -> None:
    """Fail closed for every stored/evidence-shaped attempted authority input."""

    if type(value) is dict and value.get("authority_classification") == (
        "NON_AUTHORITATIVE_EVIDENCE"
    ):
        raise H1dLiveAuthorityError("serialized H.1d evidence is not authority")
    raise H1dLiveAuthorityError("stored H.1d results are never accepted")


__all__ = [
    "ACCOUNT_ID",
    "PROFILE",
    "REGION",
    "REQUIRED_LIVE_FAMILIES",
    "RUNTIME_REVALIDATION_SOURCE_KINDS",
    "RUN_ID",
    "CallerIdentityObservation",
    "CurrentClosureSessionBinding",
    "ExpectedStateAuthentication",
    "H1dExpectedState",
    "H1dLiveAuthorityError",
    "H1dLiveAuthorityRequest",
    "H1dLiveAuthorityResult",
    "H1dLiveServices",
    "LiveReadPage",
    "LiveReadSpec",
    "RuntimeRevalidationSourceCoordinate",
    "RuntimeRevalidationSemanticAuthority",
    "SkyRelayProbeResult",
    "build_h1d_expected_state",
    "build_expected_state_authentication",
    "build_sky_relay_probe_result",
    "bind_current_closure_session_to_cutoff",
    "h1d_expected_state_from_mapping",
    "h1d_expected_state_to_mapping",
    "inspect_h1d_live_authority",
    "reject_serialized_authority_input",
    "runtime_revalidation_sources_from_mapping",
    "serialize_non_authoritative_evidence",
    "validate_runtime_revalidation_source_documents",
    "validate_h1d_expected_state",
]
