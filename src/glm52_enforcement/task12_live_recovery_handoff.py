"""Retained recovery handoff custody for the Task 12 production route.

Normal Task 11 and retained recovery share one immutable coordinate and the
same frozen 21-field body.  A compact legacy Task 11 object is foreign bytes,
not an adoption format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Mapping

from .canonical import canonical_json_bytes, canonical_sha256
from .dynamodb import ExactCheck, ExactPut, ExactUpdate, LedgerKey
from .records import canonical_record_identity, ledger_sk, validate_record
from .task12_correlation import (
    build_sky_post_handoff_record,
    validate_sky_post_handoff_record,
)


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
SUPPORTED_OPERATIONS = frozenset(
    {"RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED"}
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL_EXECUTION_STATES = frozenset(
    {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}
)
_RECOVERY_OUTCOMES = frozenset(
    {"PROVED_NOT_SENT_OWNER_DIED", "AMBIGUOUS_OWNER_DIED"}
)
_SOURCE_TYPES = {
    "activation_index": "glm52_production_activation_index",
    "control": "glm52_production_control",
    "execution": "glm52_production_execution",
    "recovery_control": "glm52_production_recovery_control",
    "sky_action": "glm52_production_action",
}
_HANDOFF_KEY = (
    "campaigns/glm52-sky-20260724/submissions/production/generations/"
    "{generation_text}/handoff/SKY_POST_HANDOFF.json"
)
_TASK10_AUTHORITY_KEY = "task13/production/task10-production-authority.json"
class Task12RecoveryHandoffError(ValueError):
    """Recovery handoff evidence is absent, mutable, or forked."""


@dataclass(frozen=True)
class RecoveryHandoffNotRequired:
    state: str
    action_identity_sha256: str
    canonical_identity_sha256: str


@dataclass(frozen=True)
class RecoveryHandoffAdoption:
    state: str
    version_id: str
    file_sha256: str
    body_sha256: str
    canonical_identity_sha256: str


@dataclass(frozen=True)
class RecoveryHandoffArmPlan:
    index: ExactCheck
    control: ExactCheck
    recovery_control: ExactUpdate
    action: ExactPut

    @property
    def transaction_plans(self) -> tuple[object, ...]:
        return (
            self.index,
            self.control,
            self.recovery_control,
            self.action,
        )


@dataclass(frozen=True)
class RecoveryHandoffConsumePlan:
    index: ExactCheck
    control: ExactCheck
    recovery_control: ExactUpdate
    action: ExactUpdate

    @property
    def transaction_plans(self) -> tuple[object, ...]:
        return (
            self.index,
            self.control,
            self.recovery_control,
            self.action,
        )


@dataclass(frozen=True)
class RecoveryHandoffCompletePlan:
    index: ExactCheck
    control: ExactCheck
    recovery_control: ExactUpdate
    action: ExactUpdate
    writer_control: ExactPut

    @property
    def transaction_plans(self) -> tuple[object, ...]:
        return (
            self.index,
            self.control,
            self.recovery_control,
            self.action,
            self.writer_control,
        )


def _fail(message: str) -> None:
    raise Task12RecoveryHandoffError(message)


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        _fail(label + " is not one exact SHA-256")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(label + " is not one exact string")
    return value


def _utc(value: object, label: str) -> str:
    exact = _text(value, label)
    try:
        parsed = datetime.strptime(exact, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise Task12RecoveryHandoffError(
            label + " is not canonical UTC"
        ) from exc
    if parsed.tzinfo is not None or parsed.microsecond:
        _fail(label + " is not canonical UTC")
    return exact


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _roles(ports: object) -> Mapping[str, object]:
    deployment = getattr(ports, "deployment", None)
    roles = getattr(deployment, "role_coordinates", None)
    if not isinstance(roles, Mapping):
        _fail("recovery handoff deployment coordinates are absent")
    return roles


def _client(ports: object, service: str) -> object:
    method = getattr(ports, "client", None)
    if not callable(method):
        _fail("recovery handoff AWS client boundary is absent")
    return method(service)


def _sole_audited_object(
    *,
    s3: object,
    bucket: str,
    key: str,
    allow_absent: bool,
) -> object | None:
    """Exhaustively require zero or one sole current non-delete version."""

    try:
        from .h1f_adapter import _audit_row, _coordinate_rows

        rows = _coordinate_rows(s3=s3, bucket=bucket, key=key)
        if not rows:
            if allow_absent:
                return None
            _fail("required immutable S3 coordinate is absent")
        return _audit_row(s3=s3, bucket=bucket, row=rows[0])
    except Task12RecoveryHandoffError:
        raise
    except (TypeError, ValueError) as exc:
        raise Task12RecoveryHandoffError(
            "immutable S3 coordinate history drifted"
        ) from exc


def _json_object(raw: object, label: str) -> dict[str, object]:
    if (
        type(raw) is not bytes
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
    ):
        _fail(label + " bytes are not canonical JSON plus LF")
    try:
        value = json.loads(raw[:-1].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Task12RecoveryHandoffError(
            label + " bytes are invalid JSON"
        ) from exc
    if type(value) is not dict or canonical_json_bytes(value) + b"\n" != raw:
        _fail(label + " bytes are not canonical JSON plus LF")
    return value


def _decision_authority(
    *,
    ports: object,
    invocation: object,
    action: Mapping[str, object],
) -> tuple[dict[str, object], str, str]:
    bucket = _text(_roles(ports).get("campaign_bucket"), "campaign bucket")
    key = _text(action.get("candidate_key"), "start decision key")
    audited = _sole_audited_object(
        s3=_client(ports, "s3"),
        bucket=bucket,
        key=key,
        allow_absent=False,
    )
    raw = getattr(audited, "raw", None)
    file_sha256 = getattr(audited, "file_sha256", None)
    version_id = getattr(audited, "version_id", None)
    if (
        file_sha256 != action.get("candidate_file_sha256")
        or type(version_id) is not str
        or not version_id
    ):
        _fail("start decision immutable version drifted")
    decision = _json_object(raw, "start decision")
    try:
        from .s3_records import build_immutable_json_candidate
        from .h1f_adapter import reconcile_exact_candidate

        candidate = build_immutable_json_candidate(
            record_kind="start-decision",
            bucket=bucket,
            key=key,
            raw=raw,
            activation_id=getattr(invocation, "activation_id"),
            generation=getattr(invocation, "generation"),
        )
        exact = reconcile_exact_candidate(
            s3=_client(ports, "s3"), candidate=candidate
        )
    except (TypeError, ValueError) as exc:
        raise Task12RecoveryHandoffError(
            "start decision metadata/checksum drifted"
        ) from exc
    if (
        exact.state != "sole-version"
        or exact.object_identity is None
        or exact.object_identity.version_id != version_id
    ):
        _fail("start decision exact adoption drifted")
    return decision, version_id, file_sha256


def _production_authority(
    *,
    ports: object,
    invocation: object,
    action: Mapping[str, object],
) -> dict[str, object]:
    bucket = _text(_roles(ports).get("campaign_bucket"), "campaign bucket")
    audited = _sole_audited_object(
        s3=_client(ports, "s3"),
        bucket=bucket,
        key=_TASK10_AUTHORITY_KEY,
        allow_absent=False,
    )
    raw = getattr(audited, "raw", None)
    value = _json_object(raw, "Task10 production authority")
    try:
        from .task10_production import production_authority_from_mapping

        exact = production_authority_from_mapping(value)
    except (TypeError, ValueError) as exc:
        raise Task12RecoveryHandoffError(
            "Task10 production authority drifted"
        ) from exc
    expected_metadata = (
        (
            "canonical-identity-sha256",
            exact.canonical_identity_sha256,
        ),
        ("record-type", "glm52_task10_production_authority_v1"),
    )
    if (
        getattr(audited, "content_type", None) != "application/json"
        or getattr(audited, "metadata", None) != expected_metadata
        or exact.activation_id != getattr(invocation, "activation_id")
        or exact.activation_ordinal
        != getattr(invocation, "activation_ordinal")
        or exact.generation != getattr(invocation, "generation")
        or exact.action_key
        != ledger_sk(
            "glm52_production_action",
            activation_id=action["activation_id"],
            generation=action["generation"],
            action_kind=action["action_kind"],
            attempt=action["attempt"],
        )
        or exact.request_body_sha256
        != action.get("request_body_sha256")
    ):
        _fail("Task10 production authority/action binding drifted")
    return asdict(exact)


def _api_server_identity(ports: object, invocation: object) -> str:
    roles = _roles(ports)
    try:
        from .task11_relay_runtime import load_task9_deployed_identity

        loaded = load_task9_deployed_identity(
            s3=_client(ports, "s3"),
            coordinate=roles.get("task9_deployed_identity_coordinate"),
            activation_id=getattr(invocation, "activation_id"),
            expected_body_sha256=_sha(
                roles.get("task9_deployed_identity_sha256"),
                "Task9 deployed identity",
            ),
            expected_bucket=_text(
                roles.get("campaign_bucket"), "campaign bucket"
            ),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Task12RecoveryHandoffError(
            "Task9 deployed API identity drifted"
        ) from exc
    return loaded.body_sha256


def _handoff_candidate(
    *,
    ports: object,
    invocation: object,
    record: Mapping[str, object],
) -> object:
    try:
        from .task12_writers import build_retained_writer_candidate

        return build_retained_writer_candidate(
            writer_kind="RecoveryHandoff",
            campaign_bucket=_text(
                _roles(ports).get("campaign_bucket"), "campaign bucket"
            ),
            activation_id=getattr(invocation, "activation_id"),
            generation=getattr(invocation, "generation"),
            authority_domain="RECOVERY",
            record=dict(record),
        )
    except (TypeError, ValueError) as exc:
        raise Task12RecoveryHandoffError(
            "recovery handoff writer candidate drifted"
        ) from exc


def _existing_handoff(
    *,
    ports: object,
    invocation: object,
    action: Mapping[str, object],
) -> RecoveryHandoffAdoption | None:
    bucket = _text(_roles(ports).get("campaign_bucket"), "campaign bucket")
    key = _HANDOFF_KEY.format(
        generation_text=getattr(invocation, "generation_text")
    )
    audited = _sole_audited_object(
        s3=_client(ports, "s3"),
        bucket=bucket,
        key=key,
        allow_absent=True,
    )
    if audited is None:
        return None
    body = _json_object(getattr(audited, "raw", None), "SKY_POST_HANDOFF")
    exact_body = validate_normal_task11_handoff(
        body=body,
        invocation=invocation,
        action=action,
    )
    try:
        from .s3_records import build_immutable_json_candidate
        from .h1f_adapter import reconcile_exact_candidate

        candidate = build_immutable_json_candidate(
            record_kind="sky-post-handoff",
            bucket=bucket,
            key=key,
            raw=getattr(audited, "raw"),
            activation_id=getattr(invocation, "activation_id"),
            generation=getattr(invocation, "generation"),
        )
        reconciled = reconcile_exact_candidate(
            s3=_client(ports, "s3"), candidate=candidate
        )
    except (TypeError, ValueError) as exc:
        raise Task12RecoveryHandoffError(
            "normal Task11 handoff metadata/checksum drifted"
        ) from exc
    version_id = getattr(audited, "version_id", None)
    if (
        reconciled.state != "sole-version"
        or reconciled.object_identity is None
        or reconciled.object_identity.version_id != version_id
    ):
        _fail("normal Task11 handoff exact adoption drifted")
    identity_body = {
        "state": "ADOPTED_NORMAL_TASK11_HANDOFF",
        "version_id": version_id,
        "file_sha256": getattr(audited, "file_sha256"),
        "body_sha256": exact_body["handoff_body_sha256"],
    }
    return RecoveryHandoffAdoption(
        **identity_body,
        canonical_identity_sha256=canonical_sha256(identity_body),
    )


def _not_required(action: Mapping[str, object]) -> RecoveryHandoffNotRequired:
    action_identity = canonical_record_identity(
        "glm52_production_action", action
    )
    body = {
        "state": "NOT_REQUIRED_ABANDONED_SKY_ACTION",
        "action_identity_sha256": action_identity,
    }
    return RecoveryHandoffNotRequired(
        **body,
        canonical_identity_sha256=canonical_sha256(body),
    )


def _mode_request(
    *,
    mode: str,
    record: Mapping[str, object],
    action: Mapping[str, object],
    audit: Mapping[str, object],
) -> dict[str, object]:
    if mode not in {"NOT_REQUIRED", "ADOPT_EXISTING", "CREATE"}:
        _fail("recovery handoff request mode drifted")
    return {
        "handoff_record": {"mode": mode, **dict(record)},
        "handoff_action": {"mode": mode, **dict(action)},
        "audit": {"mode": mode, **dict(audit)},
    }


def _exact_head_request_id(
    *,
    ports: object,
    bucket: str,
    key: str,
    version_id: str,
) -> str:
    response = _client(ports, "s3").head_object(
        Bucket=bucket,
        Key=key,
        VersionId=version_id,
        ExpectedBucketOwner=ACCOUNT_ID,
        ChecksumMode="ENABLED",
    )
    metadata = (
        response.get("ResponseMetadata")
        if type(response) is dict
        else None
    )
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or metadata.get("RetryAttempts") != 0
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or response.get("VersionId") != version_id
    ):
        _fail("recovery handoff exact HEAD is unauthenticated")
    return metadata["RequestId"]


def _validated_sources(
    *,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if type(live_sources) is not dict or set(live_sources) != set(
        _SOURCE_TYPES
    ):
        _fail("recovery handoff live source set drifted")
    activation_id = getattr(invocation, "activation_id", None)
    activation_ordinal = getattr(invocation, "activation_ordinal", None)
    generation = getattr(invocation, "generation", None)
    if (
        getattr(invocation, "operation_kind", None)
        != "RETAINED_CREATE_RECOVERY_HANDOFF_IF_REQUIRED"
        or type(activation_id) is not str
        or not activation_id
        or type(activation_ordinal) is not int
        or activation_ordinal < 1
        or type(generation) is not int
        or generation < 1
        or getattr(invocation, "generation_text", None)
        != f"{generation:08d}"
    ):
        _fail("recovery handoff invocation identity drifted")
    exact: dict[str, dict[str, object]] = {}
    campaign_identity: object = None
    for alias, record_type in _SOURCE_TYPES.items():
        try:
            record = validate_record(record_type, live_sources[alias])
        except (TypeError, ValueError) as exc:
            raise Task12RecoveryHandoffError(
                alias + " recovery handoff source drifted"
            ) from exc
        if alias == "activation_index":
            if (
                record.get("current_activation_id") != activation_id
                or record.get("current_activation_ordinal")
                != activation_ordinal
            ):
                _fail("recovery handoff activation index drifted")
            campaign_identity = record.get("campaign_identity_sha256")
        elif (
            record.get("activation_id") != activation_id
            or record.get("activation_ordinal") != activation_ordinal
            or record.get("campaign_identity_sha256") != campaign_identity
        ):
            _fail(alias + " recovery handoff activation forked")
        exact[alias] = record
    return exact


def _validate_authority_tuple(
    *,
    invocation: object,
    sources: Mapping[str, Mapping[str, object]],
) -> None:
    _validate_authority_state(
        invocation=invocation,
        sources=sources,
        expected_action_state="POST_CLASSIFIED",
    )


def _validate_authority_state(
    *,
    invocation: object,
    sources: Mapping[str, Mapping[str, object]],
    expected_action_state: str,
) -> None:
    control = sources["control"]
    execution = sources["execution"]
    recovery = sources["recovery_control"]
    action = sources["sky_action"]
    action_key = ledger_sk(
        "glm52_production_action",
        activation_id=action["activation_id"],
        generation=action["generation"],
        action_kind=action["action_kind"],
        attempt=action["attempt"],
    )
    if (
        control.get("phase") != "RECOVERY_SEALING"
        or control.get("last_sky_post_action_key") != action_key
        or control.get("last_sky_post_generation") != action.get("generation")
        or control.get("last_sky_post_state") != action.get("state")
        or control.get("active_epoch") != execution.get("epoch")
        or control.get("active_execution_arn")
        != execution.get("expected_execution_arn")
        or control.get("active_state_machine_version_arn")
        != execution.get("expected_state_machine_version_arn")
        or execution.get("state") not in _TERMINAL_EXECUTION_STATES
        or execution.get("terminal_status") != execution.get("state")
        or recovery.get("state") != "OWNED"
        or recovery.get("owner_execution_arn")
        != getattr(invocation, "state_machine_execution_arn", None)
        or recovery.get("owner_state_machine_version_arn")
        != getattr(invocation, "caller_state_machine_version_arn", None)
        or action.get("action_kind") != "SKY_POST"
        or action.get("generation") != getattr(invocation, "generation", None)
        or action.get("generation_text")
        != getattr(invocation, "generation_text", None)
        or action.get("state") != expected_action_state
    ):
        _fail("recovery handoff authority tuple drifted")


def build_recovery_handoff_record(
    *,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    decision: Mapping[str, object],
    decision_version_id: object,
    decision_file_sha256: object,
    production_authority: Mapping[str, object],
    api_server_identity_sha256: object,
) -> dict[str, object]:
    """Construct the frozen recovery document from authenticated live truth."""

    sources = _validated_sources(
        invocation=invocation,
        live_sources=live_sources,
    )
    _validate_authority_tuple(invocation=invocation, sources=sources)
    action = sources["sky_action"]
    outcome = action.get("outcome_class")
    if (
        outcome not in _RECOVERY_OUTCOMES
        or action.get("classification_evidence_kind")
        != "OWNER_DEATH_PROOF"
        or action.get("classification_evidence_body_sha256") is None
        or action.get("response_identity_sha256") is not None
    ):
        _fail("recovery handoff Sky classification drifted")
    try:
        from .glm52_sky_production_generation import (
            _validate_decision_intrinsic,
        )

        exact_decision = _validate_decision_intrinsic(decision)
    except (TypeError, ValueError) as exc:
        raise Task12RecoveryHandoffError(
            "recovery handoff start decision drifted"
        ) from exc
    decision_key = _text(action.get("candidate_key"), "decision key")
    if (
        exact_decision.get("run_id") != RUN_ID
        or exact_decision.get("campaign_identity_sha256")
        != action.get("campaign_identity_sha256")
        or exact_decision.get("generation") != action.get("generation")
        or exact_decision.get("generation_text")
        != action.get("generation_text")
        or exact_decision.get("decision") != "launch-once"
        or exact_decision.get("start_decision_body_sha256")
        != action.get("candidate_body_sha256")
        or _sha(decision_file_sha256, "decision file identity")
        != action.get("candidate_file_sha256")
    ):
        _fail("recovery handoff decision/action binding drifted")
    _text(decision_version_id, "decision VersionId")
    if (
        production_authority.get("activation_id")
        != action.get("activation_id")
        or production_authority.get("activation_ordinal")
        != action.get("activation_ordinal")
        or production_authority.get("generation")
        != action.get("generation")
        or production_authority.get("generation_text")
        != action.get("generation_text")
        or production_authority.get("action_key")
        != sources["control"].get("last_sky_post_action_key")
        or production_authority.get("expected_execution_arn")
        != sources["execution"].get("expected_execution_arn")
        or production_authority.get("workflow_version_arn")
        != sources["execution"].get("expected_state_machine_version_arn")
        or production_authority.get("request_body_sha256")
        != action.get("request_body_sha256")
    ):
        _fail("recovery handoff production authority drifted")
    consumed_at = _utc(action.get("consumed_at"), "Sky POST consumed_at")
    completed_at = _utc(
        action.get("completed_at"), "Sky POST classification completion"
    )
    started_at = (
        None
        if action.get("post_started_at") is None
        else _utc(action["post_started_at"], "Sky POST started_at")
    )
    if completed_at < consumed_at or (
        started_at is not None
        and not (consumed_at <= started_at <= completed_at)
    ):
        _fail("recovery handoff POST chronology drifted")
    try:
        return build_sky_post_handoff_record(
            run_id=RUN_ID,
            campaign_identity_sha256=action[
                "campaign_identity_sha256"
            ],
            generation=action["generation"],
            generation_text=action["generation_text"],
            submit_attempt_id=exact_decision["submit_attempt_id"],
            decision_key=decision_key,
            decision_version_id=decision_version_id,
            decision_file_sha256=decision_file_sha256,
            decision_body_sha256=exact_decision[
                "start_decision_body_sha256"
            ],
            sky_post_action_key=sources["control"][
                "last_sky_post_action_key"
            ],
            sky_post_consumed_at=consumed_at,
            sky_post_outcome_class=outcome,
            expected_sky_job_name=exact_decision["sky_job_name"],
            task_yaml_sha256=_sha(
                production_authority.get("task_yaml_sha256"),
                "Task YAML identity",
            ),
            request_body_sha256=_sha(
                production_authority.get("request_body_sha256"),
                "request body identity",
            ),
            api_server_identity_sha256=_sha(
                api_server_identity_sha256,
                "API server identity",
            ),
            sky_request_id=action.get("sky_request_id"),
            post_started_at=started_at,
            post_completed_or_lost_at=completed_at,
            binding_state="reconcile-required",
        )
    except ValueError as exc:
        raise Task12RecoveryHandoffError(
            "recovery handoff frozen body drifted"
        ) from exc


def validate_normal_task11_handoff(
    *,
    body: object,
    invocation: object,
    action: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate the shared frozen normal Task 11 body."""

    try:
        exact = validate_sky_post_handoff_record(body)
    except (TypeError, ValueError) as exc:
        raise Task12RecoveryHandoffError(
            "normal Task 11 handoff schema drifted"
        ) from exc
    if (
        exact.get("generation")
        != getattr(invocation, "generation", None)
        or exact.get("campaign_identity_sha256")
        != action.get("campaign_identity_sha256")
        or exact.get("sky_post_action_key")
        != ledger_sk(
            "glm52_production_action",
            activation_id=action.get("activation_id"),
            generation=action.get("generation"),
            action_kind=action.get("action_kind"),
            attempt=action.get("attempt"),
        )
        or action.get("state") != "POST_CLASSIFIED"
        or exact.get("sky_post_outcome_class")
        != action.get("outcome_class")
        or action.get("outcome_class")
        not in {"ACCEPTED", "KNOWN_REJECTED", "AMBIGUOUS"}
        or action.get("classification_evidence_kind") != "RELAY_RESPONSE"
        or exact.get("request_body_sha256")
        != action.get("request_body_sha256")
        or exact.get("sky_request_id") != action.get("sky_request_id")
        or exact.get("sky_post_consumed_at") != action.get("consumed_at")
        or exact.get("post_started_at") != action.get("post_started_at")
        or exact.get("post_completed_or_lost_at")
        != action.get("completed_at")
    ):
        _fail("normal Task 11 handoff authority drifted")
    return exact


def _continuation_capsule(invocation: object) -> dict[str, object]:
    operation_input = getattr(invocation, "operation_input", None)
    prior = (
        operation_input.get("task12_last_result")
        if isinstance(operation_input, Mapping)
        else None
    )
    result = prior.get("result") if isinstance(prior, Mapping) else None
    capsule = (
        result.get("owner_nonce_capsule")
        if isinstance(result, Mapping)
        else None
    )
    if type(capsule) is not dict:
        _fail("recovery handoff owner nonce continuation is absent")
    body = dict(capsule)
    supplied = body.pop("canonical_body_sha256", None)
    if supplied != canonical_sha256(body):
        _fail("recovery handoff owner nonce capsule drifted")
    return dict(capsule)


def _owner_nonce(
    *,
    ports: object,
    invocation: object,
    recovery: Mapping[str, object],
    capsule: Mapping[str, object],
) -> bytes:
    expected = {
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": getattr(invocation, "activation_id"),
        "authority_domain": "RECOVERY",
        "owner_execution_arn": recovery["owner_execution_arn"],
        "owner_state_machine_version_arn": recovery[
            "owner_state_machine_version_arn"
        ],
        "owner_attempt": recovery["owner_attempt"],
        "barrier_nonce_sha256": recovery[
            "recovery_barrier_nonce_sha256"
        ],
        "control_revision": recovery[
            "support_control_revision_at_seal"
        ],
        "owner_hard_expires_at": recovery["owner_hard_expires_at"],
    }
    try:
        from .task12_nonce_capsule import decrypt_owner_nonce_capsule

        nonce = decrypt_owner_nonce_capsule(
            ports=ports,
            capsule=capsule,
            expected_authority=expected,
        )
    except (TypeError, ValueError) as exc:
        raise Task12RecoveryHandoffError(
            "recovery handoff owner nonce could not be authenticated"
        ) from exc
    if hashlib.sha256(nonce).hexdigest() != recovery.get(
        "owner_invocation_nonce_sha256"
    ):
        _fail("recovery handoff nonce does not own recovery control")
    return nonce


def _rehash(record: Mapping[str, object]) -> dict[str, object]:
    body = dict(record)
    identity = body.pop("canonical_body_sha256", None)
    if identity is None:
        return body
    return {**body, "canonical_body_sha256": canonical_sha256(body)}


def _ledger(ports: object) -> object:
    from .dynamodb import DynamoLedgerAdapter

    return DynamoLedgerAdapter(
        client=_client(ports, "dynamodb"),
        table_name=_text(
            _roles(ports).get("ledger_table_name"), "ledger table"
        ),
    )


def _transaction(
    result: object,
    *,
    expected_records: tuple[Mapping[str, object], ...],
    label: str,
) -> None:
    try:
        from .dynamodb import TransactionResolution, WriteOutcome

        valid = (
            type(result) is TransactionResolution
            and result.outcome
            in {
                WriteOutcome.EXACT_LIVE_OWNER_COMMIT,
                WriteOutcome.EXACT_DURABLE_ADOPTION,
            }
            and all(
                any(actual == expected for actual in result.records)
                for expected in expected_records
            )
        )
    except (AttributeError, TypeError):
        valid = False
    if not valid:
        _fail(label + " transaction was not durably proven")


def _recovery_action_record(
    *,
    invocation: object,
    recovery: Mapping[str, object],
    candidate: object,
    observed_at: str,
) -> dict[str, object]:
    key = ledger_sk(
        "glm52_production_recovery_action",
        activation_id=getattr(invocation, "activation_id"),
        action_kind="RECOVERY_HANDOFF",
        attempt=1,
    )
    arming_token_identity = canonical_sha256(
        {
            "operation": "RECOVERY_HANDOFF_ARM",
            "activation_id": getattr(invocation, "activation_id"),
            "generation": getattr(invocation, "generation"),
            "candidate_identity_sha256": (
                candidate.candidate_identity_sha256
            ),
            "owner_invocation_nonce_sha256": recovery[
                "owner_invocation_nonce_sha256"
            ],
            "control_revision": recovery["revision"],
        }
    )
    body = {
        "schema_version": 1,
        "record_type": "glm52_production_recovery_action",
        "authority_domain": "RECOVERY",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "campaign_identity_sha256": recovery[
            "campaign_identity_sha256"
        ],
        "activation_id": getattr(invocation, "activation_id"),
        "activation_ordinal": getattr(invocation, "activation_ordinal"),
        "action_kind": "RECOVERY_HANDOFF",
        "attempt": 1,
        "candidate_key": str(candidate.coordinate).split("/", 3)[-1],
        "candidate_body_sha256": candidate.body_sha256,
        "request_body_sha256": candidate.candidate_identity_sha256,
        "owner_attempt": recovery["owner_attempt"],
        "owner_execution_arn": recovery["owner_execution_arn"],
        "owner_state_machine_version_arn": recovery[
            "owner_state_machine_version_arn"
        ],
        "owner_dispatch_identity_sha256": recovery[
            "owner_dispatch_identity_sha256"
        ],
        "owner_invocation_nonce_sha256": recovery[
            "owner_invocation_nonce_sha256"
        ],
        "owner_hard_expires_at": recovery["owner_hard_expires_at"],
        "authority_barrier_nonce_sha256": recovery[
            "recovery_barrier_nonce_sha256"
        ],
        "authority_audit_body_sha256": None,
        "authority_audit_closing_revision": None,
        "authorized_transition_from_revision": None,
        "authorized_transition_to_revision": None,
        "state": "ARMED",
        "armed_at": observed_at,
        "consumed_at": None,
        "completed_at": None,
        "response_identity_sha256": None,
        "reconciliation_identity_sha256": None,
        "arming_transaction_client_request_token_sha256": (
            arming_token_identity
        ),
        "consume_transaction_client_request_token_sha256": None,
        "revision": 1,
        "generation": getattr(invocation, "generation"),
        "generation_text": getattr(invocation, "generation_text"),
        "allocation_ordinal": None,
        "allocation_ordinal_text": None,
        "worker_launch_identity_sha256": None,
        "worker_launch_liability_identity_sha256": None,
    }
    try:
        return validate_record(
            "glm52_production_recovery_action", body, sk=key
        )
    except (TypeError, ValueError) as exc:
        raise Task12RecoveryHandoffError(
            "recovery handoff ARMED action drifted"
        ) from exc


def _fresh_authority_reader(
    *,
    ports: object,
    invocation: object,
) -> object:
    def read_current() -> object:
        from .h1f_adapter import (
            H1fAuthoritySnapshot,
            _audit_row,
            _list_versions,
        )

        rows = _ledger(ports).read_coherent(
            items=(
                (
                    LedgerKey(RUN_ID, ledger_sk("glm52_production_activation_index")),
                    "glm52_production_activation_index",
                ),
                (
                    LedgerKey(
                        RUN_ID,
                        ledger_sk(
                            "glm52_production_control",
                            activation_id=getattr(invocation, "activation_id"),
                        ),
                    ),
                    "glm52_production_control",
                ),
                (
                    LedgerKey(
                        RUN_ID,
                        ledger_sk(
                            "glm52_production_recovery_control",
                            activation_id=getattr(invocation, "activation_id"),
                        ),
                    ),
                    "glm52_production_recovery_control",
                ),
            )
        )
        if len(rows) != 3:
            _fail("recovery H1f current authority is incomplete")
        index, control, recovery = rows
        if (
            index["current_activation_id"]
            != getattr(invocation, "activation_id")
            or control["phase"] != "RECOVERY_SEALING"
            or recovery["state"] != "OWNED"
            or recovery["owner_execution_arn"]
            != getattr(invocation, "state_machine_execution_arn")
            or recovery["owner_state_machine_version_arn"]
            != getattr(invocation, "caller_state_machine_version_arn")
        ):
            _fail("recovery H1f current authority drifted")
        bucket = _text(
            _roles(ports).get("campaign_bucket"), "campaign bucket"
        )
        prefix = f"campaigns/{RUN_ID}/authorities/fence/"
        versions = _list_versions(
            s3=_client(ports, "s3"), bucket=bucket, prefix=prefix
        )
        matches = tuple(
            row
            for row in versions
            if not row.is_delete_marker
            and row.version_id == control["fence_head_version_id"]
        )
        if (
            len(matches) != 1
            or any(
                row.is_delete_marker
                and row.version_id == control["fence_head_version_id"]
                for row in versions
            )
        ):
            _fail("recovery H1f fence head VersionId is not unique")
        head = _audit_row(
            s3=_client(ports, "s3"), bucket=bucket, row=matches[0]
        )
        raw = head.raw
        parsed = _json_object(raw, "recovery H1f fence head")
        if (
            parsed.get("fence_body_sha256")
            != control["fence_head_body_sha256"]
            or head.file_sha256
            != hashlib.sha256(raw).hexdigest()
        ):
            _fail("recovery H1f fence head identity drifted")
        return H1fAuthoritySnapshot(
            authority_domain="RECOVERY",
            bucket=bucket,
            run_id=RUN_ID,
            activation_id=getattr(invocation, "activation_id"),
            generation=getattr(invocation, "generation"),
            epoch=recovery["owner_attempt"],
            execution_arn=recovery["owner_execution_arn"],
            barrier_nonce_sha256=recovery[
                "recovery_barrier_nonce_sha256"
            ],
            closing_revision=recovery["revision"],
            active_head_key=head.key,
            active_head_version_id=head.version_id,
            active_head_file_sha256=head.file_sha256,
            active_head_body_sha256=control[
                "fence_head_body_sha256"
            ],
        )

    return read_current


def _arm_and_consume(
    *,
    ports: object,
    invocation: object,
    sources: Mapping[str, Mapping[str, object]],
    candidate: object,
    capsule: Mapping[str, object],
    existing_armed_action: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    from .h1f_adapter import (
        FreshH1fAuditService,
        H1fAuditRequest,
    )
    from .s3_records import build_immutable_json_candidate
    from .transitions import validate_transition

    index = sources["activation_index"]
    control = sources["control"]
    recovery = sources["recovery_control"]
    raw_nonce = _owner_nonce(
        ports=ports,
        invocation=invocation,
        recovery=recovery,
        capsule=capsule,
    )
    action_key = ledger_sk(
        "glm52_production_recovery_action",
        activation_id=getattr(invocation, "activation_id"),
        action_kind="RECOVERY_HANDOFF",
        attempt=1,
    )
    if existing_armed_action is None:
        observed_at = _now()
        action = _recovery_action_record(
            invocation=invocation,
            recovery=recovery,
            candidate=candidate,
            observed_at=observed_at,
        )
        recovery_after_arm = _rehash(
            {
                **recovery,
                "revision": recovery["revision"] + 1,
                "updated_at": observed_at,
            }
        )
        recovery_after_arm = validate_record(
            "glm52_production_recovery_control", recovery_after_arm
        )
        arm_plan = RecoveryHandoffArmPlan(
            index=ExactCheck(
                LedgerKey(
                    RUN_ID,
                    ledger_sk("glm52_production_activation_index"),
                ),
                index,
            ),
            control=ExactCheck(
                LedgerKey(
                    RUN_ID,
                    ledger_sk(
                        "glm52_production_control",
                        activation_id=getattr(invocation, "activation_id"),
                    ),
                ),
                control,
            ),
            recovery_control=ExactUpdate(
                LedgerKey(
                    RUN_ID,
                    ledger_sk(
                        "glm52_production_recovery_control",
                        activation_id=getattr(invocation, "activation_id"),
                    ),
                ),
                recovery,
                recovery_after_arm,
            ),
            action=ExactPut(LedgerKey(RUN_ID, action_key), action),
        )
        arm_identity = canonical_sha256(
            {
                "operation": "RECOVERY_HANDOFF_ARM",
                "candidate_identity_sha256": (
                    candidate.candidate_identity_sha256
                ),
                "recovery_before": canonical_record_identity(
                    "glm52_production_recovery_control", recovery
                ),
                "action_identity_sha256": canonical_record_identity(
                    "glm52_production_recovery_action", action
                ),
            }
        )
        arm_result = _ledger(ports).commit_recovery_handoff_arm(
            plan=arm_plan,
            domain="RECOVERY",
            operation_identity_sha256=arm_identity,
            raw_owner_nonce=raw_nonce,
        )
        materialized_arm = tuple(
            row
            for row in getattr(arm_result, "records", ())
            if isinstance(row, Mapping)
        )
        armed_action_rows = tuple(
            row
            for row in materialized_arm
            if row.get("record_type")
            == "glm52_production_recovery_action"
        )
        recovery_rows = tuple(
            row
            for row in materialized_arm
            if row.get("record_type")
            == "glm52_production_recovery_control"
        )
        if len(armed_action_rows) != 1 or len(recovery_rows) != 1:
            _fail("recovery handoff ARM readback is incomplete")
        armed_action = dict(armed_action_rows[0])
        recovery_after_arm = dict(recovery_rows[0])
        _transaction(
            arm_result,
            expected_records=(armed_action, recovery_after_arm),
            label="recovery handoff ARM",
        )
    else:
        recovery_after_arm = recovery
        armed_action = _validate_recovery_action_binding(
            action=existing_armed_action,
            recovery=recovery_after_arm,
            candidate=candidate,
            invocation=invocation,
            allowed_states=frozenset({"ARMED"}),
        )
        arm_identity = canonical_sha256(
            {
                "operation": "RECOVERY_HANDOFF_ARM_DURABLE_ADOPTION",
                "candidate_identity_sha256": (
                    candidate.candidate_identity_sha256
                ),
                "recovery_after_arm": canonical_record_identity(
                    "glm52_production_recovery_control",
                    recovery_after_arm,
                ),
                "armed_action_identity_sha256": canonical_record_identity(
                    "glm52_production_recovery_action",
                    armed_action,
                ),
            }
        )

    s3_candidate = build_immutable_json_candidate(
        record_kind="recovery-handoff",
        bucket=candidate.campaign_bucket,
        key=_candidate_key(candidate),
        raw=candidate.raw,
        activation_id=getattr(invocation, "activation_id"),
        generation=getattr(invocation, "generation"),
    )
    try:
        audit = FreshH1fAuditService(
            authority_reader=_fresh_authority_reader(
                ports=ports, invocation=invocation
            )
        ).fresh_audit(
            s3=_client(ports, "s3"),
            request=H1fAuditRequest(
                operation_kind="S3_CREATE",
                action_key=action_key,
                candidate=s3_candidate,
            ),
        )
    except (TypeError, ValueError) as exc:
        raise Task12RecoveryHandoffError(
            "recovery handoff fresh H1f audit failed"
        ) from exc
    if (
        audit.authority_domain != "RECOVERY"
        or audit.closing_revision != recovery_after_arm["revision"]
        or audit.expected_authorized_revision
        != recovery_after_arm["revision"] + 1
    ):
        _fail("recovery handoff fresh H1f revision drifted")
    consumed_at = _now()
    action_after = dict(armed_action)
    action_after.update(
        state="CONSUMED",
        authority_audit_body_sha256=audit.canonical_body_sha256,
        authority_audit_closing_revision=audit.closing_revision,
        authorized_transition_from_revision=audit.closing_revision,
        authorized_transition_to_revision=audit.expected_authorized_revision,
        consumed_at=consumed_at,
        consume_transaction_client_request_token_sha256=canonical_sha256(
            {
                "operation": "RECOVERY_HANDOFF_CONSUME",
                "candidate_identity_sha256": (
                    candidate.candidate_identity_sha256
                ),
                "armed_action_identity_sha256": canonical_record_identity(
                    "glm52_production_recovery_action", armed_action
                ),
                "audit_identity_sha256": audit.canonical_body_sha256,
            }
        ),
        revision=armed_action["revision"] + 1,
    )
    action_after = validate_transition(
        "glm52_production_recovery_action",
        armed_action,
        action_after,
    )
    recovery_after_consume = _rehash(
        {
            **recovery_after_arm,
            "revision": recovery_after_arm["revision"] + 1,
            "updated_at": consumed_at,
        }
    )
    recovery_after_consume = validate_record(
        "glm52_production_recovery_control",
        recovery_after_consume,
    )
    consume_plan = RecoveryHandoffConsumePlan(
        index=ExactCheck(
            LedgerKey(
                RUN_ID, ledger_sk("glm52_production_activation_index")
            ),
            index,
        ),
        control=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_control",
                    activation_id=getattr(invocation, "activation_id"),
                ),
            ),
            control,
        ),
        recovery_control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_recovery_control",
                    activation_id=getattr(invocation, "activation_id"),
                ),
            ),
            recovery_after_arm,
            recovery_after_consume,
        ),
        action=ExactUpdate(
            LedgerKey(RUN_ID, action_key),
            armed_action,
            action_after,
        ),
    )
    consume_identity = canonical_sha256(
        {
            "operation": "RECOVERY_HANDOFF_CONSUME",
            "arm_identity_sha256": arm_identity,
            "audit_identity_sha256": audit.canonical_body_sha256,
            "action_before": canonical_record_identity(
                "glm52_production_recovery_action", armed_action
            ),
            "action_after": canonical_record_identity(
                "glm52_production_recovery_action", action_after
            ),
        }
    )
    consume_result = _ledger(ports).commit_recovery_handoff_consume(
        plan=consume_plan,
        domain="RECOVERY",
        operation_identity_sha256=consume_identity,
        raw_owner_nonce=raw_nonce,
    )
    materialized_consume = tuple(
        row
        for row in getattr(consume_result, "records", ())
        if isinstance(row, Mapping)
    )
    consumed_rows = tuple(
        row
        for row in materialized_consume
        if row.get("record_type") == "glm52_production_recovery_action"
    )
    if len(consumed_rows) != 1:
        _fail("recovery handoff consume readback is incomplete")
    consumed_action = dict(consumed_rows[0])
    _transaction(
        consume_result,
        expected_records=(consumed_action,),
        label="recovery handoff consume",
    )
    return consumed_action, asdict(audit)


def _writer_authorities(
    *,
    candidate: object,
    durable_action: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    from .task12_writers import (
        RetainedWriterActionAuthority,
        RetainedWriterAuditAuthority,
    )

    action_key = ledger_sk(
        "glm52_production_recovery_action",
        activation_id=durable_action["activation_id"],
        action_kind="RECOVERY_HANDOFF",
        attempt=durable_action["attempt"],
    )
    action_body = {
        "authority_domain": "RECOVERY",
        "action_kind": "RECOVERY_HANDOFF",
        "action_key": action_key,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "owner_invocation_nonce_sha256": durable_action[
            "owner_invocation_nonce_sha256"
        ],
        "state": "CONSUMED",
        "authorized_revision": durable_action[
            "authorized_transition_to_revision"
        ],
    }
    action = RetainedWriterActionAuthority(
        **action_body,
        action_identity_sha256=canonical_record_identity(
            "glm52_production_recovery_action", durable_action
        ),
    )
    audit_body = {
        "authority_domain": "RECOVERY",
        "action_kind": "RECOVERY_HANDOFF",
        "action_key": action_key,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "action_identity_sha256": action.action_identity_sha256,
        "audit_kind": "H1F_GENESIS_TO_ZERO_CHILD",
        "closing_revision": durable_action[
            "authority_audit_closing_revision"
        ],
        "authorized_revision": durable_action[
            "authorized_transition_to_revision"
        ],
        "current_revision": durable_action[
            "authorized_transition_to_revision"
        ],
        "observed_at": durable_action["consumed_at"],
    }
    audit = RetainedWriterAuditAuthority(
        **audit_body,
        canonical_identity_sha256=canonical_sha256(audit_body),
    )
    return asdict(action), asdict(audit)


def _read_recovery_action(
    *,
    ports: object,
    invocation: object,
) -> dict[str, object]:
    key = ledger_sk(
        "glm52_production_recovery_action",
        activation_id=getattr(invocation, "activation_id"),
        action_kind="RECOVERY_HANDOFF",
        attempt=1,
    )
    try:
        rows = _ledger(ports).read_coherent(
            items=(
                (
                    LedgerKey(RUN_ID, key),
                    "glm52_production_recovery_action",
                ),
            )
        )
    except Exception as exc:
        raise Task12RecoveryHandoffError(
            "recovery handoff action is absent"
        ) from exc
    if len(rows) != 1:
        _fail("recovery handoff action readback is ambiguous")
    return dict(rows[0])


def _maybe_read_recovery_action(
    *,
    ports: object,
    invocation: object,
) -> dict[str, object] | None:
    key = ledger_sk(
        "glm52_production_recovery_action",
        activation_id=getattr(invocation, "activation_id"),
        action_kind="RECOVERY_HANDOFF",
        attempt=1,
    )
    try:
        row = _ledger(ports).read_consistent_item(
            key=LedgerKey(RUN_ID, key),
            record_type="glm52_production_recovery_action",
        )
    except Exception as exc:
        raise Task12RecoveryHandoffError(
            "recovery handoff action read failed closed"
        ) from exc
    return None if row is None else dict(row)


def _candidate_key(candidate: object) -> str:
    coordinate = getattr(candidate, "coordinate", None)
    if type(coordinate) is not str or not coordinate.startswith("s3://"):
        _fail("recovery handoff candidate coordinate drifted")
    bucket_key = coordinate[5:].split("/", 1)
    if len(bucket_key) != 2 or bucket_key[0] != getattr(
        candidate, "campaign_bucket", None
    ):
        _fail("recovery handoff candidate bucket drifted")
    return bucket_key[1]


def _validate_recovery_action_binding(
    *,
    action: Mapping[str, object],
    recovery: Mapping[str, object],
    candidate: object,
    invocation: object,
    allowed_states: frozenset[str],
) -> dict[str, object]:
    try:
        exact = validate_record(
            "glm52_production_recovery_action",
            action,
            sk=ledger_sk(
                "glm52_production_recovery_action",
                activation_id=getattr(invocation, "activation_id"),
                action_kind="RECOVERY_HANDOFF",
                attempt=1,
            ),
        )
    except (TypeError, ValueError) as exc:
        raise Task12RecoveryHandoffError(
            "recovery handoff durable action drifted"
        ) from exc
    if (
        exact["state"] not in allowed_states
        or exact["authority_domain"] != "RECOVERY"
        or exact["action_kind"] != "RECOVERY_HANDOFF"
        or exact["activation_id"] != getattr(invocation, "activation_id")
        or exact["activation_ordinal"]
        != getattr(invocation, "activation_ordinal")
        or exact["generation"] != getattr(invocation, "generation")
        or exact["generation_text"]
        != getattr(invocation, "generation_text")
        or exact["attempt"] != 1
        or exact["candidate_key"] != _candidate_key(candidate)
        or exact["candidate_body_sha256"]
        != getattr(candidate, "body_sha256", None)
        or exact["request_body_sha256"]
        != getattr(candidate, "candidate_identity_sha256", None)
        or exact["owner_attempt"] != recovery["owner_attempt"]
        or exact["owner_execution_arn"]
        != recovery["owner_execution_arn"]
        or exact["owner_state_machine_version_arn"]
        != recovery["owner_state_machine_version_arn"]
        or exact["owner_dispatch_identity_sha256"]
        != recovery["owner_dispatch_identity_sha256"]
        or exact["owner_invocation_nonce_sha256"]
        != recovery["owner_invocation_nonce_sha256"]
        or exact["owner_hard_expires_at"]
        != recovery["owner_hard_expires_at"]
        or exact["authority_barrier_nonce_sha256"]
        != recovery["recovery_barrier_nonce_sha256"]
    ):
        _fail("recovery handoff action/candidate binding drifted")
    if exact["state"] == "ARMED":
        if (
            exact["revision"] != 1
            or recovery["updated_at"] != exact["armed_at"]
        ):
            _fail("recovery handoff ARMED revision drifted")
    elif exact["state"] == "CONSUMED":
        if (
            recovery["revision"]
            != exact["authorized_transition_to_revision"]
            or recovery["updated_at"] != exact["consumed_at"]
        ):
            _fail("recovery handoff CONSUMED revision drifted")
    elif exact["state"] == "COMPLETED" and (
        recovery["revision"]
        != exact["authorized_transition_to_revision"] + 1
        or recovery["updated_at"] != exact["completed_at"]
    ):
        _fail("recovery handoff COMPLETED revision drifted")
    return exact


def _completed_recovery_adoption(
    *,
    ports: object,
    invocation: object,
    recovery: Mapping[str, object],
    action: Mapping[str, object],
    candidate: object,
    version_id: str,
) -> RecoveryHandoffAdoption:
    exact_action = _validate_recovery_action_binding(
        action=action,
        recovery=recovery,
        candidate=candidate,
        invocation=invocation,
        allowed_states=frozenset({"COMPLETED"}),
    )
    control_key = ledger_sk(
        "glm52_task12_versioned_writer_control_v1",
        activation_id=getattr(invocation, "activation_id"),
        generation=getattr(invocation, "generation"),
        writer_kind="RecoveryHandoff",
    )
    try:
        rows = _ledger(ports).read_coherent(
            items=(
                (
                    LedgerKey(RUN_ID, control_key),
                    "glm52_task12_versioned_writer_control_v1",
                ),
            )
        )
    except Exception as exc:
        raise Task12RecoveryHandoffError(
            "completed recovery handoff lacks version control"
        ) from exc
    if len(rows) != 1:
        _fail("completed recovery handoff version control is ambiguous")
    control = rows[0]
    expected = {
        "activation_id": getattr(invocation, "activation_id"),
        "generation": getattr(invocation, "generation"),
        "generation_text": getattr(invocation, "generation_text"),
        "writer_kind": "RecoveryHandoff",
        "campaign_bucket": candidate.campaign_bucket,
        "object_key": _candidate_key(candidate),
        "object_version_id": version_id,
        "file_sha256": candidate.file_sha256,
        "body_sha256": candidate.body_sha256,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "writer_result_identity_sha256": exact_action[
            "response_identity_sha256"
        ],
        "published_at": exact_action["consumed_at"],
    }
    if (
        any(control.get(field) != value for field, value in expected.items())
        or exact_action["reconciliation_identity_sha256"] is not None
    ):
        _fail("completed recovery handoff version binding drifted")
    identity_body = {
        "state": "ADOPTED_COMPLETED_RECOVERY_HANDOFF",
        "version_id": version_id,
        "file_sha256": candidate.file_sha256,
        "body_sha256": candidate.body_sha256,
    }
    return RecoveryHandoffAdoption(
        **identity_body,
        canonical_identity_sha256=canonical_sha256(identity_body),
    )


def materialize_live_request(
    *,
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    ports: object,
) -> Mapping[str, object] | None:
    """Build and, for recovery create, ARM/audit/consume the live writer."""

    if operation_kind not in SUPPORTED_OPERATIONS:
        return None
    sources = _validated_sources(
        invocation=invocation,
        live_sources=live_sources,
    )
    action = sources["sky_action"]
    if action.get("state") == "ABANDONED":
        _validate_authority_state(
            invocation=invocation,
            sources=sources,
            expected_action_state="ABANDONED",
        )
        result = _not_required(action)
        payload = asdict(result)
        return _mode_request(
            mode="NOT_REQUIRED",
            record=payload,
            action={"action_identity_sha256": result.action_identity_sha256},
            audit={"canonical_identity_sha256": result.canonical_identity_sha256},
        )
    _validate_authority_tuple(invocation=invocation, sources=sources)
    if action.get("classification_evidence_kind") == "RELAY_RESPONSE":
        adoption = _existing_handoff(
            ports=ports,
            invocation=invocation,
            action=action,
        )
        if adoption is None:
            _fail("normal Task11 handoff is absent after POST classification")
        return _mode_request(
            mode="ADOPT_EXISTING",
            record=asdict(adoption),
            action={
                "action_identity_sha256": canonical_record_identity(
                    "glm52_production_action", action
                )
            },
            audit={
                "canonical_identity_sha256": adoption.canonical_identity_sha256
            },
        )
    decision, version_id, file_sha256 = _decision_authority(
        ports=ports,
        invocation=invocation,
        action=action,
    )
    production = _production_authority(
        ports=ports,
        invocation=invocation,
        action=action,
    )
    api_identity = _api_server_identity(ports, invocation)
    record = build_recovery_handoff_record(
        invocation=invocation,
        live_sources=sources,
        decision=decision,
        decision_version_id=version_id,
        decision_file_sha256=file_sha256,
        production_authority=production,
        api_server_identity_sha256=api_identity,
    )
    candidate = _handoff_candidate(
        ports=ports,
        invocation=invocation,
        record=record,
    )
    audited = _sole_audited_object(
        s3=_client(ports, "s3"),
        bucket=candidate.campaign_bucket,
        key=_candidate_key(candidate),
        allow_absent=True,
    )
    if audited is not None:
        existing = _json_object(
            getattr(audited, "raw", None), "recovery SKY_POST_HANDOFF"
        )
        if existing != record:
            _fail("existing recovery handoff bytes are foreign")
        try:
            from .s3_records import build_immutable_json_candidate
            from .h1f_adapter import reconcile_exact_candidate

            exact_s3 = build_immutable_json_candidate(
                record_kind="recovery-handoff",
                bucket=candidate.campaign_bucket,
                key=_candidate_key(candidate),
                raw=candidate.raw,
                activation_id=getattr(invocation, "activation_id"),
                generation=getattr(invocation, "generation"),
            )
            reconciliation = reconcile_exact_candidate(
                s3=_client(ports, "s3"), candidate=exact_s3
            )
        except (TypeError, ValueError) as exc:
            raise Task12RecoveryHandoffError(
                "existing recovery handoff reconciliation failed"
            ) from exc
        if (
            reconciliation.state != "sole-version"
            or reconciliation.object_identity is None
            or reconciliation.object_identity.version_id
            != getattr(audited, "version_id", None)
        ):
            _fail("existing recovery handoff version drifted")
        durable_action = _read_recovery_action(
            ports=ports, invocation=invocation
        )
        if durable_action.get("state") not in {"CONSUMED", "COMPLETED"}:
            _fail("existing recovery handoff has no consumed action")
        if durable_action["state"] == "COMPLETED":
            adoption = _completed_recovery_adoption(
                ports=ports,
                invocation=invocation,
                recovery=sources["recovery_control"],
                action=durable_action,
                candidate=candidate,
                version_id=reconciliation.object_identity.version_id,
            )
            return _mode_request(
                mode="ADOPT_EXISTING",
                record=asdict(adoption),
                action={
                    "action_identity_sha256": canonical_record_identity(
                        "glm52_production_recovery_action",
                        durable_action,
                    )
                },
                audit={
                    "canonical_identity_sha256": (
                        adoption.canonical_identity_sha256
                    )
                },
            )
        durable_action = _validate_recovery_action_binding(
            action=durable_action,
            recovery=sources["recovery_control"],
            candidate=candidate,
            invocation=invocation,
            allowed_states=frozenset({"CONSUMED"}),
        )
        capsule = _continuation_capsule(invocation)
        writer_action, writer_audit = _writer_authorities(
            candidate=candidate,
            durable_action=durable_action,
        )
        return {
            "handoff_record": {
                "mode": "ADOPT_RECOVERY",
                "record": record,
                "version_id": reconciliation.object_identity.version_id,
                "file_sha256": candidate.file_sha256,
                "body_sha256": candidate.body_sha256,
                "request_id": _exact_head_request_id(
                    ports=ports,
                    bucket=candidate.campaign_bucket,
                    key=_candidate_key(candidate),
                    version_id=reconciliation.object_identity.version_id,
                ),
            },
            "handoff_action": {
                "mode": "ADOPT_RECOVERY",
                "durable_action": durable_action,
                "writer_action": writer_action,
                "owner_nonce_capsule": capsule,
            },
            "audit": {
                "mode": "ADOPT_RECOVERY",
                "writer_audit": writer_audit,
            },
        }
    capsule = _continuation_capsule(invocation)
    durable_action = _maybe_read_recovery_action(
        ports=ports,
        invocation=invocation,
    )
    if durable_action is None:
        consumed_action, _h1f = _arm_and_consume(
            ports=ports,
            invocation=invocation,
            sources=sources,
            candidate=candidate,
            capsule=capsule,
        )
    elif durable_action["state"] == "ARMED":
        consumed_action, _h1f = _arm_and_consume(
            ports=ports,
            invocation=invocation,
            sources=sources,
            candidate=candidate,
            capsule=capsule,
            existing_armed_action=durable_action,
        )
    elif durable_action["state"] == "CONSUMED":
        consumed_action = _validate_recovery_action_binding(
            action=durable_action,
            recovery=sources["recovery_control"],
            candidate=candidate,
            invocation=invocation,
            allowed_states=frozenset({"CONSUMED"}),
        )
    else:
        _fail("recovery handoff S3 absence contradicts durable action")
    writer_action, writer_audit = _writer_authorities(
        candidate=candidate,
        durable_action=consumed_action,
    )
    return {
        "handoff_record": {"mode": "CREATE", "record": record},
        "handoff_action": {
            "mode": "CREATE",
            "durable_action": consumed_action,
            "writer_action": writer_action,
            "owner_nonce_capsule": capsule,
        },
        "audit": {"mode": "CREATE", "writer_audit": writer_audit},
    }


def execute_live_request(
    *,
    invocation: object,
    request: Mapping[str, object],
    ports: object,
    writer: object,
) -> object:
    """Execute only the already authenticated materialized handoff request."""

    if type(request) is not dict or set(request) != {
        "handoff_record",
        "handoff_action",
        "audit",
    }:
        _fail("recovery handoff execution request is not closed")
    members = tuple(request[name] for name in ("handoff_record", "handoff_action", "audit"))
    if any(type(member) is not dict for member in members):
        _fail("recovery handoff execution members are not closed")
    modes = {member.get("mode") for member in members}
    if len(modes) != 1:
        _fail("recovery handoff execution modes diverged")
    mode = modes.pop()
    if mode == "NOT_REQUIRED":
        value = dict(request["handoff_record"])
        value.pop("mode")
        try:
            result = RecoveryHandoffNotRequired(**value)
        except TypeError as exc:
            raise Task12RecoveryHandoffError(
                "recovery handoff no-action result drifted"
            ) from exc
        body = asdict(result)
        identity = body.pop("canonical_identity_sha256")
        if identity != canonical_sha256(body):
            _fail("recovery handoff no-action identity drifted")
        return result
    if mode == "ADOPT_EXISTING":
        value = dict(request["handoff_record"])
        value.pop("mode")
        try:
            result = RecoveryHandoffAdoption(**value)
        except TypeError as exc:
            raise Task12RecoveryHandoffError(
                "normal Task11 adoption result drifted"
            ) from exc
        body = asdict(result)
        identity = body.pop("canonical_identity_sha256")
        if identity != canonical_sha256(body):
            _fail("normal Task11 adoption identity drifted")
        return result
    if mode not in {"CREATE", "ADOPT_RECOVERY"}:
        _fail("recovery handoff execution mode is not closed")
    record = request["handoff_record"].get("record")
    if type(record) is not dict:
        _fail("recovery handoff execution record is absent")
    candidate = _handoff_candidate(
        ports=ports,
        invocation=invocation,
        record=record,
    )
    writer_action = request["handoff_action"].get("writer_action")
    writer_audit = request["audit"].get("writer_audit")
    try:
        from .task12_writers import (
            RetainedWriteResult,
            RetainedWriterActionAuthority,
            RetainedWriterAuditAuthority,
            validate_retained_writer_authority,
        )

        action = RetainedWriterActionAuthority(**writer_action)
        audit = RetainedWriterAuditAuthority(**writer_audit)
        validate_retained_writer_authority(
            candidate=candidate,
            action=action,
            audit=audit,
        )
    except (TypeError, ValueError) as exc:
        raise Task12RecoveryHandoffError(
            "recovery handoff writer authority drifted"
        ) from exc
    if mode == "CREATE":
        if not callable(writer):
            _fail("recovery handoff writer boundary is absent")
        return writer(
            record=record,
            action=writer_action,
            audit=writer_audit,
        )
    handoff = request["handoff_record"]
    if (
        handoff.get("file_sha256") != candidate.file_sha256
        or handoff.get("body_sha256") != candidate.body_sha256
        or type(handoff.get("version_id")) is not str
        or not handoff["version_id"]
        or type(handoff.get("request_id")) is not str
        or not handoff["request_id"]
    ):
        _fail("recovery handoff adoption evidence drifted")
    result_body = {
        "writer_kind": "RecoveryHandoff",
        "outcome": "reconciled-exact-existing",
        "coordinate": candidate.coordinate,
        "candidate_identity_sha256": candidate.candidate_identity_sha256,
        "object_version_id": handoff["version_id"],
        "response_request_ids": (handoff["request_id"],),
        "response_authenticated": True,
    }
    return RetainedWriteResult(
        **result_body,
        canonical_identity_sha256=canonical_sha256(result_body),
    )


def persist_live_successors(
    *,
    operation_kind: str,
    invocation: object,
    live_sources: Mapping[str, Mapping[str, object]],
    request: Mapping[str, object],
    domain_result: object,
    ports: object,
) -> bool:
    if operation_kind not in SUPPORTED_OPERATIONS:
        return False
    if isinstance(
        domain_result,
        (RecoveryHandoffNotRequired, RecoveryHandoffAdoption),
    ):
        return True
    try:
        from .task12_writers import (
            RetainedWriteResult,
            build_versioned_writer_control,
        )
    except ImportError as exc:
        raise Task12RecoveryHandoffError(
            "recovery handoff writer result boundary is absent"
        ) from exc
    if type(domain_result) is not RetainedWriteResult:
        _fail("recovery handoff successor requires a proven writer result")
    record = request.get("handoff_record", {}).get("record")
    if type(record) is not dict:
        _fail("recovery handoff successor record is absent")
    candidate = _handoff_candidate(
        ports=ports,
        invocation=invocation,
        record=record,
    )
    action_key = ledger_sk(
        "glm52_production_recovery_action",
        activation_id=getattr(invocation, "activation_id"),
        action_kind="RECOVERY_HANDOFF",
        attempt=1,
    )
    rows = _ledger(ports).read_coherent(
        items=(
            (
                LedgerKey(
                    RUN_ID, ledger_sk("glm52_production_activation_index")
                ),
                "glm52_production_activation_index",
            ),
            (
                LedgerKey(
                    RUN_ID,
                    ledger_sk(
                        "glm52_production_control",
                        activation_id=getattr(invocation, "activation_id"),
                    ),
                ),
                "glm52_production_control",
            ),
            (
                LedgerKey(
                    RUN_ID,
                    ledger_sk(
                        "glm52_production_recovery_control",
                        activation_id=getattr(invocation, "activation_id"),
                    ),
                ),
                "glm52_production_recovery_control",
            ),
            (
                LedgerKey(RUN_ID, action_key),
                "glm52_production_recovery_action",
            ),
        )
    )
    if len(rows) != 4:
        _fail("recovery handoff successor current authority is incomplete")
    index, control, recovery, action = rows
    current_sources = _validated_sources(
        invocation=invocation,
        live_sources={
            "activation_index": index,
            "control": control,
            "execution": live_sources["execution"],
            "recovery_control": recovery,
            "sky_action": live_sources["sky_action"],
        },
    )
    _validate_authority_tuple(
        invocation=invocation,
        sources=current_sources,
    )
    exact_action = _validate_recovery_action_binding(
        action=action,
        recovery=recovery,
        candidate=candidate,
        invocation=invocation,
        allowed_states=frozenset({"CONSUMED", "COMPLETED"}),
    )
    requested_action = request.get("handoff_action", {}).get(
        "durable_action"
    )
    if (
        exact_action["state"] == "CONSUMED"
        and requested_action != exact_action
    ):
        _fail("recovery handoff request/durable action binding drifted")
    action = exact_action
    writer_control = build_versioned_writer_control(
        candidate=candidate,
        result=domain_result,
        published_at=action["consumed_at"],
    )
    control_key = ledger_sk(
        "glm52_task12_versioned_writer_control_v1",
        activation_id=getattr(invocation, "activation_id"),
        generation=getattr(invocation, "generation"),
        writer_kind="RecoveryHandoff",
    )
    if action["state"] == "COMPLETED":
        try:
            existing = _ledger(ports).read_coherent(
                items=(
                    (
                        LedgerKey(RUN_ID, control_key),
                        "glm52_task12_versioned_writer_control_v1",
                    ),
                )
            )
        except Exception as exc:
            raise Task12RecoveryHandoffError(
                "completed recovery handoff lacks version control"
            ) from exc
        if (
            len(existing) != 1
            or existing[0] != writer_control
            or action.get("response_identity_sha256")
            != domain_result.canonical_identity_sha256
            or action.get("reconciliation_identity_sha256") is not None
        ):
            _fail("completed recovery handoff adoption drifted")
        return True
    if action["state"] != "CONSUMED":
        _fail("recovery handoff action is not consumable")
    capsule = request.get("handoff_action", {}).get("owner_nonce_capsule")
    if type(capsule) is not dict:
        _fail("recovery handoff completion nonce capsule is absent")
    raw_nonce = _owner_nonce(
        ports=ports,
        invocation=invocation,
        recovery=recovery,
        capsule=capsule,
    )
    completed_at = _now()
    from .transitions import validate_transition

    action_after = dict(action)
    action_after.update(
        state="COMPLETED",
        completed_at=completed_at,
        response_identity_sha256=domain_result.canonical_identity_sha256,
        reconciliation_identity_sha256=None,
        revision=action["revision"] + 1,
    )
    action_after = validate_transition(
        "glm52_production_recovery_action", action, action_after
    )
    recovery_after = _rehash(
        {
            **recovery,
            "revision": recovery["revision"] + 1,
            "updated_at": completed_at,
        }
    )
    recovery_after = validate_record(
        "glm52_production_recovery_control", recovery_after
    )
    plan = RecoveryHandoffCompletePlan(
        index=ExactCheck(
            LedgerKey(
                RUN_ID, ledger_sk("glm52_production_activation_index")
            ),
            index,
        ),
        control=ExactCheck(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_control",
                    activation_id=getattr(invocation, "activation_id"),
                ),
            ),
            control,
        ),
        recovery_control=ExactUpdate(
            LedgerKey(
                RUN_ID,
                ledger_sk(
                    "glm52_production_recovery_control",
                    activation_id=getattr(invocation, "activation_id"),
                ),
            ),
            recovery,
            recovery_after,
        ),
        action=ExactUpdate(
            LedgerKey(RUN_ID, action_key), action, action_after
        ),
        writer_control=ExactPut(
            LedgerKey(RUN_ID, control_key), writer_control
        ),
    )
    operation_identity = canonical_sha256(
        {
            "operation": "RECOVERY_HANDOFF_COMPLETE",
            "writer_result_identity_sha256": (
                domain_result.canonical_identity_sha256
            ),
            "action_before": canonical_record_identity(
                "glm52_production_recovery_action", action
            ),
            "action_after": canonical_record_identity(
                "glm52_production_recovery_action", action_after
            ),
            "writer_control_identity_sha256": canonical_record_identity(
                "glm52_task12_versioned_writer_control_v1",
                writer_control,
            ),
        }
    )
    result = _ledger(ports).commit_recovery_handoff_complete(
        plan=plan,
        domain="RECOVERY",
        operation_identity_sha256=operation_identity,
        raw_owner_nonce=raw_nonce,
    )
    _transaction(
        result,
        expected_records=(action_after, recovery_after, writer_control),
        label="recovery handoff completion",
    )
    return True


__all__ = [
    "SUPPORTED_OPERATIONS",
    "Task12RecoveryHandoffError",
    "build_recovery_handoff_record",
    "execute_live_request",
    "materialize_live_request",
    "persist_live_successors",
    "validate_normal_task11_handoff",
]
