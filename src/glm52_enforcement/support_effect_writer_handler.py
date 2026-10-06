"""Exact-version Task 11 immutable-effect writer Lambda entrypoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import hashlib
import json
import os
import re
from typing import Callable, Mapping, Optional

from .canonical import canonical_sha256
from .s3_adapter import S3PublicationServices
from .s3_records import S3ObjectIdentity
from .h1f_adapter import H1fAuditResult, validate_h1f_audit_result
from .task11_effect_writers import (
    EffectWriteResult,
    EffectWriterRequest,
    WRITER_KINDS,
    write_closure_handoff,
    write_fence_successor,
    write_generation_claim,
    write_start_decision,
    write_terminal_v1,
)
from .task11_support_boundary import (
    exact_input_coordinate_from_mapping,
    load_exact_input,
    source_publication_from_payload,
)


_SHA = re.compile(r"^[0-9a-f]{64}$")
_INPUT_KINDS = {
    "FenceSuccessor": "BATCH_SUCCESSOR_REQUEST",
    "ClaimWriter": "CLAIM_CREATE_REQUEST",
    "DecisionWriter": "DECISION_CREATE_REQUEST",
    "TerminalV1Writer": "TERMINAL_V1_WRITE_REQUEST",
    "ClosureHandoff": "CORRELATION_HANDOFF_REQUEST",
}
_FUNCTION_NAMES = {
    "FenceSuccessor": "keep-glm52-h1g-fence-successor",
    "ClaimWriter": "keep-glm52-h1g-claim-writer",
    "DecisionWriter": "keep-glm52-h1g-decision-writer",
    "TerminalV1Writer": "keep-glm52-h1g-terminal-v1-writer",
    "ClosureHandoff": "keep-glm52-h1g-closure-handoff",
}
_WRITERS: Mapping[
    str,
    Callable[
        ...,
        EffectWriteResult,
    ],
] = {
    "FenceSuccessor": write_fence_successor,
    "ClaimWriter": write_generation_claim,
    "DecisionWriter": write_start_decision,
    "TerminalV1Writer": write_terminal_v1,
    "ClosureHandoff": write_closure_handoff,
}
_SOURCE_KINDS = (
    "GPU_SPEND",
    "SUBMISSION_INTENT",
    "CONTROLLER_BASELINE",
    "CONTROL_PLANE_READINESS",
    "SUBMISSION_ACQUISITION",
)
_SOURCE_BINDING_NAMES = {
    "GPU_SPEND": "gpu_spend_snapshot",
    "SUBMISSION_INTENT": "intent",
    "CONTROLLER_BASELINE": "controller_baseline",
    "CONTROL_PLANE_READINESS": "must_start_control_plane_ready",
    "SUBMISSION_ACQUISITION": "submission_acquisition",
}
_CLOSURE_HANDOFF_RUNTIME_BINDINGS = {
    "run_id",
    "campaign_identity_sha256",
    "generation",
    "generation_text",
    "submit_attempt_id",
    "decision_key",
    "decision_version_id",
    "decision_file_sha256",
    "decision_body_sha256",
    "sky_post_action_key",
    "sky_post_consumed_at",
    "sky_post_outcome_class",
    "expected_sky_job_name",
    "task_yaml_sha256",
    "request_body_sha256",
    "api_server_identity_sha256",
    "sky_request_id",
    "post_started_at",
    "post_completed_or_lost_at",
    "binding_state",
}


@dataclass(frozen=True)
class EffectWriterHandlerServices:
    s3: object
    publication: S3PublicationServices


def _configuration() -> Mapping[str, str]:
    values = {
        "account_id": os.environ.get("GLM52_ACCOUNT_ID", ""),
        "region": os.environ.get("AWS_REGION", ""),
        "run_id": os.environ.get("GLM52_RUN_ID", ""),
        "activation_id": os.environ.get("GLM52_ACTIVATION_ID", ""),
        "campaign_bucket": os.environ.get("GLM52_CAMPAIGN_BUCKET", ""),
        "writer_kind": os.environ.get("GLM52_EFFECT_WRITER_KIND", ""),
        "task11_action_key": os.environ.get(
            "GLM52_TASK11_ACTION_KEY", ""
        ),
        "task11_input_key": os.environ.get(
            "GLM52_TASK11_INPUT_KEY", ""
        ),
        "task11_output_key": os.environ.get(
            "GLM52_TASK11_OUTPUT_KEY", ""
        ),
        "function_name": os.environ.get("AWS_LAMBDA_FUNCTION_NAME", ""),
        "function_version": os.environ.get(
            "AWS_LAMBDA_FUNCTION_VERSION",
            "",
        ),
        "deployment_identity_sha256": os.environ.get(
            "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256",
            "",
        ),
    }
    writer_kind = values["writer_kind"]
    if (
        values["account_id"] != "246813579024"
        or values["region"] != "us-west-2"
        or values["run_id"] != "glm52-sky-20260724"
        or writer_kind not in WRITER_KINDS
        or not values["task11_action_key"]
        or not values["task11_input_key"]
        or not values["task11_output_key"]
        or values["function_name"] != _FUNCTION_NAMES[writer_kind]
        or not values["function_version"].isdigit()
        or values["function_version"].startswith("0")
        or not values["activation_id"]
        or not values["campaign_bucket"]
        or _SHA.fullmatch(values["deployment_identity_sha256"]) is None
    ):
        raise RuntimeError("effect writer coordinates are incomplete")
    return values


def _validate_input(
    value: object,
    *,
    writer_kind: str,
    activation_id: str,
    generation: int,
) -> Mapping[str, object]:
    expected = {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "generation",
        "writer_kind",
        "action_key",
        "campaign_bucket",
        "builder_arguments",
        "canonical_identity_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("effect writer input field set drifted")
    body = dict(value)
    identity = body.pop("canonical_identity_sha256")
    if (
        value["schema_version"] != 1
        or value["record_type"] != "glm52_task11_effect_writer_input_v1"
        or value["account_id"] != "246813579024"
        or value["region"] != "us-west-2"
        or value["run_id"] != "glm52-sky-20260724"
        or value["activation_id"] != activation_id
        or value["generation"] != generation
        or value["writer_kind"] != writer_kind
        or type(value["action_key"]) is not str
        or "S3_CREATE" not in value["action_key"]
        or type(value["campaign_bucket"]) is not str
        or not value["campaign_bucket"]
        or type(value["builder_arguments"]) is not dict
        or identity != canonical_sha256(body)
    ):
        raise ValueError("effect writer input identity drifted")
    return value


def _materialize(
    value: object,
    *,
    bindings: Mapping[str, object],
) -> object:
    if type(value) is str and value.startswith("$effect."):
        if value not in bindings:
            raise ValueError("effect writer dependency placeholder is absent")
        return bindings[value]
    if type(value) is list:
        return [_materialize(item, bindings=bindings) for item in value]
    if type(value) is dict:
        return {
            key: _materialize(item, bindings=bindings)
            for key, item in value.items()
        }
    return value


def _source_dependency_bindings(values: object) -> Mapping[str, object]:
    if type(values) is not list or len(values) > len(_SOURCE_KINDS):
        raise ValueError("source dependency set is not a bounded list")
    bindings = {}
    for index, value in enumerate(values):
        kind = _SOURCE_KINDS[index]
        predecessor = (
            value.get("publication", {}).get("predecessor_version_id")
            if type(value) is dict
            and type(value.get("publication")) is dict
            else None
        )
        if type(predecessor) is not str or not predecessor:
            raise ValueError("source dependency predecessor is absent")
        publication = source_publication_from_payload(
            value,
            expected_source_kind=kind,
            expected_predecessor_version_id=predecessor,
        )
        try:
            record = json.loads(
                getattr(publication, "_task11_raw")[:-1].decode("ascii")
            )
        except (AttributeError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("source dependency bytes drifted") from exc
        bindings[
            "$source." + _SOURCE_BINDING_NAMES[kind]
        ] = {
            "key": publication.key,
            "version_id": publication.version_id,
            "record": record,
        }
    return bindings


def _runtime_dependency_bindings(
    values: object,
    *,
    writer_kind: object,
) -> Mapping[str, object]:
    expected = (
        _CLOSURE_HANDOFF_RUNTIME_BINDINGS
        if writer_kind == "ClosureHandoff"
        else set()
    )
    if type(values) is not dict or set(values) != expected:
        raise ValueError("runtime dependency set is not closed")
    bindings: dict[str, object] = {}
    for name, value in values.items():
        if name == "generation" and (
            type(value) is not int or value <= 0
        ):
            raise ValueError("runtime generation binding drifted")
        bindings["$runtime." + name] = value
    return bindings


def _dependency_bindings(values: object) -> Mapping[str, object]:
    if type(values) is not list:
        raise ValueError("effect writer dependency set is not a list")
    bindings = {}
    for value in values:
        kind = (
            value.get("result", {}).get("writer_kind")
            if type(value) is dict
            and type(value.get("result")) is dict
            else None
        )
        if kind not in WRITER_KINDS:
            raise ValueError("effect writer dependency kind drifted")
        parsed = effect_write_result_from_payload(
            value,
            expected_writer_kind=kind,
        )
        try:
            record = json.loads(parsed.raw[:-1].decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("effect writer dependency bytes drifted") from exc
        bindings["$effect." + kind] = {
            "key": parsed.object_identity.key,
            "version_id": parsed.object_identity.version_id,
            "record": record,
        }
    if len(bindings) != len(values):
        raise ValueError("effect writer dependency was duplicated")
    return bindings


def materialize_effect_builder_arguments(
    document: Mapping[str, object],
    *,
    source_publications: object,
    dependency_results: object,
    runtime_bindings: object,
) -> Mapping[str, object]:
    """Resolve only authenticated callee results and closed runtime facts."""

    writer_kind = document.get("writer_kind")
    source = _source_dependency_bindings(source_publications)
    dependencies = _dependency_bindings(dependency_results)
    runtime = _runtime_dependency_bindings(
        runtime_bindings,
        writer_kind=writer_kind,
    )
    if writer_kind == "ClosureHandoff":
        return {
            name.removeprefix("$runtime."): value
            for name, value in runtime.items()
        }
    bindings = {
        **source,
        **dependencies,
        **runtime,
    }
    return _materialize(
        document["builder_arguments"],
        bindings=bindings,
    )


def _prepared_authority(
    value: object,
    *,
    writer_kind: str,
    actions: object,
) -> H1fAuditResult | None:
    if value is None:
        return None
    if (
        writer_kind != "ClosureHandoff"
        or type(value) is not dict
        or set(value) != {"raw_owner_nonce_base64", "audit"}
        or type(value["raw_owner_nonce_base64"]) is not str
        or type(value["audit"]) is not dict
    ):
        raise ValueError("prepared writer authority is not closed")
    try:
        raw_nonce = base64.b64decode(
            value["raw_owner_nonce_base64"],
            validate=True,
        )
        audit = validate_h1f_audit_result(
            H1fAuditResult(**value["audit"])
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("prepared writer authority drifted") from exc
    adopt = getattr(actions, "adopt_owner", None)
    if (
        type(raw_nonce) is not bytes
        or len(raw_nonce) < 16
        or not callable(adopt)
    ):
        raise ValueError("prepared writer owner is unavailable")
    adopt(action_key=audit.action_key, raw_owner_nonce=raw_nonce)
    return audit


def effect_write_result_to_payload(
    value: EffectWriteResult,
) -> Mapping[str, object]:
    if type(value) is not EffectWriteResult:
        raise TypeError("effect writer result must be exact and typed")
    body = {
        "writer_kind": value.writer_kind,
        "record_kind": value.record_kind,
        "object_identity": asdict(value.object_identity),
        "candidate_identity_sha256": value.candidate_identity_sha256,
        "authority_audit_body_sha256": (
            value.authority_audit_body_sha256
        ),
        "closing_revision": value.closing_revision,
        "authorized_revision": value.authorized_revision,
        "direct_request_id": value.direct_request_id,
        "direct_server_date": value.direct_server_date,
        "direct_request_started_at": value.direct_request_started_at,
        "direct_response_received_at": (
            value.direct_response_received_at
        ),
        "direct_response_authenticated": (
            value.direct_response_authenticated
        ),
        "raw_base64": base64.b64encode(value.raw).decode("ascii"),
        "canonical_identity_sha256": value.canonical_identity_sha256,
    }
    return {
        "schema_version": 1,
        "record_type": "glm52_task11_effect_writer_response_v1",
        "result": body,
        "response_identity_sha256": canonical_sha256(body),
    }


def effect_write_result_from_payload(
    value: object,
    *,
    expected_writer_kind: str,
) -> EffectWriteResult:
    """Authenticate one exact direct-response result returned by a writer."""

    if (
        type(value) is not dict
        or set(value)
        != {
            "schema_version",
            "record_type",
            "result",
            "response_identity_sha256",
        }
        or value["schema_version"] != 1
        or value["record_type"]
        != "glm52_task11_effect_writer_response_v1"
        or type(value["result"]) is not dict
        or value["response_identity_sha256"]
        != canonical_sha256(value["result"])
    ):
        raise ValueError("effect writer response envelope drifted")
    result = value["result"]
    expected = {
        "writer_kind",
        "record_kind",
        "object_identity",
        "candidate_identity_sha256",
        "authority_audit_body_sha256",
        "closing_revision",
        "authorized_revision",
        "direct_request_id",
        "direct_server_date",
        "direct_request_started_at",
        "direct_response_received_at",
        "direct_response_authenticated",
        "raw_base64",
        "canonical_identity_sha256",
    }
    if (
        set(result) != expected
        or result["writer_kind"] != expected_writer_kind
        or result["direct_response_authenticated"] is not True
        or type(result["object_identity"]) is not dict
    ):
        raise ValueError("effect writer response field set drifted")
    try:
        raw = base64.b64decode(result["raw_base64"], validate=True)
        object_identity = S3ObjectIdentity(
            **{
                **result["object_identity"],
                "metadata": tuple(
                    tuple(item)
                    for item in result["object_identity"]["metadata"]
                ),
            }
        )
        parsed = EffectWriteResult(
            writer_kind=result["writer_kind"],
            record_kind=result["record_kind"],
            object_identity=object_identity,
            candidate_identity_sha256=result[
                "candidate_identity_sha256"
            ],
            authority_audit_body_sha256=result[
                "authority_audit_body_sha256"
            ],
            closing_revision=result["closing_revision"],
            authorized_revision=result["authorized_revision"],
            direct_request_id=result["direct_request_id"],
            direct_server_date=result["direct_server_date"],
            direct_request_started_at=result[
                "direct_request_started_at"
            ],
            direct_response_received_at=result[
                "direct_response_received_at"
            ],
            direct_response_authenticated=True,
            raw=raw,
            canonical_identity_sha256=result[
                "canonical_identity_sha256"
            ],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("effect writer response is malformed") from exc
    body = {
        "writer_kind": parsed.writer_kind,
        "record_kind": parsed.record_kind,
        "object_identity_sha256": (
            parsed.object_identity.canonical_identity_sha256
        ),
        "candidate_identity_sha256": parsed.candidate_identity_sha256,
        "authority_audit_body_sha256": (
            parsed.authority_audit_body_sha256
        ),
        "closing_revision": parsed.closing_revision,
        "authorized_revision": parsed.authorized_revision,
        "direct_request_id": parsed.direct_request_id,
        "direct_server_date": parsed.direct_server_date,
        "direct_request_started_at": parsed.direct_request_started_at,
        "direct_response_received_at": (
            parsed.direct_response_received_at
        ),
    }
    if (
        parsed.canonical_identity_sha256 != canonical_sha256(body)
        or not raw.endswith(b"\n")
        or parsed.object_identity.file_sha256
        != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("effect writer result identity drifted")
    return parsed


def main(
    event: object,
    context: object,
    *,
    services: Optional[EffectWriterHandlerServices] = None,
    config: Optional[Mapping[str, str]] = None,
) -> Mapping[str, object]:
    """Load one pinned input and execute exactly one fixed writer."""

    del context
    if (
        type(event) is not dict
        or set(event)
        != {
            "schema_version",
            "record_type",
            "activation_id",
            "generation",
            "writer_kind",
            "input_coordinate",
            "source_publications",
            "dependency_results",
            "runtime_bindings",
            "prepared_authority",
            "custody_nonce_sha256",
        }
        or event["schema_version"] != 1
        or event["record_type"]
        != "glm52_task11_effect_writer_request_v1"
        or type(event["activation_id"]) is not str
        or not event["activation_id"]
        or type(event["generation"]) is not int
        or event["generation"] <= 0
        or event["writer_kind"] not in WRITER_KINDS
        or _SHA.fullmatch(event["custody_nonce_sha256"]) is None
    ):
        raise ValueError("effect writer request is not closed")
    exact_config = dict(_configuration() if config is None else config)
    writer_kind = event["writer_kind"]
    if (
        exact_config.get("writer_kind") != writer_kind
        or exact_config.get("activation_id") != event["activation_id"]
    ):
        raise ValueError("effect writer invocation target drifted")
    coordinate = exact_input_coordinate_from_mapping(
        event["input_coordinate"],
        expected_kind=_INPUT_KINDS[writer_kind],
    )
    if (
        exact_config.get("task11_input_key") is not None
        and coordinate.key != exact_config["task11_input_key"]
    ):
        raise ValueError("effect writer input key drifted")
    if services is None:
        from .task11_production import build_effect_writer_services

        services = build_effect_writer_services(
            writer_kind=writer_kind,
            generation=event["generation"],
        )
    if type(services) is not EffectWriterHandlerServices:
        raise TypeError("effect writer services must be exact and typed")
    immutable = load_exact_input(s3=services.s3, coordinate=coordinate)
    document = _validate_input(
        immutable,
        writer_kind=writer_kind,
        activation_id=event["activation_id"],
        generation=event["generation"],
    )
    if document["campaign_bucket"] != exact_config.get("campaign_bucket"):
        raise ValueError("effect writer campaign bucket drifted")
    if (
        exact_config.get("task11_action_key") is not None
        and document["action_key"] != exact_config["task11_action_key"]
    ):
        raise ValueError("effect writer action key drifted")
    result = _WRITERS[writer_kind](
        services=services.publication,
        request=EffectWriterRequest(
            writer_kind=writer_kind,
            activation_id=document["activation_id"],
            generation=document["generation"],
            action_key=document["action_key"],
            campaign_bucket=document["campaign_bucket"],
            builder_arguments=materialize_effect_builder_arguments(
                document,
                source_publications=event["source_publications"],
                dependency_results=event["dependency_results"],
                runtime_bindings=event["runtime_bindings"],
            ),
            preauthorized_audit=_prepared_authority(
                event["prepared_authority"],
                writer_kind=writer_kind,
                actions=services.publication.actions,
            ),
        ),
    )
    if (
        type(result) is not EffectWriteResult
        or result.direct_response_authenticated is not True
        or (
            exact_config.get("task11_output_key") is not None
            and result.object_identity.key
            != exact_config["task11_output_key"]
        )
    ):
        raise ValueError("effect writer lost direct response custody")
    return effect_write_result_to_payload(result)


__all__ = [
    "EffectWriterHandlerServices",
    "effect_write_result_from_payload",
    "effect_write_result_to_payload",
    "materialize_effect_builder_arguments",
    "main",
]
