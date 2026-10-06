"""Exact-version Task 11 source-publisher Lambda entrypoint."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .canonical import canonical_sha256
from .decision_closure import (
    AUTHORITY_AUDIT_KINDS,
    SourcePublication,
    build_audit_evidence,
    build_source_publication,
)
from .s3_adapter import S3PublicationServices
from .source_publishers import (
    GpuSpendSnapshotPublicationRequest,
    GpuSpendSnapshotPublisherRuntime,
    ProductionControllerBaselinePublicationRequest,
    ProductionControllerBaselinePublisherRuntime,
    ProductionControlPlaneReadinessPublicationRequest,
    ProductionControlPlaneReadinessPublisherRuntime,
    ProductionSubmissionAcquisitionPublicationRequest,
    ProductionSubmissionAcquisitionPublisherRuntime,
    ProductionSubmissionIntentPublicationRequest,
    ProductionSubmissionIntentPublisherRuntime,
    SourcePublicationResult,
    publish_gpu_spend_snapshot,
    publish_production_control_plane_readiness,
    publish_production_controller_baseline,
    publish_production_submission_acquisition,
    publish_production_submission_intent,
)
from .task11_support_boundary import (
    exact_input_coordinate_from_mapping,
    load_exact_input,
    materialize_source_input,
    source_input_from_mapping,
    source_publication_from_payload,
    source_publication_to_payload,
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_NAMES = {
    "SourceGpuSpend": "GPU_SPEND",
    "SourceSubmissionIntent": "SUBMISSION_INTENT",
    "SourceControllerBaseline": "CONTROLLER_BASELINE",
    "SourceControlPlaneReadiness": "CONTROL_PLANE_READINESS",
    "SourceSubmissionAcquisition": "SUBMISSION_ACQUISITION",
}
_SOURCE_KINDS = tuple(_SOURCE_NAMES.values())
_INPUT_KINDS = tuple(kind + "_SOURCE_REQUEST" for kind in _SOURCE_KINDS)


@dataclass(frozen=True)
class SourcePublisherHandlerServices:
    s3: object
    publication: S3PublicationServices


def _configuration() -> Mapping[str, str]:
    values = {
        "account_id": os.environ.get("GLM52_ACCOUNT_ID", ""),
        "region": os.environ.get("AWS_REGION", ""),
        "run_id": os.environ.get("GLM52_RUN_ID", ""),
        "activation_id": os.environ.get("GLM52_ACTIVATION_ID", ""),
        "campaign_bucket": os.environ.get("GLM52_CAMPAIGN_BUCKET", ""),
        "closure_role_arn": os.environ.get("GLM52_CLOSURE_ROLE_ARN", ""),
        "publisher_role_arn": os.environ.get("GLM52_PUBLISHER_ROLE_ARN", ""),
        "source_name": os.environ.get("GLM52_SOURCE_KIND", ""),
        "task11_action_key": os.environ.get(
            "GLM52_TASK11_ACTION_KEY", ""
        ),
        "task11_input_key": os.environ.get(
            "GLM52_TASK11_INPUT_KEY", ""
        ),
        "task11_output_key": os.environ.get(
            "GLM52_TASK11_OUTPUT_KEY", ""
        ),
        "deployment_identity_sha256": os.environ.get(
            "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256",
            "",
        ),
    }
    if (
        values["account_id"] != "246813579024"
        or values["region"] != "us-west-2"
        or values["run_id"] != "glm52-sky-20260724"
        or values["source_name"] not in _SOURCE_NAMES
        or not values["task11_action_key"]
        or not values["task11_input_key"]
        or not values["task11_output_key"]
        or not values["activation_id"]
        or not values["campaign_bucket"]
        or not values["closure_role_arn"].startswith(
            "arn:aws:iam::246813579024:role/"
        )
        or not values["publisher_role_arn"].startswith(
            "arn:aws:iam::246813579024:role/"
        )
        or _SHA.fullmatch(values["deployment_identity_sha256"]) is None
    ):
        raise RuntimeError("source publisher coordinates are incomplete")
    return values


def _production_services(
    *,
    source_name: str,
    generation: int,
) -> SourcePublisherHandlerServices:
    from .task11_production import build_source_publisher_services

    return build_source_publisher_services(
        source_name=source_name,
        generation=generation,
    )


def _prior_publications(
    values: object,
) -> tuple[SourcePublication, ...]:
    if type(values) is not list:
        raise ValueError("prior source publication set is not a list")
    result = []
    predecessor = None
    for index, value in enumerate(values):
        if index == 0:
            publication = value.get("publication") if type(value) is dict else None
            predecessor = (
                publication.get("predecessor_version_id")
                if type(publication) is dict
                else None
            )
        if type(predecessor) is not str or not predecessor:
            raise ValueError("prior source predecessor is absent")
        parsed = source_publication_from_payload(
            value,
            expected_source_kind=_SOURCE_KINDS[index],
            expected_predecessor_version_id=predecessor,
        )
        result.append(parsed)
        predecessor = parsed.version_id
    return tuple(result)


def _publish(
    *,
    source_kind: str,
    config: Mapping[str, str],
    services: SourcePublisherHandlerServices,
    document: object,
    prior: tuple[SourcePublication, ...],
) -> tuple[SourcePublicationResult, bytes]:
    raw, artifacts = materialize_source_input(
        document,
        prior_publications=prior,
    )
    common = {
        "caller_arn": config["closure_role_arn"],
        "activation_id": document.activation_id,
        "generation": document.generation,
        "action_key": document.action_key,
        "raw": raw,
    }
    runtime_common = {
        "services": services.publication,
        "closure_caller_arn": config["closure_role_arn"],
        "publisher_role_arn": config["publisher_role_arn"],
        "campaign_bucket": config["campaign_bucket"],
    }
    if source_kind == "GPU_SPEND":
        result = publish_gpu_spend_snapshot(
            runtime=GpuSpendSnapshotPublisherRuntime(**runtime_common),
            request=GpuSpendSnapshotPublicationRequest(**common),
        )
    elif source_kind == "SUBMISSION_INTENT":
        result = publish_production_submission_intent(
            runtime=ProductionSubmissionIntentPublisherRuntime(
                **runtime_common
            ),
            request=ProductionSubmissionIntentPublicationRequest(**common),
        )
    elif source_kind == "CONTROLLER_BASELINE":
        result = publish_production_controller_baseline(
            runtime=ProductionControllerBaselinePublisherRuntime(
                **runtime_common
            ),
            request=ProductionControllerBaselinePublicationRequest(
                **common,
                descriptor=artifacts["descriptor"],
                intent=artifacts["intent"],
            ),
        )
    elif source_kind == "CONTROL_PLANE_READINESS":
        result = publish_production_control_plane_readiness(
            runtime=ProductionControlPlaneReadinessPublisherRuntime(
                **runtime_common
            ),
            request=ProductionControlPlaneReadinessPublicationRequest(
                **common,
                descriptor=artifacts["descriptor"],
                intent=artifacts["intent"],
                controller_baseline=artifacts["controller_baseline"],
            ),
        )
    elif source_kind == "SUBMISSION_ACQUISITION":
        record = document.source_template
        now = record.get("acquired_at")
        result = publish_production_submission_acquisition(
            runtime=ProductionSubmissionAcquisitionPublisherRuntime(
                **runtime_common
            ),
            request=ProductionSubmissionAcquisitionPublicationRequest(
                **common,
                descriptor=artifacts["descriptor"],
                intent=artifacts["intent"],
                controller_baseline=artifacts["controller_baseline"],
                must_start_control_plane_ready=artifacts[
                    "must_start_control_plane_ready"
                ],
                now=now,
            ),
        )
    else:  # pragma: no cover - configuration validation owns this
        raise RuntimeError("source publisher kind is not registered")
    return result, raw


def main(
    event: object,
    context: object,
    *,
    services: SourcePublisherHandlerServices | None = None,
    config: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Load one immutable source input and perform one accepted Task 5 call."""

    del context
    if (
        type(event) is not dict
        or set(event) != {
            "schema_version",
            "record_type",
            "activation_id",
            "generation",
            "source_kind",
            "predecessor_version_id",
            "input_coordinate",
            "prior_publications",
            "custody_nonce_sha256",
        }
        or event["schema_version"] != 1
        or event["record_type"]
        != "glm52_task11_source_publisher_request_v1"
        or type(event["activation_id"]) is not str
        or not event["activation_id"]
        or type(event["generation"]) is not int
        or event["generation"] <= 0
        or event["source_kind"] not in _SOURCE_KINDS
        or type(event["predecessor_version_id"]) is not str
        or not event["predecessor_version_id"]
        or _SHA.fullmatch(event["custody_nonce_sha256"]) is None
    ):
        raise ValueError("source publisher request is not closed")
    exact_config = dict(_configuration() if config is None else config)
    source_kind = _SOURCE_NAMES.get(exact_config.get("source_name"))
    index = _SOURCE_KINDS.index(event["source_kind"])
    if (
        source_kind != event["source_kind"]
        or event["activation_id"] != exact_config.get("activation_id")
        or len(event["prior_publications"]) != index
    ):
        raise ValueError("source publisher invocation target drifted")
    coordinate = exact_input_coordinate_from_mapping(
        event["input_coordinate"],
        expected_kind=_INPUT_KINDS[index],
    )
    if (
        exact_config.get("task11_input_key") is not None
        and coordinate.key != exact_config["task11_input_key"]
    ):
        raise ValueError("source publisher input key drifted")
    if services is None:
        services = _production_services(
            source_name=exact_config["source_name"],
            generation=event["generation"],
        )
    if type(services) is not SourcePublisherHandlerServices:
        raise TypeError("source publisher services must be exact and typed")
    immutable = load_exact_input(s3=services.s3, coordinate=coordinate)
    document = source_input_from_mapping(
        immutable,
        source_kind=source_kind,
        activation_id=event["activation_id"],
        generation=event["generation"],
    )
    if (
        exact_config.get("task11_action_key") is not None
        and document.action_key != exact_config["task11_action_key"]
    ):
        raise ValueError("source publisher action key drifted")
    prior = _prior_publications(event["prior_publications"])
    if (
        index > 0
        and event["predecessor_version_id"] != prior[-1].version_id
    ):
        raise ValueError("source publisher predecessor VersionId drifted")
    result, raw = _publish(
        source_kind=source_kind,
        config=exact_config,
        services=services,
        document=document,
        prior=prior,
    )
    if (
        type(result) is not SourcePublicationResult
        or result.direct_response_authenticated is not True
        or result.direct_request_id is None
        or result.direct_server_date is None
        or (
            exact_config.get("task11_output_key") is not None
            and result.object_identity.key
            != exact_config["task11_output_key"]
        )
    ):
        raise ValueError("source publisher lost direct response custody")
    audit = build_audit_evidence(
        kind=AUTHORITY_AUDIT_KINDS[index],
        audit_identity_sha256=result.authority_audit_body_sha256,
        invocation_identity_sha256=canonical_sha256(
            {
                "source_kind": source_kind,
                "candidate_identity_sha256": (
                    result.candidate_identity_sha256
                ),
                "direct_request_id": result.direct_request_id,
                "direct_server_date": result.direct_server_date,
                "version_id": result.object_identity.version_id,
            }
        ),
        closing_revision=result.closing_revision,
    )
    publication = build_source_publication(
        source_kind=source_kind,
        predecessor_version_id=event["predecessor_version_id"],
        bucket=result.object_identity.bucket,
        key=result.object_identity.key,
        version_id=result.object_identity.version_id,
        file_sha256=result.object_identity.file_sha256,
        body_sha256=result.object_identity.body_sha256,
        etag=result.object_identity.etag,
        checksum_sha256_base64=(
            result.object_identity.checksum_sha256_base64
        ),
        direct_request_id=result.direct_request_id,
        direct_server_date=result.direct_server_date,
        direct_response_authenticated=True,
        audit=audit,
    )
    return source_publication_to_payload(publication, raw=raw)


__all__ = ["SourcePublisherHandlerServices", "main"]
