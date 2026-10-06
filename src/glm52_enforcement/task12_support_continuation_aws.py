"""AWS materializers and effect boundaries for the support continuation.

This adapter intentionally exposes only the operations named by the support
workflow.  It does not expose raw POST, launch, arbitrary SSM, or arbitrary
EC2 mutation surfaces to the coordinator.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from types import SimpleNamespace

from .canonical import canonical_json_bytes, canonical_sha256
from .dynamodb import decode_item, encode_item
from .records import ledger_pk, ledger_sk, validate_record
from .task12_correlation import (
    CorrelatedRequest,
    RequestCorrelation,
    Task11HandoffAuthority,
    build_task11_handoff_authority,
)
from .task12_support_continuation_runtime import (
    SupportContinuationState,
    SupportEffect,
    ZeroRequestProof,
)

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_LAMBDA_VERSION_ARN = re.compile(
    r"arn:aws:lambda:us-west-2:246813579024:"
    r"function:[A-Za-z0-9_-]+:([1-9][0-9]*)\Z"
)
_INSTANCE_ID = re.compile(r"i-[0-9a-f]{17}\Z")
_REQUEST_TERMINAL = frozenset({"FAILED", "CANCELLED"})
_JOB_TERMINAL = frozenset({"FAILED", "CANCELLED", "SUCCEEDED"})
_WORKER_ACTIVE = frozenset({"pending", "running", "stopping", "stopped"})
_WORKER_TERMINAL = frozenset({"shutting-down", "terminated"})


@dataclass(frozen=True)
class AwsSupportContinuationConfig:
    account_id: str
    region: str
    run_id: str
    activation_id: str
    ledger_table_name: str
    campaign_bucket: str
    numeric_binding_version_arn: str
    retained_cancellation_version_arn: str
    worker_drain_version_arn: str
    task9_liability_watcher_version_arn: str
    terminal_v2_version_arn: str
    combined_host_instance_id: str
    quiescence_seconds: int


@dataclass(frozen=True)
class AwsSupportContinuationClients:
    s3: object
    dynamodb: object
    lambda_client: object
    ec2: object
    ssm: object
    events: object
    stepfunctions: object


def _configuration(
    environment: Mapping[str, str] | None = None,
) -> AwsSupportContinuationConfig:
    env = os.environ if environment is None else environment
    values = {
        "account_id": env.get("GLM52_ACCOUNT_ID", ""),
        "region": env.get("AWS_REGION", ""),
        "run_id": env.get("GLM52_RUN_ID", ""),
        "activation_id": env.get("GLM52_ACTIVATION_ID", ""),
        "ledger_table_name": env.get("GLM52_LEDGER_TABLE_NAME", ""),
        "campaign_bucket": env.get("GLM52_CAMPAIGN_BUCKET", ""),
        "numeric_binding_version_arn": env.get(
            "GLM52_NUMERIC_BINDING_VERSION_ARN", ""
        ),
        "retained_cancellation_version_arn": env.get(
            "GLM52_RETAINED_CANCELLATION_VERSION_ARN", ""
        ),
        "worker_drain_version_arn": env.get(
            "GLM52_WORKER_DRAIN_VERSION_ARN", ""
        ),
        "task9_liability_watcher_version_arn": env.get(
            "GLM52_TASK9_LIABILITY_WATCHER_VERSION_ARN", ""
        ),
        "terminal_v2_version_arn": env.get(
            "GLM52_TERMINAL_V2_VERSION_ARN", ""
        ),
        "combined_host_instance_id": env.get(
            "GLM52_COMBINED_HOST_INSTANCE_ID", ""
        ),
    }
    try:
        quiet = int(env.get("GLM52_SUPPORT_QUIESCENCE_SECONDS", ""))
    except ValueError as exc:
        raise RuntimeError(
            "support continuation quiet interval is invalid"
        ) from exc
    if (
        values["account_id"] != "246813579024"
        or values["region"] != "us-west-2"
        or values["run_id"] != "glm52-sky-20260724"
        or any(not value for value in values.values())
        or not values["combined_host_instance_id"].startswith("i-")
        or quiet < 1
        or quiet > 60
    ):
        raise RuntimeError(
            "support continuation AWS coordinates are incomplete"
        )
    return AwsSupportContinuationConfig(
        **values,
        quiescence_seconds=quiet,
    )


def _metadata(response: object, label: str) -> Mapping[str, object]:
    metadata = (
        response.get("ResponseMetadata")
        if type(response) is dict
        else None
    )
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or type(metadata.get("RetryAttempts")) is not int
        or metadata["RetryAttempts"] != 0
    ):
        raise ValueError(
            label + " response is not exact zero-retry"
        )
    return metadata


def _effect(kind: str, evidence: Mapping[str, object]) -> SupportEffect:
    body = {
        "effect_kind": kind,
        "outcome": "SUCCEEDED",
        "evidence": dict(evidence),
    }
    return SupportEffect(
        **body,
        canonical_identity_sha256=canonical_sha256(body),
    )


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class AwsSupportContinuationServices:
    """Concrete AWS implementation used by the published Lambda."""

    def __init__(
        self,
        *,
        config: AwsSupportContinuationConfig,
        clients: AwsSupportContinuationClients,
        caller_function_version_arn: str,
        sleeper: object = time.sleep,
        remaining_time_in_millis: object,
    ) -> None:
        self._config = config
        self._clients = clients
        expected_callees = (
            (
                config.numeric_binding_version_arn,
                "keep-glm52-h1g-numeric-binding"
            ),
            (
                config.retained_cancellation_version_arn,
                "keep-glm52-h1g-retained-cancellation"
            ),
            (
                config.terminal_v2_version_arn,
                "keep-glm52-h1g-terminal-v2-writer"
            ),
            (
                config.worker_drain_version_arn,
                "keep-glm52-h1g-worker-drain-signal"
            ),
            (
                config.task9_liability_watcher_version_arn,
                "keep-glm52-h1g-worker-launch-custody"
            ),
        )
        if (
            type(config) is not AwsSupportContinuationConfig
            or config.account_id != "246813579024"
            or config.region != "us-west-2"
            or config.run_id != "glm52-sky-20260724"
            or _INSTANCE_ID.fullmatch(
                config.combined_host_instance_id
            )
            is None
            or config.quiescence_seconds < 1
            or config.quiescence_seconds > 60
            or any(
                re.fullmatch(
                    r"arn:aws:lambda:us-west-2:246813579024:"
                    + r"function:"
                    + re.escape(name)
                    + r":[1-9][0-9]*",
                    arn,
                )
                is None
                for arn, name in expected_callees
            )
        ):
            raise TypeError(
                "support continuation configuration is not exact"
            )
        if (
            type(caller_function_version_arn) is not str
            or re.fullmatch(
                r"arn:aws:lambda:us-west-2:246813579024:"
                r"function:[A-Za-z0-9_-]+:[1-9][0-9]*",
                caller_function_version_arn,
            )
            is None
        ):
            raise TypeError(
                "support continuation caller version is not exact"
            )
        self._caller_function_version_arn = caller_function_version_arn
        if not callable(sleeper):
            raise TypeError("support continuation sleeper is absent")
        self._sleep = sleeper
        if not callable(remaining_time_in_millis):
            raise TypeError("support continuation deadline is absent")
        self._remaining_time_in_millis = remaining_time_in_millis
        self._authority: Task11HandoffAuthority | None = None
        self._task9_coordinate: Mapping[str, object] | None = None
        self._task9_sha256: str | None = None
        self._observation: Mapping[str, object] | None = None

    def _hydrate_context(self, context: object) -> None:
        if type(context) is not dict:
            raise ValueError("Task 11 context is absent")
        request = context.get("closure_request")
        if type(request) is not dict:
            raise ValueError("Task 11 closure request is absent")
        boundary = request.get("task11_boundary")
        inputs = boundary.get("inputs") if type(boundary) is dict else None
        if type(inputs) is not list or len(inputs) <= 13:
            raise ValueError(
                "Task 9 deployed identity coordinate is absent"
            )
        task9 = inputs[13]
        if type(task9) is not dict:
            raise ValueError(
                "Task 9 deployed identity coordinate drifted"
            )
        try:
            coordinate = {
                name: task9[name]
                for name in (
                    "bucket",
                    "key",
                    "version_id",
                    "file_sha256",
                )
            }
        except KeyError as exc:
            raise ValueError(
                "Task 9 deployed identity coordinate drifted"
            ) from exc
        task9_sha256 = task9.get("body_sha256")
        if _SHA.fullmatch(task9_sha256 or "") is None:
            raise ValueError("Task 9 deployed identity hash drifted")
        self._task9_coordinate = coordinate
        self._task9_sha256 = task9_sha256

    def hydrate(self, state: SupportContinuationState) -> None:
        """Rehydrate every durable coordinate for a fresh Lambda process."""

        if type(state) is not SupportContinuationState:
            raise TypeError("support continuation state is not typed")
        self._hydrate_context(state.task11_context)
        self._authority = state.handoff

    def _invoke(self, version_arn: str, payload: object) -> Mapping[str, object]:
        match = (
            _LAMBDA_VERSION_ARN.fullmatch(version_arn)
            if type(version_arn) is str
            else None
        )
        if match is None:
            raise ValueError("support callee ARN is not version-qualified")
        response = self._clients.lambda_client.invoke(
            FunctionName=version_arn,
            InvocationType="RequestResponse",
            Payload=canonical_json_bytes(payload),
        )
        _metadata(response, "support callee")
        if (
            response.get("StatusCode") != 200
            or response.get("ExecutedVersion") != match.group(1)
            or response.get("FunctionError") is not None
        ):
            raise ValueError(
                "support callee response is not exact zero-retry"
            )
        stream = response.get("Payload")
        raw = stream.read() if callable(getattr(stream, "read", None)) else stream
        if type(raw) is str:
            raw = raw.encode("utf-8")
        try:
            value = json.loads(raw)
        except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("support callee response is malformed") from exc
        if type(value) is not dict:
            raise ValueError("support callee response is not an object")
        return value

    def read_approved_task11_handoff(
        self, context: object
    ) -> Task11HandoffAuthority:
        if type(context) is not dict:
            raise ValueError("Task 11 context is absent")
        request = context.get("closure_request")
        if type(request) is not dict:
            raise ValueError("Task 11 closure request is absent")
        generation = request.get("generation")
        if type(generation) is not int or isinstance(generation, bool):
            raise ValueError("Task 11 generation is invalid")
        if request.get("activation_id") != self._config.activation_id:
            raise ValueError("Task 11 activation drifted")
        self._hydrate_context(context)
        key = (
            "campaigns/"
            + self._config.run_id
            + "/submissions/production/generations/"
            + f"{generation:08d}"
            + "/handoff/SKY_POST_HANDOFF.json"
        )
        listed = self._clients.s3.list_object_versions(
            Bucket=self._config.campaign_bucket,
            Prefix=key,
            MaxKeys=2,
            ExpectedBucketOwner=self._config.account_id,
        )
        _metadata(listed, "handoff version listing")
        versions = [
            item
            for item in listed.get("Versions", [])
            if item.get("Key") == key and item.get("IsLatest") is True
        ]
        if (
            listed.get("IsTruncated") is not False
            or len(versions) != 1
            or listed.get("DeleteMarkers")
        ):
            raise ValueError("Task 11 handoff latest version is not exact")
        version_id = versions[0].get("VersionId")
        response = self._clients.s3.get_object(
            Bucket=self._config.campaign_bucket,
            Key=key,
            VersionId=version_id,
            ExpectedBucketOwner=self._config.account_id,
        )
        _metadata(response, "handoff object read")
        stream = response.get("Body")
        raw = stream.read() if callable(getattr(stream, "read", None)) else stream
        try:
            body = json.loads(raw)
        except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Task 11 handoff bytes are malformed") from exc
        authority = build_task11_handoff_authority(
            bucket=self._config.campaign_bucket,
            key=key,
            version_id=version_id,
            file_sha256=hashlib.sha256(raw).hexdigest(),
            body=body,
            approved_task11_workflow_version_arn=context[
                "approved_task11_workflow_version_arn"
            ],
            task11_execution_arn=context["task11_execution_arn"],
        )
        self._authority = authority
        return authority

    def materialize_correlation_tuple(
        self, authority: Task11HandoffAuthority
    ) -> Mapping[str, object]:
        if (
            self._authority is None
            or authority.body_sha256 != self._authority.body_sha256
        ):
            raise ValueError("numeric handoff authority is foreign")
        body = authority.body
        return {
            "activation_id": self._config.activation_id,
            "generation": body["generation"],
            "action_key": body["sky_post_action_key"],
            "task_yaml_sha256": body["task_yaml_sha256"],
            "request_body_sha256": body["request_body_sha256"],
            "expected_sky_job_name": body["expected_sky_job_name"],
            "handoff_identity_sha256": authority.body_sha256,
        }

    def _observe(self, correlation_identity: str) -> Mapping[str, object]:
        if self._task9_coordinate is None or self._task9_sha256 is None:
            raise ValueError("numeric observation lacks Task 9 identity")
        value = self._invoke(
            self._config.numeric_binding_version_arn,
            {
                "schema_version": 1,
                "record_type": (
                    "glm52_task12_runtime_observation_request_v1"
                ),
                "run_id": self._config.run_id,
                "activation_id": self._config.activation_id,
                "correlation_identity_sha256": correlation_identity,
                "task9_deployed_identity_coordinate": (
                    self._task9_coordinate
                ),
                "task9_deployed_identity_sha256": self._task9_sha256,
            },
        )
        body = dict(value)
        identity = body.pop("canonical_identity_sha256", None)
        request_snapshot = value.get("request_snapshot")
        job_snapshot = value.get("job_snapshot")
        controller = value.get("controller_snapshot")
        request_transport = value.get("request_transport")
        job_transport = value.get("job_transport")
        top_fields = {
            "schema_version",
            "record_type",
            "account_id",
            "region",
            "run_id",
            "activation_id",
            "correlation_identity_sha256",
            "task9_deployed_identity_sha256",
            "observed_at",
            "request_snapshot",
            "job_snapshot",
            "controller_snapshot",
            "request_transport",
            "job_transport",
            "canonical_identity_sha256",
        }
        if (
            set(value) != top_fields
            or value.get("schema_version") != 1
            or value.get("record_type")
            != "glm52_task12_runtime_observation_v1"
            or value.get("account_id") != self._config.account_id
            or value.get("region") != self._config.region
            or value.get("run_id") != self._config.run_id
            or value.get("activation_id")
            != self._config.activation_id
            or value.get("correlation_identity_sha256")
            != correlation_identity
            or value.get("task9_deployed_identity_sha256")
            != self._task9_sha256
            or re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
                r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
                str(value.get("observed_at")),
            )
            is None
            or type(request_snapshot) is not dict
            or request_snapshot.get("observed_at")
            != value["observed_at"]
            or request_snapshot.get("pagination_complete") is not True
            or type(request_snapshot.get("requests")) is not list
            or request_snapshot.get("controller_snapshot") != controller
            or type(job_snapshot) is not dict
            or job_snapshot.get("observed_at") != value["observed_at"]
            or job_snapshot.get("pagination_complete") is not True
            or type(job_snapshot.get("jobs")) is not list
            or job_snapshot.get("controller_snapshot") != controller
            or type(controller) is not dict
            or type(controller.get("members")) is not list
            or _SHA.fullmatch(
                str(controller.get("evidence_identity_sha256"))
            )
            is None
            or any(
                type(transport) is not dict
                or set(transport)
                != {
                    "request_id",
                    "response_sha256",
                    "tls_peer_certificate_sha256",
                }
                or type(transport.get("request_id")) is not str
                or not transport["request_id"]
                or _SHA.fullmatch(
                    str(transport.get("tls_peer_certificate_sha256"))
                )
                is None
                for transport in (request_transport, job_transport)
            )
            or request_transport["response_sha256"]
            != canonical_sha256(request_snapshot)
            or job_transport["response_sha256"]
            != canonical_sha256(job_snapshot)
            or identity != canonical_sha256(body)
        ):
            raise ValueError("numeric observation identity drifted")
        self._observation = value
        return value

    def correlate_request(
        self,
        authority: Task11HandoffAuthority,
        correlation_tuple: object,
    ) -> RequestCorrelation:
        observation = self._observe(canonical_sha256(correlation_tuple))
        requests = observation["request_snapshot"]["requests"]
        expected_id = authority.body["sky_request_id"]
        if expected_id is None:
            matches = requests
        else:
            matches = [
                item
                for item in requests
                if item["request_id"] == expected_id
            ]
        if not matches:
            return RequestCorrelation(kind="ZERO", matches=())
        if expected_id is None and len(matches) > 1:
            return RequestCorrelation(
                kind="MULTIPLE",
                matches=tuple(
                    CorrelatedRequest(
                        request_id=item["request_id"],
                        state=item["state"],
                        body=dict(item),
                    )
                    for item in matches
                ),
            )
        if expected_id is None:
            raise ValueError(
                "UUID-less single correlation requires the exhaustive "
                "immutable-tuple adapter"
            )
        if len(matches) != 1:
            raise ValueError("direct request correlation is ambiguous")
        item = matches[0]
        return RequestCorrelation(
            kind="ONE",
            matches=(
                CorrelatedRequest(
                    request_id=item["request_id"],
                    state=item["state"],
                    body=dict(item),
                ),
            ),
        )

    def prove_zero_request(
        self,
        authority: Task11HandoffAuthority,
        correlation: RequestCorrelation,
    ) -> ZeroRequestProof:
        if correlation.kind != "ZERO" or self._observation is None:
            raise ValueError("zero-request proof lacks its first observation")
        first = self._observation
        self._sleep(self._config.quiescence_seconds)
        second = self._observe(
            canonical_sha256(
                {
                    "handoff_identity_sha256": authority.body_sha256,
                    "kind": "ZERO",
                }
            )
        )
        for observation in (first, second):
            if (
                observation["request_snapshot"]["requests"]
                or observation["job_snapshot"]["jobs"]
                or observation["controller_snapshot"]["members"]
            ):
                raise ValueError("zero-request proof observed live work")
        body = {
            "handoff_identity_sha256": authority.body_sha256,
            "pagination_complete": True,
            "first_observed_at": first["observed_at"],
            "second_observed_at": second["observed_at"],
            "request_ids": (),
            "job_ids": (),
            "worker_instance_ids": (),
            "allocation_ids": (),
            "controller_work_ids": (),
        }
        return ZeroRequestProof(
            **body,
            canonical_identity_sha256=canonical_sha256(body),
        )

    def read_numeric_binding(self, request_id: str) -> object:
        if self._observation is None or self._authority is None:
            raise ValueError("numeric binding observation is absent")
        matches = [
            item
            for item in self._observation["job_snapshot"]["jobs"]
            if item["request_id"] == request_id
        ]
        if len(matches) != 1:
            raise ValueError("positive numeric binding is not exact")
        item = matches[0]
        return {
            "request_id": request_id,
            "numeric_job_id": int(item["job_id"]),
            "sky_job_name": self._authority.body[
                "expected_sky_job_name"
            ],
            "state": item["state"],
        }

    def observe_runtime(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        observation = self._observe(
            canonical_sha256(
                {
                    "support_state": state.canonical_identity_sha256,
                    "kind": "RUNTIME_OBSERVATION",
                }
            )
        )
        return _effect(
            "RUNTIME_OBSERVATION",
            {
                "observation_identity_sha256": observation[
                    "canonical_identity_sha256"
                ],
                "observed_at": observation["observed_at"],
            },
        )

    def _transition_phase(
        self,
        *,
        before: str | None,
        after: str,
        state_identity_sha256: str | None = None,
    ) -> SupportEffect:
        key = {
            "PK": {"S": ledger_pk(self._config.run_id)},
            "SK": {
                "S": (
                    "ACTIVATION#"
                    + self._config.activation_id
                    + "#SUPPORT_DRAIN"
                )
            },
        }
        names = {"#phase": "phase"}
        state_identity = (
            canonical_sha256(
                {
                    "activation_id": self._config.activation_id,
                    "phase": after,
                }
            )
            if state_identity_sha256 is None
            else state_identity_sha256
        )
        values = {
            ":after": {"S": after},
            ":identity": {"S": state_identity},
        }
        condition = "attribute_not_exists(#phase)"
        if before is not None:
            condition = "#phase = :before"
            values[":before"] = {"S": before}
        request = {
            "TableName": self._config.ledger_table_name,
            "Key": key,
            "UpdateExpression": (
                "SET #phase = :after, "
                "support_state_identity_sha256 = :identity"
            ),
            "ConditionExpression": condition,
            "ExpressionAttributeNames": names,
            "ExpressionAttributeValues": values,
            "ReturnValues": "ALL_NEW",
        }
        try:
            response = self._clients.dynamodb.update_item(**request)
            metadata = _metadata(response, after + " transition")
        except Exception:
            readback = self._clients.dynamodb.get_item(
                TableName=self._config.ledger_table_name,
                Key=key,
                ConsistentRead=True,
                ReturnConsumedCapacity="NONE",
            )
            metadata = _metadata(readback, after + " readback")
            item = readback.get("Item")
            if (
                type(item) is not dict
                or item.get("phase") != {"S": after}
                or item.get("support_state_identity_sha256")
                != {"S": state_identity}
            ):
                raise
        return _effect(
            after,
            {
                "phase": after,
                "request_id": metadata["RequestId"],
            },
        )

    def arm_drain(self, state: SupportContinuationState) -> SupportEffect:
        return self._transition_phase(
            before=None,
            after="DRAIN_ARMED",
            state_identity_sha256=state.canonical_identity_sha256,
        )

    def request_drain(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        return self._transition_phase(
            before="DRAIN_ARMED",
            after="DRAIN_REQUESTED",
            state_identity_sha256=state.canonical_identity_sha256,
        )

    def _cancel(self, *, path: str, body: Mapping[str, object]) -> SupportEffect:
        if self._task9_coordinate is None or self._task9_sha256 is None:
            raise ValueError("cancellation lacks Task 9 identity")
        raw = canonical_json_bytes(body)
        requested_at = _utc_now()
        result = self._invoke(
            self._config.retained_cancellation_version_arn,
            {
                "schema_version": 1,
                "record_type": (
                    "glm52_task11_retained_cancellation_request_v1"
                ),
                "run_id": self._config.run_id,
                "activation_id": self._config.activation_id,
                "path": path,
                "request_body_base64": base64.b64encode(raw).decode("ascii"),
                "request_body_sha256": hashlib.sha256(raw).hexdigest(),
                "task9_deployed_identity_coordinate": (
                    self._task9_coordinate
                ),
                "task9_deployed_identity_sha256": self._task9_sha256,
            },
        )
        expected_kind = (
            "REQUEST_CANCEL" if path == "/api/cancel" else "JOB_CANCEL"
        )
        return _effect(
            expected_kind,
            {
                "action_key": (
                    "ACTIVATION#"
                    + self._config.activation_id
                    + "#"
                    + expected_kind
                ),
                "requested_at": requested_at,
                "relay_result_identity_sha256": result[
                    "canonical_identity_sha256"
                ],
                "relay_request_id": result["request_id"],
            },
        )

    def reconcile_request_cancel(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        if (
            state.request_cardinality == "MULTIPLE"
            or (
                state.correlation is not None
                and state.correlation.kind == "MULTIPLE"
            )
        ):
            raise ValueError(
                "multiple request correlation is a sealed incident; "
                "no cancellation mutation is authorized"
            )
        results = [
            self._cancel(
                path="/api/cancel",
                body={"request_id": match.request_id},
            )
            for match in state.correlation.matches
        ]
        if len(results) == 1:
            return results[0]
        return _effect(
            "REQUEST_CANCEL",
            {
                "incident": "MULTIPLE_REQUEST_MATCHES",
                "request_ids": [
                    match.request_id
                    for match in state.correlation.matches
                ],
                "cancel_effect_identity_sha256s": [
                    result.canonical_identity_sha256
                    for result in results
                ],
            },
        )

    def reconcile_job_cancel(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        return self._cancel(
            path="/jobs/cancel",
            body={"job_id": state.binding.numeric_job_id},
        )

    def _remaining(self, label: str, *, minimum_millis: int) -> int:
        value = self._remaining_time_in_millis()
        if (
            type(value) is not int
            or isinstance(value, bool)
            or value < minimum_millis
        ):
            raise ValueError(label + " lacks bounded remaining time")
        return value

    @staticmethod
    def _instance_tags(instance: object, label: str) -> dict[str, str]:
        values = instance.get("Tags") if type(instance) is dict else None
        if (
            type(values) is not list
            or any(
                type(item) is not dict
                or set(item) < {"Key", "Value"}
                or type(item["Key"]) is not str
                or type(item["Value"]) is not str
                for item in values
            )
        ):
            raise ValueError(label + " tags are malformed")
        result = {
            str(item["Key"]): str(item["Value"]) for item in values
        }
        if len(result) != len(values):
            raise ValueError(label + " tags are duplicated")
        return result

    def _describe_controller(
        self,
    ) -> tuple[str, str, Mapping[str, object]]:
        response = self._clients.ec2.describe_instances(
            InstanceIds=[self._config.combined_host_instance_id],
            DryRun=False,
        )
        metadata = _metadata(response, "controller DescribeInstances")
        reservations = response.get("Reservations")
        instances = (
            reservations[0].get("Instances")
            if type(reservations) is list and len(reservations) == 1
            and type(reservations[0]) is dict
            else None
        )
        if type(instances) is not list or len(instances) != 1:
            raise ValueError("controller DescribeInstances is not singular")
        instance = instances[0]
        tags = self._instance_tags(instance, "controller")
        state = instance.get("State") if type(instance) is dict else None
        state_name = state.get("Name") if type(state) is dict else None
        if (
            instance.get("InstanceId")
            != self._config.combined_host_instance_id
            or tags.get("RunId") != self._config.run_id
            or tags.get("ActivationId") != self._config.activation_id
            or tags.get("Purpose") != "combined-host"
            or state_name
            not in {
                "pending",
                "running",
                "stopping",
                "stopped",
                "shutting-down",
                "terminated",
            }
        ):
            raise ValueError("controller identity or tags drifted")
        return str(state_name), str(metadata["RequestId"]), instance

    @staticmethod
    def _terminal_sky_observation(
        observation: Mapping[str, object],
    ) -> dict[str, object]:
        request_snapshot = observation.get("request_snapshot")
        job_snapshot = observation.get("job_snapshot")
        controller = observation.get("controller_snapshot")
        requests = (
            request_snapshot.get("requests")
            if type(request_snapshot) is dict
            else None
        )
        jobs = (
            job_snapshot.get("jobs")
            if type(job_snapshot) is dict
            else None
        )
        members = (
            controller.get("members")
            if type(controller) is dict
            else None
        )
        if (
            type(request_snapshot) is not dict
            or request_snapshot.get("pagination_complete") is not True
            or type(job_snapshot) is not dict
            or job_snapshot.get("pagination_complete") is not True
            or type(requests) is not list
            or type(jobs) is not list
            or type(members) is not list
            or members
            or any(
                type(item) is not dict
                or item.get("state") not in _REQUEST_TERMINAL
                for item in requests
            )
            or any(
                type(item) is not dict
                or item.get("state") not in _JOB_TERMINAL
                for item in jobs
            )
        ):
            raise ValueError("terminal request/job state is absent")
        return {
            "request_members": [
                (item["request_id"], item["state"]) for item in requests
            ],
            "job_members": [
                (item["job_id"], item["request_id"], item["state"])
                for item in jobs
            ],
            "controller_members": [],
            "observed_at": observation["observed_at"],
            "identity_sha256": observation[
                "canonical_identity_sha256"
            ],
        }

    def quiesce_controller(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        if (
            state.phase != "DRAIN_REQUESTED"
            or state.request_cardinality == "MULTIPLE"
            or sum(
                item.effect_kind == "DRAIN_REQUESTED"
                for item in state.effects
            )
            != 1
        ):
            raise ValueError(
                "controller drain lacks typed DRAIN_REQUESTED authority"
            )
        minimum = (self._config.quiescence_seconds + 30) * 1000
        remaining_before = self._remaining(
            "controller quiescence",
            minimum_millis=minimum,
        )
        before_state, before_request, before_instance = (
            self._describe_controller()
        )
        authority_body = {
            "activation_id": self._config.activation_id,
            "run_id": self._config.run_id,
            "instance_id": self._config.combined_host_instance_id,
            "support_state_identity_sha256": (
                state.canonical_identity_sha256
            ),
            "drain_requested_identity_sha256": next(
                item.canonical_identity_sha256
                for item in state.effects
                if item.effect_kind == "DRAIN_REQUESTED"
            ),
            "before_instance_sha256": canonical_sha256(before_instance),
            "remaining_time_in_millis": remaining_before,
        }
        request_ids = [before_request]
        first = self._observe(
            canonical_sha256(
                {
                    "authority": canonical_sha256(authority_body),
                    "kind": "CONTROLLER_QUIESCENCE_FIRST",
                }
            )
        )
        first_evidence = self._terminal_sky_observation(first)
        self._sleep(self._config.quiescence_seconds)
        second = self._observe(
            canonical_sha256(
                {
                    "authority": canonical_sha256(authority_body),
                    "kind": "CONTROLLER_QUIESCENCE_SECOND",
                }
            )
        )
        second_evidence = self._terminal_sky_observation(second)
        if (
            first_evidence["observed_at"]
            >= second_evidence["observed_at"]
            or first_evidence["request_members"]
            != second_evidence["request_members"]
            or first_evidence["job_members"]
            != second_evidence["job_members"]
        ):
            raise ValueError("controller quiescence ancestry drifted")
        if before_state in {"pending", "running"}:
            self._remaining(
                "controller stop",
                minimum_millis=minimum,
            )
            response = self._clients.ec2.stop_instances(
                InstanceIds=[self._config.combined_host_instance_id],
                Force=False,
                Hibernate=False,
                DryRun=False,
            )
            request_ids.append(
                str(_metadata(response, "controller stop")["RequestId"])
            )
        for _attempt in range(20):
            self._remaining(
                "controller stop readback",
                minimum_millis=minimum,
            )
            after_state, request_id, _instance = (
                self._describe_controller()
            )
            request_ids.append(request_id)
            if after_state in {"stopped", "terminated"}:
                break
            self._sleep(5)
        else:
            raise ValueError("controller did not stop after bounded polling")
        ssm = self._clients.ssm.describe_instance_information(
            Filters=[
                {
                    "Key": "InstanceIds",
                    "Values": [self._config.combined_host_instance_id],
                }
            ],
            MaxResults=5,
        )
        request_ids.append(
            str(_metadata(ssm, "controller SSM readback")["RequestId"])
        )
        rows = ssm.get("InstanceInformationList")
        if (
            type(rows) is not list
            or any(
                type(item) is not dict
                or item.get("InstanceId")
                != self._config.combined_host_instance_id
                for item in rows
            )
            or any(item.get("PingStatus") == "Online" for item in rows)
        ):
            raise ValueError("controller remains reachable after stop")
        return _effect(
            "CONTROLLER_QUIESCED",
            {
                "instance_id": self._config.combined_host_instance_id,
                "ec2_state": after_state,
                "drain_authority_identity_sha256": canonical_sha256(
                    authority_body
                ),
                "first_observation_identity_sha256": first_evidence[
                    "identity_sha256"
                ],
                "second_observation_identity_sha256": second_evidence[
                    "identity_sha256"
                ],
                "request_ids": request_ids,
            },
        )

    def _discover_workers(
        self,
    ) -> tuple[dict[str, Mapping[str, object]], tuple[str, ...]]:
        token: str | None = None
        seen_tokens: set[str] = set()
        workers: dict[str, Mapping[str, object]] = {}
        request_ids: list[str] = []
        pages = 0
        records = 0
        while True:
            self._remaining(
                "worker discovery",
                minimum_millis=10_000,
            )
            if pages >= 32:
                raise ValueError(
                    "worker discovery pagination ceiling exceeded"
                )
            arguments: dict[str, object] = {
                "Filters": [
                    {
                        "Name": "tag:RunId",
                        "Values": [self._config.run_id],
                    },
                    {
                        "Name": "instance-state-name",
                        "Values": sorted(
                            _WORKER_ACTIVE | _WORKER_TERMINAL
                        ),
                    },
                ],
                "MaxResults": 1000,
                "DryRun": False,
            }
            if token is not None:
                arguments["NextToken"] = token
            response = self._clients.ec2.describe_instances(**arguments)
            pages += 1
            request_ids.append(
                str(_metadata(response, "worker discovery")["RequestId"])
            )
            reservations = response.get("Reservations")
            if type(reservations) is not list:
                raise ValueError("worker discovery page is incomplete")
            for reservation in reservations:
                instances = (
                    reservation.get("Instances")
                    if type(reservation) is dict
                    else None
                )
                if type(instances) is not list:
                    raise ValueError("worker reservation is malformed")
                for instance in instances:
                    records += 1
                    if records > 128:
                        raise ValueError(
                            "worker discovery record ceiling exceeded"
                        )
                    instance_id = (
                        instance.get("InstanceId")
                        if type(instance) is dict
                        else None
                    )
                    state = (
                        instance.get("State")
                        if type(instance) is dict
                        else None
                    )
                    state_name = (
                        state.get("Name")
                        if type(state) is dict
                        else None
                    )
                    tags = self._instance_tags(instance, "worker")
                    if instance_id == self._config.combined_host_instance_id:
                        continue
                    if (
                        type(instance_id) is not str
                        or _INSTANCE_ID.fullmatch(instance_id) is None
                        or instance_id in workers
                        or instance.get("InstanceType") != "p5.48xlarge"
                        or instance.get("InstanceLifecycle") is not None
                        or state_name
                        not in _WORKER_ACTIVE | _WORKER_TERMINAL
                        or tags.get("RunId") != self._config.run_id
                        or tags.get("activation-id")
                        != self._config.activation_id
                        or tags.get("Market") != "on-demand"
                    ):
                        raise ValueError(
                            "worker identity, market, or tags drifted"
                        )
                    workers[instance_id] = {
                        "instance_id": instance_id,
                        "state": state_name,
                        "tags_sha256": canonical_sha256(tags),
                    }
            next_token = response.get("NextToken")
            if next_token is None:
                break
            if (
                type(next_token) is not str
                or not next_token
                or next_token in seen_tokens
            ):
                raise ValueError("worker discovery pagination cycled")
            seen_tokens.add(next_token)
            token = next_token
        return workers, tuple(request_ids)

    def _read_retained_record(
        self,
        record_type: str,
    ) -> tuple[dict[str, object], str]:
        sk = ledger_sk(
            record_type,
            **(
                {"activation_id": self._config.activation_id}
                if record_type != "glm52_production_activation_index"
                else {}
            ),
        )
        self._remaining(
            "worker-drain retained state",
            minimum_millis=10_000,
        )
        response = self._clients.dynamodb.get_item(
            TableName=self._config.ledger_table_name,
            Key=encode_item(
                {
                    "PK": ledger_pk(self._config.run_id),
                    "SK": sk,
                }
            ),
            ConsistentRead=True,
            ReturnConsumedCapacity="NONE",
        )
        request_id = str(
            _metadata(
                response,
                "worker-drain retained state",
            )["RequestId"]
        )
        encoded = response.get("Item")
        if type(encoded) is not dict:
            raise ValueError("worker-drain retained state is absent")
        physical = decode_item(encoded)
        if (
            physical.pop("PK", None) != ledger_pk(self._config.run_id)
            or physical.pop("SK", None) != sk
        ):
            raise ValueError("worker-drain retained state key drifted")
        return validate_record(record_type, physical), request_id

    def _retained_history_operation_input(
        self,
        execution_arn: str,
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        token: str | None = None
        seen_tokens: set[str] = set()
        events: list[Mapping[str, object]] = []
        request_ids: list[str] = []
        pages = 0
        while True:
            self._remaining(
                "retained execution history",
                minimum_millis=10_000,
            )
            if pages >= 32:
                raise ValueError(
                    "retained execution history pagination ceiling exceeded"
                )
            arguments: dict[str, object] = {
                "executionArn": execution_arn,
                "includeExecutionData": True,
                "maxResults": 1000,
                "reverseOrder": False,
            }
            if token is not None:
                arguments["nextToken"] = token
            response = self._clients.stepfunctions.get_execution_history(
                **arguments
            )
            pages += 1
            request_ids.append(
                str(
                    _metadata(
                        response,
                        "retained execution history",
                    )["RequestId"]
                )
            )
            rows = response.get("events")
            if (
                type(rows) is not list
                or any(type(item) is not dict for item in rows)
            ):
                raise ValueError(
                    "retained execution history page is malformed"
                )
            events.extend(rows)
            if len(events) > 12_000:
                raise ValueError(
                    "retained execution history record ceiling exceeded"
                )
            next_token = response.get("nextToken")
            if next_token is None:
                break
            if (
                type(next_token) is not str
                or not next_token
                or next_token in seen_tokens
            ):
                raise ValueError(
                    "retained execution history pagination cycled"
                )
            seen_tokens.add(next_token)
            token = next_token
        event_ids = [item.get("id") for item in events]
        if (
            any(type(item) is not int for item in event_ids)
            or event_ids != sorted(event_ids)
            or len(set(event_ids)) != len(event_ids)
        ):
            raise ValueError("retained execution history order drifted")
        candidates: list[str] = []
        predecessor = (
            "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_"
            "TRANSFER_LIABILITIES"
        )
        for event in events:
            details = event.get("stateExitedEventDetails")
            if (
                event.get("type") == "TaskStateExited"
                and type(details) is dict
                and details.get("name") == predecessor
                and type(details.get("output")) is str
            ):
                candidates.append(str(details["output"]))
        if len(candidates) != 1:
            raise ValueError(
                "worker-drain predecessor history is not singular"
            )
        try:
            operation_input = json.loads(candidates[0])
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "worker-drain predecessor output is malformed"
            ) from exc
        prior = (
            operation_input.get("task12_last_result")
            if type(operation_input) is dict
            else None
        )
        if (
            type(prior) is not dict
            or set(prior)
            != {
                "schema_version",
                "record_type",
                "handler_kind",
                "operation_kind",
                "operation_input_identity_sha256",
                "outcome",
                "result",
                "canonical_body_sha256",
            }
            or prior.get("schema_version") != 1
            or prior.get("record_type")
            != "glm52_task12_lambda_result_v1"
            or prior.get("handler_kind") != "RETAINED_WORKER_DRAIN"
            or prior.get("operation_kind") != predecessor
            or prior.get("outcome") != "SUCCEEDED"
        ):
            raise ValueError(
                "worker-drain predecessor ResultPath is foreign"
            )
        prior_body = dict(prior)
        prior_identity = prior_body.pop(
            "canonical_body_sha256", None
        )
        if (
            _SHA.fullmatch(str(prior_identity)) is None
            or prior_identity != canonical_sha256(prior_body)
        ):
            raise ValueError(
                "worker-drain predecessor ResultPath drifted"
            )
        return dict(operation_input), tuple(request_ids)

    def _retained_worker_drain_envelope(
        self,
        *,
        state: SupportContinuationState,
        instance_id: str,
    ) -> Mapping[str, object]:
        index, _index_request = self._read_retained_record(
            "glm52_production_activation_index"
        )
        recovery, _recovery_request = self._read_retained_record(
            "glm52_production_recovery_control"
        )
        owner_execution = recovery.get("owner_execution_arn")
        owner_version = recovery.get(
            "owner_state_machine_version_arn"
        )
        owner_dispatch = recovery.get(
            "owner_dispatch_identity_sha256"
        )
        if (
            index.get("current_activation_id")
            != self._config.activation_id
            or recovery.get("activation_id")
            != self._config.activation_id
            or recovery.get("activation_ordinal")
            != index.get("current_activation_ordinal")
            or recovery.get("state") != "OWNED"
            or type(owner_execution) is not str
            or re.fullmatch(
                r"arn:aws:states:us-west-2:246813579024:"
                r"execution:keep-glm52-h1g-retainedlifecycle:"
                r"[A-Za-z0-9._:-]+",
                owner_execution,
            )
            is None
            or type(owner_version) is not str
            or re.fullmatch(
                r"arn:aws:states:us-west-2:246813579024:"
                r"stateMachine:keep-glm52-h1g-"
                r"retainedlifecycle:[1-9][0-9]*",
                owner_version,
            )
            is None
            or _SHA.fullmatch(str(owner_dispatch)) is None
        ):
            raise ValueError(
                "authenticated Task 12 worker-drain owner is absent"
            )
        operation_input, _history_requests = (
            self._retained_history_operation_input(owner_execution)
        )
        generation = state.handoff.body["generation"]
        generation_text = f"{generation:08d}"
        return {
            "activation_id": self._config.activation_id,
            "activation_ordinal": index[
                "current_activation_ordinal"
            ],
            "generation": generation,
            "generation_text": generation_text,
            "dispatch_identity_sha256": owner_dispatch,
            "caller_state_machine_arn": owner_version.rsplit(":", 1)[0],
            "state_machine_execution_arn": owner_execution,
            "operation_kind": (
                "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS"
            ),
            "operation_input": operation_input,
        }

    @staticmethod
    def _validate_worker_drain_result(
        value: Mapping[str, object],
        *,
        operation_input: Mapping[str, object],
        instance_id: str,
    ) -> Mapping[str, object]:
        outer_fields = {
            "schema_version",
            "record_type",
            "handler_kind",
            "operation_kind",
            "operation_input_identity_sha256",
            "outcome",
            "result",
            "canonical_body_sha256",
        }
        outer_body = dict(value)
        outer_identity = outer_body.pop(
            "canonical_body_sha256", None
        )
        result = value.get("result")
        if (
            set(value) != outer_fields
            or value.get("schema_version") != 1
            or value.get("record_type")
            != "glm52_task12_lambda_result_v1"
            or value.get("handler_kind") != "RETAINED_WORKER_DRAIN"
            or value.get("operation_kind")
            != "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS"
            or value.get("operation_input_identity_sha256")
            != canonical_sha256(operation_input)
            or value.get("outcome") != "SUCCEEDED"
            or _SHA.fullmatch(str(outer_identity)) is None
            or outer_identity != canonical_sha256(outer_body)
            or type(result) is not dict
        ):
            raise ValueError(
                "published worker-drain Lambda result drifted"
            )
        result_fields = {
            "result_kind",
            "action_kind",
            "behavior_kind",
            "predecessor_operation_kind",
            "predecessor_result_identity_sha256",
            "operation_payload_identity_sha256",
            "domain_result",
            "domain_result_identity_sha256",
            "canonical_body_sha256",
        }
        result_body = dict(result)
        result_identity = result_body.pop(
            "canonical_body_sha256", None
        )
        domain = result.get("domain_result")
        prior = operation_input.get("task12_last_result")
        if (
            set(result) != result_fields
            or result.get("result_kind")
            != (
                "glm52_task12_retained_reconcile_workers_and_"
                "allocations_result_v1"
            )
            or result.get("action_kind")
            != "reconcile_workers_and_allocations"
            or result.get("behavior_kind") != "EXTERNAL_ACTION"
            or result.get("predecessor_operation_kind")
            != (
                "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_"
                "TRANSFER_LIABILITIES"
            )
            or type(prior) is not dict
            or result.get("predecessor_result_identity_sha256")
            != prior.get("canonical_body_sha256")
            or _SHA.fullmatch(
                str(result.get("operation_payload_identity_sha256"))
            )
            is None
            or _SHA.fullmatch(str(result_identity)) is None
            or result_identity != canonical_sha256(result_body)
            or type(domain) is not dict
            or result.get("domain_result_identity_sha256")
            != canonical_sha256(domain)
        ):
            raise ValueError(
                "published worker-drain strict result drifted"
            )
        domain_fields = {
            "authority_identity_sha256",
            "candidate_identity_sha256",
            "instance_id",
            "command_id",
            "request_id",
            "response_identity_sha256",
            "retained_authority",
            "retained_authority_body_sha256",
            "state",
            "drain_complete",
            "liability_settled",
            "dispatch_identity_sha256",
        }
        retained = domain.get("retained_authority")
        dispatch_body = {
            key: domain[key]
            for key in domain_fields
            if key
            not in {
                "retained_authority",
                "dispatch_identity_sha256",
            }
        }
        if (
            set(domain) != domain_fields
            or domain.get("instance_id") != instance_id
            or re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                r"[0-9a-f]{4}-[0-9a-f]{12}",
                str(domain.get("command_id")),
            )
            is None
            or type(domain.get("request_id")) is not str
            or not domain["request_id"]
            or any(
                _SHA.fullmatch(str(domain.get(key))) is None
                for key in (
                    "authority_identity_sha256",
                    "candidate_identity_sha256",
                    "response_identity_sha256",
                    "retained_authority_body_sha256",
                    "dispatch_identity_sha256",
                )
            )
            or domain.get("state") != "SSM_ACCEPTED_NOT_DRAINED"
            or domain.get("drain_complete") is not False
            or domain.get("liability_settled") is not False
            or type(retained) is not dict
            or retained.get("instance_id") != instance_id
            or retained.get("authority_body_sha256")
            != domain.get("retained_authority_body_sha256")
            or domain.get("dispatch_identity_sha256")
            != canonical_sha256(dispatch_body)
        ):
            raise ValueError(
                "published worker-drain domain result drifted"
            )
        return {
            "function_version_arn": None,
            "lambda_result_identity_sha256": outer_identity,
            "strict_result_identity_sha256": result_identity,
            "dispatch_identity_sha256": domain[
                "dispatch_identity_sha256"
            ],
            "instance_id": instance_id,
            "command_id": domain["command_id"],
            "request_id": domain["request_id"],
            "state": domain["state"],
        }

    def _dispatch_worker_drain(
        self,
        *,
        state: SupportContinuationState,
        instance_id: str,
    ) -> Mapping[str, object]:
        envelope = self._retained_worker_drain_envelope(
            state=state,
            instance_id=instance_id,
        )
        value = self._invoke(
            self._config.worker_drain_version_arn,
            envelope,
        )
        result = dict(
            self._validate_worker_drain_result(
                value,
                operation_input=envelope["operation_input"],
                instance_id=instance_id,
            )
        )
        result["function_version_arn"] = (
            self._config.worker_drain_version_arn
        )
        return result

    def _dispatch_liability_watch(
        self,
        *,
        instance_ids: tuple[str, ...],
    ) -> Mapping[str, object]:
        if (
            not instance_ids
            or tuple(sorted(set(instance_ids))) != instance_ids
            or any(
                _INSTANCE_ID.fullmatch(instance_id) is None
                for instance_id in instance_ids
            )
        ):
            raise ValueError(
                "Task 9 guarded liability-watch target set drifted"
            )
        value = self._invoke(
            self._config.task9_liability_watcher_version_arn,
            {
                "mode": "LIABILITY_WATCH",
                "event_accelerator": "owner-failover",
            },
        )
        ids = value.get("instance_ids")
        next_scan_at = value.get("next_scan_at")
        if (
            set(value)
            != {
                "settled",
                "instance_ids",
                "next_scan_at",
                "incident",
            }
            or type(value.get("settled")) is not bool
            or type(ids) is not list
            or ids != list(instance_ids)
            or value.get("incident") is not (len(instance_ids) > 1)
            or type(next_scan_at) is not str
        ):
            raise ValueError(
                "Task 9 guarded liability-watch result drifted"
            )
        try:
            parsed = datetime.fromisoformat(
                next_scan_at.replace("Z", "+00:00")
            )
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                "Task 9 guarded liability-watch result drifted"
            ) from exc
        if (
            parsed.tzinfo is None
            or parsed.astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z")
            != next_scan_at
        ):
            raise ValueError(
                "Task 9 guarded liability-watch result drifted"
            )
        return {
            **dict(value),
            "function_version_arn": (
                self._config.task9_liability_watcher_version_arn
            ),
        }

    def reconcile_workers(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        if state.phase != "CONTROLLER_QUIESCED":
            raise ValueError(
                "worker reconciliation lacks controller quiescence"
            )
        workers, request_ids = self._discover_workers()
        active_ids = tuple(
            sorted(
                instance_id
                for instance_id, instance in workers.items()
                if instance["state"] in _WORKER_ACTIVE
            )
        )
        dispatches: list[Mapping[str, object]] = []
        guarded_watch: Mapping[str, object] | None = None
        if len(active_ids) == 1:
            instance_id = active_ids[0]
            self._remaining(
                "worker drain",
                minimum_millis=180_000,
            )
            dispatches.append(
                self._dispatch_worker_drain(
                    state=state,
                    instance_id=instance_id,
                )
            )
        if active_ids:
            self._remaining(
                "guarded worker termination",
                minimum_millis=60_000,
            )
            guarded_watch = self._dispatch_liability_watch(
                instance_ids=active_ids,
            )
        if active_ids:
            for _attempt in range(20):
                self._remaining(
                    "worker terminal readback",
                    minimum_millis=60_000,
                )
                self._sleep(5)
                after, after_request_ids = self._discover_workers()
                request_ids += after_request_ids
                nonterminal = sorted(
                    instance_id
                    for instance_id, instance in after.items()
                    if instance["state"] in _WORKER_ACTIVE
                )
                if not nonterminal:
                    workers = after
                    break
            else:
                raise ValueError(
                    "guarded worker drain lacks terminal readback"
                )
        return _effect(
            "WORKER_TERMINAL",
            {
                "pagination_complete": True,
                "instances": [
                    workers[instance_id]
                    for instance_id in sorted(workers)
                ],
                "active_instance_ids": [],
                "drain_dispatches": dispatches,
                "guarded_liability_watch": guarded_watch,
                "request_ids": list(request_ids),
                "forced_termination_count": 0,
            },
        )

    def _live_allocation_spend_proof(
        self,
        *,
        state: SupportContinuationState,
        observed_at: str,
    ) -> Mapping[str, object]:
        from .task12_live_drain_effects import _read_terminal_evidence
        from .task12_live_runtime import (
            PROVE_TERMINAL,
            _query_runtime_families,
            _spend_authority,
        )
        from .task12_support_terminal_v2 import (
            AwsSupportTerminalV2Services,
            SupportTerminalV2Clients,
            SupportTerminalV2Config,
        )

        generation = state.handoff.body["generation"]
        generation_text = f"{generation:08d}"
        request = {
            "caller_function_version_arn": (
                self._caller_function_version_arn
            ),
            "caller_state_machine_version_arn": (
                state.handoff.approved_task11_workflow_version_arn
            ),
            "caller_execution_arn": state.task11_context[
                "task11_execution_arn"
            ],
            "generation": generation,
            "generation_text": generation_text,
        }
        terminal_services = AwsSupportTerminalV2Services(
            config=SupportTerminalV2Config(
                account_id=self._config.account_id,
                run_id=self._config.run_id,
                activation_id=self._config.activation_id,
                ledger_table_name=self._config.ledger_table_name,
                campaign_bucket=self._config.campaign_bucket,
            ),
            clients=SupportTerminalV2Clients(
                s3=self._clients.s3,
                dynamodb=self._clients.dynamodb,
                stepfunctions=object(),
            ),
        )
        deployment, deployment_request_ids = (
            terminal_services._deployment(
                request,
                invoked_function_arn=(
                    self._config.terminal_v2_version_arn
                ),
            )
        )
        roles = deployment["role_coordinates"]

        class _Ports:
            def __init__(
                self,
                outer: AwsSupportContinuationServices,
            ) -> None:
                self.deployment = SimpleNamespace(
                    role_coordinates=roles
                )
                self._outer = outer

            def client(self, service: str) -> object:
                if service == "s3":
                    return self._outer._clients.s3
                if service == "dynamodb":
                    return self._outer._clients.dynamodb
                raise ValueError(
                    "allocation proof requested a foreign AWS client"
                )

        ports = _Ports(self)
        invocation = SimpleNamespace(
            activation_id=self._config.activation_id,
            activation_ordinal=deployment["activation_ordinal"],
            generation=generation,
            generation_text=generation_text,
            operation_kind=PROVE_TERMINAL,
        )
        campaign_identity = str(
            state.handoff.body["campaign_identity_sha256"]
        )
        families, family_request_ids = _query_runtime_families(
            ports=ports,
            invocation=invocation,
            campaign_identity_sha256=campaign_identity,
        )
        spend, intervals, _chain, spend_request_ids = _spend_authority(
            ports=ports,
            invocation=invocation,
            observed_at=observed_at,
            campaign_identity_sha256=campaign_identity,
            families=families,
        )
        launches = families["glm52_production_worker_launch"]
        liabilities = families[
            "glm52_production_worker_launch_liability"
        ]
        settlements = families[
            "glm52_production_worker_launch_liability_settlement"
        ]
        post_terminal = families[
            "glm52_production_post_terminal_allocation"
        ]
        if (
            any(
                row["state"]
                not in {
                    "ALLOCATION_CLOSED",
                    "REJECTED_NO_INSTANCE",
                    "ABANDONED_NOT_SENT",
                }
                for row in launches
            )
            or any(
                row["state"]
                not in {
                    "SETTLED_NO_INSTANCE_REJECTED",
                    "SETTLED_INSTANCE_CLOSED",
                }
                for row in liabilities
            )
            or len(settlements) != len(liabilities)
            or any(
                row["state"] != "ALLOCATION_CLOSED"
                for row in post_terminal
            )
            or getattr(spend, "state", None) != "CLOSED"
            or any(
                item.get("state") != "CLOSED" for item in intervals
            )
        ):
            raise ValueError(
                "live allocation or spend state is not closed"
            )
        terminal_evidence = (
            _read_terminal_evidence(
                ports=ports,
                invocation=invocation,
                recovery={
                    "campaign_identity_sha256": campaign_identity,
                },
                families=families,
            )
            if launches or liabilities
            else None
        )
        family_projection = {
            name: list(rows) for name, rows in families.items()
        }
        spend_projection = asdict(spend)
        return {
            "deployment_identity_sha256": deployment[
                "canonical_body_sha256"
            ],
            "family_identity_sha256": canonical_sha256(
                family_projection
            ),
            "family_cardinalities": {
                name: len(rows) for name, rows in families.items()
            },
            "spend_identity_sha256": spend[
                "evidence_identity_sha256"
            ]
            if type(spend) is dict
            else spend.evidence_identity_sha256,
            "spend": spend_projection,
            "intervals_identity_sha256": canonical_sha256(
                list(intervals)
            ),
            "terminal_evidence_identity_sha256": (
                None
                if terminal_evidence is None
                else terminal_evidence["canonical_body_sha256"]
            ),
            "request_ids": [
                *deployment_request_ids,
                *family_request_ids,
                *spend_request_ids,
            ],
        }

    def close_allocations(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        if state.phase != "WORKER_TERMINAL":
            raise ValueError(
                "allocation closure lacks worker terminal predecessor"
            )
        worker_effects = [
            item
            for item in state.effects
            if item.effect_kind == "WORKER_TERMINAL"
        ]
        if len(worker_effects) != 1:
            raise ValueError(
                "allocation closure lacks singular worker evidence"
            )
        self._remaining(
            "allocation terminal observation",
            minimum_millis=120_000,
        )
        first_started = _utc_now()
        first = self._observe(
            canonical_sha256(
                {
                    "support_state": state.canonical_identity_sha256,
                    "kind": "TERMINAL_QUIESCENCE_FIRST",
                }
            )
        )
        first_evidence = self._terminal_sky_observation(first)
        first_completed = str(first["observed_at"])
        self._sleep(self._config.quiescence_seconds)
        second_started = _utc_now()
        second = self._observe(
            canonical_sha256(
                {
                    "support_state": state.canonical_identity_sha256,
                    "kind": "TERMINAL_QUIESCENCE_SECOND",
                }
            )
        )
        second_evidence = self._terminal_sky_observation(second)
        second_completed = str(second["observed_at"])
        self._remaining(
            "allocation and spend terminal proof",
            minimum_millis=120_000,
        )
        allocation_spend = self._live_allocation_spend_proof(
            state=state,
            observed_at=second_completed,
        )
        worker = worker_effects[0].evidence
        if (
            first_completed < first_started
            or second_completed < second_started
            or first_completed >= second_started
            or first_evidence["request_members"]
            != second_evidence["request_members"]
            or first_evidence["job_members"]
            != second_evidence["job_members"]
            or type(worker) is not dict
            or worker.get("pagination_complete") is not True
            or worker.get("active_instance_ids") != []
            or worker.get("forced_termination_count") != 0
            or type(worker.get("instances")) is not list
            or any(
                type(item) is not dict
                or item.get("state") not in _WORKER_TERMINAL
                for item in worker["instances"]
            )
        ):
            raise ValueError(
                "terminal quiescence chronology or ancestry drifted"
            )
        return _effect(
            "ALLOCATION_CLOSED",
            {
                "terminal_worker_evidence_identity_sha256": (
                    worker_effects[0].canonical_identity_sha256
                ),
                "request_job_ancestry_sha256": canonical_sha256(
                    {
                        "requests": first_evidence["request_members"],
                        "jobs": first_evidence["job_members"],
                    }
                ),
                "allocation_spend_proof": dict(allocation_spend),
                "terminal_observations": [
                    {
                        "scan_started_at": first_started,
                        "scan_completed_at": first_completed,
                        "identity_sha256": first[
                            "canonical_identity_sha256"
                        ],
                    },
                    {
                        "scan_started_at": second_started,
                        "scan_completed_at": second_completed,
                        "identity_sha256": second[
                            "canonical_identity_sha256"
                        ],
                    },
                ],
            },
        )

    def commit_allocation_closed(
        self, state: SupportContinuationState
    ) -> None:
        if (
            state.phase != "ALLOCATION_CLOSED"
            or not state.effects
            or state.effects[-1].effect_kind != "ALLOCATION_CLOSED"
        ):
            raise ValueError(
                "allocation closure successor state is not exact"
            )
        self._transition_phase(
            before="DRAIN_REQUESTED",
            after="ALLOCATION_CLOSED",
            state_identity_sha256=state.canonical_identity_sha256,
        )

    def _request_retained_lifecycle(
        self,
        *,
        detail: Mapping[str, object],
    ) -> str:
        response = self._clients.events.put_events(
            Entries=[
                {
                    "Source": "keep.glm52.task12",
                    "DetailType": "RETAINED_LIFECYCLE_REQUESTED",
                    "Detail": canonical_json_bytes(detail).decode("ascii"),
                    "EventBusName": "default",
                }
            ]
        )
        metadata = _metadata(response, "retained lifecycle request")
        entries = response.get("Entries")
        if (
            response.get("FailedEntryCount") != 0
            or type(entries) is not list
            or len(entries) != 1
            or type(entries[0].get("EventId")) is not str
            or not entries[0]["EventId"]
        ):
            raise ValueError(
                "retained lifecycle request was not accepted exactly once"
            )
        return str(metadata["RequestId"])

    def create_terminal_v2(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        key = (
            "campaigns/"
            + self._config.run_id
            + "/submissions/production/generations/"
            + f"{state.handoff.body['generation']:08d}"
            + "/terminal/PRODUCTION_TERMINAL_V2.json"
        )
        request_body = {
            "schema_version": 1,
            "record_type": (
                "glm52_task12_support_terminal_v2_request_v1"
            ),
            "run_id": self._config.run_id,
            "activation_id": self._config.activation_id,
            "generation": state.handoff.body["generation"],
            "generation_text": (
                f"{state.handoff.body['generation']:08d}"
            ),
            "caller_function_version_arn": (
                self._caller_function_version_arn
            ),
            "caller_state_machine_version_arn": (
                state.handoff.approved_task11_workflow_version_arn
            ),
            "caller_execution_arn": state.task11_context[
                "task11_execution_arn"
            ],
            "support_state_identity_sha256": (
                state.canonical_identity_sha256
            ),
            "support_state": asdict(state),
        }
        request = {
            **request_body,
            "canonical_identity_sha256": canonical_sha256(request_body),
        }
        result = self._invoke(
            self._config.terminal_v2_version_arn,
            request,
        )
        identity = result.get("canonical_identity_sha256")
        result_body = dict(result)
        result_body.pop("canonical_identity_sha256", None)
        required = {
            "schema_version",
            "record_type",
            "outcome",
            "activation_id",
            "generation",
            "caller_function_version_arn",
            "caller_state_machine_version_arn",
            "caller_execution_arn",
            "support_state_identity_sha256",
            "coordinate",
            "object_version_id",
            "body_identity_sha256",
            "file_sha256",
            "write_request_ids",
        }
        if (
            set(result_body) != required
            or result["schema_version"] != 1
            or result["record_type"]
            != "glm52_task12_support_terminal_v2_result_v1"
            or result["outcome"] != "SUCCEEDED"
            or result["activation_id"] != self._config.activation_id
            or result["generation"] != state.handoff.body["generation"]
            or result["caller_function_version_arn"]
            != self._caller_function_version_arn
            or result["caller_state_machine_version_arn"]
            != state.handoff.approved_task11_workflow_version_arn
            or result["caller_execution_arn"]
            != state.task11_context["task11_execution_arn"]
            or result["support_state_identity_sha256"]
            != state.canonical_identity_sha256
            or result["coordinate"] != key
            or _SHA.fullmatch(result["body_identity_sha256"] or "")
            is None
            or _SHA.fullmatch(result["file_sha256"] or "") is None
            or type(result["object_version_id"]) is not str
            or not result["object_version_id"]
            or type(result["write_request_ids"]) is not list
            or not result["write_request_ids"]
            or any(
                type(item) is not str or not item
                for item in result["write_request_ids"]
            )
            or identity != canonical_sha256(result_body)
        ):
            raise ValueError(
                "activation-domain TerminalV2 writer result drifted"
            )
        return _effect(
            "TERMINAL_V2",
            {
                "writer_function_version_arn": (
                    self._config.terminal_v2_version_arn
                ),
                "coordinate": key,
                "object_version_id": result["object_version_id"],
                "body_identity_sha256": result[
                    "body_identity_sha256"
                ],
                "file_sha256": result["file_sha256"],
                "write_request_ids": result["write_request_ids"],
                "support_execution_arn": result[
                    "caller_execution_arn"
                ],
            },
        )

    def request_finalization(
        self, state: SupportContinuationState
    ) -> SupportEffect:
        body = {
            "schema_version": 1,
            "record_type": (
                "glm52_task12_retained_lifecycle_request_v1"
            ),
            "run_id": self._config.run_id,
            "activation_id": self._config.activation_id,
            "generation": state.handoff.body["generation"],
            "generation_text": (
                f"{state.handoff.body['generation']:08d}"
            ),
            "reason": "SUPPORT_FINALIZATION_REQUESTED",
            "support_state_identity_sha256": (
                state.canonical_identity_sha256
            ),
            "terminal_v2_identity_sha256": (
                state.effects[-1].evidence["body_identity_sha256"]
            ),
            "support_execution_arn": state.task11_context[
                "task11_execution_arn"
            ],
            "support_state_machine_version_arn": (
                state.handoff.approved_task11_workflow_version_arn
            ),
        }
        body["canonical_identity_sha256"] = canonical_sha256(body)
        item = {
                "PK": {"S": ledger_pk(self._config.run_id)},
                "SK": {
                    "S": (
                        "ACTIVATION#"
                        + self._config.activation_id
                        + "#FINALIZATION_REQUESTED"
                    )
                },
                "record": {
                    "S": canonical_json_bytes(body).decode("ascii")
                },
            }
        request = {
            "TableName": self._config.ledger_table_name,
            "Item": item,
            "ConditionExpression": (
                "attribute_not_exists(PK) AND attribute_not_exists(SK)"
            ),
        }
        try:
            response = self._clients.dynamodb.put_item(**request)
            metadata = _metadata(
                response, "FINALIZATION_REQUESTED write"
            )
        except Exception:
            readback = self._clients.dynamodb.get_item(
                TableName=self._config.ledger_table_name,
                Key={"PK": item["PK"], "SK": item["SK"]},
                ConsistentRead=True,
                ReturnConsumedCapacity="NONE",
            )
            metadata = _metadata(
                readback, "FINALIZATION_REQUESTED readback"
            )
            if readback.get("Item") != item:
                raise
        retained_request_id = self._request_retained_lifecycle(detail=body)
        return _effect(
            "FINALIZATION_REQUESTED",
            {
                "record_identity_sha256": body[
                    "canonical_identity_sha256"
                ],
                "write_request_id": metadata["RequestId"],
                "retained_request_id": retained_request_id,
                "exact_finalizer_route": "RETAINED_LIFECYCLE",
            },
        )


def build_aws_support_continuation_services(
    *,
    context: object,
    environment: Mapping[str, str] | None = None,
    clients: AwsSupportContinuationClients | None = None,
    sleeper: object = time.sleep,
) -> AwsSupportContinuationServices:
    """Build the closed AWS service surface for one Lambda invocation."""

    config = _configuration(environment)
    remaining_time_in_millis = getattr(
        context, "get_remaining_time_in_millis", None
    )
    if not callable(remaining_time_in_millis):
        raise TypeError(
            "support continuation Lambda remaining-time context is absent"
        )
    if clients is None:
        import boto3
        from botocore.config import Config

        client_config = Config(
            retries={"mode": "standard", "total_max_attempts": 1},
            connect_timeout=2,
            read_timeout=15,
        )

        clients = AwsSupportContinuationClients(
            s3=boto3.client(
                "s3", region_name=config.region, config=client_config
            ),
            dynamodb=boto3.client(
                "dynamodb",
                region_name=config.region,
                config=client_config,
            ),
            lambda_client=boto3.client(
                "lambda",
                region_name=config.region,
                config=client_config,
            ),
            ec2=boto3.client(
                "ec2", region_name=config.region, config=client_config
            ),
            ssm=boto3.client(
                "ssm", region_name=config.region, config=client_config
            ),
            events=boto3.client(
                "events", region_name=config.region, config=client_config
            ),
            stepfunctions=boto3.client(
                "stepfunctions",
                region_name=config.region,
                config=client_config,
            ),
        )
    if type(clients) is not AwsSupportContinuationClients:
        raise TypeError("support continuation AWS clients are not exact")
    return AwsSupportContinuationServices(
        config=config,
        clients=clients,
        caller_function_version_arn=getattr(
            context, "invoked_function_arn", None
        ),
        sleeper=sleeper,
        remaining_time_in_millis=remaining_time_in_millis,
    )


__all__ = [
    "AwsSupportContinuationClients",
    "AwsSupportContinuationConfig",
    "AwsSupportContinuationServices",
    "build_aws_support_continuation_services",
]
