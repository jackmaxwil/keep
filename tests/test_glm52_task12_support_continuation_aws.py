"""Production-edge proofs for the support continuation AWS adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from dataclasses import asdict, replace
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.dynamodb import decode_item, encode_item
from glm52_enforcement.task12_correlation import (
    CorrelatedRequest,
    RequestCorrelation,
    Task11HandoffAuthority,
)
from glm52_enforcement.task12_operations import (
    build_operation_descriptor_graph,
)
from glm52_enforcement.task12_support_continuation_aws import (
    AwsSupportContinuationClients,
    AwsSupportContinuationConfig,
    AwsSupportContinuationServices,
    build_aws_support_continuation_services,
)
from glm52_enforcement.task12_support_continuation_runtime import (
    SupportContinuationState,
    SupportEffect,
)
from glm52_enforcement.task12_support_plane import (
    Task12SupportPlaneInputs,
    render_task12_support_plane_fragment,
)
from glm52_enforcement.task12_support_terminal_v2 import (
    AwsSupportTerminalV2Services,
    SupportTerminalV2Clients,
    SupportTerminalV2Config,
    SupportTerminalV2Error,
)

SHA = "a" * 64
TASK11_EXECUTION = (
    "arn:aws:states:us-west-2:246813579024:"
    "execution:keep-glm52-h1g-support:execution-1"
)
TASK11_VERSION = (
    "arn:aws:states:us-west-2:246813579024:"
    "stateMachine:keep-glm52-h1g-support:7"
)
SUPPORT_VERSION = (
    "arn:aws:lambda:us-west-2:246813579024:"
    "function:keep-glm52-h1g-support-deadline:11"
)
NUMERIC_VERSION = (
    "arn:aws:lambda:us-west-2:246813579024:"
    "function:keep-glm52-h1g-numeric-binding:11"
)
CANCELLATION_VERSION = (
    "arn:aws:lambda:us-west-2:246813579024:"
    "function:keep-glm52-h1g-retained-cancellation:12"
)
WORKER_DRAIN_VERSION = (
    "arn:aws:lambda:us-west-2:246813579024:"
    "function:keep-glm52-h1g-worker-drain-signal:12"
)
LIABILITY_WATCHER_VERSION = (
    "arn:aws:lambda:us-west-2:246813579024:"
    "function:keep-glm52-h1g-worker-launch-custody:7"
)
TERMINAL_VERSION = (
    "arn:aws:lambda:us-west-2:246813579024:"
    "function:keep-glm52-h1g-terminal-v2-writer:13"
)


def _meta(request_id: str) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RequestId": request_id,
        "RetryAttempts": 0,
    }


class _Lambda:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def invoke(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        request = json.loads(kwargs["Payload"])
        result_body = {
            "schema_version": 1,
            "record_type": (
                "glm52_task12_support_terminal_v2_result_v1"
            ),
            "outcome": "SUCCEEDED",
            "activation_id": "activation-1",
            "generation": 1,
            "caller_function_version_arn": SUPPORT_VERSION,
            "caller_state_machine_version_arn": TASK11_VERSION,
            "caller_execution_arn": TASK11_EXECUTION,
            "support_state_identity_sha256": request[
                "support_state_identity_sha256"
            ],
            "coordinate": (
                "campaigns/glm52-sky-20260724/submissions/production/"
                "generations/00000001/terminal/"
                "PRODUCTION_TERMINAL_V2.json"
            ),
            "object_version_id": "terminal-version-1",
            "body_identity_sha256": SHA,
            "file_sha256": SHA,
            "write_request_ids": ["s3-put-1", "ddb-put-1"],
        }
        result = {
            **result_body,
            "canonical_identity_sha256": canonical_sha256(result_body),
        }
        return {
            "StatusCode": 200,
            "ExecutedVersion": str(kwargs["FunctionName"]).rsplit(":", 1)[1],
            "Payload": io.BytesIO(canonical_json_bytes(result)),
            "ResponseMetadata": _meta("lambda-1"),
        }


class _Events:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def put_events(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "FailedEntryCount": 0,
            "Entries": [{"EventId": "event-1"}],
            "ResponseMetadata": _meta("events-1"),
        }


class _Dynamo:
    def __init__(self) -> None:
        self.put_calls: list[dict[str, object]] = []

    def put_item(self, **kwargs: object) -> dict[str, object]:
        self.put_calls.append(dict(kwargs))
        return {"ResponseMetadata": _meta("ddb-finalization-1")}


def _runtime_observation(
    *,
    correlation_identity_sha256: str,
    observed_at: str,
    requests: list[dict[str, object]] | None = None,
    jobs: list[dict[str, object]] | None = None,
    controllers: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    controller = {
        "members": [] if controllers is None else controllers,
        "evidence_identity_sha256": canonical_sha256(
            [] if controllers is None else controllers
        ),
    }
    request_snapshot = {
        "observed_at": observed_at,
        "pagination_complete": True,
        "requests": [] if requests is None else requests,
        "controller_snapshot": controller,
    }
    job_snapshot = {
        "observed_at": observed_at,
        "pagination_complete": True,
        "jobs": [] if jobs is None else jobs,
        "controller_snapshot": controller,
    }
    body = {
        "schema_version": 1,
        "record_type": "glm52_task12_runtime_observation_v1",
        "account_id": "246813579024",
        "region": "us-west-2",
        "run_id": "glm52-sky-20260724",
        "activation_id": "activation-1",
        "correlation_identity_sha256": correlation_identity_sha256,
        "task9_deployed_identity_sha256": SHA,
        "observed_at": observed_at,
        "request_snapshot": request_snapshot,
        "job_snapshot": job_snapshot,
        "controller_snapshot": controller,
        "request_transport": {
            "request_id": "sky-request-read-1",
            "response_sha256": canonical_sha256(request_snapshot),
            "tls_peer_certificate_sha256": SHA,
        },
        "job_transport": {
            "request_id": "sky-job-read-1",
            "response_sha256": canonical_sha256(job_snapshot),
            "tls_peer_certificate_sha256": SHA,
        },
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


class _ObservationLambda:
    def __init__(self, observations: list[dict[str, object]]) -> None:
        self.observations = list(observations)
        self.calls: list[dict[str, object]] = []

    def invoke(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        request = json.loads(kwargs["Payload"])
        value = dict(self.observations.pop(0))
        value["correlation_identity_sha256"] = request[
            "correlation_identity_sha256"
        ]
        body = dict(value)
        body.pop("canonical_identity_sha256", None)
        value["canonical_identity_sha256"] = canonical_sha256(body)
        return {
            "StatusCode": 200,
            "ExecutedVersion": str(kwargs["FunctionName"]).rsplit(":", 1)[1],
            "Payload": io.BytesIO(canonical_json_bytes(value)),
            "ResponseMetadata": _meta("lambda-observation-1"),
        }


class _ControllerEc2:
    def __init__(
        self,
        *,
        activation_id: str,
        trace: list[str] | None = None,
    ) -> None:
        self.activation_id = activation_id
        self.trace = trace
        self.stopped = False
        self.stop_calls: list[dict[str, object]] = []
        self.describe_calls: list[dict[str, object]] = []

    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        self.describe_calls.append(dict(kwargs))
        return {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-0123456789abcdef0",
                            "State": {
                                "Name": (
                                    "stopped"
                                    if self.stopped
                                    else "running"
                                )
                            },
                            "Tags": [
                                {"Key": "RunId", "Value": "glm52-sky-20260724"},
                                {
                                    "Key": "ActivationId",
                                    "Value": self.activation_id,
                                },
                                {"Key": "Purpose", "Value": "combined-host"},
                            ],
                        }
                    ]
                }
            ],
            "ResponseMetadata": _meta("controller-describe-1"),
        }

    def stop_instances(self, **kwargs: object) -> dict[str, object]:
        self.stop_calls.append(dict(kwargs))
        if self.trace is not None:
            self.trace.append("controller-stop")
        self.stopped = True
        return {"ResponseMetadata": _meta("controller-stop-1")}

    def get_waiter(self, _name: str) -> object:
        return SimpleNamespace(wait=lambda **_kwargs: None)


class _OfflineSsm:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def describe_instance_information(
        self, **kwargs: object
    ) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "InstanceInformationList": [],
            "ResponseMetadata": _meta("ssm-offline-1"),
        }


class _PagedWorkerEc2:
    def __init__(self) -> None:
        self.describe_calls: list[dict[str, object]] = []
        self.terminate_calls: list[dict[str, object]] = []
        self.terminated = False

    def _instance(self, instance_id: str) -> dict[str, object]:
        return {
            "InstanceId": instance_id,
            "InstanceType": "p5.48xlarge",
            "State": {
                "Name": "terminated" if self.terminated else "running"
            },
            "Tags": [
                {"Key": "RunId", "Value": "glm52-sky-20260724"},
                {"Key": "activation-id", "Value": "activation-1"},
                {"Key": "Market", "Value": "on-demand"},
            ],
        }

    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        self.describe_calls.append(dict(kwargs))
        if "NextToken" not in kwargs:
            return {
                "Reservations": [
                    {"Instances": [self._instance("i-0123456789abcdef1")]}
                ],
                "NextToken": "worker-page-2",
                "ResponseMetadata": _meta("workers-page-1"),
            }
        assert kwargs["NextToken"] == "worker-page-2"
        return {
            "Reservations": [
                {"Instances": [self._instance("i-0123456789abcdef2")]}
            ],
            "ResponseMetadata": _meta("workers-page-2"),
        }

    def terminate_instances(self, **kwargs: object) -> dict[str, object]:
        self.terminate_calls.append(dict(kwargs))
        return {"ResponseMetadata": _meta("worker-terminate-1")}


class _SingleWorkerEc2:
    instance_id = "i-0123456789abcdef1"

    def __init__(self) -> None:
        self.state = "running"
        self.describe_calls: list[dict[str, object]] = []

    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        self.describe_calls.append(dict(kwargs))
        return {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": self.instance_id,
                            "InstanceType": "p5.48xlarge",
                            "Tags": [
                                {
                                    "Key": "RunId",
                                    "Value": "glm52-sky-20260724",
                                },
                                {
                                    "Key": "activation-id",
                                    "Value": "activation-1",
                                },
                                {
                                    "Key": "Market",
                                    "Value": "on-demand",
                                },
                            ],
                            "State": {"Name": self.state},
                        }
                    ]
                }
            ],
            "ResponseMetadata": _meta(
                "worker-read-" + str(len(self.describe_calls))
            ),
        }


def _worker_drain_lambda_result(
    operation_input: dict[str, object],
    *,
    instance_id: str = _SingleWorkerEc2.instance_id,
) -> dict[str, object]:
    retained_authority = {
        "schema_version": 1,
        "record_type": "glm52_task10_worker_drain_authority_v1",
        "instance_id": instance_id,
        "authority_body_sha256": SHA,
    }
    domain_result = {
        "authority_identity_sha256": SHA,
        "candidate_identity_sha256": SHA,
        "instance_id": instance_id,
        "command_id": "01234567-89ab-cdef-0123-456789abcdef",
        "request_id": "ssm-drain-1",
        "response_identity_sha256": SHA,
        "retained_authority": retained_authority,
        "retained_authority_body_sha256": SHA,
        "state": "SSM_ACCEPTED_NOT_DRAINED",
        "drain_complete": False,
        "liability_settled": False,
    }
    dispatch_body = {
        key: value
        for key, value in domain_result.items()
        if key != "retained_authority"
    }
    domain_result["dispatch_identity_sha256"] = canonical_sha256(
        dispatch_body
    )
    strict_body = {
        "result_kind": (
            "glm52_task12_retained_reconcile_workers_and_"
            "allocations_result_v1"
        ),
        "action_kind": "reconcile_workers_and_allocations",
        "behavior_kind": "EXTERNAL_ACTION",
        "predecessor_operation_kind": (
            "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_"
            "TRANSFER_LIABILITIES"
        ),
        "predecessor_result_identity_sha256": operation_input[
            "task12_last_result"
        ]["canonical_body_sha256"],
        "operation_payload_identity_sha256": SHA,
        "domain_result": domain_result,
        "domain_result_identity_sha256": canonical_sha256(domain_result),
    }
    strict = {
        **strict_body,
        "canonical_body_sha256": canonical_sha256(strict_body),
    }
    body = {
        "schema_version": 1,
        "record_type": "glm52_task12_lambda_result_v1",
        "handler_kind": "RETAINED_WORKER_DRAIN",
        "operation_kind": (
            "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS"
        ),
        "operation_input_identity_sha256": canonical_sha256(operation_input),
        "outcome": "SUCCEEDED",
        "result": strict,
    }
    return {
        **body,
        "canonical_body_sha256": canonical_sha256(body),
    }


class _GuardedWorkerLambda:
    def __init__(self, ec2: _SingleWorkerEc2) -> None:
        self.ec2 = ec2
        self.calls: list[dict[str, object]] = []

    def invoke(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        function_name = str(kwargs["FunctionName"])
        request = json.loads(kwargs["Payload"])
        if function_name == WORKER_DRAIN_VERSION:
            value = _worker_drain_lambda_result(
                request["operation_input"]
            )
        elif function_name == LIABILITY_WATCHER_VERSION:
            assert request == {
                "mode": "LIABILITY_WATCH",
                "event_accelerator": "owner-failover",
            }
            self.ec2.state = "terminated"
            value = {
                "settled": True,
                "instance_ids": [self.ec2.instance_id],
                "next_scan_at": "2026-07-29T12:01:00Z",
                "incident": False,
            }
        else:  # pragma: no cover - exact route assertion
            raise AssertionError(function_name)
        return {
            "StatusCode": 200,
            "ExecutedVersion": function_name.rsplit(":", 1)[1],
            "Payload": io.BytesIO(canonical_json_bytes(value)),
            "ResponseMetadata": _meta(
                "lambda-" + str(len(self.calls))
            ),
        }


class _GuardedMultiplicityLambda:
    def __init__(self, ec2: _PagedWorkerEc2) -> None:
        self.ec2 = ec2
        self.calls: list[dict[str, object]] = []

    def invoke(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        assert kwargs["FunctionName"] == LIABILITY_WATCHER_VERSION
        assert json.loads(kwargs["Payload"]) == {
            "mode": "LIABILITY_WATCH",
            "event_accelerator": "owner-failover",
        }
        instance_ids = [
            "i-0123456789abcdef1",
            "i-0123456789abcdef2",
        ]
        self.ec2.terminated = True
        value = {
            "settled": False,
            "instance_ids": instance_ids,
            "next_scan_at": "2026-07-29T12:01:00Z",
            "incident": True,
        }
        return {
            "StatusCode": 200,
            "ExecutedVersion": "7",
            "Payload": io.BytesIO(canonical_json_bytes(value)),
            "ResponseMetadata": _meta("lambda-multiplicity-1"),
        }


class _EndlessWorkerEc2:
    def __init__(self) -> None:
        self.describe_calls: list[dict[str, object]] = []

    def describe_instances(self, **kwargs: object) -> dict[str, object]:
        self.describe_calls.append(dict(kwargs))
        return {
            "Reservations": [],
            "NextToken": "page-" + str(len(self.describe_calls) + 1),
            "ResponseMetadata": _meta(
                "workers-page-" + str(len(self.describe_calls))
            ),
        }


class _HistoryStepFunctions:
    def __init__(
        self,
        *,
        events: list[dict[str, object]] | None = None,
        endless: bool = False,
    ) -> None:
        self.events = [] if events is None else events
        self.endless = endless
        self.calls: list[dict[str, object]] = []

    def get_execution_history(
        self, **kwargs: object
    ) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        response: dict[str, object] = {
            "events": [] if self.endless else self.events,
            "ResponseMetadata": _meta(
                "history-" + str(len(self.calls))
            ),
        }
        if self.endless:
            response["nextToken"] = (
                "history-page-" + str(len(self.calls) + 1)
            )
        return response


class _ClosureDynamo(_Dynamo):
    def __init__(self) -> None:
        super().__init__()
        self.update_calls: list[dict[str, object]] = []

    def update_item(self, **kwargs: object) -> dict[str, object]:
        self.update_calls.append(dict(kwargs))
        return {"ResponseMetadata": _meta("allocation-close-1")}


class _TerminalDynamo:
    def __init__(self, state_identity: str) -> None:
        self.state_identity = state_identity
        self.put_calls: list[dict[str, object]] = []
        operation = "RETAINED_CREATE_OR_RECONCILE_TERMINAL_V2"
        self.descriptor = build_operation_descriptor_graph(
            activation_id="activation-1",
            activation_ordinal=1,
            generation=1,
            ledger_partition_key="RUN#glm52-sky-20260724",
            campaign_bucket="keep-fixture-246813579024-us-west-2",
            kms_key_id=(
                "arn:aws:kms:us-west-2:246813579024:key/"
                "12345678-1234-4234-8234-1234567890ab"
            ),
            static_authority_sha256=SHA,
        )[operation]
        self.descriptor_sk = (
            "ACTIVATION#activation-1#TASK12_LAMBDA_INPUT#"
            "RETAINED_TERMINAL_V2#"
            + operation
            + "#00000001"
        )
        authority_body = {
            "schema_version": 1,
            "record_type": (
                "glm52_task12_lambda_invocation_authority_v1"
            ),
            "handler_kind": "RETAINED_TERMINAL_V2",
            "activation_id": "activation-1",
            "activation_ordinal": 1,
            "generation": 1,
            "generation_text": "00000001",
            "operations": {
                operation: {
                    "state_sort_key": self.descriptor_sk,
                    "state_body_sha256": self.descriptor[
                        "canonical_body_sha256"
                    ],
                }
            },
        }
        self.authority = {
            **authority_body,
            "canonical_body_sha256": canonical_sha256(
                authority_body
            ),
        }
        self.authority_raw = canonical_json_bytes(self.authority)
        authority_coordinate = {
            "bucket": "keep-fixture-246813579024-us-west-2",
            "key": (
                "campaigns/glm52-sky-20260724/task12/deployment/"
                "activation-1/00000001/TerminalV2Support/7/"
                "authority.json"
            ),
            "version_id": "authority-version-1",
            "file_sha256": hashlib.sha256(
                self.authority_raw
            ).hexdigest(),
        }
        roles = {
            "authority": authority_coordinate,
            "campaign_bucket": (
                "keep-fixture-246813579024-us-west-2"
            ),
            "spend_runtime_prefix": (
                "campaigns/glm52-sky-20260724/runtime"
            ),
            "campaign_descriptor_key": (
                "campaigns/glm52-sky-20260724/submissions/"
                "production/descriptor.json"
            ),
            "campaign_descriptor_version_id": "descriptor-version-1",
            "campaign_descriptor_file_sha256": SHA,
            "gpu_spend_approval_key": (
                "campaigns/glm52-sky-20260724/authorities/"
                "GPU_SPEND_APPROVAL.json"
            ),
            "gpu_spend_approval_version_id": "approval-version-1",
            "gpu_spend_approval_file_sha256": SHA,
            "kms_key_id": (
                "arn:aws:kms:us-west-2:246813579024:key/"
                "12345678-1234-4234-8234-1234567890ab"
            ),
            "ledger_table_name": "ledger",
            "numeric_binding_version_arn": SUPPORT_VERSION,
            "task9_deployed_identity_coordinate": {
                "bucket": "keep-fixture-246813579024-us-west-2",
                "key": "task9.json",
                "version_id": "task9-version-1",
                "file_sha256": SHA,
                "body_sha256": SHA,
            },
            "task9_deployed_identity_sha256": SHA,
            "task10_worker_descriptor_coordinate": {
                "bucket": "keep-fixture-246813579024-us-west-2",
                "key": "task10.json",
                "version_id": "task10-version-1",
                "file_sha256": SHA,
                "body_sha256": SHA,
            },
            "terminal_evidence_prefix": (
                "campaigns/glm52-sky-20260724/submissions/"
                "production/generations/00000001/terminal-evidence/"
            ),
            "support_observer_version_arn": SUPPORT_VERSION,
        }
        deployment_body = {
            "schema_version": 1,
            "record_type": "glm52_task12_lambda_deployment_v1",
            "account_id": "246813579024",
            "region": "us-west-2",
            "handler_kind": "RETAINED_TERMINAL_V2",
            "mode": "SUPPORT_OBSERVER",
            "activation_id": "activation-1",
            "activation_ordinal": 1,
            "generation": 1,
            "generation_text": "00000001",
            "function_name": "keep-glm52-h1g-terminal-v2-writer",
            "function_version": "13",
            "invoked_function_version_arn": TERMINAL_VERSION,
            "caller_state_machine_version_arn": TASK11_VERSION,
            "role_coordinates": roles,
        }
        self.deployment = {
            **deployment_body,
            "canonical_body_sha256": canonical_sha256(
                deployment_body
            ),
        }

    def get_item(self, **kwargs: object) -> dict[str, object]:
        key = decode_item(kwargs["Key"])
        if key["SK"] == "ACTIVATION_INDEX":
            body = {
                "current_activation_id": "activation-1",
                "current_activation_ordinal": 1,
            }
        elif key["SK"].startswith("TASK12_LAMBDA_DEPLOYMENT#"):
            body = self.deployment
        elif key["SK"] == self.descriptor_sk:
            body = self.descriptor
        else:
            body = {
                "phase": "ALLOCATION_CLOSED",
                "support_state_identity_sha256": self.state_identity,
            }
        return {
            "Item": encode_item({**key, **body}),
            "ResponseMetadata": _meta("ddb-get-1"),
        }

    def put_item(self, **kwargs: object) -> dict[str, object]:
        self.put_calls.append(dict(kwargs))
        return {"ResponseMetadata": _meta("ddb-put-1")}


class _TerminalS3:
    def __init__(self, authority_raw: bytes) -> None:
        self.calls: list[dict[str, object]] = []
        self.authority_raw = authority_raw

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "VersionId": "terminal-version-1",
            "ResponseMetadata": _meta("s3-put-1"),
        }

    def get_object(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["VersionId"] == "authority-version-1"
        return {
            "Body": io.BytesIO(self.authority_raw),
            "VersionId": "authority-version-1",
            "ResponseMetadata": _meta("s3-authority-get-1"),
        }


class _IamBoundLostResponseTerminalS3(_TerminalS3):
    """Exercise the exact generated role against the ambiguous-write calls."""

    def __init__(
        self,
        authority_raw: bytes,
        statements: list[dict[str, object]],
    ) -> None:
        super().__init__(authority_raw)
        self.statements = statements
        self.terminal_raw: bytes | None = None
        self.terminal_key: str | None = None

    def _authorize(
        self,
        action: str,
        resource: str,
        *,
        prefix: str | None = None,
    ) -> None:
        for statement in self.statements:
            actions = statement.get("Action")
            allowed_actions = actions if type(actions) is list else [actions]
            resources = statement.get("Resource")
            allowed_resources = (
                resources if type(resources) is list else [resources]
            )
            if action not in allowed_actions or not any(
                type(pattern) is str and fnmatchcase(resource, pattern)
                for pattern in allowed_resources
            ):
                continue
            condition = statement.get("Condition")
            if condition is None:
                return
            allowed_prefixes = (
                condition.get("ForAnyValue:StringLike", {}).get("s3:prefix")
                if type(condition) is dict
                else None
            )
            if (
                type(allowed_prefixes) is list
                and type(prefix) is str
                and any(
                    type(pattern) is str
                    and fnmatchcase(prefix, pattern)
                    for pattern in allowed_prefixes
                )
            ):
                return
        raise PermissionError(f"{action} denied for {resource}")

    def put_object(self, **kwargs: object) -> dict[str, object]:
        key = str(kwargs["Key"])
        bucket = str(kwargs["Bucket"])
        self._authorize("s3:PutObject", f"arn:aws:s3:::{bucket}/{key}")
        self.calls.append(dict(kwargs))
        self.terminal_key = key
        self.terminal_raw = bytes(kwargs["Body"])
        raise TimeoutError("response lost after durable object creation")

    def list_object_versions(
        self, **kwargs: object
    ) -> dict[str, object]:
        prefix = str(kwargs["Prefix"])
        bucket = str(kwargs["Bucket"])
        self._authorize(
            "s3:ListBucketVersions",
            f"arn:aws:s3:::{bucket}",
            prefix=prefix,
        )
        assert prefix == self.terminal_key
        return {
            "IsTruncated": False,
            "Versions": [
                {
                    "Key": self.terminal_key,
                    "VersionId": "terminal-version-1",
                }
            ],
            "ResponseMetadata": _meta("s3-list-1"),
        }

    def get_object(self, **kwargs: object) -> dict[str, object]:
        key = str(kwargs["Key"])
        bucket = str(kwargs["Bucket"])
        self._authorize(
            "s3:GetObjectVersion",
            f"arn:aws:s3:::{bucket}/{key}",
        )
        if kwargs["VersionId"] == "authority-version-1":
            return super().get_object(**kwargs)
        assert kwargs["VersionId"] == "terminal-version-1"
        assert key == self.terminal_key
        return {
            "Body": io.BytesIO(self.terminal_raw),
            "VersionId": "terminal-version-1",
            "ResponseMetadata": _meta("s3-terminal-get-1"),
        }


class _TerminalStepFunctions:
    def describe_execution(self, **_kwargs: object) -> dict[str, object]:
        return {
            "status": "RUNNING",
            "stateMachineVersionArn": TASK11_VERSION,
            "ResponseMetadata": _meta("sfn-get-1"),
        }


class _RetryingTerminalStepFunctions(_TerminalStepFunctions):
    def describe_execution(self, **kwargs: object) -> dict[str, object]:
        result = super().describe_execution(**kwargs)
        result["ResponseMetadata"]["RetryAttempts"] = 1
        return result


class _TestTerminalServices(AwsSupportTerminalV2Services):
    def _live_terminal_authority(
        self,
        request: object,
        deployment: object,
    ) -> tuple[object, tuple[object, ...], dict[str, tuple], None]:
        return (
            SimpleNamespace(
                state="CLOSED",
                ledger_head_identity_sha256="e" * 64,
                remaining_gpu_seconds=0,
                remaining_gpu_cost_usd="0.00",
            ),
            (),
            {
                "glm52_production_worker_launch": (),
                "glm52_production_worker_launch_liability": (),
                "glm52_production_worker_launch_liability_settlement": (),
                "glm52_production_post_terminal_allocation": (),
            },
            None,
        )


def _effect(kind: str) -> SupportEffect:
    body = {
        "effect_kind": kind,
        "outcome": "SUCCEEDED",
        "evidence": {"identity": SHA},
    }
    return SupportEffect(
        **body,
        canonical_identity_sha256=canonical_sha256(body),
    )


def _worker_terminal_effect() -> SupportEffect:
    body = {
        "effect_kind": "WORKER_TERMINAL",
        "outcome": "SUCCEEDED",
        "evidence": {
            "pagination_complete": True,
            "instances": [],
            "active_instance_ids": [],
            "drain_dispatches": [],
            "request_ids": ["worker-read-1"],
            "forced_termination_count": 0,
        },
    }
    return SupportEffect(
        **body,
        canonical_identity_sha256=canonical_sha256(body),
    )


def _state(*, terminal: bool = False) -> SupportContinuationState:
    handoff = Task11HandoffAuthority(
        bucket="keep-fixture-246813579024-us-west-2",
        key=(
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/handoff/SKY_POST_HANDOFF.json"
        ),
        version_id="handoff-version-1",
        file_sha256=SHA,
        body={
            "generation": 1,
            "campaign_identity_sha256": SHA,
        },
        body_sha256=SHA,
        approved_task11_workflow_version_arn=TASK11_VERSION,
        task11_execution_arn=TASK11_EXECUTION,
    )
    effects = (_effect("ALLOCATION_CLOSED"),)
    if terminal:
        effects += (
            SupportEffect(
                effect_kind="TERMINAL_V2",
                outcome="SUCCEEDED",
                evidence={"body_identity_sha256": SHA},
                canonical_identity_sha256=canonical_sha256(
                    {
                        "effect_kind": "TERMINAL_V2",
                        "outcome": "SUCCEEDED",
                        "evidence": {"body_identity_sha256": SHA},
                    }
                ),
            ),
        )
    body = {
        "phase": (
            "TERMINAL_V2_PUBLISHED" if terminal else "ALLOCATION_CLOSED"
        ),
        "task11_context": {
            "task11_execution_arn": TASK11_EXECUTION,
            "approved_task11_workflow_version_arn": TASK11_VERSION,
        },
        "handoff": asdict(handoff),
        "correlation_tuple": None,
        "correlation": None,
        "request_cardinality": "ZERO",
        "binding": None,
        "zero_request_proof": None,
        "effects": [asdict(item) for item in effects],
    }
    return SupportContinuationState(
        phase=body["phase"],
        task11_context=body["task11_context"],
        handoff=handoff,
        correlation_tuple=None,
        correlation=None,
        request_cardinality="ZERO",
        binding=None,
        zero_request_proof=None,
        effects=effects,
        canonical_identity_sha256=canonical_sha256(body),
    )


def _services() -> tuple[
    AwsSupportContinuationServices, _Lambda, _Events, _Dynamo
]:
    lambdas = _Lambda()
    events = _Events()
    dynamo = _Dynamo()
    services = AwsSupportContinuationServices(
        config=AwsSupportContinuationConfig(
            account_id="246813579024",
            region="us-west-2",
            run_id="glm52-sky-20260724",
            activation_id="activation-1",
            ledger_table_name="ledger",
            campaign_bucket="keep-fixture-246813579024-us-west-2",
            numeric_binding_version_arn=NUMERIC_VERSION,
            retained_cancellation_version_arn=CANCELLATION_VERSION,
            worker_drain_version_arn=WORKER_DRAIN_VERSION,
            task9_liability_watcher_version_arn=(
                LIABILITY_WATCHER_VERSION
            ),
            terminal_v2_version_arn=TERMINAL_VERSION,
            combined_host_instance_id="i-0123456789abcdef0",
            quiescence_seconds=1,
        ),
        clients=AwsSupportContinuationClients(
            s3=object(),
            dynamodb=dynamo,
            lambda_client=lambdas,
            ec2=object(),
            ssm=object(),
            events=events,
            stepfunctions=object(),
        ),
        caller_function_version_arn=SUPPORT_VERSION,
        sleeper=lambda _seconds: None,
        remaining_time_in_millis=lambda: 540_000,
    )
    return services, lambdas, events, dynamo


def _services_with(
    *,
    lambda_client: object | None = None,
    dynamodb: object | None = None,
    ec2: object | None = None,
    ssm: object | None = None,
    stepfunctions: object | None = None,
    remaining_time_in_millis: object = lambda: 540_000,
) -> AwsSupportContinuationServices:
    services, lambdas, events, default_dynamo = _services()
    clients = replace(
        services._clients,
        lambda_client=lambdas if lambda_client is None else lambda_client,
        dynamodb=default_dynamo if dynamodb is None else dynamodb,
        ec2=object() if ec2 is None else ec2,
        ssm=object() if ssm is None else ssm,
        events=events,
        stepfunctions=(
            object() if stepfunctions is None else stepfunctions
        ),
    )
    result = AwsSupportContinuationServices(
        config=services._config,
        clients=clients,
        caller_function_version_arn=SUPPORT_VERSION,
        sleeper=lambda _seconds: None,
        remaining_time_in_millis=remaining_time_in_millis,
    )
    result._task9_coordinate = {
        "bucket": "keep-fixture-246813579024-us-west-2",
        "key": "task9.json",
        "version_id": "task9-version-1",
        "file_sha256": SHA,
    }
    result._task9_sha256 = SHA
    return result


def _state_at(
    phase: str,
    *effects: SupportEffect,
) -> SupportContinuationState:
    base = _state()
    body = {
        "phase": phase,
        "task11_context": dict(base.task11_context),
        "handoff": asdict(base.handoff),
        "correlation_tuple": base.correlation_tuple,
        "correlation": base.correlation,
        "request_cardinality": base.request_cardinality,
        "binding": base.binding,
        "zero_request_proof": base.zero_request_proof,
        "effects": [asdict(item) for item in effects],
    }
    return SupportContinuationState(
        phase=phase,
        task11_context=base.task11_context,
        handoff=base.handoff,
        correlation_tuple=base.correlation_tuple,
        correlation=base.correlation,
        request_cardinality=base.request_cardinality,
        binding=base.binding,
        zero_request_proof=base.zero_request_proof,
        effects=effects,
        canonical_identity_sha256=canonical_sha256(body),
    )


def test_terminal_v2_uses_exact_writer_version_without_recovery_wake() -> None:
    services, lambdas, events, _dynamo = _services()

    result = services.create_terminal_v2(_state())

    assert result.effect_kind == "TERMINAL_V2"
    assert len(lambdas.calls) == 1
    assert lambdas.calls[0]["FunctionName"] == TERMINAL_VERSION
    request = json.loads(lambdas.calls[0]["Payload"])
    assert request["record_type"] == (
        "glm52_task12_support_terminal_v2_request_v1"
    )
    assert request["caller_function_version_arn"] == SUPPORT_VERSION
    assert request["caller_state_machine_version_arn"] == TASK11_VERSION
    assert request["caller_execution_arn"] == TASK11_EXECUTION
    assert events.calls == []


def test_finalization_wake_preserves_the_terminal_support_caller() -> None:
    services, lambdas, events, dynamo = _services()

    result = services.request_finalization(_state(terminal=True))

    assert result.effect_kind == "FINALIZATION_REQUESTED"
    assert lambdas.calls == []
    assert len(events.calls) == 1
    entry = events.calls[0]["Entries"][0]
    assert entry["DetailType"] == "RETAINED_LIFECYCLE_REQUESTED"
    detail = json.loads(entry["Detail"])
    assert detail["reason"] == "SUPPORT_FINALIZATION_REQUESTED"
    assert detail["support_execution_arn"] == TASK11_EXECUTION
    assert detail["support_state_machine_version_arn"] == TASK11_VERSION
    assert detail["terminal_v2_identity_sha256"] == SHA
    durable = json.loads(
        dynamo.put_calls[0]["Item"]["record"]["S"]
    )
    assert durable == detail
    assert dynamo.put_calls[0]["Item"]["PK"] == {
        "S": "RUN#glm52-sky-20260724"
    }


def test_support_drain_phase_uses_the_canonical_ledger_partition() -> None:
    dynamo = _ClosureDynamo()
    services = _services_with(dynamodb=dynamo)

    result = services.arm_drain(_state_at("RUNTIME_OBSERVED"))

    assert result.effect_kind == "DRAIN_ARMED"
    assert dynamo.update_calls[0]["Key"] == {
        "PK": {"S": "RUN#glm52-sky-20260724"},
        "SK": {"S": "ACTIVATION#activation-1#SUPPORT_DRAIN"},
    }


def test_controller_quiesce_rejects_foreign_activation_before_stop() -> None:
    ec2 = _ControllerEc2(activation_id="foreign-activation")
    services = _services_with(ec2=ec2)
    state = _state_at(
        "DRAIN_REQUESTED",
        _effect("DRAIN_ARMED"),
        _effect("DRAIN_REQUESTED"),
    )

    with pytest.raises(ValueError, match="controller identity or tags drifted"):
        services.quiesce_controller(state)

    assert len(ec2.describe_calls) == 1
    assert ec2.stop_calls == []


def test_controller_quiesce_uses_guarded_stop_and_terminal_readbacks() -> None:
    trace: list[str] = []
    ec2 = _ControllerEc2(
        activation_id="activation-1",
        trace=trace,
    )
    ssm = _OfflineSsm()
    observations = [
        _runtime_observation(
            correlation_identity_sha256=SHA,
            observed_at="2026-07-29T12:01:00Z",
        ),
        _runtime_observation(
            correlation_identity_sha256=SHA,
            observed_at="2026-07-29T12:02:00Z",
        ),
    ]
    observation_lambda = _ObservationLambda(observations)
    original_invoke = observation_lambda.invoke

    def traced_invoke(**kwargs: object) -> dict[str, object]:
        trace.append("numeric-binding-relay")
        return original_invoke(**kwargs)

    observation_lambda.invoke = traced_invoke  # type: ignore[method-assign]
    services = _services_with(
        lambda_client=observation_lambda,
        ec2=ec2,
        ssm=ssm,
    )

    result = services.quiesce_controller(
        _state_at(
            "DRAIN_REQUESTED",
            _effect("DRAIN_ARMED"),
            _effect("DRAIN_REQUESTED"),
        )
    )

    assert result.effect_kind == "CONTROLLER_QUIESCED"
    assert ec2.stop_calls == [
        {
            "InstanceIds": ["i-0123456789abcdef0"],
            "Force": False,
            "Hibernate": False,
            "DryRun": False,
        }
    ]
    assert ssm.calls == [
        {
            "Filters": [
                {
                    "Key": "InstanceIds",
                    "Values": ["i-0123456789abcdef0"],
                }
            ],
            "MaxResults": 5,
        }
    ]
    assert result.evidence["ec2_state"] == "stopped"
    assert trace == [
        "numeric-binding-relay",
        "numeric-binding-relay",
        "controller-stop",
    ]


def test_request_cancel_refuses_multiple_before_any_lambda_mutation() -> None:
    services = _services_with()
    base = _state_at("RUNTIME_OBSERVED", _effect("RUNTIME_OBSERVATION"))
    correlation = RequestCorrelation(
        kind="MULTIPLE",
        matches=(
            CorrelatedRequest(
                request_id="request-1",
                state="RUNNING",
                body={"request_id": "request-1", "state": "RUNNING"},
            ),
            CorrelatedRequest(
                request_id="request-2",
                state="RUNNING",
                body={"request_id": "request-2", "state": "RUNNING"},
            ),
        ),
    )
    state = replace(
        base,
        correlation=correlation,
        request_cardinality="MULTIPLE",
    )

    with pytest.raises(
        ValueError,
        match="multiple request correlation is a sealed incident",
    ):
        services.reconcile_request_cancel(state)

    assert services._clients.lambda_client.calls == []


def test_worker_multiplicity_uses_guarded_termination_only_route() -> None:
    """Break caught: multiple workers abort before Task 9 can drain them."""

    ec2 = _PagedWorkerEc2()
    lambdas = _GuardedMultiplicityLambda(ec2)
    services = _services_with(ec2=ec2, lambda_client=lambdas)

    result = services.reconcile_workers(
        _state_at(
            "CONTROLLER_QUIESCED",
            _effect("CONTROLLER_QUIESCED"),
        )
    )

    assert [call.get("NextToken") for call in ec2.describe_calls[:2]] == [
        None,
        "worker-page-2",
    ]
    assert ec2.terminate_calls == []
    assert [call["FunctionName"] for call in lambdas.calls] == [
        LIABILITY_WATCHER_VERSION
    ]
    assert result.evidence["active_instance_ids"] == []
    assert result.evidence["drain_dispatches"] == []
    assert result.evidence["guarded_liability_watch"]["incident"] is True
    assert result.evidence["guarded_liability_watch"]["instance_ids"] == [
        "i-0123456789abcdef1",
        "i-0123456789abcdef2",
    ]
    assert {
        item["state"] for item in result.evidence["instances"]
    } == {"terminated"}


def test_worker_drain_invokes_exact_published_routes_before_terminal_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ec2 = _SingleWorkerEc2()
    lambdas = _GuardedWorkerLambda(ec2)
    services = _services_with(lambda_client=lambdas, ec2=ec2)
    predecessor = {
        "schema_version": 1,
        "record_type": "glm52_task12_lambda_result_v1",
        "handler_kind": "RETAINED_WORKER_DRAIN",
        "operation_kind": (
            "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_"
            "TRANSFER_LIABILITIES"
        ),
        "operation_input_identity_sha256": SHA,
        "outcome": "SUCCEEDED",
        "result": {},
    }
    predecessor["canonical_body_sha256"] = canonical_sha256(predecessor)
    operation_input = {"task12_last_result": predecessor}
    envelope = {
        "activation_id": "activation-1",
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "dispatch_identity_sha256": SHA,
        "caller_state_machine_arn": (
            "arn:aws:states:us-west-2:246813579024:"
            "stateMachine:keep-glm52-h1g-retainedlifecycle"
        ),
        "state_machine_execution_arn": (
            "arn:aws:states:us-west-2:246813579024:"
            "execution:keep-glm52-h1g-retainedlifecycle:owner-1"
        ),
        "operation_kind": (
            "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS"
        ),
        "operation_input": operation_input,
    }
    monkeypatch.setattr(
        services,
        "_retained_worker_drain_envelope",
        lambda **_kwargs: envelope,
    )

    result = services.reconcile_workers(
        _state_at(
            "CONTROLLER_QUIESCED",
            _effect("CONTROLLER_QUIESCED"),
        )
    )

    assert result.effect_kind == "WORKER_TERMINAL"
    assert result.evidence["active_instance_ids"] == []
    assert result.evidence["instances"][0]["state"] == "terminated"
    assert [
        call["FunctionName"] for call in lambdas.calls
    ] == [WORKER_DRAIN_VERSION, LIABILITY_WATCHER_VERSION]
    assert all(
        call["InvocationType"] == "RequestResponse"
        for call in lambdas.calls
    )
    assert result.evidence["guarded_liability_watch"]["settled"] is True
    assert result.evidence["forced_termination_count"] == 0


def test_worker_discovery_changing_tokens_has_a_hard_page_ceiling() -> None:
    ec2 = _EndlessWorkerEc2()
    services = _services_with(ec2=ec2)

    with pytest.raises(
        ValueError,
        match="worker discovery pagination ceiling exceeded",
    ):
        services._discover_workers()

    assert len(ec2.describe_calls) == 32


def test_worker_discovery_stops_before_transport_after_deadline_margin() -> None:
    ec2 = _EndlessWorkerEc2()
    remaining = iter((120_000, 9_999))
    services = _services_with(
        ec2=ec2,
        remaining_time_in_millis=lambda: next(remaining),
    )

    with pytest.raises(
        ValueError,
        match="worker discovery lacks bounded remaining time",
    ):
        services._discover_workers()

    assert len(ec2.describe_calls) == 1


def test_worker_drain_operation_input_comes_from_bounded_retained_history() -> None:
    predecessor = {
        "schema_version": 1,
        "record_type": "glm52_task12_lambda_result_v1",
        "handler_kind": "RETAINED_WORKER_DRAIN",
        "operation_kind": (
            "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_"
            "TRANSFER_LIABILITIES"
        ),
        "operation_input_identity_sha256": SHA,
        "outcome": "SUCCEEDED",
        "result": {},
    }
    predecessor["canonical_body_sha256"] = canonical_sha256(predecessor)
    expected = {
        "activation_id": "activation-1",
        "task12_last_result": predecessor,
    }
    history = _HistoryStepFunctions(
        events=[
            {
                "id": 1,
                "type": "TaskStateExited",
                "stateExitedEventDetails": {
                    "name": (
                        "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_"
                        "TRANSFER_LIABILITIES"
                    ),
                    "output": json.dumps(expected),
                },
            }
        ]
    )
    services = _services_with(stepfunctions=history)

    result, request_ids = services._retained_history_operation_input(
        (
            "arn:aws:states:us-west-2:246813579024:"
            "execution:keep-glm52-h1g-retainedlifecycle:owner-1"
        )
    )

    assert result == expected
    assert request_ids == ("history-1",)
    assert history.calls == [
        {
            "executionArn": (
                "arn:aws:states:us-west-2:246813579024:"
                "execution:keep-glm52-h1g-retainedlifecycle:owner-1"
            ),
            "includeExecutionData": True,
            "maxResults": 1000,
            "reverseOrder": False,
        }
    ]


def test_retained_history_changing_tokens_has_a_hard_page_ceiling() -> None:
    history = _HistoryStepFunctions(endless=True)
    services = _services_with(stepfunctions=history)

    with pytest.raises(
        ValueError,
        match="retained execution history pagination ceiling exceeded",
    ):
        services._retained_history_operation_input(
            (
                "arn:aws:states:us-west-2:246813579024:"
                "execution:keep-glm52-h1g-retainedlifecycle:owner-1"
            )
        )

    assert len(history.calls) == 32


def test_foreign_worker_drain_result_blocks_task9_guarded_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ec2 = _SingleWorkerEc2()

    class _ForeignWorkerLambda(_GuardedWorkerLambda):
        def invoke(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(dict(kwargs))
            request = json.loads(kwargs["Payload"])
            value = _worker_drain_lambda_result(
                request["operation_input"],
                instance_id="i-0123456789abcdef2",
            )
            return {
                "StatusCode": 200,
                "ExecutedVersion": "12",
                "Payload": io.BytesIO(canonical_json_bytes(value)),
                "ResponseMetadata": _meta("lambda-foreign-worker"),
            }

    lambdas = _ForeignWorkerLambda(ec2)
    services = _services_with(lambda_client=lambdas, ec2=ec2)
    predecessor = {
        "schema_version": 1,
        "record_type": "glm52_task12_lambda_result_v1",
        "handler_kind": "RETAINED_WORKER_DRAIN",
        "operation_kind": (
            "RETAINED_RECONCILE_WORKER_LAUNCHES_AND_"
            "TRANSFER_LIABILITIES"
        ),
        "operation_input_identity_sha256": SHA,
        "outcome": "SUCCEEDED",
        "result": {},
    }
    predecessor["canonical_body_sha256"] = canonical_sha256(predecessor)
    monkeypatch.setattr(
        services,
        "_retained_worker_drain_envelope",
        lambda **_kwargs: {
            "activation_id": "activation-1",
            "activation_ordinal": 1,
            "generation": 1,
            "generation_text": "00000001",
            "dispatch_identity_sha256": SHA,
            "caller_state_machine_arn": (
                "arn:aws:states:us-west-2:246813579024:"
                "stateMachine:keep-glm52-h1g-retainedlifecycle"
            ),
            "state_machine_execution_arn": (
                "arn:aws:states:us-west-2:246813579024:"
                "execution:keep-glm52-h1g-retainedlifecycle:owner-1"
            ),
            "operation_kind": (
                "RETAINED_RECONCILE_WORKERS_AND_ALLOCATIONS"
            ),
            "operation_input": {"task12_last_result": predecessor},
        },
    )

    with pytest.raises(
        ValueError,
        match="published worker-drain domain result drifted",
    ):
        services.reconcile_workers(
            _state_at(
                "CONTROLLER_QUIESCED",
                _effect("CONTROLLER_QUIESCED"),
            )
        )

    assert len(lambdas.calls) == 1
    assert ec2.state == "running"


def test_allocation_close_refuses_active_sky_job_before_phase_mutation() -> None:
    active_request = {
        "request_id": "request-1",
        "state": "RUNNING",
        "created_at": "2026-07-29T12:00:00Z",
        "updated_at": "2026-07-29T12:00:00Z",
        "evidence_identity_sha256": SHA,
    }
    active_job = {
        "job_id": "1",
        "request_id": "request-1",
        "state": "RUNNING",
        "evidence_identity_sha256": SHA,
    }
    observation = _runtime_observation(
        correlation_identity_sha256=SHA,
        observed_at="2026-07-29T12:01:00Z",
        requests=[active_request],
        jobs=[active_job],
    )
    dynamo = _ClosureDynamo()
    services = _services_with(
        lambda_client=_ObservationLambda([observation]),
        dynamodb=dynamo,
    )

    with pytest.raises(ValueError, match="terminal request/job state is absent"):
        services.close_allocations(
            _state_at(
                "WORKER_TERMINAL",
                _effect("WORKER_TERMINAL"),
            )
        )

    assert dynamo.update_calls == []


def test_allocation_close_requires_authenticated_spend_and_family_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import task12_support_continuation_aws as adapter

    observations = [
        _runtime_observation(
            correlation_identity_sha256=SHA,
            observed_at="2026-07-29T12:01:00Z",
        ),
        _runtime_observation(
            correlation_identity_sha256=SHA,
            observed_at="2026-07-29T12:03:00Z",
        ),
    ]
    dynamo = _ClosureDynamo()
    services = _services_with(
        lambda_client=_ObservationLambda(observations),
        dynamodb=dynamo,
    )
    times = iter(
        ("2026-07-29T12:00:00Z", "2026-07-29T12:02:00Z")
    )
    monkeypatch.setattr(adapter, "_utc_now", lambda: next(times))

    def _refuse(**_kwargs: object) -> dict[str, object]:
        raise ValueError("live allocation or spend state is not closed")

    services._live_allocation_spend_proof = _refuse

    with pytest.raises(
        ValueError,
        match="live allocation or spend state is not closed",
    ):
        services.close_allocations(
            _state_at(
                "WORKER_TERMINAL",
                _worker_terminal_effect(),
            )
        )

    assert dynamo.update_calls == []


@pytest.mark.parametrize(
    ("retry_attempts", "executed_version"),
    [(1, "13"), (0, "12")],
)
def test_exact_lambda_invoke_rejects_retry_or_executed_version_drift(
    retry_attempts: int,
    executed_version: str,
) -> None:
    class _DriftedLambda:
        def invoke(self, **_kwargs: object) -> dict[str, object]:
            response = _Lambda().invoke(
                FunctionName=TERMINAL_VERSION,
                InvocationType="RequestResponse",
                Payload=canonical_json_bytes(
                    {"support_state_identity_sha256": SHA}
                ),
            )
            response["ExecutedVersion"] = executed_version
            response["ResponseMetadata"]["RetryAttempts"] = retry_attempts
            return response

    services = _services_with(lambda_client=_DriftedLambda())

    with pytest.raises(
        ValueError,
        match="support callee response is not exact zero-retry",
    ):
        services.create_terminal_v2(_state())


def test_default_support_clients_are_all_total_max_attempts_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _Boto3:
        @staticmethod
        def client(name: str, **kwargs: object) -> object:
            calls.append((name, dict(kwargs)))
            return object()

    monkeypatch.setitem(sys.modules, "boto3", _Boto3)
    build_aws_support_continuation_services(
        context=SimpleNamespace(
            invoked_function_arn=SUPPORT_VERSION,
            get_remaining_time_in_millis=lambda: 540_000,
        ),
        environment={
            "GLM52_ACCOUNT_ID": "246813579024",
            "AWS_REGION": "us-west-2",
            "GLM52_RUN_ID": "glm52-sky-20260724",
            "GLM52_ACTIVATION_ID": "activation-1",
            "GLM52_LEDGER_TABLE_NAME": "ledger",
            "GLM52_CAMPAIGN_BUCKET": (
                "keep-fixture-246813579024-us-west-2"
            ),
            "GLM52_NUMERIC_BINDING_VERSION_ARN": NUMERIC_VERSION,
            "GLM52_RETAINED_CANCELLATION_VERSION_ARN": (
                CANCELLATION_VERSION
            ),
            "GLM52_WORKER_DRAIN_VERSION_ARN": WORKER_DRAIN_VERSION,
            "GLM52_TASK9_LIABILITY_WATCHER_VERSION_ARN": (
                LIABILITY_WATCHER_VERSION
            ),
            "GLM52_TERMINAL_V2_VERSION_ARN": TERMINAL_VERSION,
            "GLM52_COMBINED_HOST_INSTANCE_ID": "i-0123456789abcdef0",
            "GLM52_SUPPORT_QUIESCENCE_SECONDS": "60",
        },
    )

    assert [name for name, _kwargs in calls] == [
        "s3",
        "dynamodb",
        "lambda",
        "ec2",
        "ssm",
        "events",
        "stepfunctions",
    ]
    assert all(
        kwargs["config"].retries
        == {"mode": "standard", "total_max_attempts": 1}
        and kwargs["config"].connect_timeout == 2
        and kwargs["config"].read_timeout == 15
        for _name, kwargs in calls
    )


@pytest.mark.parametrize(
    "context",
    [
        SimpleNamespace(invoked_function_arn=SUPPORT_VERSION),
        SimpleNamespace(
            invoked_function_arn=SUPPORT_VERSION,
            get_remaining_time_in_millis=540_000,
        ),
    ],
)
def test_production_builder_requires_real_lambda_remaining_time_context(
    context: object,
) -> None:
    with pytest.raises(TypeError, match="remaining-time context"):
        build_aws_support_continuation_services(
            context=context,
            environment={
                "GLM52_ACCOUNT_ID": "246813579024",
                "AWS_REGION": "us-west-2",
                "GLM52_RUN_ID": "glm52-sky-20260724",
                "GLM52_ACTIVATION_ID": "activation-1",
                "GLM52_LEDGER_TABLE_NAME": "ledger",
                "GLM52_CAMPAIGN_BUCKET": (
                    "keep-fixture-246813579024-us-west-2"
                ),
                "GLM52_NUMERIC_BINDING_VERSION_ARN": NUMERIC_VERSION,
                "GLM52_RETAINED_CANCELLATION_VERSION_ARN": (
                    CANCELLATION_VERSION
                ),
                "GLM52_WORKER_DRAIN_VERSION_ARN": WORKER_DRAIN_VERSION,
                "GLM52_TASK9_LIABILITY_WATCHER_VERSION_ARN": (
                    LIABILITY_WATCHER_VERSION
                ),
                "GLM52_TERMINAL_V2_VERSION_ARN": TERMINAL_VERSION,
                "GLM52_COMBINED_HOST_INSTANCE_ID": (
                    "i-0123456789abcdef0"
                ),
                "GLM52_SUPPORT_QUIESCENCE_SECONDS": "60",
            },
            clients=AwsSupportContinuationClients(
                s3=object(),
                dynamodb=object(),
                lambda_client=object(),
                ec2=object(),
                ssm=object(),
                events=object(),
                stepfunctions=object(),
            ),
        )


def test_default_terminal_v2_clients_are_all_total_max_attempts_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from glm52_enforcement import task12_support_terminal_v2 as terminal

    calls: list[tuple[str, dict[str, object]]] = []

    class _Boto3:
        @staticmethod
        def client(name: str, **kwargs: object) -> object:
            calls.append((name, dict(kwargs)))
            return object()

    monkeypatch.setitem(sys.modules, "boto3", _Boto3)
    terminal.build_aws_support_terminal_v2_services(
        environment={
            "GLM52_ACCOUNT_ID": "246813579024",
            "GLM52_RUN_ID": "glm52-sky-20260724",
            "GLM52_ACTIVATION_ID": "activation-1",
            "GLM52_TASK12_DEPLOYMENT_TABLE_NAME": "ledger",
            "GLM52_CAMPAIGN_BUCKET": (
                "keep-fixture-246813579024-us-west-2"
            ),
        },
    )

    assert [name for name, _kwargs in calls] == [
        "s3",
        "dynamodb",
        "stepfunctions",
    ]
    assert all(
        kwargs["config"].retries
        == {"mode": "standard", "total_max_attempts": 1}
        and kwargs["config"].connect_timeout == 2
        and kwargs["config"].read_timeout == 15
        for _name, kwargs in calls
    )


def test_terminal_v2_rejects_nonzero_retry_metadata() -> None:
    request = _support_terminal_request()
    dynamo = _TerminalDynamo(request["support_state_identity_sha256"])
    services = _TestTerminalServices(
        config=SupportTerminalV2Config(
            account_id="246813579024",
            run_id="glm52-sky-20260724",
            activation_id="activation-1",
            ledger_table_name="ledger",
            campaign_bucket="keep-fixture-246813579024-us-west-2",
        ),
        clients=SupportTerminalV2Clients(
            s3=_TerminalS3(dynamo.authority_raw),
            dynamodb=dynamo,
            stepfunctions=_RetryingTerminalStepFunctions(),
        ),
    )

    with pytest.raises(
        SupportTerminalV2Error,
        match="not zero-retry",
    ):
        services.execute(
            request,
            invoked_function_arn=TERMINAL_VERSION,
        )

    assert dynamo.put_calls == []


def _support_terminal_request() -> dict[str, object]:
    handoff = {
        "bucket": "keep-fixture-246813579024-us-west-2",
        "key": (
            "campaigns/glm52-sky-20260724/submissions/production/"
            "generations/00000001/handoff/SKY_POST_HANDOFF.json"
        ),
        "version_id": "handoff-version-1",
        "file_sha256": SHA,
        "body": {
            "generation": 1,
            "campaign_identity_sha256": SHA,
            "request_body_sha256": SHA,
        },
        "body_sha256": SHA,
        "approved_task11_workflow_version_arn": TASK11_VERSION,
        "task11_execution_arn": TASK11_EXECUTION,
    }
    zero_body = {
        "handoff_identity_sha256": SHA,
        "pagination_complete": True,
        "first_observed_at": "2026-07-29T12:00:00Z",
        "second_observed_at": "2026-07-29T12:01:00Z",
        "request_ids": [],
        "job_ids": [],
        "worker_instance_ids": [],
        "allocation_ids": [],
        "controller_work_ids": [],
    }
    zero = {
        **zero_body,
        "canonical_identity_sha256": canonical_sha256(zero_body),
    }
    worker_body = {
        "effect_kind": "WORKER_TERMINAL",
        "outcome": "SUCCEEDED",
        "evidence": {"instances": []},
    }
    worker = {
        **worker_body,
        "canonical_identity_sha256": canonical_sha256(worker_body),
    }
    observations = [
        {
            "scan_started_at": "2026-07-29T12:02:00Z",
            "scan_completed_at": "2026-07-29T12:03:00Z",
            "identity_sha256": "b" * 64,
        },
        {
            "scan_started_at": "2026-07-29T12:04:00Z",
            "scan_completed_at": "2026-07-29T12:05:00Z",
            "identity_sha256": "d" * 64,
        },
    ]
    close_body = {
        "effect_kind": "ALLOCATION_CLOSED",
        "outcome": "SUCCEEDED",
        "evidence": {"terminal_observations": observations},
    }
    close = {
        **close_body,
        "canonical_identity_sha256": canonical_sha256(close_body),
    }
    state_body = {
        "phase": "ALLOCATION_CLOSED",
        "task11_context": {
            "task11_execution_arn": TASK11_EXECUTION,
            "approved_task11_workflow_version_arn": TASK11_VERSION,
        },
        "handoff": handoff,
        "correlation_tuple": None,
        "correlation": {"kind": "ZERO", "matches": []},
        "request_cardinality": "ZERO",
        "binding": None,
        "zero_request_proof": zero,
        "effects": [worker, close],
    }
    state = {
        **state_body,
        "canonical_identity_sha256": canonical_sha256(state_body),
    }
    body = {
        "schema_version": 1,
        "record_type": "glm52_task12_support_terminal_v2_request_v1",
        "run_id": "glm52-sky-20260724",
        "activation_id": "activation-1",
        "generation": 1,
        "generation_text": "00000001",
        "caller_function_version_arn": SUPPORT_VERSION,
        "caller_state_machine_version_arn": TASK11_VERSION,
        "caller_execution_arn": TASK11_EXECUTION,
        "support_state_identity_sha256": state[
            "canonical_identity_sha256"
        ],
        "support_state": state,
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


def test_packaged_terminal_handler_executes_activation_writer_and_successor() -> None:
    request = _support_terminal_request()
    dynamo = _TerminalDynamo(request["support_state_identity_sha256"])
    s3 = _TerminalS3(dynamo.authority_raw)
    services = _TestTerminalServices(
        config=SupportTerminalV2Config(
            account_id="246813579024",
            run_id="glm52-sky-20260724",
            activation_id="activation-1",
            ledger_table_name="ledger",
            campaign_bucket="keep-fixture-246813579024-us-west-2",
        ),
        clients=SupportTerminalV2Clients(
            s3=s3,
            dynamodb=dynamo,
            stepfunctions=_TerminalStepFunctions(),
        ),
    )
    path = (
        Path(__file__).parents[1]
        / "aws/glm52-gpu/lambda/retained_terminal_v2_handler.py"
    )
    spec = importlib.util.spec_from_file_location(
        "packaged_retained_terminal_v2_handler",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.main(
        request,
        type("Context", (), {"invoked_function_arn": TERMINAL_VERSION})(),
        services=services,
    )

    assert result["outcome"] == "SUCCEEDED"
    assert result["caller_execution_arn"] == TASK11_EXECUTION
    assert len(s3.calls) == 1
    assert s3.calls[0]["IfNoneMatch"] == "*"
    assert len(dynamo.put_calls) == 2
    authority = next(
        decode_item(call["Item"])
        for call in dynamo.put_calls
        if decode_item(call["Item"]).get("record_type")
        == "glm52_task12_support_terminal_writer_authority_v1"
    )
    assert authority["action"]["authority_domain"] == "ACTIVATION"
    assert authority["audit"]["audit_kind"] == (
        "H1F_GENESIS_TO_ZERO_CHILD"
    )
    control = next(
        decode_item(call["Item"])
        for call in dynamo.put_calls
        if decode_item(call["Item"]).get("record_type")
        == "glm52_task12_versioned_writer_control_v1"
    )
    assert control["record_type"] == (
        "glm52_task12_versioned_writer_control_v1"
    )


def test_terminal_v2_generated_role_adopts_one_fresh_lost_response_version() -> None:
    """The deployed IAM must authorize every call in the real adoption branch."""

    request = _support_terminal_request()
    dynamo = _TerminalDynamo(request["support_state_identity_sha256"])
    resources = render_task12_support_plane_fragment(
        inputs=Task12SupportPlaneInputs(
            activation_id="activation-1",
            lambda_code_sha256=SHA,
            lambda_code_bucket="lambda-code",
            lambda_code_key="task12.zip",
            lambda_code_version="version-1",
            ledger_table_arn=(
                "arn:aws:dynamodb:us-west-2:246813579024:table/ledger"
            ),
            retained_kms_key_arn=(
                "arn:aws:kms:us-west-2:246813579024:key/"
                "12345678-1234-4234-8234-1234567890ab"
            ),
            model_bucket_arn=(
                "arn:aws:s3:::keep-fixture-246813579024-us-west-2"
            ),
            worker_drain_document_arn=(
                "arn:aws:ssm:us-west-2:246813579024:"
                "document/keep-glm52-worker-drain"
            ),
        )
    )["Resources"]
    statements = resources["TerminalV2Role"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    s3 = _IamBoundLostResponseTerminalS3(
        dynamo.authority_raw,
        statements,
    )
    services = _TestTerminalServices(
        config=SupportTerminalV2Config(
            account_id="246813579024",
            run_id="glm52-sky-20260724",
            activation_id="activation-1",
            ledger_table_name="ledger",
            campaign_bucket="keep-fixture-246813579024-us-west-2",
        ),
        clients=SupportTerminalV2Clients(
            s3=s3,
            dynamodb=dynamo,
            stepfunctions=_TerminalStepFunctions(),
        ),
    )

    result = services.execute(
        request,
        invoked_function_arn=TERMINAL_VERSION,
    )

    assert result["outcome"] == "SUCCEEDED"
    assert result["object_version_id"] == "terminal-version-1"
    assert "s3-list-1" in result["write_request_ids"]
    assert "s3-terminal-get-1" in result["write_request_ids"]
