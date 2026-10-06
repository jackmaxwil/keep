"""Finite, no-retry CloudFormation fence and support-lifecycle boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .cloudformation_stacks import (
    ACCOUNT_ID,
    FENCE_LOGICAL_ID,
    REGION,
    RUN_ID,
    StackKind,
    canonical_json_bytes,
    stack_name,
)
from .dynamodb import decode_item, encode_item
from .h1f_adapter import H1fAuditResult
from .records import ledger_pk, ledger_sk
from .fence_artifacts import (
    ArtifactCoordinate,
    ExecutorAuthorityClass,
    FenceArtifactEntry,
    FenceManifest,
    FenceSlot,
    FenceTransitionRequest,
    parse_fence_entry,
    parse_fence_manifest,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_ROLE_ARN = re.compile(
    rf"arn:aws:iam::{ACCOUNT_ID}:role/"
    r"[A-Za-z0-9+=,.@_-]+(?:/[A-Za-z0-9+=,.@_-]+)*\Z"
)
_SUPPORT_DELETION_ROLE_ARN = (
    f"arn:aws:iam::{ACCOUNT_ID}:role/keep-glm52-h1g-support-deletion"
)
_STACK_UUID = (
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)
_STACK_ID = re.compile(
    rf"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/"
    rf"keep-glm52-h1g-(?:fence|support)/{_STACK_UUID}\Z"
)



@dataclass(frozen=True)
class FencePostPolicyProof:
    audit: H1fAuditResult
    final_audit: H1fAuditResult
    policy_sha256: str
    observation_count: int
    separation_seconds: int
    unique_zero_child: bool
    first_observed_at: str
    second_observed_at: str
    first_evidence_sha256: str
    second_evidence_sha256: str
    first_stable_observation_sha256: str
    second_stable_observation_sha256: str
    stable_observation_sha256: str
    iam_role_inventory_sha256: str
    stable_publisher_role_arn: str
    stable_publisher_principal_id: str
    stable_publisher_assumed_arn: str
    denial_probe_operations: Tuple[str, ...]
    denial_probe_error_codes: Tuple[str, ...]
    denial_probe_attempt_counts: Tuple[int, ...]
    denial_probe_host_ids: Tuple[str, ...]
    service_request_ids: Tuple[str, ...]




class AmbiguousTransportError(RuntimeError):
    """The injected transport cannot prove whether the one request committed."""


_SUPPORT_DELETION_OPERATIONS = (
    "ReadExactFinalizationSupportDelete",
    "DescribeExactRetainedLifecycleExecution",
    "StartExactRetainedLifecycleVersion",
    "GetRetainedLifecycleExecutionHistory",
    "DescribeExactSupportStackAbsence",
)


def support_deletion_operation_surface() -> Tuple[str, ...]:
    return _SUPPORT_DELETION_OPERATIONS


@dataclass(frozen=True)
class ReviewedSupportResource:
    logical_id: str
    resource_type: str
    physical_id: str
    source_api: str
    source_response_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.logical_id) is not str
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,254}", self.logical_id)
            is None
            or type(self.resource_type) is not str
            or re.fullmatch(
                r"(?:AWS::[A-Za-z0-9]+::[A-Za-z0-9]+|"
                r"Custom::[A-Za-z0-9]+)",
                self.resource_type,
            )
            is None
            or type(self.physical_id) is not str
            or not self.physical_id
            or self.source_api != "DescribeStackResource"
            or type(self.source_response_sha256) is not str
            or _SHA256.fullmatch(self.source_response_sha256) is None
        ):
            raise ValueError("reviewed support resource identity is not exact")


def reviewed_support_resources_from_postcreate_inventory(
    value: object,
) -> Tuple[ReviewedSupportResource, ...]:
    """Project only authenticated CFN detail rows into deletion authority."""

    if type(value) is not dict:
        raise ValueError("postcreate support inventory is not one object")
    body = dict(value)
    identity = body.pop("canonical_body_sha256", None)
    rows = body.get("stack_resources")
    if (
        type(identity) is not str
        or _SHA256.fullmatch(identity) is None
        or hashlib.sha256(canonical_json_bytes(body)).hexdigest() != identity
        or body.get("schema_version") != 1
        or body.get("record_type")
        != "glm52_h1g_support_postcreate_inventory_v1"
        or type(body.get("support_stack_id")) is not str
        or _STACK_ID.fullmatch(body["support_stack_id"]) is None
        or type(rows) is not list
        or len(rows) != 202
    ):
        raise ValueError("postcreate support inventory identity is foreign")
    result = []
    for row in rows:
        if (
            type(row) is not dict
            or set(row)
            != {
                "logical_id",
                "resource_type",
                "physical_id",
                "source_api",
                "source_response_sha256",
                "action_families",
            }
            or type(row.get("action_families")) is not list
            or not row["action_families"]
            or any(
                type(action) is not str or not action
                for action in row["action_families"]
            )
        ):
            raise ValueError("postcreate support inventory row is foreign")
        result.append(
            ReviewedSupportResource(
                logical_id=row["logical_id"],
                resource_type=row["resource_type"],
                physical_id=row["physical_id"],
                source_api=row["source_api"],
                source_response_sha256=row[
                    "source_response_sha256"
                ],
            )
        )
    projected = tuple(result)
    if len({row.logical_id for row in projected}) != 202:
        raise ValueError("postcreate support logical inventory is duplicated")
    return projected


@dataclass(frozen=True)
class SupportDeletionAuthority:
    finalization_state: str
    support_stack_id: str
    support_stack_name: str
    deletion_role_arn: str
    retained_stack_id: str
    fence_stack_id: str
    lifecycle_action_identity_sha256: str
    activation_id: str
    action_key: str
    state_machine_version_arn: str
    owner_nonce_sha256: str
    request_evidence_sha256: str
    reviewed_resources: Tuple[ReviewedSupportResource, ...]
    reviewed_inventory_sha256: str

    def __post_init__(self) -> None:
        def exact_stack_id(kind: StackKind, value: object) -> bool:
            return (
                type(value) is str
                and re.fullmatch(
                    rf"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/"
                    rf"{re.escape(stack_name(kind))}/{_STACK_UUID}",
                    value,
                )
                is not None
            )

        if (
            self.finalization_state != "SUPPORT_FINALIZED"
            or self.support_stack_name != stack_name(StackKind.SUPPORT)
            or not exact_stack_id(StackKind.SUPPORT, self.support_stack_id)
            or not exact_stack_id(StackKind.RETAINED, self.retained_stack_id)
            or not exact_stack_id(StackKind.FENCE, self.fence_stack_id)
            or len(
                {
                    self.support_stack_id,
                    self.retained_stack_id,
                    self.fence_stack_id,
                }
            )
            != 3
            or self.deletion_role_arn != _SUPPORT_DELETION_ROLE_ARN
            or type(self.lifecycle_action_identity_sha256) is not str
            or _SHA256.fullmatch(self.lifecycle_action_identity_sha256) is None
            or type(self.activation_id) is not str
            or _SAFE_ID.fullmatch(self.activation_id) is None
            or self.action_key
            != ledger_sk(
                "glm52_production_finalization_action",
                activation_id=self.activation_id,
                action_kind="SUPPORT_DELETE",
                attempt=1,
            )
            or type(self.state_machine_version_arn) is not str
            or not self.state_machine_version_arn.startswith(
                f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:"
            )
            or type(self.reviewed_resources) is not tuple
            or len(self.reviewed_resources) != 173
            or any(
                type(resource) is not ReviewedSupportResource
                for resource in self.reviewed_resources
            )
            or len({resource.logical_id for resource in self.reviewed_resources})
            != len(self.reviewed_resources)
        ):
            raise ValueError("support deletion authority is not exact and finalized")
        for name, value in (
            ("owner_nonce_sha256", self.owner_nonce_sha256),
            ("request_evidence_sha256", self.request_evidence_sha256),
            ("reviewed_inventory_sha256", self.reviewed_inventory_sha256),
        ):
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} is not exact SHA-256")
        expected_inventory_sha256 = hashlib.sha256(
            canonical_json_bytes(
                [
                    {
                        "logical_id": resource.logical_id,
                        "resource_type": resource.resource_type,
                        "physical_id": resource.physical_id,
                        "source_api": resource.source_api,
                        "source_response_sha256": (
                            resource.source_response_sha256
                        ),
                    }
                    for resource in self.reviewed_resources
                ]
            )
        ).hexdigest()
        if expected_inventory_sha256 != self.reviewed_inventory_sha256:
            raise ValueError("reviewed support resource inventory digest mismatches")


@dataclass(frozen=True)
class SupportDeletionWorkflowExecution:
    """Authenticated identity of the one durable support-delete execution."""

    execution_arn: str
    input_sha256: str
    status: str
    adopted: bool


@dataclass(frozen=True)
class SupportDeletionWorkflowReconciliation:
    execution_arn: str
    history_pages: int
    history_events: int
    update_task_schedules: int
    delete_task_schedules: int
    terminal_absence_confirmed: bool


def support_deletion_workflow_history_event_ceiling() -> int:
    """Conservative 1-hour Standard history bound for 5-second readbacks."""

    fixed_events = 64
    maximum_readback_loops = 3600 // 5
    maximum_events_per_readback_loop = 11
    return fixed_events + (
        maximum_readback_loops * maximum_events_per_readback_loop
    )


def build_support_deletion_workflow_definition() -> Mapping[str, object]:
    """Build the retained Standard graph that durably owns both mutations.

    The two CloudFormation AWS SDK integrations are Step Functions Task
    states.  Neither has a Retry, and every error edge is read-only
    reconciliation or terminal failure.  A process-local adapter therefore
    never commits a possibly-sent marker and then calls CloudFormation.
    """

    return {
        "Comment": (
            "Exact-version retained H.1g support deletion; Standard "
            "exactly-once Task ownership and no mutation resubmission"
        ),
        "StartAt": "DescribeProtectedSupportStack",
        "TimeoutSeconds": 3600,
        "States": {
            "DescribeProtectedSupportStack": {
                "Type": "Task",
                "Resource": (
                    "arn:aws:states:::aws-sdk:cloudformation:describeStacks"
                ),
                "Parameters": {"StackName.$": "$.support_stack_id"},
                "ResultPath": "$.protected_stack",
                "Next": "RequireStableProtectedSupportStack",
            },
            "RequireStableProtectedSupportStack": {
                "Type": "Choice",
                "Choices": [
                    {
                        "And": [
                            {
                                "Variable": "$.protected_stack.Stacks[0].StackId",
                                "StringEqualsPath": "$.support_stack_id",
                            },
                            {
                                "Variable": (
                                    "$.protected_stack.Stacks[0].StackStatus"
                                ),
                                "StringEquals": "UPDATE_COMPLETE",
                            },
                            {
                                "Variable": (
                                    "$.protected_stack.Stacks[0]."
                                    "EnableTerminationProtection"
                                ),
                                "BooleanEquals": True,
                            },
                        ],
                        "Next": "DisableSupportTerminationProtection",
                    }
                ],
                "Default": "ProtectedSupportPrestateInvalid",
            },
            "DisableSupportTerminationProtection": {
                "Type": "Task",
                "Resource": (
                    "arn:aws:states:::aws-sdk:cloudformation:"
                    "updateTerminationProtection"
                ),
                "Parameters": {
                    "EnableTerminationProtection": False,
                    "StackName.$": "$.support_stack_id",
                },
                "ResultPath": "$.update_termination_protection",
                "Catch": [
                    {
                        "ErrorEquals": ["States.ALL"],
                        "ResultPath": "$.update_ambiguity",
                        "Next": "ReconcileTerminationProtectionFalse",
                    }
                ],
                "Next": "ReconcileTerminationProtectionFalse",
            },
            "ReconcileTerminationProtectionFalse": {
                "Type": "Task",
                "Resource": (
                    "arn:aws:states:::aws-sdk:cloudformation:describeStacks"
                ),
                "Parameters": {"StackName.$": "$.support_stack_id"},
                "ResultPath": "$.unprotected_stack",
                "Next": "RequireTerminationProtectionFalse",
            },
            "RequireTerminationProtectionFalse": {
                "Type": "Choice",
                "Choices": [
                    {
                        "And": [
                            {
                                "Variable": (
                                    "$.unprotected_stack.Stacks[0].StackId"
                                ),
                                "StringEqualsPath": "$.support_stack_id",
                            },
                            {
                                "Variable": (
                                    "$.unprotected_stack.Stacks[0]."
                                    "EnableTerminationProtection"
                                ),
                                "BooleanEquals": False,
                            },
                        ],
                        "Next": "DeleteSupportStack",
                    }
                ],
                "Default": "TerminationProtectionFalseUnresolved",
            },
            "DeleteSupportStack": {
                "Type": "Task",
                "Resource": (
                    "arn:aws:states:::aws-sdk:cloudformation:deleteStack"
                ),
                "Parameters": {
                    "RoleARN.$": "$.support_deletion_role_arn",
                    "StackName.$": "$.support_stack_id",
                },
                "ResultPath": "$.delete_stack",
                "Catch": [
                    {
                        "ErrorEquals": ["States.ALL"],
                        "ResultPath": "$.delete_ambiguity",
                        "Next": "WaitForSupportStackAbsence",
                    }
                ],
                "Next": "WaitForSupportStackAbsence",
            },
            "WaitForSupportStackAbsence": {
                "Type": "Wait",
                "Seconds": 5,
                "Next": "DescribeSupportStackForAbsence",
            },
            "DescribeSupportStackForAbsence": {
                "Type": "Task",
                "Resource": (
                    "arn:aws:states:::aws-sdk:cloudformation:describeStacks"
                ),
                "Parameters": {"StackName.$": "$.support_stack_id"},
                "ResultPath": "$.delete_readback",
                "Catch": [
                    {
                        "ErrorEquals": ["States.ALL"],
                        "ResultPath": "$.absence_candidate",
                        "Next": "RequireExactAbsentError",
                    }
                ],
                "Next": "RequireDeletionStillCoherent",
            },
            "RequireDeletionStillCoherent": {
                "Type": "Choice",
                "Choices": [
                    {
                        "And": [
                            {
                                "Variable": (
                                    "$.delete_readback.Stacks[0].StackId"
                                ),
                                "StringEqualsPath": "$.support_stack_id",
                            },
                            {
                                "Variable": (
                                    "$.delete_readback.Stacks[0].StackStatus"
                                ),
                                "StringEquals": "DELETE_IN_PROGRESS",
                            },
                        ],
                        "Next": "WaitForSupportStackAbsence",
                    }
                ],
                "Default": "SupportDeletionReadbackIncoherent",
            },
            "RequireExactAbsentError": {
                "Type": "Choice",
                "Choices": [
                    {
                        "And": [
                            {
                                "Variable": "$.absence_candidate.Error",
                                "StringEquals": (
                                    "CloudFormation.ValidationException"
                                ),
                            },
                            {
                                "Variable": "$.absence_candidate.Cause",
                                "StringMatches": "*does not exist*",
                            },
                        ],
                        "Next": "SupportStackAbsenceConfirmed",
                    }
                ],
                "Default": "SupportDeletionReadbackIncoherent",
            },
            "SupportStackAbsenceConfirmed": {"Type": "Succeed"},
            "ProtectedSupportPrestateInvalid": {
                "Type": "Fail",
                "Error": "ProtectedSupportPrestateInvalid",
            },
            "TerminationProtectionFalseUnresolved": {
                "Type": "Fail",
                "Error": "TerminationProtectionFalseUnresolved",
            },
            "SupportDeletionReadbackIncoherent": {
                "Type": "Fail",
                "Error": "SupportDeletionReadbackIncoherent",
            },
        },
    }


def _support_deletion_workflow_coordinates(
    authority: SupportDeletionAuthority,
) -> Tuple[str, str, str, str]:
    version_arn = authority.state_machine_version_arn
    marker = (
        f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:"
        "keep-glm52-h1g-retained-lifecycle:"
    )
    if (
        type(version_arn) is not str
        or not version_arn.startswith(marker)
        or not version_arn.removeprefix(marker).isdigit()
        or int(version_arn.removeprefix(marker)) <= 0
    ):
        raise ValueError("support deletion requires the exact retained version")
    body = {
        "schema_version": 1,
        "mode": "SUPPORT_DELETE",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": authority.activation_id,
        "support_stack_id": authority.support_stack_id,
        "support_deletion_role_arn": authority.deletion_role_arn,
        "finalization_action_identity_sha256": (
            authority.lifecycle_action_identity_sha256
        ),
        "request_evidence_sha256": authority.request_evidence_sha256,
        "reviewed_inventory_sha256": authority.reviewed_inventory_sha256,
    }
    input_bytes = canonical_json_bytes(body)
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    name = (
        f"h1g-delete-{authority.activation_id[:24]}-"
        f"{input_sha256[:16]}"
    )
    execution_arn = (
        f"arn:aws:states:{REGION}:{ACCOUNT_ID}:execution:"
        f"keep-glm52-h1g-retained-lifecycle:{name}"
    )
    return name, input_bytes.decode("ascii"), input_sha256, execution_arn


def _step_functions_error_code(exc: BaseException) -> Optional[str]:
    response = getattr(exc, "response", None)
    detail = response.get("Error") if type(response) is dict else None
    code = detail.get("Code") if type(detail) is dict else None
    return code if type(code) is str else None


def _authenticated_step_functions_error(
    exc: BaseException,
    *,
    code: str,
    operation: str,
) -> bool:
    if _step_functions_error_code(exc) != code:
        return False
    response = getattr(exc, "response", None)
    metadata = (
        response.get("ResponseMetadata")
        if type(response) is dict
        else None
    )
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") not in {400, 404}
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
    ):
        raise ValueError(
            operation + " error is not authenticated"
        ) from exc
    return True


def _execution_not_found(exc: Exception) -> bool:
    return _authenticated_step_functions_error(
        exc,
        code="ExecutionDoesNotExist",
        operation="DescribeExecution",
    )


def _execution_already_exists(exc: Exception) -> bool:
    return _authenticated_step_functions_error(
        exc,
        code="ExecutionAlreadyExists",
        operation="StartExecution",
    )


class SupportDeletionWorkflowAdapter:
    """Start or adopt the one exact retained Standard deletion execution."""

    def __init__(
        self,
        *,
        client: object,
        read_client: Optional[object] = None,
        finalization_reader: Optional[object] = None,
    ) -> None:
        self._client = client
        self._read_client = read_client
        self._finalization_reader = finalization_reader

    def _require_live_finalization(
        self, authority: SupportDeletionAuthority
    ) -> None:
        if self._finalization_reader is None:
            raise ValueError(
                "support deletion lacks live finalization readback"
            )
        read = _service_method(
            self._finalization_reader,
            "read_finalization_support_delete",
        )
        response = _success_response(
            read(
                LedgerPartitionKey=ledger_pk(RUN_ID),
                ActionKey=authority.action_key,
                ActivationId=authority.activation_id,
            ),
            "ReadFinalizationSupportDelete",
        )
        expected = {
            "FinalizationState": "SUPPORT_FINALIZED",
            "SupportStackId": authority.support_stack_id,
            "ActionKey": authority.action_key,
            "LifecycleActionIdentitySha256": (
                authority.lifecycle_action_identity_sha256
            ),
            "StateMachineVersionArn": (
                authority.state_machine_version_arn
            ),
            "OwnerNonceSha256": authority.owner_nonce_sha256,
            "RequestEvidenceSha256": authority.request_evidence_sha256,
            "ReviewedInventorySha256": (
                authority.reviewed_inventory_sha256
            ),
        }
        if any(response.get(key) != value for key, value in expected.items()):
            raise ValueError("live finalization support delete is foreign")

    def _describe(
        self,
        *,
        authority: SupportDeletionAuthority,
        execution_arn: str,
        expected_input: str,
    ) -> Optional[SupportDeletionWorkflowExecution]:
        describe = _service_method(self._client, "describe_execution")
        try:
            response = describe(executionArn=execution_arn)
        except Exception as exc:
            if _execution_not_found(exc):
                return None
            raise
        item = _success_response(
            response, "DescribeRetainedSupportDeletionExecution"
        )
        if item.get("exists") is False:
            return None
        status = item.get("status")
        if (
            item.get("executionArn") != execution_arn
            or item.get("input") != expected_input
            or item.get("stateMachineVersionArn")
            != authority.state_machine_version_arn
            or status
            not in {
                "RUNNING",
                "SUCCEEDED",
                "FAILED",
                "TIMED_OUT",
                "ABORTED",
            }
        ):
            raise ValueError("retained deletion execution identity drifted")
        return SupportDeletionWorkflowExecution(
            execution_arn=execution_arn,
            input_sha256=hashlib.sha256(
                expected_input.encode("ascii")
            ).hexdigest(),
            status=status,
            adopted=True,
        )

    def start_or_adopt(
        self, *, authority: SupportDeletionAuthority
    ) -> SupportDeletionWorkflowExecution:
        if type(authority) is not SupportDeletionAuthority:
            raise TypeError("support deletion requires exact authority")
        self._require_live_finalization(authority)
        name, input_text, input_sha256, execution_arn = (
            _support_deletion_workflow_coordinates(authority)
        )
        existing = self._describe(
            authority=authority,
            execution_arn=execution_arn,
            expected_input=input_text,
        )
        if existing is not None:
            return existing
        start = _service_method(self._client, "start_execution")
        request = {
            "stateMachineArn": authority.state_machine_version_arn,
            "name": name,
            "input": input_text,
        }
        try:
            response = start(**request)
        except Exception as exc:
            if (
                not isinstance(exc, AmbiguousTransportError)
                and not _execution_already_exists(exc)
            ):
                raise
            adopted = self._describe(
                authority=authority,
                execution_arn=execution_arn,
                expected_input=input_text,
            )
            if adopted is None:
                raise ValueError(
                    "ambiguous retained deletion start is unresolved"
                )
            return adopted
        item = _success_response(
            response, "StartRetainedSupportDeletionExecution"
        )
        if item.get("executionArn") != execution_arn:
            raise ValueError("retained deletion start returned a foreign ARN")
        return SupportDeletionWorkflowExecution(
            execution_arn=execution_arn,
            input_sha256=input_sha256,
            status="RUNNING",
            adopted=False,
        )

    def reconcile(
        self, *, authority: SupportDeletionAuthority
    ) -> SupportDeletionWorkflowReconciliation:
        """Validate one complete execution history and exact stack absence."""

        if type(authority) is not SupportDeletionAuthority:
            raise TypeError("support deletion requires exact authority")
        _, input_text, _, execution_arn = (
            _support_deletion_workflow_coordinates(authority)
        )
        execution = self._describe(
            authority=authority,
            execution_arn=execution_arn,
            expected_input=input_text,
        )
        if execution is None or execution.status != "SUCCEEDED":
            raise ValueError("retained deletion execution is not successful")
        get_history = _service_method(
            self._client, "get_execution_history"
        )
        events = []
        pages = 0
        next_token: Optional[str] = None
        while True:
            request: Dict[str, object] = {
                "executionArn": execution_arn,
                "includeExecutionData": True,
                "maxResults": 1000,
                "reverseOrder": False,
            }
            if next_token is not None:
                request["nextToken"] = next_token
            page = _success_response(
                get_history(**request),
                "GetRetainedSupportDeletionExecutionHistory",
            )
            page_events = page.get("events")
            if (
                type(page_events) is not list
                or any(type(event) is not dict for event in page_events)
            ):
                raise ValueError("retained deletion history page is malformed")
            events.extend(page_events)
            pages += 1
            if (
                pages > 12
                or len(events)
                > support_deletion_workflow_history_event_ceiling()
            ):
                raise ValueError("retained deletion history ceiling exceeded")
            token = page.get("nextToken")
            if token is None:
                break
            if type(token) is not str or not token:
                raise ValueError("retained deletion history token is malformed")
            next_token = token
        event_ids = [event.get("id") for event in events]
        if (
            not events
            or any(type(event_id) is not int for event_id in event_ids)
            or event_ids != sorted(event_ids)
            or len(set(event_ids)) != len(event_ids)
        ):
            raise ValueError("retained deletion history order is incoherent")
        rendered_events = [
            json.dumps(event, sort_keys=True, separators=(",", ":"))
            for event in events
        ]
        update_schedules = sum(
            event.get("type") == "TaskScheduled"
            and "updateTerminationProtection" in rendered
            for event, rendered in zip(events, rendered_events)
        )
        delete_schedules = sum(
            event.get("type") == "TaskScheduled"
            and "deleteStack" in rendered
            for event, rendered in zip(events, rendered_events)
        )
        terminal_absence = any(
            event.get("type") == "SucceedStateEntered"
            and type(event.get("stateEnteredEventDetails")) is dict
            and event["stateEnteredEventDetails"].get("name")
            == "SupportStackAbsenceConfirmed"
            for event in events
        )
        if (
            update_schedules != 1
            or delete_schedules != 1
            or not terminal_absence
            or events[-1].get("type") != "ExecutionSucceeded"
        ):
            raise ValueError(
                "retained deletion history does not prove one-shot absence"
            )
        if self._read_client is None:
            raise ValueError("support absence reconciliation client is absent")
        describe = _service_method(
            self._read_client, "describe_stacks"
        )
        try:
            describe(StackName=authority.support_stack_id)
        except Exception as exc:
            response = getattr(exc, "response", None)
            error = (
                response.get("Error")
                if type(response) is dict
                else None
            )
            if (
                type(error) is not dict
                or error.get("Code") != "ValidationError"
                or type(error.get("Message")) is not str
                or "does not exist" not in error["Message"]
                or authority.support_stack_id not in error["Message"]
            ):
                raise ValueError(
                    "support absence reconciliation failed"
                ) from exc
        else:
            raise ValueError("support stack remains present after deletion")
        audit = _service_method(
            self._read_client, "audit_support_resources"
        )
        _validate_support_residual_absence(
            response=audit(
                StackId=authority.support_stack_id,
                Resources=[
                    {
                        "LogicalResourceId": resource.logical_id,
                        "ResourceType": resource.resource_type,
                        "PhysicalResourceId": resource.physical_id,
                    }
                    for resource in authority.reviewed_resources
                ],
            ),
            authority=authority,
        )
        return SupportDeletionWorkflowReconciliation(
            execution_arn=execution_arn,
            history_pages=pages,
            history_events=len(events),
            update_task_schedules=update_schedules,
            delete_task_schedules=delete_schedules,
            terminal_absence_confirmed=True,
        )


SupportDeletionAdapter = SupportDeletionWorkflowAdapter


def derive_support_deletion_request_evidence_sha256(
    authority: SupportDeletionAuthority,
) -> str:
    if type(authority) is not SupportDeletionAuthority:
        raise TypeError("support deletion request requires exact authority")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "run_id": RUN_ID,
                "support_stack_id": authority.support_stack_id,
                "retained_stack_id": authority.retained_stack_id,
                "fence_stack_id": authority.fence_stack_id,
                "deletion_role_arn": authority.deletion_role_arn,
                "lifecycle_action_identity_sha256": (
                    authority.lifecycle_action_identity_sha256
                ),
                "reviewed_inventory_sha256": (
                    authority.reviewed_inventory_sha256
                ),
            }
        )
    ).hexdigest()












def _validate_support_residual_absence(
    *, response: object, authority: SupportDeletionAuthority
) -> None:
    item = _success_response(response, "AuditSupportResources")
    expected_resources = [
        {
            "LogicalResourceId": resource.logical_id,
            "ResourceType": resource.resource_type,
            "PhysicalResourceId": resource.physical_id,
            "Status": "ABSENT",
        }
        for resource in authority.reviewed_resources
    ]
    if (
        item.get("StackId") != authority.support_stack_id
        or type(item.get("ObservationId")) is not str
        or not item["ObservationId"]
        or item.get("Resources") != expected_resources
    ):
        raise ValueError("reviewed support resource absence is not exact")









def _service_method(service: object, name: str) -> object:
    try:
        method = getattr(service, name)
    except Exception as exc:
        raise ValueError(f"fresh authority service lacks {name}") from exc
    if not callable(method):
        raise ValueError(f"fresh authority service lacks {name}")
    return method




def _success_response(response: object, operation: str) -> Mapping[str, object]:
    if type(response) is not dict:
        raise ValueError(f"{operation} response is malformed")
    metadata = response.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
    ):
        raise ValueError(f"{operation} response metadata is not exact success")
    return response




# ---------------------------------------------------------------------------
# H.1g v2 execution boundary
# ---------------------------------------------------------------------------
#
# Historical Task 11 fence evidence models above remain available for audit
# decoding only.  The sole public executor below accepts closed v2 artifacts.




class FenceExecutorV2Error(ValueError):
    """A v2 transition fact is not exact or cannot authorize execution."""


_V2_ROLE_ID = re.compile(r"[A-Z0-9]{16,128}\Z")
_V2_CHANGE_SET_ARN = re.compile(
    rf"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:changeSet/"
    rf"[A-Za-z][-A-Za-z0-9]{{0,127}}/{_STACK_UUID}\Z"
)
_V2_TIME = re.compile(
    r"20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z\Z"
)
_V2_AUTHORITY_ROLE_FIELDS = frozenset(
    {
        "role_arn",
        "role_id",
        "trust_policy_sha256",
        "permission_policy_sha256",
    }
)
_V2_EXECUTOR_ROW_FIELDS = frozenset(
    {
        "authority_class",
        "api_caller_role",
        "cloudformation_service_role",
    }
)
_V2_TERMINAL_CHANGE_SET_STATUSES = frozenset({"DELETE_COMPLETE", "FAILED"})
_V2_SELECTED_CHANGE_SET_STATUS = ("CREATE_COMPLETE", "AVAILABLE")
_V2_RECORD_TYPES = {
    "create_authority": "glm52_fence_create_authority_v2",
    "execute_authority": "glm52_fence_execute_authority_v2",
    "prestate_create": "glm52_fence_create_prestate_v2",
    "prestate_execute": "glm52_fence_execute_prestate_v2",
    "prepared": "glm52_prepared_fence_change_set_v2",
    "change_set_evidence": "glm52_fence_change_set_evidence_v2",
    "change_set_list": "glm52_fence_change_set_list_digest_v2",
    "execution_result": "glm52_fence_execution_result_v2",
    "freeze_delta_audit": "glm52_freeze_execute_delta_audit_v2",
}


def _v2_fail(message: str) -> None:
    raise FenceExecutorV2Error(message)


def _v2_text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        _v2_fail(name + " is not exact text")
    return value


def _v2_sha(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _v2_fail(name + " is not exact SHA-256")
    return value


def _v2_role_arn(value: object, name: str) -> str:
    if type(value) is not str or _ROLE_ARN.fullmatch(value) is None:
        _v2_fail(name + " is not an exact same-account role ARN")
    return value


def _v2_role_id(value: object, name: str) -> str:
    if type(value) is not str or _V2_ROLE_ID.fullmatch(value) is None:
        _v2_fail(name + " is not an exact RoleId")
    return value


def _v2_time(value: object, name: str) -> str:
    if type(value) is not str or _V2_TIME.fullmatch(value) is None:
        _v2_fail(name + " is not an exact UTC observation time")
    return value


def _v2_exact_mapping(
    value: object, fields: Iterable[str], name: str
) -> Mapping[str, object]:
    expected = frozenset(fields)
    if type(value) is not dict or frozenset(value) != expected:
        _v2_fail(name + " fields are not exact")
    return value


def _v2_deep_freeze(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {
                key: _v2_deep_freeze(item)
                for key, item in value.items()
            }
        )
    if type(value) in (list, tuple):
        return tuple(_v2_deep_freeze(item) for item in value)
    if value is None or type(value) in (str, int, bool):
        return value
    _v2_fail("record contains a non-canonical value")


def _v2_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _v2_thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_v2_thaw(item) for item in value]
    return value


def _v2_canonical_sha256(value: object) -> str:
    try:
        return hashlib.sha256(canonical_json_bytes(_v2_thaw(value))).hexdigest()
    except (TypeError, ValueError, UnicodeError) as exc:
        raise FenceExecutorV2Error("record is not canonical JSON") from exc


def _v2_validate_identity(
    value: Mapping[str, object], *, name: str
) -> None:
    identity = _v2_sha(value.get("canonical_identity_sha256"), name)
    unsigned = dict(value)
    unsigned.pop("canonical_identity_sha256")
    if _v2_canonical_sha256(unsigned) != identity:
        _v2_fail(name + " canonical identity drifted")


def _v2_authority_role(value: object, name: str) -> Mapping[str, object]:
    role = _v2_exact_mapping(value, _V2_AUTHORITY_ROLE_FIELDS, name)
    _v2_role_arn(role["role_arn"], name + ".role_arn")
    _v2_role_id(role["role_id"], name + ".role_id")
    _v2_sha(role["trust_policy_sha256"], name + ".trust_policy_sha256")
    _v2_sha(
        role["permission_policy_sha256"],
        name + ".permission_policy_sha256",
    )
    return role


def _v2_manifest_executor_row(
    manifest: FenceManifest,
    authority_class: ExecutorAuthorityClass,
) -> Mapping[str, object]:
    value = manifest.to_dict()
    rows = value.get("executor_inventory")
    if type(rows) not in (list, tuple):
        _v2_fail("manifest executor_inventory is not exact")
    matches = [
        row
        for row in rows
        if type(row) is dict
        and row.get("authority_class") == authority_class.value
    ]
    if len(matches) != 1:
        _v2_fail("manifest executor authority class is not singular")
    row = _v2_exact_mapping(
        matches[0], _V2_EXECUTOR_ROW_FIELDS, "executor_inventory row"
    )
    _v2_authority_role(row["api_caller_role"], "api_caller_role")
    service = _v2_authority_role(
        row["cloudformation_service_role"],
        "cloudformation_service_role",
    )
    fence_role = value.get("fence_service_role")
    if (
        type(fence_role) is not dict
        or fence_role.get("arn") != service["role_arn"]
    ):
        _v2_fail(
            "manifest FenceServiceRole is not the executor CloudFormation role"
        )
    return row


@dataclass(frozen=True)
class FenceCreateAuthorityV2:
    request_identity_sha256: str
    authority_class: ExecutorAuthorityClass
    api_caller_role_arn: str
    api_caller_role_id: str
    api_caller_trust_policy_sha256: str
    api_caller_permission_policy_sha256: str
    cloudformation_service_role_arn: str
    cloudformation_service_role_id: str
    cloudformation_service_role_trust_policy_sha256: str
    cloudformation_service_role_permission_policy_sha256: str
    action_key: str
    authorized_revision: int
    issued_at: str
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        _v2_validate_create_authority_fields(self)

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "record_type": _V2_RECORD_TYPES["create_authority"],
            "request_identity_sha256": self.request_identity_sha256,
            "authority_class": self.authority_class.value,
            "api_caller_role_arn": self.api_caller_role_arn,
            "api_caller_role_id": self.api_caller_role_id,
            "api_caller_trust_policy_sha256": (
                self.api_caller_trust_policy_sha256
            ),
            "api_caller_permission_policy_sha256": (
                self.api_caller_permission_policy_sha256
            ),
            "cloudformation_service_role_arn": (
                self.cloudformation_service_role_arn
            ),
            "cloudformation_service_role_id": (
                self.cloudformation_service_role_id
            ),
            "cloudformation_service_role_trust_policy_sha256": (
                self.cloudformation_service_role_trust_policy_sha256
            ),
            "cloudformation_service_role_permission_policy_sha256": (
                self.cloudformation_service_role_permission_policy_sha256
            ),
            "action_key": self.action_key,
            "authorized_revision": self.authorized_revision,
            "issued_at": self.issued_at,
            "canonical_identity_sha256": self.canonical_identity_sha256,
        }


@dataclass(frozen=True)
class FenceExecuteAuthorityV2:
    request_identity_sha256: str
    prepared_identity_sha256: str
    authority_class: ExecutorAuthorityClass
    api_caller_role_arn: str
    api_caller_role_id: str
    api_caller_trust_policy_sha256: str
    api_caller_permission_policy_sha256: str
    cloudformation_service_role_arn: str
    cloudformation_service_role_id: str
    cloudformation_service_role_trust_policy_sha256: str
    cloudformation_service_role_permission_policy_sha256: str
    action_key: str
    authorized_revision: int
    issued_at: str
    retained_failover: bool
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        _v2_validate_execute_authority_fields(self)

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "record_type": _V2_RECORD_TYPES["execute_authority"],
            "request_identity_sha256": self.request_identity_sha256,
            "prepared_identity_sha256": self.prepared_identity_sha256,
            "authority_class": self.authority_class.value,
            "api_caller_role_arn": self.api_caller_role_arn,
            "api_caller_role_id": self.api_caller_role_id,
            "api_caller_trust_policy_sha256": (
                self.api_caller_trust_policy_sha256
            ),
            "api_caller_permission_policy_sha256": (
                self.api_caller_permission_policy_sha256
            ),
            "cloudformation_service_role_arn": (
                self.cloudformation_service_role_arn
            ),
            "cloudformation_service_role_id": (
                self.cloudformation_service_role_id
            ),
            "cloudformation_service_role_trust_policy_sha256": (
                self.cloudformation_service_role_trust_policy_sha256
            ),
            "cloudformation_service_role_permission_policy_sha256": (
                self.cloudformation_service_role_permission_policy_sha256
            ),
            "action_key": self.action_key,
            "authorized_revision": self.authorized_revision,
            "issued_at": self.issued_at,
            "retained_failover": self.retained_failover,
            "canonical_identity_sha256": self.canonical_identity_sha256,
        }


def _v2_validate_common_authority_fields(value: object) -> None:
    _v2_sha(
        getattr(value, "request_identity_sha256", None),
        "authority.request_identity_sha256",
    )
    if type(getattr(value, "authority_class", None)) is not ExecutorAuthorityClass:
        _v2_fail("authority_class is not exact")
    _v2_role_arn(
        getattr(value, "api_caller_role_arn", None),
        "authority.api_caller_role_arn",
    )
    _v2_role_id(
        getattr(value, "api_caller_role_id", None),
        "authority.api_caller_role_id",
    )
    _v2_role_arn(
        getattr(value, "cloudformation_service_role_arn", None),
        "authority.cloudformation_service_role_arn",
    )
    _v2_role_id(
        getattr(value, "cloudformation_service_role_id", None),
        "authority.cloudformation_service_role_id",
    )
    for field in (
        "api_caller_trust_policy_sha256",
        "api_caller_permission_policy_sha256",
        "cloudformation_service_role_trust_policy_sha256",
        "cloudformation_service_role_permission_policy_sha256",
    ):
        _v2_sha(getattr(value, field, None), "authority." + field)
    _v2_text(getattr(value, "action_key", None), "authority.action_key")
    revision = getattr(value, "authorized_revision", None)
    if type(revision) is not int or revision < 1:
        _v2_fail("authority.authorized_revision is not exact")
    _v2_time(getattr(value, "issued_at", None), "authority.issued_at")


def _v2_validate_create_authority_fields(
    value: FenceCreateAuthorityV2,
) -> None:
    _v2_validate_common_authority_fields(value)
    _v2_validate_identity(value.to_dict(), name="create authority")


def _v2_validate_execute_authority_fields(
    value: FenceExecuteAuthorityV2,
) -> None:
    _v2_validate_common_authority_fields(value)
    _v2_sha(
        value.prepared_identity_sha256,
        "execute authority prepared_identity_sha256",
    )
    if type(value.retained_failover) is not bool:
        _v2_fail("execute authority retained_failover is not exact")
    _v2_validate_identity(value.to_dict(), name="execute authority")


_V2_CREATE_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "request_identity_sha256",
        "authority_class",
        "api_caller_role_arn",
        "api_caller_role_id",
        "api_caller_trust_policy_sha256",
        "api_caller_permission_policy_sha256",
        "cloudformation_service_role_arn",
        "cloudformation_service_role_id",
        "cloudformation_service_role_trust_policy_sha256",
        "cloudformation_service_role_permission_policy_sha256",
        "action_key",
        "authorized_revision",
        "issued_at",
        "canonical_identity_sha256",
    }
)
_V2_EXECUTE_AUTHORITY_FIELDS = _V2_CREATE_AUTHORITY_FIELDS | frozenset(
    {"prepared_identity_sha256", "retained_failover"}
)


def parse_fence_create_authority(
    value: object,
) -> FenceCreateAuthorityV2:
    item = _v2_exact_mapping(
        value, _V2_CREATE_AUTHORITY_FIELDS, "create authority"
    )
    if (
        item["schema_version"] != 2
        or item["record_type"] != _V2_RECORD_TYPES["create_authority"]
    ):
        _v2_fail("create authority v1/non-v2 is execution-ineligible")
    try:
        authority_class = ExecutorAuthorityClass(item["authority_class"])
    except (TypeError, ValueError) as exc:
        raise FenceExecutorV2Error("create authority class is foreign") from exc
    return FenceCreateAuthorityV2(
        request_identity_sha256=item["request_identity_sha256"],  # type: ignore[arg-type]
        authority_class=authority_class,
        api_caller_role_arn=item["api_caller_role_arn"],  # type: ignore[arg-type]
        api_caller_role_id=item["api_caller_role_id"],  # type: ignore[arg-type]
        api_caller_trust_policy_sha256=item[
            "api_caller_trust_policy_sha256"
        ],  # type: ignore[arg-type]
        api_caller_permission_policy_sha256=item[
            "api_caller_permission_policy_sha256"
        ],  # type: ignore[arg-type]
        cloudformation_service_role_arn=item[
            "cloudformation_service_role_arn"
        ],  # type: ignore[arg-type]
        cloudformation_service_role_id=item[
            "cloudformation_service_role_id"
        ],  # type: ignore[arg-type]
        cloudformation_service_role_trust_policy_sha256=item[
            "cloudformation_service_role_trust_policy_sha256"
        ],  # type: ignore[arg-type]
        cloudformation_service_role_permission_policy_sha256=item[
            "cloudformation_service_role_permission_policy_sha256"
        ],  # type: ignore[arg-type]
        action_key=item["action_key"],  # type: ignore[arg-type]
        authorized_revision=item["authorized_revision"],  # type: ignore[arg-type]
        issued_at=item["issued_at"],  # type: ignore[arg-type]
        canonical_identity_sha256=item[
            "canonical_identity_sha256"
        ],  # type: ignore[arg-type]
    )


def parse_fence_execute_authority(
    value: object,
) -> FenceExecuteAuthorityV2:
    item = _v2_exact_mapping(
        value, _V2_EXECUTE_AUTHORITY_FIELDS, "execute authority"
    )
    if (
        item["schema_version"] != 2
        or item["record_type"] != _V2_RECORD_TYPES["execute_authority"]
    ):
        _v2_fail("execute authority v1/non-v2 is execution-ineligible")
    try:
        authority_class = ExecutorAuthorityClass(item["authority_class"])
    except (TypeError, ValueError) as exc:
        raise FenceExecutorV2Error("execute authority class is foreign") from exc
    return FenceExecuteAuthorityV2(
        request_identity_sha256=item["request_identity_sha256"],  # type: ignore[arg-type]
        prepared_identity_sha256=item[
            "prepared_identity_sha256"
        ],  # type: ignore[arg-type]
        authority_class=authority_class,
        api_caller_role_arn=item["api_caller_role_arn"],  # type: ignore[arg-type]
        api_caller_role_id=item["api_caller_role_id"],  # type: ignore[arg-type]
        api_caller_trust_policy_sha256=item[
            "api_caller_trust_policy_sha256"
        ],  # type: ignore[arg-type]
        api_caller_permission_policy_sha256=item[
            "api_caller_permission_policy_sha256"
        ],  # type: ignore[arg-type]
        cloudformation_service_role_arn=item[
            "cloudformation_service_role_arn"
        ],  # type: ignore[arg-type]
        cloudformation_service_role_id=item[
            "cloudformation_service_role_id"
        ],  # type: ignore[arg-type]
        cloudformation_service_role_trust_policy_sha256=item[
            "cloudformation_service_role_trust_policy_sha256"
        ],  # type: ignore[arg-type]
        cloudformation_service_role_permission_policy_sha256=item[
            "cloudformation_service_role_permission_policy_sha256"
        ],  # type: ignore[arg-type]
        action_key=item["action_key"],  # type: ignore[arg-type]
        authorized_revision=item["authorized_revision"],  # type: ignore[arg-type]
        issued_at=item["issued_at"],  # type: ignore[arg-type]
        retained_failover=item["retained_failover"],  # type: ignore[arg-type]
        canonical_identity_sha256=item[
            "canonical_identity_sha256"
        ],  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class FencePrestateSnapshot:
    request_identity_sha256: str
    manifest_identity_sha256: str
    entry_identity_sha256: str
    phase: str
    stack_id: str
    logical_id: str
    stack_role_arn: str
    stack_role_id: str
    termination_protection: bool
    stack_policy_sha256: str
    original_template_body_sha256: str
    processed_template_body_sha256: str
    direct_policy_sha256: str
    bucket_control_plane_identity_sha256: str
    role_inventory_identity_sha256: str
    h1f_observation_identity_sha256: str
    observed_at: str
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "request_identity_sha256",
            "manifest_identity_sha256",
            "entry_identity_sha256",
            "stack_policy_sha256",
            "original_template_body_sha256",
            "processed_template_body_sha256",
            "direct_policy_sha256",
            "bucket_control_plane_identity_sha256",
            "role_inventory_identity_sha256",
            "h1f_observation_identity_sha256",
        ):
            _v2_sha(getattr(self, field), "prestate." + field)
        if self.phase not in {"CREATE", "EXECUTE"}:
            _v2_fail("prestate phase is not exact")
        if _STACK_ID.fullmatch(self.stack_id) is None:
            _v2_fail("prestate stack_id is not exact")
        if self.logical_id != FENCE_LOGICAL_ID:
            _v2_fail("prestate logical_id is not exact")
        _v2_role_arn(self.stack_role_arn, "prestate.stack_role_arn")
        _v2_role_id(self.stack_role_id, "prestate.stack_role_id")
        if self.termination_protection is not True:
            _v2_fail("fence termination protection is not enabled")
        _v2_time(self.observed_at, "prestate.observed_at")
        _v2_validate_identity(self.to_dict(), name="fence prestate")

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "record_type": _V2_RECORD_TYPES[
                "prestate_" + self.phase.lower()
            ],
            "request_identity_sha256": self.request_identity_sha256,
            "manifest_identity_sha256": self.manifest_identity_sha256,
            "entry_identity_sha256": self.entry_identity_sha256,
            "phase": self.phase,
            "stack_id": self.stack_id,
            "logical_id": self.logical_id,
            "stack_role_arn": self.stack_role_arn,
            "stack_role_id": self.stack_role_id,
            "termination_protection": self.termination_protection,
            "stack_policy_sha256": self.stack_policy_sha256,
            "original_template_body_sha256": (
                self.original_template_body_sha256
            ),
            "processed_template_body_sha256": (
                self.processed_template_body_sha256
            ),
            "direct_policy_sha256": self.direct_policy_sha256,
            "bucket_control_plane_identity_sha256": (
                self.bucket_control_plane_identity_sha256
            ),
            "role_inventory_identity_sha256": (
                self.role_inventory_identity_sha256
            ),
            "h1f_observation_identity_sha256": (
                self.h1f_observation_identity_sha256
            ),
            "observed_at": self.observed_at,
            "canonical_identity_sha256": self.canonical_identity_sha256,
        }


_V2_PRESTATE_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "request_identity_sha256",
        "manifest_identity_sha256",
        "entry_identity_sha256",
        "phase",
        "stack_id",
        "logical_id",
        "stack_role_arn",
        "stack_role_id",
        "termination_protection",
        "stack_policy_sha256",
        "original_template_body_sha256",
        "processed_template_body_sha256",
        "direct_policy_sha256",
        "bucket_control_plane_identity_sha256",
        "role_inventory_identity_sha256",
        "h1f_observation_identity_sha256",
        "observed_at",
        "canonical_identity_sha256",
    }
)


def parse_fence_prestate_snapshot(value: object) -> FencePrestateSnapshot:
    item = _v2_exact_mapping(value, _V2_PRESTATE_FIELDS, "fence prestate")
    phase = item["phase"]
    if phase not in {"CREATE", "EXECUTE"}:
        _v2_fail("prestate phase is not exact")
    if (
        item["schema_version"] != 2
        or item["record_type"]
        != _V2_RECORD_TYPES["prestate_" + str(phase).lower()]
    ):
        _v2_fail("prestate v1/non-v2 is execution-ineligible")
    return FencePrestateSnapshot(
        request_identity_sha256=item["request_identity_sha256"],  # type: ignore[arg-type]
        manifest_identity_sha256=item["manifest_identity_sha256"],  # type: ignore[arg-type]
        entry_identity_sha256=item["entry_identity_sha256"],  # type: ignore[arg-type]
        phase=phase,  # type: ignore[arg-type]
        stack_id=item["stack_id"],  # type: ignore[arg-type]
        logical_id=item["logical_id"],  # type: ignore[arg-type]
        stack_role_arn=item["stack_role_arn"],  # type: ignore[arg-type]
        stack_role_id=item["stack_role_id"],  # type: ignore[arg-type]
        termination_protection=item["termination_protection"],  # type: ignore[arg-type]
        stack_policy_sha256=item["stack_policy_sha256"],  # type: ignore[arg-type]
        original_template_body_sha256=item[
            "original_template_body_sha256"
        ],  # type: ignore[arg-type]
        processed_template_body_sha256=item[
            "processed_template_body_sha256"
        ],  # type: ignore[arg-type]
        direct_policy_sha256=item["direct_policy_sha256"],  # type: ignore[arg-type]
        bucket_control_plane_identity_sha256=item[
            "bucket_control_plane_identity_sha256"
        ],  # type: ignore[arg-type]
        role_inventory_identity_sha256=item[
            "role_inventory_identity_sha256"
        ],  # type: ignore[arg-type]
        h1f_observation_identity_sha256=item[
            "h1f_observation_identity_sha256"
        ],  # type: ignore[arg-type]
        observed_at=item["observed_at"],  # type: ignore[arg-type]
        canonical_identity_sha256=item[
            "canonical_identity_sha256"
        ],  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class FreezeExecuteDeltaAudit:
    request_identity_sha256: str
    prepared_identity_sha256: str
    required_equal_before_sha256: str
    required_equal_after_sha256: str
    expected_source_delta_sha256: str
    quarantined_delta_sha256: Optional[str]
    classification: str
    batch_execution_eligible: bool
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "request_identity_sha256",
            "prepared_identity_sha256",
            "required_equal_before_sha256",
            "required_equal_after_sha256",
            "expected_source_delta_sha256",
        ):
            _v2_sha(getattr(self, field), "freeze delta." + field)
        if self.required_equal_before_sha256 != self.required_equal_after_sha256:
            _v2_fail("freeze required-equal state drifted")
        if self.quarantined_delta_sha256 is not None:
            _v2_sha(
                self.quarantined_delta_sha256,
                "freeze delta.quarantined_delta_sha256",
            )
        if self.classification not in {
            "EXACT",
            "FREEZE_EXPECTED_SOURCE_DELTA",
            "FREEZE_QUARANTINED_DRIFT",
        }:
            _v2_fail("freeze delta classification is foreign")
        if type(self.batch_execution_eligible) is not bool:
            _v2_fail("freeze batch eligibility is not exact")
        if (
            self.classification == "FREEZE_QUARANTINED_DRIFT"
            and (
                self.quarantined_delta_sha256 is None
                or self.batch_execution_eligible
            )
        ):
            _v2_fail("quarantined freeze drift did not permanently exclude BATCH")
        if (
            self.classification != "FREEZE_QUARANTINED_DRIFT"
            and self.quarantined_delta_sha256 is not None
        ):
            _v2_fail("non-quarantined freeze audit carries quarantined drift")
        if (
            self.classification != "FREEZE_QUARANTINED_DRIFT"
            and not self.batch_execution_eligible
        ):
            _v2_fail("expected/exact freeze delta unexpectedly excluded BATCH")
        _v2_validate_identity(self.to_dict(), name="freeze delta audit")

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "record_type": "glm52_freeze_execute_delta_audit_v2",
            "request_identity_sha256": self.request_identity_sha256,
            "prepared_identity_sha256": self.prepared_identity_sha256,
            "required_equal_before_sha256": (
                self.required_equal_before_sha256
            ),
            "required_equal_after_sha256": self.required_equal_after_sha256,
            "expected_source_delta_sha256": (
                self.expected_source_delta_sha256
            ),
            "quarantined_delta_sha256": self.quarantined_delta_sha256,
            "classification": self.classification,
            "batch_execution_eligible": self.batch_execution_eligible,
            "canonical_identity_sha256": self.canonical_identity_sha256,
        }


def parse_freeze_execute_delta_audit(
    value: object,
) -> FreezeExecuteDeltaAudit:
    fields = {
        "schema_version",
        "record_type",
        "request_identity_sha256",
        "prepared_identity_sha256",
        "required_equal_before_sha256",
        "required_equal_after_sha256",
        "expected_source_delta_sha256",
        "quarantined_delta_sha256",
        "classification",
        "batch_execution_eligible",
        "canonical_identity_sha256",
    }
    item = _v2_exact_mapping(value, fields, "freeze delta audit")
    if (
        item["schema_version"] != 2
        or item["record_type"] != "glm52_freeze_execute_delta_audit_v2"
    ):
        _v2_fail("freeze delta audit v1/non-v2 is execution-ineligible")
    return FreezeExecuteDeltaAudit(
        request_identity_sha256=item["request_identity_sha256"],  # type: ignore[arg-type]
        prepared_identity_sha256=item["prepared_identity_sha256"],  # type: ignore[arg-type]
        required_equal_before_sha256=item[
            "required_equal_before_sha256"
        ],  # type: ignore[arg-type]
        required_equal_after_sha256=item[
            "required_equal_after_sha256"
        ],  # type: ignore[arg-type]
        expected_source_delta_sha256=item[
            "expected_source_delta_sha256"
        ],  # type: ignore[arg-type]
        quarantined_delta_sha256=item[
            "quarantined_delta_sha256"
        ],  # type: ignore[arg-type]
        classification=item["classification"],  # type: ignore[arg-type]
        batch_execution_eligible=item[
            "batch_execution_eligible"
        ],  # type: ignore[arg-type]
        canonical_identity_sha256=item[
            "canonical_identity_sha256"
        ],  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class PreparedFenceChangeSet:
    request_identity_sha256: str
    manifest_identity_sha256: str
    entry_identity_sha256: str
    prestate_identity_sha256: str
    request_skeleton_sha256: str
    change_set_arn: str
    stack_id: str
    slot: FenceSlot
    create_authority_identity_sha256: str
    create_authority_class: ExecutorAuthorityClass
    create_api_caller_role_arn: str
    create_api_caller_role_id: str
    cloudformation_service_role_arn: str
    cloudformation_service_role_id: str
    template_sha256: str
    template_body_sha256: str
    policy_sha256: str
    expected_prestate_stack_role_arn: str
    expected_poststate_stack_role_arn: str
    selected_change_set_description_sha256: str
    created_at: str
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "request_identity_sha256",
            "manifest_identity_sha256",
            "entry_identity_sha256",
            "prestate_identity_sha256",
            "request_skeleton_sha256",
            "create_authority_identity_sha256",
            "template_sha256",
            "template_body_sha256",
            "policy_sha256",
            "selected_change_set_description_sha256",
        ):
            _v2_sha(getattr(self, field), "prepared." + field)
        if _V2_CHANGE_SET_ARN.fullmatch(self.change_set_arn) is None:
            _v2_fail("prepared change_set_arn is not service-assigned")
        if _STACK_ID.fullmatch(self.stack_id) is None:
            _v2_fail("prepared stack_id is not exact")
        if type(self.slot) is not FenceSlot:
            _v2_fail("prepared slot is not exact")
        if type(self.create_authority_class) is not ExecutorAuthorityClass:
            _v2_fail("prepared create authority class is not exact")
        _v2_role_arn(
            self.create_api_caller_role_arn,
            "prepared.create_api_caller_role_arn",
        )
        _v2_role_id(
            self.create_api_caller_role_id,
            "prepared.create_api_caller_role_id",
        )
        _v2_role_arn(
            self.cloudformation_service_role_arn,
            "prepared.cloudformation_service_role_arn",
        )
        _v2_role_id(
            self.cloudformation_service_role_id,
            "prepared.cloudformation_service_role_id",
        )
        _v2_role_arn(
            self.expected_prestate_stack_role_arn,
            "prepared.expected_prestate_stack_role_arn",
        )
        _v2_role_arn(
            self.expected_poststate_stack_role_arn,
            "prepared.expected_poststate_stack_role_arn",
        )
        if (
            self.cloudformation_service_role_arn
            != self.expected_poststate_stack_role_arn
        ):
            _v2_fail("prepared skeleton does not use FenceServiceRole")
        if self.slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE:
            if (
                self.expected_prestate_stack_role_arn
                == self.expected_poststate_stack_role_arn
            ):
                _v2_fail("PREPARE does not perform the sole role cutover")
        elif (
            self.expected_prestate_stack_role_arn
            != self.expected_poststate_stack_role_arn
        ):
            _v2_fail("post-PREPARE slot attempts a stack-role transition")
        _v2_time(self.created_at, "prepared.created_at")
        _v2_validate_identity(self.to_dict(), name="prepared change set")

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "record_type": _V2_RECORD_TYPES["prepared"],
            "request_identity_sha256": self.request_identity_sha256,
            "manifest_identity_sha256": self.manifest_identity_sha256,
            "entry_identity_sha256": self.entry_identity_sha256,
            "prestate_identity_sha256": self.prestate_identity_sha256,
            "request_skeleton_sha256": self.request_skeleton_sha256,
            "change_set_arn": self.change_set_arn,
            "stack_id": self.stack_id,
            "slot": self.slot.value,
            "create_authority_identity_sha256": (
                self.create_authority_identity_sha256
            ),
            "create_authority_class": self.create_authority_class.value,
            "create_api_caller_role_arn": self.create_api_caller_role_arn,
            "create_api_caller_role_id": self.create_api_caller_role_id,
            "cloudformation_service_role_arn": (
                self.cloudformation_service_role_arn
            ),
            "cloudformation_service_role_id": (
                self.cloudformation_service_role_id
            ),
            "template_sha256": self.template_sha256,
            "template_body_sha256": self.template_body_sha256,
            "policy_sha256": self.policy_sha256,
            "expected_prestate_stack_role_arn": (
                self.expected_prestate_stack_role_arn
            ),
            "expected_poststate_stack_role_arn": (
                self.expected_poststate_stack_role_arn
            ),
            "selected_change_set_description_sha256": (
                self.selected_change_set_description_sha256
            ),
            "created_at": self.created_at,
            "canonical_identity_sha256": self.canonical_identity_sha256,
        }


_V2_PREPARED_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "request_identity_sha256",
        "manifest_identity_sha256",
        "entry_identity_sha256",
        "prestate_identity_sha256",
        "request_skeleton_sha256",
        "change_set_arn",
        "stack_id",
        "slot",
        "create_authority_identity_sha256",
        "create_authority_class",
        "create_api_caller_role_arn",
        "create_api_caller_role_id",
        "cloudformation_service_role_arn",
        "cloudformation_service_role_id",
        "template_sha256",
        "template_body_sha256",
        "policy_sha256",
        "expected_prestate_stack_role_arn",
        "expected_poststate_stack_role_arn",
        "selected_change_set_description_sha256",
        "created_at",
        "canonical_identity_sha256",
    }
)


def parse_prepared_fence_change_set(value: object) -> PreparedFenceChangeSet:
    item = _v2_exact_mapping(
        value, _V2_PREPARED_FIELDS, "prepared change set"
    )
    if (
        item["schema_version"] != 2
        or item["record_type"] != _V2_RECORD_TYPES["prepared"]
    ):
        _v2_fail("prepared change set v1/non-v2 is execution-ineligible")
    try:
        slot = FenceSlot(item["slot"])
        authority_class = ExecutorAuthorityClass(
            item["create_authority_class"]
        )
    except (TypeError, ValueError) as exc:
        raise FenceExecutorV2Error("prepared enum is foreign") from exc
    return PreparedFenceChangeSet(
        request_identity_sha256=item["request_identity_sha256"],  # type: ignore[arg-type]
        manifest_identity_sha256=item["manifest_identity_sha256"],  # type: ignore[arg-type]
        entry_identity_sha256=item["entry_identity_sha256"],  # type: ignore[arg-type]
        prestate_identity_sha256=item["prestate_identity_sha256"],  # type: ignore[arg-type]
        request_skeleton_sha256=item["request_skeleton_sha256"],  # type: ignore[arg-type]
        change_set_arn=item["change_set_arn"],  # type: ignore[arg-type]
        stack_id=item["stack_id"],  # type: ignore[arg-type]
        slot=slot,
        create_authority_identity_sha256=item[
            "create_authority_identity_sha256"
        ],  # type: ignore[arg-type]
        create_authority_class=authority_class,
        create_api_caller_role_arn=item[
            "create_api_caller_role_arn"
        ],  # type: ignore[arg-type]
        create_api_caller_role_id=item[
            "create_api_caller_role_id"
        ],  # type: ignore[arg-type]
        cloudformation_service_role_arn=item[
            "cloudformation_service_role_arn"
        ],  # type: ignore[arg-type]
        cloudformation_service_role_id=item[
            "cloudformation_service_role_id"
        ],  # type: ignore[arg-type]
        template_sha256=item["template_sha256"],  # type: ignore[arg-type]
        template_body_sha256=item["template_body_sha256"],  # type: ignore[arg-type]
        policy_sha256=item["policy_sha256"],  # type: ignore[arg-type]
        expected_prestate_stack_role_arn=item[
            "expected_prestate_stack_role_arn"
        ],  # type: ignore[arg-type]
        expected_poststate_stack_role_arn=item[
            "expected_poststate_stack_role_arn"
        ],  # type: ignore[arg-type]
        selected_change_set_description_sha256=item[
            "selected_change_set_description_sha256"
        ],  # type: ignore[arg-type]
        created_at=item["created_at"],  # type: ignore[arg-type]
        canonical_identity_sha256=item[
            "canonical_identity_sha256"
        ],  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class FenceChangeSetEvidence:
    request_identity_sha256: str
    prepared_identity_sha256: str
    change_set_arn: str
    stack_id: str
    status: str
    execution_status: str
    description: Mapping[str, object]
    captured_at: str
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        _v2_sha(
            self.request_identity_sha256,
            "change-set evidence.request_identity_sha256",
        )
        _v2_sha(
            self.prepared_identity_sha256,
            "change-set evidence.prepared_identity_sha256",
        )
        if _V2_CHANGE_SET_ARN.fullmatch(self.change_set_arn) is None:
            _v2_fail("change-set evidence ARN is foreign")
        if _STACK_ID.fullmatch(self.stack_id) is None:
            _v2_fail("change-set evidence stack is foreign")
        _v2_text(self.status, "change-set evidence.status")
        _v2_text(
            self.execution_status, "change-set evidence.execution_status"
        )
        if not isinstance(self.description, Mapping):
            _v2_fail("change-set evidence description is not exact")
        _v2_time(self.captured_at, "change-set evidence.captured_at")
        _v2_validate_identity(self.to_dict(), name="change-set evidence")

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "record_type": _V2_RECORD_TYPES["change_set_evidence"],
            "request_identity_sha256": self.request_identity_sha256,
            "prepared_identity_sha256": self.prepared_identity_sha256,
            "change_set_arn": self.change_set_arn,
            "stack_id": self.stack_id,
            "status": self.status,
            "execution_status": self.execution_status,
            "description": _v2_thaw(self.description),
            "captured_at": self.captured_at,
            "canonical_identity_sha256": self.canonical_identity_sha256,
        }


@dataclass(frozen=True)
class FenceChangeSetInventoryDigest:
    request_identity_sha256: str
    prepared_identity_sha256: str
    stack_id: str
    selected_change_set_arn: str
    visible_count: int
    page_count: int
    snapshot_identity_sha256s: Tuple[str, ...]
    sealed_at: str
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        _v2_sha(
            self.request_identity_sha256,
            "change-set inventory.request_identity_sha256",
        )
        _v2_sha(
            self.prepared_identity_sha256,
            "change-set inventory.prepared_identity_sha256",
        )
        if (
            _STACK_ID.fullmatch(self.stack_id) is None
            or _V2_CHANGE_SET_ARN.fullmatch(self.selected_change_set_arn)
            is None
        ):
            _v2_fail("change-set inventory coordinates are foreign")
        if (
            type(self.visible_count) is not int
            or not 1 <= self.visible_count <= 16
            or type(self.page_count) is not int
            or not 1 <= self.page_count <= 16
            or type(self.snapshot_identity_sha256s) is not tuple
            or len(self.snapshot_identity_sha256s) != self.visible_count
            or len(set(self.snapshot_identity_sha256s))
            != self.visible_count
        ):
            _v2_fail("change-set inventory cardinality is not exact")
        for identity in self.snapshot_identity_sha256s:
            _v2_sha(identity, "change-set inventory snapshot identity")
        _v2_time(self.sealed_at, "change-set inventory.sealed_at")
        _v2_validate_identity(self.to_dict(), name="change-set inventory")

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "record_type": _V2_RECORD_TYPES["change_set_list"],
            "request_identity_sha256": self.request_identity_sha256,
            "prepared_identity_sha256": self.prepared_identity_sha256,
            "stack_id": self.stack_id,
            "selected_change_set_arn": self.selected_change_set_arn,
            "visible_count": self.visible_count,
            "page_count": self.page_count,
            "snapshot_identity_sha256s": list(
                self.snapshot_identity_sha256s
            ),
            "sealed_at": self.sealed_at,
            "canonical_identity_sha256": self.canonical_identity_sha256,
        }


def parse_change_set_evidence(value: object) -> FenceChangeSetEvidence:
    fields = {
        "schema_version",
        "record_type",
        "request_identity_sha256",
        "prepared_identity_sha256",
        "change_set_arn",
        "stack_id",
        "status",
        "execution_status",
        "description",
        "captured_at",
        "canonical_identity_sha256",
    }
    item = _v2_exact_mapping(value, fields, "change-set evidence")
    if (
        item["schema_version"] != 2
        or item["record_type"] != _V2_RECORD_TYPES["change_set_evidence"]
    ):
        _v2_fail("change-set evidence v1/non-v2 is ineligible")
    description = item["description"]
    if type(description) is not dict:
        _v2_fail("change-set evidence description is not exact")
    return FenceChangeSetEvidence(
        request_identity_sha256=item["request_identity_sha256"],  # type: ignore[arg-type]
        prepared_identity_sha256=item["prepared_identity_sha256"],  # type: ignore[arg-type]
        change_set_arn=item["change_set_arn"],  # type: ignore[arg-type]
        stack_id=item["stack_id"],  # type: ignore[arg-type]
        status=item["status"],  # type: ignore[arg-type]
        execution_status=item["execution_status"],  # type: ignore[arg-type]
        description=_v2_deep_freeze(description),  # type: ignore[arg-type]
        captured_at=item["captured_at"],  # type: ignore[arg-type]
        canonical_identity_sha256=item[
            "canonical_identity_sha256"
        ],  # type: ignore[arg-type]
    )


def parse_change_set_inventory_digest(
    value: object,
) -> FenceChangeSetInventoryDigest:
    fields = {
        "schema_version",
        "record_type",
        "request_identity_sha256",
        "prepared_identity_sha256",
        "stack_id",
        "selected_change_set_arn",
        "visible_count",
        "page_count",
        "snapshot_identity_sha256s",
        "sealed_at",
        "canonical_identity_sha256",
    }
    item = _v2_exact_mapping(value, fields, "change-set inventory")
    if (
        item["schema_version"] != 2
        or item["record_type"] != _V2_RECORD_TYPES["change_set_list"]
        or type(item["snapshot_identity_sha256s"]) not in (list, tuple)
    ):
        _v2_fail("change-set inventory v1/non-v2 is ineligible")
    return FenceChangeSetInventoryDigest(
        request_identity_sha256=item["request_identity_sha256"],  # type: ignore[arg-type]
        prepared_identity_sha256=item["prepared_identity_sha256"],  # type: ignore[arg-type]
        stack_id=item["stack_id"],  # type: ignore[arg-type]
        selected_change_set_arn=item["selected_change_set_arn"],  # type: ignore[arg-type]
        visible_count=item["visible_count"],  # type: ignore[arg-type]
        page_count=item["page_count"],  # type: ignore[arg-type]
        snapshot_identity_sha256s=tuple(
            item["snapshot_identity_sha256s"]  # type: ignore[arg-type]
        ),
        sealed_at=item["sealed_at"],  # type: ignore[arg-type]
        canonical_identity_sha256=item[
            "canonical_identity_sha256"
        ],  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class FenceExecutionResult:
    request_identity_sha256: str
    manifest_identity_sha256: str
    entry_identity_sha256: str
    prepared_identity_sha256: str
    change_set_inventory_identity_sha256: str
    change_set_arn: str
    stack_id: str
    slot: FenceSlot
    execute_authority_identity_sha256: str
    execute_authority_class: ExecutorAuthorityClass
    execute_api_caller_role_arn: str
    execute_api_caller_role_id: str
    expected_poststate_stack_role_arn: str
    observed_poststate_stack_role_arn: str
    observed_poststate_stack_role_id: str
    original_template_body_sha256: str
    processed_template_body_sha256: str
    poststate_policy_sha256: str
    first_stable_snapshot_identity_sha256: str
    second_stable_snapshot_identity_sha256: str
    stabilization_first_evidence_sha256: str
    stabilization_second_evidence_sha256: str
    stabilization_identity_sha256: str
    freeze_delta_classification: Optional[str]
    batch_execution_eligible: bool
    completed_at: str
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "request_identity_sha256",
            "manifest_identity_sha256",
            "entry_identity_sha256",
            "prepared_identity_sha256",
            "change_set_inventory_identity_sha256",
            "execute_authority_identity_sha256",
            "original_template_body_sha256",
            "processed_template_body_sha256",
            "poststate_policy_sha256",
            "first_stable_snapshot_identity_sha256",
            "second_stable_snapshot_identity_sha256",
            "stabilization_first_evidence_sha256",
            "stabilization_second_evidence_sha256",
            "stabilization_identity_sha256",
        ):
            _v2_sha(getattr(self, field), "execution result." + field)
        if (
            _V2_CHANGE_SET_ARN.fullmatch(self.change_set_arn) is None
            or _STACK_ID.fullmatch(self.stack_id) is None
            or type(self.slot) is not FenceSlot
            or type(self.execute_authority_class)
            is not ExecutorAuthorityClass
        ):
            _v2_fail("execution result service identity is foreign")
        _v2_role_arn(
            self.execute_api_caller_role_arn,
            "execution result.execute_api_caller_role_arn",
        )
        _v2_role_id(
            self.execute_api_caller_role_id,
            "execution result.execute_api_caller_role_id",
        )
        _v2_role_arn(
            self.expected_poststate_stack_role_arn,
            "execution result.expected_poststate_stack_role_arn",
        )
        _v2_role_arn(
            self.observed_poststate_stack_role_arn,
            "execution result.observed_poststate_stack_role_arn",
        )
        _v2_role_id(
            self.observed_poststate_stack_role_id,
            "execution result.observed_poststate_stack_role_id",
        )
        if (
            self.expected_poststate_stack_role_arn
            != self.observed_poststate_stack_role_arn
            or self.original_template_body_sha256
            != self.processed_template_body_sha256
            or self.first_stable_snapshot_identity_sha256
            != self.second_stable_snapshot_identity_sha256
            or self.stabilization_first_evidence_sha256
            == self.stabilization_second_evidence_sha256
            or type(self.batch_execution_eligible) is not bool
        ):
            _v2_fail("execution result did not stabilize exact poststate")
        if self.slot is FenceSlot.SOURCE_FAMILIES_FROZEN:
            if self.freeze_delta_classification not in {
                "EXACT",
                "FREEZE_EXPECTED_SOURCE_DELTA",
                "FREEZE_QUARANTINED_DRIFT",
            }:
                _v2_fail("freeze result omitted its delta classification")
            if (
                self.freeze_delta_classification
                == "FREEZE_QUARANTINED_DRIFT"
                and self.batch_execution_eligible
            ):
                _v2_fail("quarantined freeze result still authorizes BATCH")
        elif self.freeze_delta_classification is not None:
            _v2_fail("non-freeze result carries a freeze delta classification")
        _v2_time(self.completed_at, "execution result.completed_at")
        _v2_validate_identity(self.to_dict(), name="fence execution result")

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "record_type": _V2_RECORD_TYPES["execution_result"],
            "request_identity_sha256": self.request_identity_sha256,
            "manifest_identity_sha256": self.manifest_identity_sha256,
            "entry_identity_sha256": self.entry_identity_sha256,
            "prepared_identity_sha256": self.prepared_identity_sha256,
            "change_set_inventory_identity_sha256": (
                self.change_set_inventory_identity_sha256
            ),
            "change_set_arn": self.change_set_arn,
            "stack_id": self.stack_id,
            "slot": self.slot.value,
            "execute_authority_identity_sha256": (
                self.execute_authority_identity_sha256
            ),
            "execute_authority_class": self.execute_authority_class.value,
            "execute_api_caller_role_arn": (
                self.execute_api_caller_role_arn
            ),
            "execute_api_caller_role_id": self.execute_api_caller_role_id,
            "expected_poststate_stack_role_arn": (
                self.expected_poststate_stack_role_arn
            ),
            "observed_poststate_stack_role_arn": (
                self.observed_poststate_stack_role_arn
            ),
            "observed_poststate_stack_role_id": (
                self.observed_poststate_stack_role_id
            ),
            "original_template_body_sha256": (
                self.original_template_body_sha256
            ),
            "processed_template_body_sha256": (
                self.processed_template_body_sha256
            ),
            "poststate_policy_sha256": self.poststate_policy_sha256,
            "first_stable_snapshot_identity_sha256": (
                self.first_stable_snapshot_identity_sha256
            ),
            "second_stable_snapshot_identity_sha256": (
                self.second_stable_snapshot_identity_sha256
            ),
            "stabilization_first_evidence_sha256": (
                self.stabilization_first_evidence_sha256
            ),
            "stabilization_second_evidence_sha256": (
                self.stabilization_second_evidence_sha256
            ),
            "stabilization_identity_sha256": (
                self.stabilization_identity_sha256
            ),
            "freeze_delta_classification": self.freeze_delta_classification,
            "batch_execution_eligible": self.batch_execution_eligible,
            "completed_at": self.completed_at,
            "canonical_identity_sha256": self.canonical_identity_sha256,
        }


_V2_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "request_identity_sha256",
        "manifest_identity_sha256",
        "entry_identity_sha256",
        "prepared_identity_sha256",
        "change_set_inventory_identity_sha256",
        "change_set_arn",
        "stack_id",
        "slot",
        "execute_authority_identity_sha256",
        "execute_authority_class",
        "execute_api_caller_role_arn",
        "execute_api_caller_role_id",
        "expected_poststate_stack_role_arn",
        "observed_poststate_stack_role_arn",
        "observed_poststate_stack_role_id",
        "original_template_body_sha256",
        "processed_template_body_sha256",
        "poststate_policy_sha256",
        "first_stable_snapshot_identity_sha256",
        "second_stable_snapshot_identity_sha256",
        "stabilization_first_evidence_sha256",
        "stabilization_second_evidence_sha256",
        "stabilization_identity_sha256",
        "freeze_delta_classification",
        "batch_execution_eligible",
        "completed_at",
        "canonical_identity_sha256",
    }
)


def parse_fence_execution_result(value: object) -> FenceExecutionResult:
    item = _v2_exact_mapping(
        value, _V2_RESULT_FIELDS, "fence execution result"
    )
    if (
        item["schema_version"] != 2
        or item["record_type"] != _V2_RECORD_TYPES["execution_result"]
    ):
        _v2_fail("fence execution result v1/non-v2 is execution-ineligible")
    try:
        slot = FenceSlot(item["slot"])
        authority_class = ExecutorAuthorityClass(
            item["execute_authority_class"]
        )
    except (TypeError, ValueError) as exc:
        raise FenceExecutorV2Error("execution result enum is foreign") from exc
    return FenceExecutionResult(
        request_identity_sha256=item["request_identity_sha256"],  # type: ignore[arg-type]
        manifest_identity_sha256=item["manifest_identity_sha256"],  # type: ignore[arg-type]
        entry_identity_sha256=item["entry_identity_sha256"],  # type: ignore[arg-type]
        prepared_identity_sha256=item["prepared_identity_sha256"],  # type: ignore[arg-type]
        change_set_inventory_identity_sha256=item[
            "change_set_inventory_identity_sha256"
        ],  # type: ignore[arg-type]
        change_set_arn=item["change_set_arn"],  # type: ignore[arg-type]
        stack_id=item["stack_id"],  # type: ignore[arg-type]
        slot=slot,
        execute_authority_identity_sha256=item[
            "execute_authority_identity_sha256"
        ],  # type: ignore[arg-type]
        execute_authority_class=authority_class,
        execute_api_caller_role_arn=item[
            "execute_api_caller_role_arn"
        ],  # type: ignore[arg-type]
        execute_api_caller_role_id=item[
            "execute_api_caller_role_id"
        ],  # type: ignore[arg-type]
        expected_poststate_stack_role_arn=item[
            "expected_poststate_stack_role_arn"
        ],  # type: ignore[arg-type]
        observed_poststate_stack_role_arn=item[
            "observed_poststate_stack_role_arn"
        ],  # type: ignore[arg-type]
        observed_poststate_stack_role_id=item[
            "observed_poststate_stack_role_id"
        ],  # type: ignore[arg-type]
        original_template_body_sha256=item[
            "original_template_body_sha256"
        ],  # type: ignore[arg-type]
        processed_template_body_sha256=item[
            "processed_template_body_sha256"
        ],  # type: ignore[arg-type]
        poststate_policy_sha256=item["poststate_policy_sha256"],  # type: ignore[arg-type]
        first_stable_snapshot_identity_sha256=item[
            "first_stable_snapshot_identity_sha256"
        ],  # type: ignore[arg-type]
        second_stable_snapshot_identity_sha256=item[
            "second_stable_snapshot_identity_sha256"
        ],  # type: ignore[arg-type]
        stabilization_first_evidence_sha256=item[
            "stabilization_first_evidence_sha256"
        ],  # type: ignore[arg-type]
        stabilization_second_evidence_sha256=item[
            "stabilization_second_evidence_sha256"
        ],  # type: ignore[arg-type]
        stabilization_identity_sha256=item[
            "stabilization_identity_sha256"
        ],  # type: ignore[arg-type]
        freeze_delta_classification=item[
            "freeze_delta_classification"
        ],  # type: ignore[arg-type]
        batch_execution_eligible=item[
            "batch_execution_eligible"
        ],  # type: ignore[arg-type]
        completed_at=item["completed_at"],  # type: ignore[arg-type]
        canonical_identity_sha256=item[
            "canonical_identity_sha256"
        ],  # type: ignore[arg-type]
    )


def _v2_duplicate_rejecting_object(raw: object, name: str) -> Mapping[str, object]:
    if type(raw) is bytes:
        try:
            text = raw.decode("ascii")
        except UnicodeError as exc:
            raise FenceExecutorV2Error(name + " is not ASCII") from exc
    elif type(raw) is str:
        text = raw
        try:
            raw = text.encode("ascii")
        except UnicodeError as exc:
            raise FenceExecutorV2Error(name + " is not ASCII") from exc
    else:
        _v2_fail(name + " is not bytes/text")

    def pairs(items: Sequence[Tuple[str, object]]) -> dict:
        result: Dict[str, object] = {}
        for key, item in items:
            if key in result:
                raise FenceExecutorV2Error(
                    name + " contains a duplicate JSON key"
                )
            result[key] = item
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except FenceExecutorV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FenceExecutorV2Error(name + " is not strict JSON") from exc
    if type(value) is not dict:
        _v2_fail(name + " is not one JSON object")
    if canonical_json_bytes(value) != raw:
        _v2_fail(name + " is not canonical JSON")
    return _v2_deep_freeze(value)  # type: ignore[return-value]


def parse_direct_policy_document(raw: object) -> Mapping[str, object]:
    """Strictly parse canonical S3 policy bytes, rejecting every duplicate key."""

    return _v2_duplicate_rejecting_object(raw, "direct bucket policy")


def direct_policy_sha256(raw: object) -> str:
    parse_direct_policy_document(raw)
    if type(raw) is str:
        raw = raw.encode("ascii")
    return hashlib.sha256(raw).hexdigest()  # type: ignore[arg-type]


def validate_fence_template_bytes(raw: object) -> Mapping[str, object]:
    if type(raw) is not bytes or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _v2_fail("fence template is not one canonical LF-terminated file")
    value = _v2_duplicate_rejecting_object(raw[:-1], "fence template")
    if frozenset(value) != {
        "AWSTemplateFormatVersion",
        "Resources",
    }:
        _v2_fail("fence template top-level shape is not exact")
    resources = value["Resources"]
    if (
        value["AWSTemplateFormatVersion"] != "2010-09-09"
        or not isinstance(resources, Mapping)
        or frozenset(resources) != {FENCE_LOGICAL_ID}
    ):
        _v2_fail("fence template resource ownership is not exact")
    resource = resources[FENCE_LOGICAL_ID]
    if (
        not isinstance(resource, Mapping)
        or frozenset(resource)
        != {
            "DeletionPolicy",
            "Properties",
            "Type",
            "UpdateReplacePolicy",
        }
        or resource["Type"] != "AWS::S3::BucketPolicy"
        or resource["DeletionPolicy"] != "Retain"
        or resource["UpdateReplacePolicy"] != "Retain"
    ):
        _v2_fail("fence template does not retain the sole policy resource")
    return value


def validate_fence_stack_policy(raw: object) -> str:
    policy = _v2_duplicate_rejecting_object(raw, "fence stack policy")
    statements = policy.get("Statement")
    if type(statements) is not tuple:
        _v2_fail("fence stack policy statements are not exact")
    allow_modify = 0
    deny_replace_delete = 0
    resource = "LogicalResourceId/" + FENCE_LOGICAL_ID
    for statement in statements:
        if not isinstance(statement, Mapping):
            _v2_fail("fence stack policy statement is malformed")
        effect = statement.get("Effect")
        action = statement.get("Action")
        target = statement.get("Resource")
        actions = (action,) if type(action) is str else action
        if type(actions) is not tuple or any(type(item) is not str for item in actions):
            _v2_fail("fence stack policy action is malformed")
        if effect == "Allow" and "Update:Modify" in actions:
            if tuple(actions) != ("Update:Modify",) or target != resource:
                _v2_fail("fence stack policy modify allow is broader than exact")
            allow_modify += 1
        if effect == "Deny" and set(actions) == {
            "Update:Replace",
            "Update:Delete",
        }:
            if target != resource:
                _v2_fail("fence stack policy retain deny targets another resource")
            deny_replace_delete += 1
        if effect == "Allow" and any(item != "Update:Modify" for item in actions):
            _v2_fail("fence stack policy contains another allow")
    if allow_modify != 1 or deny_replace_delete != 1:
        _v2_fail("fence stack policy does not close replace/delete")
    if type(raw) is str:
        raw = raw.encode("ascii")
    return hashlib.sha256(raw).hexdigest()  # type: ignore[arg-type]


def collect_visible_change_sets(
    *,
    client: object,
    stack_id: str,
) -> Tuple[Tuple[Mapping[str, object], ...], int]:
    """Exhaustively page a bounded visible set; item 17 fails immediately."""

    if type(stack_id) is not str or _STACK_ID.fullmatch(stack_id) is None:
        _v2_fail("change-set inventory stack_id is foreign")
    rows: list[Mapping[str, object]] = []
    next_token: Optional[str] = None
    seen_tokens: set[str] = set()
    page_count = 0
    while True:
        request: Dict[str, object] = {"StackName": stack_id}
        if next_token is not None:
            request["NextToken"] = next_token
        response = _success_response(
            _service_method(client, "list_change_sets")(**request),
            "ListChangeSets",
        )
        page = response.get("Summaries")
        if type(page) is not list or any(type(row) is not dict for row in page):
            _v2_fail("ListChangeSets summaries are malformed")
        page_count += 1
        for row in page:
            rows.append(MappingProxyType(dict(row)))
            if len(rows) > 16:
                _v2_fail("CHANGE_SET_INVENTORY_EXHAUSTED")
        token = response.get("NextToken")
        if token is None:
            return tuple(rows), page_count
        if (
            type(token) is not str
            or not token
            or token in seen_tokens
            or page_count >= 16
        ):
            _v2_fail("ListChangeSets pagination is ambiguous or unbounded")
        seen_tokens.add(token)
        next_token = token


def _v2_validate_selected_change_set(
    description: Mapping[str, object],
    *,
    stack_id: str,
    entry: FenceArtifactEntry,
) -> None:
    value = entry.to_dict()
    skeleton = entry.request_skeleton
    if (
        description.get("StackId") != stack_id
        or description.get("ChangeSetName") != skeleton["ChangeSetName"]
        or (
            description.get("Status"),
            description.get("ExecutionStatus"),
        )
        != _V2_SELECTED_CHANGE_SET_STATUS
        or description.get("Parameters") not in (None, [])
        or description.get("Capabilities") not in (None, [])
    ):
        _v2_fail("selected change set identity/status is not exact")
    changes = description.get("Changes")
    if type(changes) is not list or len(changes) != 1:
        _v2_fail("selected change set is not a one-resource modify")
    change = changes[0]
    resource = (
        change.get("ResourceChange")
        if type(change) is dict and change.get("Type") == "Resource"
        else None
    )
    if (
        type(resource) is not dict
        or resource.get("Action") != "Modify"
        or resource.get("LogicalResourceId") != FENCE_LOGICAL_ID
        or resource.get("ResourceType") != "AWS::S3::BucketPolicy"
        or resource.get("Replacement") not in ("False", False)
        or resource.get("Scope") != ["Properties"]
        or type(resource.get("Details", [])) is not list
    ):
        _v2_fail("CHANGE_SET_SHAPE_UNPROVED")
    if description.get("ChangeSetId") is None:
        _v2_fail("selected change set omitted its service ARN")
    if value["request_skeleton"]["StackName"] != stack_id:
        _v2_fail("selected entry skeleton targets another stack")


class DynamoFenceRecordStore:
    """Conditional immutable v2 evidence rows over the existing ledger table."""

    def __init__(self, *, client: object, table_name: str) -> None:
        self._client = client
        self._table_name = _v2_text(table_name, "fence ledger table")

    @staticmethod
    def sort_key(
        *,
        record_type: str,
        request_identity_sha256: str,
        slot: FenceSlot,
        child_identity: Optional[str] = None,
    ) -> str:
        _v2_sha(request_identity_sha256, "record request identity")
        if type(slot) is not FenceSlot:
            _v2_fail("record slot is not exact")
        suffix = ""
        if child_identity is not None:
            _v2_sha(child_identity, "record child identity")
            suffix = "#" + child_identity
        return (
            "GLM52_FENCE_V2#"
            + slot.value
            + "#"
            + request_identity_sha256
            + "#"
            + record_type
            + suffix
        )

    def _key(
        self,
        *,
        record_type: str,
        request_identity_sha256: str,
        slot: FenceSlot,
        child_identity: Optional[str] = None,
    ) -> Mapping[str, object]:
        return {
            "PK": ledger_pk(RUN_ID),
            "SK": self.sort_key(
                record_type=record_type,
                request_identity_sha256=request_identity_sha256,
                slot=slot,
                child_identity=child_identity,
            ),
        }

    def put_immutable(
        self,
        *,
        value: Mapping[str, object],
        request_identity_sha256: str,
        slot: FenceSlot,
        child_identity: Optional[str] = None,
    ) -> None:
        if type(value) is not dict:
            _v2_fail("immutable fence row is not one exact dictionary")
        record_type = _v2_text(
            value.get("record_type"), "immutable fence row record_type"
        )
        key = self._key(
            record_type=record_type,
            request_identity_sha256=request_identity_sha256,
            slot=slot,
            child_identity=child_identity,
        )
        physical = {**key, **value}
        try:
            response = self._client.put_item(
                TableName=self._table_name,
                Item=encode_item(physical),
                ConditionExpression=(
                    "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
                ),
                ExpressionAttributeNames={"#pk": "PK", "#sk": "SK"},
            )
            _success_response(response, "PutFenceEvidence")
            return
        except Exception:
            existing = self.get_immutable(
                record_type=record_type,
                request_identity_sha256=request_identity_sha256,
                slot=slot,
                child_identity=child_identity,
            )
            if existing != dict(value):
                raise

    def get_immutable(
        self,
        *,
        record_type: str,
        request_identity_sha256: str,
        slot: FenceSlot,
        child_identity: Optional[str] = None,
    ) -> Mapping[str, object]:
        key = self._key(
            record_type=record_type,
            request_identity_sha256=request_identity_sha256,
            slot=slot,
            child_identity=child_identity,
        )
        response = _success_response(
            self._client.get_item(
                TableName=self._table_name,
                Key=encode_item(key),
                ConsistentRead=True,
            ),
            "GetFenceEvidence",
        )
        raw = response.get("Item")
        if type(raw) is not dict:
            _v2_fail("immutable fence row is absent")
        value = decode_item(raw)
        if value.pop("PK", None) != key["PK"] or value.pop("SK", None) != key["SK"]:
            _v2_fail("immutable fence row key drifted")
        return MappingProxyType(value)


def _v2_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _v2_client_token(*parts: str) -> str:
    return hashlib.sha256(
        canonical_json_bytes({"parts": list(parts)})
    ).hexdigest()


def _v2_record_with_identity(value: Mapping[str, object]) -> dict:
    result = dict(value)
    result["canonical_identity_sha256"] = _v2_canonical_sha256(result)
    return result


def _v2_validate_request_binding(
    *,
    request: FenceTransitionRequest,
    manifest: FenceManifest,
    entry: FenceArtifactEntry,
) -> Mapping[str, object]:
    if (
        type(request) is not FenceTransitionRequest
        or type(manifest) is not FenceManifest
        or type(entry) is not FenceArtifactEntry
    ):
        _v2_fail("executor requires parsed Task-1 v2 contracts")
    request_value = request.to_dict()
    manifest_value = manifest.to_dict()
    entry_value = entry.to_dict()
    if (
        request_value["manifest_coordinate"]["canonical_identity_sha256"]
        != manifest.canonical_identity_sha256
        or request_value["manifest_stage"]
        != manifest.manifest_stage.value
        or request.slot is not entry.slot
        or request_value["selected_entry_identity_sha256"]
        != entry.entry_identity_sha256
        or request_value["request_skeleton_sha256"]
        != entry.request_skeleton_sha256
        or tuple(request_value["allowed_create_authority_classes"])
        != entry.allowed_create_authority_classes
        or tuple(request_value["allowed_execute_authority_classes"])
        != entry.allowed_execute_authority_classes
        or request_value["predecessor_head"]
        not in [dict(row) for row in entry.allowed_predecessor_heads]
    ):
        _v2_fail("full request/stage/graph/entry identity drifted")
    skeleton = entry.request_skeleton
    if (
        skeleton.get("RoleARN")
        != manifest_value["fence_service_role"].get("arn")
        or _v2_canonical_sha256(skeleton)
        != entry.request_skeleton_sha256
    ):
        _v2_fail("request skeleton does not use the exact FenceServiceRole")
    return entry_value


def _v2_validate_authority_binding(
    *,
    request: FenceTransitionRequest,
    manifest: FenceManifest,
    entry: FenceArtifactEntry,
    authority: object,
    phase: str,
) -> Mapping[str, object]:
    expected_type = (
        FenceCreateAuthorityV2
        if phase == "CREATE"
        else FenceExecuteAuthorityV2
    )
    if type(authority) is not expected_type:
        _v2_fail(phase + " authority type is not exact")
    if authority.request_identity_sha256 != request.canonical_identity_sha256:
        _v2_fail(phase + " authority request identity drifted")
    allowed = (
        entry.allowed_create_authority_classes
        if phase == "CREATE"
        else entry.allowed_execute_authority_classes
    )
    if authority.authority_class.value not in allowed:
        _v2_fail("EXECUTOR_AUTHORITY_CLASS_MISMATCH")
    row = _v2_manifest_executor_row(manifest, authority.authority_class)
    caller = row["api_caller_role"]
    service = row["cloudformation_service_role"]
    if (
        authority.api_caller_role_arn != caller["role_arn"]
        or authority.api_caller_role_id != caller["role_id"]
        or authority.api_caller_trust_policy_sha256
        != caller["trust_policy_sha256"]
        or authority.api_caller_permission_policy_sha256
        != caller["permission_policy_sha256"]
        or authority.cloudformation_service_role_arn
        != service["role_arn"]
        or authority.cloudformation_service_role_id != service["role_id"]
        or authority.cloudformation_service_role_trust_policy_sha256
        != service["trust_policy_sha256"]
        or authority.cloudformation_service_role_permission_policy_sha256
        != service["permission_policy_sha256"]
    ):
        _v2_fail("executor caller/service role identity drifted")
    return row


def _v2_validate_prestate_binding(
    *,
    request: FenceTransitionRequest,
    manifest: FenceManifest,
    entry: FenceArtifactEntry,
    prestate: FencePrestateSnapshot,
    phase: str,
) -> Mapping[str, object]:
    if type(prestate) is not FencePrestateSnapshot or prestate.phase != phase:
        _v2_fail("prestate snapshot phase/type is not exact")
    manifest_value = manifest.to_dict()
    entry_value = entry.to_dict()
    expected_prestate_role = (
        manifest_value["migration_service_role"]
        if entry.slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE
        else manifest_value["fence_service_role"]
    )
    expected_poststate_role = manifest_value["fence_service_role"]
    if (
        type(expected_prestate_role) is not dict
        or type(expected_poststate_role) is not dict
        or entry_value["expected_prestate_stack_role_arn"]
        != expected_prestate_role.get("arn")
        or entry_value["expected_poststate_stack_role_arn"]
        != expected_poststate_role.get("arn")
        or prestate.stack_role_id != expected_prestate_role.get("role_id")
    ):
        _v2_fail("PREPARE is not the sole exact service-role cutover")
    if (
        prestate.request_identity_sha256
        != request.canonical_identity_sha256
        or prestate.manifest_identity_sha256
        != manifest.canonical_identity_sha256
        or prestate.entry_identity_sha256
        != entry.entry_identity_sha256
        or prestate.stack_id != manifest_value["stack_id"]
        or prestate.logical_id != manifest_value["logical_id"]
        or prestate.stack_role_arn
        != entry_value["expected_prestate_stack_role_arn"]
        or prestate.direct_policy_sha256
        != entry_value["expected_prestate_policy_sha256"]
        or prestate.original_template_body_sha256
        != prestate.processed_template_body_sha256
    ):
        _v2_fail("full request/prestate identity drifted")
    return entry_value


def _v2_load_exact_s3_version(
    *,
    s3: object,
    bucket: str,
    key: str,
    version_id: str,
    file_sha256: str,
) -> bytes:
    if bucket != "keep-glm52-models-246813579024-us-west-2":
        _v2_fail("pinned artifact bucket is foreign")
    _v2_text(key, "pinned artifact key")
    if (
        type(version_id) is not str
        or not version_id
        or version_id == "null"
        or len(version_id.encode("utf-8")) > 1024
    ):
        _v2_fail("pinned artifact VersionId is not exact")
    _v2_sha(file_sha256, "pinned artifact file_sha256")
    versions: list[Mapping[str, object]] = []
    markers: list[Mapping[str, object]] = []
    key_marker: Optional[str] = None
    version_marker: Optional[str] = None
    seen: set[Tuple[str, str]] = set()
    for _page in range(16):
        request: Dict[str, object] = {
            "Bucket": bucket,
            "Prefix": key,
            "ExpectedBucketOwner": ACCOUNT_ID,
        }
        if key_marker is not None:
            request["KeyMarker"] = key_marker
            request["VersionIdMarker"] = version_marker
        response = _success_response(
            _service_method(s3, "list_object_versions")(**request),
            "ListObjectVersions",
        )
        page_versions = response.get("Versions", [])
        page_markers = response.get("DeleteMarkers", [])
        if (
            type(page_versions) is not list
            or type(page_markers) is not list
            or any(type(row) is not dict for row in page_versions)
            or any(type(row) is not dict for row in page_markers)
        ):
            _v2_fail("fixed-key history response is malformed")
        versions.extend(row for row in page_versions if row.get("Key") == key)
        markers.extend(row for row in page_markers if row.get("Key") == key)
        if response.get("IsTruncated") is False:
            break
        next_key = response.get("NextKeyMarker")
        next_version = response.get("NextVersionIdMarker")
        if (
            type(next_key) is not str
            or not next_key
            or type(next_version) is not str
            or not next_version
            or (next_key, next_version) in seen
        ):
            _v2_fail("fixed-key history pagination is ambiguous")
        seen.add((next_key, next_version))
        key_marker = next_key
        version_marker = next_version
    else:
        _v2_fail("fixed-key history pagination is unbounded")
    if (
        markers
        or len(versions) != 1
        or versions[0].get("VersionId") != version_id
        or versions[0].get("IsLatest") is not True
    ):
        _v2_fail("FIXED_KEY_HISTORY_DRIFT")
    response = _success_response(
        _service_method(s3, "get_object")(
            Bucket=bucket,
            Key=key,
            VersionId=version_id,
            ExpectedBucketOwner=ACCOUNT_ID,
            ChecksumMode="ENABLED",
        ),
        "GetObjectVersion",
    )
    if response.get("VersionId") != version_id:
        _v2_fail("pinned artifact read returned another VersionId")
    body = response.get("Body")
    if type(body) is bytes:
        raw = body
    else:
        read_body = getattr(body, "read", None)
        if not callable(read_body):
            _v2_fail("pinned artifact body is unreadable")
        raw = read_body()
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != file_sha256:
        _v2_fail("pinned artifact file hash drifted")
    return raw


def load_pinned_fence_manifest(
    *,
    s3: object,
    coordinate: ArtifactCoordinate,
) -> FenceManifest:
    """Independently reload one singular manifest version and exact file hash."""

    if type(coordinate) is not ArtifactCoordinate:
        _v2_fail("manifest loader requires one parsed ArtifactCoordinate")
    raw = _v2_load_exact_s3_version(
        s3=s3,
        bucket=coordinate.bucket,
        key=coordinate.key,
        version_id=coordinate.version_id,
        file_sha256=coordinate.file_sha256,
    )
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _v2_fail("manifest file is not singular LF-terminated canonical JSON")
    value = _v2_duplicate_rejecting_object(raw[:-1], "fence manifest")
    manifest = parse_fence_manifest(_v2_thaw(value))
    if (
        manifest.canonical_identity_sha256
        != coordinate.canonical_identity_sha256
    ):
        _v2_fail("pinned manifest canonical identity drifted")
    return manifest


def load_pinned_fence_entry_template(
    *,
    s3: object,
    entry: FenceArtifactEntry,
) -> FenceArtifactEntry:
    """Reload the selected template independently from its versioned URL."""

    if type(entry) is not FenceArtifactEntry:
        _v2_fail("template loader requires one parsed v2 entry")
    value = entry.to_dict()
    raw = _v2_load_exact_s3_version(
        s3=s3,
        bucket="keep-glm52-models-246813579024-us-west-2",
        key=value["template_key"],
        version_id=value["version_id"],
        file_sha256=value["template_sha256"],
    )
    validate_fence_template_bytes(raw)
    parsed = parse_fence_entry(value, template_bytes=raw)
    if parsed.entry_identity_sha256 != entry.entry_identity_sha256:
        _v2_fail("pinned template changed its manifest entry identity")
    return parsed


class FenceExecutor:
    """Strict v2-only CloudFormation transition executor."""

    def __init__(
        self,
        *,
        client: object,
        record_store: object,
    ) -> None:
        self._client = client
        self._record_store = record_store

    def _load(
        self,
        *,
        request: FenceTransitionRequest,
        record_type: str,
        child_identity: Optional[str] = None,
    ) -> Mapping[str, object]:
        if type(request) is not FenceTransitionRequest:
            _v2_fail("record loader requires a v2 transition request")
        value = _service_method(
            self._record_store, "get_immutable"
        )(
            record_type=record_type,
            request_identity_sha256=request.canonical_identity_sha256,
            slot=request.slot,
            child_identity=child_identity,
        )
        if not isinstance(value, Mapping):
            _v2_fail("immutable fence row is not a mapping")
        return dict(value)

    def load_create_authority(
        self, request: FenceTransitionRequest
    ) -> FenceCreateAuthorityV2:
        return parse_fence_create_authority(
            self._load(
                request=request,
                record_type=_V2_RECORD_TYPES["create_authority"],
            )
        )

    def load_prestate_snapshot(
        self, request: FenceTransitionRequest, phase: str
    ) -> FencePrestateSnapshot:
        if phase not in {"CREATE", "EXECUTE"}:
            _v2_fail("prestate loader phase is not exact")
        return parse_fence_prestate_snapshot(
            self._load(
                request=request,
                record_type=_V2_RECORD_TYPES[
                    "prestate_" + phase.lower()
                ],
            )
        )

    def load_prepared_change_set(
        self, request: FenceTransitionRequest
    ) -> PreparedFenceChangeSet:
        return parse_prepared_fence_change_set(
            self._load(
                request=request,
                record_type=_V2_RECORD_TYPES["prepared"],
            )
        )

    def load_execute_authority(
        self,
        request: FenceTransitionRequest,
        prepared: PreparedFenceChangeSet,
    ) -> FenceExecuteAuthorityV2:
        authority = parse_fence_execute_authority(
            self._load(
                request=request,
                record_type=_V2_RECORD_TYPES["execute_authority"],
            )
        )
        if (
            type(prepared) is not PreparedFenceChangeSet
            or authority.prepared_identity_sha256
            != prepared.canonical_identity_sha256
        ):
            _v2_fail("execute authority does not bind prepared custody")
        return authority

    def load_freeze_execute_delta_audit(
        self,
        request: FenceTransitionRequest,
        prepared: PreparedFenceChangeSet,
    ) -> FreezeExecuteDeltaAudit:
        audit = parse_freeze_execute_delta_audit(
            self._load(
                request=request,
                record_type=_V2_RECORD_TYPES["freeze_delta_audit"],
            )
        )
        if (
            type(prepared) is not PreparedFenceChangeSet
            or audit.prepared_identity_sha256
            != prepared.canonical_identity_sha256
        ):
            _v2_fail("freeze delta audit does not bind prepared custody")
        return audit

    def prepare_change_set(
        self,
        *,
        request: FenceTransitionRequest,
        manifest: FenceManifest,
        entry: FenceArtifactEntry,
        authority: FenceCreateAuthorityV2,
        prestate: FencePrestateSnapshot,
    ) -> PreparedFenceChangeSet:
        entry_value = _v2_validate_request_binding(
            request=request, manifest=manifest, entry=entry
        )
        _v2_validate_authority_binding(
            request=request,
            manifest=manifest,
            entry=entry,
            authority=authority,
            phase="CREATE",
        )
        _v2_validate_prestate_binding(
            request=request,
            manifest=manifest,
            entry=entry,
            prestate=prestate,
            phase="CREATE",
        )
        skeleton = dict(entry.request_skeleton)
        if "change_set_arn" in skeleton or "ClientToken" in skeleton:
            _v2_fail("pre-create skeleton carries a dynamic field")
        token = _v2_client_token(
            request.canonical_identity_sha256,
            authority.canonical_identity_sha256,
            "CREATE",
        )
        response = _success_response(
            _service_method(self._client, "create_change_set")(
                **skeleton, ClientToken=token
            ),
            "CreateChangeSet",
        )
        change_set_arn = response.get("Id")
        if (
            type(change_set_arn) is not str
            or _V2_CHANGE_SET_ARN.fullmatch(change_set_arn) is None
            or response.get("StackId") != skeleton["StackName"]
        ):
            _v2_fail("CreateChangeSet omitted its dynamic service identity")
        description = _success_response(
            _service_method(self._client, "describe_change_set")(
                ChangeSetName=change_set_arn
            ),
            "DescribeChangeSet",
        )
        _v2_validate_selected_change_set(
            description,
            stack_id=skeleton["StackName"],  # type: ignore[arg-type]
            entry=entry,
        )
        if description.get("ChangeSetId") != change_set_arn:
            _v2_fail("Create/Describe change-set ARN mismatch")
        base = {
            "schema_version": 2,
            "record_type": _V2_RECORD_TYPES["prepared"],
            "request_identity_sha256": request.canonical_identity_sha256,
            "manifest_identity_sha256": manifest.canonical_identity_sha256,
            "entry_identity_sha256": entry.entry_identity_sha256,
            "prestate_identity_sha256": prestate.canonical_identity_sha256,
            "request_skeleton_sha256": entry.request_skeleton_sha256,
            "change_set_arn": change_set_arn,
            "stack_id": skeleton["StackName"],
            "slot": entry.slot.value,
            "create_authority_identity_sha256": (
                authority.canonical_identity_sha256
            ),
            "create_authority_class": authority.authority_class.value,
            "create_api_caller_role_arn": authority.api_caller_role_arn,
            "create_api_caller_role_id": authority.api_caller_role_id,
            "cloudformation_service_role_arn": (
                authority.cloudformation_service_role_arn
            ),
            "cloudformation_service_role_id": (
                authority.cloudformation_service_role_id
            ),
            "template_sha256": entry_value["template_sha256"],
            "template_body_sha256": entry_value["template_body_sha256"],
            "policy_sha256": entry.policy_sha256,
            "expected_prestate_stack_role_arn": entry_value[
                "expected_prestate_stack_role_arn"
            ],
            "expected_poststate_stack_role_arn": entry_value[
                "expected_poststate_stack_role_arn"
            ],
            "selected_change_set_description_sha256": (
                _v2_canonical_sha256(description)
            ),
            "created_at": _v2_now(),
        }
        prepared = parse_prepared_fence_change_set(
            _v2_record_with_identity(base)
        )
        _service_method(self._record_store, "put_immutable")(
            value=prepared.to_dict(),
            request_identity_sha256=request.canonical_identity_sha256,
            slot=request.slot,
            child_identity=None,
        )
        return prepared

    def _snapshot_change_sets(
        self,
        *,
        request: FenceTransitionRequest,
        prepared: PreparedFenceChangeSet,
        entry: FenceArtifactEntry,
    ) -> FenceChangeSetInventoryDigest:
        summaries, page_count = collect_visible_change_sets(
            client=self._client, stack_id=prepared.stack_id
        )
        selected = [
            row
            for row in summaries
            if row.get("ChangeSetId") == prepared.change_set_arn
        ]
        if len(selected) != 1:
            _v2_fail("selected prepared change set is not singularly visible")
        for row in summaries:
            if row.get("ChangeSetId") == prepared.change_set_arn:
                if (
                    row.get("Status"),
                    row.get("ExecutionStatus"),
                ) != _V2_SELECTED_CHANGE_SET_STATUS:
                    _v2_fail("selected visible change set is not available")
            elif row.get("Status") not in _V2_TERMINAL_CHANGE_SET_STATUSES:
                _v2_fail("another visible change set is nonterminal")
        snapshots: list[FenceChangeSetEvidence] = []
        captured_at = _v2_now()
        for row in summaries:
            change_set_arn = row.get("ChangeSetId")
            if (
                type(change_set_arn) is not str
                or _V2_CHANGE_SET_ARN.fullmatch(change_set_arn) is None
            ):
                _v2_fail("visible change set omitted an exact ARN")
            description = _success_response(
                _service_method(self._client, "describe_change_set")(
                    ChangeSetName=change_set_arn
                ),
                "DescribeChangeSet",
            )
            if description.get("ChangeSetId") != change_set_arn:
                _v2_fail("visible change-set description changed identity")
            if change_set_arn == prepared.change_set_arn:
                _v2_validate_selected_change_set(
                    description, stack_id=prepared.stack_id, entry=entry
                )
            base = {
                "schema_version": 2,
                "record_type": _V2_RECORD_TYPES["change_set_evidence"],
                "request_identity_sha256": request.canonical_identity_sha256,
                "prepared_identity_sha256": (
                    prepared.canonical_identity_sha256
                ),
                "change_set_arn": change_set_arn,
                "stack_id": prepared.stack_id,
                "status": description.get("Status"),
                "execution_status": description.get("ExecutionStatus"),
                "description": dict(description),
                "captured_at": captured_at,
            }
            snapshot = parse_change_set_evidence(
                _v2_record_with_identity(base)
            )
            snapshots.append(snapshot)
            _service_method(self._record_store, "put_immutable")(
                value=snapshot.to_dict(),
                request_identity_sha256=request.canonical_identity_sha256,
                slot=request.slot,
                child_identity=hashlib.sha256(
                    change_set_arn.encode("ascii")
                ).hexdigest(),
            )
        list_base = {
            "schema_version": 2,
            "record_type": _V2_RECORD_TYPES["change_set_list"],
            "request_identity_sha256": request.canonical_identity_sha256,
            "prepared_identity_sha256": prepared.canonical_identity_sha256,
            "stack_id": prepared.stack_id,
            "selected_change_set_arn": prepared.change_set_arn,
            "visible_count": len(snapshots),
            "page_count": page_count,
            "snapshot_identity_sha256s": [
                snapshot.canonical_identity_sha256
                for snapshot in snapshots
            ],
            "sealed_at": captured_at,
        }
        inventory = parse_change_set_inventory_digest(
            _v2_record_with_identity(list_base)
        )
        _service_method(self._record_store, "put_immutable")(
            value=inventory.to_dict(),
            request_identity_sha256=request.canonical_identity_sha256,
            slot=request.slot,
            child_identity=None,
        )
        return inventory

    def execute_prepared_change_set(
        self,
        *,
        request: FenceTransitionRequest,
        manifest: FenceManifest,
        entry: FenceArtifactEntry,
        prepared: PreparedFenceChangeSet,
        authority: FenceExecuteAuthorityV2,
        prestate: FencePrestateSnapshot,
        freeze_delta_audit: Optional[FreezeExecuteDeltaAudit] = None,
    ) -> FenceExecutionResult:
        _v2_validate_request_binding(
            request=request, manifest=manifest, entry=entry
        )
        _v2_validate_authority_binding(
            request=request,
            manifest=manifest,
            entry=entry,
            authority=authority,
            phase="EXECUTE",
        )
        _v2_validate_prestate_binding(
            request=request,
            manifest=manifest,
            entry=entry,
            prestate=prestate,
            phase="EXECUTE",
        )
        if (
            type(prepared) is not PreparedFenceChangeSet
            or prepared.request_identity_sha256
            != request.canonical_identity_sha256
            or prepared.manifest_identity_sha256
            != manifest.canonical_identity_sha256
            or prepared.entry_identity_sha256
            != entry.entry_identity_sha256
            or prepared.stack_id != entry.request_skeleton["StackName"]
            or prepared.slot is not entry.slot
            or authority.prepared_identity_sha256
            != prepared.canonical_identity_sha256
        ):
            _v2_fail("prepared change-set custody drifted")
        if authority.retained_failover:
            if (
                authority.authority_class
                is not ExecutorAuthorityClass.RETAINED_PRE_SUPPORT
                or entry.slot
                not in {
                    FenceSlot.SOURCE_FAMILIES_FROZEN,
                    FenceSlot.CLOSED_SOURCE,
                }
            ):
                _v2_fail("retained failover is not execute-only freeze/close")
        if entry.slot is FenceSlot.SOURCE_FAMILIES_FROZEN:
            if (
                type(freeze_delta_audit) is not FreezeExecuteDeltaAudit
                or freeze_delta_audit.request_identity_sha256
                != request.canonical_identity_sha256
                or freeze_delta_audit.prepared_identity_sha256
                != prepared.canonical_identity_sha256
            ):
                _v2_fail("pre-armed freeze omitted required-equal delta audit")
        elif freeze_delta_audit is not None:
            _v2_fail("freeze delta audit is forbidden for another slot")
        inventory = self._snapshot_change_sets(
            request=request, prepared=prepared, entry=entry
        )
        execute_token = _v2_client_token(
            request.canonical_identity_sha256,
            prepared.canonical_identity_sha256,
            authority.canonical_identity_sha256,
            "EXECUTE",
        )
        _success_response(
            _service_method(self._client, "execute_change_set")(
                ChangeSetName=prepared.change_set_arn,
                StackName=prepared.stack_id,
                ClientRequestToken=execute_token,
            ),
            "ExecuteChangeSet",
        )
        poststate = _success_response(
            _service_method(self._client, "observe_fence_poststate")(
                StackName=prepared.stack_id,
                LogicalResourceId=FENCE_LOGICAL_ID,
                ExpectedBucketOwner=ACCOUNT_ID,
            ),
            "ObserveFencePoststate",
        )
        required = {
            "stack_id",
            "stack_role_arn",
            "stack_role_id",
            "termination_protection",
            "stack_policy_sha256",
            "original_template_body_sha256",
            "processed_template_body_sha256",
            "direct_policy",
            "first_stable_snapshot_identity_sha256",
            "second_stable_snapshot_identity_sha256",
            "stabilization_first_evidence_sha256",
            "stabilization_second_evidence_sha256",
            "stabilization_identity_sha256",
        }
        if not required.issubset(poststate):
            _v2_fail("poststate observation is incomplete")
        direct_hash = direct_policy_sha256(poststate["direct_policy"])
        if (
            poststate["stack_id"] != prepared.stack_id
            or poststate["stack_role_arn"]
            != prepared.expected_poststate_stack_role_arn
            or poststate["termination_protection"] is not True
            or poststate["original_template_body_sha256"]
            != prepared.template_body_sha256
            or poststate["processed_template_body_sha256"]
            != prepared.template_body_sha256
            or direct_hash != prepared.policy_sha256
            or poststate["stack_role_id"]
            != authority.cloudformation_service_role_id
            or poststate["stack_policy_sha256"]
            != prestate.stack_policy_sha256
            or poststate["first_stable_snapshot_identity_sha256"]
            != poststate["second_stable_snapshot_identity_sha256"]
            or poststate["stabilization_first_evidence_sha256"]
            == poststate["stabilization_second_evidence_sha256"]
        ):
            _v2_fail("poststate role/template/direct-policy identity drifted")
        base = {
            "schema_version": 2,
            "record_type": _V2_RECORD_TYPES["execution_result"],
            "request_identity_sha256": request.canonical_identity_sha256,
            "manifest_identity_sha256": manifest.canonical_identity_sha256,
            "entry_identity_sha256": entry.entry_identity_sha256,
            "prepared_identity_sha256": prepared.canonical_identity_sha256,
            "change_set_inventory_identity_sha256": (
                inventory.canonical_identity_sha256
            ),
            "change_set_arn": prepared.change_set_arn,
            "stack_id": prepared.stack_id,
            "slot": entry.slot.value,
            "execute_authority_identity_sha256": (
                authority.canonical_identity_sha256
            ),
            "execute_authority_class": authority.authority_class.value,
            "execute_api_caller_role_arn": authority.api_caller_role_arn,
            "execute_api_caller_role_id": authority.api_caller_role_id,
            "expected_poststate_stack_role_arn": (
                prepared.expected_poststate_stack_role_arn
            ),
            "observed_poststate_stack_role_arn": poststate["stack_role_arn"],
            "observed_poststate_stack_role_id": poststate["stack_role_id"],
            "original_template_body_sha256": (
                poststate["original_template_body_sha256"]
            ),
            "processed_template_body_sha256": (
                poststate["processed_template_body_sha256"]
            ),
            "poststate_policy_sha256": direct_hash,
            "first_stable_snapshot_identity_sha256": poststate[
                "first_stable_snapshot_identity_sha256"
            ],
            "second_stable_snapshot_identity_sha256": poststate[
                "second_stable_snapshot_identity_sha256"
            ],
            "stabilization_first_evidence_sha256": poststate[
                "stabilization_first_evidence_sha256"
            ],
            "stabilization_second_evidence_sha256": poststate[
                "stabilization_second_evidence_sha256"
            ],
            "stabilization_identity_sha256": poststate[
                "stabilization_identity_sha256"
            ],
            "freeze_delta_classification": (
                None
                if freeze_delta_audit is None
                else freeze_delta_audit.classification
            ),
            "batch_execution_eligible": (
                True
                if freeze_delta_audit is None
                else freeze_delta_audit.batch_execution_eligible
            ),
            "completed_at": _v2_now(),

        }
        result = parse_fence_execution_result(
            _v2_record_with_identity(base)
        )
        _service_method(self._record_store, "put_immutable")(
            value=result.to_dict(),
            request_identity_sha256=request.canonical_identity_sha256,
            slot=request.slot,
            child_identity=None,
        )
        return result
def execute_prepare_v2(
    *,
    executor: FenceExecutor,
    request: FenceTransitionRequest,
    manifest: FenceManifest,
    entry: FenceArtifactEntry,
) -> FenceExecutionResult:
    """Execute only the pinned PREPARE transition through strong record loads."""

    if (
        type(executor) is not FenceExecutor
        or type(request) is not FenceTransitionRequest
        or type(manifest) is not FenceManifest
        or type(entry) is not FenceArtifactEntry
        or request.slot is not FenceSlot.PREPARE_GENESIS_LIVE_STATE
        or entry.slot is not FenceSlot.PREPARE_GENESIS_LIVE_STATE
    ):
        _v2_fail("execute_prepare_v2 accepts only exact PREPARE v2 contracts")
    prepared = executor.prepare_change_set(
        request=request,
        manifest=manifest,
        entry=entry,
        authority=executor.load_create_authority(request),
        prestate=executor.load_prestate_snapshot(request, "CREATE"),
    )
    return executor.execute_prepared_change_set(
        request=request,
        manifest=manifest,
        entry=entry,
        prepared=prepared,
        authority=executor.load_execute_authority(request, prepared),
        prestate=executor.load_prestate_snapshot(request, "EXECUTE"),
    )
