from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json

from botocore.exceptions import ClientError
import pytest

from glm52_enforcement.fence_artifacts import (
    ExecutorAuthorityClass,
    ManifestStage,
    FenceSlot,
)
from glm52_enforcement.fence_executor import (
    DynamoFenceRecordStore,
    FenceExecutorV2Error,
    ReviewedSupportResource,
    SupportDeletionAuthority,
    collect_visible_change_sets,
    derive_support_deletion_request_evidence_sha256,
    direct_policy_sha256,
    load_pinned_fence_entry_template,
    parse_change_set_evidence,
    parse_fence_create_authority,
    parse_fence_execution_result,
    parse_freeze_execute_delta_audit,
    parse_prepared_fence_change_set,
    support_deletion_operation_surface,
)
from glm52_enforcement.records import ledger_pk, ledger_sk
from test_glm52_h1g_fence_artifacts import (
    _build_bootstrap_manifest,
    _build_entries,
    _build_source_manifest,
)


STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:"
    "stack/keep-glm52-h1g-fence/12345678-1234-1234-1234-123456789abc"
)
ACTIVATION_ID = "activation-00000001"
CHANGE_SET_ARN = (
    "arn:aws:cloudformation:us-west-2:246813579024:"
    "changeSet/exact-change-set/12345678-1234-1234-1234-123456789abc"
)
FENCE_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/keep-glm52-h1g-fence-service"
)
MIGRATION_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/keep-glm52-h1g-migration-service"
)
SUPPORT_STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:"
    "stack/keep-glm52-h1g-support/87654321-4321-4321-4321-cba987654321"
)
SUPPORT_DELETION_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/keep-glm52-h1g-support-deletion"
)


def _identified(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["canonical_identity_sha256"] = hashlib.sha256(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    return result


def _prepared_value(
    *,
    slot: FenceSlot = FenceSlot.PREPARE_GENESIS_LIVE_STATE,
    expected_prestate_role: str = MIGRATION_ROLE_ARN,
    expected_poststate_role: str = FENCE_ROLE_ARN,
) -> dict[str, object]:
    return _identified(
        {
            "schema_version": 2,
            "record_type": "glm52_prepared_fence_change_set_v2",
            "request_identity_sha256": "1" * 64,
            "manifest_identity_sha256": "2" * 64,
            "entry_identity_sha256": "3" * 64,
            "prestate_identity_sha256": "4" * 64,
            "request_skeleton_sha256": "5" * 64,
            "change_set_arn": CHANGE_SET_ARN,
            "stack_id": STACK_ID,
            "slot": slot.value,
            "create_authority_identity_sha256": "6" * 64,
            "create_authority_class": "RETAINED_PRE_SUPPORT",
            "create_api_caller_role_arn": (
                "arn:aws:iam::246813579024:"
                "role/keep-glm52-h1g-pre-support-fence-executor"
            ),
            "create_api_caller_role_id": "AROAPRESUPPORT0001",
            "cloudformation_service_role_arn": FENCE_ROLE_ARN,
            "cloudformation_service_role_id": "AROAFENCESERVICE01",
            "template_sha256": "7" * 64,
            "template_body_sha256": "8" * 64,
            "policy_sha256": "9" * 64,
            "expected_prestate_stack_role_arn": expected_prestate_role,
            "expected_poststate_stack_role_arn": expected_poststate_role,
            "selected_change_set_description_sha256": "a" * 64,
            "created_at": "2026-07-31T01:02:03Z",
        }
    )


def _freeze_audit_value(
    *,
    classification: str,
    quarantined: str | None,
    batch_eligible: bool,
) -> dict[str, object]:
    return _identified(
        {
            "schema_version": 2,
            "record_type": "glm52_freeze_execute_delta_audit_v2",
            "request_identity_sha256": "1" * 64,
            "prepared_identity_sha256": "2" * 64,
            "required_equal_before_sha256": "3" * 64,
            "required_equal_after_sha256": "3" * 64,
            "expected_source_delta_sha256": "4" * 64,
            "quarantined_delta_sha256": quarantined,
            "classification": classification,
            "batch_execution_eligible": batch_eligible,
        }
    )


class _InventoryClient:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def list_change_sets(self, **request: object) -> dict[str, object]:
        assert request == {"StackName": STACK_ID}
        return {
            "Summaries": self.rows,
            "IsTruncated": False,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "list-change-sets",
            },
        }


class _PinnedTemplateS3:
    def __init__(self, *, key: str, version_id: str, raw: bytes) -> None:
        self.key = key
        self.version_id = version_id
        self.raw = raw
        self.extra_version = False

    def list_object_versions(self, **request: object) -> dict[str, object]:
        assert request == {
            "Bucket": "keep-glm52-models-246813579024-us-west-2",
            "Prefix": self.key,
            "ExpectedBucketOwner": "246813579024",
        }
        versions = [
            {
                "Key": self.key,
                "VersionId": self.version_id,
                "IsLatest": not self.extra_version,
            }
        ]
        if self.extra_version:
            versions.append(
                {
                    "Key": self.key,
                    "VersionId": "later-version",
                    "IsLatest": True,
                }
            )
        return {
            "Versions": versions,
            "DeleteMarkers": [],
            "IsTruncated": False,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "history",
            },
        }

    def get_object(self, **request: object) -> dict[str, object]:
        assert request == {
            "Bucket": "keep-glm52-models-246813579024-us-west-2",
            "Key": self.key,
            "VersionId": self.version_id,
            "ExpectedBucketOwner": "246813579024",
            "ChecksumMode": "ENABLED",
        }
        return {
            "Body": self.raw,
            "VersionId": self.version_id,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "get-version",
            },
        }


def test_v1_transition_records_are_audit_only_and_execution_ineligible() -> None:
    with pytest.raises(FenceExecutorV2Error, match="fields|v1|non-v2"):
        parse_fence_execution_result(
            {
                "schema_version": 1,
                "record_type": "glm52_task11_fence_result_v1",
            }
        )


def test_manifest_class_matrix_and_service_role_cutover_are_closed() -> None:
    bootstrap = _build_bootstrap_manifest()
    source = _build_source_manifest(bootstrap)
    entries = {
        entry.slot: entry
        for entry in bootstrap.entries + source.entries
    }
    retained = ExecutorAuthorityClass.RETAINED_PRE_SUPPORT.value
    support = ExecutorAuthorityClass.SUPPORT_RUNTIME.value
    expected = {
        FenceSlot.PREPARE_GENESIS_LIVE_STATE: ((retained,), (retained,)),
        FenceSlot.RESERVATION_ONLY: ((support,), (support,)),
        FenceSlot.SOURCE_FAMILIES_FROZEN: (
            (support,),
            (support, retained),
        ),
        FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION: ((support,), (support,)),
        FenceSlot.CLOSED_SOURCE: (
            (support, retained),
            (support, retained),
        ),
        FenceSlot.TERMINAL: ((support,), (support,)),
    }
    migration_role = bootstrap.to_dict()["migration_service_role"]["arn"]
    fence_role = bootstrap.to_dict()["fence_service_role"]["arn"]
    for slot, (create_classes, execute_classes) in expected.items():
        entry = entries[slot]
        value = entry.to_dict()
        assert entry.allowed_create_authority_classes == create_classes
        assert entry.allowed_execute_authority_classes == execute_classes
        assert value["request_skeleton"]["RoleARN"] == fence_role
        assert value["expected_poststate_stack_role_arn"] == fence_role
        assert value["expected_prestate_stack_role_arn"] == (
            migration_role
            if slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE
            else fence_role
        )


def test_direct_policy_parser_rejects_duplicate_and_noncanonical_json() -> None:
    canonical = (
        b'{"Statement":[],"Version":"2012-10-17"}'
    )
    assert direct_policy_sha256(canonical) == hashlib.sha256(
        canonical
    ).hexdigest()
    with pytest.raises(FenceExecutorV2Error, match="duplicate"):
        direct_policy_sha256(
            b'{"Statement":[],"Statement":[],"Version":"2012-10-17"}'
        )
    with pytest.raises(FenceExecutorV2Error, match="canonical"):
        direct_policy_sha256(
            b'{"Version": "2012-10-17", "Statement": []}'
        )


def test_visible_change_set_inventory_accepts_zero_and_sixteen_but_not_seventeen() -> None:
    empty, pages = collect_visible_change_sets(
        client=_InventoryClient([]),
        stack_id=STACK_ID,
    )
    assert empty == ()
    assert pages == 1
    sixteen = [
        {
            "ChangeSetId": CHANGE_SET_ARN.replace(
                "12345678-1234-1234-1234-123456789abc",
                f"12345678-1234-1234-1234-{index:012d}",
            ),
            "Status": "FAILED",
            "ExecutionStatus": "UNAVAILABLE",
        }
        for index in range(16)
    ]
    rows, pages = collect_visible_change_sets(
        client=_InventoryClient(sixteen),
        stack_id=STACK_ID,
    )
    assert len(rows) == 16
    assert pages == 1
    with pytest.raises(
        FenceExecutorV2Error, match="CHANGE_SET_INVENTORY_EXHAUSTED"
    ):
        collect_visible_change_sets(
            client=_InventoryClient(sixteen + [dict(sixteen[0])]),
            stack_id=STACK_ID,
        )


def test_pinned_template_loader_rechecks_singular_history_version_hash_and_url() -> None:
    manifest = _build_bootstrap_manifest()
    entry = _build_entries(
        ManifestStage.BOOTSTRAP,
        bridge_seed_hash=manifest.bridge_seed.policy_sha256,
    )[0]
    assert entry.slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE
    assert (
        entry.entry_identity_sha256
        == manifest.entry(
            FenceSlot.PREPARE_GENESIS_LIVE_STATE
        ).entry_identity_sha256
    )
    raw = entry.template_bytes
    value = entry.to_dict()
    s3 = _PinnedTemplateS3(
        key=value["template_key"],
        version_id=value["version_id"],
        raw=raw,
    )
    loaded = load_pinned_fence_entry_template(s3=s3, entry=entry)
    assert loaded.entry_identity_sha256 == entry.entry_identity_sha256
    assert "?versionId=" in value["template_url"]
    s3.extra_version = True
    with pytest.raises(FenceExecutorV2Error, match="FIXED_KEY_HISTORY_DRIFT"):
        load_pinned_fence_entry_template(s3=s3, entry=entry)


def test_prepared_change_set_owns_dynamic_arn_and_prepare_is_the_only_role_cutover() -> None:
    prepared = parse_prepared_fence_change_set(_prepared_value())
    assert prepared.change_set_arn == CHANGE_SET_ARN
    assert (
        prepared.expected_prestate_stack_role_arn
        == MIGRATION_ROLE_ARN
    )
    assert prepared.expected_poststate_stack_role_arn == FENCE_ROLE_ARN
    extra = _prepared_value()
    extra["static_change_set_arn"] = CHANGE_SET_ARN
    with pytest.raises(FenceExecutorV2Error, match="fields"):
        parse_prepared_fence_change_set(extra)
    with pytest.raises(FenceExecutorV2Error, match="role transition"):
        parse_prepared_fence_change_set(
            _prepared_value(
                slot=FenceSlot.RESERVATION_ONLY,
                expected_prestate_role=MIGRATION_ROLE_ARN,
                expected_poststate_role=FENCE_ROLE_ARN,
            )
        )
    later = parse_prepared_fence_change_set(
        _prepared_value(
            slot=FenceSlot.RESERVATION_ONLY,
            expected_prestate_role=FENCE_ROLE_ARN,
            expected_poststate_role=FENCE_ROLE_ARN,
        )
    )
    assert later.expected_prestate_stack_role_arn == FENCE_ROLE_ARN


def test_authority_parser_rejects_dynamic_arn_and_foreign_class() -> None:
    base = {
        "schema_version": 2,
        "record_type": "glm52_fence_create_authority_v2",
        "request_identity_sha256": "1" * 64,
        "authority_class": ExecutorAuthorityClass.RETAINED_PRE_SUPPORT.value,
        "api_caller_role_arn": (
            "arn:aws:iam::246813579024:"
            "role/keep-glm52-h1g-pre-support-fence-executor"
        ),
        "api_caller_role_id": "AROAPRESUPPORT0001",
        "api_caller_trust_policy_sha256": "2" * 64,
        "api_caller_permission_policy_sha256": "3" * 64,
        "cloudformation_service_role_arn": FENCE_ROLE_ARN,
        "cloudformation_service_role_id": "AROAFENCESERVICE01",
        "cloudformation_service_role_trust_policy_sha256": "4" * 64,
        "cloudformation_service_role_permission_policy_sha256": "5" * 64,
        "action_key": "create-authority",
        "authorized_revision": 1,
        "issued_at": "2026-07-31T01:02:03Z",
    }
    parsed = parse_fence_create_authority(_identified(base))
    assert (
        parsed.authority_class
        is ExecutorAuthorityClass.RETAINED_PRE_SUPPORT
    )
    smuggled = _identified(base)
    smuggled["change_set_arn"] = CHANGE_SET_ARN
    with pytest.raises(FenceExecutorV2Error, match="fields"):
        parse_fence_create_authority(smuggled)
    foreign = dict(base)
    foreign["authority_class"] = "MIGRATION_SERVICE_ROLE"
    with pytest.raises(FenceExecutorV2Error, match="foreign"):
        parse_fence_create_authority(_identified(foreign))


def test_change_set_snapshot_parser_deep_freezes_full_description() -> None:
    base = {
        "schema_version": 2,
        "record_type": "glm52_fence_change_set_evidence_v2",
        "request_identity_sha256": "1" * 64,
        "prepared_identity_sha256": "2" * 64,
        "change_set_arn": CHANGE_SET_ARN,
        "stack_id": STACK_ID,
        "status": "CREATE_COMPLETE",
        "execution_status": "AVAILABLE",
        "description": {
            "Changes": [{"ResourceChange": {"Action": "Modify"}}],
            "ResponseMetadata": {"RequestId": "describe"},
        },
        "captured_at": "2026-07-31T01:02:03Z",
    }
    snapshot = parse_change_set_evidence(_identified(base))
    with pytest.raises(TypeError):
        snapshot.description["Status"] = "FAILED"
    changes = snapshot.description["Changes"]
    with pytest.raises(TypeError):
        changes[0]["ResourceChange"]["Action"] = "Remove"


def test_prearmed_freeze_accepts_expected_delta_and_quarantines_other_drift() -> None:
    expected = parse_freeze_execute_delta_audit(
        _freeze_audit_value(
            classification="FREEZE_EXPECTED_SOURCE_DELTA",
            quarantined=None,
            batch_eligible=True,
        )
    )
    assert expected.batch_execution_eligible is True
    quarantined = parse_freeze_execute_delta_audit(
        _freeze_audit_value(
            classification="FREEZE_QUARANTINED_DRIFT",
            quarantined="5" * 64,
            batch_eligible=False,
        )
    )
    assert quarantined.batch_execution_eligible is False
    with pytest.raises(FenceExecutorV2Error, match="exclude BATCH"):
        parse_freeze_execute_delta_audit(
            _freeze_audit_value(
                classification="FREEZE_QUARANTINED_DRIFT",
                quarantined="5" * 64,
                batch_eligible=True,
            )
        )

class _ConditionalDynamo:
    def __init__(self) -> None:
        self.item: dict[str, object] | None = None

    def put_item(self, **request: object) -> dict[str, object]:
        if self.item is not None:
            raise RuntimeError("ConditionalCheckFailedException")
        self.item = request["Item"]
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "put-immutable",
            }
        }

    def get_item(self, **request: object) -> dict[str, object]:
        assert self.item is not None
        return {
            "Item": self.item,
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "get-immutable",
            },
        }


def test_dynamo_snapshot_rows_are_conditionally_immutable() -> None:
    dynamo = _ConditionalDynamo()
    store = DynamoFenceRecordStore(
        client=dynamo, table_name="glm52-ledger"
    )
    value = _identified(
        {
            "schema_version": 2,
            "record_type": "glm52_fence_change_set_evidence_v2",
            "request_identity_sha256": "1" * 64,
            "prepared_identity_sha256": "2" * 64,
            "change_set_arn": CHANGE_SET_ARN,
            "stack_id": STACK_ID,
            "status": "CREATE_COMPLETE",
            "execution_status": "AVAILABLE",
            "description": {"Changes": []},
            "captured_at": "2026-07-31T01:02:03Z",
        }
    )
    kwargs = {
        "request_identity_sha256": "1" * 64,
        "slot": FenceSlot.PREPARE_GENESIS_LIVE_STATE,
        "child_identity": "2" * 64,
    }
    store.put_immutable(value=value, **kwargs)
    store.put_immutable(value=value, **kwargs)
    changed = dict(value)
    changed["status"] = "FAILED"
    with pytest.raises(RuntimeError, match="ConditionalCheckFailed"):
        store.put_immutable(value=changed, **kwargs)


def _support_authority() -> SupportDeletionAuthority:
    first = (
        ReviewedSupportResource(
            logical_id="SupportLogGroup",
            resource_type="AWS::Logs::LogGroup",
            physical_id="/aws/keep/glm52/support",
            source_api="DescribeStackResource",
            source_response_sha256="1" * 64,
        ),
    )
    resources = first + tuple(
        ReviewedSupportResource(
            logical_id=f"SupportResource{index:03d}",
            resource_type="AWS::Logs::LogGroup",
            physical_id=f"/aws/keep/glm52/support/{index:03d}",
            source_api="DescribeStackResource",
            source_response_sha256=f"{index:064x}",
        )
        for index in range(2, 174)
    )
    inventory_sha256 = hashlib.sha256(
        json.dumps(
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
                for resource in resources
            ],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    authority = SupportDeletionAuthority(
        finalization_state="SUPPORT_FINALIZED",
        support_stack_id=SUPPORT_STACK_ID,
        support_stack_name="keep-glm52-h1g-support",
        deletion_role_arn=SUPPORT_DELETION_ROLE_ARN,
        retained_stack_id=(
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "stack/keep-glm52-gpu/11111111-1111-1111-1111-111111111111"
        ),
        fence_stack_id=STACK_ID,
        lifecycle_action_identity_sha256="9" * 64,
        activation_id=ACTIVATION_ID,
        action_key=ledger_sk(
            "glm52_production_finalization_action",
            activation_id=ACTIVATION_ID,
            action_kind="SUPPORT_DELETE",
            attempt=1,
        ),
        state_machine_version_arn=(
            "arn:aws:states:us-west-2:246813579024:"
            "stateMachine:glm52-h1g-finalization:1"
        ),
        owner_nonce_sha256="7" * 64,
        request_evidence_sha256="0" * 64,
        reviewed_resources=resources,
        reviewed_inventory_sha256=inventory_sha256,
    )
    return replace(
        authority,
        request_evidence_sha256=(
            derive_support_deletion_request_evidence_sha256(authority)
        ),
    )


class _FinalizationReader:
    def __init__(self, authority: SupportDeletionAuthority) -> None:
        self.authority = authority
        self.calls = 0

    def read_finalization_support_delete(
        self, **request: object
    ) -> dict[str, object]:
        self.calls += 1
        assert request == {
            "LedgerPartitionKey": ledger_pk("glm52-sky-20260724"),
            "ActionKey": self.authority.action_key,
            "ActivationId": self.authority.activation_id,
        }
        return {
            "FinalizationState": "SUPPORT_FINALIZED",
            "SupportStackId": self.authority.support_stack_id,
            "ActionKey": self.authority.action_key,
            "LifecycleActionIdentitySha256": (
                self.authority.lifecycle_action_identity_sha256
            ),
            "StateMachineVersionArn": (
                self.authority.state_machine_version_arn
            ),
            "OwnerNonceSha256": self.authority.owner_nonce_sha256,
            "RequestEvidenceSha256": (
                self.authority.request_evidence_sha256
            ),
            "ReviewedInventorySha256": (
                self.authority.reviewed_inventory_sha256
            ),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": f"finalization-{self.calls}",
            },
        }


_MISSING_RESPONSE_METADATA = object()


def _step_functions_client_error(
    code: str,
    metadata: object = _MISSING_RESPONSE_METADATA,
    *,
    operation: str,
) -> ClientError:
    response: dict[str, object] = {
        "Error": {
            "Code": code,
            "Message": f"{operation} rejected",
        },
    }
    if metadata is not _MISSING_RESPONSE_METADATA and metadata is not None:
        response["ResponseMetadata"] = metadata
    error = ClientError(response, operation)
    if metadata is None:
        error.response["ResponseMetadata"] = None
    return error


def _retained_workflow_authority() -> SupportDeletionAuthority:
    return replace(
        _support_authority(),
        state_machine_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-retained-lifecycle:17"
        ),
    )


@pytest.mark.parametrize("status_code", [400, 404])
def test_authenticated_execution_absence_authorizes_one_start(
    status_code: int,
) -> None:
    """Break caught: valid Step Functions absence can no longer start."""

    from glm52_enforcement.fence_executor import (
        SupportDeletionWorkflowAdapter,
    )

    absence = _step_functions_client_error(
        "ExecutionDoesNotExist",
        {
            "HTTPStatusCode": status_code,
            "RequestId": f"describe-absence-{status_code}",
            "RetryAttempts": 0,
        },
        operation="DescribeExecution",
    )

    class Client:
        def __init__(self) -> None:
            self.describe_calls = 0
            self.start_calls = 0

        def describe_execution(self, **request: object) -> object:
            self.describe_calls += 1
            raise absence

        def start_execution(self, **request: object) -> dict[str, object]:
            self.start_calls += 1
            return {
                "executionArn": (
                    "arn:aws:states:us-west-2:246813579024:execution:"
                    "keep-glm52-h1g-retained-lifecycle:"
                    f"{request['name']}"
                ),
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": f"start-{status_code}",
                },
            }

    authority = _retained_workflow_authority()
    client = Client()
    result = SupportDeletionWorkflowAdapter(
        client=client,
        finalization_reader=_FinalizationReader(authority),
    ).start_or_adopt(authority=authority)

    assert result.adopted is False
    assert result.status == "RUNNING"
    assert client.describe_calls == 1
    assert client.start_calls == 1


@pytest.mark.parametrize(
    "metadata",
    [
        _MISSING_RESPONSE_METADATA,
        None,
        {},
        {"RequestId": "request", "RetryAttempts": 0},
        {
            "HTTPStatusCode": 500,
            "RequestId": "request",
            "RetryAttempts": 0,
        },
        {"HTTPStatusCode": 400, "RetryAttempts": 0},
        {
            "HTTPStatusCode": 400,
            "RequestId": "",
            "RetryAttempts": 0,
        },
        {
            "HTTPStatusCode": 400,
            "RequestId": 7,
            "RetryAttempts": 0,
        },
        {"HTTPStatusCode": 400, "RequestId": "request"},
        {
            "HTTPStatusCode": 400,
            "RequestId": "request",
            "RetryAttempts": "0",
        },
        {
            "HTTPStatusCode": 400,
            "RequestId": "request",
            "RetryAttempts": 1,
        },
    ],
    ids=[
        "missing",
        "not-a-mapping",
        "empty",
        "missing-http-status",
        "wrong-http-status",
        "missing-request-id",
        "empty-request-id",
        "non-string-request-id",
        "missing-retry-attempts",
        "malformed-retry-attempts",
        "retried",
    ],
)
def test_unauthenticated_execution_absence_cannot_authorize_start(
    metadata: object,
) -> None:
    """Break caught: a code-only or retried absence starts deletion."""

    from glm52_enforcement.fence_executor import (
        SupportDeletionWorkflowAdapter,
    )

    absence = _step_functions_client_error(
        "ExecutionDoesNotExist",
        metadata,
        operation="DescribeExecution",
    )

    class Client:
        def __init__(self) -> None:
            self.start_calls = 0

        def describe_execution(self, **request: object) -> object:
            raise absence

        def start_execution(self, **request: object) -> object:
            self.start_calls += 1
            raise AssertionError("unauthenticated absence authorized start")

    authority = _retained_workflow_authority()
    client = Client()
    adapter = SupportDeletionWorkflowAdapter(
        client=client,
        finalization_reader=_FinalizationReader(authority),
    )

    with pytest.raises(ValueError, match="not authenticated") as caught:
        adapter.start_or_adopt(authority=authority)
    assert caught.value.__cause__ is absence
    assert client.start_calls == 0


@pytest.mark.parametrize(
    "code",
    [
        "ExecutionDoesNotExistException",
        "ResourceNotFoundException",
        "AccessDeniedException",
    ],
)
def test_unrelated_describe_errors_cannot_authorize_start(code: str) -> None:
    """Break caught: a near-match or unrelated error is treated as absence."""

    from glm52_enforcement.fence_executor import (
        SupportDeletionWorkflowAdapter,
    )

    error = _step_functions_client_error(
        code,
        {
            "HTTPStatusCode": 400,
            "RequestId": f"describe-{code}",
            "RetryAttempts": 0,
        },
        operation="DescribeExecution",
    )

    class Client:
        def __init__(self) -> None:
            self.start_calls = 0

        def describe_execution(self, **request: object) -> object:
            raise error

        def start_execution(self, **request: object) -> object:
            self.start_calls += 1
            raise AssertionError("unrelated describe error authorized start")

    authority = _retained_workflow_authority()
    client = Client()
    adapter = SupportDeletionWorkflowAdapter(
        client=client,
        finalization_reader=_FinalizationReader(authority),
    )

    with pytest.raises(ClientError) as caught:
        adapter.start_or_adopt(authority=authority)
    assert caught.value is error
    assert client.start_calls == 0


@pytest.mark.parametrize("status_code", [400, 404])
def test_authenticated_execution_already_exists_authorizes_adoption(
    status_code: int,
) -> None:
    """Break caught: authenticated idempotent start cannot adopt."""

    from glm52_enforcement.fence_executor import (
        SupportDeletionWorkflowAdapter,
    )

    absence = _step_functions_client_error(
        "ExecutionDoesNotExist",
        {
            "HTTPStatusCode": 404,
            "RequestId": "describe-absence",
            "RetryAttempts": 0,
        },
        operation="DescribeExecution",
    )
    already_exists = _step_functions_client_error(
        "ExecutionAlreadyExists",
        {
            "HTTPStatusCode": status_code,
            "RequestId": f"start-exists-{status_code}",
            "RetryAttempts": 0,
        },
        operation="StartExecution",
    )

    class Client:
        def __init__(self) -> None:
            self.describe_calls = 0
            self.start_calls = 0
            self.start_request: dict[str, object] | None = None

        def describe_execution(
            self, **request: object
        ) -> dict[str, object]:
            self.describe_calls += 1
            if self.describe_calls == 1:
                raise absence
            assert self.start_request is not None
            return {
                "executionArn": request["executionArn"],
                "input": self.start_request["input"],
                "stateMachineVersionArn": self.start_request[
                    "stateMachineArn"
                ],
                "status": "RUNNING",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": f"describe-adopt-{status_code}",
                },
            }

        def start_execution(self, **request: object) -> object:
            self.start_calls += 1
            self.start_request = dict(request)
            raise already_exists

    authority = _retained_workflow_authority()
    client = Client()
    result = SupportDeletionWorkflowAdapter(
        client=client,
        finalization_reader=_FinalizationReader(authority),
    ).start_or_adopt(authority=authority)

    assert result.adopted is True
    assert result.status == "RUNNING"
    assert client.describe_calls == 2
    assert client.start_calls == 1


@pytest.mark.parametrize(
    "metadata",
    [
        _MISSING_RESPONSE_METADATA,
        None,
        {},
        {"RequestId": "request", "RetryAttempts": 0},
        {
            "HTTPStatusCode": 500,
            "RequestId": "request",
            "RetryAttempts": 0,
        },
        {"HTTPStatusCode": 400, "RetryAttempts": 0},
        {
            "HTTPStatusCode": 400,
            "RequestId": "",
            "RetryAttempts": 0,
        },
        {
            "HTTPStatusCode": 400,
            "RequestId": 7,
            "RetryAttempts": 0,
        },
        {"HTTPStatusCode": 400, "RequestId": "request"},
        {
            "HTTPStatusCode": 400,
            "RequestId": "request",
            "RetryAttempts": "0",
        },
        {
            "HTTPStatusCode": 400,
            "RequestId": "request",
            "RetryAttempts": 1,
        },
    ],
    ids=[
        "missing",
        "not-a-mapping",
        "empty",
        "missing-http-status",
        "wrong-http-status",
        "missing-request-id",
        "empty-request-id",
        "non-string-request-id",
        "missing-retry-attempts",
        "malformed-retry-attempts",
        "retried",
    ],
)
def test_unauthenticated_execution_already_exists_cannot_authorize_adoption(
    metadata: object,
) -> None:
    """Break caught: a code-only or retried collision adopts execution."""

    from glm52_enforcement.fence_executor import (
        SupportDeletionWorkflowAdapter,
    )

    absence = _step_functions_client_error(
        "ExecutionDoesNotExist",
        {
            "HTTPStatusCode": 404,
            "RequestId": "describe-absence",
            "RetryAttempts": 0,
        },
        operation="DescribeExecution",
    )
    already_exists = _step_functions_client_error(
        "ExecutionAlreadyExists",
        metadata,
        operation="StartExecution",
    )

    class Client:
        def __init__(self) -> None:
            self.describe_calls = 0
            self.start_calls = 0

        def describe_execution(self, **request: object) -> object:
            self.describe_calls += 1
            if self.describe_calls == 1:
                raise absence
            raise AssertionError(
                "unauthenticated already-exists authorized adoption"
            )

        def start_execution(self, **request: object) -> object:
            self.start_calls += 1
            raise already_exists

    authority = _retained_workflow_authority()
    client = Client()
    adapter = SupportDeletionWorkflowAdapter(
        client=client,
        finalization_reader=_FinalizationReader(authority),
    )

    with pytest.raises(ValueError, match="not authenticated") as caught:
        adapter.start_or_adopt(authority=authority)
    assert caught.value.__cause__ is already_exists
    assert client.describe_calls == 1
    assert client.start_calls == 1


@pytest.mark.parametrize(
    "code",
    [
        "ExecutionAlreadyExistsException",
        "ConflictException",
        "AccessDeniedException",
    ],
)
def test_unrelated_start_errors_cannot_authorize_adoption(code: str) -> None:
    """Break caught: a near-match or unrelated start error is adopted."""

    from glm52_enforcement.fence_executor import (
        SupportDeletionWorkflowAdapter,
    )

    absence = _step_functions_client_error(
        "ExecutionDoesNotExist",
        {
            "HTTPStatusCode": 404,
            "RequestId": "describe-absence",
            "RetryAttempts": 0,
        },
        operation="DescribeExecution",
    )
    start_error = _step_functions_client_error(
        code,
        {
            "HTTPStatusCode": 400,
            "RequestId": f"start-{code}",
            "RetryAttempts": 0,
        },
        operation="StartExecution",
    )

    class Client:
        def __init__(self) -> None:
            self.describe_calls = 0

        def describe_execution(self, **request: object) -> object:
            self.describe_calls += 1
            if self.describe_calls == 1:
                raise absence
            raise AssertionError("unrelated start error authorized adoption")

        def start_execution(self, **request: object) -> object:
            raise start_error

    authority = _retained_workflow_authority()
    client = Client()
    adapter = SupportDeletionWorkflowAdapter(
        client=client,
        finalization_reader=_FinalizationReader(authority),
    )

    with pytest.raises(ClientError) as caught:
        adapter.start_or_adopt(authority=authority)
    assert caught.value is start_error
    assert client.describe_calls == 1








def test_reviewed_support_inventory_requires_authenticated_full_cfn_detail() -> None:
    """Break caught: a partial self-asserted resource list authorizes deletion."""

    resource = ReviewedSupportResource(
        logical_id="SupportResource001",
        resource_type="AWS::Logs::LogGroup",
        physical_id="/aws/keep/glm52/support/001",
        source_api="DescribeStackResource",
        source_response_sha256="a" * 64,
    )
    resources = tuple(
        replace(
            resource,
            logical_id=f"SupportResource{index:03d}",
            physical_id=f"/aws/keep/glm52/support/{index:03d}",
            source_response_sha256=f"{index:064x}",
        )
        for index in range(1, 174)
    )
    inventory_sha256 = hashlib.sha256(
        json.dumps(
            [
                {
                    "logical_id": row.logical_id,
                    "resource_type": row.resource_type,
                    "physical_id": row.physical_id,
                    "source_api": row.source_api,
                    "source_response_sha256": (
                        row.source_response_sha256
                    ),
                }
                for row in resources
            ],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    base = _support_authority()
    full = replace(
        base,
        reviewed_resources=resources,
        reviewed_inventory_sha256=inventory_sha256,
        request_evidence_sha256="0" * 64,
    )
    full = replace(
        full,
        request_evidence_sha256=(
            derive_support_deletion_request_evidence_sha256(full)
        ),
    )
    assert len(full.reviewed_resources) == 173
    with pytest.raises(ValueError):
        replace(
            full,
            reviewed_resources=full.reviewed_resources[:-1],
        )
















def test_support_deletion_rejects_non_uuid_retained_stack_identity() -> None:
    authority = _support_authority()
    with pytest.raises(ValueError):
        SupportDeletionAuthority(
            **{
                **authority.__dict__,
                "retained_stack_id": (
                    "arn:aws:cloudformation:us-west-2:246813579024:"
                    "stack/keep-glm52-gpu/not-a-uuid"
                ),
            }
        )






def test_support_deletion_authority_cannot_target_retained_or_fence_stack() -> None:
    """Break caught: teardown authority is retargeted outside support."""

    authority = _support_authority()
    for alternate in (authority.retained_stack_id, authority.fence_stack_id):
        with pytest.raises(ValueError):
            SupportDeletionAuthority(
                **{
                    **authority.__dict__,
                    "support_stack_id": alternate,
                }
            )
    with pytest.raises(ValueError):
        SupportDeletionAuthority(
            **{
                **authority.__dict__,
                "deletion_role_arn": (
                    "arn:aws:iam::246813579024:"
                    "role/keep-glm52-h1g-retained-support-lifecycle"
                ),
            }
        )


def test_standard_deletion_requires_stable_protected_prestate_and_exact_version() -> None:
    """Break caught: an unstable stack or alias/unqualified workflow can mutate."""

    from glm52_enforcement.fence_executor import (
        SupportDeletionWorkflowAdapter,
        build_support_deletion_workflow_definition,
    )

    definition = build_support_deletion_workflow_definition()
    states = definition["States"]
    choice = states["RequireStableProtectedSupportStack"]
    assert choice["Default"] == "ProtectedSupportPrestateInvalid"
    conditions = choice["Choices"][0]["And"]
    assert {
        condition.get("StringEquals")
        for condition in conditions
        if "StringEquals" in condition
    } == {"UPDATE_COMPLETE"}
    assert {
        condition.get("BooleanEquals")
        for condition in conditions
        if "BooleanEquals" in condition
    } == {True}
    assert states["DisableSupportTerminationProtection"]["Catch"][0][
        "Next"
    ] == "ReconcileTerminationProtectionFalse"
    assert states["DeleteSupportStack"]["Catch"][0]["Next"] == (
        "WaitForSupportStackAbsence"
    )
    assert states["RequireExactAbsentError"]["Choices"][0]["And"][1][
        "StringMatches"
    ] == "*does not exist*"

    class NoCalls:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"foreign workflow reached client: {name}")

    authority = _support_authority()
    for foreign in (
        (
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-retained-lifecycle"
        ),
        (
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-retained-lifecycle:PROD"
        ),
        (
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "foreign-retained-lifecycle:17"
        ),
    ):
        foreign_authority = replace(
            authority,
            state_machine_version_arn=foreign,
        )
        adapter = SupportDeletionWorkflowAdapter(
            client=NoCalls(),
            finalization_reader=_FinalizationReader(foreign_authority),
        )
        with pytest.raises(ValueError, match="exact retained version"):
            adapter.start_or_adopt(
                authority=foreign_authority
            )
    adapter = SupportDeletionWorkflowAdapter(
        client=NoCalls(),
        finalization_reader=_FinalizationReader(authority),
    )
    with pytest.raises(TypeError):
        adapter.start_or_adopt(authority=object())


def test_support_delete_mutations_are_owned_by_one_exact_standard_execution() -> None:
    """Break caught: a local commit can precede either CloudFormation call."""

    from glm52_enforcement.fence_executor import (
        SupportDeletionWorkflowAdapter,
        build_support_deletion_workflow_definition,
    )

    definition = build_support_deletion_workflow_definition()
    states = definition["States"]
    mutation_resources = {
        "DisableSupportTerminationProtection": (
            "arn:aws:states:::aws-sdk:cloudformation:"
            "updateTerminationProtection"
        ),
        "DeleteSupportStack": (
            "arn:aws:states:::aws-sdk:cloudformation:deleteStack"
        ),
    }
    for state_name, resource in mutation_resources.items():
        state = states[state_name]
        assert state["Type"] == "Task"
        assert state["Resource"] == resource
        assert "Retry" not in state
        assert all(
            catcher["Next"] != state_name
            for catcher in state.get("Catch", [])
        )

    class ProcessLost(BaseException):
        pass

    class DurableStandardClient:
        def __init__(self) -> None:
            self.execution: dict[str, object] | None = None
            self.start_calls = 0
            self.describe_calls = 0

        def describe_execution(
            self, **request: object
        ) -> dict[str, object]:
            self.describe_calls += 1
            if (
                self.execution is None
                or request != {
                    "executionArn": self.execution["executionArn"]
                }
            ):
                return {
                    "exists": False,
                    "ResponseMetadata": {
                        "HTTPStatusCode": 200,
                        "RequestId": f"describe-{self.describe_calls}",
                    },
                }
            return {
                **self.execution,
                "exists": True,
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": f"describe-{self.describe_calls}",
                },
            }

        def start_execution(self, **request: object) -> dict[str, object]:
            self.start_calls += 1
            assert request["stateMachineArn"].endswith(":17")
            execution_arn = (
                "arn:aws:states:us-west-2:246813579024:execution:"
                f"keep-glm52-h1g-retained-lifecycle:{request['name']}"
            )
            self.execution = {
                "executionArn": execution_arn,
                "name": request["name"],
                "input": request["input"],
                "stateMachineVersionArn": request["stateMachineArn"],
                "status": "RUNNING",
            }
            raise ProcessLost("after Step Functions accepted the execution")

    authority = replace(
        _support_authority(),
        state_machine_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-retained-lifecycle:17"
        ),
    )
    client = DurableStandardClient()
    adapter = SupportDeletionWorkflowAdapter(
        client=client,
        finalization_reader=_FinalizationReader(authority),
    )

    with pytest.raises(ProcessLost):
        adapter.start_or_adopt(authority=authority)
    adopted = adapter.start_or_adopt(authority=authority)

    assert adopted.status == "RUNNING"
    assert adopted.adopted is True
    assert client.start_calls == 1
    assert not hasattr(client, "update_termination_protection")
    assert not hasattr(client, "delete_stack")


def test_process_local_support_delete_adapter_is_not_production_reachable() -> None:
    """Break caught: production can still call CloudFormation after a local commit."""

    import glm52_enforcement.fence_executor as fence_executor

    assert (
        fence_executor.SupportDeletionAdapter
        is fence_executor.SupportDeletionWorkflowAdapter
    )
    source = inspect.getsource(fence_executor.SupportDeletionAdapter)
    assert "update_termination_protection" not in source
    assert "delete_stack" not in source


def test_syntactic_finalization_without_live_readback_has_zero_effect() -> None:
    """Break caught: a caller-constructed authority starts deletion by itself."""

    from glm52_enforcement.fence_executor import (
        SupportDeletionWorkflowAdapter,
    )

    class NoExecutionCalls:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"syntactic authority reached {name}")

    adapter = SupportDeletionWorkflowAdapter(client=NoExecutionCalls())
    authority = replace(
        _support_authority(),
        state_machine_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-retained-lifecycle:17"
        ),
    )
    with pytest.raises(ValueError, match="live finalization"):
        adapter.start_or_adopt(authority=authority)


def test_ambiguous_mutation_paths_are_read_only_and_cannot_restart_a_task() -> None:
    """Break caught: an ASL Catch or loop resubmits either CFN mutation."""

    from glm52_enforcement.fence_executor import (
        build_support_deletion_workflow_definition,
    )

    states = build_support_deletion_workflow_definition()["States"]
    update = states["DisableSupportTerminationProtection"]
    delete = states["DeleteSupportStack"]
    assert update["Catch"] == [
        {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.update_ambiguity",
            "Next": "ReconcileTerminationProtectionFalse",
        }
    ]
    assert delete["Catch"] == [
        {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.delete_ambiguity",
            "Next": "WaitForSupportStackAbsence",
        }
    ]
    assert states["ReconcileTerminationProtectionFalse"]["Resource"].endswith(
        "cloudformation:describeStacks"
    )
    assert states["DescribeSupportStackForAbsence"]["Resource"].endswith(
        "cloudformation:describeStacks"
    )
    for state_name, state in states.items():
        if state_name not in {
            "DisableSupportTerminationProtection",
            "DeleteSupportStack",
        }:
            assert state.get("Next") not in {
                "DisableSupportTerminationProtection",
                "DeleteSupportStack",
            }


def test_mutation_states_are_outside_every_workflow_cycle() -> None:
    """Break caught: a Choice/Catch edge makes a CFN Task resubmittable."""

    from glm52_enforcement.fence_executor import (
        build_support_deletion_workflow_definition,
    )

    states = build_support_deletion_workflow_definition()["States"]

    def successors(state: dict[str, object]) -> set[str]:
        result = set()
        if isinstance(state.get("Next"), str):
            result.add(state["Next"])
        if isinstance(state.get("Default"), str):
            result.add(state["Default"])
        for branch in state.get("Choices", []):
            if isinstance(branch.get("Next"), str):
                result.add(branch["Next"])
        for catcher in state.get("Catch", []):
            if isinstance(catcher.get("Next"), str):
                result.add(catcher["Next"])
        return result

    graph = {name: successors(state) for name, state in states.items()}

    def reachable(start: str) -> set[str]:
        seen = set()
        pending = list(graph[start])
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(graph.get(current, ()))
        return seen

    update = "DisableSupportTerminationProtection"
    delete = "DeleteSupportStack"
    assert update not in reachable(update)
    assert delete not in reachable(delete)
    assert update not in reachable(
        states[update]["Catch"][0]["Next"]
    )
    assert {update, delete}.isdisjoint(
        reachable(states[delete]["Catch"][0]["Next"])
    )


def test_standard_deletion_history_proves_one_shot_tasks_and_terminal_absence() -> None:
    """Break caught: a successful execution lacks exact history/absence proof."""

    from glm52_enforcement.fence_executor import (
        SupportDeletionWorkflowAdapter,
    )

    class States:
        def __init__(self) -> None:
            self.execution: dict[str, object] | None = None

        def describe_execution(
            self, **request: object
        ) -> dict[str, object]:
            if self.execution is None:
                return {
                    "exists": False,
                    "ResponseMetadata": {
                        "HTTPStatusCode": 200,
                        "RequestId": "describe-absent",
                    },
                }
            assert request == {
                "executionArn": self.execution["executionArn"]
            }
            return {
                **self.execution,
                "exists": True,
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "describe-present",
                },
            }

        def start_execution(self, **request: object) -> dict[str, object]:
            execution_arn = (
                "arn:aws:states:us-west-2:246813579024:execution:"
                f"keep-glm52-h1g-retained-lifecycle:{request['name']}"
            )
            self.execution = {
                "executionArn": execution_arn,
                "input": request["input"],
                "stateMachineVersionArn": request["stateMachineArn"],
                "status": "RUNNING",
            }
            return {
                "executionArn": execution_arn,
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "start",
                },
            }

        def get_execution_history(
            self, **request: object
        ) -> dict[str, object]:
            expected = {
                "executionArn": self.execution["executionArn"],
                "includeExecutionData": True,
                "maxResults": 1000,
                "reverseOrder": False,
            }
            page_two = request.pop("nextToken", None)
            assert request == expected
            events = [
                {"id": 1, "type": "ExecutionStarted"},
                {
                    "id": 2,
                    "type": "TaskScheduled",
                    "taskScheduledEventDetails": {
                        "resource": (
                            "arn:aws:states:::aws-sdk:cloudformation:"
                            "updateTerminationProtection"
                        )
                    },
                },
                {"id": 3, "type": "TaskSucceeded"},
                {
                    "id": 4,
                    "type": "TaskScheduled",
                    "taskScheduledEventDetails": {
                        "resource": (
                            "arn:aws:states:::aws-sdk:cloudformation:"
                            "deleteStack"
                        )
                    },
                },
                {"id": 5, "type": "TaskSucceeded"},
                {
                    "id": 6,
                    "type": "SucceedStateEntered",
                    "stateEnteredEventDetails": {
                        "name": "SupportStackAbsenceConfirmed"
                    },
                },
                {"id": 7, "type": "ExecutionSucceeded"},
            ]
            return {
                "events": events[4:] if page_two else events[:4],
                **({} if page_two else {"nextToken": "page-2"}),
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "history-2" if page_two else "history-1",
                },
            }

    class AbsentError(Exception):
        def __init__(self, stack_id: str) -> None:
            super().__init__("absent")
            self.response = {
                "Error": {
                    "Code": "ValidationError",
                    "Message": f"Stack with id {stack_id} does not exist",
                }
            }

    class CloudFormationRead:
        def __init__(self) -> None:
            self.replacement_present = False

        def describe_stacks(self, **request: object) -> object:
            raise AbsentError(request["StackName"])

        def audit_support_resources(
            self, **request: object
        ) -> dict[str, object]:
            resources = [
                {**resource, "Status": "ABSENT"}
                for resource in request["Resources"]
            ]
            if self.replacement_present:
                resources[0]["PhysicalResourceId"] = (
                    "/aws/keep/glm52/replacement"
                )
                resources[0]["Status"] = "PRESENT"
            return {
                "StackId": request["StackId"],
                "ObservationId": "all-173-absent",
                "Resources": resources,
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "RequestId": "residual-audit",
                },
            }

    authority = replace(
        _support_authority(),
        state_machine_version_arn=(
            "arn:aws:states:us-west-2:246813579024:stateMachine:"
            "keep-glm52-h1g-retained-lifecycle:17"
        ),
    )
    states = States()
    read_client = CloudFormationRead()
    adapter = SupportDeletionWorkflowAdapter(
        client=states,
        read_client=read_client,
        finalization_reader=_FinalizationReader(authority),
    )
    adapter.start_or_adopt(authority=authority)
    states.execution["status"] = "SUCCEEDED"
    result = adapter.reconcile(authority=authority)
    assert result.history_pages == 2
    assert result.history_events == 7
    assert result.update_task_schedules == 1
    assert result.delete_task_schedules == 1
    assert result.terminal_absence_confirmed is True
    read_client.replacement_present = True
    with pytest.raises(ValueError, match="absence"):
        adapter.reconcile(authority=authority)
    assert support_deletion_operation_surface() == (
        "ReadExactFinalizationSupportDelete",
        "DescribeExactRetainedLifecycleExecution",
        "StartExactRetainedLifecycleVersion",
        "GetRetainedLifecycleExecutionHistory",
        "DescribeExactSupportStackAbsence",
    )
