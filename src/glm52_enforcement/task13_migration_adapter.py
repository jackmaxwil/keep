"""Production Task 13 port for the reviewed seven-operation stack migration."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping
from urllib.parse import quote

from .canonical import canonical_json_bytes

if TYPE_CHECKING:
    from .fence_executor import FenceExecutionResult
    from .task13_staged_deployment import DisabledSupportDeploymentEvidence
from .cloudformation_stacks import (
    ACCOUNT_ID,
    REGION,
    RUN_ID,
    BootstrapCoordinate,
    CanonicalMigrationStateStore,
    MigrationBootstrapResult,
    MigrationBundle,
    MigrationCoordinator,
    MigrationEvidence,
    MigrationExecutionAuthority,
    MigrationExecutionResult,
    MigrationRoleEvidenceV2,
    StackIdentity,
    StackKind,
    FENCE_SERVICE_ROLE_ARN_V2,
    MIGRATION_SERVICE_ROLE_ARN_V2,
    STACK_MIGRATION_TRANSFER_CHECKPOINT_V2,
    StackMigrationAuthorityV2,
    StackMigrationEvidenceV2,
    StackMigrationTransferBundleV2,
    StackMigrationTransferCheckpointV2,
    TemplateArtifact,
    build_stack_migration_transfer_checkpoint_v2,
    build_stack_migration_v2,
    initial_migration_execution_state,
    initial_stack_migration_execution_state_v2,
    migration_execution_state_from_projection,
    migration_execution_state_projection,
    parse_stack_migration_transfer_checkpoint_v2,
)
from .dynamodb import decode_item, encode_item


LEDGER_TABLE_NAME = "keep-glm52-h1g-ledger-v1"
_STATE_PREFIX = "H1G_STACK_MIGRATION#"
_SHA256 = frozenset("0123456789abcdef")
MIGRATION_TEMPLATE_KEYS: Mapping[str, str] = {
    "bridge-seed-owner": "task13/migration/bridge-seed-owner.json",
    "retention-only": "task13/migration/retention-only.json",
    "post-retain": "task13/migration/post-retain.json",
    "fence-import": "task13/migration/fence-import.json",
    "fence-transfer": "task13/migration/fence-transfer.json",
    "disabled-support": "task13/migration/disabled-support.json",
}


class Task13MigrationAdapterError(ValueError):
    """The sealed migration input or durable state failed closed."""


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise Task13MigrationAdapterError(
            label + " is not one lowercase SHA-256"
        )
    return value


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise Task13MigrationAdapterError(
            label + " is not one exact JSON object"
        )
    return dict(value)


def _success(value: object, operation: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise Task13MigrationAdapterError(
            operation + " returned no exact object"
        )
    metadata = value.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") not in {200, 201}
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or metadata.get("RetryAttempts") != 0
    ):
        raise Task13MigrationAdapterError(
            operation + " lacks zero-retry success metadata"
        )
    return value


def _error_metadata(error: BaseException, operation: str) -> None:
    response = getattr(error, "response", None)
    metadata = (
        response.get("ResponseMetadata")
        if type(response) is dict
        else None
    )
    if (
        type(metadata) is not dict
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
        or type(metadata.get("HTTPStatusCode")) is not int
        or metadata.get("RetryAttempts") != 0
    ):
        raise Task13MigrationAdapterError(
            operation + " error lacks zero-retry metadata"
        ) from error


def _stack_identity(value: object) -> StackIdentity:
    item = _mapping(value, "migration stack identity")
    if set(item) != {
        "kind",
        "name",
        "stack_id",
        "account_id",
        "region",
        "termination_protection",
        "tags",
    }:
        raise Task13MigrationAdapterError(
            "migration stack identity fields are not exact"
        )
    tags = _mapping(item["tags"], "migration stack tags")
    return StackIdentity(
        kind=StackKind(str(item["kind"])),
        name=str(item["name"]),
        stack_id=str(item["stack_id"]),
        account_id=str(item["account_id"]),
        region=str(item["region"]),
        termination_protection=item["termination_protection"],
        tags=tuple(sorted(tags.items())),
    )


def _bootstrap_coordinate(
    value: object,
    *,
    expected_kind: StackKind,
) -> BootstrapCoordinate:
    item = _mapping(value, "migration bootstrap coordinate")
    if set(item) != {"kind", "template_url", "version_id"}:
        raise Task13MigrationAdapterError(
            "migration bootstrap coordinate fields are not exact"
        )
    coordinate = BootstrapCoordinate(
        kind=StackKind(str(item["kind"])),
        template_url=str(item["template_url"]),
        version_id=str(item["version_id"]),
    )
    if coordinate.kind is not expected_kind:
        raise Task13MigrationAdapterError(
            "migration bootstrap coordinate kind drifted"
        )
    return coordinate


def parse_execution_authority(
    value: object,
) -> MigrationExecutionAuthority:
    item = _mapping(value, "migration execution authority")
    if set(item) != {
        "deployment_role_arn",
        "deployment_role_id",
        "fence_bootstrap",
        "support_bootstrap",
        "import_change_set_name",
        "action_identity_sha256",
    }:
        raise Task13MigrationAdapterError(
            "migration execution authority fields are not exact"
        )
    return MigrationExecutionAuthority(
        deployment_role_arn=str(item["deployment_role_arn"]),
        deployment_role_id=str(item["deployment_role_id"]),
        fence_bootstrap=_bootstrap_coordinate(
            item["fence_bootstrap"],
            expected_kind=StackKind.FENCE,
        ),
        support_bootstrap=_bootstrap_coordinate(
            item["support_bootstrap"],
            expected_kind=StackKind.SUPPORT,
        ),
        import_change_set_name=str(item["import_change_set_name"]),
        action_identity_sha256=str(item["action_identity_sha256"]),
    )


def parse_migration_evidence(value: object) -> MigrationEvidence:
    item = _mapping(value, "migration evidence")
    if set(item) != {
        "retained",
        "fence",
        "support",
        "current_policy_logical_id",
        "current_policy_physical_id",
        "current_policy_stack_id",
        "bucket_name",
        "import_identifier",
        "live_policy",
        "direct_policy_readbacks",
        "retained_deployment_role_id",
        "retained_resource_physical_ids",
        "retained_export_names",
        "support_export_names",
    }:
        raise Task13MigrationAdapterError(
            "migration evidence fields are not exact"
        )
    import_identifier = _mapping(
        item["import_identifier"],
        "migration import identifier",
    )
    direct = item["direct_policy_readbacks"]
    if type(direct) is not list:
        raise Task13MigrationAdapterError(
            "migration direct policy readbacks are not exact"
        )
    tuples: dict[str, tuple[str, ...]] = {}
    for field in (
        "retained_resource_physical_ids",
        "retained_export_names",
        "support_export_names",
    ):
        values = item[field]
        if (
            type(values) is not list
            or any(type(entry) is not str or not entry for entry in values)
        ):
            raise Task13MigrationAdapterError(
                field + " is not one exact string array"
            )
        tuples[field] = tuple(values)
    return MigrationEvidence(
        retained=_stack_identity(item["retained"]),
        fence=_stack_identity(item["fence"]),
        support=_stack_identity(item["support"]),
        current_policy_logical_id=str(
            item["current_policy_logical_id"]
        ),
        current_policy_physical_id=str(
            item["current_policy_physical_id"]
        ),
        current_policy_stack_id=str(item["current_policy_stack_id"]),
        bucket_name=str(item["bucket_name"]),
        import_identifier=tuple(sorted(import_identifier.items())),
        live_policy=_mapping(item["live_policy"], "live bucket policy"),
        direct_policy_readbacks=tuple(
            _mapping(readback, "direct bucket-policy readback")
            for readback in direct
        ),
        retained_deployment_role_id=str(
            item["retained_deployment_role_id"]
        ),
        retained_resource_physical_ids=tuples[
            "retained_resource_physical_ids"
        ],
        retained_export_names=tuples["retained_export_names"],
        support_export_names=tuples["support_export_names"],
    )


def parse_bootstrap_result(value: object) -> MigrationBootstrapResult:
    item = _mapping(value, "migration bootstrap result")
    if set(item) != {
        "fence",
        "support",
        "action_identity_sha256",
        "state_revision",
        "reconciled_creates",
    }:
        raise Task13MigrationAdapterError(
            "migration bootstrap result fields are not exact"
        )
    reconciled = item["reconciled_creates"]
    if type(reconciled) is not list:
        raise Task13MigrationAdapterError(
            "migration reconciled creates are not exact"
        )
    return MigrationBootstrapResult(
        fence=_stack_identity(item["fence"]),
        support=_stack_identity(item["support"]),
        action_identity_sha256=str(item["action_identity_sha256"]),
        state_revision=item["state_revision"],
        reconciled_creates=tuple(
            StackKind(str(kind)) for kind in reconciled
        ),
    )


def parse_migration_bundle(value: object) -> MigrationBundle:
    item = _mapping(value, "migration bundle")
    if set(item) != {"manifest", "artifacts"}:
        raise Task13MigrationAdapterError(
            "migration bundle fields are not exact"
        )
    manifest = _mapping(item["manifest"], "migration manifest")
    rows = item["artifacts"]
    if type(rows) is not list:
        raise Task13MigrationAdapterError(
            "migration artifact inventory is not exact"
        )
    artifacts = []
    for value_row in rows:
        row = _mapping(value_row, "migration template artifact")
        if set(row) != {
            "stage",
            "stack_id",
            "template_url",
            "version_id",
            "body_base64",
            "sha256",
            "template_body_sha256",
            "policy_sha256",
            "template",
        }:
            raise Task13MigrationAdapterError(
                "migration template artifact fields are not exact"
            )
        try:
            body = base64.b64decode(
                str(row["body_base64"]),
                validate=True,
            )
        except (ValueError, TypeError) as exc:
            raise Task13MigrationAdapterError(
                "migration template body encoding is invalid"
            ) from exc
        template = _mapping(row["template"], "migration template")
        sha = _sha256(row["sha256"], "migration artifact identity")
        template_sha = _sha256(
            row["template_body_sha256"],
            "migration template identity",
        )
        policy_sha = row["policy_sha256"]
        if (
            hashlib.sha256(body).hexdigest() != sha
            or body != canonical_json_bytes(template) + b"\n"
            or hashlib.sha256(canonical_json_bytes(template)).hexdigest()
            != template_sha
            or (
                policy_sha is not None
                and _sha256(
                    policy_sha,
                    "migration policy identity",
                )
                != policy_sha
            )
        ):
            raise Task13MigrationAdapterError(
                "migration artifact bytes or identity drifted"
            )
        artifacts.append(
            TemplateArtifact(
                stage=str(row["stage"]),
                stack_id=str(row["stack_id"]),
                template_url=str(row["template_url"]),
                version_id=str(row["version_id"]),
                body=body,
                sha256=sha,
                template_body_sha256=template_sha,
                policy_sha256=policy_sha,
                template=template,
            )
        )
    return MigrationBundle(
        artifacts=tuple(artifacts),
        manifest=manifest,
    )


def execution_authority_projection(
    value: MigrationExecutionAuthority,
) -> Mapping[str, object]:
    if type(value) is not MigrationExecutionAuthority:
        raise TypeError("execution authority projection requires exact value")
    return {
        "deployment_role_arn": value.deployment_role_arn,
        "deployment_role_id": value.deployment_role_id,
        "fence_bootstrap": {
            "kind": value.fence_bootstrap.kind.value,
            "template_url": value.fence_bootstrap.template_url,
            "version_id": value.fence_bootstrap.version_id,
        },
        "support_bootstrap": {
            "kind": value.support_bootstrap.kind.value,
            "template_url": value.support_bootstrap.template_url,
            "version_id": value.support_bootstrap.version_id,
        },
        "import_change_set_name": value.import_change_set_name,
        "action_identity_sha256": value.action_identity_sha256,
    }


def _stack_identity_projection(
    value: StackIdentity,
) -> Mapping[str, object]:
    if type(value) is not StackIdentity:
        raise TypeError("stack identity projection requires exact value")
    return {
        "kind": value.kind.value,
        "name": value.name,
        "stack_id": value.stack_id,
        "account_id": value.account_id,
        "region": value.region,
        "termination_protection": value.termination_protection,
        "tags": dict(value.tags),
    }


def migration_evidence_projection(
    value: MigrationEvidence,
) -> Mapping[str, object]:
    if type(value) is not MigrationEvidence:
        raise TypeError("migration evidence projection requires exact value")
    return {
        "retained": _stack_identity_projection(value.retained),
        "fence": _stack_identity_projection(value.fence),
        "support": _stack_identity_projection(value.support),
        "current_policy_logical_id": value.current_policy_logical_id,
        "current_policy_physical_id": value.current_policy_physical_id,
        "current_policy_stack_id": value.current_policy_stack_id,
        "bucket_name": value.bucket_name,
        "import_identifier": dict(value.import_identifier),
        "live_policy": dict(value.live_policy),
        "direct_policy_readbacks": [
            dict(readback) for readback in value.direct_policy_readbacks
        ],
        "retained_deployment_role_id": (
            value.retained_deployment_role_id
        ),
        "retained_resource_physical_ids": list(
            value.retained_resource_physical_ids
        ),
        "retained_export_names": list(value.retained_export_names),
        "support_export_names": list(value.support_export_names),
    }


def bootstrap_result_projection(
    value: MigrationBootstrapResult,
) -> Mapping[str, object]:
    if type(value) is not MigrationBootstrapResult:
        raise TypeError("bootstrap result projection requires exact value")
    return {
        "fence": _stack_identity_projection(value.fence),
        "support": _stack_identity_projection(value.support),
        "action_identity_sha256": value.action_identity_sha256,
        "state_revision": value.state_revision,
        "reconciled_creates": [
            kind.value for kind in value.reconciled_creates
        ],
    }


def migration_bundle_projection(
    value: MigrationBundle,
) -> Mapping[str, object]:
    if type(value) is not MigrationBundle:
        raise TypeError("migration bundle projection requires exact value")
    return {
        "manifest": dict(value.manifest),
        "artifacts": [
            {
                "stage": artifact.stage,
                "stack_id": artifact.stack_id,
                "template_url": artifact.template_url,
                "version_id": artifact.version_id,
                "body_base64": base64.b64encode(
                    artifact.body
                ).decode("ascii"),
                "sha256": artifact.sha256,
                "template_body_sha256": (
                    artifact.template_body_sha256
                ),
                "policy_sha256": artifact.policy_sha256,
                "template": dict(artifact.template),
            }
            for artifact in value.artifacts
        ],
    }


def sealed_migration_projection(
    *,
    authority: MigrationExecutionAuthority,
    evidence: MigrationEvidence,
    bundle: MigrationBundle,
    bootstrap_result: MigrationBootstrapResult,
    durable_state: Mapping[str, object],
) -> Mapping[str, object]:
    document = {
        "schema_version": 1,
        "record_type": "glm52_task13_sealed_stack_migration_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "authority": execution_authority_projection(authority),
        "evidence": migration_evidence_projection(evidence),
        "bundle": migration_bundle_projection(bundle),
        "bootstrap_result": bootstrap_result_projection(
            bootstrap_result
        ),
        "durable_state": dict(durable_state),
    }
    document["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(document)
    ).hexdigest()
    validate_sealed_migration_projection(document)
    return document


@dataclass(frozen=True)
class SealedMigrationRuntime:
    """Exact typed inputs and durable coordinator for one migration."""

    coordinator: MigrationCoordinator
    authority: MigrationExecutionAuthority
    evidence: MigrationEvidence
    bundle: MigrationBundle
    bootstrap_result: MigrationBootstrapResult


@dataclass(frozen=True)
class ValidatedSealedMigrationProjection:
    """Pure typed form of the self-hashed sealed migration document."""

    authority: MigrationExecutionAuthority
    evidence: MigrationEvidence
    bundle: MigrationBundle
    bootstrap_result: MigrationBootstrapResult
    durable_state: Mapping[str, object]


@dataclass(frozen=True)
class ValidatedMigrationSeed:
    """Pure typed seed for the first in-route migration mutation."""

    authority: MigrationExecutionAuthority
    initial_state: Mapping[str, object]
    publication_plan: tuple[Mapping[str, str], ...]
    canonical_identity_sha256: str


@dataclass(frozen=True)
class SeededMigrationRuntime:
    """Coordinator and backend initialized from the mutation-free seed."""

    coordinator: MigrationCoordinator
    backend: "DynamoDbMigrationStateBackend"
    seed: ValidatedMigrationSeed


def validate_stack_migration_seed_projection(
    value: object,
) -> ValidatedMigrationSeed:
    """Validate a mutation-free seed for the journaled coordinator route."""

    document = _mapping(value, "stack migration seed")
    if set(document) != {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "authority",
        "initial_state",
        "publication_plan",
        "canonical_identity_sha256",
    }:
        raise Task13MigrationAdapterError(
            "stack migration seed fields are not exact"
        )
    claimed = document.pop("canonical_identity_sha256")
    if (
        document.get("schema_version") != 1
        or document.get("record_type")
        != "glm52_task13_stack_migration_seed_v1"
        or document.get("account_id") != ACCOUNT_ID
        or document.get("region") != REGION
        or document.get("run_id") != RUN_ID
        or _sha256(claimed, "stack migration seed identity")
        != hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    ):
        raise Task13MigrationAdapterError(
            "stack migration seed identity drifted"
        )
    authority = parse_execution_authority(document["authority"])
    state = migration_execution_state_from_projection(
        document["initial_state"]
    )
    expected_state = initial_migration_execution_state(authority)
    if (
        state != expected_state
        or state.revision != 0
        or any(
            mutation.status.value != "NOT_SUBMITTED"
            for mutation in state.mutations
        )
    ):
        raise Task13MigrationAdapterError(
            "stack migration seed state is not pristine revision zero"
        )
    plan = document["publication_plan"]
    if type(plan) is not list or len(plan) != len(
        MIGRATION_TEMPLATE_KEYS
    ):
        raise Task13MigrationAdapterError(
            "stack migration publication plan is not exact"
        )
    expected_plan = [
        {
            "stage": stage,
            "bucket": (
                "keep-glm52-models-246813579024-us-west-2"
            ),
            "key": key,
        }
        for stage, key in MIGRATION_TEMPLATE_KEYS.items()
    ]
    if plan != expected_plan:
        raise Task13MigrationAdapterError(
            "stack migration publication plan drifted"
        )
    return ValidatedMigrationSeed(
        authority=authority,
        initial_state=migration_execution_state_projection(state),
        publication_plan=tuple(
            dict(row) for row in expected_plan
        ),
        canonical_identity_sha256=str(claimed),
    )


class DynamoDbMigrationStateBackend:
    """Conditional-revision backend for the retained campaign ledger."""

    def __init__(
        self,
        *,
        client: object,
        table_name: str = LEDGER_TABLE_NAME,
    ) -> None:
        if table_name != LEDGER_TABLE_NAME:
            raise Task13MigrationAdapterError(
                "migration state table is not the retained ledger"
            )
        self._client = client
        self._table_name = table_name

    @staticmethod
    def _key(action_identity_sha256: str) -> dict[str, object]:
        _sha256(
            action_identity_sha256,
            "migration action identity",
        )
        return {
            "PK": {"S": f"RUN#{RUN_ID}"},
            "SK": {
                "S": _STATE_PREFIX + action_identity_sha256
            },
        }
    def _read_item(
        self,
        action_identity_sha256: str,
    ) -> dict[str, object] | None:
        method = getattr(self._client, "get_item", None)
        if not callable(method):
            raise Task13MigrationAdapterError(
                "migration state client lacks get_item"
            )
        response = _success(
            method(
                TableName=self._table_name,
                Key=self._key(action_identity_sha256),
                ConsistentRead=True,
                ReturnConsumedCapacity="NONE",
            ),
            "GetMigrationState",
        )
        raw = response.get("Item")
        if raw is None:
            return None
        decoded = decode_item(raw)
        if set(decoded) != {
            "PK",
            "SK",
            "ActionIdentitySha256",
            "Revision",
            "State",
            "StateBodySha256",
        }:
            raise Task13MigrationAdapterError(
                "migration state item fields are not exact"
            )
        expected_pk = f"RUN#{RUN_ID}"
        expected_sk = _STATE_PREFIX + action_identity_sha256
        if (
            decoded["PK"] != expected_pk
            or decoded["SK"] != expected_sk
            or decoded["ActionIdentitySha256"]
            != action_identity_sha256
            or type(decoded["Revision"]) is not int
            or decoded["Revision"] < 0
            or type(decoded["State"]) is not dict
            or type(decoded["StateBodySha256"]) is not str
            or decoded["StateBodySha256"]
            != hashlib.sha256(
                canonical_json_bytes(decoded["State"])
            ).hexdigest()
            or decoded["State"].get("revision")
            != decoded["Revision"]
        ):
            raise Task13MigrationAdapterError(
                "migration state item identity drifted"
            )
        return decoded

    def read_state(
        self,
        *,
        action_identity_sha256: str,
    ) -> dict[str, object]:
        item = self._read_item(action_identity_sha256)
        if item is None:
            raise Task13MigrationAdapterError(
                "durable migration state is absent"
            )
        return {
            "State": item["State"],
            "StateBodySha256": item["StateBodySha256"],
        }

    def initialize_state(
        self,
        seed: ValidatedMigrationSeed,
    ) -> Mapping[str, object]:
        """Conditionally create revision zero and reconcile ambiguity."""

        if type(seed) is not ValidatedMigrationSeed:
            raise TypeError(
                "migration state initialization requires exact seed"
            )
        action = seed.authority.action_identity_sha256
        item = {
            "PK": f"RUN#{RUN_ID}",
            "SK": _STATE_PREFIX + action,
            "ActionIdentitySha256": action,
            "Revision": 0,
            "State": dict(seed.initial_state),
            "StateBodySha256": hashlib.sha256(
                canonical_json_bytes(seed.initial_state)
            ).hexdigest(),
        }
        observed = self._read_item(action)
        if observed is None:
            method = getattr(self._client, "put_item", None)
            if not callable(method):
                raise Task13MigrationAdapterError(
                    "migration state client lacks put_item"
                )
            try:
                response = method(
                    TableName=self._table_name,
                    Item=encode_item(item),
                    ConditionExpression=(
                        "attribute_not_exists(PK) "
                        "AND attribute_not_exists(SK)"
                    ),
                    ReturnConsumedCapacity="NONE",
                    ReturnItemCollectionMetrics="NONE",
                    ReturnValues="NONE",
                )
            except Exception as exc:
                _error_metadata(exc, "InitializeMigrationState")
            else:
                _success(response, "InitializeMigrationState")
            observed = self._read_item(action)
        if observed != item:
            raise Task13MigrationAdapterError(
                "initial migration state readback is not exact"
            )
        return {
            "State": observed["State"],
            "StateBodySha256": observed["StateBodySha256"],
        }

    def conditional_write_state(
        self,
        *,
        action_identity_sha256: str,
        expected_revision: int,
        State: Mapping[str, object],
        StateBodySha256: str,
    ) -> dict[str, object]:
        if (
            type(expected_revision) is not int
            or expected_revision < 0
            or type(State) is not dict
            or State.get("revision") != expected_revision + 1
            or State.get("action_identity_sha256")
            != action_identity_sha256
            or _sha256(
                StateBodySha256,
                "migration state body identity",
            )
            != hashlib.sha256(canonical_json_bytes(State)).hexdigest()
        ):
            raise Task13MigrationAdapterError(
                "migration state transition is not exact"
            )
        item = {
            "PK": f"RUN#{RUN_ID}",
            "SK": _STATE_PREFIX + action_identity_sha256,
            "ActionIdentitySha256": action_identity_sha256,
            "Revision": expected_revision + 1,
            "State": dict(State),
            "StateBodySha256": StateBodySha256,
        }
        method = getattr(self._client, "put_item", None)
        if not callable(method):
            raise Task13MigrationAdapterError(
                "migration state client lacks put_item"
            )
        try:
            response = method(
                TableName=self._table_name,
                Item=encode_item(item),
                ConditionExpression=(
                    "ActionIdentitySha256 = :action "
                    "AND Revision = :expected"
                ),
                ExpressionAttributeValues=encode_item(
                    {
                        ":action": action_identity_sha256,
                        ":expected": expected_revision,
                    }
                ),
                ReturnConsumedCapacity="NONE",
                ReturnItemCollectionMetrics="NONE",
                ReturnValues="NONE",
            )
        except Exception as exc:
            _error_metadata(exc, "PutMigrationState")
        else:
            _success(response, "PutMigrationState")
        observed = self._read_item(action_identity_sha256)
        if observed != item:
            raise Task13MigrationAdapterError(
                "conditional migration state write was not exact"
            )
        return {
            "Committed": True,
            "PriorRevision": expected_revision,
            "State": dict(State),
            "StateBodySha256": StateBodySha256,
        }


def build_seeded_migration_runtime(
    *,
    cloudformation: object,
    dynamodb: object,
    value: object,
) -> SeededMigrationRuntime:
    """Initialize/adopt revision zero and return the reviewed coordinator."""

    seed = validate_stack_migration_seed_projection(value)
    backend = DynamoDbMigrationStateBackend(client=dynamodb)
    observed = backend._read_item(  # noqa: SLF001 - same-module adapter
        seed.authority.action_identity_sha256
    )
    if observed is None:
        backend.initialize_state(seed)
    else:
        try:
            state = migration_execution_state_from_projection(
                observed["State"]
            )
            initial = migration_execution_state_from_projection(
                seed.initial_state
            )
        except (TypeError, ValueError) as exc:
            raise Task13MigrationAdapterError(
                "existing migration state is invalid"
            ) from exc
        state_static = migration_execution_state_projection(state)
        initial_static = migration_execution_state_projection(initial)
        state_static.pop("revision")
        initial_static.pop("revision")
        state_static.pop("mutations")
        initial_static.pop("mutations")
        if (
            state_static != initial_static
            or state.revision < 0
            or observed["StateBodySha256"]
            != hashlib.sha256(
                canonical_json_bytes(observed["State"])
            ).hexdigest()
        ):
            raise Task13MigrationAdapterError(
                "existing migration state differs from the seed"
            )
    return SeededMigrationRuntime(
        coordinator=MigrationCoordinator(
            client=cloudformation,
            state_store=CanonicalMigrationStateStore(backend=backend),
        ),
        backend=backend,
        seed=seed,
    )
def build_existing_seeded_migration_runtime(
    *,
    cloudformation: object,
    dynamodb: object,
    value: object,
) -> SeededMigrationRuntime:
    """Open one existing exact migration journal without creating it."""

    seed = validate_stack_migration_seed_projection(value)
    backend = DynamoDbMigrationStateBackend(client=dynamodb)
    observed = backend._read_item(  # noqa: SLF001 - same-module adapter
        seed.authority.action_identity_sha256
    )
    if observed is None:
        raise Task13MigrationAdapterError(
            "existing migration state is absent; read-only adoption refused"
        )
    try:
        state = migration_execution_state_from_projection(observed["State"])
        initial = migration_execution_state_from_projection(seed.initial_state)
    except (TypeError, ValueError) as exc:
        raise Task13MigrationAdapterError(
            "existing migration state is invalid"
        ) from exc
    state_static = migration_execution_state_projection(state)
    initial_static = migration_execution_state_projection(initial)
    state_static.pop("revision")
    initial_static.pop("revision")
    state_static.pop("mutations")
    initial_static.pop("mutations")
    if (
        state_static != initial_static
        or state.revision < 0
        or observed["StateBodySha256"]
        != hashlib.sha256(
            canonical_json_bytes(observed["State"])
        ).hexdigest()
    ):
        raise Task13MigrationAdapterError(
            "existing migration state differs from the seed"
        )
    return SeededMigrationRuntime(
        coordinator=MigrationCoordinator(
            client=cloudformation,
            state_store=CanonicalMigrationStateStore(backend=backend),
        ),
        backend=backend,
        seed=seed,
    )




def bootstrap_seeded_migration(
    runtime: SeededMigrationRuntime,
) -> MigrationBootstrapResult:
    if type(runtime) is not SeededMigrationRuntime:
        raise TypeError(
            "migration bootstrap requires exact seeded runtime"
        )
    return runtime.coordinator.bootstrap(
        authority=runtime.seed.authority
    )


def validate_sealed_migration_projection(
    value: object,
) -> ValidatedSealedMigrationProjection:
    """Validate all sealed migration components without AWS access."""

    item = _mapping(value, "sealed migration input")
    if set(item) != {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "authority",
        "evidence",
        "bundle",
        "bootstrap_result",
        "durable_state",
        "canonical_identity_sha256",
    }:
        raise Task13MigrationAdapterError(
            "sealed migration input fields are not exact"
        )
    identity = item.pop("canonical_identity_sha256")
    if (
        item.get("schema_version") != 1
        or item.get("record_type")
        != "glm52_task13_sealed_stack_migration_v1"
        or item.get("account_id") != ACCOUNT_ID
        or item.get("region") != REGION
        or item.get("run_id") != RUN_ID
        or _sha256(identity, "sealed migration identity")
        != hashlib.sha256(canonical_json_bytes(item)).hexdigest()
    ):
        raise Task13MigrationAdapterError(
            "sealed migration input identity drifted"
        )
    authority = parse_execution_authority(item["authority"])
    evidence = parse_migration_evidence(item["evidence"])
    bundle = parse_migration_bundle(item["bundle"])
    bootstrap_result = parse_bootstrap_result(item["bootstrap_result"])
    state = migration_execution_state_from_projection(
        item["durable_state"]
    )
    if (
        authority.action_identity_sha256
        != bootstrap_result.action_identity_sha256
        or authority.action_identity_sha256
        != state.action_identity_sha256
        or evidence.fence != bootstrap_result.fence
        or evidence.support != bootstrap_result.support
        or state.revision < bootstrap_result.state_revision
    ):
        raise Task13MigrationAdapterError(
            "sealed migration components are incoherent"
        )
    return ValidatedSealedMigrationProjection(
        authority=authority,
        evidence=evidence,
        bundle=bundle,
        bootstrap_result=bootstrap_result,
        durable_state=dict(item["durable_state"]),
    )


def build_sealed_migration_runtime(
    *,
    cloudformation: object,
    dynamodb: object,
    value: object,
) -> SealedMigrationRuntime:
    validated = validate_sealed_migration_projection(value)
    authority = validated.authority
    backend = DynamoDbMigrationStateBackend(client=dynamodb)
    stored = backend.read_state(
        action_identity_sha256=authority.action_identity_sha256
    )
    try:
        sealed_state = migration_execution_state_from_projection(
            validated.durable_state
        )
        observed_state = migration_execution_state_from_projection(
            stored["State"]
        )
    except (TypeError, ValueError) as exc:
        raise Task13MigrationAdapterError(
            "sealed or durable migration state is invalid"
        ) from exc
    sealed_static = migration_execution_state_projection(sealed_state)
    observed_static = migration_execution_state_projection(observed_state)
    sealed_static.pop("revision")
    sealed_mutations = sealed_static.pop("mutations")
    observed_static.pop("revision")
    observed_mutations = observed_static.pop("mutations")
    status_rank = {
        "NOT_SUBMITTED": 0,
        "SUBMITTED": 1,
        "COMPLETE": 2,
    }
    monotonic = (
        observed_state.revision >= sealed_state.revision
        and observed_static == sealed_static
        and type(sealed_mutations) is list
        and type(observed_mutations) is list
        and len(observed_mutations) == len(sealed_mutations)
    )
    if monotonic:
        for sealed_row, observed_row in zip(
            sealed_mutations,
            observed_mutations,
            strict=True,
        ):
            if (
                type(sealed_row) is not dict
                or type(observed_row) is not dict
                or sealed_row.get("key") != observed_row.get("key")
                or status_rank.get(str(observed_row.get("status")), -1)
                < status_rank.get(str(sealed_row.get("status")), -1)
                or (
                    sealed_row.get("request_sha256") is not None
                    and observed_row.get("request_sha256")
                    != sealed_row.get("request_sha256")
                )
                or (
                    sealed_row.get("effect_identity") is not None
                    and observed_row.get("effect_identity")
                    != sealed_row.get("effect_identity")
                )
            ):
                monotonic = False
                break
    if not monotonic:
        raise Task13MigrationAdapterError(
            "durable migration state is not a monotonic continuation "
            "of the sealed state"
        )
    return SealedMigrationRuntime(
        coordinator=MigrationCoordinator(
            client=cloudformation,
            state_store=CanonicalMigrationStateStore(backend=backend),
        ),
        authority=authority,
        evidence=validated.evidence,
        bundle=validated.bundle,
        bootstrap_result=validated.bootstrap_result,
    )


def execute_sealed_migration(
    runtime: SealedMigrationRuntime,
) -> MigrationExecutionResult:
    if type(runtime) is not SealedMigrationRuntime:
        raise TypeError(
            "sealed migration execution requires exact runtime"
        )
    raise Task13MigrationAdapterError(
        "stack-migration v1 is audit-only and execution-ineligible"
    )



STACK_MIGRATION_OPERATION_7_RECORD_TYPE_V2 = (
    "STACK_MIGRATION_OPERATION_7_EVIDENCE_V2"
)


@dataclass(frozen=True)
class StackMigrationOperation7EvidenceV2:
    """Immutable evidence for the sole disabled-support replacement."""

    record_type: str
    checkpoint_identity_sha256: str
    prepare_execution_identity_sha256: str
    disabled_support_identity_sha256: str
    retained_stack_id: str
    fence_stack_id: str
    support_stack_id: str
    support_prestate_template_sha256: str
    support_poststate_template_sha256: str
    disabled_support_profile_sha256: str
    no_launch_evidence_sha256: str
    api_caller: MigrationRoleEvidenceV2
    migration_service_role: MigrationRoleEvidenceV2
    fence_service_role: MigrationRoleEvidenceV2
    fence_associated_role_arn: str
    fence_associated_role_id: str
    operation_7_status: str
    mutation_scope: str
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        roles = (
            self.api_caller,
            self.migration_service_role,
            self.fence_service_role,
        )
        if any(type(role) is not MigrationRoleEvidenceV2 for role in roles):
            raise TypeError("operation-7 role evidence must be exact v2")
        if (
            self.record_type != STACK_MIGRATION_OPERATION_7_RECORD_TYPE_V2
            or self.operation_7_status != "COMPLETE"
            or self.mutation_scope != "SUPPORT_REPLACEMENT_ONLY"
            or self.fence_associated_role_arn != self.fence_service_role.role_arn
            or self.fence_associated_role_id != self.fence_service_role.role_id
            or self.support_prestate_template_sha256
            == self.support_poststate_template_sha256
            or len({role.role_arn for role in roles}) != 3
            or len({role.role_id for role in roles}) != 3
        ):
            raise Task13MigrationAdapterError(
                "operation-7 evidence is not one disabled-support replacement"
            )
        for kind, stack_id in (
            (StackKind.RETAINED, self.retained_stack_id),
            (StackKind.FENCE, self.fence_stack_id),
            (StackKind.SUPPORT, self.support_stack_id),
        ):
            StackIdentity(
                kind=kind,
                name={
                    StackKind.RETAINED: "keep-glm52-gpu",
                    StackKind.FENCE: "keep-glm52-h1g-fence",
                    StackKind.SUPPORT: "keep-glm52-h1g-support",
                }[kind],
                stack_id=stack_id,
                account_id=ACCOUNT_ID,
                region=REGION,
                termination_protection=True,
                tags=(
                    ("Authority", "H1g"),
                    ("Campaign", "GLM-5.2"),
                    ("Environment", "production"),
                    ("ManagedBy", "CloudFormation"),
                    ("Project", "KEEP"),
                    ("RunId", RUN_ID),
                ),
            )
        for label, value in (
            ("checkpoint_identity_sha256", self.checkpoint_identity_sha256),
            (
                "prepare_execution_identity_sha256",
                self.prepare_execution_identity_sha256,
            ),
            (
                "disabled_support_identity_sha256",
                self.disabled_support_identity_sha256,
            ),
            (
                "support_prestate_template_sha256",
                self.support_prestate_template_sha256,
            ),
            (
                "support_poststate_template_sha256",
                self.support_poststate_template_sha256,
            ),
            (
                "disabled_support_profile_sha256",
                self.disabled_support_profile_sha256,
            ),
            ("no_launch_evidence_sha256", self.no_launch_evidence_sha256),
        ):
            _sha256(value, label)
        if self.canonical_identity_sha256 != _operation_7_identity_sha256(self):
            raise Task13MigrationAdapterError(
                "operation-7 evidence canonical identity drifted"
            )

    def to_dict(self) -> Mapping[str, object]:
        return _operation_7_projection(self, include_identity=True)


def _operation_7_projection(
    evidence: StackMigrationOperation7EvidenceV2, *, include_identity: bool
) -> dict[str, object]:
    value: dict[str, object] = {
        "record_type": evidence.record_type,
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "checkpoint_identity_sha256": evidence.checkpoint_identity_sha256,
        "prepare_execution_identity_sha256": (
            evidence.prepare_execution_identity_sha256
        ),
        "disabled_support_identity_sha256": (
            evidence.disabled_support_identity_sha256
        ),
        "retained_stack_id": evidence.retained_stack_id,
        "fence_stack_id": evidence.fence_stack_id,
        "support_stack_id": evidence.support_stack_id,
        "support_prestate_template_sha256": (
            evidence.support_prestate_template_sha256
        ),
        "support_poststate_template_sha256": (
            evidence.support_poststate_template_sha256
        ),
        "disabled_support_profile_sha256": (
            evidence.disabled_support_profile_sha256
        ),
        "no_launch_evidence_sha256": evidence.no_launch_evidence_sha256,
        "api_caller": evidence.api_caller.to_dict(),
        "migration_service_role": evidence.migration_service_role.to_dict(),
        "fence_service_role": evidence.fence_service_role.to_dict(),
        "fence_associated_role_arn": evidence.fence_associated_role_arn,
        "fence_associated_role_id": evidence.fence_associated_role_id,
        "operation_7_status": evidence.operation_7_status,
        "mutation_scope": evidence.mutation_scope,
    }
    if include_identity:
        value["canonical_identity_sha256"] = evidence.canonical_identity_sha256
    return value


def _operation_7_identity_sha256(
    evidence: StackMigrationOperation7EvidenceV2,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            _operation_7_projection(evidence, include_identity=False)
        )
    ).hexdigest()


def _operation_7_identity_from_fields(fields: Mapping[str, object]) -> str:
    projection = {
        "record_type": fields["record_type"],
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "checkpoint_identity_sha256": fields["checkpoint_identity_sha256"],
        "prepare_execution_identity_sha256": (
            fields["prepare_execution_identity_sha256"]
        ),
        "disabled_support_identity_sha256": (
            fields["disabled_support_identity_sha256"]
        ),
        "retained_stack_id": fields["retained_stack_id"],
        "fence_stack_id": fields["fence_stack_id"],
        "support_stack_id": fields["support_stack_id"],
        "support_prestate_template_sha256": (
            fields["support_prestate_template_sha256"]
        ),
        "support_poststate_template_sha256": (
            fields["support_poststate_template_sha256"]
        ),
        "disabled_support_profile_sha256": (
            fields["disabled_support_profile_sha256"]
        ),
        "no_launch_evidence_sha256": fields["no_launch_evidence_sha256"],
        "api_caller": fields["api_caller"].to_dict(),
        "migration_service_role": fields["migration_service_role"].to_dict(),
        "fence_service_role": fields["fence_service_role"].to_dict(),
        "fence_associated_role_arn": fields["fence_associated_role_arn"],
        "fence_associated_role_id": fields["fence_associated_role_id"],
        "operation_7_status": fields["operation_7_status"],
        "mutation_scope": fields["mutation_scope"],
    }
    return hashlib.sha256(canonical_json_bytes(projection)).hexdigest()


def _load_disabled_support_template_artifact(
    *,
    runtime: SeededMigrationRuntime,
    disabled_support: "DisabledSupportDeploymentEvidence",
    stack_id: str,
) -> TemplateArtifact:
    coordinate = disabled_support.support_template_coordinate
    response = _success(
        runtime.coordinator._client.get_object(  # noqa: SLF001
            Bucket=coordinate["bucket"],
            Key=coordinate["key"],
            VersionId=coordinate["version_id"],
            ExpectedBucketOwner=ACCOUNT_ID,
        ),
        "GetObject(DisabledSupportTemplate)",
    )
    body = response.get("Body")
    raw = body.read() if callable(getattr(body, "read", None)) else body
    if type(raw) is not bytes:
        raise Task13MigrationAdapterError(
            "disabled support template body is not exact bytes"
        )
    try:
        template = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Task13MigrationAdapterError(
            "disabled support template is not canonical JSON"
        ) from exc
    canonical = canonical_json_bytes(template)
    if (
        type(template) is not dict
        or raw != canonical + b"\n"
        or hashlib.sha256(raw).hexdigest()
        != coordinate["file_sha256"]
        or hashlib.sha256(canonical).hexdigest()
        != coordinate["body_sha256"]
        or coordinate["body_sha256"]
        != disabled_support.support_template_sha256
    ):
        raise Task13MigrationAdapterError(
            "disabled support template coordinate drifted"
        )
    version_id = coordinate["version_id"]
    template_url = (
        f"https://{coordinate['bucket']}.s3.{REGION}.amazonaws.com/"
        f"{quote(str(coordinate['key']), safe='/')}?versionId="
        f"{quote(str(version_id), safe='')}"
    )
    return TemplateArtifact(
        stage="disabled-support",
        stack_id=stack_id,
        template_url=template_url,
        version_id=version_id,
        body=raw,
        sha256=coordinate["file_sha256"],
        template_body_sha256=coordinate["body_sha256"],
        policy_sha256=None,
        template=template,
    )


def execute_stack_migration_operation_7_v2(
    *,
    runtime: SeededMigrationRuntime,
    checkpoint: StackMigrationTransferCheckpointV2,
    prepare_result: "FenceExecutionResult",
    disabled_support: "DisabledSupportDeploymentEvidence",
    observed_api_caller: MigrationRoleEvidenceV2,
    observed_migration_service_role: MigrationRoleEvidenceV2,
    observed_fence_service_role: MigrationRoleEvidenceV2,
) -> StackMigrationOperation7EvidenceV2:
    """Perform only the journaled support-anchor replacement."""

    from .fence_executor import FenceExecutionResult
    from .task13_staged_deployment import DisabledSupportDeploymentEvidence

    if (
        type(runtime) is not SeededMigrationRuntime
        or type(checkpoint) is not StackMigrationTransferCheckpointV2
        or type(prepare_result) is not FenceExecutionResult
        or type(disabled_support) is not DisabledSupportDeploymentEvidence
        or observed_api_caller != checkpoint.api_caller
        or observed_migration_service_role
        != checkpoint.migration_service_role
        or observed_fence_service_role != checkpoint.fence_service_role
        or runtime.seed.authority.deployment_role_arn
        != observed_migration_service_role.role_arn
        or runtime.seed.authority.deployment_role_id
        != observed_migration_service_role.role_id
    ):
        raise Task13MigrationAdapterError(
            "operation 7 runtime and live role identities are not exact"
        )
    artifact = _load_disabled_support_template_artifact(
        runtime=runtime,
        disabled_support=disabled_support,
        stack_id=checkpoint.support_stack_id,
    )
    prestate_sha256, poststate_sha256 = (
        runtime.coordinator.execute_support_anchor_replacement_v2(
            checkpoint=checkpoint,
            authority=runtime.seed.authority,
            artifact=artifact,
            expected_direct_policy_sha256=(
                prepare_result.poststate_policy_sha256
            ),
        )
    )
    return build_stack_migration_operation_7_evidence_v2(
        checkpoint=checkpoint,
        prepare_result=prepare_result,
        disabled_support=disabled_support,
        observed_api_caller=observed_api_caller,
        observed_migration_service_role=observed_migration_service_role,
        observed_fence_service_role=observed_fence_service_role,
        support_stack_id=checkpoint.support_stack_id,
        support_prestate_template_sha256=prestate_sha256,
        support_poststate_template_sha256=poststate_sha256,
    )


def read_stack_migration_operation_7_effect_v2(
    *,
    runtime: SeededMigrationRuntime,
    checkpoint: StackMigrationTransferCheckpointV2,
    prepare_result: "FenceExecutionResult",
    disabled_support: "DisabledSupportDeploymentEvidence",
    observed_api_caller: MigrationRoleEvidenceV2,
    observed_migration_service_role: MigrationRoleEvidenceV2,
    observed_fence_service_role: MigrationRoleEvidenceV2,
) -> StackMigrationOperation7EvidenceV2:
    """Read the exact journaled operation-7 effect without mutation."""

    from .fence_executor import FenceExecutionResult
    from .task13_staged_deployment import DisabledSupportDeploymentEvidence

    if (
        type(runtime) is not SeededMigrationRuntime
        or type(checkpoint) is not StackMigrationTransferCheckpointV2
        or type(prepare_result) is not FenceExecutionResult
        or type(disabled_support) is not DisabledSupportDeploymentEvidence
        or observed_api_caller != checkpoint.api_caller
        or observed_migration_service_role
        != checkpoint.migration_service_role
        or observed_fence_service_role != checkpoint.fence_service_role
        or runtime.seed.authority.deployment_role_arn
        != observed_migration_service_role.role_arn
        or runtime.seed.authority.deployment_role_id
        != observed_migration_service_role.role_id
    ):
        raise Task13MigrationAdapterError(
            "operation 7 read runtime and live role identities are not exact"
        )

    artifact = _load_disabled_support_template_artifact(
        runtime=runtime,
        disabled_support=disabled_support,
        stack_id=checkpoint.support_stack_id,
    )
    prestate_sha256, poststate_sha256 = (
        runtime.coordinator.read_support_anchor_replacement_effect_v2(
            checkpoint=checkpoint,
            authority=runtime.seed.authority,
            artifact=artifact,
            expected_direct_policy_sha256=(
                prepare_result.poststate_policy_sha256
            ),
        )
    )
    return build_stack_migration_operation_7_evidence_v2(
        checkpoint=checkpoint,
        prepare_result=prepare_result,
        disabled_support=disabled_support,
        observed_api_caller=observed_api_caller,
        observed_migration_service_role=observed_migration_service_role,
        observed_fence_service_role=observed_fence_service_role,
        support_stack_id=checkpoint.support_stack_id,
        support_prestate_template_sha256=prestate_sha256,
        support_poststate_template_sha256=poststate_sha256,
    )


def build_stack_migration_operation_7_evidence_v2(
    *,
    checkpoint: StackMigrationTransferCheckpointV2,
    prepare_result: "FenceExecutionResult",
    disabled_support: "DisabledSupportDeploymentEvidence",
    observed_api_caller: MigrationRoleEvidenceV2,
    observed_migration_service_role: MigrationRoleEvidenceV2,
    observed_fence_service_role: MigrationRoleEvidenceV2,
    support_stack_id: str,
    support_prestate_template_sha256: str,
    support_poststate_template_sha256: str,
) -> StackMigrationOperation7EvidenceV2:
    """Gate operation 7 on exact PREPARE and disabled-support evidence."""

    from .fence_artifacts import ExecutorAuthorityClass, FenceSlot
    from .fence_executor import FenceExecutionResult
    from .task13_staged_deployment import DisabledSupportDeploymentEvidence

    if (
        type(checkpoint) is not StackMigrationTransferCheckpointV2
        or type(prepare_result) is not FenceExecutionResult
        or type(disabled_support) is not DisabledSupportDeploymentEvidence
    ):
        raise TypeError("operation 7 requires exact typed v2 prerequisites")
    if (
        checkpoint.operation_7_status != "NOT_SUBMITTED"
        or prepare_result.slot is not FenceSlot.PREPARE_GENESIS_LIVE_STATE
        or prepare_result.execute_authority_class
        is not ExecutorAuthorityClass.RETAINED_PRE_SUPPORT
        or prepare_result.execute_api_caller_role_arn
        != observed_api_caller.role_arn
        or prepare_result.execute_api_caller_role_id
        != observed_api_caller.role_id
        or prepare_result.stack_id != checkpoint.fence_stack_id
        or prepare_result.expected_poststate_stack_role_arn
        != checkpoint.fence_service_role.role_arn
        or prepare_result.observed_poststate_stack_role_arn
        != checkpoint.fence_service_role.role_arn
        or prepare_result.observed_poststate_stack_role_id
        != checkpoint.fence_service_role.role_id
        or prepare_result.original_template_body_sha256
        != prepare_result.processed_template_body_sha256
        or prepare_result.first_stable_snapshot_identity_sha256
        != prepare_result.second_stable_snapshot_identity_sha256
        or prepare_result.stabilization_first_evidence_sha256
        == prepare_result.stabilization_second_evidence_sha256
        or prepare_result.poststate_policy_sha256
        == checkpoint.bridge_seed_policy_sha256
    ):
        raise Task13MigrationAdapterError(
            "operation 7 requires exact stabilized PREPARE role/policy readback"
        )
    if (
        disabled_support.prepare_entry_identity_sha256
        != prepare_result.entry_identity_sha256
        or support_stack_id != checkpoint.support_stack_id
        or support_prestate_template_sha256
        != checkpoint.support_prestate_template_sha256
        or support_poststate_template_sha256
        != disabled_support.support_template_sha256
        or disabled_support.support_template_coordinate.get("body_sha256")
        != disabled_support.support_template_sha256
    ):
        raise Task13MigrationAdapterError(
            "operation 7 requires the exact disabled-support artifact"
        )
    if (
        observed_api_caller != checkpoint.api_caller
        or observed_migration_service_role != checkpoint.migration_service_role
        or observed_fence_service_role != checkpoint.fence_service_role
    ):
        raise Task13MigrationAdapterError(
            "operation 7 role ARN/RoleId/trust/permission evidence drifted"
        )
    fields = {
        "record_type": STACK_MIGRATION_OPERATION_7_RECORD_TYPE_V2,
        "checkpoint_identity_sha256": checkpoint.canonical_identity_sha256,
        "prepare_execution_identity_sha256": (
            prepare_result.canonical_identity_sha256
        ),
        "disabled_support_identity_sha256": (
            disabled_support.canonical_identity_sha256
        ),
        "retained_stack_id": checkpoint.retained_stack_id,
        "fence_stack_id": checkpoint.fence_stack_id,
        "support_stack_id": checkpoint.support_stack_id,
        "support_prestate_template_sha256": support_prestate_template_sha256,
        "support_poststate_template_sha256": support_poststate_template_sha256,
        "disabled_support_profile_sha256": (
            disabled_support.disabled_support_profile_sha256
        ),
        "no_launch_evidence_sha256": (
            disabled_support.no_launch_evidence_sha256
        ),
        "api_caller": observed_api_caller,
        "migration_service_role": observed_migration_service_role,
        "fence_service_role": observed_fence_service_role,
        "fence_associated_role_arn": (
            prepare_result.observed_poststate_stack_role_arn
        ),
        "fence_associated_role_id": (
            prepare_result.observed_poststate_stack_role_id
        ),
        "operation_7_status": "COMPLETE",
        "mutation_scope": "SUPPORT_REPLACEMENT_ONLY",
    }
    return StackMigrationOperation7EvidenceV2(
        **fields,
        canonical_identity_sha256=_operation_7_identity_from_fields(fields),
    )


def parse_stack_migration_operation_7_evidence_v2(
    value: object,
) -> StackMigrationOperation7EvidenceV2:
    if type(value) is not dict:
        raise TypeError("operation-7 evidence must be one object")
    expected = {
        "record_type",
        "account_id",
        "region",
        "run_id",
        "checkpoint_identity_sha256",
        "prepare_execution_identity_sha256",
        "disabled_support_identity_sha256",
        "retained_stack_id",
        "fence_stack_id",
        "support_stack_id",
        "support_prestate_template_sha256",
        "support_poststate_template_sha256",
        "disabled_support_profile_sha256",
        "no_launch_evidence_sha256",
        "api_caller",
        "migration_service_role",
        "fence_service_role",
        "fence_associated_role_arn",
        "fence_associated_role_id",
        "operation_7_status",
        "mutation_scope",
        "canonical_identity_sha256",
    }
    if set(value) != expected or (
        value["account_id"],
        value["region"],
        value["run_id"],
    ) != (ACCOUNT_ID, REGION, RUN_ID):
        raise Task13MigrationAdapterError(
            "operation-7 evidence fields/scope are not exact"
        )

    def role(label: str) -> MigrationRoleEvidenceV2:
        item = value[label]
        if type(item) is not dict or set(item) != {
            "role_arn",
            "role_id",
            "trust_policy_sha256",
            "permission_policy_sha256",
        }:
            raise Task13MigrationAdapterError(
                f"operation-7 {label} is not exact"
            )
        return MigrationRoleEvidenceV2(**item)

    return StackMigrationOperation7EvidenceV2(
        record_type=value["record_type"],
        checkpoint_identity_sha256=value["checkpoint_identity_sha256"],
        prepare_execution_identity_sha256=value[
            "prepare_execution_identity_sha256"
        ],
        disabled_support_identity_sha256=value[
            "disabled_support_identity_sha256"
        ],
        retained_stack_id=value["retained_stack_id"],
        fence_stack_id=value["fence_stack_id"],
        support_stack_id=value["support_stack_id"],
        support_prestate_template_sha256=value[
            "support_prestate_template_sha256"
        ],
        support_poststate_template_sha256=value[
            "support_poststate_template_sha256"
        ],
        disabled_support_profile_sha256=value[
            "disabled_support_profile_sha256"
        ],
        no_launch_evidence_sha256=value["no_launch_evidence_sha256"],
        api_caller=role("api_caller"),
        migration_service_role=role("migration_service_role"),
        fence_service_role=role("fence_service_role"),
        fence_associated_role_arn=value["fence_associated_role_arn"],
        fence_associated_role_id=value["fence_associated_role_id"],
        operation_7_status=value["operation_7_status"],
        mutation_scope=value["mutation_scope"],
        canonical_identity_sha256=value["canonical_identity_sha256"],
    )


__all__ = [
    "DynamoDbMigrationStateBackend",
    "MigrationRoleEvidenceV2",
    "StackMigrationAuthorityV2",
    "StackMigrationEvidenceV2",
    "StackMigrationTransferBundleV2",
    "FENCE_SERVICE_ROLE_ARN_V2",
    "MIGRATION_SERVICE_ROLE_ARN_V2",
    "STACK_MIGRATION_TRANSFER_CHECKPOINT_V2",
    "LEDGER_TABLE_NAME",
    "MIGRATION_TEMPLATE_KEYS",
    "STACK_MIGRATION_OPERATION_7_RECORD_TYPE_V2",
    "SeededMigrationRuntime",
    "SealedMigrationRuntime",
    "StackMigrationOperation7EvidenceV2",
    "StackMigrationTransferCheckpointV2",
    "Task13MigrationAdapterError",
    "ValidatedSealedMigrationProjection",
    "ValidatedMigrationSeed",
    "build_existing_seeded_migration_runtime",
    "build_sealed_migration_runtime",
    "build_stack_migration_operation_7_evidence_v2",
    "build_stack_migration_transfer_checkpoint_v2",
    "build_stack_migration_v2",
    "build_seeded_migration_runtime",
    "bootstrap_result_projection",
    "bootstrap_seeded_migration",
    "execution_authority_projection",
    "initial_stack_migration_execution_state_v2",
    "execute_stack_migration_operation_7_v2",
    "execute_sealed_migration",
    "migration_bundle_projection",
    "migration_evidence_projection",
    "parse_bootstrap_result",
    "parse_execution_authority",
    "parse_migration_bundle",
    "parse_migration_evidence",
    "parse_stack_migration_operation_7_evidence_v2",
    "parse_stack_migration_transfer_checkpoint_v2",
    "read_stack_migration_operation_7_effect_v2",
    "sealed_migration_projection",
    "validate_sealed_migration_projection",
    "validate_stack_migration_seed_projection",
]
