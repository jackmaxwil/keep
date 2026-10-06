"""Pure three-stack CloudFormation ownership and migration contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
from typing import Dict, Mapping, Optional, Tuple
from urllib.parse import parse_qs, urlparse


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
MIGRATION_SERVICE_ROLE_ARN_V2 = (
    f"arn:aws:iam::{ACCOUNT_ID}:role/keep-glm52-h1g-cloudformation-deployment"
)
FENCE_SERVICE_ROLE_ARN_V2 = (
    f"arn:aws:iam::{ACCOUNT_ID}:role/keep-glm52-h1g-fence-service"
)
FENCE_LOGICAL_ID = "H1gProductionFenceBucketPolicy"
CONTAINER_ANCHOR = "ContainerAnchor"

STACK_TAGS: Tuple[Tuple[str, str], ...] = (
    ("Authority", "H1g"),
    ("Campaign", "GLM-5.2"),
    ("Environment", "production"),
    ("ManagedBy", "CloudFormation"),
    ("Project", "KEEP"),
    ("RunId", RUN_ID),
)


class StackKind(Enum):
    RETAINED = "retained"
    FENCE = "fence"
    SUPPORT = "support"


_STACK_NAMES = {
    StackKind.RETAINED: "keep-glm52-gpu",
    StackKind.FENCE: "keep-glm52-h1g-fence",
    StackKind.SUPPORT: "keep-glm52-h1g-support",
}

STACK_OWNERSHIP: Mapping[StackKind, Tuple[str, ...]] = {
    StackKind.RETAINED: (
        "existing-retained-infrastructure",
        "ledger-and-campaign-kms",
        "retained-lifecycle-and-finalization",
        "stable-source-publisher-identities",
    ),
    StackKind.FENCE: ("model-bucket-policy",),
    StackKind.SUPPORT: (
        "support-private-network",
        "support-combined-host-and-volumes",
        "support-tls-and-secrets",
        "support-production-workflow",
        "support-runtime-functions",
    ),
}
_BUCKET = re.compile(
    r"(?=.{3,63}\Z)(?![0-9]+(?:\.[0-9]+){3}\Z)"
    r"(?!xn--)(?!sthree-)(?!amzn-s3-demo-)"
    r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\Z"
)


def stack_name(kind: StackKind) -> str:
    if type(kind) is not StackKind:
        raise TypeError("stack kind must be exact StackKind")
    return _STACK_NAMES[kind]


def validate_stack_ownership(
    ownership: Mapping[StackKind, Tuple[str, ...]],
) -> Mapping[StackKind, Tuple[str, ...]]:
    if type(ownership) is not dict or set(ownership) != set(StackKind):
        raise ValueError("ownership must contain exactly the three stacks")
    observed = []
    for kind in StackKind:
        families = ownership[kind]
        if type(families) is not tuple or not families:
            raise ValueError("each stack requires a nonempty exact ownership tuple")
        for family in families:
            if type(family) is not str or not family or family in observed:
                raise ValueError("resource families must be nonempty and nonoverlapping")
            observed.append(family)
    return ownership


def validate_bootstrap_template(
    template: Mapping[str, object],
) -> Mapping[str, object]:
    expected = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            CONTAINER_ANCHOR: {
                "Type": "AWS::CloudFormation::WaitConditionHandle"
            }
        },
    }
    if type(template) is not dict or template != expected:
        raise ValueError("bootstrap template is not the exact inert v1 container")
    return template


def _resources(template: Mapping[str, object]) -> Dict[str, object]:
    if type(template) is not dict:
        raise ValueError("template must be an exact JSON object")
    resources = template.get("Resources")
    if type(resources) is not dict or not resources:
        raise ValueError("template must contain a nonempty Resources object")
    return resources


def _exact_policy_owner(
    template: Mapping[str, object], current_policy_logical_id: str
) -> Mapping[str, object]:
    if type(current_policy_logical_id) is not str or not current_policy_logical_id:
        raise ValueError("current policy logical ID is required")
    resources = _resources(template)
    policy_ids = [
        logical_id
        for logical_id, resource in resources.items()
        if type(resource) is dict
        and resource.get("Type") == "AWS::S3::BucketPolicy"
    ]
    if policy_ids != [current_policy_logical_id]:
        raise ValueError("archived template must have one exact bucket-policy owner")
    policy = resources[current_policy_logical_id]
    if type(policy) is not dict:
        raise ValueError("bucket-policy resource is malformed")
    return policy


def build_retention_only_template(
    *,
    archived_template: Mapping[str, object],
    current_policy_logical_id: str,
) -> Dict[str, object]:
    _exact_policy_owner(archived_template, current_policy_logical_id)
    result = deepcopy(archived_template)
    policy = _resources(result)[current_policy_logical_id]
    if type(policy) is not dict:
        raise ValueError("bucket-policy resource is malformed")
    if "DeletionPolicy" in policy or "UpdateReplacePolicy" in policy:
        raise ValueError("retention attributes must not preexist or be guessed")
    policy["DeletionPolicy"] = "Retain"
    policy["UpdateReplacePolicy"] = "Retain"
    return result


def build_post_retain_template(
    *,
    retention_template: Mapping[str, object],
    current_policy_logical_id: str,
) -> Dict[str, object]:
    policy = _exact_policy_owner(retention_template, current_policy_logical_id)
    if (
        policy.get("DeletionPolicy") != "Retain"
        or policy.get("UpdateReplacePolicy") != "Retain"
    ):
        raise ValueError("policy cannot be removed before both retain attributes")
    result = deepcopy(retention_template)
    del _resources(result)[current_policy_logical_id]
    return result


def canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("value is not canonical JSON data") from exc
    return encoded


def _validate_bucket_name(bucket_name: str) -> str:
    if (
        type(bucket_name) is not str
        or _BUCKET.fullmatch(bucket_name) is None
        or ".." in bucket_name
        or ".-" in bucket_name
        or "-." in bucket_name
        or bucket_name.endswith(("-s3alias", "--ol-s3", ".mrap", "--x-s3", "--table-s3"))
        or "*" in bucket_name
    ):
        raise ValueError("bucket name is not one exact general-purpose bucket")
    return bucket_name


def _validate_literal_policy(policy: Mapping[str, object]) -> Mapping[str, object]:
    if type(policy) is not dict or not policy:
        raise ValueError("canonical live policy must be a nonempty JSON object")
    raw = canonical_json_bytes(policy)
    if b"{{resolve:" in raw or b"Fn::" in raw or b"!Ref" in raw or b"!Sub" in raw:
        raise ValueError("canonical live policy must be literal and resolved")
    return policy


def build_fence_import_template(
    *,
    bootstrap_template: Mapping[str, object],
    bucket_name: str,
    canonical_live_policy: Mapping[str, object],
) -> Dict[str, object]:
    validate_bootstrap_template(bootstrap_template)
    _validate_bucket_name(bucket_name)
    _validate_literal_policy(canonical_live_policy)
    result = deepcopy(bootstrap_template)
    resources = _resources(result)
    resources[FENCE_LOGICAL_ID] = {
        "Type": "AWS::S3::BucketPolicy",
        "DeletionPolicy": "Retain",
        "UpdateReplacePolicy": "Retain",
        "Properties": {
            "Bucket": bucket_name,
            "PolicyDocument": deepcopy(canonical_live_policy),
        },
    }
    return result



def _checkpoint_projection_from_fields(
    fields: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "record_type": fields["record_type"],
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "retained_stack_id": fields["retained_stack_id"],
        "fence_stack_id": fields["fence_stack_id"],
        "support_stack_id": fields["support_stack_id"],
        "bucket_name": fields["bucket_name"],
        "retained_owner_logical_id": fields["retained_owner_logical_id"],
        "retained_owner_physical_id": fields["retained_owner_physical_id"],
        "bridge_seed_owner_branch": fields["bridge_seed_owner_branch"],
        "bridge_seed_policy_sha256": fields["bridge_seed_policy_sha256"],
        "expected_live_preseed_policy_sha256": (
            fields["expected_live_preseed_policy_sha256"]
        ),
        "bridge_seed_owner_template_sha256": (
            fields["bridge_seed_owner_template_sha256"]
        ),
        "fence_import_template_sha256": fields["fence_import_template_sha256"],
        "fence_transfer_template_sha256": (
            fields["fence_transfer_template_sha256"]
        ),
        "active_original_template_sha256": (
            fields["active_original_template_sha256"]
        ),
        "active_processed_template_sha256": (
            fields["active_processed_template_sha256"]
        ),
        "direct_policy_sha256": list(fields["direct_policy_sha256"]),
        "stack_policy_sha256": fields["stack_policy_sha256"],
        "api_caller": fields["api_caller"].to_dict(),
        "migration_service_role": fields["migration_service_role"].to_dict(),
        "fence_service_role": fields["fence_service_role"].to_dict(),
        "associated_stack_role_arn": fields["associated_stack_role_arn"],
        "associated_stack_role_id": fields["associated_stack_role_id"],
        "rollback_stack_role_arn": fields["rollback_stack_role_arn"],
        "import_change_set_id": fields["import_change_set_id"],
        "completed_mutation_keys": list(fields["completed_mutation_keys"]),
        "operation_7_status": fields["operation_7_status"],
        "support_state": fields["support_state"],
        "support_prestate_template_sha256": (
            fields["support_prestate_template_sha256"]
        ),
        "deletion_policy": fields["deletion_policy"],
        "update_replace_policy": fields["update_replace_policy"],
    }


def parse_stack_migration_transfer_checkpoint_v2(
    value: object,
) -> StackMigrationTransferCheckpointV2:
    if type(value) is not dict:
        raise TypeError("stack migration transfer checkpoint must be an object")
    expected = {
        "record_type",
        "account_id",
        "region",
        "run_id",
        "retained_stack_id",
        "fence_stack_id",
        "support_stack_id",
        "bucket_name",
        "retained_owner_logical_id",
        "retained_owner_physical_id",
        "bridge_seed_owner_branch",
        "bridge_seed_policy_sha256",
        "expected_live_preseed_policy_sha256",
        "bridge_seed_owner_template_sha256",
        "fence_import_template_sha256",
        "fence_transfer_template_sha256",
        "active_original_template_sha256",
        "active_processed_template_sha256",
        "direct_policy_sha256",
        "stack_policy_sha256",
        "api_caller",
        "migration_service_role",
        "fence_service_role",
        "associated_stack_role_arn",
        "associated_stack_role_id",
        "rollback_stack_role_arn",
        "import_change_set_id",
        "completed_mutation_keys",
        "operation_7_status",
        "support_state",
        "support_prestate_template_sha256",
        "deletion_policy",
        "update_replace_policy",
        "canonical_identity_sha256",
    }
    if set(value) != expected:
        raise ValueError("stack migration transfer checkpoint fields are not exact")
    if (
        value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["run_id"] != RUN_ID
    ):
        raise ValueError("stack migration transfer checkpoint scope drifted")

    def role(label: str) -> MigrationRoleEvidenceV2:
        item = value[label]
        if type(item) is not dict or set(item) != {
            "role_arn",
            "role_id",
            "trust_policy_sha256",
            "permission_policy_sha256",
        }:
            raise ValueError(f"{label} evidence is not exact")
        return MigrationRoleEvidenceV2(**item)

    direct = value["direct_policy_sha256"]
    completed = value["completed_mutation_keys"]
    if type(direct) is not list or type(completed) is not list:
        raise ValueError("checkpoint arrays are not exact")
    return StackMigrationTransferCheckpointV2(
        record_type=value["record_type"],
        retained_stack_id=value["retained_stack_id"],
        fence_stack_id=value["fence_stack_id"],
        support_stack_id=value["support_stack_id"],
        bucket_name=value["bucket_name"],
        retained_owner_logical_id=value["retained_owner_logical_id"],
        retained_owner_physical_id=value["retained_owner_physical_id"],
        bridge_seed_owner_branch=value["bridge_seed_owner_branch"],
        bridge_seed_policy_sha256=value["bridge_seed_policy_sha256"],
        expected_live_preseed_policy_sha256=value[
            "expected_live_preseed_policy_sha256"
        ],
        bridge_seed_owner_template_sha256=(
            value["bridge_seed_owner_template_sha256"]
        ),
        fence_import_template_sha256=value["fence_import_template_sha256"],
        fence_transfer_template_sha256=value["fence_transfer_template_sha256"],
        active_original_template_sha256=value[
            "active_original_template_sha256"
        ],
        active_processed_template_sha256=value[
            "active_processed_template_sha256"
        ],
        direct_policy_sha256=tuple(direct),
        stack_policy_sha256=value["stack_policy_sha256"],
        api_caller=role("api_caller"),
        migration_service_role=role("migration_service_role"),
        fence_service_role=role("fence_service_role"),
        associated_stack_role_arn=value["associated_stack_role_arn"],
        associated_stack_role_id=value["associated_stack_role_id"],
        rollback_stack_role_arn=value["rollback_stack_role_arn"],
        import_change_set_id=value["import_change_set_id"],
        completed_mutation_keys=tuple(completed),
        operation_7_status=value["operation_7_status"],
        support_state=value["support_state"],
        support_prestate_template_sha256=value[
            "support_prestate_template_sha256"
        ],
        deletion_policy=value["deletion_policy"],
        update_replace_policy=value["update_replace_policy"],
        canonical_identity_sha256=value["canonical_identity_sha256"],
    )


def _checkpoint_fields_identity_sha256(
    fields: Mapping[str, object],
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(_checkpoint_projection_from_fields(fields))
    ).hexdigest()



_SUPPORT_FORBIDDEN_TYPES = {
    "AWS::DynamoDB::Table",
    "AWS::KMS::Key",
    "AWS::KMS::Alias",
    "AWS::S3::BucketPolicy",
}
_REHEARSAL_BUCKET_NAME = (
    f"keep-glm52-h1g-rehearsal-{ACCOUNT_ID}-{REGION}"
)
_RESOURCE_IDENTITY_PROPERTIES: Mapping[str, Tuple[str, ...]] = {
    "AWS::S3::Bucket": ("BucketName",),
    "AWS::Logs::LogGroup": ("LogGroupName",),
    "AWS::Lambda::Function": ("FunctionName",),
    "AWS::SecretsManager::Secret": ("Name",),
    "AWS::IAM::Role": ("RoleName",),
}
_LOGICAL_ID = re.compile(r"[A-Za-z][A-Za-z0-9]{0,254}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STACK_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_CHANGE_SET_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_ROLE_ID = re.compile(r"AROA[A-Z0-9]{16,128}\Z")
_VERSION_ID = re.compile(r"[A-Za-z0-9._+=/-]{1,1024}\Z")


@dataclass(frozen=True)
class StackIdentity:
    kind: StackKind
    name: str
    stack_id: str
    account_id: str
    region: str
    termination_protection: bool
    tags: Tuple[Tuple[str, str], ...]

    def __post_init__(self) -> None:
        if type(self.kind) is not StackKind or self.name != stack_name(self.kind):
            raise ValueError("stack identity has wrong kind or exact name")
        expected_prefix = (
            f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:"
            f"stack/{self.name}/"
        )
        if type(self.stack_id) is not str:
            raise ValueError("stack identity is not the exact protected deployment")
        suffix = self.stack_id.removeprefix(expected_prefix)
        if (
            not self.stack_id.startswith(expected_prefix)
            or _STACK_UUID.fullmatch(suffix) is None
            or self.account_id != ACCOUNT_ID
            or self.region != REGION
            or type(self.termination_protection) is not bool
            or not self.termination_protection
            or self.tags != STACK_TAGS
        ):
            raise ValueError("stack identity is not the exact protected deployment")


@dataclass(frozen=True)
class TemplateCoordinate:
    stage: str
    stack_id: str
    template_url: str
    version_id: str

    def __post_init__(self) -> None:
        if type(self.stage) is not str or not self.stage:
            raise ValueError("template stage is required")
        _validate_template_url(self.template_url, self.version_id)


@dataclass(frozen=True)
class MigrationEvidence:
    retained: StackIdentity
    fence: StackIdentity
    support: StackIdentity
    current_policy_logical_id: str
    current_policy_physical_id: str
    current_policy_stack_id: str
    bucket_name: str
    import_identifier: Tuple[Tuple[str, str], ...]
    live_policy: Mapping[str, object]
    direct_policy_readbacks: Tuple[Mapping[str, object], ...]
    retained_deployment_role_id: str
    retained_resource_physical_ids: Tuple[str, ...]
    retained_export_names: Tuple[str, ...]
    support_export_names: Tuple[str, ...]


@dataclass(frozen=True)
class TemplateArtifact:
    stage: str
    stack_id: str
    template_url: str
    version_id: str
    body: bytes
    sha256: str
    template_body_sha256: str
    policy_sha256: Optional[str]
    template: Mapping[str, object]


@dataclass(frozen=True)
class MigrationBundle:
    artifacts: Tuple[TemplateArtifact, ...]
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class BootstrapCoordinate:
    kind: StackKind
    template_url: str
    version_id: str

    def __post_init__(self) -> None:
        if self.kind not in {StackKind.FENCE, StackKind.SUPPORT}:
            raise ValueError("bootstrap coordinate is only for a new container stack")
        _validate_template_url(self.template_url, self.version_id)


@dataclass(frozen=True)
class MigrationExecutionAuthority:
    deployment_role_arn: str
    deployment_role_id: str
    fence_bootstrap: BootstrapCoordinate
    support_bootstrap: BootstrapCoordinate
    import_change_set_name: str
    action_identity_sha256: str

    def __post_init__(self) -> None:
        role = re.fullmatch(
            rf"arn:aws:iam::{ACCOUNT_ID}:role/"
            r"[A-Za-z0-9+=,.@_-]+(?:/[A-Za-z0-9+=,.@_-]+)*",
            self.deployment_role_arn,
        )
        if (
            role is None
            or type(self.deployment_role_id) is not str
            or _ROLE_ID.fullmatch(self.deployment_role_id) is None
            or type(self.fence_bootstrap) is not BootstrapCoordinate
            or self.fence_bootstrap.kind is not StackKind.FENCE
            or type(self.support_bootstrap) is not BootstrapCoordinate
            or self.support_bootstrap.kind is not StackKind.SUPPORT
            or type(self.import_change_set_name) is not str
            or re.fullmatch(
                r"[A-Za-z][-A-Za-z0-9]{0,127}", self.import_change_set_name
            )
            is None
            or type(self.action_identity_sha256) is not str
            or _SHA256.fullmatch(self.action_identity_sha256) is None
        ):
            raise ValueError("migration execution authority is not exact")


class MigrationMutationStatus(Enum):
    NOT_SUBMITTED = "NOT_SUBMITTED"
    SUBMITTED = "SUBMITTED"
    COMPLETE = "COMPLETE"


_MIGRATION_MUTATION_KEYS = (
    "CreateBootstrapFenceStack",
    "CreateBootstrapSupportStack",
    "EstablishRetainedBridgeSeed",
    "UpdateRetainedWithRetainAttributes",
    "UpdateRetainedRemovePolicy",
    "ImportPolicyIntoFenceStack.CreateChangeSet",
    "ImportPolicyIntoFenceStack.ExecuteChangeSet",
    "UpdateFenceRemoveAnchor",
    "UpdateSupportReplaceAnchor",
)


@dataclass(frozen=True)
class MigrationMutationRecord:
    key: str
    status: MigrationMutationStatus
    request_sha256: Optional[str]
    effect_identity: Optional[str]

    def __post_init__(self) -> None:
        identity_keys = {
            "CreateBootstrapFenceStack",
            "CreateBootstrapSupportStack",
            "ImportPolicyIntoFenceStack.CreateChangeSet",
        }
        if (
            self.key not in _MIGRATION_MUTATION_KEYS
            or type(self.status) is not MigrationMutationStatus
            or (
                self.status is MigrationMutationStatus.NOT_SUBMITTED
                and self.request_sha256 is not None
            )
            or (
                self.status is not MigrationMutationStatus.NOT_SUBMITTED
                and (
                    type(self.request_sha256) is not str
                    or _SHA256.fullmatch(self.request_sha256) is None
                )
            )
            or (
                self.key not in identity_keys
                and self.effect_identity is not None
            )
            or (
                self.effect_identity is not None
                and (
                    type(self.effect_identity) is not str
                    or not self.effect_identity
                )
            )
            or (
                self.status is MigrationMutationStatus.NOT_SUBMITTED
                and self.effect_identity is not None
            )
            or (
                self.status is MigrationMutationStatus.COMPLETE
                and self.key in identity_keys
                and self.effect_identity is None
            )
        ):
            raise ValueError("migration mutation state is not exact")


@dataclass(frozen=True)
class MigrationExecutionState:
    run_id: str
    account_id: str
    region: str
    deployment_role_arn: str
    deployment_role_id: str
    fence_stack_name: str
    support_stack_name: str
    fence_bootstrap: BootstrapCoordinate
    support_bootstrap: BootstrapCoordinate
    import_change_set_name: str
    stack_tags: Tuple[Tuple[str, str], ...]
    action_identity_sha256: str
    revision: int
    mutations: Tuple[MigrationMutationRecord, ...]

    def __post_init__(self) -> None:
        if (
            self.run_id != RUN_ID
            or self.account_id != ACCOUNT_ID
            or self.region != REGION
            or self.fence_stack_name != stack_name(StackKind.FENCE)
            or self.support_stack_name != stack_name(StackKind.SUPPORT)
            or type(self.deployment_role_arn) is not str
            or type(self.deployment_role_id) is not str
            or type(self.fence_bootstrap) is not BootstrapCoordinate
            or self.fence_bootstrap.kind is not StackKind.FENCE
            or type(self.support_bootstrap) is not BootstrapCoordinate
            or self.support_bootstrap.kind is not StackKind.SUPPORT
            or type(self.import_change_set_name) is not str
            or self.stack_tags != STACK_TAGS
            or type(self.action_identity_sha256) is not str
            or _SHA256.fullmatch(self.action_identity_sha256) is None
            or type(self.revision) is not int
            or self.revision < 0
            or type(self.mutations) is not tuple
            or any(
                type(record) is not MigrationMutationRecord
                for record in self.mutations
            )
            or tuple(record.key for record in self.mutations)
            != _MIGRATION_MUTATION_KEYS
        ):
            raise ValueError("migration execution state is not exact")
        try:
            MigrationExecutionAuthority(
                deployment_role_arn=self.deployment_role_arn,
                deployment_role_id=self.deployment_role_id,
                fence_bootstrap=self.fence_bootstrap,
                support_bootstrap=self.support_bootstrap,
                import_change_set_name=self.import_change_set_name,
                action_identity_sha256=self.action_identity_sha256,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("migration execution authority state is invalid") from exc
        records = {record.key: record for record in self.mutations}
        for key, kind in (
            ("CreateBootstrapFenceStack", StackKind.FENCE),
            ("CreateBootstrapSupportStack", StackKind.SUPPORT),
        ):
            effect = records[key].effect_identity
            if effect is not None:
                StackIdentity(
                    kind=kind,
                    name=stack_name(kind),
                    stack_id=effect,
                    account_id=self.account_id,
                    region=self.region,
                    termination_protection=True,
                    tags=self.stack_tags,
                )
        change_set_id = records[
            "ImportPolicyIntoFenceStack.CreateChangeSet"
        ].effect_identity
        if change_set_id is not None:
            _validate_change_set_id(
                change_set_id,
                expected_name=self.import_change_set_name,
            )


@dataclass(frozen=True)
class MigrationBootstrapResult:
    fence: StackIdentity
    support: StackIdentity
    action_identity_sha256: str
    state_revision: int
    reconciled_creates: Tuple[StackKind, ...]

    def __post_init__(self) -> None:
        if (
            type(self.fence) is not StackIdentity
            or self.fence.kind is not StackKind.FENCE
            or type(self.support) is not StackIdentity
            or self.support.kind is not StackKind.SUPPORT
            or self.fence.stack_id == self.support.stack_id
            or type(self.action_identity_sha256) is not str
            or _SHA256.fullmatch(self.action_identity_sha256) is None
            or type(self.state_revision) is not int
            or self.state_revision < 1
            or type(self.reconciled_creates) is not tuple
            or any(
                kind not in {StackKind.FENCE, StackKind.SUPPORT}
                for kind in self.reconciled_creates
            )
            or len(set(self.reconciled_creates))
            != len(self.reconciled_creates)
        ):
            raise ValueError("migration bootstrap result is not exact")


def initial_migration_execution_state(
    authority: MigrationExecutionAuthority,
) -> MigrationExecutionState:
    if type(authority) is not MigrationExecutionAuthority:
        raise TypeError("initial migration state requires exact authority")
    return MigrationExecutionState(
        run_id=RUN_ID,
        account_id=ACCOUNT_ID,
        region=REGION,
        deployment_role_arn=authority.deployment_role_arn,
        deployment_role_id=authority.deployment_role_id,
        fence_stack_name=stack_name(StackKind.FENCE),
        support_stack_name=stack_name(StackKind.SUPPORT),
        fence_bootstrap=authority.fence_bootstrap,
        support_bootstrap=authority.support_bootstrap,
        import_change_set_name=authority.import_change_set_name,
        stack_tags=STACK_TAGS,
        action_identity_sha256=authority.action_identity_sha256,
        revision=0,
        mutations=tuple(
            MigrationMutationRecord(
                key=key,
                status=MigrationMutationStatus.NOT_SUBMITTED,
                request_sha256=None,
                effect_identity=None,
            )
            for key in _MIGRATION_MUTATION_KEYS
        ),
    )


def migration_execution_state_projection(
    state: MigrationExecutionState,
) -> Mapping[str, object]:
    if type(state) is not MigrationExecutionState:
        raise TypeError("migration state projection requires exact state")
    return {
        "schema_version": 2,
        "record_type": "glm52_h1g_migration_execution_state_v2",
        "run_id": state.run_id,
        "account_id": state.account_id,
        "region": state.region,
        "deployment_role_arn": state.deployment_role_arn,
        "deployment_role_id": state.deployment_role_id,
        "fence_stack_name": state.fence_stack_name,
        "support_stack_name": state.support_stack_name,
        "fence_bootstrap": {
            "kind": state.fence_bootstrap.kind.value,
            "template_url": state.fence_bootstrap.template_url,
            "version_id": state.fence_bootstrap.version_id,
        },
        "support_bootstrap": {
            "kind": state.support_bootstrap.kind.value,
            "template_url": state.support_bootstrap.template_url,
            "version_id": state.support_bootstrap.version_id,
        },
        "import_change_set_name": state.import_change_set_name,
        "stack_tags": dict(state.stack_tags),
        "action_identity_sha256": state.action_identity_sha256,
        "revision": state.revision,
        "mutations": [
            {
                "key": record.key,
                "status": record.status.value,
                "request_sha256": record.request_sha256,
                "effect_identity": record.effect_identity,
            }
            for record in state.mutations
        ],
    }


def migration_execution_state_from_projection(
    value: object,
) -> MigrationExecutionState:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "record_type",
        "run_id",
        "account_id",
        "region",
        "deployment_role_arn",
        "deployment_role_id",
        "fence_stack_name",
        "support_stack_name",
        "fence_bootstrap",
        "support_bootstrap",
        "import_change_set_name",
        "stack_tags",
        "action_identity_sha256",
        "revision",
        "mutations",
    }:
        raise ValueError("migration execution state projection is not exact")
    if (
        value["schema_version"] != 2
        or value["record_type"]
        != "glm52_h1g_migration_execution_state_v2"
        or type(value["mutations"]) is not list
    ):
        raise ValueError("migration execution state projection header is invalid")
    bootstrap_items = {}
    for field, expected_kind in (
        ("fence_bootstrap", StackKind.FENCE),
        ("support_bootstrap", StackKind.SUPPORT),
    ):
        item = value[field]
        if type(item) is not dict or set(item) != {
            "kind",
            "template_url",
            "version_id",
        }:
            raise ValueError("migration bootstrap state is not exact")
        try:
            coordinate = BootstrapCoordinate(
                kind=StackKind(item["kind"]),
                template_url=item["template_url"],
                version_id=item["version_id"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("migration bootstrap state is invalid") from exc
        if coordinate.kind is not expected_kind:
            raise ValueError("migration bootstrap kind is invalid")
        bootstrap_items[field] = coordinate
    tags = value["stack_tags"]
    if (
        type(tags) is not dict
        or any(
            type(key) is not str or type(item) is not str
            for key, item in tags.items()
        )
    ):
        raise ValueError("migration state tags are not exact")
    records = []
    for item in value["mutations"]:
        if type(item) is not dict or set(item) != {
            "key",
            "status",
            "request_sha256",
            "effect_identity",
        }:
            raise ValueError("migration mutation projection is not exact")
        try:
            status = MigrationMutationStatus(item["status"])
        except (TypeError, ValueError) as exc:
            raise ValueError("migration mutation status is invalid") from exc
        records.append(
            MigrationMutationRecord(
                key=item["key"],
                status=status,
                request_sha256=item["request_sha256"],
                effect_identity=item["effect_identity"],
            )
        )
    state = MigrationExecutionState(
        run_id=value["run_id"],
        account_id=value["account_id"],
        region=value["region"],
        deployment_role_arn=value["deployment_role_arn"],
        deployment_role_id=value["deployment_role_id"],
        fence_stack_name=value["fence_stack_name"],
        support_stack_name=value["support_stack_name"],
        fence_bootstrap=bootstrap_items["fence_bootstrap"],
        support_bootstrap=bootstrap_items["support_bootstrap"],
        import_change_set_name=value["import_change_set_name"],
        stack_tags=tuple(sorted(tags.items())),
        action_identity_sha256=value["action_identity_sha256"],
        revision=value["revision"],
        mutations=tuple(records),
    )
    _validate_execution_progression(state)
    return state


def migration_bootstrap_result_from_state(
    state: MigrationExecutionState,
    *,
    reconciled_creates: Tuple[StackKind, ...] = (),
) -> MigrationBootstrapResult:
    if type(state) is not MigrationExecutionState:
        raise TypeError("bootstrap result requires exact durable state")
    records = {record.key: record for record in state.mutations}
    fence_record = records["CreateBootstrapFenceStack"]
    support_record = records["CreateBootstrapSupportStack"]
    if (
        fence_record.status is not MigrationMutationStatus.COMPLETE
        or support_record.status is not MigrationMutationStatus.COMPLETE
        or fence_record.effect_identity is None
        or support_record.effect_identity is None
    ):
        raise ValueError("both bootstrap containers must be complete before build")
    return MigrationBootstrapResult(
        fence=StackIdentity(
            kind=StackKind.FENCE,
            name=state.fence_stack_name,
            stack_id=fence_record.effect_identity,
            account_id=state.account_id,
            region=state.region,
            termination_protection=True,
            tags=state.stack_tags,
        ),
        support=StackIdentity(
            kind=StackKind.SUPPORT,
            name=state.support_stack_name,
            stack_id=support_record.effect_identity,
            account_id=state.account_id,
            region=state.region,
            termination_protection=True,
            tags=state.stack_tags,
        ),
        action_identity_sha256=state.action_identity_sha256,
        state_revision=state.revision,
        reconciled_creates=reconciled_creates,
    )


class CanonicalMigrationStateStore:
    """Exact conditional-revision port for a durable canonical state backend."""

    def __init__(self, *, backend: object) -> None:
        self._backend = backend

    def load(self, *, action_identity_sha256: str) -> MigrationExecutionState:
        read = getattr(self._backend, "read_state", None)
        if not callable(read):
            raise ValueError("migration state backend lacks read_state")
        response = read(action_identity_sha256=action_identity_sha256)
        if (
            type(response) is not dict
            or set(response) != {"State", "StateBodySha256"}
            or type(response["State"]) is not dict
            or response["StateBodySha256"]
            != hashlib.sha256(
                canonical_json_bytes(response["State"])
            ).hexdigest()
        ):
            raise ValueError("durable migration state readback is not canonical")
        state = migration_execution_state_from_projection(response["State"])
        if state.action_identity_sha256 != action_identity_sha256:
            raise ValueError("durable migration state belongs to another action")
        return state

    def save(
        self,
        *,
        expected_revision: int,
        state: MigrationExecutionState,
    ) -> MigrationExecutionState:
        if (
            type(expected_revision) is not int
            or expected_revision < 0
            or type(state) is not MigrationExecutionState
            or state.revision != expected_revision + 1
        ):
            raise ValueError("migration state revision transition is not exact")
        write = getattr(self._backend, "conditional_write_state", None)
        if not callable(write):
            raise ValueError(
                "migration state backend lacks conditional_write_state"
            )
        projection = migration_execution_state_projection(state)
        body_sha256 = hashlib.sha256(
            canonical_json_bytes(projection)
        ).hexdigest()
        response = write(
            action_identity_sha256=state.action_identity_sha256,
            expected_revision=expected_revision,
            State=projection,
            StateBodySha256=body_sha256,
        )
        if (
            type(response) is not dict
            or response.get("Committed") is not True
            or response.get("PriorRevision") != expected_revision
            or response.get("State") != projection
            or response.get("StateBodySha256") != body_sha256
        ):
            raise ValueError("conditional migration state commit is not exact")
        return migration_execution_state_from_projection(response["State"])


@dataclass(frozen=True)
class MigrationExecutionResult:
    fence_stack_id: str
    support_stack_id: str
    import_change_set_id: str
    operation_count: int
    reconciled_creates: Tuple[StackKind, ...]
    direct_policy_sha256: Tuple[str, str, str]


class AmbiguousMigrationTransportError(RuntimeError):
    """The one submitted migration mutation has an unknown transport outcome."""


class StackAlreadyExistsError(RuntimeError):
    """CreateStack definitely reported that the deterministic name exists."""


class StackNotFoundError(RuntimeError):
    """DescribeStacks proved the exact deterministic stack name absent."""


class ChangeSetNotFoundError(RuntimeError):
    """DescribeChangeSet proved the deterministic change-set name absent."""


def _validate_template_url(template_url: str, version_id: str) -> None:
    if (
        type(template_url) is not str
        or type(version_id) is not str
        or _VERSION_ID.fullmatch(version_id) is None
        or "*" in template_url
    ):
        raise ValueError("immutable template URL and VersionId are required")
    parsed = urlparse(template_url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.fragment
        or not parsed.hostname
        or not parsed.hostname.endswith(f".s3.{REGION}.amazonaws.com")
        or not parsed.path.startswith("/")
        or parsed.path in {"", "/"}
        or query != {"versionId": [version_id]}
    ):
        raise ValueError("template URL is not one exact regional versioned S3 URL")


def build_support_disabled_transition(
    *,
    bootstrap_template: Mapping[str, object],
    reviewed_inventory: Mapping[str, object],
    reviewed_inventory_sha256: str,
    retained_resource_physical_ids: Tuple[str, ...],
    rehearsal_bucket_name: str,
) -> Dict[str, object]:
    validate_bootstrap_template(bootstrap_template)
    if type(reviewed_inventory) is not dict or not reviewed_inventory:
        raise ValueError("reviewed support inventory must be nonempty")
    if (
        type(reviewed_inventory_sha256) is not str
        or _SHA256.fullmatch(reviewed_inventory_sha256) is None
        or hashlib.sha256(canonical_json_bytes(reviewed_inventory)).hexdigest()
        != reviewed_inventory_sha256
    ):
        raise ValueError("reviewed support inventory SHA-256 mismatch")
    if (
        type(retained_resource_physical_ids) is not tuple
        or not retained_resource_physical_ids
        or len(set(retained_resource_physical_ids))
        != len(retained_resource_physical_ids)
        or any(type(value) is not str or not value for value in retained_resource_physical_ids)
        or rehearsal_bucket_name != _REHEARSAL_BUCKET_NAME
        or rehearsal_bucket_name in retained_resource_physical_ids
    ):
        raise ValueError("support inventory owner identities are not exact")
    for logical_id, resource in reviewed_inventory.items():
        if (
            type(logical_id) is not str
            or _LOGICAL_ID.fullmatch(logical_id) is None
            or logical_id
            in {
                CONTAINER_ANCHOR,
                FENCE_LOGICAL_ID,
                "ModelBucket",
                "EvidenceBucket",
                "CampaignKmsKey",
                "H1gLedger",
            }
            or type(resource) is not dict
            or type(resource.get("Type")) is not str
            or resource["Type"] in _SUPPORT_FORBIDDEN_TYPES
        ):
            raise ValueError("reviewed inventory contains non-support authority")
        properties = resource.get("Properties", {})
        if type(properties) is not dict:
            raise ValueError("reviewed support resource properties are not exact")
        resource_type = resource["Type"]
        for property_name in _RESOURCE_IDENTITY_PROPERTIES.get(resource_type, ()):
            physical_id = properties.get(property_name)
            if (
                physical_id is not None
                and (
                    type(physical_id) is not str
                    or not physical_id
                    or physical_id in retained_resource_physical_ids
                )
            ):
                raise ValueError("support resource aliases a retained physical identity")
        if (
            resource_type == "AWS::S3::Bucket"
            and properties.get("BucketName") != rehearsal_bucket_name
        ):
            raise ValueError("only the exact Task7 rehearsal bucket is support-owned")
    result = deepcopy(bootstrap_template)
    result["Resources"] = deepcopy(reviewed_inventory)
    return result


def _import_values(value: object) -> Tuple[str, ...]:
    found = []
    if type(value) is dict:
        for key, child in value.items():
            if key == "Fn::ImportValue":
                if type(child) is not str or not child:
                    raise ValueError("stack import must be one exact output name")
                found.append(child)
            found.extend(_import_values(child))
    elif type(value) is list:
        for child in value:
            found.extend(_import_values(child))
    return tuple(found)


def validate_one_way_stack_graph(
    *,
    retained_template: Mapping[str, object],
    fence_template: Mapping[str, object],
    support_template: Mapping[str, object],
    retained_export_names: Tuple[str, ...],
    support_export_names: Tuple[str, ...],
) -> bool:
    for template in (retained_template, fence_template, support_template):
        _resources(template)
    retained_imports = _import_values(retained_template)
    fence_imports = _import_values(fence_template)
    support_imports = _import_values(support_template)
    for label, names in (
        ("retained", retained_export_names),
        ("support", support_export_names),
    ):
        if (
            type(names) is not tuple
            or len(set(names)) != len(names)
            or any(type(name) is not str or not name for name in names)
        ):
            raise ValueError(f"{label} export allowlist is not exact")
    retained_exports = set(retained_export_names)
    support_exports = set(support_export_names)
    if retained_exports & support_exports:
        raise ValueError("export ownership is ambiguous")
    if set(retained_imports) & support_exports:
        raise ValueError("retained infrastructure imports a support output")
    if fence_imports:
        raise ValueError("retained fence stack cannot import any output")
    if not set(support_imports).issubset(retained_exports):
        raise ValueError("support stack imports an unowned or support output")
    return True


_MIGRATION_STAGES: Tuple[Tuple[str, StackKind], ...] = (
    ("retention-only", StackKind.RETAINED),
    ("post-retain", StackKind.RETAINED),
    ("fence-import", StackKind.FENCE),
    ("fence-transfer", StackKind.FENCE),
    ("disabled-support", StackKind.SUPPORT),
)

_MIGRATION_OPERATIONS = (
    "CreateBootstrapFenceStack",
    "CreateBootstrapSupportStack",
    "UpdateRetainedWithRetainAttributes",
    "UpdateRetainedRemovePolicy",
    "ImportPolicyIntoFenceStack",
    "UpdateFenceRemoveAnchor",
    "UpdateSupportReplaceAnchor",
)


def migration_operation_plan() -> Tuple[str, ...]:
    return _MIGRATION_OPERATIONS


def _artifact(
    *,
    stage: str,
    template: Mapping[str, object],
    coordinate: TemplateCoordinate,
) -> TemplateArtifact:
    canonical_body = canonical_json_bytes(template)
    body = canonical_body + b"\n"
    resources = _resources(template)
    policy_sha256 = None
    if FENCE_LOGICAL_ID in resources:
        resource = resources[FENCE_LOGICAL_ID]
        if type(resource) is not dict or type(resource.get("Properties")) is not dict:
            raise ValueError("fence policy resource is malformed")
        policy_sha256 = hashlib.sha256(
            canonical_json_bytes(resource["Properties"]["PolicyDocument"])
        ).hexdigest()
    return TemplateArtifact(
        stage=stage,
        stack_id=coordinate.stack_id,
        template_url=coordinate.template_url,
        version_id=coordinate.version_id,
        body=body,
        sha256=hashlib.sha256(body).hexdigest(),
        template_body_sha256=hashlib.sha256(canonical_body).hexdigest(),
        policy_sha256=policy_sha256,
        template=deepcopy(template),
    )


def build_stack_migration(
    *,
    archived_template: Mapping[str, object],
    bootstrap_template: Mapping[str, object],
    bootstrap_result: MigrationBootstrapResult,
    evidence: MigrationEvidence,
    template_coordinates: Tuple[TemplateCoordinate, ...],
    reviewed_support_inventory: Mapping[str, object],
    reviewed_support_inventory_sha256: str,
) -> MigrationBundle:
    raise ValueError(
        "stack-migration v1 inputs are audit-only and execution-ineligible"
    )


def _successful_response(response: object, operation: str) -> Mapping[str, object]:
    if type(response) is not dict:
        raise ValueError(f"{operation} response is not exact")
    metadata = response.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") not in {200, 201}
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
    ):
        raise ValueError(f"{operation} response lacks successful request evidence")
    return response


def _migration_token(
    *, authority: MigrationExecutionAuthority, operation: str, identity: str
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "run_id": RUN_ID,
                "operation": operation,
                "identity": identity,
                "action_identity_sha256": authority.action_identity_sha256,
            }
        )
    ).hexdigest()


def _validate_change_set_id(
    change_set_id: str, *, expected_name: str
) -> str:
    if type(change_set_id) is not str or type(expected_name) is not str:
        raise ValueError("change set service identity is absent")
    prefix = (
        f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:"
        f"changeSet/{expected_name}/"
    )
    suffix = change_set_id.removeprefix(prefix)
    if (
        not change_set_id.startswith(prefix)
        or _CHANGE_SET_UUID.fullmatch(suffix) is None
    ):
        raise ValueError("change set service identity is not exact")
    return change_set_id


def _stack_tags_from_readback(value: object) -> Tuple[Tuple[str, str], ...]:
    if type(value) is not list:
        raise ValueError("stack tags are absent")
    tags = []
    for item in value:
        if (
            type(item) is not dict
            or set(item) != {"Key", "Value"}
            or type(item["Key"]) is not str
            or type(item["Value"]) is not str
        ):
            raise ValueError("stack tags are not exact")
        tags.append((item["Key"], item["Value"]))
    return tuple(sorted(tags))


def _exact_stack_readback(
    response: object,
    *,
    identity: StackIdentity,
    role_arn: str,
    status: str,
) -> None:
    item = _successful_response(response, "DescribeStacks")
    stacks = item.get("Stacks")
    if type(stacks) is not list or len(stacks) != 1 or type(stacks[0]) is not dict:
        raise ValueError("stack readback is absent or duplicated")
    stack = stacks[0]
    if (
        stack.get("StackId") != identity.stack_id
        or stack.get("StackName") != identity.name
        or stack.get("StackStatus") != status
        or stack.get("EnableTerminationProtection") is not True
        or stack.get("RoleARN") != role_arn
        or _stack_tags_from_readback(stack.get("Tags")) != STACK_TAGS
    ):
        raise ValueError("stack readback does not match exact protected identity")


def _direct_policy_sha256(
    client: object, *, bucket_name: str, expected_sha256: str
) -> str:
    response = _successful_response(
        client.get_bucket_policy(
            Bucket=bucket_name,
            ExpectedBucketOwner=ACCOUNT_ID,
        ),
        "GetBucketPolicy",
    )
    raw_policy = response.get("Policy")
    if type(raw_policy) is str:
        try:
            policy = json.loads(raw_policy)
        except json.JSONDecodeError as exc:
            raise ValueError("direct policy readback is not JSON") from exc
    elif type(raw_policy) is dict:
        policy = raw_policy
    else:
        raise ValueError("direct policy readback is absent")
    observed = hashlib.sha256(canonical_json_bytes(policy)).hexdigest()
    if observed != expected_sha256:
        raise ValueError("direct bucket policy changed during migration")
    return observed


_STABILIZATION_READ_LIMIT = 8
_CREATE_PROGRESS = ("REVIEW_IN_PROGRESS", "CREATE_IN_PROGRESS")
_UPDATE_PROGRESS = (
    "UPDATE_IN_PROGRESS",
    "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS",
)
_IMPORT_PROGRESS = ("IMPORT_IN_PROGRESS",)
_STACK_TERMINAL = ("CREATE_COMPLETE", "UPDATE_COMPLETE", "IMPORT_COMPLETE")


def _request_sha256(request: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(request)).hexdigest()


def _validate_execution_progression(state: MigrationExecutionState) -> None:
    incomplete_seen = False
    submitted_seen = False
    for record in state.mutations:
        if record.status is MigrationMutationStatus.COMPLETE:
            if incomplete_seen:
                raise ValueError("migration execution state is out of order")
            continue
        if record.status is MigrationMutationStatus.SUBMITTED:
            if submitted_seen or incomplete_seen:
                raise ValueError("migration execution has multiple submitted mutations")
            submitted_seen = True
        incomplete_seen = True


def _state_record(
    state: MigrationExecutionState, key: str
) -> MigrationMutationRecord:
    return next(record for record in state.mutations if record.key == key)


def _persist_mutation(
    *,
    store: object,
    state: MigrationExecutionState,
    key: str,
    status: MigrationMutationStatus,
    request_sha256: str,
    effect_identity: Optional[str] = None,
) -> MigrationExecutionState:
    records = []
    for record in state.mutations:
        if record.key == key:
            records.append(
                MigrationMutationRecord(
                    key=key,
                    status=status,
                    request_sha256=request_sha256,
                    effect_identity=effect_identity,
                )
            )
        else:
            records.append(record)
    proposed = MigrationExecutionState(
        run_id=state.run_id,
        account_id=state.account_id,
        region=state.region,
        deployment_role_arn=state.deployment_role_arn,
        deployment_role_id=state.deployment_role_id,
        fence_stack_name=state.fence_stack_name,
        support_stack_name=state.support_stack_name,
        fence_bootstrap=state.fence_bootstrap,
        support_bootstrap=state.support_bootstrap,
        import_change_set_name=state.import_change_set_name,
        stack_tags=state.stack_tags,
        action_identity_sha256=state.action_identity_sha256,
        revision=state.revision + 1,
        mutations=tuple(records),
    )
    _validate_execution_progression(proposed)
    save = getattr(store, "save", None)
    if not callable(save):
        raise ValueError("migration state store lacks exact save")
    saved = save(expected_revision=state.revision, state=proposed)
    if type(saved) is not MigrationExecutionState or saved != proposed:
        raise ValueError("migration state persistence readback mismatches")
    return saved


def _begin_mutation(
    *,
    store: object,
    state: MigrationExecutionState,
    key: str,
    request: Mapping[str, object],
) -> Tuple[MigrationExecutionState, bool, str]:
    digest = _request_sha256(request)
    record = _state_record(state, key)
    if record.status is MigrationMutationStatus.NOT_SUBMITTED:
        state = _persist_mutation(
            store=store,
            state=state,
            key=key,
            status=MigrationMutationStatus.SUBMITTED,
            request_sha256=digest,
            effect_identity=None,
        )
        return state, True, digest
    if record.request_sha256 != digest:
        raise ValueError("persisted migration request digest is stale")
    return state, False, digest


def _complete_mutation(
    *,
    store: object,
    state: MigrationExecutionState,
    key: str,
    request: Mapping[str, object],
    effect_identity: Optional[str] = None,
) -> MigrationExecutionState:
    digest = _request_sha256(request)
    record = _state_record(state, key)
    if record.status is MigrationMutationStatus.COMPLETE:
        if (
            record.request_sha256 != digest
            or (
                effect_identity is not None
                and record.effect_identity != effect_identity
            )
        ):
            raise ValueError("completed migration request digest is stale")
        return state
    if (
        record.status is MigrationMutationStatus.SUBMITTED
        and record.request_sha256 != digest
    ):
        raise ValueError("submitted migration request digest is stale")
    return _persist_mutation(
        store=store,
        state=state,
        key=key,
        status=MigrationMutationStatus.COMPLETE,
        request_sha256=digest,
        effect_identity=(
            effect_identity
            if effect_identity is not None
            else record.effect_identity
        ),
    )


def _bind_effect_identity(
    *,
    store: object,
    state: MigrationExecutionState,
    key: str,
    request: Mapping[str, object],
    effect_identity: str,
) -> MigrationExecutionState:
    digest = _request_sha256(request)
    record = _state_record(state, key)
    if (
        record.status is MigrationMutationStatus.NOT_SUBMITTED
        or record.request_sha256 != digest
    ):
        raise ValueError("effect identity lacks persisted exact intent")
    if record.effect_identity is not None:
        if record.effect_identity != effect_identity:
            raise ValueError("service effect identity changed after binding")
        return state
    return _persist_mutation(
        store=store,
        state=state,
        key=key,
        status=record.status,
        request_sha256=digest,
        effect_identity=effect_identity,
    )


def _describe_exact_stack(
    client: object,
    *,
    lookup: str,
    identity: StackIdentity,
    role_arn: str,
) -> Optional[Mapping[str, object]]:
    try:
        response = client.describe_stacks(StackName=lookup)
    except StackNotFoundError:
        return None
    item = _successful_response(response, "DescribeStacks")
    stacks = item.get("Stacks")
    if type(stacks) is not list or len(stacks) != 1 or type(stacks[0]) is not dict:
        raise ValueError("stack readback is absent or duplicated")
    stack = stacks[0]
    if (
        stack.get("StackId") != identity.stack_id
        or stack.get("StackName") != identity.name
        or stack.get("EnableTerminationProtection") is not True
        or stack.get("RoleARN") != role_arn
        or _stack_tags_from_readback(stack.get("Tags")) != STACK_TAGS
        or type(stack.get("StackStatus")) is not str
    ):
        raise ValueError("stack readback replaced or mismatched protected identity")
    return stack


def _describe_named_stack(
    client: object,
    *,
    kind: StackKind,
    role_arn: str,
) -> Optional[Tuple[StackIdentity, Mapping[str, object]]]:
    name = stack_name(kind)
    try:
        response = client.describe_stacks(StackName=name)
    except StackNotFoundError:
        return None
    item = _successful_response(response, "DescribeStacks")
    stacks = item.get("Stacks")
    if type(stacks) is not list or len(stacks) != 1 or type(stacks[0]) is not dict:
        raise ValueError("stack readback is absent or duplicated")
    stack = stacks[0]
    if (
        stack.get("StackName") != name
        or stack.get("EnableTerminationProtection") is not True
        or stack.get("RoleARN") != role_arn
        or _stack_tags_from_readback(stack.get("Tags")) != STACK_TAGS
        or type(stack.get("StackStatus")) is not str
    ):
        raise ValueError("named stack readback is not exact")
    identity = StackIdentity(
        kind=kind,
        name=name,
        stack_id=stack.get("StackId"),
        account_id=ACCOUNT_ID,
        region=REGION,
        termination_protection=True,
        tags=STACK_TAGS,
    )
    return identity, stack


def _exact_json_object(
    pairs: object,
) -> Mapping[str, object]:
    if type(pairs) is not list:
        raise ValueError("JSON object pairs are not exact")
    result = {}
    for pair in pairs:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
        ):
            raise ValueError("JSON object member is not exact")
        key, value = pair
        if key in result:
            raise ValueError("TemplateBody JSON contains a duplicate key")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"TemplateBody JSON contains non-finite constant {value}")


def _validate_exact_json_tree(value: object) -> object:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("TemplateBody JSON object key is not exact string")
            _validate_exact_json_tree(item)
        return value
    if type(value) is list:
        for item in value:
            _validate_exact_json_tree(item)
        return value
    if type(value) is float and not math.isfinite(value):
        raise ValueError("TemplateBody JSON number is non-finite")
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise ValueError("TemplateBody contains a noncanonical JSON value")


def _template_body_mapping(
    *, item: Mapping[str, object], operation: str
) -> Mapping[str, object]:
    raw = item.get("TemplateBody")
    if type(raw) is dict:
        template = raw
    elif type(raw) is str and raw:
        try:
            template = json.loads(
                raw,
                object_pairs_hook=_exact_json_object,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{operation} TemplateBody must be exact JSON"
            ) from exc
    else:
        raise ValueError(f"{operation} TemplateBody has an unsupported shape")
    _validate_exact_json_tree(template)
    if type(template) is not dict:
        raise ValueError(f"{operation} TemplateBody JSON root is not an object")
    return template


def _get_template_sha256(
    client: object, *, request: Mapping[str, object], operation: str
) -> str:
    item = _successful_response(
        client.get_template(**request),
        operation,
    )
    template = _template_body_mapping(
        item=item, operation=operation
    )
    return hashlib.sha256(canonical_json_bytes(template)).hexdigest()


def _stack_template_sha256(client: object, *, stack_id: str) -> str:
    return _get_template_sha256(
        client,
        request={"StackName": stack_id, "TemplateStage": "Processed"},
        operation="GetTemplate(Stack)",
    )


def _stabilize_stack(
    client: object,
    *,
    lookup: str,
    identity: StackIdentity,
    role_arn: str,
    expected_template_sha256: Tuple[str, ...],
    transient_template_sha256: Tuple[str, ...],
    terminal_statuses: Tuple[str, ...],
    progress_statuses: Tuple[str, ...],
    allow_absent: bool,
) -> Tuple[str, str]:
    for _index in range(_STABILIZATION_READ_LIMIT):
        stack = _describe_exact_stack(
            client,
            lookup=lookup,
            identity=identity,
            role_arn=role_arn,
        )
        if stack is None:
            if allow_absent:
                continue
            raise ValueError("required migration stack is absent")
        status = stack["StackStatus"]
        if status in progress_statuses:
            continue
        if status not in terminal_statuses:
            raise ValueError("migration stack entered rollback/failure/mismatched state")
        template_sha256 = _stack_template_sha256(
            client, stack_id=identity.stack_id
        )
        if template_sha256 in expected_template_sha256:
            return status, template_sha256
        if template_sha256 in transient_template_sha256:
            continue
        raise ValueError("migration stack terminal template is not exact")
    raise ValueError("migration stack stabilization read limit exhausted")


class MigrationCoordinator:
    """Resumable, persisted-intent executor for the closed H.1g migration."""

    def __init__(self, *, client: object, state_store: object) -> None:
        self._client = client
        self._state_store = state_store

    def _load_state(
        self, authority: MigrationExecutionAuthority
    ) -> MigrationExecutionState:
        load = getattr(self._state_store, "load", None)
        if not callable(load):
            raise ValueError("migration state store lacks exact load")
        state = load(
            action_identity_sha256=authority.action_identity_sha256
        )
        if (
            type(state) is not MigrationExecutionState
            or state.action_identity_sha256
            != authority.action_identity_sha256
            or state.account_id != ACCOUNT_ID
            or state.region != REGION
            or state.deployment_role_arn != authority.deployment_role_arn
            or state.deployment_role_id != authority.deployment_role_id
            or state.fence_stack_name != stack_name(StackKind.FENCE)
            or state.support_stack_name != stack_name(StackKind.SUPPORT)
            or state.fence_bootstrap != authority.fence_bootstrap
            or state.support_bootstrap != authority.support_bootstrap
            or state.import_change_set_name
            != authority.import_change_set_name
            or state.stack_tags != STACK_TAGS
        ):
            raise ValueError("migration execution state readback is not exact")
        _validate_execution_progression(state)
        return state

    def _bootstrap_container(
        self,
        *,
        state: MigrationExecutionState,
        key: str,
        kind: StackKind,
        coordinate: BootstrapCoordinate,
        authority: MigrationExecutionAuthority,
    ) -> Tuple[MigrationExecutionState, StackIdentity, bool]:
        name = stack_name(kind)
        request = {
            "StackName": name,
            "TemplateURL": coordinate.template_url,
            "RoleARN": authority.deployment_role_arn,
            "EnableTerminationProtection": True,
            "Tags": [
                {"Key": tag_key, "Value": value}
                for tag_key, value in STACK_TAGS
            ],
            "ClientRequestToken": _migration_token(
                authority=authority,
                operation=f"create-{kind.value}",
                identity=hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "stack_name": name,
                            "template_url": coordinate.template_url,
                            "version_id": coordinate.version_id,
                            "role_arn": authority.deployment_role_arn,
                            "tags": dict(STACK_TAGS),
                        }
                    )
                ).hexdigest(),
            ),
        }
        record = _state_record(state, key)
        observed = _describe_named_stack(
            self._client, kind=kind, role_arn=authority.deployment_role_arn
        )
        if record.status is MigrationMutationStatus.NOT_SUBMITTED and observed:
            raise ValueError("caller-untracked deterministic stack cannot be adopted")
        state, submit, _digest = _begin_mutation(
            store=self._state_store,
            state=state,
            key=key,
            request=request,
        )
        reconciled = not submit
        returned_stack_id = None
        if submit:
            try:
                response = self._client.create_stack(**request)
            except (AmbiguousMigrationTransportError, StackAlreadyExistsError):
                reconciled = True
            else:
                item = _successful_response(response, "CreateStack")
                returned_stack_id = item.get("StackId")
                StackIdentity(
                    kind=kind,
                    name=name,
                    stack_id=returned_stack_id,
                    account_id=ACCOUNT_ID,
                    region=REGION,
                    termination_protection=True,
                    tags=STACK_TAGS,
                )
        identity = None
        bootstrap_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "AWSTemplateFormatVersion": "2010-09-09",
                    "Resources": {
                        CONTAINER_ANCHOR: {
                            "Type": "AWS::CloudFormation::WaitConditionHandle"
                        }
                    },
                }
            )
        ).hexdigest()
        for _index in range(_STABILIZATION_READ_LIMIT):
            described = _describe_named_stack(
                self._client,
                kind=kind,
                role_arn=authority.deployment_role_arn,
            )
            if described is None:
                continue
            candidate, stack = described
            if (
                record.effect_identity is not None
                and candidate.stack_id != record.effect_identity
            ):
                raise ValueError("bound stack identity was replaced")
            if (
                returned_stack_id is not None
                and candidate.stack_id != returned_stack_id
            ):
                raise ValueError("CreateStack response and name readback disagree")
            if stack["StackStatus"] in _CREATE_PROGRESS:
                continue
            if stack["StackStatus"] != "CREATE_COMPLETE":
                raise ValueError("bootstrap stack entered a non-create terminal state")
            if (
                _stack_template_sha256(
                    self._client, stack_id=candidate.stack_id
                )
                != bootstrap_sha256
            ):
                raise ValueError("bootstrap stack template is not exact")
            identity = candidate
            break
        if identity is None:
            raise ValueError("bootstrap stack stabilization read limit exhausted")
        state = _bind_effect_identity(
            store=self._state_store,
            state=state,
            key=key,
            request=request,
            effect_identity=identity.stack_id,
        )
        state = _complete_mutation(
            store=self._state_store,
            state=state,
            key=key,
            request=request,
            effect_identity=identity.stack_id,
        )
        return state, identity, reconciled

    def bootstrap(
        self, *, authority: MigrationExecutionAuthority
    ) -> MigrationBootstrapResult:
        if type(authority) is not MigrationExecutionAuthority:
            raise TypeError("migration bootstrap requires exact authority")
        state = self._load_state(authority)
        reconciled = []
        state, _fence, was_reconciled = self._bootstrap_container(
            state=state,
            key="CreateBootstrapFenceStack",
            kind=StackKind.FENCE,
            coordinate=authority.fence_bootstrap,
            authority=authority,
        )
        if was_reconciled:
            reconciled.append(StackKind.FENCE)
        state, _support, was_reconciled = self._bootstrap_container(
            state=state,
            key="CreateBootstrapSupportStack",
            kind=StackKind.SUPPORT,
            coordinate=authority.support_bootstrap,
            authority=authority,
        )
        if was_reconciled:
            reconciled.append(StackKind.SUPPORT)
        return migration_bootstrap_result_from_state(
            state, reconciled_creates=tuple(reconciled)
        )

    def _update(
        self,
        *,
        state: MigrationExecutionState,
        key: str,
        identity: StackIdentity,
        artifact: TemplateArtifact,
        authority: MigrationExecutionAuthority,
        accepted_target_sha256: Tuple[str, ...],
        prior_template_sha256: Tuple[str, ...],
    ) -> MigrationExecutionState:
        if artifact.stack_id != identity.stack_id:
            raise ValueError("migration artifact targets the wrong stack")
        request = {
            "StackName": identity.stack_id,
            "TemplateURL": artifact.template_url,
            "RoleARN": authority.deployment_role_arn,
            "ClientRequestToken": _migration_token(
                authority=authority,
                operation=f"update-{artifact.stage}",
                identity=artifact.sha256,
            ),
        }
        stack = _describe_exact_stack(
            self._client,
            lookup=identity.stack_id,
            identity=identity,
            role_arn=authority.deployment_role_arn,
        )
        if stack is None:
            raise ValueError("update target stack is absent")
        if stack["StackStatus"] in _STACK_TERMINAL:
            current_sha256 = _stack_template_sha256(
                self._client, stack_id=identity.stack_id
            )
            if current_sha256 in accepted_target_sha256:
                return _complete_mutation(
                    store=self._state_store,
                    state=state,
                    key=key,
                    request=request,
                )
            if current_sha256 not in prior_template_sha256:
                raise ValueError("update target is not an exact predecessor")
        elif stack["StackStatus"] not in _UPDATE_PROGRESS:
            raise ValueError("update target entered rollback/failure state")
        if (
            stack["StackStatus"] in _UPDATE_PROGRESS
            and _state_record(state, key).status
            is MigrationMutationStatus.NOT_SUBMITTED
        ):
            raise ValueError("untracked stack update is in progress")
        state, submit, _digest = _begin_mutation(
            store=self._state_store,
            state=state,
            key=key,
            request=request,
        )
        if submit:
            try:
                response = self._client.update_stack(**request)
            except AmbiguousMigrationTransportError:
                pass
            else:
                _successful_response(response, "UpdateStack")
        _stabilize_stack(
            self._client,
            lookup=identity.stack_id,
            identity=identity,
            role_arn=authority.deployment_role_arn,
            expected_template_sha256=accepted_target_sha256,
            transient_template_sha256=prior_template_sha256,
            terminal_statuses=("UPDATE_COMPLETE",),
            progress_statuses=_UPDATE_PROGRESS,
            allow_absent=False,
        )
        return _complete_mutation(
            store=self._state_store,
            state=state,
            key=key,
            request=request,
        )

    def _import_request(
        self,
        *,
        evidence: MigrationEvidence,
        authority: MigrationExecutionAuthority,
        artifact: TemplateArtifact,
    ) -> Mapping[str, object]:
        return {
            "StackName": evidence.fence.stack_id,
            "ChangeSetName": authority.import_change_set_name,
            "ChangeSetType": "IMPORT",
            "TemplateURL": artifact.template_url,
            "RoleARN": authority.deployment_role_arn,
            "ResourcesToImport": [
                {
                    "ResourceType": "AWS::S3::BucketPolicy",
                    "LogicalResourceId": FENCE_LOGICAL_ID,
                    "ResourceIdentifier": {"Bucket": evidence.bucket_name},
                }
            ],
            "IncludeNestedStacks": False,
            "ClientToken": _migration_token(
                authority=authority,
                operation="import-policy",
                identity=artifact.sha256,
            ),
        }

    def _describe_import_change_set(
        self,
        *,
        evidence: MigrationEvidence,
        authority: MigrationExecutionAuthority,
        artifact: TemplateArtifact,
        expected_change_set_id: Optional[str] = None,
    ) -> Optional[Mapping[str, object]]:
        try:
            response = self._client.describe_change_set(
                ChangeSetName=authority.import_change_set_name,
                StackName=evidence.fence.stack_id,
            )
        except ChangeSetNotFoundError:
            return None
        item = _successful_response(response, "DescribeChangeSet")
        expected_changes = [
            {
                "Type": "Resource",
                "ResourceChange": {
                    "Action": "Import",
                    "LogicalResourceId": FENCE_LOGICAL_ID,
                    "PhysicalResourceId": evidence.bucket_name,
                    "ResourceType": "AWS::S3::BucketPolicy",
                    "Replacement": "False",
                    "Scope": [],
                    "Details": [],
                },
            }
        ]
        if (
            type(item.get("ChangeSetId")) is not str
            or not item["ChangeSetId"]
            or item.get("ChangeSetName") != authority.import_change_set_name
            or item.get("StackId") != evidence.fence.stack_id
            or item.get("StackName") != evidence.fence.name
            or item.get("IncludeNestedStacks") is not False
            or type(item.get("Status")) is not str
            or type(item.get("ExecutionStatus")) is not str
        ):
            raise ValueError("import change set identity/readback is not exact")
        change_set_id = _validate_change_set_id(
            item["ChangeSetId"],
            expected_name=authority.import_change_set_name,
        )
        if (
            expected_change_set_id is not None
            and change_set_id != expected_change_set_id
        ):
            raise ValueError("import change set service identity was replaced")
        if item["Status"] == "CREATE_COMPLETE":
            if item.get("Changes") != expected_changes:
                raise ValueError("import change set resource delta is not exact")
            original_sha256 = _get_template_sha256(
                self._client,
                request={
                    "ChangeSetName": item["ChangeSetId"],
                    "TemplateStage": "Original",
                },
                operation="GetTemplate(ChangeSet Original)",
            )
            processed_sha256 = _get_template_sha256(
                self._client,
                request={
                    "ChangeSetName": item["ChangeSetId"],
                    "TemplateStage": "Processed",
                },
                operation="GetTemplate(ChangeSet Processed)",
            )
            if (
                original_sha256 != artifact.template_body_sha256
                or processed_sha256 != artifact.template_body_sha256
            ):
                raise ValueError("import change set template is not exact")
        elif item["Status"] not in {"CREATE_PENDING", "CREATE_IN_PROGRESS"}:
            raise ValueError("import change set progress shape is not exact")
        elif (
            item.get("Changes") is not None
            and item.get("Changes") != []
            and item.get("Changes") != expected_changes
        ):
            raise ValueError("import change set progress delta is incoherent")
        return item

    def _stabilize_change_set_creation(
        self,
        *,
        evidence: MigrationEvidence,
        authority: MigrationExecutionAuthority,
        artifact: TemplateArtifact,
        expected_change_set_id: Optional[str],
    ) -> Mapping[str, object]:
        for _index in range(_STABILIZATION_READ_LIMIT):
            item = self._describe_import_change_set(
                evidence=evidence,
                authority=authority,
                artifact=artifact,
                expected_change_set_id=expected_change_set_id,
            )
            if item is None:
                continue
            if item["Status"] in {"CREATE_PENDING", "CREATE_IN_PROGRESS"}:
                continue
            if (
                item["Status"] == "CREATE_COMPLETE"
                and item["ExecutionStatus"]
                in {"AVAILABLE", "EXECUTE_IN_PROGRESS", "EXECUTE_COMPLETE"}
            ):
                return item
            raise ValueError("import change set create failed or rolled back")
        raise ValueError("import change set creation stabilization exhausted")

    def _create_import_change_set(
        self,
        *,
        state: MigrationExecutionState,
        evidence: MigrationEvidence,
        authority: MigrationExecutionAuthority,
        artifact: TemplateArtifact,
        fence_stage_sha256: str,
        fence_transfer_sha256: str,
    ) -> Tuple[MigrationExecutionState, Mapping[str, object]]:
        request = self._import_request(
            evidence=evidence,
            authority=authority,
            artifact=artifact,
        )
        key = "ImportPolicyIntoFenceStack.CreateChangeSet"
        record = _state_record(state, key)
        expected_change_set_id = record.effect_identity
        existing = self._describe_import_change_set(
            evidence=evidence,
            authority=authority,
            artifact=artifact,
            expected_change_set_id=expected_change_set_id,
        )
        if (
            existing is not None
            and record.status is MigrationMutationStatus.NOT_SUBMITTED
        ):
            raise ValueError(
                "caller-untracked import change set cannot be adopted"
            )
        if fence_stage_sha256 in {
            artifact.template_body_sha256,
            fence_transfer_sha256,
        }:
            if existing is None:
                execute_record = _state_record(
                    state,
                    "ImportPolicyIntoFenceStack.ExecuteChangeSet",
                )
                if (
                    fence_stage_sha256 != fence_transfer_sha256
                    or record.status is not MigrationMutationStatus.COMPLETE
                    or expected_change_set_id is None
                    or execute_record.status
                    is not MigrationMutationStatus.COMPLETE
                ):
                    raise ValueError(
                        "import change set disappeared before exact transfer state"
                    )
                return state, {
                    "ChangeSetId": expected_change_set_id,
                    "ExecutionStatus": "EXECUTE_COMPLETE",
                }
            change_set_id = existing["ChangeSetId"]
            state = _bind_effect_identity(
                store=self._state_store,
                state=state,
                key=key,
                request=request,
                effect_identity=change_set_id,
            )
            state = _complete_mutation(
                store=self._state_store,
                state=state,
                key=key,
                request=request,
                effect_identity=change_set_id,
            )
            return state, existing
        if existing is not None and existing["Status"] == "CREATE_COMPLETE":
            change_set_id = existing["ChangeSetId"]
            state = _bind_effect_identity(
                store=self._state_store,
                state=state,
                key=key,
                request=request,
                effect_identity=change_set_id,
            )
            state = _complete_mutation(
                store=self._state_store,
                state=state,
                key=key,
                request=request,
                effect_identity=change_set_id,
            )
            return state, existing
        existing_in_progress = existing is not None
        state, submit, _digest = _begin_mutation(
            store=self._state_store,
            state=state,
            key=key,
            request=request,
        )
        if submit and not existing_in_progress:
            try:
                response = self._client.create_change_set(**request)
            except AmbiguousMigrationTransportError:
                pass
            else:
                item = _successful_response(response, "CreateChangeSet")
                returned_id = _validate_change_set_id(
                    item.get("Id"),
                    expected_name=authority.import_change_set_name,
                )
                state = _bind_effect_identity(
                    store=self._state_store,
                    state=state,
                    key=key,
                    request=request,
                    effect_identity=returned_id,
                )
                expected_change_set_id = returned_id
        record = _state_record(state, key)
        expected_change_set_id = record.effect_identity
        described = self._stabilize_change_set_creation(
            evidence=evidence,
            authority=authority,
            artifact=artifact,
            expected_change_set_id=expected_change_set_id,
        )
        change_set_id = described["ChangeSetId"]
        state = _bind_effect_identity(
            store=self._state_store,
            state=state,
            key=key,
            request=request,
            effect_identity=change_set_id,
        )
        state = _complete_mutation(
            store=self._state_store,
            state=state,
            key=key,
            request=request,
            effect_identity=change_set_id,
        )
        return state, described

    def _execute_import(
        self,
        *,
        state: MigrationExecutionState,
        evidence: MigrationEvidence,
        authority: MigrationExecutionAuthority,
        artifact: TemplateArtifact,
        described: Mapping[str, object],
        fence_transfer_sha256: str,
        bootstrap_sha256: str,
    ) -> Tuple[MigrationExecutionState, str]:
        change_set_id = _validate_change_set_id(
            described["ChangeSetId"],
            expected_name=authority.import_change_set_name,
        )
        create_record = _state_record(
            state, "ImportPolicyIntoFenceStack.CreateChangeSet"
        )
        if create_record.effect_identity != change_set_id:
            raise ValueError("import execution is not bound to the created change set")
        execute_request = {
            "ChangeSetName": change_set_id,
            "StackName": evidence.fence.stack_id,
            "ClientRequestToken": _migration_token(
                authority=authority,
                operation="execute-import",
                identity=change_set_id,
            ),
        }
        key = "ImportPolicyIntoFenceStack.ExecuteChangeSet"
        fence_sha256 = _stack_template_sha256(
            self._client, stack_id=evidence.fence.stack_id
        )
        if fence_sha256 in {
            artifact.template_body_sha256,
            fence_transfer_sha256,
        }:
            state = _complete_mutation(
                store=self._state_store,
                state=state,
                key=key,
                request=execute_request,
            )
            return state, str(change_set_id)
        execution_status = described["ExecutionStatus"]
        if execution_status == "AVAILABLE":
            state, submit, _digest = _begin_mutation(
                store=self._state_store,
                state=state,
                key=key,
                request=execute_request,
            )
            if submit:
                try:
                    response = self._client.execute_change_set(**execute_request)
                except AmbiguousMigrationTransportError:
                    pass
                else:
                    _successful_response(response, "ExecuteChangeSet")
        elif execution_status == "EXECUTE_IN_PROGRESS":
            state, _submit, _digest = _begin_mutation(
                store=self._state_store,
                state=state,
                key=key,
                request=execute_request,
            )
        elif execution_status != "EXECUTE_COMPLETE":
            raise ValueError("import change set is not executable or complete")
        for _index in range(_STABILIZATION_READ_LIMIT):
            item = self._describe_import_change_set(
                evidence=evidence,
                authority=authority,
                artifact=artifact,
                expected_change_set_id=change_set_id,
            )
            if item is None:
                raise ValueError("submitted import change set disappeared")
            if item["ExecutionStatus"] in {"AVAILABLE", "EXECUTE_IN_PROGRESS"}:
                continue
            if item["ExecutionStatus"] != "EXECUTE_COMPLETE":
                raise ValueError("import change set execution failed")
            _stabilize_stack(
                self._client,
                lookup=evidence.fence.stack_id,
                identity=evidence.fence,
                role_arn=authority.deployment_role_arn,
                expected_template_sha256=(artifact.template_body_sha256,),
                transient_template_sha256=(bootstrap_sha256,),
                terminal_statuses=("IMPORT_COMPLETE",),
                progress_statuses=_IMPORT_PROGRESS,
                allow_absent=False,
            )
            state = _complete_mutation(
                store=self._state_store,
                state=state,
                key=key,
                request=execute_request,
            )
            return state, str(item["ChangeSetId"])
        raise ValueError("import change set execution stabilization exhausted")

    def bootstrap_v2(
        self, *, authority: "StackMigrationAuthorityV2"
    ) -> MigrationBootstrapResult:
        """Create/reconcile inert fence and support containers under MigrationServiceRole."""

        return self.bootstrap(authority=_legacy_authority_from_v2(authority))

    def _bridge_seed_request_v2(
        self,
        *,
        bundle: "StackMigrationTransferBundleV2",
        evidence: "StackMigrationEvidenceV2",
        authority: "StackMigrationAuthorityV2",
    ) -> Mapping[str, object]:
        artifacts = {artifact.stage: artifact for artifact in bundle.artifacts}
        if set(artifacts) != {
            "bridge-seed-owner",
            "retention-only",
            "post-retain",
            "fence-import",
            "fence-transfer",
        }:
            raise ValueError("bridge seed artifacts are not exact")
        bridge_artifact = artifacts["bridge-seed-owner"]
        return {
            "StackName": evidence.retained.stack_id,
            "TemplateURL": bridge_artifact.template_url,
            "RoleARN": authority.migration_service_role.role_arn,
            "ClientRequestToken": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "record_type": "glm52_h1g_bridge_seed_update_v2",
                        "action_identity_sha256": (
                            authority.action_identity_sha256
                        ),
                        "template_sha256": bridge_artifact.sha256,
                        "api_caller": authority.api_caller.to_dict(),
                        "service_role": (
                            authority.migration_service_role.to_dict()
                        ),
                    }
                )
            ).hexdigest(),
        }

    def _validate_bridge_inputs_v2(
        self,
        *,
        bundle: "StackMigrationTransferBundleV2",
        evidence: "StackMigrationEvidenceV2",
        authority: "StackMigrationAuthorityV2",
        bootstrap_result: MigrationBootstrapResult,
    ) -> MigrationExecutionState:
        if (
            type(bundle) is not StackMigrationTransferBundleV2
            or type(evidence) is not StackMigrationEvidenceV2
            or type(authority) is not StackMigrationAuthorityV2
            or type(bootstrap_result) is not MigrationBootstrapResult
        ):
            raise TypeError("bridge seed requires exact v2 migration inputs")
        if (
            bundle.manifest.get("record_type")
            != "glm52_h1g_stack_migration_transfer_manifest_v2"
            or evidence.fence != bootstrap_result.fence
            or evidence.support != bootstrap_result.support
            or bootstrap_result.action_identity_sha256
            != authority.action_identity_sha256
        ):
            raise ValueError("bridge seed bundle/bootstrap identity drifted")
        state = self._load_state(_legacy_authority_from_v2(authority))
        durable_bootstrap = migration_bootstrap_result_from_state(state)
        if (
            durable_bootstrap.fence != bootstrap_result.fence
            or durable_bootstrap.support != bootstrap_result.support
        ):
            raise ValueError("durable bootstrap StackIds drifted")
        return state

    def _read_bridge_seed_effect_v2(
        self,
        *,
        bundle: "StackMigrationTransferBundleV2",
        evidence: "StackMigrationEvidenceV2",
        authority: "StackMigrationAuthorityV2",
        bootstrap_result: MigrationBootstrapResult,
        require_journaled: bool,
    ) -> "BridgeSeedEstablishedV2":
        state = self._validate_bridge_inputs_v2(
            bundle=bundle,
            evidence=evidence,
            authority=authority,
            bootstrap_result=bootstrap_result,
        )
        request = self._bridge_seed_request_v2(
            bundle=bundle,
            evidence=evidence,
            authority=authority,
        )
        record = _state_record(state, "EstablishRetainedBridgeSeed")
        request_sha256 = _request_sha256(request)
        if require_journaled and (
            record.status is MigrationMutationStatus.NOT_SUBMITTED
            or record.request_sha256 != request_sha256
        ):
            raise ValueError("bridge seed mutation was not journaled exactly")
        original_sha256 = _get_template_sha256(
            self._client,
            request={
                "StackName": evidence.retained.stack_id,
                "TemplateStage": "Original",
            },
            operation="GetTemplateOriginal(BridgeSeed)",
        )
        processed_sha256 = _get_template_sha256(
            self._client,
            request={
                "StackName": evidence.retained.stack_id,
                "TemplateStage": "Processed",
            },
            operation="GetTemplateProcessed(BridgeSeed)",
        )
        if (
            original_sha256 != bundle.bridge_seed_plan.template_body_sha256
            or processed_sha256
            != bundle.bridge_seed_plan.template_body_sha256
        ):
            raise ValueError("retained bridge template readback is not exact")
        direct_sha256 = _direct_policy_sha256(
            self._client,
            bucket_name=evidence.bucket_name,
            expected_sha256=bundle.bridge_seed_plan.seed_policy_sha256,
        )
        return build_bridge_seed_established_v2(
            action_identity_sha256=authority.action_identity_sha256,
            retained=evidence.retained,
            fence=evidence.fence,
            support=evidence.support,
            bucket_name=evidence.bucket_name,
            plan=bundle.bridge_seed_plan,
            original_template_body_sha256=original_sha256,
            processed_template_body_sha256=processed_sha256,
            direct_policy_sha256=direct_sha256,
            api_caller=authority.api_caller,
            migration_service_role=authority.migration_service_role,
            fence_service_role=authority.fence_service_role,
        )

    def read_bridge_seed_effect_v2(
        self,
        *,
        bundle: "StackMigrationTransferBundleV2",
        evidence: "StackMigrationEvidenceV2",
        authority: "StackMigrationAuthorityV2",
        bootstrap_result: MigrationBootstrapResult,
    ) -> "BridgeSeedEstablishedV2":
        """Read one exact journaled bridge effect without mutation."""

        return self._read_bridge_seed_effect_v2(
            bundle=bundle,
            evidence=evidence,
            authority=authority,
            bootstrap_result=bootstrap_result,
            require_journaled=True,
        )

    def establish_bridge_seed_v2(
        self,
        *,
        bundle: "StackMigrationTransferBundleV2",
        evidence: "StackMigrationEvidenceV2",
        authority: "StackMigrationAuthorityV2",
        bootstrap_result: MigrationBootstrapResult,
    ) -> "BridgeSeedEstablishedV2":
        """Create or exactly adopt only the retained bridge-seed effect."""

        state = self._validate_bridge_inputs_v2(
            bundle=bundle,
            evidence=evidence,
            authority=authority,
            bootstrap_result=bootstrap_result,
        )
        request = self._bridge_seed_request_v2(
            bundle=bundle,
            evidence=evidence,
            authority=authority,
        )
        processed_sha256 = _stack_template_sha256(
            self._client, stack_id=evidence.retained.stack_id
        )
        plan = bundle.bridge_seed_plan
        if processed_sha256 not in {
            plan.preseed_template_body_sha256,
            plan.template_body_sha256,
        }:
            raise ValueError("retained stack is not an exact bridge predecessor")
        record = _state_record(state, "EstablishRetainedBridgeSeed")
        if (
            record.status is MigrationMutationStatus.COMPLETE
            and processed_sha256 != plan.template_body_sha256
        ):
            raise ValueError("completed bridge mutation effect was replaced")
        if (
            record.status is MigrationMutationStatus.SUBMITTED
            and processed_sha256 == plan.preseed_template_body_sha256
        ):
            raise ValueError(
                "possibly-sent bridge mutation is absent; resend is forbidden"
            )
        state, submit, _digest = _begin_mutation(
            store=self._state_store,
            state=state,
            key="EstablishRetainedBridgeSeed",
            request=request,
        )
        if submit and processed_sha256 == plan.preseed_template_body_sha256:
            try:
                response = self._client.update_stack(**request)
            except AmbiguousMigrationTransportError:
                pass
            else:
                _successful_response(response, "UpdateStack(BridgeSeed)")
        _stabilize_stack(
            self._client,
            lookup=evidence.retained.stack_id,
            identity=evidence.retained,
            role_arn=authority.migration_service_role.role_arn,
            expected_template_sha256=(plan.template_body_sha256,),
            transient_template_sha256=(plan.preseed_template_body_sha256,),
            terminal_statuses=("UPDATE_COMPLETE",),
            progress_statuses=_UPDATE_PROGRESS,
            allow_absent=False,
        )
        result = self._read_bridge_seed_effect_v2(
            bundle=bundle,
            evidence=evidence,
            authority=authority,
            bootstrap_result=bootstrap_result,
            require_journaled=True,
        )
        _complete_mutation(
            store=self._state_store,
            state=state,
            key="EstablishRetainedBridgeSeed",
            request=request,
        )
        return result

    def execute_operations_1_to_6_v2(
        self,
        *,
        bundle: "StackMigrationTransferBundleV2",
        evidence: "StackMigrationEvidenceV2",
        authority: "StackMigrationAuthorityV2",
        bootstrap_result: MigrationBootstrapResult,
        bridge_seed: "BridgeSeedEstablishedV2",
    ) -> "StackMigrationTransferCheckpointV2":
        """Execute only ownership transfer and stop before support operation 7."""

        if (
            type(bundle) is not StackMigrationTransferBundleV2
            or type(evidence) is not StackMigrationEvidenceV2
            or type(authority) is not StackMigrationAuthorityV2
            or type(bootstrap_result) is not MigrationBootstrapResult
            or type(bridge_seed) is not BridgeSeedEstablishedV2
        ):
            raise TypeError("operations 1-6 require exact v2 migration inputs")
        if (
            bundle.manifest.get("record_type")
            != "glm52_h1g_stack_migration_transfer_manifest_v2"
            or bundle.manifest.get("operation_7_status")
            != MigrationMutationStatus.NOT_SUBMITTED.value
            or evidence.fence != bootstrap_result.fence
            or evidence.support != bootstrap_result.support
            or bootstrap_result.action_identity_sha256
            != authority.action_identity_sha256
            or bridge_seed.action_identity_sha256
            != authority.action_identity_sha256
            or bridge_seed.retained_stack_id != evidence.retained.stack_id
            or bridge_seed.fence_stack_id != evidence.fence.stack_id
            or bridge_seed.support_stack_id != evidence.support.stack_id
            or bridge_seed.bucket_name != evidence.bucket_name
            or bridge_seed.owner_branch != bundle.bridge_seed_plan.owner_branch
            or bridge_seed.owner_logical_id
            != bundle.bridge_seed_plan.owner_logical_id
            or bridge_seed.bridge_template_body_sha256
            != bundle.bridge_seed_plan.template_body_sha256
            or bridge_seed.seed_policy_sha256
            != bundle.bridge_seed_plan.seed_policy_sha256
            or bridge_seed.direct_policy_sha256
            != bundle.bridge_seed_plan.seed_policy_sha256
            or bridge_seed.api_caller != authority.api_caller
            or bridge_seed.migration_service_role
            != authority.migration_service_role
            or bridge_seed.fence_service_role != authority.fence_service_role
            or bridge_seed.policy_committed is not True
            or bridge_seed.execution_eligible is not True
        ):
            raise ValueError("operations 1-6 bundle/bootstrap identity drifted")
        artifacts = {artifact.stage: artifact for artifact in bundle.artifacts}
        expected_stages = {
            "bridge-seed-owner",
            "retention-only",
            "post-retain",
            "fence-import",
            "fence-transfer",
        }
        if set(artifacts) != expected_stages:
            raise ValueError("operations 1-6 artifacts are not exact")
        legacy_authority = _legacy_authority_from_v2(authority)
        state = self._load_state(legacy_authority)
        state_bootstrap = migration_bootstrap_result_from_state(state)
        if (
            state_bootstrap.fence != bootstrap_result.fence
            or state_bootstrap.support != bootstrap_result.support
        ):
            raise ValueError("durable bootstrap StackIds drifted")
        if (
            _state_record(state, "EstablishRetainedBridgeSeed").status
            is not MigrationMutationStatus.COMPLETE
        ):
            raise ValueError("durable bridge seed is not committed")
        seed_policy = _resources(bundle.bridge_seed_plan.template)[
            bundle.bridge_seed_plan.owner_logical_id
        ]["Properties"]["PolicyDocument"]
        legacy_evidence = MigrationEvidence(
            retained=evidence.retained,
            fence=evidence.fence,
            support=evidence.support,
            current_policy_logical_id=(
                evidence.current_policy_logical_id
                or bundle.bridge_seed_plan.owner_logical_id
            ),
            current_policy_physical_id=evidence.bucket_name,
            current_policy_stack_id=evidence.retained.stack_id,
            bucket_name=evidence.bucket_name,
            import_identifier=evidence.import_identifier,
            live_policy=seed_policy,
            direct_policy_readbacks=(seed_policy, seed_policy, seed_policy),
            retained_deployment_role_id=authority.migration_service_role.role_id,
            retained_resource_physical_ids=(
                evidence.retained_resource_physical_ids
            ),
            retained_export_names=evidence.retained_export_names,
            support_export_names=evidence.support_export_names,
        )
        bridge_artifact = artifacts["bridge-seed-owner"]
        retained_current_sha256 = _stack_template_sha256(
            self._client, stack_id=evidence.retained.stack_id
        )
        if retained_current_sha256 not in {
            bridge_artifact.template_body_sha256,
            artifacts["retention-only"].template_body_sha256,
            artifacts["post-retain"].template_body_sha256,
        }:
            raise ValueError("retained stack is not an exact post-bridge stage")
        seed_sha256 = bundle.bridge_seed_plan.seed_policy_sha256
        first_direct = _direct_policy_sha256(
            self._client,
            bucket_name=evidence.bucket_name,
            expected_sha256=seed_sha256,
        )
        state = self._update(
            state=state,
            key="UpdateRetainedWithRetainAttributes",
            identity=evidence.retained,
            artifact=artifacts["retention-only"],
            authority=legacy_authority,
            accepted_target_sha256=(
                artifacts["retention-only"].template_body_sha256,
                artifacts["post-retain"].template_body_sha256,
            ),
            prior_template_sha256=(
                bridge_artifact.template_body_sha256,
            ),
        )
        state = self._update(
            state=state,
            key="UpdateRetainedRemovePolicy",
            identity=evidence.retained,
            artifact=artifacts["post-retain"],
            authority=legacy_authority,
            accepted_target_sha256=(
                artifacts["post-retain"].template_body_sha256,
            ),
            prior_template_sha256=(
                artifacts["retention-only"].template_body_sha256,
            ),
        )
        second_direct = _direct_policy_sha256(
            self._client,
            bucket_name=evidence.bucket_name,
            expected_sha256=seed_sha256,
        )
        fence_stage_sha256 = _stack_template_sha256(
            self._client, stack_id=evidence.fence.stack_id
        )
        if fence_stage_sha256 not in {
            bundle.manifest["bootstrap_template_body_sha256"],
            artifacts["fence-import"].template_body_sha256,
            artifacts["fence-transfer"].template_body_sha256,
        }:
            raise ValueError("fence stack is not an exact transfer stage")
        state, described = self._create_import_change_set(
            state=state,
            evidence=legacy_evidence,
            authority=legacy_authority,
            artifact=artifacts["fence-import"],
            fence_stage_sha256=fence_stage_sha256,
            fence_transfer_sha256=(
                artifacts["fence-transfer"].template_body_sha256
            ),
        )
        state, change_set_id = self._execute_import(
            state=state,
            evidence=legacy_evidence,
            authority=legacy_authority,
            artifact=artifacts["fence-import"],
            described=described,
            fence_transfer_sha256=(
                artifacts["fence-transfer"].template_body_sha256
            ),
            bootstrap_sha256=bundle.manifest[
                "bootstrap_template_body_sha256"
            ],
        )
        state = self._update(
            state=state,
            key="UpdateFenceRemoveAnchor",
            identity=evidence.fence,
            artifact=artifacts["fence-transfer"],
            authority=legacy_authority,
            accepted_target_sha256=(
                artifacts["fence-transfer"].template_body_sha256,
            ),
            prior_template_sha256=(
                artifacts["fence-import"].template_body_sha256,
            ),
        )
        records = {record.key: record for record in state.mutations}
        if (
            tuple(
                key
                for key in _MIGRATION_MUTATION_KEYS
                if key != "EstablishRetainedBridgeSeed"
                and records[key].status is MigrationMutationStatus.COMPLETE
            )
            != STACK_MIGRATION_TRANSFER_COMPLETED_KEYS_V2
            or records["UpdateSupportReplaceAnchor"].status
            is not MigrationMutationStatus.NOT_SUBMITTED
        ):
            raise ValueError("operation-6 durable checkpoint is not exact")
        original_sha256 = _get_template_sha256(
            self._client,
            request={
                "StackName": evidence.fence.stack_id,
                "TemplateStage": "Original",
            },
            operation="GetTemplateOriginal",
        )
        processed_sha256 = _get_template_sha256(
            self._client,
            request={
                "StackName": evidence.fence.stack_id,
                "TemplateStage": "Processed",
            },
            operation="GetTemplateProcessed",
        )
        support_prestate_sha256 = _stack_template_sha256(
            self._client, stack_id=evidence.support.stack_id
        )
        if (
            original_sha256
            != artifacts["fence-transfer"].template_body_sha256
            or processed_sha256
            != artifacts["fence-transfer"].template_body_sha256
            or support_prestate_sha256
            != bundle.manifest["bootstrap_template_body_sha256"]
            or _stack_policy_sha256_v2(
                self._client, stack_id=evidence.fence.stack_id
            )
            != FENCE_STACK_POLICY_SHA256_V2
        ):
            raise ValueError("operation-6 template/stack-policy readback drifted")
        final_direct = _direct_policy_sha256(
            self._client,
            bucket_name=evidence.bucket_name,
            expected_sha256=seed_sha256,
        )
        if (first_direct, second_direct, final_direct) != (
            seed_sha256,
            seed_sha256,
            seed_sha256,
        ):
            raise ValueError("bridge-seed direct policy equality drifted")
        return build_stack_migration_transfer_checkpoint_v2(
            bridge_seed_plan=bundle.bridge_seed_plan,
            retained=evidence.retained,
            fence=evidence.fence,
            support=evidence.support,
            bucket_name=evidence.bucket_name,
            fence_import_template_sha256=(
                artifacts["fence-import"].template_body_sha256
            ),
            fence_transfer_template_sha256=(
                artifacts["fence-transfer"].template_body_sha256
            ),
            active_original_template_sha256=original_sha256,
            active_processed_template_sha256=processed_sha256,
            direct_policy_sha256=(second_direct, final_direct),
            api_caller=authority.api_caller,
            migration_service_role=authority.migration_service_role,
            fence_service_role=authority.fence_service_role,
            associated_stack_role_arn=(
                authority.migration_service_role.role_arn
            ),
            associated_stack_role_id=(
                authority.migration_service_role.role_id
            ),
            import_change_set_id=change_set_id,
            support_prestate_template_sha256=support_prestate_sha256,
        )

    def read_operations_1_to_6_effect_v2(
        self,
        *,
        bundle: "StackMigrationTransferBundleV2",
        evidence: "StackMigrationEvidenceV2",
        authority: "StackMigrationAuthorityV2",
        bootstrap_result: MigrationBootstrapResult,
        bridge_seed: "BridgeSeedEstablishedV2",
    ) -> "StackMigrationTransferCheckpointV2":
        """Read the exact operation-6 checkpoint without mutation."""

        state = self._validate_bridge_inputs_v2(
            bundle=bundle,
            evidence=evidence,
            authority=authority,
            bootstrap_result=bootstrap_result,
        )
        if (
            type(bridge_seed) is not BridgeSeedEstablishedV2
            or bridge_seed.action_identity_sha256
            != authority.action_identity_sha256
            or bridge_seed.canonical_identity_sha256
            != parse_bridge_seed_established_v2(
                bridge_seed.to_dict()
            ).canonical_identity_sha256
        ):
            raise ValueError("bridge seed prerequisite is not exact")
        records = {record.key: record for record in state.mutations}
        if (
            tuple(
                key
                for key in _MIGRATION_MUTATION_KEYS
                if key != "EstablishRetainedBridgeSeed"
                and records[key].status is MigrationMutationStatus.COMPLETE
            )
            != STACK_MIGRATION_TRANSFER_COMPLETED_KEYS_V2
            or records["UpdateSupportReplaceAnchor"].status
            is not MigrationMutationStatus.NOT_SUBMITTED
        ):
            raise ValueError("operation-6 durable state is not exact")
        change_set_id = records[
            "ImportPolicyIntoFenceStack.CreateChangeSet"
        ].effect_identity
        if type(change_set_id) is not str:
            raise ValueError("operation-6 import identity is absent")
        artifacts = {artifact.stage: artifact for artifact in bundle.artifacts}
        if set(artifacts) != {
            "bridge-seed-owner",
            "retention-only",
            "post-retain",
            "fence-import",
            "fence-transfer",
        }:
            raise ValueError("operation-6 artifacts are not exact")
        if (
            _stack_template_sha256(
                self._client, stack_id=evidence.retained.stack_id
            )
            != artifacts["post-retain"].template_body_sha256
        ):
            raise ValueError("retained post-transfer template drifted")
        original_sha256 = _get_template_sha256(
            self._client,
            request={
                "StackName": evidence.fence.stack_id,
                "TemplateStage": "Original",
            },
            operation="GetTemplateOriginal(Operation6Read)",
        )
        processed_sha256 = _get_template_sha256(
            self._client,
            request={
                "StackName": evidence.fence.stack_id,
                "TemplateStage": "Processed",
            },
            operation="GetTemplateProcessed(Operation6Read)",
        )
        support_prestate_sha256 = _stack_template_sha256(
            self._client, stack_id=evidence.support.stack_id
        )
        if (
            original_sha256
            != artifacts["fence-transfer"].template_body_sha256
            or processed_sha256
            != artifacts["fence-transfer"].template_body_sha256
            or support_prestate_sha256
            != bundle.manifest["bootstrap_template_body_sha256"]
            or _stack_policy_sha256_v2(
                self._client, stack_id=evidence.fence.stack_id
            )
            != FENCE_STACK_POLICY_SHA256_V2
        ):
            raise ValueError("operation-6 readback drifted")
        direct_sha256 = _direct_policy_sha256(
            self._client,
            bucket_name=evidence.bucket_name,
            expected_sha256=bundle.bridge_seed_plan.seed_policy_sha256,
        )
        return build_stack_migration_transfer_checkpoint_v2(
            bridge_seed_plan=bundle.bridge_seed_plan,
            retained=evidence.retained,
            fence=evidence.fence,
            support=evidence.support,
            bucket_name=evidence.bucket_name,
            fence_import_template_sha256=(
                artifacts["fence-import"].template_body_sha256
            ),
            fence_transfer_template_sha256=(
                artifacts["fence-transfer"].template_body_sha256
            ),
            active_original_template_sha256=original_sha256,
            active_processed_template_sha256=processed_sha256,
            direct_policy_sha256=(direct_sha256, direct_sha256),
            api_caller=authority.api_caller,
            migration_service_role=authority.migration_service_role,
            fence_service_role=authority.fence_service_role,
            associated_stack_role_arn=(
                authority.migration_service_role.role_arn
            ),
            associated_stack_role_id=(
                authority.migration_service_role.role_id
            ),
            import_change_set_id=change_set_id,
            support_prestate_template_sha256=support_prestate_sha256,
        )

    def _support_anchor_request_v2(
        self,
        *,
        authority: MigrationExecutionAuthority,
        artifact: TemplateArtifact,
        support: StackIdentity,
    ) -> Mapping[str, object]:
        if artifact.stack_id != support.stack_id:
            raise ValueError("support replacement artifact targets another stack")
        return {
            "StackName": support.stack_id,
            "TemplateURL": artifact.template_url,
            "RoleARN": authority.deployment_role_arn,
            "ClientRequestToken": _migration_token(
                authority=authority,
                operation=f"update-{artifact.stage}",
                identity=artifact.sha256,
            ),
        }

    def read_support_anchor_replacement_effect_v2(
        self,
        *,
        checkpoint: "StackMigrationTransferCheckpointV2",
        authority: MigrationExecutionAuthority,
        artifact: TemplateArtifact,
        expected_direct_policy_sha256: str,
    ) -> Tuple[str, str]:
        """Read the exact operation-7 support effect without mutation."""

        if (
            type(checkpoint) is not StackMigrationTransferCheckpointV2
            or type(authority) is not MigrationExecutionAuthority
            or type(artifact) is not TemplateArtifact
            or authority.deployment_role_arn
            != checkpoint.migration_service_role.role_arn
            or authority.deployment_role_id
            != checkpoint.migration_service_role.role_id
            or artifact.stage != "disabled-support"
            or artifact.stack_id != checkpoint.support_stack_id
            or artifact.template_body_sha256
            == checkpoint.support_prestate_template_sha256
        ):
            raise ValueError("support replacement read contract is not exact")
        state = self._load_state(authority)
        request = self._support_anchor_request_v2(
            authority=authority,
            artifact=artifact,
            support=StackIdentity(
                kind=StackKind.SUPPORT,
                name=stack_name(StackKind.SUPPORT),
                stack_id=checkpoint.support_stack_id,
                account_id=ACCOUNT_ID,
                region=REGION,
                termination_protection=True,
                tags=STACK_TAGS,
            ),
        )
        record = _state_record(state, "UpdateSupportReplaceAnchor")
        if (
            record.status is MigrationMutationStatus.NOT_SUBMITTED
            or record.request_sha256 != _request_sha256(request)
        ):
            raise ValueError("support replacement is not exactly journaled")
        support = StackIdentity(
            kind=StackKind.SUPPORT,
            name=stack_name(StackKind.SUPPORT),
            stack_id=checkpoint.support_stack_id,
            account_id=ACCOUNT_ID,
            region=REGION,
            termination_protection=True,
            tags=STACK_TAGS,
        )
        stack = _describe_exact_stack(
            self._client,
            lookup=support.stack_id,
            identity=support,
            role_arn=authority.deployment_role_arn,
        )
        if stack is None or stack["StackStatus"] != "UPDATE_COMPLETE":
            raise ValueError("support replacement did not reach exact completion")
        original_sha256 = _get_template_sha256(
            self._client,
            request={
                "StackName": support.stack_id,
                "TemplateStage": "Original",
            },
            operation="GetTemplateOriginal(SupportReplacement)",
        )
        processed_sha256 = _get_template_sha256(
            self._client,
            request={
                "StackName": support.stack_id,
                "TemplateStage": "Processed",
            },
            operation="GetTemplateProcessed(SupportReplacement)",
        )
        if (
            original_sha256 != artifact.template_body_sha256
            or processed_sha256 != artifact.template_body_sha256
        ):
            raise ValueError("support replacement template readback drifted")
        _direct_policy_sha256(
            self._client,
            bucket_name=checkpoint.bucket_name,
            expected_sha256=expected_direct_policy_sha256,
        )
        return (
            checkpoint.support_prestate_template_sha256,
            processed_sha256,
        )

    def execute_support_anchor_replacement_v2(
        self,
        *,
        checkpoint: "StackMigrationTransferCheckpointV2",
        authority: MigrationExecutionAuthority,
        artifact: TemplateArtifact,
        expected_direct_policy_sha256: str,
    ) -> Tuple[str, str]:
        """Journal and perform only the approved support-anchor replacement."""

        if (
            type(checkpoint) is not StackMigrationTransferCheckpointV2
            or checkpoint.operation_7_status
            != MigrationMutationStatus.NOT_SUBMITTED.value
            or type(authority) is not MigrationExecutionAuthority
            or type(artifact) is not TemplateArtifact
        ):
            raise TypeError("support replacement requires exact typed inputs")
        state = self._load_state(authority)
        records = {record.key: record for record in state.mutations}
        if (
            tuple(
                key
                for key in _MIGRATION_MUTATION_KEYS
                if key != "EstablishRetainedBridgeSeed"
                and records[key].status is MigrationMutationStatus.COMPLETE
            )
            != STACK_MIGRATION_TRANSFER_COMPLETED_KEYS_V2
            or records["UpdateSupportReplaceAnchor"].status
            is MigrationMutationStatus.COMPLETE
        ):
            raise ValueError("support replacement predecessor state is not exact")
        _direct_policy_sha256(
            self._client,
            bucket_name=checkpoint.bucket_name,
            expected_sha256=expected_direct_policy_sha256,
        )
        support = StackIdentity(
            kind=StackKind.SUPPORT,
            name=stack_name(StackKind.SUPPORT),
            stack_id=checkpoint.support_stack_id,
            account_id=ACCOUNT_ID,
            region=REGION,
            termination_protection=True,
            tags=STACK_TAGS,
        )
        self._update(
            state=state,
            key="UpdateSupportReplaceAnchor",
            identity=support,
            artifact=artifact,
            authority=authority,
            accepted_target_sha256=(artifact.template_body_sha256,),
            prior_template_sha256=(
                checkpoint.support_prestate_template_sha256,
            ),
        )
        return self.read_support_anchor_replacement_effect_v2(
            checkpoint=checkpoint,
            authority=authority,
            artifact=artifact,
            expected_direct_policy_sha256=expected_direct_policy_sha256,
        )

    def execute(
        self,
        *,
        bundle: MigrationBundle,
        evidence: MigrationEvidence,
        authority: MigrationExecutionAuthority,
        bootstrap_result: MigrationBootstrapResult,
    ) -> MigrationExecutionResult:
        raise ValueError(
            "stack-migration v1 inputs are audit-only and execution-ineligible"
        )


STACK_MIGRATION_TRANSFER_CHECKPOINT_RECORD_TYPE_V2 = (
    "STACK_MIGRATION_TRANSFER_CHECKPOINT_V2"
)
STACK_MIGRATION_TRANSFER_CHECKPOINT_V2 = (
    STACK_MIGRATION_TRANSFER_CHECKPOINT_RECORD_TYPE_V2
)
STACK_MIGRATION_TRANSFER_COMPLETED_KEYS_V2: Tuple[str, ...] = (
    "CreateBootstrapFenceStack",
    "CreateBootstrapSupportStack",
    "UpdateRetainedWithRetainAttributes",
    "UpdateRetainedRemovePolicy",
    "ImportPolicyIntoFenceStack.CreateChangeSet",
    "ImportPolicyIntoFenceStack.ExecuteChangeSet",
    "UpdateFenceRemoveAnchor",
)


@dataclass(frozen=True)
class MigrationRoleEvidenceV2:
    """One exact live IAM role identity used by the split migration."""

    role_arn: str
    role_id: str
    trust_policy_sha256: str
    permission_policy_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.role_arn) is not str
            or re.fullmatch(
                rf"arn:aws:iam::{ACCOUNT_ID}:role/"
                r"[A-Za-z0-9+=,.@_-]+(?:/[A-Za-z0-9+=,.@_-]+)*",
                self.role_arn,
            )
            is None
            or type(self.role_id) is not str
            or _ROLE_ID.fullmatch(self.role_id) is None
            or type(self.trust_policy_sha256) is not str
            or _SHA256.fullmatch(self.trust_policy_sha256) is None
            or type(self.permission_policy_sha256) is not str
            or _SHA256.fullmatch(self.permission_policy_sha256) is None
        ):
            raise ValueError("migration role evidence v2 is not exact")

    def to_dict(self) -> Dict[str, str]:
        return {
            "role_arn": self.role_arn,
            "role_id": self.role_id,
            "trust_policy_sha256": self.trust_policy_sha256,
            "permission_policy_sha256": self.permission_policy_sha256,
        }


def parse_migration_role_evidence_v2(
    value: object,
) -> MigrationRoleEvidenceV2:
    if type(value) is not dict or set(value) != {
        "role_arn",
        "role_id",
        "trust_policy_sha256",
        "permission_policy_sha256",
    }:
        raise ValueError("migration role evidence schema is not exact")
    return MigrationRoleEvidenceV2(
        role_arn=value["role_arn"],
        role_id=value["role_id"],
        trust_policy_sha256=value["trust_policy_sha256"],
        permission_policy_sha256=value["permission_policy_sha256"],
    )


@dataclass(frozen=True)
class BridgeSeedOwnershipPlanV2:
    """Immutable retained-owner Add/Modify plan for the canonical seed."""

    owner_branch: str
    owner_logical_id: str
    template_body: bytes
    preseed_template_body_sha256: str
    expected_live_preseed_policy_sha256: str
    template_body_sha256: str
    seed_policy_sha256: str
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        if (
            self.owner_branch not in {"ADD", "MODIFY"}
            or type(self.owner_logical_id) is not str
            or _LOGICAL_ID.fullmatch(self.owner_logical_id) is None
            or type(self.template_body) is not bytes
            or not self.template_body
            or hashlib.sha256(self.template_body).hexdigest()
            != self.template_body_sha256
            or _SHA256.fullmatch(self.preseed_template_body_sha256) is None
            or _SHA256.fullmatch(
                self.expected_live_preseed_policy_sha256
            )
            is None
            or _SHA256.fullmatch(self.seed_policy_sha256) is None
        ):
            raise ValueError("bridge-seed ownership plan v2 is not exact")
        projection = {
            "record_type": "glm52_h1g_bridge_seed_ownership_plan_v2",
            "owner_branch": self.owner_branch,
            "owner_logical_id": self.owner_logical_id,
            "preseed_template_body_sha256": self.preseed_template_body_sha256,
            "expected_live_preseed_policy_sha256": (
                self.expected_live_preseed_policy_sha256
            ),
            "template_body_sha256": self.template_body_sha256,
            "seed_policy_sha256": self.seed_policy_sha256,
        }
        if (
            hashlib.sha256(canonical_json_bytes(projection)).hexdigest()
            != self.canonical_identity_sha256
        ):
            raise ValueError("bridge-seed ownership plan identity drifted")

    @property
    def template(self) -> Mapping[str, object]:
        value = json.loads(self.template_body)
        if type(value) is not dict:
            raise AssertionError("validated bridge-seed template is not an object")
        return value

BRIDGE_SEED_ESTABLISHED_RECORD_TYPE_V2 = (
    "glm52_h1g_bridge_seed_established_v2"
)


@dataclass(frozen=True)
class BridgeSeedEstablishedV2:
    """Exact retained-stack bridge effect after canonical live readback."""

    record_type: str
    action_identity_sha256: str
    retained_stack_id: str
    fence_stack_id: str
    support_stack_id: str
    bucket_name: str
    owner_branch: str
    owner_logical_id: str
    preseed_template_body_sha256: str
    bridge_template_body_sha256: str
    seed_policy_sha256: str
    original_template_body_sha256: str
    processed_template_body_sha256: str
    direct_policy_sha256: str
    api_caller: MigrationRoleEvidenceV2
    migration_service_role: MigrationRoleEvidenceV2
    fence_service_role: MigrationRoleEvidenceV2
    policy_committed: bool
    execution_eligible: bool
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        for role in (
            self.api_caller,
            self.migration_service_role,
            self.fence_service_role,
        ):
            if type(role) is not MigrationRoleEvidenceV2:
                raise TypeError("bridge result role evidence must be exact v2")
        if (
            self.record_type != BRIDGE_SEED_ESTABLISHED_RECORD_TYPE_V2
            or self.owner_branch not in {"ADD", "MODIFY"}
            or _LOGICAL_ID.fullmatch(self.owner_logical_id) is None
            or self.policy_committed is not True
            or self.execution_eligible is not True
            or self.original_template_body_sha256
            != self.bridge_template_body_sha256
            or self.processed_template_body_sha256
            != self.bridge_template_body_sha256
            or self.direct_policy_sha256 != self.seed_policy_sha256
            or self.migration_service_role.role_arn
            != MIGRATION_SERVICE_ROLE_ARN_V2
            or self.fence_service_role.role_arn
            != FENCE_SERVICE_ROLE_ARN_V2
        ):
            raise ValueError("bridge seed established result is not exact")
        for kind, stack_id in (
            (StackKind.RETAINED, self.retained_stack_id),
            (StackKind.FENCE, self.fence_stack_id),
            (StackKind.SUPPORT, self.support_stack_id),
        ):
            StackIdentity(
                kind=kind,
                name=stack_name(kind),
                stack_id=stack_id,
                account_id=ACCOUNT_ID,
                region=REGION,
                termination_protection=True,
                tags=STACK_TAGS,
            )
        for value in (
            self.action_identity_sha256,
            self.preseed_template_body_sha256,
            self.bridge_template_body_sha256,
            self.seed_policy_sha256,
            self.original_template_body_sha256,
            self.processed_template_body_sha256,
            self.direct_policy_sha256,
        ):
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise ValueError("bridge result contains a non-SHA-256 identity")
        _validate_bucket_name(self.bucket_name)
        if self.canonical_identity_sha256 != hashlib.sha256(
            canonical_json_bytes(
                _bridge_seed_established_projection(
                    self, include_identity=False
                )
            )
        ).hexdigest():
            raise ValueError("bridge seed established identity drifted")

    def to_dict(self) -> Mapping[str, object]:
        return _bridge_seed_established_projection(self, include_identity=True)


def _bridge_seed_established_projection(
    value: BridgeSeedEstablishedV2, *, include_identity: bool
) -> Dict[str, object]:
    projected: Dict[str, object] = {
        "record_type": value.record_type,
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "action_identity_sha256": value.action_identity_sha256,
        "retained_stack_id": value.retained_stack_id,
        "fence_stack_id": value.fence_stack_id,
        "support_stack_id": value.support_stack_id,
        "bucket_name": value.bucket_name,
        "owner_branch": value.owner_branch,
        "owner_logical_id": value.owner_logical_id,
        "preseed_template_body_sha256": value.preseed_template_body_sha256,
        "bridge_template_body_sha256": value.bridge_template_body_sha256,
        "seed_policy_sha256": value.seed_policy_sha256,
        "original_template_body_sha256": value.original_template_body_sha256,
        "processed_template_body_sha256": value.processed_template_body_sha256,
        "direct_policy_sha256": value.direct_policy_sha256,
        "api_caller": value.api_caller.to_dict(),
        "migration_service_role": value.migration_service_role.to_dict(),
        "fence_service_role": value.fence_service_role.to_dict(),
        "policy_committed": value.policy_committed,
        "execution_eligible": value.execution_eligible,
    }
    if include_identity:
        projected["canonical_identity_sha256"] = (
            value.canonical_identity_sha256
        )
    return projected


def _bridge_seed_established_identity_from_fields(
    fields: Mapping[str, object],
) -> str:
    projected = {
        "record_type": fields["record_type"],
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "action_identity_sha256": fields["action_identity_sha256"],
        "retained_stack_id": fields["retained_stack_id"],
        "fence_stack_id": fields["fence_stack_id"],
        "support_stack_id": fields["support_stack_id"],
        "bucket_name": fields["bucket_name"],
        "owner_branch": fields["owner_branch"],
        "owner_logical_id": fields["owner_logical_id"],
        "preseed_template_body_sha256": fields[
            "preseed_template_body_sha256"
        ],
        "bridge_template_body_sha256": fields[
            "bridge_template_body_sha256"
        ],
        "seed_policy_sha256": fields["seed_policy_sha256"],
        "original_template_body_sha256": fields[
            "original_template_body_sha256"
        ],
        "processed_template_body_sha256": fields[
            "processed_template_body_sha256"
        ],
        "direct_policy_sha256": fields["direct_policy_sha256"],
        "api_caller": fields["api_caller"].to_dict(),
        "migration_service_role": fields[
            "migration_service_role"
        ].to_dict(),
        "fence_service_role": fields["fence_service_role"].to_dict(),
        "policy_committed": fields["policy_committed"],
        "execution_eligible": fields["execution_eligible"],
    }
    return hashlib.sha256(canonical_json_bytes(projected)).hexdigest()


def build_bridge_seed_established_v2(
    *,
    action_identity_sha256: str,
    retained: StackIdentity,
    fence: StackIdentity,
    support: StackIdentity,
    bucket_name: str,
    plan: BridgeSeedOwnershipPlanV2,
    original_template_body_sha256: str,
    processed_template_body_sha256: str,
    direct_policy_sha256: str,
    api_caller: MigrationRoleEvidenceV2,
    migration_service_role: MigrationRoleEvidenceV2,
    fence_service_role: MigrationRoleEvidenceV2,
) -> BridgeSeedEstablishedV2:
    if (
        type(plan) is not BridgeSeedOwnershipPlanV2
        or type(retained) is not StackIdentity
        or retained.kind is not StackKind.RETAINED
        or type(fence) is not StackIdentity
        or fence.kind is not StackKind.FENCE
        or type(support) is not StackIdentity
        or support.kind is not StackKind.SUPPORT
    ):
        raise TypeError("bridge result requires exact migration identities")
    fields = {
        "record_type": BRIDGE_SEED_ESTABLISHED_RECORD_TYPE_V2,
        "action_identity_sha256": action_identity_sha256,
        "retained_stack_id": retained.stack_id,
        "fence_stack_id": fence.stack_id,
        "support_stack_id": support.stack_id,
        "bucket_name": bucket_name,
        "owner_branch": plan.owner_branch,
        "owner_logical_id": plan.owner_logical_id,
        "preseed_template_body_sha256": plan.preseed_template_body_sha256,
        "bridge_template_body_sha256": plan.template_body_sha256,
        "seed_policy_sha256": plan.seed_policy_sha256,
        "original_template_body_sha256": original_template_body_sha256,
        "processed_template_body_sha256": processed_template_body_sha256,
        "direct_policy_sha256": direct_policy_sha256,
        "api_caller": api_caller,
        "migration_service_role": migration_service_role,
        "fence_service_role": fence_service_role,
        "policy_committed": True,
        "execution_eligible": True,
    }
    return BridgeSeedEstablishedV2(
        **fields,
        canonical_identity_sha256=(
            _bridge_seed_established_identity_from_fields(fields)
        ),
    )


def parse_bridge_seed_established_v2(
    value: object,
) -> BridgeSeedEstablishedV2:
    expected = {
        "record_type",
        "account_id",
        "region",
        "run_id",
        "action_identity_sha256",
        "retained_stack_id",
        "fence_stack_id",
        "support_stack_id",
        "bucket_name",
        "owner_branch",
        "owner_logical_id",
        "preseed_template_body_sha256",
        "bridge_template_body_sha256",
        "seed_policy_sha256",
        "original_template_body_sha256",
        "processed_template_body_sha256",
        "direct_policy_sha256",
        "api_caller",
        "migration_service_role",
        "fence_service_role",
        "policy_committed",
        "execution_eligible",
        "canonical_identity_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("bridge seed established schema is not exact")
    if (
        value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["run_id"] != RUN_ID
    ):
        raise ValueError("bridge seed established scope is foreign")
    return BridgeSeedEstablishedV2(
        record_type=value["record_type"],
        action_identity_sha256=value["action_identity_sha256"],
        retained_stack_id=value["retained_stack_id"],
        fence_stack_id=value["fence_stack_id"],
        support_stack_id=value["support_stack_id"],
        bucket_name=value["bucket_name"],
        owner_branch=value["owner_branch"],
        owner_logical_id=value["owner_logical_id"],
        preseed_template_body_sha256=value[
            "preseed_template_body_sha256"
        ],
        bridge_template_body_sha256=value[
            "bridge_template_body_sha256"
        ],
        seed_policy_sha256=value["seed_policy_sha256"],
        original_template_body_sha256=value[
            "original_template_body_sha256"
        ],
        processed_template_body_sha256=value[
            "processed_template_body_sha256"
        ],
        direct_policy_sha256=value["direct_policy_sha256"],
        api_caller=parse_migration_role_evidence_v2(value["api_caller"]),
        migration_service_role=parse_migration_role_evidence_v2(
            value["migration_service_role"]
        ),
        fence_service_role=parse_migration_role_evidence_v2(
            value["fence_service_role"]
        ),
        policy_committed=value["policy_committed"],
        execution_eligible=value["execution_eligible"],
        canonical_identity_sha256=value["canonical_identity_sha256"],
    )


def _exact_bridge_seed_policy(bridge_seed: object) -> Mapping[str, object]:
    from .fence_artifacts import (
        BridgeSeedArtifact,
        parse_bridge_seed_artifact,
    )

    if type(bridge_seed) is not BridgeSeedArtifact:
        raise TypeError("bridge seed must be exact BridgeSeedArtifact")
    authenticated = parse_bridge_seed_artifact(
        bridge_seed.to_dict(), raw_bytes=bridge_seed.raw_bytes
    )
    if not authenticated.raw_bytes:
        raise ValueError("bridge seed raw policy bytes are required")
    try:
        policy = json.loads(
            authenticated.raw_bytes,
            object_pairs_hook=_exact_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("bridge seed policy is not strict JSON") from exc
    _validate_literal_policy(policy)
    if canonical_json_bytes(policy) != authenticated.raw_bytes:
        raise ValueError("bridge seed policy bytes are not canonical")
    if (
        hashlib.sha256(authenticated.raw_bytes).hexdigest()
        != authenticated.policy_sha256
    ):
        raise ValueError("bridge seed policy hash drifted")
    return policy


def build_bridge_seed_ownership_plan_v2(
    *,
    retained_template: Mapping[str, object],
    bucket_name: str,
    bridge_seed: object,
    current_policy_logical_id: Optional[str],
) -> BridgeSeedOwnershipPlanV2:
    """Build the sole retained-owner Add/Modify transaction before transfer."""

    _validate_bucket_name(bucket_name)
    resources = _resources(retained_template)
    policy = _exact_bridge_seed_policy(bridge_seed)
    seed_projection = bridge_seed.to_dict()
    existing = tuple(
        logical_id
        for logical_id, resource in resources.items()
        if type(resource) is dict
        and resource.get("Type") == "AWS::S3::BucketPolicy"
    )
    if current_policy_logical_id is None:
        if existing:
            raise ValueError("bridge-seed ADD branch found an existing policy owner")
        owner_branch = "ADD"
        owner_logical_id = FENCE_LOGICAL_ID
    else:
        if (
            type(current_policy_logical_id) is not str
            or _LOGICAL_ID.fullmatch(current_policy_logical_id) is None
            or existing != (current_policy_logical_id,)
        ):
            raise ValueError("bridge-seed MODIFY owner is not singular and exact")
        owner_branch = "MODIFY"
        owner_logical_id = current_policy_logical_id
    result = deepcopy(retained_template)
    result_resources = _resources(result)
    result_resources[owner_logical_id] = {
        "Type": "AWS::S3::BucketPolicy",
        "Properties": {
            "Bucket": bucket_name,
            "PolicyDocument": deepcopy(policy),
        },
    }
    template_body = canonical_json_bytes(result)
    seed_sha256 = hashlib.sha256(canonical_json_bytes(policy)).hexdigest()
    identity_projection = {
        "record_type": "glm52_h1g_bridge_seed_ownership_plan_v2",
        "owner_branch": owner_branch,
        "owner_logical_id": owner_logical_id,
        "preseed_template_body_sha256": hashlib.sha256(
            canonical_json_bytes(retained_template)
        ).hexdigest(),
        "expected_live_preseed_policy_sha256": seed_projection[
            "expected_live_preseed_policy_sha256"
        ],
        "template_body_sha256": hashlib.sha256(template_body).hexdigest(),
        "seed_policy_sha256": seed_sha256,
    }
    return BridgeSeedOwnershipPlanV2(
        owner_branch=owner_branch,
        owner_logical_id=owner_logical_id,
        template_body=template_body,
        preseed_template_body_sha256=identity_projection[
            "preseed_template_body_sha256"
        ],
        expected_live_preseed_policy_sha256=identity_projection[
            "expected_live_preseed_policy_sha256"
        ],
        template_body_sha256=identity_projection["template_body_sha256"],
        seed_policy_sha256=seed_sha256,
        canonical_identity_sha256=hashlib.sha256(
            canonical_json_bytes(identity_projection)
        ).hexdigest(),
    )


def build_fence_transfer_template_v2(
    *, fence_import_template: Mapping[str, object]
) -> Dict[str, object]:
    """Remove only the inert anchor while preserving exact seed ownership."""

    if type(fence_import_template) is not dict or set(fence_import_template) != {
        "AWSTemplateFormatVersion",
        "Resources",
    }:
        raise ValueError("fence import template is not exact")
    resources = _resources(fence_import_template)
    if set(resources) != {CONTAINER_ANCHOR, FENCE_LOGICAL_ID}:
        raise ValueError("fence import template has foreign resources")
    if resources[CONTAINER_ANCHOR] != {
        "Type": "AWS::CloudFormation::WaitConditionHandle"
    }:
        raise ValueError("fence import anchor is not exact")
    policy = resources[FENCE_LOGICAL_ID]
    if (
        type(policy) is not dict
        or policy.get("Type") != "AWS::S3::BucketPolicy"
        or policy.get("DeletionPolicy") != "Retain"
        or policy.get("UpdateReplacePolicy") != "Retain"
        or "Condition" in policy
        or type(policy.get("Properties")) is not dict
        or set(policy["Properties"]) != {"Bucket", "PolicyDocument"}
    ):
        raise ValueError("fence import owner is not exact and unconditional")
    _validate_bucket_name(policy["Properties"]["Bucket"])
    _validate_literal_policy(policy["Properties"]["PolicyDocument"])
    result = deepcopy(fence_import_template)
    del _resources(result)[CONTAINER_ANCHOR]
    return result


def fence_stack_policy_v2() -> Mapping[str, object]:
    """Return the immutable fence-stack policy used throughout migration."""

    return {
        "Statement": [
            {
                "Effect": "Deny",
                "Action": ["Update:Replace", "Update:Delete"],
                "Principal": "*",
                "Resource": f"LogicalResourceId/{FENCE_LOGICAL_ID}",
            },
            {
                "Effect": "Allow",
                "Action": "Update:Modify",
                "Principal": "*",
                "Resource": f"LogicalResourceId/{FENCE_LOGICAL_ID}",
            },
        ]
    }


FENCE_STACK_POLICY_SHA256_V2 = hashlib.sha256(
    canonical_json_bytes(fence_stack_policy_v2())
).hexdigest()


@dataclass(frozen=True)
class StackMigrationTransferCheckpointV2:
    """Immutable close of operations 1-6 before PREPARE and support op7."""

    record_type: str
    retained_stack_id: str
    fence_stack_id: str
    support_stack_id: str
    bucket_name: str
    retained_owner_logical_id: str
    retained_owner_physical_id: str
    bridge_seed_owner_branch: str
    bridge_seed_policy_sha256: str
    expected_live_preseed_policy_sha256: str
    bridge_seed_owner_template_sha256: str
    fence_import_template_sha256: str
    fence_transfer_template_sha256: str
    active_original_template_sha256: str
    active_processed_template_sha256: str
    direct_policy_sha256: Tuple[str, str]
    stack_policy_sha256: str
    api_caller: MigrationRoleEvidenceV2
    migration_service_role: MigrationRoleEvidenceV2
    fence_service_role: MigrationRoleEvidenceV2
    associated_stack_role_arn: str
    associated_stack_role_id: str
    rollback_stack_role_arn: str
    import_change_set_id: str
    completed_mutation_keys: Tuple[str, ...]
    operation_7_status: str
    support_state: str
    support_prestate_template_sha256: str
    deletion_policy: str
    update_replace_policy: str
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        for identity in (
            self.api_caller,
            self.migration_service_role,
            self.fence_service_role,
        ):
            if type(identity) is not MigrationRoleEvidenceV2:
                raise TypeError("checkpoint role evidence must be exact v2")
        stack_ids = (
            self.retained_stack_id,
            self.fence_stack_id,
            self.support_stack_id,
        )
        if (
            self.record_type != STACK_MIGRATION_TRANSFER_CHECKPOINT_RECORD_TYPE_V2
            or len(set(stack_ids)) != 3
            or self.bridge_seed_owner_branch not in {"ADD", "MODIFY"}
            or self.completed_mutation_keys
            != STACK_MIGRATION_TRANSFER_COMPLETED_KEYS_V2
            or self.operation_7_status != MigrationMutationStatus.NOT_SUBMITTED.value
            or self.support_state != "INERT_ANCHOR"
            or self.deletion_policy != "Retain"
            or self.update_replace_policy != "Retain"
            or self.stack_policy_sha256 != FENCE_STACK_POLICY_SHA256_V2
            or self.active_original_template_sha256
            != self.fence_transfer_template_sha256
            or self.active_processed_template_sha256
            != self.fence_transfer_template_sha256
            or self.direct_policy_sha256
            != (
                self.bridge_seed_policy_sha256,
                self.bridge_seed_policy_sha256,
            )
            or self.associated_stack_role_arn
            != self.migration_service_role.role_arn
            or self.associated_stack_role_id
            != self.migration_service_role.role_id
            or self.rollback_stack_role_arn
            != self.migration_service_role.role_arn
            or self.migration_service_role.role_arn
            != MIGRATION_SERVICE_ROLE_ARN_V2
            or self.fence_service_role.role_arn
            != FENCE_SERVICE_ROLE_ARN_V2
        ):
            raise ValueError("stack migration transfer checkpoint v2 is not exact")
        if (
            len(
                {
                    self.api_caller.role_arn,
                    self.migration_service_role.role_arn,
                    self.fence_service_role.role_arn,
                }
            )
            != 3
            or len(
                {
                    self.api_caller.role_id,
                    self.migration_service_role.role_id,
                    self.fence_service_role.role_id,
                }
            )
            != 3
        ):
            raise ValueError("migration API caller and service roles are not distinct")
        for kind, stack_id in (
            (StackKind.RETAINED, self.retained_stack_id),
            (StackKind.FENCE, self.fence_stack_id),
            (StackKind.SUPPORT, self.support_stack_id),
        ):
            StackIdentity(
                kind=kind,
                name=stack_name(kind),
                stack_id=stack_id,
                account_id=ACCOUNT_ID,
                region=REGION,
                termination_protection=True,
                tags=STACK_TAGS,
            )
        for value in (
            self.bridge_seed_policy_sha256,
            self.expected_live_preseed_policy_sha256,
            self.bridge_seed_owner_template_sha256,
            self.fence_import_template_sha256,
            self.fence_transfer_template_sha256,
            self.active_original_template_sha256,
            self.active_processed_template_sha256,
            self.support_prestate_template_sha256,
        ):
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise ValueError("checkpoint contains a non-SHA-256 identity")
        _validate_bucket_name(self.bucket_name)
        if (
            _LOGICAL_ID.fullmatch(self.retained_owner_logical_id) is None
            or self.retained_owner_physical_id != self.bucket_name
        ):
            raise ValueError("checkpoint retained owner identity is not exact")
        _validate_change_set_id(
            self.import_change_set_id,
            expected_name=self.import_change_set_id.split("/")[1]
            if "/" in self.import_change_set_id
            else "",
        )
        if self.canonical_identity_sha256 != _checkpoint_identity_sha256(self):
            raise ValueError("stack migration transfer checkpoint identity drifted")

    def to_dict(self) -> Mapping[str, object]:
        return _checkpoint_projection(self, include_identity=True)


def _checkpoint_projection(
    checkpoint: StackMigrationTransferCheckpointV2, *, include_identity: bool
) -> Dict[str, object]:
    value: Dict[str, object] = {
        "record_type": checkpoint.record_type,
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "retained_stack_id": checkpoint.retained_stack_id,
        "fence_stack_id": checkpoint.fence_stack_id,
        "support_stack_id": checkpoint.support_stack_id,
        "bucket_name": checkpoint.bucket_name,
        "retained_owner_logical_id": checkpoint.retained_owner_logical_id,
        "retained_owner_physical_id": checkpoint.retained_owner_physical_id,
        "bridge_seed_owner_branch": checkpoint.bridge_seed_owner_branch,
        "bridge_seed_policy_sha256": checkpoint.bridge_seed_policy_sha256,
        "expected_live_preseed_policy_sha256": (
            checkpoint.expected_live_preseed_policy_sha256
        ),
        "bridge_seed_owner_template_sha256": (
            checkpoint.bridge_seed_owner_template_sha256
        ),
        "fence_import_template_sha256": checkpoint.fence_import_template_sha256,
        "fence_transfer_template_sha256": (
            checkpoint.fence_transfer_template_sha256
        ),
        "active_original_template_sha256": (
            checkpoint.active_original_template_sha256
        ),
        "active_processed_template_sha256": (
            checkpoint.active_processed_template_sha256
        ),
        "direct_policy_sha256": list(checkpoint.direct_policy_sha256),
        "stack_policy_sha256": checkpoint.stack_policy_sha256,
        "api_caller": checkpoint.api_caller.to_dict(),
        "migration_service_role": checkpoint.migration_service_role.to_dict(),
        "fence_service_role": checkpoint.fence_service_role.to_dict(),
        "associated_stack_role_arn": checkpoint.associated_stack_role_arn,
        "associated_stack_role_id": checkpoint.associated_stack_role_id,
        "rollback_stack_role_arn": checkpoint.rollback_stack_role_arn,
        "import_change_set_id": checkpoint.import_change_set_id,
        "completed_mutation_keys": list(checkpoint.completed_mutation_keys),
        "operation_7_status": checkpoint.operation_7_status,
        "support_state": checkpoint.support_state,
        "support_prestate_template_sha256": (
            checkpoint.support_prestate_template_sha256
        ),
        "deletion_policy": checkpoint.deletion_policy,
        "update_replace_policy": checkpoint.update_replace_policy,
    }
    if include_identity:
        value["canonical_identity_sha256"] = checkpoint.canonical_identity_sha256
    return value


def _checkpoint_identity_sha256(
    checkpoint: StackMigrationTransferCheckpointV2,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(_checkpoint_projection(checkpoint, include_identity=False))
    ).hexdigest()


def build_stack_migration_transfer_checkpoint_v2(
    *,
    bridge_seed_plan: BridgeSeedOwnershipPlanV2,
    retained: StackIdentity,
    fence: StackIdentity,
    support: StackIdentity,
    bucket_name: str,
    fence_import_template_sha256: str,
    fence_transfer_template_sha256: str,
    active_original_template_sha256: str,
    active_processed_template_sha256: str,
    direct_policy_sha256: Tuple[str, str],
    api_caller: MigrationRoleEvidenceV2,
    migration_service_role: MigrationRoleEvidenceV2,
    fence_service_role: MigrationRoleEvidenceV2,
    associated_stack_role_arn: str,
    associated_stack_role_id: str,
    import_change_set_id: str,
    support_prestate_template_sha256: str,
    completed_mutation_keys: Tuple[str, ...] = (
        STACK_MIGRATION_TRANSFER_COMPLETED_KEYS_V2
    ),
    operation_7_status: str = MigrationMutationStatus.NOT_SUBMITTED.value,
) -> StackMigrationTransferCheckpointV2:
    if (
        type(bridge_seed_plan) is not BridgeSeedOwnershipPlanV2
        or type(retained) is not StackIdentity
        or retained.kind is not StackKind.RETAINED
        or type(fence) is not StackIdentity
        or fence.kind is not StackKind.FENCE
        or type(support) is not StackIdentity
        or support.kind is not StackKind.SUPPORT
    ):
        raise TypeError("checkpoint requires exact retained/fence/support identities")
    fields = {
        "record_type": STACK_MIGRATION_TRANSFER_CHECKPOINT_RECORD_TYPE_V2,
        "retained_stack_id": retained.stack_id,
        "fence_stack_id": fence.stack_id,
        "support_stack_id": support.stack_id,
        "bucket_name": bucket_name,
        "retained_owner_logical_id": bridge_seed_plan.owner_logical_id,
        "retained_owner_physical_id": bucket_name,
        "bridge_seed_owner_branch": bridge_seed_plan.owner_branch,
        "bridge_seed_policy_sha256": bridge_seed_plan.seed_policy_sha256,
        "expected_live_preseed_policy_sha256": (
            bridge_seed_plan.expected_live_preseed_policy_sha256
        ),
        "bridge_seed_owner_template_sha256": (
            bridge_seed_plan.template_body_sha256
        ),
        "fence_import_template_sha256": fence_import_template_sha256,
        "fence_transfer_template_sha256": fence_transfer_template_sha256,
        "active_original_template_sha256": active_original_template_sha256,
        "active_processed_template_sha256": active_processed_template_sha256,
        "direct_policy_sha256": direct_policy_sha256,
        "stack_policy_sha256": FENCE_STACK_POLICY_SHA256_V2,
        "api_caller": api_caller,
        "migration_service_role": migration_service_role,
        "fence_service_role": fence_service_role,
        "associated_stack_role_arn": associated_stack_role_arn,
        "associated_stack_role_id": associated_stack_role_id,
        "rollback_stack_role_arn": migration_service_role.role_arn,
        "import_change_set_id": import_change_set_id,
        "completed_mutation_keys": completed_mutation_keys,
        "operation_7_status": operation_7_status,
        "support_state": "INERT_ANCHOR",
        "support_prestate_template_sha256": support_prestate_template_sha256,
        "deletion_policy": "Retain",
        "update_replace_policy": "Retain",
    }
    return StackMigrationTransferCheckpointV2(
        **fields,
        canonical_identity_sha256=_checkpoint_fields_identity_sha256(fields),
    )


@dataclass(frozen=True)
class StackMigrationAuthorityV2:
    """Distinct API-caller and CloudFormation service-role authority."""

    api_caller: MigrationRoleEvidenceV2
    migration_service_role: MigrationRoleEvidenceV2
    fence_service_role: MigrationRoleEvidenceV2
    fence_bootstrap: BootstrapCoordinate
    support_bootstrap: BootstrapCoordinate
    import_change_set_name: str
    action_identity_sha256: str

    def __post_init__(self) -> None:
        roles = (
            self.api_caller,
            self.migration_service_role,
            self.fence_service_role,
        )
        if (
            any(type(role) is not MigrationRoleEvidenceV2 for role in roles)
            or len({role.role_arn for role in roles}) != 3
            or len({role.role_id for role in roles}) != 3
            or self.migration_service_role.role_arn
            != MIGRATION_SERVICE_ROLE_ARN_V2
            or self.fence_service_role.role_arn
            != FENCE_SERVICE_ROLE_ARN_V2
            or type(self.fence_bootstrap) is not BootstrapCoordinate
            or self.fence_bootstrap.kind is not StackKind.FENCE
            or type(self.support_bootstrap) is not BootstrapCoordinate
            or self.support_bootstrap.kind is not StackKind.SUPPORT
            or type(self.import_change_set_name) is not str
            or re.fullmatch(r"[A-Za-z][-A-Za-z0-9]{0,127}", self.import_change_set_name)
            is None
            or type(self.action_identity_sha256) is not str
            or _SHA256.fullmatch(self.action_identity_sha256) is None
        ):
            raise ValueError("stack migration authority v2 is not exact")


@dataclass(frozen=True)
class StackMigrationEvidenceV2:
    """Authenticated pre-transfer stack and direct-policy evidence."""

    retained: StackIdentity
    fence: StackIdentity
    support: StackIdentity
    current_policy_logical_id: Optional[str]
    current_policy_physical_id: Optional[str]
    current_policy_stack_id: Optional[str]
    bucket_name: str
    import_identifier: Tuple[Tuple[str, str], ...]
    preseed_policy: Mapping[str, object]
    direct_preseed_policy_readbacks: Tuple[Mapping[str, object], ...]
    retained_resource_physical_ids: Tuple[str, ...]
    retained_export_names: Tuple[str, ...]
    support_export_names: Tuple[str, ...]


@dataclass(frozen=True)
class StackMigrationTransferBundleV2:
    """Only bridge-seed ownership and operations 1-6 artifacts."""

    bridge_seed_plan: BridgeSeedOwnershipPlanV2
    artifacts: Tuple[TemplateArtifact, ...]
    manifest: Mapping[str, object]
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.bridge_seed_plan) is not BridgeSeedOwnershipPlanV2
            or type(self.artifacts) is not tuple
            or tuple(artifact.stage for artifact in self.artifacts)
            != (
                "bridge-seed-owner",
                "retention-only",
                "post-retain",
                "fence-import",
                "fence-transfer",
            )
            or type(self.manifest) is not dict
            or self.manifest.get("record_type")
            != "glm52_h1g_stack_migration_transfer_manifest_v2"
            or self.manifest.get("execution_eligible") is not True
            or self.manifest.get("direct_put_bucket_policy_fallback") is not False
        ):
            raise ValueError("stack migration transfer bundle v2 is not exact")
        projection = {
            "bridge_seed_plan_identity_sha256": (
                self.bridge_seed_plan.canonical_identity_sha256
            ),
            "artifact_sha256s": [artifact.sha256 for artifact in self.artifacts],
            "manifest": self.manifest,
        }
        if (
            hashlib.sha256(canonical_json_bytes(projection)).hexdigest()
            != self.canonical_identity_sha256
        ):
            raise ValueError("stack migration transfer bundle identity drifted")


def _legacy_authority_from_v2(
    authority: StackMigrationAuthorityV2,
) -> MigrationExecutionAuthority:
    if type(authority) is not StackMigrationAuthorityV2:
        raise TypeError("stack migration authority must be exact v2")
    return MigrationExecutionAuthority(
        deployment_role_arn=authority.migration_service_role.role_arn,
        deployment_role_id=authority.migration_service_role.role_id,
        fence_bootstrap=authority.fence_bootstrap,
        support_bootstrap=authority.support_bootstrap,
        import_change_set_name=authority.import_change_set_name,
        action_identity_sha256=authority.action_identity_sha256,
    )


def initial_stack_migration_execution_state_v2(
    authority: StackMigrationAuthorityV2,
) -> MigrationExecutionState:
    return initial_migration_execution_state(_legacy_authority_from_v2(authority))


def build_stack_migration_v2(
    *,
    bootstrap_template: Mapping[str, object],
    bootstrap_result: MigrationBootstrapResult,
    evidence: StackMigrationEvidenceV2,
    authority: StackMigrationAuthorityV2,
    bridge_seed_plan: BridgeSeedOwnershipPlanV2,
    template_coordinates: Tuple[TemplateCoordinate, ...],
) -> StackMigrationTransferBundleV2:
    """Build bridge-seed ownership and operations 1-6 only."""

    if (
        type(bootstrap_result) is not MigrationBootstrapResult
        or type(evidence) is not StackMigrationEvidenceV2
        or type(authority) is not StackMigrationAuthorityV2
        or type(bridge_seed_plan) is not BridgeSeedOwnershipPlanV2
    ):
        raise TypeError("stack migration v2 build inputs are not exact")
    if (
        evidence.fence != bootstrap_result.fence
        or evidence.support != bootstrap_result.support
        or evidence.import_identifier != (("Bucket", evidence.bucket_name),)
        or evidence.current_policy_physical_id
        not in {None, evidence.bucket_name}
        or evidence.current_policy_stack_id
        not in {None, evidence.retained.stack_id}
        or (
            bridge_seed_plan.owner_branch == "ADD"
            and any(
                item is not None
                for item in (
                    evidence.current_policy_logical_id,
                    evidence.current_policy_physical_id,
                    evidence.current_policy_stack_id,
                )
            )
        )
        or (
            bridge_seed_plan.owner_branch == "MODIFY"
            and evidence.current_policy_logical_id
            != bridge_seed_plan.owner_logical_id
        )
    ):
        raise ValueError("stack migration v2 owner/bootstrap evidence drifted")
    _validate_bucket_name(evidence.bucket_name)
    _validate_literal_policy(evidence.preseed_policy)
    preseed_bytes = canonical_json_bytes(evidence.preseed_policy)
    if (
        type(evidence.direct_preseed_policy_readbacks) is not tuple
        or len(evidence.direct_preseed_policy_readbacks) != 2
        or any(
            canonical_json_bytes(readback) != preseed_bytes
            for readback in evidence.direct_preseed_policy_readbacks
        )
        or hashlib.sha256(preseed_bytes).hexdigest()
        != bridge_seed_plan.expected_live_preseed_policy_sha256
        or bridge_seed_plan.template
        != json.loads(bridge_seed_plan.template_body)
    ):
        raise ValueError("stack migration v2 preseed evidence is not exact")
    validate_bootstrap_template(bootstrap_template)
    retention = build_retention_only_template(
        archived_template=bridge_seed_plan.template,
        current_policy_logical_id=bridge_seed_plan.owner_logical_id,
    )
    post_retain = build_post_retain_template(
        retention_template=retention,
        current_policy_logical_id=bridge_seed_plan.owner_logical_id,
    )
    seed_policy = _resources(bridge_seed_plan.template)[
        bridge_seed_plan.owner_logical_id
    ]["Properties"]["PolicyDocument"]
    fence_import = build_fence_import_template(
        bootstrap_template=bootstrap_template,
        bucket_name=evidence.bucket_name,
        canonical_live_policy=seed_policy,
    )
    fence_transfer = build_fence_transfer_template_v2(
        fence_import_template=fence_import
    )
    stages = (
        ("bridge-seed-owner", StackKind.RETAINED, bridge_seed_plan.template),
        ("retention-only", StackKind.RETAINED, retention),
        ("post-retain", StackKind.RETAINED, post_retain),
        ("fence-import", StackKind.FENCE, fence_import),
        ("fence-transfer", StackKind.FENCE, fence_transfer),
    )
    if (
        type(template_coordinates) is not tuple
        or len(template_coordinates) != len(stages)
    ):
        raise ValueError("five immutable transfer coordinates are required")
    coordinates = {coordinate.stage: coordinate for coordinate in template_coordinates}
    if (
        len(coordinates) != len(stages)
        or tuple(coordinates) != tuple(stage for stage, _kind, _template in stages)
    ):
        raise ValueError("transfer coordinates are not in exact order")
    identities = {
        StackKind.RETAINED: evidence.retained,
        StackKind.FENCE: evidence.fence,
        StackKind.SUPPORT: evidence.support,
    }
    artifacts = tuple(
        _artifact(
            stage=stage,
            template=template,
            coordinate=coordinates[stage],
        )
        for stage, kind, template in stages
        if coordinates[stage].stack_id == identities[kind].stack_id
    )
    if len(artifacts) != len(stages):
        raise ValueError("transfer coordinate targets the wrong StackId")
    manifest = {
        "schema_version": 2,
        "record_type": "glm52_h1g_stack_migration_transfer_manifest_v2",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "bootstrap_template_body_sha256": hashlib.sha256(
            canonical_json_bytes(bootstrap_template)
        ).hexdigest(),
        "bucket_name": evidence.bucket_name,
        "bridge_seed_owner_branch": bridge_seed_plan.owner_branch,
        "bridge_seed_policy_sha256": bridge_seed_plan.seed_policy_sha256,
        "preseed_template_body_sha256": (
            bridge_seed_plan.preseed_template_body_sha256
        ),
        "preseed_policy_sha256": hashlib.sha256(preseed_bytes).hexdigest(),
        "api_caller": authority.api_caller.to_dict(),
        "migration_service_role": authority.migration_service_role.to_dict(),
        "fence_service_role": authority.fence_service_role.to_dict(),
        "stack_policy_sha256": FENCE_STACK_POLICY_SHA256_V2,
        "artifacts": [
            {
                "stage": artifact.stage,
                "stack_id": artifact.stack_id,
                "template_url": artifact.template_url,
                "version_id": artifact.version_id,
                "sha256": artifact.sha256,
                "template_body_sha256": artifact.template_body_sha256,
                "policy_sha256": artifact.policy_sha256,
            }
            for artifact in artifacts
        ],
        "operation_7_status": MigrationMutationStatus.NOT_SUBMITTED.value,
        "execution_eligible": True,
        "direct_put_bucket_policy_fallback": False,
    }
    projection = {
        "bridge_seed_plan_identity_sha256": (
            bridge_seed_plan.canonical_identity_sha256
        ),
        "artifact_sha256s": [artifact.sha256 for artifact in artifacts],
        "manifest": manifest,
    }
    return StackMigrationTransferBundleV2(
        bridge_seed_plan=bridge_seed_plan,
        artifacts=artifacts,
        manifest=manifest,
        canonical_identity_sha256=hashlib.sha256(
            canonical_json_bytes(projection)
        ).hexdigest(),
    )


def _stack_policy_sha256_v2(client: object, *, stack_id: str) -> str:
    response = _successful_response(
        client.get_stack_policy(StackName=stack_id),
        "GetStackPolicy",
    )
    raw = response.get("StackPolicyBody")
    if type(raw) is str:
        try:
            value = json.loads(
                raw,
                object_pairs_hook=_exact_json_object,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("fence stack policy is not strict JSON") from exc
    elif type(raw) is dict:
        value = raw
    else:
        raise ValueError("GetStackPolicy omitted exact policy body")
    if value != fence_stack_policy_v2():
        raise ValueError("fence stack policy is not immutable v2")
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
