"""Restart-safe Task 13 H.1g staged deployment temporal spine.

The coordinator records one append-only journal and admits exactly this order:
bridge seed, migration operations 1--6 checkpoint, manifest-last bootstrap
publication, PREPARE execution/stabilization, two independent support-input
snapshots, disabled support, gated operation 7, and singular runtime identity.
It never grants launch or submission authority.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .canonical import canonical_json_bytes, canonical_sha256
from .fence_artifacts import (
    BOOTSTRAP_PUBLICATION_ORDER,
    ArtifactCoordinate,
    parse_artifact_coordinate,
)
from .task13_support_input_materialization import (
    SupportBuildInputs,
    support_build_inputs_from_mapping,
    support_build_inputs_identity,
    support_build_inputs_projection,
)

if TYPE_CHECKING:
    from .fence_executor import FenceExecutionResult
    from .support_fence_handler import SupportRuntimeIdentity
    from .task13_migration_adapter import (
        StackMigrationOperation7EvidenceV2,
        StackMigrationTransferCheckpointV2,
    )

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
MODEL_BUCKET_NAME = "keep-glm52-models-246813579024-us-west-2"
RETAINED_STACK_NAME = "keep-glm52-gpu"
FENCE_STACK_NAME = "keep-glm52-h1g-fence"
SUPPORT_STACK_NAME = "keep-glm52-h1g-support"
FENCE_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/keep-glm52-h1g-fence-service"
)
SUPPORT_ROLE_ARN = (
    "arn:aws:iam::246813579024:"
    "role/keep-glm52-h1g-cloudformation-deployment"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ACTIVATION_ID = re.compile(r"[a-z0-9][a-z0-9-]{2,63}\Z")
_STACK_ID = re.compile(
    r"arn:aws:cloudformation:us-west-2:246813579024:stack/"
    r"keep-glm52-gpu/[0-9a-f-]{36}\Z"
)
_LAMBDA_VERSION_ARN = re.compile(
    r"arn:aws:lambda:us-west-2:246813579024:function:"
    r"[A-Za-z0-9_-]+:[1-9][0-9]*\Z"
)
_JOURNAL_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "sequence",
        "request_identity_sha256",
        "step",
        "state",
        "evidence",
        "previous_record_sha256",
        "record_sha256",
    }
)
_COORDINATE_FIELDS = frozenset(
    {
        "artifact_kind",
        "bucket",
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
    }
)


class StagedDeploymentError(ValueError):
    """The staged route or its durable evidence failed closed."""


class DeploymentStep(Enum):
    """The one legal pre-source deployment order."""

    RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED = (
        "RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED"
    )
    BRIDGE_SEED_PUBLISHED = "BRIDGE_SEED_PUBLISHED"
    BRIDGE_SEED_ESTABLISHED = "BRIDGE_SEED_ESTABLISHED"
    STACK_MIGRATION_OPERATIONS_1_TO_6 = "STACK_MIGRATION_OPERATIONS_1_TO_6"
    BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED = (
        "BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED"
    )
    RETAINED_FENCE_RUNTIME_DEPLOYED = "RETAINED_FENCE_RUNTIME_DEPLOYED"
    PREPARE_EXECUTED_STABILIZED = "PREPARE_EXECUTED_STABILIZED"
    SUPPORT_INPUT_SNAPSHOT_ONE = "SUPPORT_INPUT_SNAPSHOT_ONE"
    SUPPORT_INPUT_SNAPSHOT_TWO = "SUPPORT_INPUT_SNAPSHOT_TWO"
    DISABLED_SUPPORT_DEPLOYED = "DISABLED_SUPPORT_DEPLOYED"
    STACK_MIGRATION_OPERATION_7 = "STACK_MIGRATION_OPERATION_7"
    SUPPORT_RUNTIME_IDENTITY_COMMITTED = (
        "SUPPORT_RUNTIME_IDENTITY_COMMITTED"
    )
    NO_WORKER_ACTIVATION_PROVED = "NO_WORKER_ACTIVATION_PROVED"


_MUTATING_STEPS = frozenset(
    {
        DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED,
        DeploymentStep.BRIDGE_SEED_PUBLISHED,
        DeploymentStep.BRIDGE_SEED_ESTABLISHED,
        DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6,
        DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED,
        DeploymentStep.RETAINED_FENCE_RUNTIME_DEPLOYED,
        DeploymentStep.PREPARE_EXECUTED_STABILIZED,
        DeploymentStep.DISABLED_SUPPORT_DEPLOYED,
        DeploymentStep.STACK_MIGRATION_OPERATION_7,
        DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED,
    }
)


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise StagedDeploymentError(label + " must be one lowercase SHA-256")
    return value


def _detached(value: object, label: str) -> object:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StagedDeploymentError(label + " is not canonical JSON data") from exc


def _identity_projection(value: Mapping[str, object], label: str) -> None:
    identity = _sha(value.get("canonical_identity_sha256"), label + " identity")
    unsigned = dict(value)
    unsigned.pop("canonical_identity_sha256")
    if canonical_sha256(unsigned) != identity:
        raise StagedDeploymentError(label + " canonical identity drifted")


@dataclass(frozen=True)
class BootstrapFencePublication:
    """The seed plus four bootstrap templates and manifest-last receipt."""

    coordinates: tuple[ArtifactCoordinate, ...]
    prepare_entry_identity_sha256: str
    checkpoint_identity_sha256: str
    seed_adopted: bool
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.coordinates) is not tuple
            or len(self.coordinates) != len(BOOTSTRAP_PUBLICATION_ORDER)
            or tuple(item.key for item in self.coordinates)
            != BOOTSTRAP_PUBLICATION_ORDER
            or any(type(item) is not ArtifactCoordinate for item in self.coordinates)
            or self.seed_adopted is not True
        ):
            raise StagedDeploymentError(
                "bootstrap publication is not seed plus four templates and manifest last"
            )
        _sha(self.prepare_entry_identity_sha256, "PREPARE entry identity")
        _sha(self.checkpoint_identity_sha256, "operation-6 checkpoint identity")
        value = self.to_dict()
        _identity_projection(value, "bootstrap publication")

    @property
    def manifest_coordinate(self) -> ArtifactCoordinate:
        return self.coordinates[-1]

    @property
    def seed_coordinate(self) -> ArtifactCoordinate:
        return self.coordinates[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "record_type": "glm52_h1g_bootstrap_fence_publication_v2",
            "coordinates": [item.to_dict() for item in self.coordinates],
            "prepare_entry_identity_sha256": self.prepare_entry_identity_sha256,
            "checkpoint_identity_sha256": self.checkpoint_identity_sha256,
            "seed_adopted": self.seed_adopted,
            "canonical_identity_sha256": self.canonical_identity_sha256,
        }


def parse_bootstrap_fence_publication(value: object) -> BootstrapFencePublication:
    fields = {
        "schema_version",
        "record_type",
        "coordinates",
        "prepare_entry_identity_sha256",
        "checkpoint_identity_sha256",
        "seed_adopted",
        "canonical_identity_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise StagedDeploymentError("bootstrap publication schema is not exact")
    if (
        value["schema_version"] != 2
        or value["record_type"] != "glm52_h1g_bootstrap_fence_publication_v2"
        or type(value["coordinates"]) is not list
    ):
        raise StagedDeploymentError("bootstrap publication identity is not exact")
    return BootstrapFencePublication(
        coordinates=tuple(
            parse_artifact_coordinate(item) for item in value["coordinates"]
        ),
        prepare_entry_identity_sha256=_sha(
            value["prepare_entry_identity_sha256"], "PREPARE entry identity"
        ),
        checkpoint_identity_sha256=_sha(
            value["checkpoint_identity_sha256"], "checkpoint identity"
        ),
        seed_adopted=value["seed_adopted"],
        canonical_identity_sha256=_sha(
            value["canonical_identity_sha256"], "publication identity"
        ),
    )


@dataclass(frozen=True)
class DisabledSupportDeploymentEvidence:
    """The sole operation-7 input: a deployed, launch-inert support profile."""

    bootstrap_manifest_coordinate: ArtifactCoordinate
    prepare_entry_identity_sha256: str
    support_build_inputs_identity_sha256: str
    disabled_support_profile_sha256: str
    support_template_coordinate: Mapping[str, object]
    support_template_sha256: str
    no_launch_evidence_sha256: str
    canonical_identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.bootstrap_manifest_coordinate) is not ArtifactCoordinate:
            raise StagedDeploymentError("disabled support manifest coordinate is not exact")
        for field, value in (
            ("PREPARE entry identity", self.prepare_entry_identity_sha256),
            ("support build input identity", self.support_build_inputs_identity_sha256),
            ("disabled support profile identity", self.disabled_support_profile_sha256),
            ("support template identity", self.support_template_sha256),
            ("no-launch evidence identity", self.no_launch_evidence_sha256),
        ):
            _sha(value, field)
        coordinate = self.support_template_coordinate
        if (
            type(coordinate) is not dict
            or set(coordinate) != _COORDINATE_FIELDS
            or coordinate.get("artifact_kind") != "DISABLED_SUPPORT_TEMPLATE"
            or coordinate.get("bucket") != MODEL_BUCKET_NAME
            or type(coordinate.get("key")) is not str
            or type(coordinate.get("version_id")) is not str
            or not coordinate["version_id"]
        ):
            raise StagedDeploymentError("disabled support template coordinate is not exact")
        _sha(coordinate.get("file_sha256"), "support template file hash")
        _sha(coordinate.get("body_sha256"), "support template body hash")
        _identity_projection(self.to_dict(), "disabled support deployment")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "record_type": "glm52_h1g_disabled_support_deployment_v1",
            "profile_record_type": "disabled_support_profile_v1",
            "bootstrap_manifest_coordinate": (
                self.bootstrap_manifest_coordinate.to_dict()
            ),
            "prepare_entry_identity_sha256": self.prepare_entry_identity_sha256,
            "support_build_inputs_identity_sha256": (
                self.support_build_inputs_identity_sha256
            ),
            "disabled_support_profile_sha256": self.disabled_support_profile_sha256,
            "support_template_coordinate": dict(self.support_template_coordinate),
            "support_template_sha256": self.support_template_sha256,
            "no_launch_evidence_sha256": self.no_launch_evidence_sha256,
            "worker_activation_allowed": False,
            "source_action_allowed": False,
            "canonical_identity_sha256": self.canonical_identity_sha256,
        }


def parse_disabled_support_deployment_evidence(
    value: object,
) -> DisabledSupportDeploymentEvidence:
    fields = {
        "schema_version",
        "record_type",
        "profile_record_type",
        "bootstrap_manifest_coordinate",
        "prepare_entry_identity_sha256",
        "support_build_inputs_identity_sha256",
        "disabled_support_profile_sha256",
        "support_template_coordinate",
        "support_template_sha256",
        "no_launch_evidence_sha256",
        "worker_activation_allowed",
        "source_action_allowed",
        "canonical_identity_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise StagedDeploymentError("disabled support deployment schema is not exact")
    if (
        value["schema_version"] != 1
        or value["record_type"] != "glm52_h1g_disabled_support_deployment_v1"
        or value["profile_record_type"] != "disabled_support_profile_v1"
        or value["worker_activation_allowed"] is not False
        or value["source_action_allowed"] is not False
    ):
        raise StagedDeploymentError("disabled support deployment is not launch-inert")
    return DisabledSupportDeploymentEvidence(
        bootstrap_manifest_coordinate=parse_artifact_coordinate(
            value["bootstrap_manifest_coordinate"]
        ),
        prepare_entry_identity_sha256=_sha(
            value["prepare_entry_identity_sha256"], "PREPARE entry identity"
        ),
        support_build_inputs_identity_sha256=_sha(
            value["support_build_inputs_identity_sha256"],
            "support build input identity",
        ),
        disabled_support_profile_sha256=_sha(
            value["disabled_support_profile_sha256"],
            "disabled support profile identity",
        ),
        support_template_coordinate=_detached(
            value["support_template_coordinate"], "support template coordinate"
        ),
        support_template_sha256=_sha(
            value["support_template_sha256"], "support template identity"
        ),
        no_launch_evidence_sha256=_sha(
            value["no_launch_evidence_sha256"], "no-launch evidence identity"
        ),
        canonical_identity_sha256=_sha(
            value["canonical_identity_sha256"], "disabled support identity"
        ),
    )


@dataclass(frozen=True)
class StagedDeploymentRequest:
    """Immutable local request for the v2 staged transaction."""

    schema_version: int
    record_type: str
    activation_id: str
    journal_path: Path
    production_request: Mapping[str, object]

    def __post_init__(self) -> None:
        journal = Path(self.journal_path)
        if (
            self.schema_version != 2
            or self.record_type != "glm52_task13_staged_deployment_request_v2"
            or type(self.activation_id) is not str
            or _ACTIVATION_ID.fullmatch(self.activation_id) is None
            or not journal.is_absolute()
            or not journal.parent.is_dir()
            or journal.parent.is_symlink()
            or journal.parent.resolve(strict=True) != journal.parent
            or journal.name in {"", ".", ".."}
            or type(self.production_request) is not dict
            or not self.production_request
        ):
            raise StagedDeploymentError("staged deployment v2 request is not exact")
        object.__setattr__(self, "journal_path", journal)
        object.__setattr__(
            self,
            "production_request",
            _detached(self.production_request, "production request"),
        )


class StagedDeploymentOperations(Protocol):
    """Typed production boundary for each temporal-spine transition."""
    def deploy_retained_bootstrap_runtime(
        self,
        request: StagedDeploymentRequest,
    ) -> Mapping[str, object]: ...

    def publish_bridge_seed(
        self,
        request: StagedDeploymentRequest,
        bootstrap_runtime: Mapping[str, object],
    ) -> object: ...


    def establish_bridge_seed(
        self,
        request: StagedDeploymentRequest,
        bridge_seed_publication: object,
    ) -> object: ...

    def complete_stack_migration_operations_1_to_6(
        self, request: StagedDeploymentRequest, bridge_seed: object
    ) -> "StackMigrationTransferCheckpointV2": ...

    def publish_bootstrap_fence_artifacts(
        self,
        request: StagedDeploymentRequest,
        checkpoint: "StackMigrationTransferCheckpointV2",
        bootstrap_runtime: Mapping[str, object],
    ) -> BootstrapFencePublication: ...
    def deploy_retained_fence_runtime(
        self,
        request: StagedDeploymentRequest,
        checkpoint: "StackMigrationTransferCheckpointV2",
        publication: BootstrapFencePublication,
    ) -> Mapping[str, object]: ...

    def execute_prepare(
        self,
        request: StagedDeploymentRequest,
        checkpoint: "StackMigrationTransferCheckpointV2",
        publication: BootstrapFencePublication,
    ) -> "FenceExecutionResult": ...

    def collect_support_input_snapshot(
        self,
        request: StagedDeploymentRequest,
        checkpoint: "StackMigrationTransferCheckpointV2",
        prepare_result: "FenceExecutionResult",
        publication: BootstrapFencePublication,
    ) -> SupportBuildInputs: ...

    def deploy_disabled_support(
        self,
        request: StagedDeploymentRequest,
        checkpoint: "StackMigrationTransferCheckpointV2",
        support_inputs: SupportBuildInputs,
    ) -> DisabledSupportDeploymentEvidence: ...

    def complete_stack_migration_operation_7(
        self,
        request: StagedDeploymentRequest,
        checkpoint: "StackMigrationTransferCheckpointV2",
        prepare_result: "FenceExecutionResult",
        disabled_support: DisabledSupportDeploymentEvidence,
    ) -> "StackMigrationOperation7EvidenceV2": ...

    def commit_support_runtime_identity(
        self,
        request: StagedDeploymentRequest,
        operation_7: "StackMigrationOperation7EvidenceV2",
        disabled_support: DisabledSupportDeploymentEvidence,
    ) -> "SupportRuntimeIdentity": ...

    def prove_no_worker_activation(
        self,
        request: StagedDeploymentRequest,
        runtime_identity: "SupportRuntimeIdentity",
    ) -> Mapping[str, object]: ...

    def reconcile_mutation(
        self,
        step: DeploymentStep,
        request: StagedDeploymentRequest,
        committed: Mapping[DeploymentStep, object],
        possible_send_evidence: Mapping[str, object],
    ) -> object: ...

    def adopt_committed(
        self,
        step: DeploymentStep,
        request: StagedDeploymentRequest,
        evidence: Mapping[str, object],
        committed: Mapping[DeploymentStep, object],
    ) -> object: ...


@dataclass(frozen=True)
class StagedDeploymentResult:
    activation_id: str
    completed_steps: tuple[DeploymentStep, ...]
    journal_sha256: str
    checkpoint: object
    bootstrap_publication: BootstrapFencePublication
    prepare_result: object
    support_inputs: SupportBuildInputs
    disabled_support: DisabledSupportDeploymentEvidence
    operation_7_evidence: object
    runtime_identity: object
    staged_infrastructure_evidence_path: Path
    staged_infrastructure_identity_sha256: str
    final_worker_count: int
    raw_ec2_launch_calls: int
    worker_activation_allowed: bool = False


@dataclass(frozen=True)
class _JournalState:
    committed: dict[DeploymentStep, Mapping[str, object]]
    possible_send_step: DeploymentStep | None
    possible_send_evidence: Mapping[str, object] | None
    next_sequence: int
    previous_record_sha256: str | None


def _request_identity(request: StagedDeploymentRequest) -> str:
    return canonical_sha256(
        {
            "schema_version": request.schema_version,
            "record_type": request.record_type,
            "activation_id": request.activation_id,
            "production_request": request.production_request,
        }
    )


def _projection(step: DeploymentStep, evidence: object) -> Mapping[str, object]:
    if step is DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED:
        if type(evidence) is not BootstrapFencePublication:
            raise StagedDeploymentError("bootstrap publication result type is not exact")
        return evidence.to_dict()
    if step is DeploymentStep.BRIDGE_SEED_PUBLISHED:
        from .fence_bootstrap_publication import BridgeSeedPublicationV2

        if type(evidence) is not BridgeSeedPublicationV2:
            raise StagedDeploymentError(
                "bridge seed publication result type is not exact"
            )
        return evidence.to_dict()
    if step in {
        DeploymentStep.SUPPORT_INPUT_SNAPSHOT_ONE,
        DeploymentStep.SUPPORT_INPUT_SNAPSHOT_TWO,
    }:
        if type(evidence) is not SupportBuildInputs:
            raise StagedDeploymentError("support snapshot result type is not exact")
        return support_build_inputs_projection(evidence)
    if step is DeploymentStep.DISABLED_SUPPORT_DEPLOYED:
        if type(evidence) is not DisabledSupportDeploymentEvidence:
            raise StagedDeploymentError("disabled support result type is not exact")
        return evidence.to_dict()
    if step is DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6:
        from .task13_migration_adapter import StackMigrationTransferCheckpointV2

        if type(evidence) is not StackMigrationTransferCheckpointV2:
            raise StagedDeploymentError("operation-6 checkpoint result type is not exact")
    elif step is DeploymentStep.PREPARE_EXECUTED_STABILIZED:
        from .fence_executor import FenceExecutionResult

        if type(evidence) is not FenceExecutionResult:
            raise StagedDeploymentError("PREPARE execution result type is not exact")
    elif step is DeploymentStep.STACK_MIGRATION_OPERATION_7:
        from .task13_migration_adapter import StackMigrationOperation7EvidenceV2

        if type(evidence) is not StackMigrationOperation7EvidenceV2:
            raise StagedDeploymentError("operation-7 result type is not exact")
    elif step is DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED:
        from .support_plane import SupportRuntimeIdentity

        if type(evidence) is not SupportRuntimeIdentity:
            raise StagedDeploymentError("support runtime identity type is not exact")
    if type(evidence) is dict:
        return _detached(evidence, step.value + " evidence")
    to_dict = getattr(evidence, "to_dict", None)
    if not callable(to_dict):
        raise StagedDeploymentError(step.value + " result has no public projection")
    projected = to_dict()
    if type(projected) is not dict:
        raise StagedDeploymentError(step.value + " public projection is not exact")
    return _detached(projected, step.value + " evidence")


def _validate_evidence(step: DeploymentStep, evidence: object) -> dict[str, object]:
    projected = dict(_projection(step, evidence))
    if step is DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED:
        required = {
            "schema_version",
            "record_type",
            "retained_stack_id",
            "template_sha256",
            "retained_update_identity_sha256",
            "materializer_function_version_arn",
            "publisher_role_arn",
            "worker_activation_allowed",
            "source_action_allowed",
            "invoker_role_arn",
            "canonical_identity_sha256",
        }
        if (
            set(projected) != required
            or projected["schema_version"] != 2
            or projected["record_type"]
            != "glm52_h1g_retained_bootstrap_runtime_deployment_v2"
            or type(projected["retained_stack_id"]) is not str
            or _STACK_ID.fullmatch(projected["retained_stack_id"]) is None
            or type(projected["materializer_function_version_arn"]) is not str
            or _LAMBDA_VERSION_ARN.fullmatch(
                projected["materializer_function_version_arn"]
            )
            is None
            or projected["publisher_role_arn"]
            != (
                "arn:aws:iam::246813579024:role/"
                "keep-glm52-h1g-fence-bootstrap-artifact-publisher"
            )
            or projected["invoker_role_arn"]
            != (
                "arn:aws:iam::246813579024:role/"
                "keep-glm52-h1g-fence-bootstrap-materializer-invoker"
            )
            or projected["worker_activation_allowed"] is not False
            or projected["source_action_allowed"] is not False
        ):
            raise StagedDeploymentError(
                "retained bootstrap runtime deployment is not exact"
            )
        _sha(projected["template_sha256"], "bootstrap runtime template")
        _sha(
            projected["retained_update_identity_sha256"],
            "bootstrap retained update identity",
        )
        _identity_projection(projected, "bootstrap runtime deployment")
    elif step is DeploymentStep.BRIDGE_SEED_PUBLISHED:
        if (
            projected.get("record_type")
            != "glm52_h1g_bridge_seed_publication_v2"
            or type(projected.get("artifact")) is not dict
            or projected.get("disposition") not in {"WRITTEN", "ADOPTED"}
        ):
            raise StagedDeploymentError(
                "bridge seed publication evidence is not exact"
            )
        _identity_projection(projected, "bridge seed publication")
        projected["disposition"] = "ADOPTED"
        projected.pop("canonical_identity_sha256")
        projected["canonical_identity_sha256"] = canonical_sha256(projected)
    elif step is DeploymentStep.RETAINED_FENCE_RUNTIME_DEPLOYED:
        required = {
            "schema_version",
            "record_type",
            "retained_stack_id",
            "template_sha256",
            "bootstrap_manifest_coordinate",
            "retained_update_identity_sha256",
            "pre_support_executor_function_version_arn",
            "source_settlement_function_version_arn",
            "worker_activation_allowed",
            "source_action_allowed",
            "canonical_identity_sha256",
        }
        if (
            set(projected) != required
            or projected["schema_version"] != 2
            or projected["record_type"]
            != "glm52_h1g_retained_fence_runtime_deployment_v2"
            or type(projected["retained_stack_id"]) is not str
            or _STACK_ID.fullmatch(projected["retained_stack_id"]) is None
            or any(
                type(projected[field]) is not str
                or _LAMBDA_VERSION_ARN.fullmatch(projected[field]) is None
                for field in (
                    "pre_support_executor_function_version_arn",
                    "source_settlement_function_version_arn",
                )
            )
            or projected["worker_activation_allowed"] is not False
            or projected["source_action_allowed"] is not False
        ):
            raise StagedDeploymentError(
                "retained fence runtime deployment is not exact"
            )
        _sha(projected["template_sha256"], "fence runtime template")
        parse_artifact_coordinate(projected["bootstrap_manifest_coordinate"])
        _sha(
            projected["retained_update_identity_sha256"],
            "fence retained update identity",
        )
        _identity_projection(projected, "fence runtime deployment")
    elif step is DeploymentStep.BRIDGE_SEED_ESTABLISHED:
        if (
            projected.get("record_type")
            != "glm52_h1g_bridge_seed_established_v2"
            or projected.get("policy_committed") is not True
            or projected.get("execution_eligible") is not True
        ):
            raise StagedDeploymentError("bridge seed is not committed v2 evidence")
    elif step is DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6:
        if (
            projected.get("record_type")
            != "STACK_MIGRATION_TRANSFER_CHECKPOINT_V2"
            or projected.get("operation_7_status") != "NOT_SUBMITTED"
        ):
            raise StagedDeploymentError("operation-6 checkpoint is not durable and gated")
    elif step is DeploymentStep.PREPARE_EXECUTED_STABILIZED:
        if (
            projected.get("record_type") != "glm52_fence_execution_result_v2"
            or projected.get("slot") != "PREPARE_GENESIS_LIVE_STATE"
            or projected.get("original_template_body_sha256")
            != projected.get("processed_template_body_sha256")
            or projected.get("first_stable_snapshot_identity_sha256")
            != projected.get("second_stable_snapshot_identity_sha256")
            or projected.get("stabilization_first_evidence_sha256")
            == projected.get("stabilization_second_evidence_sha256")
        ):
            raise StagedDeploymentError("PREPARE did not execute and stabilize exactly")
    elif step is DeploymentStep.STACK_MIGRATION_OPERATION_7:
        if (
            projected.get("record_type")
            != "STACK_MIGRATION_OPERATION_7_EVIDENCE_V2"
            or projected.get("operation_7_status") != "COMPLETE"
            or projected.get("mutation_scope") != "SUPPORT_REPLACEMENT_ONLY"
        ):
            raise StagedDeploymentError("operation 7 evidence is not exact")
    elif step is DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED:
        if (
            projected.get("record_type")
            != "glm52_h1g_support_runtime_identity_v1"
            or projected.get("schema_version") != 1
            or type(projected.get("expected_contract")) is not dict
            or type(projected.get("live_identity")) is not dict
        ):
            raise StagedDeploymentError(
                "support runtime identity is not singularly committed"
            )
    elif step is DeploymentStep.NO_WORKER_ACTIVATION_PROVED:
        required = {
            "schema_version",
            "record_type",
            "worker_count",
            "raw_ec2_launch_calls",
            "source_action_calls",
            "launch_authority_present",
            "complete",
        }
        if (
            set(projected) != required
            or projected["record_type"] != "glm52_h1g_no_launch_evidence_v2"
            or projected["worker_count"] != 0
            or projected["raw_ec2_launch_calls"] != 0
            or projected["source_action_calls"] != 0
            or projected["launch_authority_present"] is not False
            or projected["complete"] is not True
        ):
            raise StagedDeploymentError("final no-launch evidence is not exact")
    return projected


def _validate_cross_step_evidence(
    step: DeploymentStep,
    evidence: Mapping[str, object],
    committed: Mapping[DeploymentStep, object],
) -> None:
    expected_prior = tuple(DeploymentStep)[: tuple(DeploymentStep).index(step)]
    if tuple(committed) != expected_prior:
        raise StagedDeploymentError(step.value + " predecessors are incomplete or reordered")
    if step is DeploymentStep.BRIDGE_SEED_PUBLISHED:
        runtime = _projection(
            DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED,
            committed[DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED],
        )
        if evidence.get("materializer_function_version_arn") != runtime.get(
            "materializer_function_version_arn"
        ):
            raise StagedDeploymentError(
                "bridge seed was not published by the retained materializer"
            )
    elif step is DeploymentStep.BRIDGE_SEED_ESTABLISHED:
        publication = _projection(
            DeploymentStep.BRIDGE_SEED_PUBLISHED,
            committed[DeploymentStep.BRIDGE_SEED_PUBLISHED],
        )
        artifact = publication.get("artifact")
        if (
            type(artifact) is not dict
            or evidence.get("seed_policy_sha256")
            != artifact.get("policy_sha256")
        ):
            raise StagedDeploymentError(
                "established bridge seed is not the published artifact"
            )
    elif step is DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED:
        checkpoint = _projection(
            DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6,
            committed[DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6],
        )
        if evidence.get("checkpoint_identity_sha256") != checkpoint.get(
            "canonical_identity_sha256"
        ):
            raise StagedDeploymentError("bootstrap publication is not checkpoint-bound")
    elif step is DeploymentStep.RETAINED_FENCE_RUNTIME_DEPLOYED:
        publication = _projection(
            DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED,
            committed[DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED],
        )
        if evidence.get("bootstrap_manifest_coordinate") != publication[
            "coordinates"
        ][-1]:
            raise StagedDeploymentError(
                "retained fence runtime is not manifest-bound"
            )
    elif step is DeploymentStep.PREPARE_EXECUTED_STABILIZED:
        publication = _projection(
            DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED,
            committed[DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED],
        )
        if evidence.get("entry_identity_sha256") != publication.get(
            "prepare_entry_identity_sha256"
        ):
            raise StagedDeploymentError("PREPARE result is not the published unique entry")
    elif step in {
        DeploymentStep.SUPPORT_INPUT_SNAPSHOT_ONE,
        DeploymentStep.SUPPORT_INPUT_SNAPSHOT_TWO,
    }:
        prepare = _projection(
            DeploymentStep.PREPARE_EXECUTED_STABILIZED,
            committed[DeploymentStep.PREPARE_EXECUTED_STABILIZED],
        )
        publication = _projection(
            DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED,
            committed[DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED],
        )
        if (
            evidence.get("prepare_entry_identity_sha256")
            != prepare.get("entry_identity_sha256")
            or evidence.get("bootstrap_manifest_coordinate")
            != publication["coordinates"][-1]
        ):
            raise StagedDeploymentError("support snapshot is not post-PREPARE and manifest-bound")
        if step is DeploymentStep.SUPPORT_INPUT_SNAPSHOT_TWO:
            first = _projection(
                DeploymentStep.SUPPORT_INPUT_SNAPSHOT_ONE,
                committed[DeploymentStep.SUPPORT_INPUT_SNAPSHOT_ONE],
            )
            if canonical_json_bytes(first) != canonical_json_bytes(evidence):
                raise StagedDeploymentError("independent support snapshots drifted")
    elif step is DeploymentStep.DISABLED_SUPPORT_DEPLOYED:
        second = committed[DeploymentStep.SUPPORT_INPUT_SNAPSHOT_TWO]
        publication = _projection(
            DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED,
            committed[DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED],
        )
        if (
            evidence.get("support_build_inputs_identity_sha256")
            != support_build_inputs_identity(second)
            or evidence.get("bootstrap_manifest_coordinate")
            != publication["coordinates"][-1]
            or evidence.get("prepare_entry_identity_sha256")
            != publication.get("prepare_entry_identity_sha256")
        ):
            raise StagedDeploymentError("disabled support is not snapshot-bound")
    elif step is DeploymentStep.STACK_MIGRATION_OPERATION_7:
        disabled = _projection(
            DeploymentStep.DISABLED_SUPPORT_DEPLOYED,
            committed[DeploymentStep.DISABLED_SUPPORT_DEPLOYED],
        )
        checkpoint = _projection(
            DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6,
            committed[DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6],
        )
        prepare = _projection(
            DeploymentStep.PREPARE_EXECUTED_STABILIZED,
            committed[DeploymentStep.PREPARE_EXECUTED_STABILIZED],
        )
        if (
            evidence.get("disabled_support_identity_sha256")
            != disabled.get("canonical_identity_sha256")
            or evidence.get("checkpoint_identity_sha256")
            != checkpoint.get("canonical_identity_sha256")
            or evidence.get("prepare_execution_identity_sha256")
            != prepare.get("canonical_identity_sha256")
        ):
            raise StagedDeploymentError("operation 7 is not gated by exact prerequisites")
    elif step is DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED:
        disabled = _projection(
            DeploymentStep.DISABLED_SUPPORT_DEPLOYED,
            committed[DeploymentStep.DISABLED_SUPPORT_DEPLOYED],
        )
        expected = evidence.get("expected_contract")
        if (
            evidence.get("bootstrap_manifest_coordinate")
            != disabled.get("bootstrap_manifest_coordinate")
            or type(expected) is not dict
            or expected.get("disabled_support_profile_sha256")
            != disabled.get("disabled_support_profile_sha256")
        ):
            raise StagedDeploymentError("runtime identity was committed before operation 7")


def _journal_record(
    *,
    sequence: int,
    request_identity_sha256: str,
    step: DeploymentStep,
    state: str,
    evidence: Mapping[str, object],
    previous_record_sha256: str | None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_task13_staged_deployment_journal_v2",
        "sequence": sequence,
        "request_identity_sha256": request_identity_sha256,
        "step": step.value,
        "state": state,
        "evidence": dict(evidence),
        "previous_record_sha256": previous_record_sha256,
    }
    body["record_sha256"] = canonical_sha256(body)
    return body


def _open_locked_journal(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
        os.close(descriptor)
        raise StagedDeploymentError("staged journal is not one owner-only regular file")
    return descriptor


def _read_all(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 65536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _append_record(descriptor: int, record: Mapping[str, object]) -> None:
    raw = canonical_json_bytes(record) + b"\n"
    os.lseek(descriptor, 0, os.SEEK_END)
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written <= 0:
            raise OSError("staged journal append made no progress")
        offset += written
    os.fsync(descriptor)


def _load_journal(descriptor: int, request_identity_sha256: str) -> _JournalState:
    raw = _read_all(descriptor)
    if not raw:
        return _JournalState({}, None, None, 0, None)
    if not raw.endswith(b"\n"):
        raise StagedDeploymentError("staged journal has a torn record")
    committed: dict[DeploymentStep, Mapping[str, object]] = {}
    possible_step: DeploymentStep | None = None
    possible_evidence: Mapping[str, object] | None = None
    previous: str | None = None
    sequence = 0
    for line in raw.splitlines():
        try:
            row = json.loads(line.decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise StagedDeploymentError("staged journal contains malformed JSON") from exc
        if (
            type(row) is not dict
            or set(row) != _JOURNAL_FIELDS
            or canonical_json_bytes(row) != line
            or row["schema_version"] != 2
            or row["record_type"] != "glm52_task13_staged_deployment_journal_v2"
            or row["sequence"] != sequence
            or row["request_identity_sha256"] != request_identity_sha256
            or row["previous_record_sha256"] != previous
            or row["state"] not in {"POSSIBLY_SENT", "COMMITTED"}
            or type(row["evidence"]) is not dict
        ):
            raise StagedDeploymentError("staged journal chain or schema drifted")
        identity = row["record_sha256"]
        unsigned = dict(row)
        unsigned.pop("record_sha256")
        if _sha(identity, "journal record identity") != canonical_sha256(unsigned):
            raise StagedDeploymentError("staged journal record identity drifted")
        try:
            step = DeploymentStep(row["step"])
        except (TypeError, ValueError) as exc:
            raise StagedDeploymentError("staged journal step is foreign") from exc
        expected_step = tuple(DeploymentStep)[len(committed)]
        if step is not expected_step:
            raise StagedDeploymentError("staged journal steps are reordered")
        if row["state"] == "POSSIBLY_SENT":
            if step not in _MUTATING_STEPS or possible_step is not None:
                raise StagedDeploymentError("staged journal mutation state is incoherent")
            possible_step = step
            possible_evidence = row["evidence"]
        else:
            if step in _MUTATING_STEPS and possible_step is not step:
                raise StagedDeploymentError("mutation committed without possible-send seal")
            committed[step] = row["evidence"]
            possible_step = None
            possible_evidence = None
        previous = identity
        sequence += 1
    return _JournalState(
        committed,
        possible_step,
        possible_evidence,
        sequence,
        previous,
    )


def _invoke_fresh(
    step: DeploymentStep,
    request: StagedDeploymentRequest,
    operations: StagedDeploymentOperations,
    committed: Mapping[DeploymentStep, object],
) -> object:
    if step is DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED:
        return operations.deploy_retained_bootstrap_runtime(request)
    if step is DeploymentStep.BRIDGE_SEED_PUBLISHED:
        return operations.publish_bridge_seed(
            request,
            committed[DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED],
        )
    if step is DeploymentStep.BRIDGE_SEED_ESTABLISHED:
        return operations.establish_bridge_seed(
            request,
            committed[DeploymentStep.BRIDGE_SEED_PUBLISHED],
        )
    if step is DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6:
        return operations.complete_stack_migration_operations_1_to_6(
            request, committed[DeploymentStep.BRIDGE_SEED_ESTABLISHED]
        )
    if step is DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED:
        return operations.publish_bootstrap_fence_artifacts(
            request,
            committed[DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6],
            committed[DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED],
        )
    if step is DeploymentStep.RETAINED_FENCE_RUNTIME_DEPLOYED:
        return operations.deploy_retained_fence_runtime(
            request,
            committed[DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6],
            committed[DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED],
        )
    if step is DeploymentStep.PREPARE_EXECUTED_STABILIZED:
        return operations.execute_prepare(
            request,
            committed[DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6],
            committed[DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED],
        )
    if step in {
        DeploymentStep.SUPPORT_INPUT_SNAPSHOT_ONE,
        DeploymentStep.SUPPORT_INPUT_SNAPSHOT_TWO,
    }:
        return operations.collect_support_input_snapshot(
            request,
            committed[DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6],
            committed[DeploymentStep.PREPARE_EXECUTED_STABILIZED],
            committed[DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED],
        )
    if step is DeploymentStep.DISABLED_SUPPORT_DEPLOYED:
        return operations.deploy_disabled_support(
            request,
            committed[DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6],
            committed[DeploymentStep.SUPPORT_INPUT_SNAPSHOT_TWO],
        )
    if step is DeploymentStep.STACK_MIGRATION_OPERATION_7:
        return operations.complete_stack_migration_operation_7(
            request,
            committed[DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6],
            committed[DeploymentStep.PREPARE_EXECUTED_STABILIZED],
            committed[DeploymentStep.DISABLED_SUPPORT_DEPLOYED],
        )
    if step is DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED:
        return operations.commit_support_runtime_identity(
            request,
            committed[DeploymentStep.STACK_MIGRATION_OPERATION_7],
            committed[DeploymentStep.DISABLED_SUPPORT_DEPLOYED],
        )
    if step is DeploymentStep.NO_WORKER_ACTIVATION_PROVED:
        return operations.prove_no_worker_activation(
            request,
            committed[DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED],
        )
    raise AssertionError("unreachable staged deployment step")


def _rehydrate(step: DeploymentStep, value: Mapping[str, object]) -> object:
    if step is DeploymentStep.BRIDGE_SEED_PUBLISHED:
        from .fence_bootstrap_publication import (
            parse_bridge_seed_publication_v2,
        )

        return parse_bridge_seed_publication_v2(value)
    if step is DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED:
        return parse_bootstrap_fence_publication(value)
    if step in {
        DeploymentStep.SUPPORT_INPUT_SNAPSHOT_ONE,
        DeploymentStep.SUPPORT_INPUT_SNAPSHOT_TWO,
    }:
        return support_build_inputs_from_mapping(value)
    if step is DeploymentStep.DISABLED_SUPPORT_DEPLOYED:
        return parse_disabled_support_deployment_evidence(value)
    if step is DeploymentStep.PREPARE_EXECUTED_STABILIZED:
        from .fence_executor import parse_fence_execution_result

        return parse_fence_execution_result(value)
    if step is DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6:
        from .task13_migration_adapter import (
            parse_stack_migration_transfer_checkpoint_v2,
        )

        return parse_stack_migration_transfer_checkpoint_v2(value)
    if step is DeploymentStep.STACK_MIGRATION_OPERATION_7:
        from .task13_migration_adapter import (
            parse_stack_migration_operation_7_evidence_v2,
        )

        return parse_stack_migration_operation_7_evidence_v2(value)
    if step is DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED:
        from .support_plane import parse_support_runtime_identity

        return parse_support_runtime_identity(value)
    return dict(value)


def _materialize_aggregate_evidence(
    request: StagedDeploymentRequest,
    committed: Mapping[DeploymentStep, object],
    journal_sha256: str,
) -> tuple[Path, str]:
    output = Path(str(request.production_request.get("output_directory", "")))
    if (
        not output.is_absolute()
        or not output.is_dir()
        or output.is_symlink()
        or output.resolve(strict=True) != output
    ):
        raise StagedDeploymentError("aggregate output directory is not exact")
    path = output / "staged-infrastructure-evidence-v2.json"
    unsigned: dict[str, object] = {
        "schema_version": 2,
        "record_type": "glm52_task13_staged_infrastructure_evidence_v2",
        "activation_id": request.activation_id,
        "completed_steps": [step.value for step in DeploymentStep],
        "journal_sha256": journal_sha256,
        "retained_bootstrap_runtime_identity_sha256": _projection(
            DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED,
            committed[DeploymentStep.RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED],
        ).get("canonical_identity_sha256"),
        "bridge_seed_publication_identity_sha256": _projection(
            DeploymentStep.BRIDGE_SEED_PUBLISHED,
            committed[DeploymentStep.BRIDGE_SEED_PUBLISHED],
        ).get("canonical_identity_sha256"),
        "checkpoint_identity_sha256": _projection(
            DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6,
            committed[DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6],
        ).get("canonical_identity_sha256"),
        "bootstrap_publication_identity_sha256": _projection(
            DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED,
            committed[DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED],
        ).get("canonical_identity_sha256"),
        "retained_fence_runtime_identity_sha256": _projection(
            DeploymentStep.RETAINED_FENCE_RUNTIME_DEPLOYED,
            committed[DeploymentStep.RETAINED_FENCE_RUNTIME_DEPLOYED],
        ).get("canonical_identity_sha256"),
        "prepare_result_identity_sha256": _projection(
            DeploymentStep.PREPARE_EXECUTED_STABILIZED,
            committed[DeploymentStep.PREPARE_EXECUTED_STABILIZED],
        ).get("canonical_identity_sha256"),
        "support_build_inputs_identity_sha256": support_build_inputs_identity(
            committed[DeploymentStep.SUPPORT_INPUT_SNAPSHOT_TWO]
        ),
        "disabled_support_identity_sha256": _projection(
            DeploymentStep.DISABLED_SUPPORT_DEPLOYED,
            committed[DeploymentStep.DISABLED_SUPPORT_DEPLOYED],
        ).get("canonical_identity_sha256"),
        "operation_7_identity_sha256": _projection(
            DeploymentStep.STACK_MIGRATION_OPERATION_7,
            committed[DeploymentStep.STACK_MIGRATION_OPERATION_7],
        ).get("canonical_identity_sha256"),
        "runtime_identity_sha256": _projection(
            DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED,
            committed[DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED],
        ).get("canonical_identity_sha256"),
        "no_launch_evidence": _projection(
            DeploymentStep.NO_WORKER_ACTIVATION_PROVED,
            committed[DeploymentStep.NO_WORKER_ACTIVATION_PROVED],
        ),
        "worker_activation_allowed": False,
    }
    for field in (
        "retained_bootstrap_runtime_identity_sha256",
        "bridge_seed_publication_identity_sha256",
        "checkpoint_identity_sha256",
        "bootstrap_publication_identity_sha256",
        "retained_fence_runtime_identity_sha256",
        "prepare_result_identity_sha256",
        "support_build_inputs_identity_sha256",
        "disabled_support_identity_sha256",
        "operation_7_identity_sha256",
        "runtime_identity_sha256",
    ):
        _sha(unsigned[field], field)
    identity = canonical_sha256(unsigned)
    body = dict(unsigned)
    body["canonical_identity_sha256"] = identity
    raw = canonical_json_bytes(body) + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != raw:
            raise StagedDeploymentError("aggregate evidence already exists and differs")
        return path, identity
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY
        | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("aggregate evidence write made no progress")
            offset += written
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path, identity


def run_staged_deployment(
    request: StagedDeploymentRequest,
    operations: StagedDeploymentOperations,
) -> StagedDeploymentResult:
    """Execute or reconcile the exact v2 temporal spine without resend."""

    if type(request) is not StagedDeploymentRequest:
        raise StagedDeploymentError("staged deployment requires an exact v2 request")
    request_identity = _request_identity(request)
    descriptor = _open_locked_journal(request.journal_path)
    try:
        journal = _load_journal(descriptor, request_identity)
        committed: dict[DeploymentStep, object] = {}
        for step, stored in journal.committed.items():
            expected = _rehydrate(step, stored)
            adopted = operations.adopt_committed(
                step, request, stored, committed
            )
            observed = _validate_evidence(step, adopted)
            _validate_cross_step_evidence(step, observed, committed)
            if canonical_json_bytes(observed) != canonical_json_bytes(
                _projection(step, expected)
            ):
                raise StagedDeploymentError(
                    step.value + " committed evidence drifted during adoption"
                )
            committed[step] = adopted

        sequence = journal.next_sequence
        previous = journal.previous_record_sha256
        if journal.possible_send_step is not None:
            assert journal.possible_send_evidence is not None
            step = journal.possible_send_step
            reconciled = operations.reconcile_mutation(
                step,
                request,
                committed,
                journal.possible_send_evidence,
            )
            evidence = _validate_evidence(step, reconciled)
            _validate_cross_step_evidence(step, evidence, committed)
            record = _journal_record(
                sequence=sequence,
                request_identity_sha256=request_identity,
                step=step,
                state="COMMITTED",
                evidence=evidence,
                previous_record_sha256=previous,
            )
            _append_record(descriptor, record)
            committed[step] = reconciled
            sequence += 1
            previous = str(record["record_sha256"])

        for step in DeploymentStep:
            if step in committed:
                continue
            if step in _MUTATING_STEPS:
                seal = {
                    "operation_identity_sha256": canonical_sha256(
                        {
                            "request_identity_sha256": request_identity,
                            "step": step.value,
                            "committed_evidence_sha256": canonical_sha256(
                                {
                                    item.value: _projection(item, evidence)
                                    for item, evidence in committed.items()
                                }
                            ),
                        }
                    )
                }
                possible = _journal_record(
                    sequence=sequence,
                    request_identity_sha256=request_identity,
                    step=step,
                    state="POSSIBLY_SENT",
                    evidence=seal,
                    previous_record_sha256=previous,
                )
                _append_record(descriptor, possible)
                sequence += 1
                previous = str(possible["record_sha256"])
            fresh = _invoke_fresh(step, request, operations, committed)
            evidence = _validate_evidence(step, fresh)
            _validate_cross_step_evidence(step, evidence, committed)
            record = _journal_record(
                sequence=sequence,
                request_identity_sha256=request_identity,
                step=step,
                state="COMMITTED",
                evidence=evidence,
                previous_record_sha256=previous,
            )
            _append_record(descriptor, record)
            committed[step] = fresh
            sequence += 1
            previous = str(record["record_sha256"])

        if tuple(committed) != tuple(DeploymentStep):
            raise StagedDeploymentError("staged deployment did not complete its exact spine")
        final = _projection(
            DeploymentStep.NO_WORKER_ACTIVATION_PROVED,
            committed[DeploymentStep.NO_WORKER_ACTIVATION_PROVED],
        )
        journal_sha256 = hashlib.sha256(_read_all(descriptor)).hexdigest()
        aggregate_path, aggregate_identity = _materialize_aggregate_evidence(
            request,
            committed,
            journal_sha256,
        )
        return StagedDeploymentResult(
            activation_id=request.activation_id,
            completed_steps=tuple(committed),
            journal_sha256=journal_sha256,
            checkpoint=committed[
                DeploymentStep.STACK_MIGRATION_OPERATIONS_1_TO_6
            ],
            bootstrap_publication=committed[
                DeploymentStep.BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED
            ],
            prepare_result=committed[
                DeploymentStep.PREPARE_EXECUTED_STABILIZED
            ],
            support_inputs=committed[DeploymentStep.SUPPORT_INPUT_SNAPSHOT_TWO],
            disabled_support=committed[DeploymentStep.DISABLED_SUPPORT_DEPLOYED],
            operation_7_evidence=committed[
                DeploymentStep.STACK_MIGRATION_OPERATION_7
            ],
            runtime_identity=committed[
                DeploymentStep.SUPPORT_RUNTIME_IDENTITY_COMMITTED
            ],
            staged_infrastructure_evidence_path=aggregate_path,
            staged_infrastructure_identity_sha256=aggregate_identity,
            final_worker_count=int(final["worker_count"]),
            raw_ec2_launch_calls=int(final["raw_ec2_launch_calls"]),
            worker_activation_allowed=False,
        )
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = [
    "ACCOUNT_ID",
    "BootstrapFencePublication",
    "DeploymentStep",
    "DisabledSupportDeploymentEvidence",
    "FENCE_ROLE_ARN",
    "MODEL_BUCKET_NAME",
    "PROFILE",
    "REGION",
    "SUPPORT_ROLE_ARN",
    "StagedDeploymentError",
    "StagedDeploymentOperations",
    "StagedDeploymentRequest",
    "StagedDeploymentResult",
    "parse_bootstrap_fence_publication",
    "parse_disabled_support_deployment_evidence",
    "run_staged_deployment",
]
