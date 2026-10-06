from __future__ import annotations

import json
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_sha256
from glm52_enforcement.task13_staged_deployment import (
    DeploymentStep,
    StagedDeploymentError,
    StagedDeploymentRequest,
    _load_journal,
    _open_locked_journal,
    _validate_evidence,
    run_staged_deployment,
)

SHA_A = "a" * 64


def _request(tmp_path: Path) -> StagedDeploymentRequest:
    return StagedDeploymentRequest(
        schema_version=2,
        record_type="glm52_task13_staged_deployment_request_v2",
        activation_id="glm52-v2-amber-quartz",
        journal_path=(tmp_path / "staged-v2.jsonl").resolve(),
        production_request={
            "output_directory": str(tmp_path.resolve()),
            "sealed_request_sha256": SHA_A,
        },
    )


def _seed() -> dict[str, object]:
    return {
        "schema_version": 2,
        "record_type": "glm52_h1g_bridge_seed_established_v2",
        "policy_committed": True,
        "execution_eligible": True,
        "canonical_identity_sha256": SHA_A,
    }


class FailsAfterBootstrap:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def deploy_retained_bootstrap_runtime(self, request):
        del request
        self.calls.append("bootstrap-runtime")
        unsigned = {
            "schema_version": 2,
            "record_type": (
                "glm52_h1g_retained_bootstrap_runtime_deployment_v2"
            ),
            "retained_stack_id": (
                "arn:aws:cloudformation:us-west-2:246813579024:stack/"
                "keep-glm52-gpu/11111111-1111-4111-8111-111111111111"
            ),
            "template_sha256": "b" * 64,
            "materializer_function_version_arn": (
                "arn:aws:lambda:us-west-2:246813579024:function:"
                "keep-glm52-h1g-fence-bootstrap-materializer:1"
            ),
            "retained_update_identity_sha256": "d" * 64,
            "publisher_role_arn": (
                "arn:aws:iam::246813579024:role/"
                "keep-glm52-h1g-fence-bootstrap-artifact-publisher"
            ),
            "invoker_role_arn": (
                "arn:aws:iam::246813579024:role/"
                "keep-glm52-h1g-fence-bootstrap-materializer-invoker"
            ),
            "worker_activation_allowed": False,
            "source_action_allowed": False,
        }
        return {
            **unsigned,
            "canonical_identity_sha256": canonical_sha256(unsigned),
        }

    def publish_bridge_seed(self, request, bootstrap_runtime):
        del request, bootstrap_runtime
        self.calls.append("publish-seed")
        raise RuntimeError("stop after ordering proof")

    def reconcile_mutation(
        self,
        step,
        request,
        committed,
        possible_send_evidence,
    ):
        del request, committed, possible_send_evidence
        self.calls.append("reconcile:" + step.value)
        raise RuntimeError("stop after ordering proof")

    def adopt_committed(self, step, request, evidence, committed):
        del step, request, committed
        self.calls.append("adopt")
        return evidence


def test_temporal_spine_is_exact_and_operation_7_follows_disabled_support() -> None:
    assert tuple(step.value for step in DeploymentStep) == (
        "RETAINED_BOOTSTRAP_RUNTIME_DEPLOYED",
        "BRIDGE_SEED_PUBLISHED",
        "BRIDGE_SEED_ESTABLISHED",
        "STACK_MIGRATION_OPERATIONS_1_TO_6",
        "BOOTSTRAP_FENCE_ARTIFACTS_PUBLISHED",
        "RETAINED_FENCE_RUNTIME_DEPLOYED",
        "PREPARE_EXECUTED_STABILIZED",
        "SUPPORT_INPUT_SNAPSHOT_ONE",
        "SUPPORT_INPUT_SNAPSHOT_TWO",
        "DISABLED_SUPPORT_DEPLOYED",
        "STACK_MIGRATION_OPERATION_7",
        "SUPPORT_RUNTIME_IDENTITY_COMMITTED",
        "NO_WORKER_ACTIVATION_PROVED",
    )


def test_v1_staged_request_is_execution_ineligible(tmp_path: Path) -> None:
    with pytest.raises(StagedDeploymentError, match="v2"):
        StagedDeploymentRequest(
            schema_version=1,
            record_type="glm52_task13_staged_deployment_request_v1",
            activation_id="glm52-v2-amber-quartz",
            journal_path=(tmp_path / "old.jsonl").resolve(),
            production_request={"legacy": True},
        )


def test_seed_publication_journal_projection_normalizes_write_disposition() -> None:
    from types import SimpleNamespace

    from glm52_enforcement.fence_bootstrap_publication import (
        BridgeSeedPublicationV2,
    )
    from glm52_enforcement.task13_fixed_artifacts import (
        FixedKeyPublicationDisposition,
    )

    artifact = {"artifact": "seed"}
    materializer_arn = (
        "arn:aws:lambda:us-west-2:246813579024:function:"
        "keep-glm52-h1g-fence-bootstrap-materializer:1"
    )
    unsigned = {
        "schema_version": 2,
        "record_type": "glm52_h1g_bridge_seed_publication_v2",
        "artifact": artifact,
        "materializer_function_version_arn": materializer_arn,
        "disposition": "WRITTEN",
    }
    publication = BridgeSeedPublicationV2(
        artifact=SimpleNamespace(to_dict=lambda: artifact),
        materializer_function_version_arn=materializer_arn,
        disposition=FixedKeyPublicationDisposition.WRITTEN,
        canonical_identity_sha256=canonical_sha256(unsigned),
    )
    projection = _validate_evidence(
        DeploymentStep.BRIDGE_SEED_PUBLISHED,
        publication,
    )
    identity = projection.pop("canonical_identity_sha256")
    assert projection["disposition"] == "ADOPTED"
    assert identity == canonical_sha256(projection)


def test_possible_send_is_reconciled_without_bootstrap_redeploy(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    first = FailsAfterBootstrap()
    with pytest.raises(RuntimeError, match="ordering proof"):
        run_staged_deployment(request, first)
    assert first.calls == ["bootstrap-runtime", "publish-seed"]

    rows = [
        json.loads(line)
        for line in request.journal_path.read_text().splitlines()
    ]
    assert [row["state"] for row in rows] == [
        "POSSIBLY_SENT",
        "COMMITTED",
        "POSSIBLY_SENT",
    ]
    second = FailsAfterBootstrap()
    with pytest.raises(RuntimeError, match="ordering proof"):
        run_staged_deployment(request, second)
    assert second.calls[:2] == ["adopt", "reconcile:BRIDGE_SEED_PUBLISHED"]
    assert "bootstrap-runtime" not in second.calls


def test_journal_rejects_reordered_or_foreign_step(tmp_path: Path) -> None:
    request = _request(tmp_path)
    identity = canonical_sha256(
        {
            "schema_version": request.schema_version,
            "record_type": request.record_type,
            "activation_id": request.activation_id,
            "production_request": request.production_request,
        }
    )
    row = {
        "schema_version": 2,
        "record_type": "glm52_task13_staged_deployment_journal_v2",
        "sequence": 0,
        "request_identity_sha256": identity,
        "step": "STACK_MIGRATION_OPERATION_7",
        "state": "POSSIBLY_SENT",
        "evidence": {"operation_identity_sha256": SHA_A},
        "previous_record_sha256": None,
    }
    row["record_sha256"] = canonical_sha256(row)
    request.journal_path.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    request.journal_path.chmod(0o600)
    descriptor = _open_locked_journal(request.journal_path)
    try:
        with pytest.raises(StagedDeploymentError, match="reordered"):
            _load_journal(descriptor, identity)
    finally:
        import os

        os.close(descriptor)
