"""Task 13 deterministic, disabled campaign-package contracts."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.decision_closure import build_closure_request
from glm52_enforcement.task13_transport_gates import SEMANTIC_GATE_PINS
from glm52_task13_staged_fixture_support import (
    build_staged_infrastructure_evidence,
)

ACCOUNT = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
RUN_ID = "glm52-sky-20260724"
ACTIVATION = "approved-20260728"
COLLECTOR_ARN = (
    "arn:aws:lambda:us-west-2:246813579024:function:"
    "keep-glm52-h1g-rehearsal-collector:19"
)
COLLECTOR_VERSION = "19"
SHA_A = hashlib.sha256(b"task11-review").hexdigest()
SHA_B = hashlib.sha256(b"task12-review").hexdigest()
SHA_C = hashlib.sha256(b"retained-template").hexdigest()
SHA_D = hashlib.sha256(b"fence-template").hexdigest()
SHA_E = hashlib.sha256(b"support-template").hexdigest()
SHA_F = hashlib.sha256(b"h100-input").hexdigest()
SHA_G = hashlib.sha256(b"bootstrap-template").hexdigest()
SHA_H = hashlib.sha256(b"cache-seed-input").hexdigest()


def _task11_request() -> dict[str, object]:
    return asdict(
        build_closure_request(
            activation_id=ACTIVATION,
            generation=1,
            candidate_identity_sha256="9" * 64,
            initial_source_predecessor_version_id="3LgSourcePredecessor",
            admission_version_arn=(
                "arn:aws:lambda:us-west-2:246813579024:function:"
                "keep-glm52-h1g-launch-admission:17"
            ),
            numeric_binding_version_arn=(
                "arn:aws:lambda:us-west-2:246813579024:function:"
                "keep-glm52-h1g-numeric-binding:23"
            ),
            closure_budget_status="CLOSURE_BUDGET_PROVEN",
        )
    )


def _task11_boundary() -> dict[str, object]:
    return {
        "bucket": "keep-glm52-models-246813579024-us-west-2",
        "key": (
            "campaigns/glm52-sky-20260724/authorities/task11/"
            "approved-20260728/00000001/TASK11_BOUNDARY.json"
        ),
        "version_id": "3LgTask11BoundaryVersion",
        "file_sha256": "7" * 64,
        "body_sha256": "8" * 64,
    }


def _api():
    try:
        return importlib.import_module(
            "glm52_enforcement.task13_campaign_package"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "Task 13 campaign-package implementation is missing",
            pytrace=False,
        )


def _coordinate(
    kind: str,
    key: str,
    sha256: str,
) -> dict[str, object]:
    pin = SEMANTIC_GATE_PINS.get(kind)
    return {
        "artifact_kind": kind,
        "bucket": "keep-glm52-models-246813579024-us-west-2",
        "key": key,
        "version_id": "3LgExactImmutableVersion",
        "file_sha256": (
            hashlib.sha256((kind + "-file").encode()).hexdigest()
            if pin is None
            else pin["file_sha256"]
        ),
        "body_sha256": sha256 if pin is None else pin["body_sha256"],
    }


_STAGED_STEPS = [
    "ACCOUNT_LIVE_BASELINE",
    "BOOTSTRAP_INERT_ANCHORS",
    "PROVE_BUCKET_POLICY_ABSENT",
    "CREATE_FENCE_CHANGE_SET",
    "INSPECT_FENCE_CHANGE_SET",
    "EXECUTE_FENCE_UPDATE",
    "MATERIALIZE_PRE_SUPPORT",
    "UPDATE_RETAINED_PRE_SUPPORT",
    "MATERIALIZE_FULL_SUPPORT_INPUTS",
    "BUILD_PUBLISH_SUPPORT",
    "CAPTURE_PRECREATE_ORPHAN_AUTHORITY",
    "CREATE_SUPPORT_CHANGE_SET",
    "INSPECT_SUPPORT_CHANGE_SET",
    "EXECUTE_SUPPORT_CHANGE_SET",
    "MATERIALIZE_POSTCREATE_FRAGMENT",
    "UPDATE_RETAINED_FINAL",
    "POSTPUBLICATION_AUTHORITY",
    "FINAL_EXACT_READBACK",
    "PROVE_NO_WORKER_ACTIVATION",
]
RETAINED_STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:stack/"
    "keep-glm52-gpu/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
)
FENCE_STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:stack/"
    "keep-glm52-h1g-fence/bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
)
SUPPORT_STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:stack/"
    "keep-glm52-h1g-support/cccccccc-dddd-4eee-8fff-aaaaaaaaaaaa"
)
DEPLOYMENT_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/"
    "keep-glm52-h1g-cloudformation-deployment"
)


def _staged_evidence(
    artifacts: list[dict[str, object]],
) -> dict[str, object]:
    return build_staged_infrastructure_evidence(
        artifacts,
        activation_id=ACTIVATION,
        retained_stack_id=RETAINED_STACK_ID,
        fence_stack_id=FENCE_STACK_ID,
        support_stack_id=SUPPORT_STACK_ID,
    )

    # Retained below temporarily as fixture-shape documentation.
    by_kind = {row["artifact_kind"]: row for row in artifacts}
    fixed_artifacts = [
        by_kind[kind]
        for kind in sorted(
            {
                "BOOTSTRAP_TEMPLATE",
                "FENCE_TEMPLATE",
                "RETAINED_FOUNDATION_TEMPLATE",
                "RETAINED_PRE_SUPPORT_TEMPLATE",
                "RETAINED_TEMPLATE",
                "SUPPORT_TEMPLATE",
                "SUPPORT_INPUTS",
            }
        )
    ]
    support_archive_sha = hashlib.sha256(
        b"support-lambda-archive"
    ).hexdigest()
    layer_archive_sha = hashlib.sha256(
        b"cryptography-layer-archive"
    ).hexdigest()
    journal_path = (
        Path(__file__).resolve().parent
        / "fixtures/glm52_task13_staged_deployment_journal_v1.jsonl"
    )
    journal_raw = journal_path.read_bytes()
    journal_records = [
        json.loads(line) for line in journal_raw.splitlines()
    ]
    staged_request = {
        "schema_version": 1,
        "record_type": "glm52_task13_staged_deployment_request_v1",
        "activation_id": ACTIVATION,
        "journal_path": str(journal_path),
        "production_request": {
            "fixture_contract": "sealed-staged-route-v1",
        },
    }
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task13_staged_infrastructure_evidence_v1",
        "account_id": ACCOUNT,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "staged_request": staged_request,
        "staged_request_identity_sha256": hashlib.sha256(
            canonical_json_bytes(staged_request)
        ).hexdigest(),
        "staged_journal_path": str(journal_path),
        "staged_journal_size_bytes": len(journal_raw),
        "staged_journal_sha256": hashlib.sha256(
            journal_raw
        ).hexdigest(),
        "staged_journal_records": journal_records,
        "completed_steps": list(_STAGED_STEPS),
        "retained_foundation": {
            "stack_id": RETAINED_STACK_ID,
            "stack_status": "UPDATE_COMPLETE",
            "change_set_id": (
                "arn:aws:cloudformation:us-west-2:246813579024:changeSet/"
                "glm52-task13-retained-foundation-v1/foundation-uuid"
            ),
            "template_coordinate": by_kind[
                "RETAINED_FOUNDATION_TEMPLATE"
            ],
            "template_body_sha256": by_kind[
                "RETAINED_FOUNDATION_TEMPLATE"
            ]["body_sha256"],
            "readback_sha256": hashlib.sha256(
                b"retained-foundation-readback"
            ).hexdigest(),
        },
        "fence": {
            "stack_id": FENCE_STACK_ID,
            "stack_status": "UPDATE_COMPLETE",
            "change_set_type": "UPDATE",
            "policy_logical_id": "H1gProductionFenceBucketPolicy",
            "template_coordinate": by_kind["FENCE_TEMPLATE"],
            "template_body_sha256": by_kind["FENCE_TEMPLATE"][
                "body_sha256"
            ],
            "readback_sha256": hashlib.sha256(
                b"fence-readback"
            ).hexdigest(),
        },
        "pre_support": {
            "stack_id": RETAINED_STACK_ID,
            "stack_status": "UPDATE_COMPLETE",
            "change_set_type": "UPDATE",
            "role_arn": DEPLOYMENT_ROLE_ARN,
            "template_coordinate": by_kind[
                "RETAINED_PRE_SUPPORT_TEMPLATE"
            ],
            "retained_fragment_sha256": hashlib.sha256(
                b"pre-support-fragment"
            ).hexdigest(),
            "readback_sha256": hashlib.sha256(
                b"pre-support-readback"
            ).hexdigest(),
        },
        "support_stack": {
            "stack_id": SUPPORT_STACK_ID,
            "stack_status": "UPDATE_COMPLETE",
            "change_set_type": "UPDATE",
            "worker_activation_enabled": False,
            "template_coordinate": by_kind["SUPPORT_TEMPLATE"],
            "inputs_coordinate": by_kind["SUPPORT_INPUTS"],
            "support_lambda_archive": {
                "bucket": "keep-glm52-models-246813579024-us-west-2",
                "key": (
                    "task13/artifacts/support-lambda/"
                    + support_archive_sha
                    + ".zip"
                ),
                "version_id": "3LgSupportLambdaArchiveVersion",
                "size_bytes": 4096,
                "file_sha256": support_archive_sha,
            },
            "cryptography_layer_archive": {
                "bucket": "keep-glm52-models-246813579024-us-west-2",
                "key": (
                    "task13/artifacts/"
                    "cryptography-layer-python312-x86_64/"
                    + layer_archive_sha
                    + ".zip"
                ),
                "version_id": "3LgCryptographyLayerArchiveVersion",
                "size_bytes": 8192,
                "file_sha256": layer_archive_sha,
            },
            "cryptography_layer_version_arn": (
                "arn:aws:lambda:us-west-2:246813579024:layer:"
                "keep-glm52-h1g-cryptography-py312-x86-64:7"
            ),
            "cryptography_layer_code_sha256": base64.b64encode(
                bytes.fromhex(layer_archive_sha)
            ).decode("ascii"),
            "readback_sha256": hashlib.sha256(
                b"support-stack-readback"
            ).hexdigest(),
        },
        "postcreate_final": {
            "retained_stack_id": RETAINED_STACK_ID,
            "support_stack_id": SUPPORT_STACK_ID,
            "stack_status": "UPDATE_COMPLETE",
            "change_set_type": "UPDATE",
            "role_arn": DEPLOYMENT_ROLE_ARN,
            "template_coordinate": by_kind["RETAINED_TEMPLATE"],
            "retained_fragment_sha256": hashlib.sha256(
                b"postcreate-final-fragment"
            ).hexdigest(),
            "postpublication_authority_sha256": hashlib.sha256(
                b"postpublication-authority"
            ).hexdigest(),
            "readback_sha256": hashlib.sha256(
                b"postcreate-final-readback"
            ).hexdigest(),
        },
        "fixed_artifacts": copy.deepcopy(fixed_artifacts),
        "fixed_artifacts_identity_sha256": hashlib.sha256(
            canonical_json_bytes(fixed_artifacts)
        ).hexdigest(),
        "final_readback": {
            "retained_stack_id": RETAINED_STACK_ID,
            "fence_stack_id": FENCE_STACK_ID,
            "support_stack_id": SUPPORT_STACK_ID,
            "retained_stack_status": "UPDATE_COMPLETE",
            "fence_stack_status": "UPDATE_COMPLETE",
            "support_stack_status": "UPDATE_COMPLETE",
            "retained_template_sha256": by_kind["RETAINED_TEMPLATE"][
                "body_sha256"
            ],
            "fence_template_sha256": by_kind["FENCE_TEMPLATE"][
                "body_sha256"
            ],
            "support_template_sha256": by_kind["SUPPORT_TEMPLATE"][
                "body_sha256"
            ],
            "pending_change_sets": 0,
            "active_p5_instance_ids": [],
            "worker_activation_attempts": 0,
            "raw_ec2_launch_calls": 0,
        },
    }
    body["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    return body


def _rehash_staged_evidence(
    evidence: dict[str, object],
) -> None:
    staged_request = evidence["staged_request"]
    assert type(staged_request) is dict
    request_identity = hashlib.sha256(
        canonical_json_bytes(staged_request)
    ).hexdigest()
    evidence["staged_request_identity_sha256"] = request_identity
    records = evidence["staged_journal_records"]
    assert type(records) is list
    previous = None
    for sequence, row in enumerate(records, 1):
        assert type(row) is dict
        row["sequence"] = sequence
        row["request_identity_sha256"] = request_identity
        row["previous_record_sha256"] = previous
        unsigned_row = dict(row)
        unsigned_row.pop("record_sha256", None)
        row["record_sha256"] = hashlib.sha256(
            canonical_json_bytes(unsigned_row)
        ).hexdigest()
        previous = row["record_sha256"]
    journal_raw = b"".join(
        canonical_json_bytes(row) + b"\n" for row in records
    )
    evidence["staged_journal_size_bytes"] = len(journal_raw)
    evidence["staged_journal_sha256"] = hashlib.sha256(
        journal_raw
    ).hexdigest()
    unsigned = dict(evidence)
    unsigned.pop("canonical_identity_sha256", None)
    evidence["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()


def _production_request(
    *,
    clean_coordinate: object = None,
) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "record_type": "glm52_task13_campaign_package_request_v1",
        "account_id": ACCOUNT,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "collector_version_arn": COLLECTOR_ARN,
        "task11_request": _task11_request(),
        "task11_boundary": _task11_boundary(),
        "retained_stack_id": RETAINED_STACK_ID,
        "fence_stack_name": "keep-glm52-h1g-fence",
        "support_stack_name": "keep-glm52-h1g-support",
        "fence_change_set_name": "glm52-task13-fence-disabled-0001",
        "support_change_set_name": "glm52-task13-support-disabled-0001",
        "retained_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-cloudformation-deployment"
        ),
        "fence_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-fence-service"
        ),
        "support_role_arn": (
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-cloudformation-deployment"
        ),
        "monitor_descriptor_path": (
            "/tmp/glm52-full-run-20260729/production/"
            "campaign-descriptor-v2.json"
        ),
        "artifacts": [
            _coordinate(
                "TASK11_REVIEW_APPROVAL",
                "reviews/task11/approval.json",
                SHA_A,
            ),
            _coordinate(
                "TASK12_REVIEW_APPROVAL",
                "reviews/task12/approval.json",
                SHA_B,
            ),
            _coordinate(
                "RETAINED_FOUNDATION_TEMPLATE",
                "task13/templates/retained-foundation.yaml",
                hashlib.sha256(b"retained-foundation-template").hexdigest(),
            ),
            _coordinate(
                "RETAINED_PRE_SUPPORT_TEMPLATE",
                "task13/templates/retained-pre-support.yaml",
                hashlib.sha256(b"retained-pre-support-template").hexdigest(),
            ),
            _coordinate(
                "RETAINED_TEMPLATE",
                "task13/templates/retained.yaml",
                SHA_C,
            ),
            _coordinate(
                "FENCE_TEMPLATE",
                "task13/migration/fence-transfer.json",
                SHA_D,
            ),
            _coordinate(
                "SUPPORT_TEMPLATE",
                "task13/templates/support-disabled.yaml",
                SHA_E,
            ),
            _coordinate(
                "SUPPORT_INPUTS",
                "task13/inputs/support-build-inputs.json",
                hashlib.sha256(b"SUPPORT_INPUTS").hexdigest(),
            ),
            _coordinate(
                "H100_QUALIFICATION_INPUT",
                (
                    f"task13/activations/{ACTIVATION}/"
                    "qualification/h100-input.json"
                ),
                SHA_F,
            ),
            _coordinate(
                "BOOTSTRAP_TEMPLATE",
                "task13/templates/container-bootstrap-v1.json",
                SHA_G,
            ),
            _coordinate(
                "QUALIFICATION_CACHE_SEED_INPUT",
                (
                    f"task13/activations/{ACTIVATION}/"
                    "qualification/cache-seed-input.json"
                ),
                SHA_H,
            ),
            *[
                _coordinate(kind, key, hashlib.sha256(kind.encode()).hexdigest())
                for kind, key in (
                    ("T01_T25_GATE", "task13/gates/t01-t25.json"),
                    (
                        "TRANSPORT_22_MUTANT_GATE",
                        "task13/gates/transport-22-mutants.json",
                    ),
                    (
                        "REPOSITORY_ARCHIVE",
                        f"task13/activations/{ACTIVATION}/archive/repo-tar.json",
                    ),
                    ("ACCEPTED_BASELINE", "task13/inputs/accepted-baseline.json"),
                    ("PROMPT_PACK", "task13/inputs/prompt-pack.json"),
                    (
                        "TRAINING_CONFIGURATION",
                        "task13/inputs/training-configuration.json",
                    ),
                    ("GPU_SPEND_APPROVAL", "task13/approvals/gpu-spend.json"),
                    ("SUPPORT_APPROVAL", "task13/approvals/support-plane.json"),
                    (
                        "RESIDUAL_LIABILITY_APPROVAL",
                        "task13/approvals/residual-liability.json",
                    ),
                    (
                        "PRODUCTION_DESCRIPTOR",
                        (
                            f"task13/activations/{ACTIVATION}/"
                            "inputs/campaign-descriptor-v2.json"
                        ),
                    ),
                    (
                        "TASK10_PRODUCTION_AUTHORITY",
                        "task13/production/task10-production-authority.json",
                    ),
                    (
                        "TASK10_WORKER_DESCRIPTOR",
                        "task13/production/task10-worker-descriptor.json",
                    ),
                    (
                        "TASK10_TASK_INPUTS",
                        "task13/production/task10-task-inputs.json",
                    ),
                )
            ],
            *(
                [copy.deepcopy(clean_coordinate)]
                if clean_coordinate is not None
                else []
            ),
        ],
    }
    payload["staged_infrastructure_evidence"] = _staged_evidence(
        payload["artifacts"]
    )
    return payload


def _prequalification_request() -> dict[str, object]:
    payload = _production_request()
    payload["artifacts"] = [
        item
        for item in payload["artifacts"]
        if item["artifact_kind"]
        not in {
            "CLEAN_REHEARSAL",
            "TASK10_PRODUCTION_AUTHORITY",
            "TASK10_WORKER_DESCRIPTOR",
            "TASK10_TASK_INPUTS",
        }
    ]
    payload["staged_infrastructure_evidence"] = _staged_evidence(
        payload["artifacts"]
    )
    return payload


def request() -> dict[str, object]:
    return _prequalification_request()


def _production_successor_fixture() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    from glm52_enforcement.task13_clean_rehearsal import (
        clean_rehearsal_coordinate,
    )
    from test_glm52_task13_clean_rehearsal import _build, _fixture

    fixture = _fixture()
    evidence = _build(fixture)
    clean_coordinate = clean_rehearsal_coordinate(
        evidence=evidence,
        version_id="clean-evidence-version",
    )
    production_request = _production_request(
        clean_coordinate=clean_coordinate,
    )
    production_request["artifacts"] = [
        (
            fixture["repository_archive_coordinate"]
            if row["artifact_kind"] == "REPOSITORY_ARCHIVE"
            else row
        )
        for row in production_request["artifacts"]
    ]
    production_request["staged_infrastructure_evidence"] = (
        _staged_evidence(production_request["artifacts"])
    )
    return fixture["predecessor_package"], production_request, evidence


def _production_package() -> dict[str, object]:
    api = _api()
    predecessor, production_request, evidence = (
        _production_successor_fixture()
    )
    return api.build_campaign_package(
        production_request,
        predecessor_package=predecessor,
        predecessor_reviewed_artifacts=predecessor["reviewed_artifacts"],
        clean_rehearsal_evidence=evidence,
    )


def test_transport_gate_coordinates_are_semantic_observation_pins() -> None:
    """Break caught: Task 13 packages accept label-only transport gates."""

    api = _api()
    package = api.build_campaign_package(_prequalification_request())
    assert package["semantic_transport_gates"] == SEMANTIC_GATE_PINS

    wrong = _prequalification_request()
    gate = next(
        row
        for row in wrong["artifacts"]
        if row["artifact_kind"] == "T01_T25_GATE"
    )
    gate["body_sha256"] = "f" * 64
    with pytest.raises(
        api.CampaignPackageError,
        match="semantic transport gate",
    ):
        api.build_campaign_package(wrong)


def test_two_generation_package_contract_is_immutable_and_additive() -> None:
    """Break caught: H100 and launch share one mutable package identity."""

    api = _api()
    prequalification, production_request, evidence = (
        _production_successor_fixture()
    )
    assert prequalification["package_phase"] == "PREQUALIFICATION"
    assert prequalification["predecessor_identity"] is None
    assert len(prequalification["reviewed_artifacts"]) == 21
    assert "CLEAN_REHEARSAL" not in {
        row["artifact_kind"]
        for row in prequalification["reviewed_artifacts"]
    }

    production = api.build_campaign_package(
        production_request,
        predecessor_package=prequalification,
        predecessor_reviewed_artifacts=prequalification[
            "reviewed_artifacts"
        ],
        clean_rehearsal_evidence=evidence,
    )
    assert production["package_phase"] == "PRODUCTION"
    assert len(production["reviewed_artifacts"]) == 25
    assert production["predecessor_identity"] == {
        "package_identity_sha256": prequalification[
            "canonical_identity_sha256"
        ],
        "reviewed_artifacts_identity_sha256": hashlib.sha256(
            canonical_json_bytes(prequalification["reviewed_artifacts"])
        ).hexdigest(),
        "clean_rehearsal_evidence": evidence,
    }
    assert {
        row["artifact_kind"]
        for row in production["reviewed_artifacts"]
    } - {
        row["artifact_kind"]
        for row in prequalification["reviewed_artifacts"]
    } == set(api.PRODUCTION_ONLY_ARTIFACT_KINDS)
    predecessor_by_kind = {
        row["artifact_kind"]: row
        for row in prequalification["reviewed_artifacts"]
    }
    production_by_kind = {
        row["artifact_kind"]: row
        for row in production["reviewed_artifacts"]
    }
    assert all(
        production_by_kind[kind] == coordinate
        for kind, coordinate in predecessor_by_kind.items()
    )


def test_prequalification_rejects_every_clean_rehearsal_coordinate() -> None:
    api = _api()
    _, _, evidence = _production_successor_fixture()
    from glm52_enforcement.task13_clean_rehearsal import (
        clean_rehearsal_coordinate,
    )

    prequalification_request = _prequalification_request()
    prequalification_request["artifacts"].append(
        clean_rehearsal_coordinate(
            evidence=evidence,
            version_id="clean-evidence-version",
        )
    )
    prequalification_request["staged_infrastructure_evidence"] = (
        _staged_evidence(prequalification_request["artifacts"])
    )

    with pytest.raises(api.CampaignPackageError, match="artifact kinds"):
        api.build_campaign_package(prequalification_request)


def test_production_rejects_legacy_clean_rehearsal_key() -> None:
    api = _api()
    predecessor, production_request, evidence = (
        _production_successor_fixture()
    )
    clean = next(
        row
        for row in production_request["artifacts"]
        if row["artifact_kind"] == "CLEAN_REHEARSAL"
    )
    clean["key"] = "task13/gates/clean-rehearsal.json"

    with pytest.raises(api.CampaignPackageError, match="artifact key"):
        api.build_campaign_package(
            production_request,
            predecessor_package=predecessor,
            predecessor_reviewed_artifacts=predecessor[
                "reviewed_artifacts"
            ],
            clean_rehearsal_evidence=evidence,
        )


def test_production_successor_requires_clean_rehearsal_evidence() -> None:
    api = _api()
    predecessor, production_request, _ = _production_successor_fixture()

    with pytest.raises(
        api.CampaignPackageError,
        match="clean rehearsal",
    ):
        api.build_campaign_package(
            production_request,
            predecessor_package=predecessor,
            predecessor_reviewed_artifacts=predecessor[
                "reviewed_artifacts"
            ],
        )


def test_production_successor_adds_only_proven_clean_rehearsal() -> None:
    api = _api()
    predecessor, production_request, evidence = (
        _production_successor_fixture()
    )

    production = api.build_campaign_package(
        production_request,
        predecessor_package=predecessor,
        predecessor_reviewed_artifacts=predecessor[
            "reviewed_artifacts"
        ],
        clean_rehearsal_evidence=evidence,
    )

    assert production["predecessor_identity"][
        "clean_rehearsal_evidence"
    ] == evidence
    assert api.canonical_campaign_package_bytes(production) == (
        canonical_json_bytes(production) + b"\n"
    )
    predecessor_by_kind = {
        row["artifact_kind"]: row
        for row in predecessor["reviewed_artifacts"]
    }
    production_by_kind = {
        row["artifact_kind"]: row
        for row in production["reviewed_artifacts"]
    }
    assert {
        kind
        for kind in predecessor_by_kind
        if production_by_kind[kind] != predecessor_by_kind[kind]
    } == set()
    assert set(production_by_kind) - set(predecessor_by_kind) == {
        "CLEAN_REHEARSAL",
        "TASK10_PRODUCTION_AUTHORITY",
        "TASK10_WORKER_DESCRIPTOR",
        "TASK10_TASK_INPUTS",
    }

    wrong = copy.deepcopy(evidence)
    wrong["predecessor_package"]["package_identity_sha256"] = "f" * 64
    unsigned = dict(wrong)
    unsigned.pop("canonical_identity_sha256")
    wrong["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(api.CampaignPackageError, match="lineage|proof"):
        api.build_campaign_package(
            production_request,
            predecessor_package=predecessor,
            predecessor_reviewed_artifacts=predecessor[
                "reviewed_artifacts"
            ],
            clean_rehearsal_evidence=wrong,
        )


def test_fresh_prequalification_excludes_every_post_h100_coordinate() -> None:
    """Break caught: a fresh run requires Task10 outputs from its own future."""

    api = _api()
    package = api.build_campaign_package(_prequalification_request())
    kinds = {
        row["artifact_kind"] for row in package["reviewed_artifacts"]
    }
    staged_kinds = {
        row["artifact_kind"]
        for row in package["disabled_deployment"][
            "staged_infrastructure_evidence"
        ]["fixed_artifacts"]
    }

    assert kinds.isdisjoint(api.PRODUCTION_ONLY_ARTIFACT_KINDS)
    assert staged_kinds == set(
        api.STAGED_INFRASTRUCTURE_ARTIFACT_KINDS
    )
    assert {
        "QUALIFICATION_CACHE_SEED_INPUT",
        "H100_QUALIFICATION_INPUT",
    }.issubset(kinds)
    assert {
        row["artifact_kind"]
        for row in package["production_retry_plan"][
            "immutable_inputs"
        ]
    }.isdisjoint(api.PRODUCTION_ONLY_ARTIFACT_KINDS)


def test_serialized_production_rejects_coherent_predecessor_substitution() -> None:
    """Break caught: two SHA labels replace the exact predecessor package."""

    api = _api()
    production = _production_package()
    production["predecessor_identity"][
        "package_identity_sha256"
    ] = "0" * 64
    production["predecessor_identity"][
        "reviewed_artifacts_identity_sha256"
    ] = "1" * 64
    unsigned = dict(production)
    unsigned.pop("canonical_identity_sha256")
    production["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()

    with pytest.raises(
        api.CampaignPackageError,
        match="predecessor",
    ):
        api.canonical_campaign_package_bytes(production)


def test_package_rejects_coherently_rehashed_staged_journal_forgery() -> None:
    """Break caught: a self-hashed 19-step label list replaces route truth."""

    api = _api()
    forged = _prequalification_request()
    evidence = forged["staged_infrastructure_evidence"]
    records = evidence["staged_journal_records"]
    records[0]["evidence"]["worker_activation_attempts"] = 1
    previous = None
    for sequence, row in enumerate(records, 1):
        row["sequence"] = sequence
        row["previous_record_sha256"] = previous
        unsigned_row = dict(row)
        unsigned_row.pop("record_sha256")
        row["record_sha256"] = hashlib.sha256(
            canonical_json_bytes(unsigned_row)
        ).hexdigest()
        previous = row["record_sha256"]
    journal_raw = b"".join(
        canonical_json_bytes(row) + b"\n" for row in records
    )
    evidence["staged_journal_size_bytes"] = len(journal_raw)
    evidence["staged_journal_sha256"] = hashlib.sha256(
        journal_raw
    ).hexdigest()
    unsigned_evidence = dict(evidence)
    unsigned_evidence.pop("canonical_identity_sha256")
    evidence["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned_evidence)
    ).hexdigest()

    with pytest.raises(
        api.CampaignPackageError,
        match="journal activated a worker",
    ):
        api.build_campaign_package(forged)


def test_package_rejects_coherent_foreign_staged_production_request() -> None:
    """Break caught: the journal authenticates an unrelated staged route."""

    api = _api()
    forged = _prequalification_request()
    evidence = forged["staged_infrastructure_evidence"]
    evidence["staged_request"]["production_request"] = {
        "foreign_route": True,
    }
    _rehash_staged_evidence(evidence)

    with pytest.raises(
        api.CampaignPackageError,
        match="production request ancestry",
    ):
        api.build_campaign_package(forged)


def test_package_rejects_outer_summary_spliced_from_another_journal() -> None:
    """Break caught: self-hashed outer summaries do not match journal rows."""

    api = _api()
    forged = _prequalification_request()
    evidence = forged["staged_infrastructure_evidence"]
    evidence["fence"]["readback_sha256"] = "0" * 64
    unsigned = dict(evidence)
    unsigned.pop("canonical_identity_sha256")
    evidence["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()

    with pytest.raises(
        api.CampaignPackageError,
        match="journal summary",
    ):
        api.build_campaign_package(forged)


def test_serialized_successor_rejects_coherent_monitor_mutation() -> None:
    """Break caught: successor changes executable behavior after H100."""

    api = _api()
    forged = _production_package()
    forged["monitor_contract"]["descriptor_path"] = (
        "/tmp/foreign/campaign-descriptor-v2.json"
    )
    forged["monitor_contract"]["environment"][
        "CAMPAIGN_DESCRIPTOR"
    ] = "/tmp/foreign/campaign-descriptor-v2.json"
    unsigned = dict(forged)
    unsigned.pop("canonical_identity_sha256")
    forged["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()

    with pytest.raises(
        api.CampaignPackageError,
        match="differs outside",
    ):
        api.canonical_campaign_package_bytes(forged)


def test_serialized_successor_pins_retry_inputs_to_reviewed_artifacts() -> None:
    api = _api()
    forged = _production_package()
    immutable_inputs = forged["production_retry_plan"]["immutable_inputs"]
    repository_archive_index = next(
        index
        for index, row in enumerate(immutable_inputs)
        if row["artifact_kind"] == "REPOSITORY_ARCHIVE"
    )
    repository_archive = dict(immutable_inputs[repository_archive_index])
    repository_archive["version_id"] = "foreign-archive-version"
    immutable_inputs[repository_archive_index] = repository_archive
    unsigned = dict(forged)
    unsigned.pop("canonical_identity_sha256")
    forged["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()

    with pytest.raises(
        api.CampaignPackageError,
        match="immutable inputs drifted",
    ):
        api.canonical_campaign_package_bytes(forged)


def test_serialized_successor_rejects_unknown_retry_input_kind() -> None:
    api = _api()
    forged = _production_package()
    forged["production_retry_plan"]["immutable_inputs"][0][
        "artifact_kind"
    ] = "FOREIGN_AUTHORITY"
    unsigned = dict(forged)
    unsigned.pop("canonical_identity_sha256")
    forged["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()

    with pytest.raises(
        api.CampaignPackageError,
        match="artifact kind is not allowed",
    ):
        api.canonical_campaign_package_bytes(forged)

@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_serialized_successor_requires_exact_production_retry_inputs(
    mutation: str,
) -> None:
    api = _api()
    forged = _production_package()
    immutable_inputs = forged["production_retry_plan"]["immutable_inputs"]
    production_authority = next(
        row
        for row in immutable_inputs
        if row["artifact_kind"] == "TASK10_PRODUCTION_AUTHORITY"
    )
    if mutation == "missing":
        immutable_inputs.remove(production_authority)
    else:
        immutable_inputs.append(dict(production_authority))
    unsigned = dict(forged)
    unsigned.pop("canonical_identity_sha256")
    forged["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()

    with pytest.raises(
        api.CampaignPackageError,
        match="immutable inputs drifted",
    ):
        api.canonical_campaign_package_bytes(forged)



def test_production_successor_rejects_wrong_or_missing_h100_authority() -> None:
    """Break caught: an unrelated package is nominated as predecessor."""

    api = _api()
    prequalification, production_request, evidence = (
        _production_successor_fixture()
    )
    with pytest.raises(
        api.CampaignPackageError,
        match="predecessor",
    ):
        api.build_campaign_package(
            production_request,
            clean_rehearsal_evidence=evidence,
        )

    wrong = copy.deepcopy(prequalification)
    wrong["reviewed_artifacts"][0]["body_sha256"] = "f" * 64
    unhashed = dict(wrong)
    del unhashed["canonical_identity_sha256"]
    wrong["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(unhashed)
    ).hexdigest()
    with pytest.raises(
        api.CampaignPackageError,
        match="predecessor",
    ):
        api.build_campaign_package(
            production_request,
            predecessor_package=wrong,
            predecessor_reviewed_artifacts=prequalification[
                "reviewed_artifacts"
            ],
            clean_rehearsal_evidence=evidence,
        )


def test_production_successor_rejects_non_artifact_predecessor_drift() -> None:
    """Break caught: successor adopts H100 proof from a different collector."""

    api = _api()
    _, production_request, evidence = _production_successor_fixture()
    predecessor_request = _prequalification_request()
    production_archive = next(
        row
        for row in production_request["artifacts"]
        if row["artifact_kind"] == "REPOSITORY_ARCHIVE"
    )
    predecessor_request["artifacts"] = [
        production_archive
        if row["artifact_kind"] == "REPOSITORY_ARCHIVE"
        else row
        for row in predecessor_request["artifacts"]
    ]
    predecessor_request["staged_infrastructure_evidence"] = _staged_evidence(
        predecessor_request["artifacts"]
    )
    predecessor_request["collector_version_arn"] = (
        "arn:aws:lambda:us-west-2:246813579024:function:"
        "keep-glm52-h1g-rehearsal-collector:2"
    )
    predecessor = api.build_campaign_package(predecessor_request)
    with pytest.raises(
        api.CampaignPackageError,
        match="differs by more than",
    ):
        api.build_campaign_package(
            production_request,
            predecessor_package=predecessor,
            predecessor_reviewed_artifacts=predecessor[
                "reviewed_artifacts"
            ],
            clean_rehearsal_evidence=evidence,
        )


def test_builds_exact_twenty_run_rehearsal_and_single_finalize() -> None:
    """Break caught: deployment rehearsal uses a fixture-shaped run set."""

    package = _production_package()
    rehearsal = package["rehearsal"]
    invokes = rehearsal["collect_invocations"]
    assert len(invokes) == 20
    assert [row["event"]["measurement_id"] for row in invokes] == [
        f"measurement-{index:02d}" for index in range(1, 21)
    ]
    assert [row["event"]["scenario"] for row in invokes] == [
        "THROTTLING",
        "PAGINATION",
        "NETWORK_AMBIGUITY",
        *(["NONE"] * 17),
    ]
    assert all(
        row["function_version_arn"] == COLLECTOR_ARN
        and row["qualifier"] == COLLECTOR_VERSION
        and set(row["event"])
        == {
            "schema_version",
            "record_type",
            "activation_id",
            "measurement_id",
            "scenario",
            "task11_request",
            "task11_boundary",
        }
        for row in invokes
    )
    assert all(
        row["event"]["task11_request"] == _task11_request()
        and row["event"]["task11_boundary"] == _task11_boundary()
        for row in invokes
    )
    assert rehearsal["batches"] == [
        {
            "batch_id": "cold-start-01",
            "concurrency": 5,
            "measurement_ids": [
                "measurement-01",
                "measurement-02",
                "measurement-03",
                "measurement-04",
                "measurement-05",
            ],
            "required_distinct_cold_environments": 5,
        },
        {
            "batch_id": "remaining-02",
            "concurrency": 1,
            "measurement_ids": [
                f"measurement-{index:02d}" for index in range(6, 21)
            ],
            "required_distinct_cold_environments": 0,
        },
    ]
    assert rehearsal["finalize_invocation"] == {
        "function_version_arn": COLLECTOR_ARN,
        "qualifier": COLLECTOR_VERSION,
        "event": {
            "schema_version": 1,
            "record_type": "glm52_task11_finalize_rehearsal_gate_v1",
            "activation_id": ACTIVATION,
        },
    }


def test_package_is_canonical_self_identifying_and_no_execute() -> None:
    """Break caught: generation order changes bytes or grants launch authority."""

    api = _api()
    first = api.build_campaign_package(request())
    second = api.build_campaign_package(copy.deepcopy(request()))
    assert first == second
    assert api.canonical_campaign_package_bytes(first).endswith(b"\n")
    assert json.loads(api.canonical_campaign_package_bytes(first)) == first
    identity = first["canonical_identity_sha256"]
    unhashed = dict(first)
    del unhashed["canonical_identity_sha256"]
    assert identity == hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest()
    assert first["execution_authority"] == {
        "mode": "STAGED_INFRASTRUCTURE_ADOPTED_NO_WORKER",
        "live_aws_executed": True,
        "change_set_execution_authorized": False,
        "change_set_execution_requires_scoped_authority": False,
        "qualification_cache_seed_authorized": False,
        "h100_qualification_authorized": False,
        "production_launch_authorized": False,
    }


def test_package_binds_repository_archive_to_exact_activation() -> None:
    payload = request()
    archive = next(
        row
        for row in payload["artifacts"]
        if row["artifact_kind"] == "REPOSITORY_ARCHIVE"
    )
    archive["key"] = (
        "task13/activations/approved-20260729/archive/repo-tar.json"
    )

    with pytest.raises(
        _api().CampaignPackageError,
        match="repository archive activation lineage drifted",
    ):
        _api().build_campaign_package(payload)

@pytest.mark.parametrize(
    "artifact_kind",
    (
        "QUALIFICATION_CACHE_SEED_INPUT",
        "H100_QUALIFICATION_INPUT",
        "PRODUCTION_DESCRIPTOR",
    ),
)
def test_package_rejects_cross_activation_mutable_artifact(
    artifact_kind: str,
) -> None:
    payload = request()
    coordinate = next(
        row
        for row in payload["artifacts"]
        if row["artifact_kind"] == artifact_kind
    )
    coordinate["key"] = str(coordinate["key"]).replace(
        ACTIVATION,
        "approved-20260729",
    )

    with pytest.raises(
        _api().CampaignPackageError,
        match=artifact_kind + " activation lineage drifted",
    ):
        _api().build_campaign_package(payload)


@pytest.mark.parametrize(
    "artifact_kind",
    (
        "QUALIFICATION_CACHE_SEED_INPUT",
        "H100_QUALIFICATION_INPUT",
        "PRODUCTION_DESCRIPTOR",
    ),
)
def test_serialized_package_rejects_cross_activation_mutable_artifact(
    artifact_kind: str,
) -> None:
    api = _api()
    package = _production_package()
    coordinate = next(
        row
        for row in package["reviewed_artifacts"]
        if row["artifact_kind"] == artifact_kind
    )
    coordinate["key"] = str(coordinate["key"]).replace(
        ACTIVATION,
        "approved-20260729",
    )
    unhashed = dict(package)
    del unhashed["canonical_identity_sha256"]
    package["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(unhashed)
    ).hexdigest()

    with pytest.raises(
        api.CampaignPackageError,
        match=artifact_kind + " activation lineage drifted",
    ):
        api.canonical_campaign_package_bytes(package)


def test_package_adopts_staged_infrastructure_and_contains_no_deploy_route() -> None:
    """Break caught: the campaign package is built before its stack anchors."""

    package = _api().build_campaign_package(request())
    deployment = package["disabled_deployment"]

    assert deployment["state"] == "ADOPTED_DISABLED"
    assert deployment["staged_infrastructure_evidence"] == request()[
        "staged_infrastructure_evidence"
    ]
    assert "bootstrap" not in deployment
    assert "create_change_set_commands" not in deployment
    assert "execute_change_sets" not in deployment
    assert "task12_orphan_authority" not in deployment
    assert package["execution_authority"]["live_aws_executed"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda evidence: evidence["retained_foundation"].update(
                {"readback_sha256": "f" * 64}
            ),
            "staged infrastructure identity",
        ),
        (
            lambda evidence: evidence["support_stack"].update(
                {
                    "template_coordinate": evidence["fence"][
                        "template_coordinate"
                    ]
                }
            ),
            "staged infrastructure identity",
        ),
        (
            lambda evidence: evidence["completed_steps"].remove(
                "UPDATE_RETAINED_FINAL"
            ),
            "staged infrastructure identity",
        ),
        (
            lambda evidence: evidence["final_readback"].update(
                {"active_p5_instance_ids": ["i-0123456789abcdef0"]}
            ),
            "staged infrastructure identity",
        ),
    ],
)
def test_staged_infrastructure_evidence_is_self_identifying_and_cross_bound(
    mutate,
    message: str,
) -> None:
    """Break caught: label-only stack success is accepted as deployment proof."""

    payload = request()
    mutate(payload["staged_infrastructure_evidence"])
    with pytest.raises(_api().CampaignPackageError, match=message):
        _api().build_campaign_package(payload)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("bucket", "foreign-campaign-bucket", "artifact bucket"),
        ("key", "task13/templates/foreign-retained.yaml", "artifact key"),
        ("version_id", "latest", "immutable version"),
        ("file_sha256", "not-a-sha256", "file_sha256"),
        ("body_sha256", "not-a-sha256", "body_sha256"),
    ],
)
def test_canonical_package_rejects_coherently_rehashed_foreign_coordinate(
    field: str,
    replacement: str,
    message: str,
) -> None:
    """Break caught: a self-consistent package swaps a reviewed artifact."""

    api = _api()
    package = _production_package()
    coordinate = next(
        row
        for row in package["reviewed_artifacts"]
        if row["artifact_kind"] == "RETAINED_TEMPLATE"
    )
    coordinate[field] = replacement
    unhashed = dict(package)
    del unhashed["canonical_identity_sha256"]
    package["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(unhashed)
    ).hexdigest()

    with pytest.raises(api.CampaignPackageError, match=message):
        api.canonical_campaign_package_bytes(package)


def test_disabled_adoption_uses_exact_staged_stack_tags() -> None:
    """Break caught: package tags differ from the staged live stacks."""

    deployment = _api().build_campaign_package(request())[
        "disabled_deployment"
    ]
    expected = [
        ["Authority", "H1g"],
        ["Campaign", "GLM-5.2"],
        ["Environment", "production"],
        ["ManagedBy", "CloudFormation"],
        ["Project", "KEEP"],
        ["RunId", RUN_ID],
    ]
    assert deployment["tags"] == expected
    evidence = deployment["staged_infrastructure_evidence"]
    assert evidence["final_readback"]["pending_change_sets"] == 0
    assert "create_change_set_commands" not in deployment


def test_disabled_change_sets_and_anchors_use_separate_service_roles() -> None:
    """Break caught: one deployment role was reused for the fence stack."""

    deployment = _api().build_campaign_package(request())[
        "disabled_deployment"
    ]
    assert deployment["retained_role_arn"].endswith(
        "/keep-glm52-h1g-cloudformation-deployment"
    )
    assert deployment["fence_role_arn"].endswith(
        "/keep-glm52-h1g-fence-service"
    )
    assert deployment["support_role_arn"].endswith(
        "/keep-glm52-h1g-cloudformation-deployment"
    )
    evidence = deployment["staged_infrastructure_evidence"]
    assert evidence["pre_support"]["role_arn"] == (
        deployment["retained_role_arn"]
    )
    assert evidence["postcreate_final"]["role_arn"] == (
        deployment["retained_role_arn"]
    )
    assert "bootstrap" not in deployment


def test_rejects_missing_reviews_mutable_versions_and_unknown_fields() -> None:
    """Break caught: latest/current or unreviewed bytes enter the campaign."""

    api = _api()
    missing = request()
    missing["artifacts"] = missing["artifacts"][1:]
    with pytest.raises(api.CampaignPackageError, match="artifact kinds"):
        api.build_campaign_package(missing)

    mutable = request()
    mutable["artifacts"][0]["version_id"] = "null"
    with pytest.raises(api.CampaignPackageError, match="version"):
        api.build_campaign_package(mutable)

    unknown = request()
    unknown["fallback_to_current_version"] = True
    with pytest.raises(api.CampaignPackageError, match="field set"):
        api.build_campaign_package(unknown)

    duplicate = request()
    duplicate["artifacts"].append(copy.deepcopy(duplicate["artifacts"][0]))
    with pytest.raises(api.CampaignPackageError, match="artifact kinds"):
        api.build_campaign_package(duplicate)


def test_invoke_result_validators_require_exact_version_and_payload() -> None:
    """Break caught: asynchronous, function-error, or foreign-version output passes."""

    api = _api()
    collect_payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task11_collect_rehearsal_result_v1",
        "status": "DEPLOYED_REHEARSAL_RECORDED",
        "account_id": ACCOUNT,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "measurement_id": "measurement-01",
        "collector_function_version_arn": COLLECTOR_ARN,
        "key": (
            "rehearsal/measurements/approved-20260728/"
            + ("a" * 64)
            + "/measurement-01.json"
        ),
        "version_id": "3LgMeasurementVersion",
        "file_sha256": "b" * 64,
        "checksum_sha256_base64": base64.b64encode(b"c" * 32).decode(),
    }
    collect_payload["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(collect_payload)
    ).hexdigest()
    result = {
        "StatusCode": 200,
        "ExecutedVersion": COLLECTOR_VERSION,
        "Payload": json.dumps(collect_payload).encode(),
    }
    assert api.validate_collect_invoke_result(
        result,
        expected_function_version_arn=COLLECTOR_ARN,
        expected_version=COLLECTOR_VERSION,
        expected_measurement_id="measurement-01",
    ) == collect_payload

    wrong_version = dict(result, ExecutedVersion="$LATEST")
    with pytest.raises(api.CampaignPackageError, match="ExecutedVersion"):
        api.validate_collect_invoke_result(
            wrong_version,
            expected_function_version_arn=COLLECTOR_ARN,
            expected_version=COLLECTOR_VERSION,
            expected_measurement_id="measurement-01",
        )

    function_error = dict(result, FunctionError="Unhandled")
    with pytest.raises(api.CampaignPackageError, match="FunctionError"):
        api.validate_collect_invoke_result(
            function_error,
            expected_function_version_arn=COLLECTOR_ARN,
            expected_version=COLLECTOR_VERSION,
            expected_measurement_id="measurement-01",
        )

    finalize_payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_task11_finalize_rehearsal_gate_result_v1",
        "status": "CLOSURE_BUDGET_PROVEN",
        "account_id": ACCOUNT,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "collector_function_version_arn": COLLECTOR_ARN,
        "measurement_count": 20,
        "cold_environment_count": 5,
        "key": (
            "rehearsal/gates/approved-20260728/CLOSURE_BUDGET.json"
        ),
        "version_id": "3LgGateVersion",
        "file_sha256": "d" * 64,
        "body_sha256": "e" * 64,
        "checksum_sha256_base64": base64.b64encode(b"f" * 32).decode(),
        "measurements_identity_sha256": "1" * 64,
    }
    finalize_payload["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(finalize_payload)
    ).hexdigest()
    final_result = {
        "StatusCode": 200,
        "ExecutedVersion": COLLECTOR_VERSION,
        "Payload": json.dumps(finalize_payload).encode(),
    }
    assert api.validate_finalize_invoke_result(
        final_result,
        expected_function_version_arn=COLLECTOR_ARN,
        expected_version=COLLECTOR_VERSION,
    ) == finalize_payload
    bad_count = copy.deepcopy(final_result)
    bad_count["Payload"] = json.dumps(
        dict(finalize_payload, measurement_count=19)
    ).encode()
    with pytest.raises(api.CampaignPackageError, match="payload"):
        api.validate_finalize_invoke_result(
            bad_count,
            expected_function_version_arn=COLLECTOR_ARN,
            expected_version=COLLECTOR_VERSION,
        )


def test_deployment_manifest_is_explicit_disabled_and_inspectable() -> None:
    """Break caught: a command is ambient, writable, or silently executable."""

    package = _api().build_campaign_package(request())
    deployment = package["disabled_deployment"]
    assert deployment["state"] == "ADOPTED_DISABLED"
    assert deployment["approved_support_envelope"] == {
        "support_work_stop_hours": 68,
        "delete_request_deadline_hours": 71,
        "absence_expected_hours": 72,
        "support_host_count": 1,
        "host_root_volume_count": 1,
        "support_data_volume_count": 1,
        "support_data_volume_gib": 50,
        "support_nat_count": 1,
        "support_eip_count": 1,
        "interface_endpoint_eni_count": 2,
        "secret_count": 8,
        "secret_versions_per_secret": 1,
        "support_workflow_count": 1,
        "workflow_history_events_max": 12000,
        "accepted_support_lambda_invocations": 10000,
        "lambda_reserved_concurrency_per_function": 1,
        "transient_log_retention_days": 14,
        "application_log_ingestion_mib_max": 512,
        "forensic_snapshot_count": 1,
        "forensic_snapshot_retention_days": 7,
        "nat_processed_gib_max": 10,
    }
    assert deployment["resource_inventory"]["cloudformation_cardinality"] == {
        "AWS::EC2::Subnet": 3,
        "AWS::EC2::RouteTable": 3,
        "AWS::EC2::SubnetRouteTableAssociation": 3,
        "AWS::EC2::Route": 1,
        "AWS::EC2::NatGateway": 1,
        "AWS::EC2::EIP": 1,
        "AWS::EC2::VPCEndpoint": 3,
        "AWS::EC2::Instance": 1,
        "AWS::EC2::Volume": 1,
        "AWS::SecretsManager::Secret": 8,
        "AWS::S3::Bucket": 1,
        "AWS::Lambda::Function": 1,
        "AWS::Lambda::Version": 1,
        "AWS::StepFunctions::StateMachine": 1,
        "AWS::StepFunctions::StateMachineVersion": 1,
    }
    assert deployment["resource_inventory"][
        "task10_production_workflow"
    ] == {
        "resource_count": 14,
        "state_machine_logical_id": "Task10ProductionStateMachine",
        "state_machine_version_logical_id": (
            "Task10ProductionStateMachineVersion"
        ),
        "state_machine_name": "keep-glm52-h1g-production",
        "workflow_role_logical_id": "Task10ProductionWorkflowRole",
        "reconciliation_role_logical_id": (
            "Task10ProductionReconciliationRole"
        ),
        "reconciliation_log_group_logical_id": (
            "Task10ProductionReconciliationLogGroup"
        ),
        "reconciliation_function_logical_id": (
            "Task10ProductionReconciliationFunction"
        ),
        "reconciliation_function_version_logical_id": (
            "Task10ProductionReconciliationFunctionVersion"
        ),
        "reconciliation_function_name": (
            "keep-glm52-h1g-task10-capacity-reconciliation"
        ),
        "invoke_permission_logical_id": (
            "Task10ProductionReconciliationInvokePermission"
        ),
        "error_alarm_logical_id": (
            "Task10ProductionReconciliationErrorAlarm"
        ),
        "liability_watcher_role_logical_id": (
            "Task9LiabilityWatcherRole"
        ),
        "liability_watcher_log_group_logical_id": (
            "Task9LiabilityWatcherLogGroup"
        ),
        "liability_watcher_function_logical_id": (
            "Task9LiabilityWatcherFunction"
        ),
        "liability_watcher_function_version_logical_id": (
            "Task9LiabilityWatcherFunctionVersion"
        ),
        "liability_watcher_error_alarm_logical_id": (
            "Task9LiabilityWatcherErrorAlarm"
        ),
        "sole_sender_state": "RunInternalSixAzSoleSender",
        "terminal_writer_state": "RunInternalSixAzSoleSender",
        "maximum_ec2_calls": 6,
        "ordered_availability_zones": [
            f"us-west-2{letter}" for letter in "abcdef"
        ],
    }
    from glm52_enforcement.task10_support_plane import (
        Task10SupportInputs,
        render_task10_support_plane_fragment,
    )

    task10_fragment = render_task10_support_plane_fragment(
        inputs=Task10SupportInputs(
            lambda_code_bucket="keep-glm52-code",
            lambda_code_key="support/support.zip",
            lambda_code_version="opaque-version-1",
            lambda_code_sha256="a" * 64,
            ledger_table_arn=(
                "arn:aws:dynamodb:us-west-2:246813579024:"
                "table/keep-glm52-h1g-ledger-v1"
            ),
            campaign_bucket_arn=(
                "arn:aws:s3:::keep-glm52-models-"
                "246813579024-us-west-2"
            ),
        )
    )
    inventory = deployment["resource_inventory"][
        "task10_production_workflow"
    ]
    declared_logical_ids = {
        value
        for key, value in inventory.items()
        if key.endswith("_logical_id")
    }
    assert declared_logical_ids == set(task10_fragment["Resources"])
    definition = json.loads(
        task10_fragment["Resources"]["Task10ProductionStateMachine"][
            "Properties"
        ]["DefinitionString"]
    )
    assert inventory["sole_sender_state"] == definition["StartAt"]
    assert inventory["terminal_writer_state"] in definition["States"]
    assert task10_fragment["Metadata"][
        "Task10ProductionResourceCount"
    ] == inventory["resource_count"]
    assert deployment["fence_parameters"] == []
    assert deployment["support_parameters"] == []
    assert deployment["retained_stack_id"].split("/")[1] == "keep-glm52-gpu"
    assert deployment["adoption_readback_contract"] == {
        "operation_kind": "staged-infrastructure-adoption",
        "read_only": True,
        "exact_evidence_required": True,
        "staged_request_identity_sha256": deployment[
            "staged_infrastructure_evidence"
        ]["staged_request_identity_sha256"],
        "staged_journal_sha256": deployment[
            "staged_infrastructure_evidence"
        ]["staged_journal_sha256"],
        "fixed_artifacts_identity_sha256": deployment[
            "staged_infrastructure_evidence"
        ]["fixed_artifacts_identity_sha256"],
        "worker_activation_allowed": False,
    }
    assert {
        "bootstrap",
        "create_change_set_commands",
        "inspection_commands",
        "execute_change_sets",
        "post_deploy_readback_commands",
    }.isdisjoint(deployment)
    all_commands = package["cleanup_and_reconciliation"]["commands"]
    assert all(type(row["argv"]) is list for row in all_commands)
    rendered = "\n".join(" ".join(row["argv"]) for row in all_commands)
    assert "--profile keep-gpu" in rendered
    assert "--region us-west-2" in rendered
    assert ACCOUNT in json.dumps(all_commands)
    assert "$LATEST" not in rendered
    assert "latest" not in rendered.lower()
    assert "put-object" not in rendered
    assert "run-instances" not in rendered
    assert "request-spot-instances" not in rendered
    assert "write_gate" not in rendered
    assert "secret" not in rendered.lower()
    assert all(
        row["operation"] in package["command_operation_allowlist"]
        for row in all_commands
    )


def test_negative_probes_h100_six_az_retry_and_monitor_stay_gated() -> None:
    """Break caught: qualification or p5 launch bypasses immutable evidence gates."""

    package = _production_package()
    probes = package["negative_iam_probes"]
    assert [probe["expected_error_code"] for probe in probes] == [
        "AccessDenied",
        "AccessDenied",
        "AccessDenied",
        "AccessDenied",
    ]
    assert len({probe["probe_id"] for probe in probes}) == 4
    qualification = package["h100_qualification"]
    assert qualification["driver_record_type"] == (
        "glm52_task13_repository_driver_v2"
    )
    assert qualification["driver_operation_kind"] == "h100-qualification"
    assert qualification["authorized"] is False
    assert qualification["instance_type"] == "p5.48xlarge"
    assert qualification["required_training_steps"] == 2
    seed = package["qualification_cache_seed"]
    assert seed["driver_record_type"] == "glm52_task13_repository_driver_v2"
    assert seed["driver_operation_kind"] == "qualification-cache-seed"
    assert seed["authorized"] is False
    assert seed["maximum_gpu_hours"] == 6
    assert seed["required_teacher_rows"] == 1
    assert seed["teardown_required"] is True
    retry = package["production_retry_plan"]
    expected_route = [
        "aws/glm52-gpu/scripts/submit_sky_campaign.sh",
        "--production",
        "start",
    ]
    assert retry["route"] == expected_route
    custody = retry["execution_custody"]
    assert custody == {
        "schema_version": 1,
        "record_type": (
            "glm52_task13_reviewed_repository_execution_custody_v1"
        ),
        "execution_root_source": "REVIEWED_REPOSITORY_ARCHIVE",
        "repository_archive_manifest": next(
            row
            for row in package["reviewed_artifacts"]
            if row["artifact_kind"] == "REPOSITORY_ARCHIVE"
        ),
        "archive_payload_exact_version_required": True,
        "archive_payload_file_sha256_required": True,
        "safe_archive_extraction_required": True,
        "live_worktree_execution_allowed": False,
        "production_route": expected_route,
        "required_executable_paths": [
            "aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py",
            "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh",
            "aws/glm52-gpu/scripts/submit_sky_campaign.py",
            "aws/glm52-gpu/scripts/submit_sky_campaign.sh",
            "src/glm52_enforcement/task10_production.py",
        ],
        "executable_file_sha256_required": True,
        "repository_driver_manifests": {
            "h100-qualification": next(
                row
                for row in package["reviewed_artifacts"]
                if row["artifact_kind"] == "H100_QUALIFICATION_INPUT"
            ),
            "qualification-cache-seed": next(
                row
                for row in package["reviewed_artifacts"]
                if row["artifact_kind"]
                == "QUALIFICATION_CACHE_SEED_INPUT"
            ),
        },
        "canonical_identity_sha256": custody[
            "canonical_identity_sha256"
        ],
    }
    custody_body = dict(custody)
    custody_identity = custody_body.pop("canonical_identity_sha256")
    assert custody_identity == hashlib.sha256(
        canonical_json_bytes(custody_body)
    ).hexdigest()
    assert retry["workflow_reconciliation_contract"] == {
        "writer_owner": "TASK10_VERSIONED_PRODUCTION_WORKFLOW",
        "writer_handler": (
            "aws/glm52-gpu/lambda/"
            "task10_sole_sender_handler.py"
        ),
        "state_machine_logical_id": "Task10ProductionStateMachine",
        "state_machine_version_logical_id": (
            "Task10ProductionStateMachineVersion"
        ),
        "state_machine_name": "keep-glm52-h1g-production",
        "reconciliation_function_logical_id": (
            "Task10ProductionReconciliationFunction"
        ),
        "reconciliation_function_version_logical_id": (
            "Task10ProductionReconciliationFunctionVersion"
        ),
        "reconciliation_function_name": (
            "keep-glm52-h1g-task10-capacity-reconciliation"
        ),
        "terminal_writer_state": "RunInternalSixAzSoleSender",
        "workflow_invocation_count": 1,
        "bucket": (
            "keep-glm52-models-246813579024-us-west-2"
        ),
        "key": (
            f"campaigns/{RUN_ID}/submissions/production/"
            "generations/00000001/workflow/LAUNCH_OUTCOME.json"
        ),
        "schema_version": 1,
        "record_type": "glm52_task10_capacity_reconciliation_v1",
        "classifications": [
            "RUNNING",
            "WORKER_ALLOCATED",
            "CAPACITY_EXHAUSTED",
            "FAILED",
        ],
        "capacity_outcome_fields": [
            "attempt",
            "availability_zone",
            "outcome",
        ],
        "exact_version_id_required": True,
        "canonical_self_hash_required": True,
    }
    assert [
        row["artifact_kind"] for row in retry["immutable_inputs"]
    ] == [
        "REPOSITORY_ARCHIVE",
        "ACCEPTED_BASELINE",
        "PROMPT_PACK",
        "TRAINING_CONFIGURATION",
        "GPU_SPEND_APPROVAL",
        "SUPPORT_APPROVAL",
        "RESIDUAL_LIABILITY_APPROVAL",
        "PRODUCTION_DESCRIPTOR",
        "CLEAN_REHEARSAL",
        "TASK10_PRODUCTION_AUTHORITY",
        "TASK10_WORKER_DESCRIPTOR",
        "TASK10_TASK_INPUTS",
    ]
    assert retry["capacity_type"] == "ON_DEMAND"
    assert retry["max_active_instances"] == 1
    assert retry["availability_zones"] == [
        f"us-west-2{letter}" for letter in "abcdef"
    ]
    assert retry["attempts"] == [
        {
            "attempt": index,
            "availability_zone": f"us-west-2{letter}",
            "on_capacity_failure": (
                "ROTATE_NEXT_AZ" if letter != "f" else "STOP_CAPACITY_EXHAUSTED"
            ),
        }
        for index, letter in enumerate("abcdef", 1)
    ]
    assert retry["required_gates"] == package["required_gates"]
    assert retry["authorized"] is False
    assert retry["production_authority_contract"] == {
        "minimum_remaining_gpu_seconds_exclusive": 3600,
        "action_kind": "PRODUCTION_SUBMISSION",
        "action_count": 1,
        "required_terminal_markers": [
            "CAMPAIGN_DRAINED.json",
            "TERMINAL_VERIFIED.json",
        ],
        "monitor_route": (
            "aws/glm52-gpu/scripts/sky_campaign_break_glass.sh status"
        ),
    }
    monitor = package["monitor_contract"]
    assert monitor["descriptor_path"] == request()["monitor_descriptor_path"]
    assert monitor["argv"] == [
        "aws/glm52-gpu/scripts/sky_campaign_break_glass.sh",
        "status",
    ]
    assert monitor["environment"] == {
        "AWS_PROFILE": PROFILE,
        "CAMPAIGN_DESCRIPTOR": request()["monitor_descriptor_path"],
        "SKY_BIN": (
            "/Users/jack.mazac/.local/share/keep/"
            "skypilot-0.13.0/bin/sky"
        ),
        "SKYPILOT_CONFIG": (
            "/Users/jack.mazac/.local/share/keep/"
            "skypilot-0.13.0/server-config.yaml"
        ),
    }
    assert monitor["profile"] == PROFILE
    assert monitor["environment"]["CAMPAIGN_DESCRIPTOR"] == (
        monitor["descriptor_path"]
    )


def test_package_rejects_live_worktree_execution_custody(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: reviewed archive metadata permits live-checkout execution."""

    api = _api()
    monkeypatch.setattr(
        api,
        "validate_staged_infrastructure_evidence",
        lambda value, **_kwargs: copy.deepcopy(value),
    )
    package = api.build_campaign_package(request())
    custody = package["production_retry_plan"]["execution_custody"]
    custody["live_worktree_execution_allowed"] = True
    custody_body = dict(custody)
    custody_body.pop("canonical_identity_sha256")
    custody["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(custody_body)
    ).hexdigest()
    package_body = dict(package)
    package_body.pop("canonical_identity_sha256")
    package["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(package_body)
    ).hexdigest()

    with pytest.raises(
        api.CampaignPackageError,
        match="execution custody",
    ):
        api.canonical_campaign_package_bytes(package)


def test_acceptance_archive_and_production_inputs_are_immutable_gates() -> None:
    """Break caught: focused tests substitute for Task13 acceptance authority."""

    package = _production_package()
    kinds = {
        row["artifact_kind"] for row in package["reviewed_artifacts"]
    }
    required_inputs = {
        "REPOSITORY_ARCHIVE",
        "CLEAN_REHEARSAL",
        "ACCEPTED_BASELINE",
        "PROMPT_PACK",
        "TRAINING_CONFIGURATION",
        "GPU_SPEND_APPROVAL",
        "SUPPORT_APPROVAL",
            "RESIDUAL_LIABILITY_APPROVAL",
            "PRODUCTION_DESCRIPTOR",
            "TASK10_PRODUCTION_AUTHORITY",
            "TASK10_WORKER_DESCRIPTOR",
            "TASK10_TASK_INPUTS",
        }
    assert {
        "T01_T25_GATE",
        "TRANSPORT_22_MUTANT_GATE",
        *required_inputs,
    }.issubset(kinds)
    assert {
        "T01_T25_PROVEN",
        "TRANSPORT_22_MUTANTS_PROVEN",
        "EXACT_ARCHIVE_PROVEN",
        "CLEAN_REHEARSAL_PROVEN",
    }.issubset(package["required_gates"])
    assert {
        row["artifact_kind"]
        for row in package["production_retry_plan"]["immutable_inputs"]
    } == required_inputs


def test_builder_script_is_deterministic_and_no_overwrite(tmp_path: Path) -> None:
    """Break caught: packaging contacts AWS or replaces reviewed evidence."""

    script = (
        Path(__file__).parents[1]
        / "aws/glm52-gpu/scripts/build_glm52_task13_campaign_package.py"
    )
    request_path = tmp_path / "request.json"
    output_a = tmp_path / "package-a.json"
    output_b = tmp_path / "package-b.json"
    request_path.write_bytes(canonical_json_bytes(request()) + b"\n")
    for output in (output_a, output_b):
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--request",
                str(request_path),
                "--output",
                str(output),
            ],
            cwd=Path(__file__).parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    assert output_a.read_bytes() == output_b.read_bytes()
    before = output_a.read_bytes()
    refused = subprocess.run(
        [
            sys.executable,
            str(script),
            "--request",
            str(request_path),
            "--output",
            str(output_a),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode != 0
    assert output_a.read_bytes() == before


def test_package_builder_cli_requires_and_consumes_clean_evidence(
    tmp_path: Path,
) -> None:
    from glm52_enforcement.task13_clean_rehearsal import (
        clean_rehearsal_coordinate,
    )
    from test_glm52_task13_clean_rehearsal import _build, _fixture

    fixture = _fixture()
    evidence = _build(fixture)
    clean_coordinate = clean_rehearsal_coordinate(
        evidence=evidence,
        version_id="clean-evidence-version-0001",
    )
    production_request = _production_request(
        clean_coordinate=clean_coordinate,
    )
    replacements = {
        "REPOSITORY_ARCHIVE": fixture["repository_archive_coordinate"],
        "CLEAN_REHEARSAL": clean_coordinate,
    }
    production_request["artifacts"] = [
        replacements.get(row["artifact_kind"], row)
        for row in production_request["artifacts"]
    ]
    production_request["staged_infrastructure_evidence"] = _staged_evidence(
        production_request["artifacts"]
    )
    values = {
        "request": production_request,
        "predecessor": fixture["predecessor_package"],
        "predecessor-artifacts": fixture[
            "predecessor_reviewed_artifacts"
        ],
        "clean-evidence": evidence,
    }
    paths = {}
    for label, value in values.items():
        path = (tmp_path / (label + ".json")).resolve()
        path.write_bytes(canonical_json_bytes(value) + b"\n")
        paths[label] = path
    script = (
        Path(__file__).parents[1]
        / "aws/glm52-gpu/scripts/build_glm52_task13_campaign_package.py"
    )
    output = (tmp_path / "production-package.json").resolve()
    command = [
        sys.executable,
        str(script),
        "--request",
        str(paths["request"]),
        "--predecessor-package",
        str(paths["predecessor"]),
        "--predecessor-reviewed-artifacts",
        str(paths["predecessor-artifacts"]),
        "--clean-rehearsal-evidence",
        str(paths["clean-evidence"]),
        "--output",
        str(output),
    ]

    completed = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    package = json.loads(output.read_bytes())
    assert (
        package["predecessor_identity"]["clean_rehearsal_evidence"]
        == evidence
    )

    missing_output = (tmp_path / "missing-evidence.json").resolve()
    missing = subprocess.run(
        [
            item
            for item in command[:-2]
            if item
            not in {
                "--clean-rehearsal-evidence",
                str(paths["clean-evidence"]),
            }
        ]
        + ["--output", str(missing_output)],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode == 64
    assert not missing_output.exists()

    prequalification_output = (
        tmp_path / "prequalification-with-evidence.json"
    ).resolve()
    prequalification_path = (tmp_path / "prequalification.json").resolve()
    prequalification_request = request()
    prequalification_path.write_bytes(
        canonical_json_bytes(prequalification_request) + b"\n"
    )
    unused = subprocess.run(
        [
            sys.executable,
            str(script),
            "--request",
            str(prequalification_path),
            "--clean-rehearsal-evidence",
            str(paths["clean-evidence"]),
            "--output",
            str(prequalification_output),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert unused.returncode == 64
    assert not prequalification_output.exists()


def test_prequalification_input_builder_emits_exact_request_and_reviewed_list(
    tmp_path: Path,
) -> None:
    candidate = request()
    coordinates = candidate.pop("artifacts")
    base_path = tmp_path / "base.json"
    coordinates_path = tmp_path / "coordinates.json"
    request_path = tmp_path / "request.json"
    reviewed_path = tmp_path / "reviewed.json"
    base_path.write_bytes(canonical_json_bytes(candidate) + b"\n")
    coordinates_path.write_bytes(canonical_json_bytes(coordinates) + b"\n")
    script = (
        Path(__file__).parents[1]
        / "aws/glm52-gpu/scripts/"
        "build_glm52_task13_prequalification_inputs.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--base-request",
            str(base_path),
            "--artifact-coordinates",
            str(coordinates_path),
            "--request",
            str(request_path),
            "--reviewed-artifacts",
            str(reviewed_path),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    exact_request = json.loads(request_path.read_bytes())
    reviewed = json.loads(reviewed_path.read_bytes())
    package = _api().build_campaign_package(exact_request)
    assert package["package_phase"] == "PREQUALIFICATION"
    assert reviewed == package["reviewed_artifacts"]


def test_builder_cli_requires_and_exact_reads_production_predecessor(
    tmp_path: Path,
) -> None:
    """Break caught: successor lineage exists only in an in-process helper."""

    api = _api()
    predecessor, production_request, evidence = (
        _production_successor_fixture()
    )
    request_path = (tmp_path / "production-request.json").resolve()
    predecessor_path = (tmp_path / "predecessor.json").resolve()
    predecessor_artifacts_path = (
        tmp_path / "predecessor-artifacts.json"
    ).resolve()
    clean_evidence_path = (tmp_path / "clean-evidence.json").resolve()
    output_path = (tmp_path / "production-package.json").resolve()
    request_path.write_bytes(
        canonical_json_bytes(production_request) + b"\n"
    )
    predecessor_path.write_bytes(
        api.canonical_campaign_package_bytes(predecessor)
    )
    predecessor_artifacts_path.write_bytes(
        canonical_json_bytes(predecessor["reviewed_artifacts"]) + b"\n"
    )
    clean_evidence_path.write_bytes(
        canonical_json_bytes(evidence) + b"\n"
    )
    script = (
        Path(__file__).parents[1]
        / "aws/glm52-gpu/scripts/build_glm52_task13_campaign_package.py"
    )
    missing = subprocess.run(
        [
            sys.executable,
            str(script),
            "--request",
            str(request_path),
            "--clean-rehearsal-evidence",
            str(clean_evidence_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode != 0
    assert not output_path.exists()

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--request",
            str(request_path),
            "--predecessor-package",
            str(predecessor_path),
            "--predecessor-reviewed-artifacts",
            str(predecessor_artifacts_path),
            "--clean-rehearsal-evidence",
            str(clean_evidence_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    package = json.loads(output_path.read_bytes())
    assert package["package_phase"] == "PRODUCTION"
    assert package["predecessor_identity"][
        "package_identity_sha256"
    ] == predecessor["canonical_identity_sha256"]


def test_final_package_requires_preexisting_task10_authority_coordinate() -> None:
    """Break caught: final package invents its own production authority."""

    predecessor, mutable, evidence = _production_successor_fixture()
    authority = next(
        item
        for item in mutable["artifacts"]
        if item["artifact_kind"] == "TASK10_PRODUCTION_AUTHORITY"
    )
    authority["version_id"] = "null"
    with pytest.raises(
        _api().CampaignPackageError,
        match="immutable version",
    ):
        _api().build_campaign_package(
            mutable,
            predecessor_package=predecessor,
            predecessor_reviewed_artifacts=predecessor[
                "reviewed_artifacts"
            ],
            clean_rehearsal_evidence=evidence,
        )
