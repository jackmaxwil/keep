"""Task 13 staged, injected, resumable campaign runner."""

from __future__ import annotations

import ast
import atexit
import base64
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.decision_closure import (
    CLOSURE_PHASE_CEILINGS,
    REHEARSAL_PATH_PROOFS,
    SUFFIX_PHASE_CEILINGS,
    ZERO_PRODUCTION_EFFECTS,
    PhaseSpan,
    build_closure_request,
    build_deployed_gate_document,
    build_rehearsal_measurement,
    rehearsal_measurement_from_mapping,
)
from glm52_enforcement.task13_campaign_package import (
    CAMPAIGN_OWNER_APPROVAL_SHA256,
    build_campaign_package,
    canonical_campaign_package_bytes,
)
from glm52_enforcement.task13_controller_authority import (
    ControllerAuthorityIssuerError,
    issue_controller_execution_authority,
    issue_controller_operation_capability,
)
from glm52_enforcement.task13_transport_gates import SEMANTIC_GATE_PINS
from glm52_task13_staged_fixture_support import (
    build_staged_infrastructure_evidence,
)
from glm52_task13_signer_support import (
    ephemeral_controller_authority_signer,
    patch_controller_authority_consumer_pins,
    patched_python_script_command,
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
REPOSITORY_COORDINATOR = (
    Path(__file__).parents[1]
    / "aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py"
)
COORDINATOR_SHA = hashlib.sha256(
    REPOSITORY_COORDINATOR.read_bytes()
).hexdigest()
FULL_RUN_WORK, SIGNING_PRIVATE_KEY, SIGNER = (
    ephemeral_controller_authority_signer()
)
_SIGNED_AUTHORITY_CACHE: dict[
    tuple[str, str, str], object
] = {}
_SIGNED_V2_JOURNAL_ROOT = Path(
    tempfile.mkdtemp(
        prefix="glm52-task13-signed-v2-tests-",
        dir="/private/tmp",
    )
).resolve(strict=True)
_SIGNED_V2_JOURNAL_ROOT.chmod(0o700)
atexit.register(
    shutil.rmtree,
    _SIGNED_V2_JOURNAL_ROOT,
    ignore_errors=True,
)


def _api():
    try:
        return importlib.import_module(
            "glm52_enforcement.task13_campaign_runner"
        )
    except ModuleNotFoundError:
        pytest.fail(
            "Task 13 executable campaign runner is missing",
            pytrace=False,
        )


def _coordinator_api():
    spec = importlib.util.spec_from_file_location(
        "glm52_task13_production_coordinator_test",
        REPOSITORY_COORDINATOR,
    )
    if spec is None or spec.loader is None:
        pytest.fail("Task 13 production coordinator is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    patch_controller_authority_consumer_pins(
        module,
        SIGNING_PRIVATE_KEY.with_suffix(".pub"),
    )
    return module


def _script_api(script_name: str):
    script = (
        Path(__file__).parents[1]
        / "aws/glm52-gpu/scripts"
        / script_name
    )
    spec = importlib.util.spec_from_file_location(
        script.stem + "_argv_contract_test",
        script,
    )
    if spec is None or spec.loader is None:
        pytest.fail(f"{script_name} is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coordinator_envelope(
    coordinator: object,
    *,
    action: str,
    kind: str,
    request: object,
) -> dict[str, object]:
    api = _api()
    package = _package()
    coordinator_sha256 = coordinator._self_sha256()
    authority = _issued_authority(
        package,
        "deploy-disabled",
        coordinator_sha256,
    )
    operation_id = "test:" + action + ":" + kind
    request_identity = canonical_sha256(request)
    if action == "inspect":
        journal_store_path = "READ_ONLY"
        journal_store_identity = canonical_sha256(
            {
                "authority": authority.canonical_identity_sha256,
                "stage": authority.stage,
                "state": "READ_ONLY",
            }
        )
        journal_record_identities = [
            authority.canonical_identity_sha256
        ]
        journal_state = "READ_ONLY"
    else:
        descriptor, raw_path = tempfile.mkstemp(
            prefix="operation-",
            suffix=".jsonl",
            dir=_SIGNED_V2_JOURNAL_ROOT,
        )
        os.close(descriptor)
        path = Path(raw_path).resolve(strict=True)
        path.chmod(0o600)
        journal = api.FileJournalStore(path)
        for sequence, state in (
            (1, "PREPARED"),
            (2, "POSSIBLY_SENT"),
        ):
            body = {
                "schema_version": 1,
                "record_type": "glm52_task13_runner_journal_v1",
                "operation_id": operation_id,
                "operation_kind": kind,
                "stage": authority.stage,
                "sequence": sequence,
                "state": state,
                "request_identity_sha256": request_identity,
                "result_identity_sha256": None,
            }
            journal.append(
                {
                    **body,
                    "canonical_identity_sha256": canonical_sha256(body),
                }
            )
        binding = journal.authority_binding(operation_id)
        journal_store_path = binding["journal_store_path"]
        journal_store_identity = binding[
            "journal_store_identity_sha256"
        ]
        journal_record_identities = binding[
            "journal_record_identities"
        ]
        journal_state = binding["journal_state"]
    prior_execute = (
        canonical_sha256(
            {
                "stage": authority.stage,
                "operation_kind": kind,
                "operation_id": operation_id,
                "request_identity_sha256": request_identity,
            }
        )
        if action == "reconcile"
        else None
    )
    capability = issue_controller_operation_capability(
        authority=authority,
        action=action,
        operation_kind=kind,
        operation_id=operation_id,
        request_identity_sha256=request_identity,
        journal_store_path=journal_store_path,
        journal_store_identity_sha256=journal_store_identity,
        journal_record_identities=journal_record_identities,
        journal_state=journal_state,
        prior_execute_capability_identity_sha256=prior_execute,
        ttl_seconds=60,
        signer=SIGNER,
    )
    return {
        "schema_version": 2,
        "record_type": "glm52_task13_coordinator_request_v2",
        "action": action,
        "operation_kind": kind,
        "operation_id": operation_id,
        "request": request,
        "coordinator_executable_sha256": coordinator_sha256,
        "controller_execution_authority": asdict(authority),
        "operation_capability": asdict(capability),
    }


def _fixture_template_body(kind: str) -> dict[str, object]:
    if kind == "FENCE_TEMPLATE":
        return {
            "Resources": {
                "H1gProductionFenceBucketPolicy": {
                    "Type": "AWS::S3::BucketPolicy"
                }
            }
        }
    if kind != "SUPPORT_TEMPLATE":
        return {
            "fixture_template_kind": kind,
            "Resources": {},
        }
    support_archive_sha = hashlib.sha256(
        b"support-lambda-archive"
    ).hexdigest()
    return {
        "Resources": {
            "SupportFunctionRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "RoleName": (
                        "keep-glm52-h1g-support-function-role"
                    ),
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": "lambda.amazonaws.com"
                                },
                                "Action": "sts:AssumeRole",
                            }
                        ],
                    },
                },
            },
            "SupportFunction": {
                "Type": "AWS::Lambda::Function",
                "Properties": {
                    "FunctionName": (
                        "keep-glm52-h1g-support-function"
                    ),
                    "Code": {
                        "S3Bucket": (
                            "keep-glm52-models-"
                            "246813579024-us-west-2"
                        ),
                        "S3Key": (
                            "task13/artifacts/support-lambda/"
                            + support_archive_sha
                            + ".zip"
                        ),
                        "S3ObjectVersion": (
                            "3LgSupportLambdaArchiveVersion"
                        ),
                    },
                    "Handler": "support_custom_resource_handler.main",
                    "Runtime": "python3.12",
                    "Architectures": ["x86_64"],
                    "MemorySize": 256,
                    "Timeout": 840,
                    "ReservedConcurrentExecutions": 1,
                    "Role": {
                        "Fn::GetAtt": [
                            "SupportFunctionRole",
                            "Arn",
                        ]
                    },
                    "Environment": {
                        "Variables": {
                            "GLM52_ACTIVATION_ID": ACTIVATION,
                            "GLM52_RUN_ID": RUN_ID,
                        }
                    },
                    "Layers": [
                        "arn:aws:lambda:us-west-2:"
                        "246813579024:layer:"
                        "keep-glm52-h1g-cryptography-"
                        "py312-x86-64:7"
                    ],
                },
            },
            "SupportVersion": {
                "Type": "AWS::Lambda::Version",
                "Properties": {
                    "FunctionName": {"Ref": "SupportFunction"},
                    "Description": "code=" + support_archive_sha,
                },
            },
        }
    }


def _coordinate(kind: str, key: str) -> dict[str, object]:
    pin = SEMANTIC_GATE_PINS.get(kind)
    template_body = _fixture_template_body(kind)
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
        "body_sha256": (
            hashlib.sha256(
                canonical_json_bytes(template_body)
            ).hexdigest()
            if kind
            in {"RETAINED_TEMPLATE", "FENCE_TEMPLATE", "SUPPORT_TEMPLATE"}
            else hashlib.sha256(kind.encode()).hexdigest()
            if pin is None
            else pin["body_sha256"]
        ),
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


def _package(phase: str = "PRODUCTION") -> dict[str, object]:
    request = {
        "schema_version": 1,
        "record_type": "glm52_task13_campaign_package_request_v1",
        "account_id": ACCOUNT,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "collector_version_arn": COLLECTOR_ARN,
        "task11_request": asdict(
            build_closure_request(
                activation_id=ACTIVATION,
                generation=1,
                candidate_identity_sha256="9" * 64,
                initial_source_predecessor_version_id=(
                    "3LgSourcePredecessor"
                ),
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
        ),
        "task11_boundary": {
            "bucket": (
                "keep-glm52-models-246813579024-us-west-2"
            ),
            "key": (
                "campaigns/glm52-sky-20260724/authorities/task11/"
                "approved-20260728/00000001/TASK11_BOUNDARY.json"
            ),
            "version_id": "3LgTask11BoundaryVersion",
            "file_sha256": "7" * 64,
            "body_sha256": "8" * 64,
        },
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
            ),
            _coordinate(
                "TASK12_REVIEW_APPROVAL",
                "reviews/task12/approval.json",
            ),
            _coordinate(
                "RETAINED_FOUNDATION_TEMPLATE",
                "task13/templates/retained-foundation.yaml",
            ),
            _coordinate(
                "RETAINED_PRE_SUPPORT_TEMPLATE",
                "task13/templates/retained-pre-support.yaml",
            ),
            _coordinate(
                "RETAINED_TEMPLATE",
                "task13/templates/retained.yaml",
            ),
            _coordinate(
                "FENCE_TEMPLATE",
                "task13/migration/fence-transfer.json",
            ),
            _coordinate(
                "SUPPORT_TEMPLATE",
                "task13/templates/support-disabled.yaml",
            ),
            _coordinate(
                "SUPPORT_INPUTS",
                "task13/inputs/support-build-inputs.json",
            ),
            _coordinate(
                "H100_QUALIFICATION_INPUT",
                (
                    f"task13/activations/{ACTIVATION}/"
                    "qualification/h100-input.json"
                ),
            ),
            _coordinate(
                "BOOTSTRAP_TEMPLATE",
                "task13/templates/container-bootstrap-v1.json",
            ),
            _coordinate(
                "QUALIFICATION_CACHE_SEED_INPUT",
                (
                    f"task13/activations/{ACTIVATION}/"
                    "qualification/cache-seed-input.json"
                ),
            ),
            *[
                _coordinate(kind, key)
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
        ],
    }
    from glm52_enforcement.task13_clean_rehearsal import (
        build_clean_rehearsal_evidence,
        clean_rehearsal_coordinate,
    )
    from test_glm52_task13_clean_rehearsal import _fixture

    clean_inputs = _fixture()
    request["artifacts"] = [
        clean_inputs["repository_archive_coordinate"]
        if row["artifact_kind"] == "REPOSITORY_ARCHIVE"
        else row
        for row in request["artifacts"]
    ]
    request["staged_infrastructure_evidence"] = _staged_evidence(
        request["artifacts"]
    )
    prequalification_request = copy.deepcopy(request)
    prequalification_request["artifacts"] = sorted(
        (
            row
            for row in prequalification_request["artifacts"]
            if row["artifact_kind"]
            not in {
                "CLEAN_REHEARSAL",
                "TASK10_PRODUCTION_AUTHORITY",
                "TASK10_WORKER_DESCRIPTOR",
                "TASK10_TASK_INPUTS",
            }
        ),
        key=lambda row: row["artifact_kind"],
    )
    prequalification_request["staged_infrastructure_evidence"] = (
        _staged_evidence(prequalification_request["artifacts"])
    )
    predecessor = build_campaign_package(prequalification_request)
    if phase == "PREQUALIFICATION":
        return predecessor
    assert phase == "PRODUCTION"

    evidence = build_clean_rehearsal_evidence(
        predecessor_package=predecessor,
        predecessor_package_bytes=canonical_campaign_package_bytes(
            predecessor
        ),
        predecessor_reviewed_artifacts=predecessor["reviewed_artifacts"],
        repository_archive_coordinate=clean_inputs[
            "repository_archive_coordinate"
        ],
        repository_archive_manifest=clean_inputs[
            "repository_archive_manifest"
        ],
        repository_archive_manifest_bytes=clean_inputs[
            "repository_archive_manifest_bytes"
        ],
        finalize_invoke_result=clean_inputs["finalize_invoke_result"],
        gate_readback=clean_inputs["gate_readback"],
        gate_readback_bytes=clean_inputs["gate_readback_bytes"],
    )
    clean_coordinate = clean_rehearsal_coordinate(
        evidence=evidence,
        version_id="clean-evidence-version-0001",
    )
    task10_rows = [
        row
        for row in request["artifacts"]
        if row["artifact_kind"]
        in {
            "TASK10_PRODUCTION_AUTHORITY",
            "TASK10_WORKER_DESCRIPTOR",
            "TASK10_TASK_INPUTS",
        }
    ]
    production_request = copy.deepcopy(request)
    production_request["artifacts"] = sorted(
        [
            *predecessor["reviewed_artifacts"],
            clean_coordinate,
            *task10_rows,
        ],
        key=lambda row: row["artifact_kind"],
    )
    production_request["staged_infrastructure_evidence"] = _staged_evidence(
        production_request["artifacts"]
    )
    return build_campaign_package(
        production_request,
        predecessor_package=predecessor,
        predecessor_reviewed_artifacts=predecessor["reviewed_artifacts"],
        clean_rehearsal_evidence=evidence,
    )


class MemoryJournal:
    def __init__(self, *, _leaf: bool = False) -> None:
        self.rows: list[dict[str, object]] = []
        self.lock = threading.Lock()
        self.scoped = (
            {}
            if _leaf
            else {
                "fence": MemoryJournal(_leaf=True),
                "support": MemoryJournal(_leaf=True),
            }
        )

    def for_scope(self, scope: str) -> "MemoryJournal":
        if scope == "campaign":
            return self
        return self.scoped[scope]

    def entries(self, operation_id: str) -> tuple[dict[str, object], ...]:
        with self.lock:
            return tuple(
                copy.deepcopy(row)
                for row in self.rows
                if row["operation_id"] == operation_id
            )

    def append(self, row: dict[str, object]) -> None:
        with self.lock:
            self.rows.append(copy.deepcopy(row))


class NoExternalServices:
    def __getattr__(self, name: str):
        raise AssertionError("validate stage attempted external service: " + name)


def _spans(
    values: tuple[tuple[str, int], ...],
    *,
    start: int = 0,
) -> tuple[PhaseSpan, ...]:
    result = []
    cursor = start
    for name, duration in values:
        result.append(PhaseSpan(name, cursor, cursor + duration))
        cursor += duration
    return tuple(result)


def _measurement(index: int, *, reused_environment: bool = False) -> dict[str, object]:
    suffix_durations = (1, 1, 10, 1, 1, 1, 1, 1)
    closure_durations = (
        10,
        10,
        10,
        10,
        sum(suffix_durations),
        10,
        10,
        10,
    )
    closure = _spans(
        tuple(zip(CLOSURE_PHASE_CEILINGS, closure_durations))
    )
    suffix = _spans(
        tuple(zip(SUFFIX_PHASE_CEILINGS, suffix_durations)),
        start=closure[4].started_monotonic_seconds,
    )
    faults = {
        1: "THROTTLING",
        2: "PAGINATION",
        3: "NETWORK_AMBIGUITY",
    }
    measurement = build_rehearsal_measurement(
        rehearsal_id=f"measurement-{index:02d}",
        provenance="DEPLOYED_REHEARSAL",
        lambda_environment_id=(
            "environment-reused"
            if reused_environment and index <= 5
            else f"environment-{index:02d}"
        ),
        cold_start=index <= 5,
        path_proofs=REHEARSAL_PATH_PROOFS,
        canary_bucket="keep-glm52-h1g-rehearsal",
        canary_key=f"rehearsal/canary/{index:02d}/decision.json",
        canary_can_satisfy_production_authority=False,
        canary_worker_readable=False,
        canary_admission_readable=False,
        effect_counts=ZERO_PRODUCTION_EFFECTS,
        relay_call_count=0,
        closure_spans=closure,
        suffix_spans=suffix,
        first_policy_readback_monotonic_seconds=(
            suffix[2].started_monotonic_seconds
        ),
        second_policy_readback_monotonic_seconds=(
            suffix[2].started_monotonic_seconds + 10
        ),
        injected_failure_kind=faults.get(index, "NONE"),
        failure_unwind_seconds=10 if index <= 3 else 0,
    )
    return json.loads(canonical_json_bytes(asdict(measurement)))


def _self_hash(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["canonical_identity_sha256"] = canonical_sha256(value)
    return result


def _marker(key: str, digest_character: str) -> dict[str, object]:
    body = {
        "key": key,
        "version_id": "3Lg" + digest_character * 16,
        "file_sha256": digest_character * 64,
        "body_sha256": digest_character * 64,
    }
    return {
        **body,
        "canonical_identity_sha256": canonical_sha256(body),
    }


class FiniteServices:
    coordinator_executable_sha256 = COORDINATOR_SHA
    def __init__(
        self,
        *,
        reused_environment: bool = False,
        wrong_version_at: int | None = None,
        lost_response_operation: str | None = None,
        launch_accept_at: int = 6,
        stale_credentials: bool = False,
        foreign_h100_marker: bool = False,
        marker_readback_drift: bool = False,
        bad_launch_authority: bool = False,
        terminal_proof_drift: bool = False,
        opaque_cf_evidence: bool = False,
        artifact_readback_drift: bool = False,
    ) -> None:
        self.reused_environment = reused_environment
        self.wrong_version_at = wrong_version_at
        self.lost_response_operation = lost_response_operation
        self.launch_accept_at = launch_accept_at
        self.stale_credentials = stale_credentials
        self.foreign_h100_marker = foreign_h100_marker
        self.marker_readback_drift = marker_readback_drift
        self.bad_launch_authority = bad_launch_authority
        self.terminal_proof_drift = terminal_proof_drift
        self.opaque_cf_evidence = opaque_cf_evidence
        self.artifact_readback_drift = artifact_readback_drift
        self.calls: list[tuple[str, str, object]] = []
        self.results: dict[str, object] = {}
        self.lock = threading.Lock()
        self.active_collect = 0
        self.max_active_collect = 0
        self.cold_barrier = threading.Barrier(5)
        self.lost_once = False
        self.gate_raw: bytes | None = None
        self.launch_requests: list[dict[str, object]] = []
        self.bootstrap_stack_ids: dict[str, str] = {}

    @staticmethod
    def _measurement_index(request: object) -> int:
        assert type(request) is dict
        event = request["event"]
        assert type(event) is dict
        return int(str(event["measurement_id"]).rsplit("-", 1)[1])

    def _launch_authority_result(
        self,
        request: dict[str, object],
    ) -> dict[str, object]:
        attempt = int(request["attempt"])
        body = {
            "account_id": ACCOUNT,
            "region": REGION,
            "run_id": RUN_ID,
            "activation_id": ACTIVATION,
            "attempt": attempt,
            "availability_zone": request["availability_zone"],
            "instance_type": "p5.48xlarge",
            "capacity_type": "ON_DEMAND",
            "same_token_identity_sha256": "1" * 64,
            "reserve_identity_sha256": "6" * 64,
            "spend_authority_identity_sha256": "7" * 64,
            "action_identity_sha256": "8" * 64,
            "task9_custody_identity_sha256": "9" * 64,
            "shared_attempt_counter": attempt,
            "immutable_inputs_identity_sha256": request[
                "immutable_inputs_identity_sha256"
            ],
            "sole_sender_authority": copy.deepcopy(
                request["sole_sender_authority"]
            ),
            "authority_record": _marker(
                (
                    f"campaigns/{RUN_ID}/authorities/task13/"
                    f"{ACTIVATION}/launch-attempt-{attempt:02d}.json"
                ),
                str(attempt),
            ),
            "spend_authority": {
                "remaining_gpu_seconds": 7200,
                "remaining_gpu_usd": "110.08",
                "gpu_reserve_seconds": 900,
                "gpu_reserve_usd": "13.76",
                "root_volume_tail_usd_max": "0.01",
                "ledger_version_id": "3LgSpendLedgerVersion",
                "ledger_body_sha256": "2" * 64,
            },
            "action_authority": {
                "action_kind": "PRODUCTION_SUBMISSION",
                "action_count": 1,
                "action_version_id": "3LgProductionAction",
                "action_body_sha256": "3" * 64,
            },
            "liability_action": {
                "action_kind": "SAME_TOKEN_COMPLETE",
                "attempt": attempt,
                "shared_attempt_counter": attempt,
                "action_version_id": "3LgLiabilityAction%02d" % attempt,
                "action_body_sha256": "4" * 64,
            },
        }
        if self.bad_launch_authority:
            body["spend_authority"]["gpu_reserve_usd"] = "13.77"
        return _self_hash(body)

    @staticmethod
    def _sole_sender_authority_result() -> dict[str, object]:
        return {
            "table_name": "keep-glm52-h1g-ledger-v1",
            "partition_key": RUN_ID,
            "sort_key": (
                f"ACTIVATION#{ACTIVATION}#"
                "TASK10_SOLE_SENDER_AUTHORITY#00000001"
            ),
            "authority_identity_sha256": "a" * 64,
            "source_closure_identity_sha256": "b" * 64,
        }

    def _collect_result(self, request: object) -> dict[str, object]:
        index = self._measurement_index(request)
        measurement_id = f"measurement-{index:02d}"
        payload = _self_hash(
            {
                "schema_version": 1,
                "record_type": "glm52_task11_collect_rehearsal_result_v1",
                "status": "DEPLOYED_REHEARSAL_RECORDED",
                "account_id": ACCOUNT,
                "region": REGION,
                "run_id": RUN_ID,
                "activation_id": ACTIVATION,
                "measurement_id": measurement_id,
                "collector_function_version_arn": COLLECTOR_ARN,
                "key": (
                    f"rehearsal/measurements/{ACTIVATION}/"
                    + ("a" * 64)
                    + f"/{measurement_id}.json"
                ),
                "version_id": f"version-{index:02d}",
                "file_sha256": hashlib.sha256(
                    measurement_id.encode()
                ).hexdigest(),
                "checksum_sha256_base64": base64.b64encode(
                    hashlib.sha256(measurement_id.encode()).digest()
                ).decode(),
            }
        )
        return {
            "StatusCode": 200,
            "ExecutedVersion": (
                "$LATEST" if self.wrong_version_at == index else "19"
            ),
            "Payload": canonical_json_bytes(payload),
        }

    def _finalize_result(self) -> dict[str, object]:
        measurements = tuple(
            rehearsal_measurement_from_mapping(_measurement(index))
            for index in range(1, 21)
        )
        gate_raw = build_deployed_gate_document(
            account_id=ACCOUNT,
            region=REGION,
            run_id=RUN_ID,
            activation_id=ACTIVATION,
            deployment_identity_sha256="d" * 64,
            measurements=measurements,
        )
        gate = json.loads(gate_raw)
        self.gate_raw = gate_raw
        payload = _self_hash(
            {
                "schema_version": 1,
                "record_type": (
                    "glm52_task11_finalize_rehearsal_gate_result_v1"
                ),
                "status": "CLOSURE_BUDGET_PROVEN",
                "account_id": ACCOUNT,
                "region": REGION,
                "run_id": RUN_ID,
                "activation_id": ACTIVATION,
                "collector_function_version_arn": COLLECTOR_ARN,
                "key": (
                    f"rehearsal/gates/{ACTIVATION}/"
                    "CLOSURE_BUDGET.json"
                ),
                "version_id": "gate-version-01",
                "file_sha256": hashlib.sha256(gate_raw).hexdigest(),
                "body_sha256": gate["canonical_body_sha256"],
                "checksum_sha256_base64": base64.b64encode(
                    hashlib.sha256(gate_raw).digest()
                ).decode(),
                "measurement_count": 20,
                "cold_environment_count": 5,
                "measurements_identity_sha256": gate[
                    "measurements_identity_sha256"
                ],
            }
        )
        return {
            "StatusCode": 200,
            "ExecutedVersion": "19",
            "Payload": canonical_json_bytes(payload),
        }

    def execute(
        self,
        operation_kind: str,
        operation_id: str,
        request: object,
    ) -> object:
        with self.lock:
            self.calls.append(("execute", operation_kind, operation_id))
        if operation_kind == "migration-bootstrap":
            stack_name = request["target_stack_name"]
            stack_ids = {
                "keep-glm52-h1g-fence": (
                    "arn:aws:cloudformation:us-west-2:246813579024:"
                    "stack/keep-glm52-h1g-fence/"
                    "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
                ),
                "keep-glm52-h1g-support": (
                    "arn:aws:cloudformation:us-west-2:246813579024:"
                    "stack/keep-glm52-h1g-support/"
                    "cccccccc-dddd-4eee-8fff-aaaaaaaaaaaa"
                ),
            }
            self.bootstrap_stack_ids[stack_name] = stack_ids[stack_name]
            result = {
                "status": "BOOTSTRAPPED_STACK",
                "retained_stack_id": request["retained_stack_id"],
                "stack_name": stack_name,
                "stack_id": stack_ids[stack_name],
                "termination_protection": True,
                "anchor_logical_id": "ContainerAnchor",
                "template_body_sha256": request["template"][
                    "body_sha256"
                ],
            }
        elif operation_kind == "collector-invoke":
            index = self._measurement_index(request)
            with self.lock:
                self.active_collect += 1
                self.max_active_collect = max(
                    self.max_active_collect,
                    self.active_collect,
                )
            if index <= 5:
                self.cold_barrier.wait(timeout=5)
            result = self._collect_result(request)
            with self.lock:
                self.active_collect -= 1
        elif operation_kind == "finalize-invoke":
            result = self._finalize_result()
        elif operation_kind == "qualification-cache-seed":
            manifest_sha = "9" * 64
            result = {
                "status": "SEED_READY",
                "instance_type": "p5.48xlarge",
                "capacity_type": "ON_DEMAND",
                "active_instance_count": 0,
                "gpu_seconds": 900,
                "teacher_row_count": 1,
                "qualification_cache_prefix": (
                    f"qualification-cache/seeds/{RUN_ID}/"
                    f"{manifest_sha}/"
                ),
                "qualification_cache_manifest_sha256": manifest_sha,
                "seed_ready": _marker(
                    f"campaigns/{RUN_ID}/qualification/"
                    "QUALIFICATION_CACHE_SEED_READY.json",
                    "7",
                ),
                "teacher_ready": _marker(
                    f"qualification-cache/seeds/{RUN_ID}/"
                    f"{manifest_sha}/TEACHER_CACHE_READY.json",
                    "8",
                ),
                "teardown_verified": True,
            }
        elif operation_kind == "h100-qualification":
            assert type(request) is dict
            result = {
                "status": "QUALIFIED",
                "instance_type": "p5.48xlarge",
                "capacity_type": "ON_DEMAND",
                "training_steps": 2,
                "cross_node_resume": True,
                "peak_memory_gib": 69,
                "instance_ids": [
                    "i-00000000000000001",
                    "i-00000000000000002",
                ],
                "qualification_cache_prefix": request[
                    "qualification_cache_prefix"
                ],
                "qualification_cache_manifest_sha256": request[
                    "qualification_cache_manifest_sha256"
                ],
                "source_ready": _marker(
                    f"campaigns/{RUN_ID}/qualification/"
                    "SOURCE_NODE_READY.json",
                    "a",
                ),
                "termination_requested": _marker(
                    f"campaigns/{RUN_ID}/qualification/"
                    "QUALIFICATION_TERMINATION_REQUESTED.json",
                    "b",
                ),
                "source_termination_verified": True,
                "max_concurrent_active_instances": 1,
                "active_instance_count": 0,
                "h100_resume_ready": _marker(
                    (
                        "campaigns/foreign-run/qualification/"
                        if self.foreign_h100_marker
                        else f"campaigns/{RUN_ID}/qualification/"
                    )
                    + "H100_RESUME_READY.json",
                    "c",
                ),
            }
        elif operation_kind == "sole-sender-authority-materialize":
            result = self._sole_sender_authority_result()
        elif operation_kind == "launch-authority-materialize":
            assert type(request) is dict
            result = self._launch_authority_result(request)
        elif operation_kind == "guarded-launch":
            assert type(request) is dict
            self.launch_requests.append(copy.deepcopy(request))
            accepted = self.launch_accept_at <= 6
            capacity_outcomes = [
                {
                    "attempt": index,
                    "availability_zone": (
                        f"us-west-2{'abcdef'[index - 1]}"
                    ),
                    "outcome": (
                        "WORKER_ALLOCATED"
                        if accepted and index == self.launch_accept_at
                        else "CAPACITY_REJECTED"
                    ),
                }
                for index in range(
                    1,
                    self.launch_accept_at + 1
                    if accepted
                    else 7,
                )
            ]
            result = {
                "status": (
                    "WORKER_ALLOCATED"
                    if accepted
                    else "CAPACITY_EXHAUSTED"
                ),
                "accepted": accepted,
                "workflow_invocation_count": 1,
                "instance_type": "p5.48xlarge",
                "capacity_type": "ON_DEMAND",
                "execution_arn": (
                    "arn:aws:states:us-west-2:246813579024:execution:"
                    "keep-glm52-h1g-production:task13-test"
                ),
                "instance_id": (
                    f"i-{self.launch_accept_at:017d}"
                    if accepted
                    else None
                ),
                "availability_zone": (
                    capacity_outcomes[-1]["availability_zone"]
                    if accepted
                    else None
                ),
                "capacity_outcomes": capacity_outcomes,
                "workflow_output": {
                    "bucket": (
                        "keep-glm52-models-"
                        "246813579024-us-west-2"
                    ),
                    "key": (
                        f"campaigns/{RUN_ID}/submissions/production/"
                        "generations/00000001/workflow/"
                        "LAUNCH_OUTCOME.json"
                    ),
                    "version_id": "3LgWorkflowOutput",
                    "file_sha256": "f" * 64,
                    "body_sha256": "e" * 64,
                },
                "same_token_identity_sha256": request[
                    "execution_authority"
                ]["same_token_identity_sha256"],
                "reserve_identity_sha256": "6" * 64,
                "spend_authority_identity_sha256": "7" * 64,
                "action_identity_sha256": "8" * 64,
                "task9_custody_identity_sha256": "9" * 64,
                "spend_authority": (
                    {
                        "account_id": ACCOUNT,
                        "run_id": RUN_ID,
                        "remaining_gpu_seconds": 7200,
                        "ledger_version_id": "spend-ledger-version",
                        "ledger_body_sha256": "d" * 64,
                    }
                    if accepted
                    else None
                ),
                "action_authority": (
                    {
                        "activation_id": request["activation_id"],
                        "action_kind": "PRODUCTION_SUBMISSION",
                        "action_count": 1,
                        "action_version_id": "action-version",
                        "action_body_sha256": "e" * 64,
                    }
                    if accepted
                    else None
                ),
                "terminal_contract": (
                    {
                        "required_terminal_markers": [
                            "CAMPAIGN_DRAINED.json",
                            "TERMINAL_VERIFIED.json",
                        ],
                        "monitor_route": (
                            "aws/glm52-gpu/scripts/"
                            "sky_campaign_break_glass.sh status"
                        ),
                    }
                    if accepted
                    else None
                ),
            }
            workflow_reconciliation = {
                "classification": result["status"],
                "execution_arn": result["execution_arn"],
                "capacity_outcomes": copy.deepcopy(
                    result["capacity_outcomes"]
                ),
                "instance_id": result["instance_id"],
                "availability_zone": result["availability_zone"],
                "workflow_output": copy.deepcopy(
                    result["workflow_output"]
                ),
                "same_token_identity_sha256": result[
                    "same_token_identity_sha256"
                ],
                "reserve_identity_sha256": result[
                    "reserve_identity_sha256"
                ],
                "spend_authority_identity_sha256": result[
                    "spend_authority_identity_sha256"
                ],
                "action_identity_sha256": result[
                    "action_identity_sha256"
                ],
                "task9_custody_identity_sha256": result[
                    "task9_custody_identity_sha256"
                ],
            }
            workflow_reconciliation[
                "observation_identity_sha256"
            ] = canonical_sha256(workflow_reconciliation)
            result["workflow_reconciliation"] = workflow_reconciliation
        elif operation_kind == "orphan-authority-precreate":
            assert type(request) is dict
            body = {
                "record_type": (
                    "glm52_task12_orphan_precreate_artifact_v1"
                ),
                "activation_id": request["activation_id"],
                "path": request["precreate_output"],
                "file_sha256": "1" * 64,
                "body_sha256": "2" * 64,
            }
            result = {**body, "canonical_identity_sha256": canonical_sha256(body)}
        elif operation_kind == "orphan-authority-postcreate":
            assert type(request) is dict
            authority = {
                "record_type": (
                    "glm52_task12_activation_orphan_coordinate_v1"
                ),
                "activation_id": request["activation_id"],
                "bucket": "keep-glm52-models-246813579024-us-west-2",
                "key": (
                    f"campaigns/{RUN_ID}/task12/orphans/"
                    f"{request['activation_id']}/"
                    + "3" * 64
                    + ".json"
                ),
                "version_id": "orphan-authority-version-1",
                "file_sha256": "4" * 64,
                "authority_body_sha256": "3" * 64,
            }
            support = {
                "path": (
                    str(request["postcreate_output_dir"])
                    + "/support-postcreate-manifest-v1.json"
                ),
                "file_sha256": "5" * 64,
                "body_sha256": "6" * 64,
            }
            body = {
                "schema_version": 1,
                "record_type": (
                    "glm52_task13_task12_orphan_activation_manifest_v1"
                ),
                "account_id": ACCOUNT,
                "region": REGION,
                "run_id": RUN_ID,
                "activation_id": request["activation_id"],
                "precreate_authority": request["precreate_authority"],
                "support_postcreate_manifest": support,
                "activation_authority": authority,
                "support_inventory_identity_sha256": request[
                    "support_inventory_identity_sha256"
                ],
                "path": request["activation_manifest"],
            }
            result = {**body, "canonical_identity_sha256": canonical_sha256(body)}
        else:
            assert type(request) is dict
            result = {
                "status": "COMMITTED",
                "operation": request["operation"],
                "request_identity_sha256": canonical_sha256(request),
            }
        self.results[operation_id] = copy.deepcopy(result)
        if (
            operation_id == self.lost_response_operation
            and not self.lost_once
        ):
            self.lost_once = True
            raise RuntimeError("simulated lost response")
        return result

    def reconcile(
        self,
        operation_kind: str,
        operation_id: str,
        request: object,
    ) -> object:
        with self.lock:
            self.calls.append(("reconcile", operation_kind, operation_id))
        if operation_id not in self.results:
            return {"state": "UNRESOLVED"}
        return {
            "state": "COMMITTED",
            "result": copy.deepcopy(self.results[operation_id]),
        }

    def inspect(self, operation_kind: str, request: object) -> object:
        with self.lock:
            self.calls.append(("inspect", operation_kind, copy.deepcopy(request)))
        if operation_kind == "staged-infrastructure-adoption":
            assert type(request) is dict
            observed = copy.deepcopy(request)
            if self.opaque_cf_evidence:
                observed["final_readback"]["support_stack_status"] = (
                    "UPDATE_IN_PROGRESS"
                )
            return observed
        if operation_kind == "migration-bootstrap-state":
            assert type(request) is dict
            names = [
                "keep-glm52-h1g-fence",
                "keep-glm52-h1g-support",
            ]
            if set(self.bootstrap_stack_ids) == set(names):
                return {
                    "status": "BOOTSTRAPPED",
                    "retained_stack_id": request["retained_stack_id"],
                    "fence_stack_id": self.bootstrap_stack_ids[names[0]],
                    "support_stack_id": self.bootstrap_stack_ids[names[1]],
                    "termination_protection": True,
                    "anchor_logical_id": "ContainerAnchor",
                    "template_body_sha256": request["template"][
                        "body_sha256"
                    ],
                }
            return {
                "status": "BOOTSTRAP_REQUIRED",
                "retained_stack_id": request["retained_stack_id"],
                "absent_stack_names": [
                    name
                    for name in names
                    if name not in self.bootstrap_stack_ids
                ],
                "existing_stack_ids": dict(self.bootstrap_stack_ids),
            }
        if operation_kind == "measurement":
            assert type(request) is dict
            payload = request["payload"]
            assert type(payload) is dict
            index = int(
                str(payload["measurement_id"]).rsplit("-", 1)[1]
            )
            return {
                "key": payload["key"],
                "version_id": payload["version_id"],
                "file_sha256": payload["file_sha256"],
                "measurement": _measurement(
                    index,
                    reused_environment=self.reused_environment,
                ),
            }
        if operation_kind == "gate":
            assert type(request) is dict
            payload = request["payload"]
            assert type(payload) is dict
            assert self.gate_raw is not None
            summary = {
                "key": payload["key"],
                "version_id": payload["version_id"],
                "file_sha256": payload["file_sha256"],
                "body_sha256": payload["body_sha256"],
                "status": payload["status"],
                "measurement_count": payload["measurement_count"],
                "cold_environment_count": payload["cold_environment_count"],
                "measurements_identity_sha256": payload[
                    "measurements_identity_sha256"
                ],
            }
            return {"summary": summary, "content": self.gate_raw}
        if operation_kind == "negative-iam-probe":
            return {"error_code": "AccessDenied"}
        if operation_kind == "p5-zero":
            return {"active_p5_instance_ids": []}
        if operation_kind == "monitor":
            return {
                "status": "RUNNING",
                "instance_ids": ["i-00000000000000006"],
            }
        if operation_kind == "credential-guard":
            observed = int(time.time()) - (7200 if self.stale_credentials else 0)
            return {
                "account_id": ACCOUNT,
                "observed_epoch_seconds": observed,
                "expiration_epoch_seconds": observed + 7200,
                "seconds_remaining": 7200,
            }
        if operation_kind == "immutable-marker":
            assert type(request) is dict
            observed = copy.deepcopy(request)
            if self.marker_readback_drift:
                observed["body_sha256"] = "f" * 64
            return observed
        if operation_kind == "immutable-artifact":
            assert type(request) is dict
            observed = copy.deepcopy(request)
            if self.artifact_readback_drift:
                observed["body_sha256"] = "f" * 64
            return observed
        if operation_kind == "terminal-proof":
            body = {
                "status": "TERMINAL_PROVEN",
                "account_id": ACCOUNT,
                "region": REGION,
                "run_id": RUN_ID,
                "activation_id": ACTIVATION,
                "sky_state": "SUCCEEDED",
                "ec2_billable_instance_ids": [],
                "campaign_drained": _marker(
                    f"campaigns/{RUN_ID}/submissions/production/generations/"
                    "00000001/terminal/CAMPAIGN_DRAINED.json",
                    "5",
                ),
                "terminal_verified": _marker(
                    f"campaigns/{RUN_ID}/submissions/production/generations/"
                    "00000001/terminal/TERMINAL_VERIFIED.json",
                    "6",
                ),
                "allocations": [
                    {
                        "allocation_kind": "qualification-cache-seed",
                        "instance_id": "i-00000000000000001",
                        "gpu_seconds": 900,
                        "gpu_cost_usd": "13.76",
                    },
                    {
                        "allocation_kind": "production",
                        "instance_id": "i-00000000000000006",
                        "gpu_seconds": 3600,
                        "gpu_cost_usd": "55.04",
                    },
                ],
                "total_gpu_seconds": 4500,
                "total_gpu_cost_usd": "68.80",
                "remaining_gpu_seconds": 81900,
                "remaining_gpu_usd": "1252.16",
                "retained_costs": {
                    name: "0.00"
                    for name in (
                        "controller",
                        "storage",
                        "checksum",
                        "network",
                        "lambda",
                        "logs",
                        "workflow",
                        "queue",
                        "notification",
                        "root_volume",
                        "incident_tail",
                    )
                },
            }
            if self.terminal_proof_drift:
                body["ec2_billable_instance_ids"] = [
                    "i-00000000000000006"
                ]
            return {
                "proof": _self_hash(body),
                "terminal_verified_value": {
                    "fixture": "terminal-verified"
                },
                "campaign_drained_value": {
                    "fixture": "campaign-drained"
                },
                "spend_ledger_raw": b"fixture-spend-ledger\n",
            }
        assert type(request) is dict
        command_id = request["command_id"]
        if command_id in {"read-caller-identity", "assert-caller-account"}:
            evidence = {"account_id": ACCOUNT}
        elif command_id == "read-credential-expiry":
            observed = int(time.time()) - (7200 if self.stale_credentials else 0)
            evidence = {
                "account_id": ACCOUNT,
                "observed_epoch_seconds": observed,
                "expiration_epoch_seconds": observed + 7200,
                "seconds_remaining": 7200,
            }
        elif command_id in {
            "read-deployment-cloudformation-role",
            "read-fence-cloudformation-role",
        }:
            role_name = (
                "keep-glm52-h1g-fence-service"
                if command_id == "read-fence-cloudformation-role"
                else "keep-glm52-h1g-cloudformation-deployment"
            )
            evidence = {
                "role_arn": (
                    "arn:aws:iam::246813579024:role/" + role_name
                ),
                "role_id": "AROAEXACTROLEID",
            }
        elif str(request["operation"]).startswith("cloudformation:"):
            if self.opaque_cf_evidence:
                evidence = {
                    "semantic_status": "VERIFIED",
                    "resource_identity_sha256": canonical_sha256(request),
                }
            else:
                argv = request["argv"]
                stack_id = argv[argv.index("--stack-name") + 1]
                label = next(
                    candidate
                    for candidate in ("retained", "fence", "support")
                    if candidate in command_id
                )
                template_kind = {
                    "retained": "RETAINED_TEMPLATE",
                    "fence": "FENCE_TEMPLATE",
                    "support": "SUPPORT_TEMPLATE",
                }[label]
                template_sha = hashlib.sha256(
                    template_kind.encode()
                ).hexdigest()
                if command_id.startswith("describe-"):
                    change_set_name = argv[
                        argv.index("--change-set-name") + 1
                    ]
                    changes = [
                        {
                            "action": "Modify",
                            "logical_resource_id": (
                                "H1gProductionFenceBucketPolicy"
                                if label == "fence"
                                else "DisabledSupportPlane"
                            ),
                            "resource_type": (
                                "AWS::S3::BucketPolicy"
                                if label == "fence"
                                else "AWS::StepFunctions::StateMachine"
                            ),
                            "replacement": "False",
                        }
                    ]
                    evidence = {
                        "change_set_arn": (
                            "arn:aws:cloudformation:us-west-2:"
                            "246813579024:changeSet/"
                            f"{change_set_name}/"
                            "dddddddd-eeee-4fff-8aaa-bbbbbbbbbbbb"
                        ),
                        "stack_id": stack_id,
                        "status": "CREATE_COMPLETE",
                        "execution_status": "AVAILABLE",
                        "change_set_type": "UPDATE",
                        "role_arn": (
                            "arn:aws:iam::246813579024:role/"
                            + (
                                "keep-glm52-h1g-fence-service"
                                if label == "fence"
                                else (
                                    "keep-glm52-h1g-"
                                    "cloudformation-deployment"
                                )
                            )
                        ),
                        "template_body_sha256": template_sha,
                        "changes": changes,
                        "changes_identity_sha256": canonical_sha256(changes),
                    }
                elif command_id.startswith("inspect-"):
                    change_set_name = argv[
                        argv.index("--change-set-name") + 1
                    ]
                    evidence = {
                        "change_set_arn": (
                            "arn:aws:cloudformation:us-west-2:"
                            "246813579024:changeSet/"
                            f"{change_set_name}/"
                            "dddddddd-eeee-4fff-8aaa-bbbbbbbbbbbb"
                        ),
                        "stack_id": stack_id,
                        "template_stage": "Processed",
                        "template_body_sha256": template_sha,
                    }
                elif command_id.startswith("readback-") and command_id.endswith(
                    "-stack"
                ):
                    evidence = {
                        "stack_id": stack_id,
                        "stack_status": "UPDATE_COMPLETE",
                        "termination_protection": True,
                        "tags_identity_sha256": canonical_sha256(
                            [
                                ["Authority", "H1g"],
                                ["Campaign", "GLM-5.2"],
                                ["Environment", "production"],
                                ["ManagedBy", "CloudFormation"],
                                ["Project", "KEEP"],
                                ["RunId", RUN_ID],
                            ]
                        ),
                    }
                elif command_id.startswith("inventory-"):
                    resources = [
                        {
                            "logical_resource_id": "ContainerAnchor",
                            "resource_type": (
                                "AWS::CloudFormation::WaitConditionHandle"
                            ),
                            "resource_status": "CREATE_COMPLETE",
                        }
                    ]
                    evidence = {
                        "stack_id": stack_id,
                        "resource_summaries": resources,
                        "resource_summaries_identity_sha256": (
                            canonical_sha256(resources)
                        ),
                    }
                else:
                    evidence = {
                        "stack_id": stack_id,
                        "template_stage": "Original",
                        "template_body_sha256": template_sha,
                    }
        else:
            raise AssertionError("unexpected inspection command")
        return {
            "status": "INSPECTED",
            "operation": request["operation"],
            "request_identity_sha256": canonical_sha256(request),
            "evidence": evidence,
        }


def _issued_authority(
    package: dict[str, object],
    stage: str,
    coordinator_executable_sha256: str,
):
    key = (
        str(package["canonical_identity_sha256"]),
        stage,
        coordinator_executable_sha256,
    )
    authority = _SIGNED_AUTHORITY_CACHE.get(key)
    if authority is None:
        authority = issue_controller_execution_authority(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage=stage,
            ttl_seconds=900,
            coordinator_executable_sha256=(
                coordinator_executable_sha256
            ),
            owner_approval_sha256=CAMPAIGN_OWNER_APPROVAL_SHA256,
            signer=SIGNER,
        )
        _SIGNED_AUTHORITY_CACHE[key] = authority
    return authority


def _authority(
    api: object,
    package: dict[str, object],
    stage: str,
    **updates: object,
):
    authority = _issued_authority(package, stage, COORDINATOR_SHA)
    assert type(authority) is api.ControllerExecutionAuthority
    return replace(authority, **updates) if updates else authority


def test_external_stage_rejects_expired_or_foreign_coordinator_authority() -> None:
    """Break caught: a local boolean file authorizes an arbitrary coordinator."""

    api = _api()
    package = _package()
    services = FiniteServices()
    journal = MemoryJournal()
    now = int(time.time())
    for authority in (
        _authority(
            api,
            package,
            "deploy-disabled",
            issued_at_epoch_seconds=now - 901,
            expires_at_epoch_seconds=now - 1,
        ),
        _authority(
            api,
            package,
            "deploy-disabled",
            coordinator_executable_sha256="f" * 64,
        ),
        _authority(
            api,
            package,
            "deploy-disabled",
            owner_approval_sha256="f" * 64,
        ),
        _authority(
            api,
            package,
            "deploy-disabled",
            execution_custody_identity_sha256="f" * 64,
        ),
    ):
        with pytest.raises(api.CampaignRunnerError, match="authority"):
            api.run_campaign_stage(
                package=package,
                reviewed_artifacts=package["reviewed_artifacts"],
                stage="deploy-disabled",
                services=services,
                journal=journal,
                authority=authority,
            )
    assert services.calls == []


def test_mutation_refuses_stale_credential_observation_before_possibly_sent() -> None:
    """Break caught: replayed expiry arithmetic authorizes a current mutation."""

    api = _api()
    services = FiniteServices(stale_credentials=True)
    journal = MemoryJournal().for_scope("campaign")
    with pytest.raises(api.CampaignRunnerError, match="credential"):
        api._mutation(
            services=services,
            journal=journal,
            stage="collect-first-five",
            operation_kind="collector-invoke",
            operation_id="collect:stale-credential-proof",
            request={"request": "exact"},
            validate_result=lambda value: value,
        )
    assert not any(call[0] == "execute" for call in services.calls)


def test_validate_stage_is_default_inert_and_binds_reviewed_artifacts() -> None:
    """Break caught: package validation performs AWS work or accepts new bytes."""

    api = _api()
    package = _package()
    journal = MemoryJournal()
    result = api.run_campaign_stage(
        package=package,
        reviewed_artifacts=copy.deepcopy(package["reviewed_artifacts"]),
        stage="validate",
        services=NoExternalServices(),
        journal=journal,
        authority=None,
    )
    assert result["status"] == "VALIDATED_NO_EXTERNAL_EXECUTION"
    assert result["stage"] == "validate"
    assert result["package_identity_sha256"] == package[
        "canonical_identity_sha256"
    ]
    assert result["reviewed_artifacts_identity_sha256"] == canonical_sha256(
        package["reviewed_artifacts"]
    )
    assert journal.rows == []

    drifted = copy.deepcopy(package["reviewed_artifacts"])
    drifted[0]["body_sha256"] = "f" * 64
    with pytest.raises(api.CampaignRunnerError, match="reviewed artifacts"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=drifted,
            stage="validate",
            services=NoExternalServices(),
            journal=journal,
            authority=None,
        )

    forged = copy.deepcopy(package)
    forged["semantic_transport_gates"]["T01_T25_GATE"]["row_count"] = 24
    unhashed = dict(forged)
    del unhashed["canonical_identity_sha256"]
    forged["canonical_identity_sha256"] = canonical_sha256(unhashed)
    with pytest.raises(
        api.CampaignRunnerError,
        match="campaign package identity",
    ):
        api.run_campaign_stage(
            package=forged,
            reviewed_artifacts=forged["reviewed_artifacts"],
            stage="validate",
            services=NoExternalServices(),
            journal=journal,
            authority=None,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("bucket", "foreign-campaign-bucket"),
        ("key", "task13/templates/foreign-retained.yaml"),
    ],
)
def test_runner_rejects_coherently_rehashed_foreign_artifact_coordinate(
    field: str,
    replacement: str,
) -> None:
    """Break caught: runner trusts a rehashed foreign reviewed coordinate."""

    from glm52_enforcement import task13_campaign_runner as api

    package = _package()
    coordinate = next(
        row
        for row in package["reviewed_artifacts"]
        if row["artifact_kind"] == "RETAINED_TEMPLATE"
    )
    coordinate[field] = replacement
    unhashed = dict(package)
    del unhashed["canonical_identity_sha256"]
    package["canonical_identity_sha256"] = canonical_sha256(unhashed)

    with pytest.raises(
        api.CampaignRunnerError,
        match="campaign package identity",
    ):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=copy.deepcopy(
                package["reviewed_artifacts"]
            ),
            stage="validate",
            services=NoExternalServices(),
            journal=MemoryJournal(),
            authority=None,
        )


def test_launch_bridge_builder_binds_package_artifacts_and_route(
    tmp_path: Path,
) -> None:
    from glm52_enforcement.task10_task13_bridge import (
        task13_launch_bridge_from_mapping,
    )

    package = _package()
    package_path = (tmp_path / "package.json").resolve()
    reviewed_path = (tmp_path / "reviewed.json").resolve()
    controller_path = (tmp_path / "controller.json").resolve()
    h100_path = (tmp_path / "h100-coordinate.json").resolve()
    output_path = (tmp_path / "launch-bridge.json").resolve()
    package_path.write_bytes(canonical_campaign_package_bytes(package))
    reviewed_path.write_bytes(
        canonical_json_bytes(package["reviewed_artifacts"]) + b"\n"
    )
    controller_path.write_bytes(
        canonical_json_bytes({"activation_id": ACTIVATION}) + b"\n"
    )
    h100_path.write_bytes(
        canonical_json_bytes(
            {
                "bucket": (
                    "keep-glm52-models-246813579024-us-west-2"
                ),
                "key": (
                    "campaigns/glm52-sky-20260724/qualification/"
                    "H100_RESUME_READY.json"
                ),
                "version_id": "3LgH100ResumeReady",
                "file_sha256": "a" * 64,
                "body_sha256": "b" * 64,
            }
        )
        + b"\n"
    )
    script = (
        Path(__file__).parents[1]
        / "aws/glm52-gpu/scripts/build_glm52_task13_launch_bridge.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--package",
            str(package_path),
            "--reviewed-artifacts",
            str(reviewed_path),
            "--controller-authority",
            str(controller_path),
            "--h100-resume-ready-coordinate",
            str(h100_path),
            "--task13-route-binding-identity",
            "e" * 64,
            "--fence-journal",
            str((tmp_path / "fence.jsonl").resolve()),
            "--support-journal",
            str((tmp_path / "support.jsonl").resolve()),
            "--campaign-journal",
            str((tmp_path / "campaign.jsonl").resolve()),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    bridge = task13_launch_bridge_from_mapping(
        json.loads(output_path.read_bytes())
    )
    assert bridge.package_path == str(package_path)
    assert bridge.reviewed_artifacts_path == str(reviewed_path)
    assert bridge.task13_route_binding_identity_sha256 == "e" * 64


def test_prequalification_cannot_launch_and_successor_authenticates_journal() -> None:
    """Break caught: a launch adopts an ambient H100 completion row."""

    api = _api()
    prequalification = _package("PREQUALIFICATION")
    production = _package()
    services = FiniteServices()
    journal = MemoryJournal()
    with pytest.raises(
        ControllerAuthorityIssuerError,
        match="prequalification package cannot",
    ):
        _authority(
            api,
            prequalification,
            "launch",
        )
    assert services.calls == []

    campaign_journal = journal.for_scope("campaign")
    api._record_stage(
        campaign_journal,
        stage="h100-qualification",
        package_identity_sha256=prequalification[
            "canonical_identity_sha256"
        ],
        reviewed_artifacts_identity_sha256=canonical_sha256(
            prequalification["reviewed_artifacts"]
        ),
        summary={"status": "QUALIFIED"},
    )
    api._require_stage(
        campaign_journal,
        "h100-qualification",
        package=production,
    )


def test_stage_journal_rejects_cross_package_replay() -> None:
    """Break caught: a committed stage name is treated as global authority."""

    api = _api()
    package = _package()
    journal = MemoryJournal().for_scope("campaign")
    api._record_stage(
        journal,
        stage="h100-qualification",
        package_identity_sha256="0" * 64,
        reviewed_artifacts_identity_sha256="1" * 64,
        summary={"status": "QUALIFIED"},
    )
    with pytest.raises(api.CampaignRunnerError, match="foreign package"):
        api._require_stage(
            journal,
            "h100-qualification",
            package=package,
        )


def test_production_stage_rejects_prequalification_launch_replay() -> None:
    """Break caught: terminal adopts an impossible predecessor launch row."""

    api = _api()
    prequalification = _package("PREQUALIFICATION")
    production = _package()
    journal = MemoryJournal().for_scope("campaign")
    api._record_stage(
        journal,
        stage="launch",
        package_identity_sha256=prequalification[
            "canonical_identity_sha256"
        ],
        reviewed_artifacts_identity_sha256=canonical_sha256(
            prequalification["reviewed_artifacts"]
        ),
        summary={"status": "STARTED"},
    )
    with pytest.raises(api.CampaignRunnerError, match="foreign package"):
        api._require_stage(
            journal,
            "launch",
            package=production,
        )


def test_cli_defaults_to_validate_without_coordinator_or_journal(
    tmp_path: Path,
) -> None:
    """Break caught: invoking the CLI with no stage mutates AWS."""

    package = _package()
    package_path = tmp_path / "package.json"
    artifacts_path = tmp_path / "artifacts.json"
    package_path.write_bytes(canonical_campaign_package_bytes(package))
    artifacts_path.write_bytes(
        canonical_json_bytes(package["reviewed_artifacts"]) + b"\n"
    )
    script = (
        Path(__file__).parents[1]
        / "aws/glm52-gpu/scripts/run_glm52_task13_campaign.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--package",
            str(package_path),
            "--reviewed-artifacts",
            str(artifacts_path),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["status"] == "VALIDATED_NO_EXTERNAL_EXECUTION"
    assert parsed["stage"] == "validate"
    assert not (tmp_path / "journal.jsonl").exists()


def test_cli_external_stage_requires_durable_pinned_controller(
    tmp_path: Path,
) -> None:
    """Break caught: an explicit live stage falls back to ambient clients."""

    package = _package()
    package_path = tmp_path / "package.json"
    artifacts_path = tmp_path / "artifacts.json"
    package_path.write_bytes(canonical_campaign_package_bytes(package))
    artifacts_path.write_bytes(
        canonical_json_bytes(package["reviewed_artifacts"]) + b"\n"
    )
    script = (
        Path(__file__).parents[1]
        / "aws/glm52-gpu/scripts/run_glm52_task13_campaign.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--package",
            str(package_path),
            "--reviewed-artifacts",
            str(artifacts_path),
            "--stage",
            "deploy-disabled",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert "SHA-pinned coordinator" in result.stderr
    assert not (tmp_path / "journal.jsonl").exists()


@pytest.mark.parametrize(
    "omitted",
    ["--full-run-work", "--signing-private-key"],
)
def test_cli_external_stage_requires_explicit_signing_custody(
    tmp_path: Path,
    omitted: str,
) -> None:
    """Break caught: live signing falls back to an ambient work root or key."""

    api = _api()
    package = _package()
    package_path = (tmp_path / "package.json").resolve()
    artifacts_path = (tmp_path / "artifacts.json").resolve()
    authority_path = (tmp_path / "authority.json").resolve()
    journal_paths = [
        (tmp_path / name).resolve()
        for name in (
            "fence-journal.jsonl",
            "support-journal.jsonl",
            "campaign-journal.jsonl",
        )
    ]
    package_path.write_bytes(canonical_campaign_package_bytes(package))
    artifacts_path.write_bytes(
        canonical_json_bytes(package["reviewed_artifacts"]) + b"\n"
    )
    authority_path.write_bytes(
        canonical_json_bytes(
            asdict(_authority(api, package, "deploy-disabled"))
        )
        + b"\n"
    )
    arguments = [
        sys.executable,
        str(
            Path(__file__).parents[1]
            / "aws/glm52-gpu/scripts/run_glm52_task13_campaign.py"
        ),
        "--package",
        str(package_path),
        "--reviewed-artifacts",
        str(artifacts_path),
        "--stage",
        "deploy-disabled",
        "--authority",
        str(authority_path),
        "--fence-journal",
        str(journal_paths[0]),
        "--support-journal",
        str(journal_paths[1]),
        "--campaign-journal",
        str(journal_paths[2]),
    ]
    for flag, value in (
        ("--full-run-work", FULL_RUN_WORK),
        ("--signing-private-key", SIGNING_PRIVATE_KEY),
    ):
        if flag != omitted:
            arguments.extend([flag, str(value)])
    result = subprocess.run(
        arguments,
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert "external stage requires authority" in result.stderr
    assert all(not path.exists() for path in journal_paths)


@pytest.mark.parametrize(
    ("stage", "package_phase", "capture_arguments"),
    [
        ("deploy-disabled", "prequalification", ()),
        ("collect-first-five", "prequalification", ()),
        ("collect-remaining", "prequalification", ()),
        (
            "finalize",
            "prequalification",
            (
                "--finalize-invoke-metadata-output",
                "/private/tmp/finalize-invoke-metadata.json",
                "--finalize-payload-output",
                "/private/tmp/finalize-payload.json",
                "--gate-readback-output",
                "/private/tmp/finalize-gate-readback.json",
            ),
        ),
        ("qualification-cache-seed", "prequalification", ()),
        ("h100-qualification", "prequalification", ()),
        ("launch", "production", ()),
        ("monitor", "production", ()),
        ("terminal", "production", ()),
    ],
)
def test_runbook_external_runner_vectors_parse_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    package_phase: str,
    capture_arguments: tuple[str, ...],
) -> None:
    """Break caught: an operator runner vector omits explicit signing custody."""

    module = _script_api("run_glm52_task13_campaign.py")
    full_run_work = Path("/private/tmp/glm52-full-run-reviewed")
    signing_private_key = (
        full_run_work / "controller-authority-ed25519"
    )
    observed: dict[str, object] = {}

    class Capture:
        def commit(self) -> None:
            observed["capture_committed"] = True

        def abort(self) -> None:
            observed["capture_aborted"] = True

    def signer(
        *,
        private_key_path: Path,
        full_run_work: Path,
    ) -> object:
        observed["private_key_path"] = private_key_path
        observed["full_run_work"] = full_run_work
        return object()

    def run_campaign_stage(**kwargs: object) -> dict[str, object]:
        observed["stage"] = kwargs["stage"]
        observed["finalization_capture"] = kwargs[
            "finalization_capture"
        ]
        return {"status": "STAGE_COMMITTED"}

    monkeypatch.setattr(module, "_read_canonical", lambda _path: {})
    monkeypatch.setattr(module, "_authority", lambda _value: object())
    monkeypatch.setattr(module, "_full_run_work", lambda path: path)
    monkeypatch.setattr(module, "ControllerAuthoritySigner", signer)
    monkeypatch.setattr(
        module,
        "SubprocessCoordinator",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(module, "FileJournalStore", lambda path: path)
    monkeypatch.setattr(
        module,
        "CampaignJournalStores",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        module,
        "_FinalizationCaptureFiles",
        lambda **_kwargs: Capture(),
    )
    monkeypatch.setattr(module, "run_campaign_stage", run_campaign_stage)

    arguments = [
        "--package",
        f"/private/tmp/{package_phase}-package.json",
        "--reviewed-artifacts",
        f"/private/tmp/{package_phase}-reviewed.json",
        "--stage",
        stage,
        "--authority",
        "/private/tmp/stage-authority.json",
        "--full-run-work",
        str(full_run_work),
        "--signing-private-key",
        str(signing_private_key),
        "--fence-journal",
        "/private/tmp/fence.jsonl",
        "--support-journal",
        "/private/tmp/support.jsonl",
        "--campaign-journal",
        "/private/tmp/campaign.jsonl",
        *capture_arguments,
    ]
    assert module.main(arguments) == 0
    assert observed["stage"] == stage
    assert observed["full_run_work"] == full_run_work
    assert observed["private_key_path"] == signing_private_key
    assert (observed["finalization_capture"] is not None) == (
        stage == "finalize"
    )


def test_prequalification_builder_runbook_vector_requires_staged_evidence(
) -> None:
    """Break caught: the prequalification builder exits before construction."""

    module = _script_api(
        "build_glm52_task13_prequalification_sources.py"
    )
    coordinate_sources = [
        f"/private/tmp/source-{index:02d}.json"
        for index in range(18)
    ]
    arguments = [
        "--activation-id",
        ACTIVATION,
        "--collector-version-arn",
        COLLECTOR_ARN,
        "--task11-request",
        "/private/tmp/task11-request.json",
        "--task11-boundary",
        "/private/tmp/task11-boundary.json",
        "--staged-infrastructure-evidence",
        "/private/tmp/staged-infrastructure-evidence.json",
        "--retained-stack-id",
        RETAINED_STACK_ID,
        "--fence-change-set-name",
        "task13-fence-disabled",
        "--support-change-set-name",
        "task13-support-disabled",
        "--monitor-descriptor-path",
        "/private/tmp/campaign-descriptor.json",
    ]
    for source in coordinate_sources:
        arguments.extend(["--coordinate-source", source])
    arguments.extend(
        [
            "--base-request-output",
            "/private/tmp/base-request.json",
            "--coordinates-output",
            "/private/tmp/prequalification-21.json",
            "--request-output",
            "/private/tmp/prequalification-request.json",
            "--reviewed-artifacts-output",
            "/private/tmp/prequalification-reviewed.json",
        ]
    )

    parsed = module._parser().parse_args(arguments)
    assert parsed.staged_infrastructure_evidence == Path(
        "/private/tmp/staged-infrastructure-evidence.json"
    )
    assert parsed.coordinate_source == [
        Path(source) for source in coordinate_sources
    ]

    evidence_index = arguments.index(
        "--staged-infrastructure-evidence"
    )
    omitted = (
        arguments[:evidence_index]
        + arguments[evidence_index + 2 :]
    )
    with pytest.raises(SystemExit) as caught:
        module._parser().parse_args(omitted)
    assert caught.value.code == 2


def test_archive_publisher_runbook_vector_requires_manifest_output(
) -> None:
    """Break caught: archive publication leaves no exact local manifest."""

    module = _script_api(
        "materialize_glm52_task13_fixed_artifacts.py"
    )
    arguments = [
        "publish-archive",
        "--activation-id",
        ACTIVATION,
        "--archive",
        "/private/tmp/repo.tar.gz",
        "--bucket",
        "keep-glm52-models-246813579024-us-west-2",
        "--manifest-output",
        "/private/tmp/repository-archive-manifest.json",
        "--coordinate-output",
        "/private/tmp/repository-archive.json",
    ]

    parsed = module._parser().parse_args(arguments)
    assert parsed.manifest_output == Path(
        "/private/tmp/repository-archive-manifest.json"
    )
    manifest_index = arguments.index("--manifest-output")
    omitted = (
        arguments[:manifest_index]
        + arguments[manifest_index + 2 :]
    )
    with pytest.raises(SystemExit) as caught:
        module._parser().parse_args(omitted)
    assert caught.value.code == 2


def test_production_reviewed_joiner_runbook_vector_has_exact_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: production authority is assembled by an inline join."""

    module = _script_api(
        "build_glm52_production_reviewed_artifacts.py"
    )
    arguments = [
        "--prequalification-reviewed-artifacts",
        "/private/tmp/prequalification-reviewed.json",
        "--clean-rehearsal-coordinate",
        "/private/tmp/clean-rehearsal-coordinate.json",
        "--task10-production-authority",
        "/private/tmp/task10-production-authority.json",
        "--task10-worker-descriptor",
        "/private/tmp/task10-worker-descriptor.json",
        "--task10-task-inputs",
        "/private/tmp/task10-task-inputs.json",
        "--output",
        "/private/tmp/production-reviewed-25.json",
    ]
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "_read",
        lambda path, _label: path,
    )

    def build_production_reviewed_artifacts(
        **kwargs: object,
    ) -> list[object]:
        observed["inputs"] = kwargs
        return []

    monkeypatch.setattr(
        module,
        "build_production_reviewed_artifacts",
        build_production_reviewed_artifacts,
    )
    monkeypatch.setattr(
        module,
        "_write_new",
        lambda path, value: observed.update(
            output=path,
            value=value,
        ),
    )

    assert module.main(arguments) == 0
    assert observed["inputs"] == {
        "prequalification_reviewed_artifacts": Path(
            "/private/tmp/prequalification-reviewed.json"
        ),
        "clean_rehearsal_coordinate": Path(
            "/private/tmp/clean-rehearsal-coordinate.json"
        ),
        "task10_production_authority": Path(
            "/private/tmp/task10-production-authority.json"
        ),
        "task10_worker_descriptor": Path(
            "/private/tmp/task10-worker-descriptor.json"
        ),
        "task10_task_inputs": Path(
            "/private/tmp/task10-task-inputs.json"
        ),
    }
    assert observed["output"] == Path(
        "/private/tmp/production-reviewed-25.json"
    )


@pytest.mark.parametrize(
    "omitted",
    [
        "--prequalification-reviewed-artifacts",
        "--clean-rehearsal-coordinate",
        "--task10-production-authority",
        "--task10-worker-descriptor",
        "--task10-task-inputs",
        "--output",
    ],
)
def test_production_reviewed_joiner_rejects_missing_runbook_input(
    omitted: str,
) -> None:
    """Break caught: a production-only coordinate silently disappears."""

    module = _script_api(
        "build_glm52_production_reviewed_artifacts.py"
    )
    arguments = [
        "--prequalification-reviewed-artifacts",
        "/private/tmp/prequalification-reviewed.json",
        "--clean-rehearsal-coordinate",
        "/private/tmp/clean-rehearsal-coordinate.json",
        "--task10-production-authority",
        "/private/tmp/task10-production-authority.json",
        "--task10-worker-descriptor",
        "/private/tmp/task10-worker-descriptor.json",
        "--task10-task-inputs",
        "/private/tmp/task10-task-inputs.json",
        "--output",
        "/private/tmp/production-reviewed-25.json",
    ]
    omitted_index = arguments.index(omitted)
    incomplete = (
        arguments[:omitted_index]
        + arguments[omitted_index + 2 :]
    )
    with pytest.raises(SystemExit) as caught:
        module.main(incomplete)
    assert caught.value.code == 2


def test_cli_rejects_aliasing_fence_support_and_campaign_journals(
    tmp_path: Path,
) -> None:
    """Break caught: one file authorizes replay across three effect domains."""

    api = _api()
    package = _package()
    package_path = tmp_path / "package.json"
    artifacts_path = tmp_path / "artifacts.json"
    authority_path = tmp_path / "authority.json"
    journal_path = (tmp_path / "aliased.jsonl").resolve()
    package_path.write_bytes(canonical_campaign_package_bytes(package))
    artifacts_path.write_bytes(
        canonical_json_bytes(package["reviewed_artifacts"]) + b"\n"
    )
    authority_path.write_bytes(
        canonical_json_bytes(
            asdict(_authority(api, package, "deploy-disabled"))
        )
        + b"\n"
    )
    script = (
        Path(__file__).parents[1]
        / "aws/glm52-gpu/scripts/run_glm52_task13_campaign.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--package",
            str(package_path),
            "--reviewed-artifacts",
            str(artifacts_path),
            "--stage",
            "deploy-disabled",
            "--authority",
            str(authority_path),
            "--full-run-work",
            str(FULL_RUN_WORK),
            "--signing-private-key",
            str(SIGNING_PRIVATE_KEY),
            "--fence-journal",
            str(journal_path),
            "--support-journal",
            str(journal_path),
            "--campaign-journal",
            str(journal_path),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert "distinct" in result.stderr
    assert not journal_path.exists()


def test_file_journal_is_canonical_append_only_and_private(
    tmp_path: Path,
) -> None:
    """Break caught: resumability depends on process memory or loose files."""

    api = _api()
    path = (tmp_path / "journal.jsonl").resolve()
    journal = api.FileJournalStore(path)
    row = {
        "operation_id": "test:one",
        "state": "PREPARED",
    }
    journal.append(row)
    assert journal.entries("test:one") == (row,)
    assert path.read_bytes() == canonical_json_bytes(row) + b"\n"
    assert path.stat().st_mode & 0o777 == 0o600


def test_file_journal_rejects_loose_existing_mode_and_fsyncs_first_create_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: crash loses POSSIBLY_SENT and restart replays a mutation."""

    api = _api()
    loose = (tmp_path / "loose.jsonl").resolve()
    loose.write_bytes(b"")
    loose.chmod(0o644)
    with pytest.raises(api.CampaignRunnerError, match="0600"):
        api.FileJournalStore(loose)

    path = (tmp_path / "durable.jsonl").resolve()
    fsynced_modes: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        fsynced_modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(api.os, "fsync", recording_fsync)
    journal = api.FileJournalStore(path)
    journal.append({"operation_id": "test:first", "state": "PREPARED"})
    assert any(stat.S_ISREG(mode) for mode in fsynced_modes)
    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)


def test_campaign_journal_stores_reject_path_inode_and_symlink_aliases(
    tmp_path: Path,
) -> None:
    """Break caught: two stack journals resolve to one replay authority."""

    api = _api()
    fence_path = (tmp_path / "fence.jsonl").resolve()
    support_path = (tmp_path / "support.jsonl").resolve()
    campaign_path = (tmp_path / "campaign.jsonl").resolve()
    stores = api.CampaignJournalStores(
        fence=api.FileJournalStore(fence_path),
        support=api.FileJournalStore(support_path),
        campaign=api.FileJournalStore(campaign_path),
    )
    assert stores.for_scope("fence").path == fence_path
    assert stores.for_scope("support").path == support_path
    assert stores.for_scope("campaign").path == campaign_path

    with pytest.raises(api.CampaignRunnerError, match="distinct"):
        api.CampaignJournalStores(
            fence=api.FileJournalStore(fence_path),
            support=api.FileJournalStore(fence_path),
            campaign=api.FileJournalStore(campaign_path),
        )

    fence_path.write_bytes(b"")
    fence_path.chmod(0o600)
    os.link(fence_path, support_path)
    with pytest.raises(api.CampaignRunnerError, match="alias"):
        api.CampaignJournalStores(
            fence=api.FileJournalStore(fence_path),
            support=api.FileJournalStore(support_path),
            campaign=api.FileJournalStore(campaign_path),
        )
    support_path.unlink()
    support_path.symlink_to(fence_path)
    with pytest.raises(api.CampaignRunnerError, match="symlink"):
        api.FileJournalStore(support_path)


def test_subprocess_coordinator_is_sha_pinned_and_canonical(
    tmp_path: Path,
) -> None:
    """Break caught: the runner shells out or trusts an unreviewed adapter."""

    api = _api()
    executable = (tmp_path / "finite-coordinator").resolve()
    executable.write_text(
        """#!/usr/bin/python3
import json
import sys
request = json.loads(sys.stdin.buffer.read())
assert request["schema_version"] == 2
assert request["record_type"] == "glm52_task13_coordinator_request_v2"
assert request["controller_execution_authority"]["schema_version"] == 2
assert request["operation_capability"]["schema_version"] == 2
response = {
    "schema_version": 2,
    "record_type": "glm52_task13_coordinator_response_v2",
    "action": request["action"],
    "operation_kind": request["operation_kind"],
    "operation_id": request["operation_id"],
    "result": {"observed": request["request"]},
}
sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\\n")
""",
        encoding="ascii",
    )
    executable.chmod(0o700)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    coordinator = api.SubprocessCoordinator(
        executable,
        digest,
        authority_signer=SIGNER,
    )
    coordinator.bind_controller_authority(
        _issued_authority(
            _package(),
            "deploy-disabled",
            digest,
        )
    )
    assert coordinator.inspect("proof", {"value": "literal;$(false)"}) == {
        "observed": {"value": "literal;$(false)"}
    }
    with pytest.raises(api.CampaignRunnerError, match="identity drifted"):
        api.SubprocessCoordinator(executable, "0" * 64)


def test_subprocess_coordinator_rehashes_immediately_before_every_exec(
    tmp_path: Path,
) -> None:
    """Break caught: a verified path is replaced before the controller execs it."""

    api = _api()
    executable = (tmp_path / "replaceable-coordinator").resolve()
    original = (
        "#!/usr/bin/python3\n"
        "import json,sys\n"
        "r=json.loads(sys.stdin.buffer.read())\n"
        "o={'schema_version':2,'record_type':"
        "'glm52_task13_coordinator_response_v2','action':r['action'],"
        "'operation_kind':r['operation_kind'],'operation_id':"
        "r['operation_id'],'result':{'version':'original'}}\n"
        "sys.stdout.write(json.dumps(o,sort_keys=True,separators=(',',':'))+'\\n')\n"
    )
    replacement = original.replace("original", "replacement")
    executable.write_text(original, encoding="ascii")
    executable.chmod(0o700)
    coordinator = api.SubprocessCoordinator(
        executable,
        hashlib.sha256(original.encode("ascii")).hexdigest(),
        authority_signer=SIGNER,
    )
    coordinator.bind_controller_authority(
        _issued_authority(
            _package(),
            "deploy-disabled",
            hashlib.sha256(original.encode("ascii")).hexdigest(),
        )
    )
    executable.write_text(replacement, encoding="ascii")
    executable.chmod(0o700)
    with pytest.raises(api.CampaignRunnerError, match="identity drifted"):
        coordinator.inspect("proof", {"value": "closed"})


def test_subprocess_coordinator_allows_bounded_long_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: the runner kills the guarded H100 route after 120 seconds."""

    api = _api()
    executable = (tmp_path / "qualification-coordinator").resolve()
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
    executable.chmod(0o700)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    observed_timeout: list[int] = []
    real_run = subprocess.run

    def run(argv: list[str], **kwargs: object) -> object:
        if argv[0] == "/usr/bin/ssh-keygen":
            return real_run(argv, **kwargs)
        observed_timeout.append(int(kwargs["timeout"]))
        request = json.loads(kwargs["input"])
        assert request["schema_version"] == 2
        assert (
            request["record_type"]
            == "glm52_task13_coordinator_request_v2"
        )
        assert request["controller_execution_authority"][
            "schema_version"
        ] == 2
        assert request["operation_capability"]["schema_version"] == 2
        response = {
            "schema_version": 2,
            "record_type": "glm52_task13_coordinator_response_v2",
            "action": request["action"],
            "operation_kind": request["operation_kind"],
            "operation_id": request["operation_id"],
            "result": {"status": "QUALIFIED"},
        }
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=canonical_json_bytes(response) + b"\n",
            stderr=b"",
        )

    monkeypatch.setattr(api.subprocess, "run", run)
    coordinator = api.SubprocessCoordinator(
        executable,
        digest,
        authority_signer=SIGNER,
    )
    authority = _issued_authority(
        _package(),
        "h100-qualification",
        digest,
    )
    coordinator.bind_controller_authority(authority)
    operation_id = "h100:test"
    request = {"closed": True}
    journal = api.FileJournalStore(
        (tmp_path / "qualification-journal.jsonl").resolve()
    )
    request_identity = canonical_sha256(request)
    for sequence, state in ((1, "PREPARED"), (2, "POSSIBLY_SENT")):
        body = {
            "schema_version": 1,
            "record_type": "glm52_task13_runner_journal_v1",
            "operation_id": operation_id,
            "operation_kind": "h100-qualification",
            "stage": authority.stage,
            "sequence": sequence,
            "state": state,
            "request_identity_sha256": request_identity,
            "result_identity_sha256": None,
        }
        journal.append(
            {
                **body,
                "canonical_identity_sha256": canonical_sha256(body),
            }
        )
    coordinator.bind_operation_journal(
        action="execute",
        operation_kind="h100-qualification",
        operation_id=operation_id,
        journal=journal,
    )
    result = coordinator.execute(
        "h100-qualification",
        operation_id,
        request,
    )
    assert result == {"status": "QUALIFIED"}
    assert observed_timeout == [24 * 60 * 60 + 300]


def test_repository_owned_coordinator_is_the_only_cli_execution_boundary(
    tmp_path: Path,
) -> None:
    """Break caught: CLI callers supply an arbitrary fake coordinator."""

    api = _api()
    coordinator = api.SubprocessCoordinator(
        REPOSITORY_COORDINATOR.resolve(),
        COORDINATOR_SHA,
    )
    assert coordinator.inspect("health", {}) == {
        "status": "READY",
        "account_id": ACCOUNT,
        "region": REGION,
        "profile": PROFILE,
    }

    package = _package()
    package_path = (tmp_path / "package.json").resolve()
    artifacts_path = (tmp_path / "artifacts.json").resolve()
    authority_path = (tmp_path / "authority.json").resolve()
    fence_journal_path = (tmp_path / "fence-journal.jsonl").resolve()
    support_journal_path = (tmp_path / "support-journal.jsonl").resolve()
    campaign_journal_path = (tmp_path / "campaign-journal.jsonl").resolve()
    package_path.write_bytes(canonical_campaign_package_bytes(package))
    artifacts_path.write_bytes(
        canonical_json_bytes(package["reviewed_artifacts"]) + b"\n"
    )
    foreign = asdict(_authority(api, package, "deploy-disabled"))
    foreign["coordinator_executable_sha256"] = "0" * 64
    unsigned = dict(foreign)
    del unsigned["canonical_identity_sha256"]
    foreign["canonical_identity_sha256"] = canonical_sha256(unsigned)
    authority_path.write_bytes(canonical_json_bytes(foreign) + b"\n")
    script = (
        Path(__file__).parents[1]
        / "aws/glm52-gpu/scripts/run_glm52_task13_campaign.py"
    )
    completed = subprocess.run(
        patched_python_script_command(
            public_key_path=SIGNING_PRIVATE_KEY.with_suffix(".pub"),
            script_path=script,
            arguments=[
                "--package",
                str(package_path),
                "--reviewed-artifacts",
                str(artifacts_path),
                "--stage",
                "deploy-disabled",
                "--authority",
                str(authority_path),
                "--full-run-work",
                str(FULL_RUN_WORK),
                "--signing-private-key",
                str(SIGNING_PRIVATE_KEY),
                "--fence-journal",
                str(fence_journal_path),
                "--support-journal",
                str(support_journal_path),
                "--campaign-journal",
                str(campaign_journal_path),
            ],
        ),
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 64
    assert "controller authority signature or TTL drifted" in completed.stderr
    assert not fence_journal_path.exists()
    assert not support_journal_path.exists()
    assert not campaign_journal_path.exists()


@pytest.mark.parametrize(
    ("action", "kind", "handler_name"),
    [
        ("inspect", "migration-bootstrap-state", "_bootstrap_state"),
        ("inspect", "negative-iam-probe", "_negative_iam_probe"),
        ("inspect", "measurement", "_measurement_readback"),
        ("inspect", "gate", "_gate_readback"),
        ("inspect", "terminal-proof", "_terminal_proof"),
        ("inspect", "monitor", "_monitor"),
        ("execute", "migration-bootstrap", "_execute_bootstrap"),
        ("execute", "collector-invoke", "_lambda_invoke"),
        ("execute", "finalize-invoke", "_lambda_invoke"),
        (
            "execute",
            "qualification-cache-seed",
            "_execute_qualification_cache_seed",
        ),
        (
            "execute",
            "h100-qualification",
            "_execute_h100_qualification",
        ),
        (
            "execute",
            "launch-authority-materialize",
            "_execute_launch_authority_materialize",
        ),
        ("execute", "guarded-launch", "_execute_guarded_launch"),
        ("reconcile", "migration-bootstrap", "_reconcile_bootstrap"),
        ("reconcile", "collector-invoke", "_reconcile_collector_invoke"),
        ("reconcile", "finalize-invoke", "_reconcile_finalize_invoke"),
        (
            "reconcile",
            "qualification-cache-seed",
            "_reconcile_qualification_cache_seed",
        ),
        (
            "reconcile",
            "h100-qualification",
            "_reconcile_h100_qualification",
        ),
        (
            "reconcile",
            "launch-authority-materialize",
            "_reconcile_launch_authority_materialize",
        ),
        ("reconcile", "guarded-launch", "_reconcile_guarded_launch"),
    ],
)
def test_production_coordinator_dispatches_every_nontrivial_runner_route(
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    kind: str,
    handler_name: str,
) -> None:
    """Break caught: an exact runner route falls into the unknown-operation stop."""

    coordinator = _coordinator_api()
    expected = {"route": action + "/" + kind}
    monkeypatch.setattr(
        coordinator,
        handler_name,
        lambda request: expected,
        raising=False,
    )
    monkeypatch.setattr(
        coordinator,
        "_current_credential_evidence",
        lambda: {
            "account_id": ACCOUNT,
            "observed_epoch_seconds": 1,
            "expiration_epoch_seconds": 7201,
            "seconds_remaining": 7200,
        },
    )
    if action == "execute":
        monkeypatch.setattr(
            coordinator,
            "_reserve_execute_custody",
            lambda authority, capability: None,
        )
    elif action == "reconcile":
        monkeypatch.setattr(
            coordinator,
            "_read_operation_custody",
            lambda authority, capability: (
                coordinator._expected_operation_custody(
                    authority,
                    capability,
                )
            ),
        )
    if action in {"execute", "reconcile"}:
        monkeypatch.setattr(
            coordinator,
            "_commit_operation_custody",
            lambda authority, capability, result: None,
        )
    response = coordinator._dispatch(
        _coordinator_envelope(
            coordinator,
            action=action,
            kind=kind,
            request={"closed": True},
        )
    )
    want = expected
    if action == "reconcile":
        want = {"state": "COMMITTED", "result": expected}
    assert response["result"] == want
    monkeypatch.setattr(
        coordinator,
        handler_name,
        lambda request: (_ for _ in ()).throw(
            ValueError("positive truth unavailable")
        ),
    )
    with pytest.raises(ValueError, match="positive truth unavailable"):
        coordinator._dispatch(
            _coordinator_envelope(
                coordinator,
                action=action,
                kind=kind,
                request={"closed": True},
            )
        )


def test_production_coordinator_inventory_matches_exact_runner_call_sites() -> None:
    """Break caught: a new runner operation has no production coordinator route."""

    coordinator = _coordinator_api()
    source = (
        Path(__file__).parents[1]
        / "src/glm52_enforcement/task13_campaign_runner.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    inspect_kinds: set[str] = set()
    mutation_kinds: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "_inspect"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            inspect_kinds.add(node.args[1].value)
        if isinstance(node.func, ast.Name) and node.func.id == "_mutation":
            for keyword in node.keywords:
                if (
                    keyword.arg == "operation_kind"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    mutation_kinds.add(keyword.value.value)
    assert coordinator._INSPECT_KINDS == inspect_kinds | {"health"}
    assert coordinator._EXECUTE_KINDS == mutation_kinds
    assert coordinator._RECONCILE_KINDS == mutation_kinds


def test_staged_adoption_coordinator_exact_reads_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: staged adoption trusts a label instead of live truth."""

    coordinator = _coordinator_api()
    monkeypatch.setenv(
        "GLM52_TASK13_REPO_ROOT",
        str(Path(__file__).parents[1]),
    )
    package = _package()
    evidence = package["disabled_deployment"][
        "staged_infrastructure_evidence"
    ]
    calls: list[list[str]] = []
    unrelated_event_body = {
        "eventSource": "ec2.amazonaws.com",
        "eventName": "RunInstances",
        "awsRegion": REGION,
        "recipientAccountId": ACCOUNT,
        "eventID": "11111111-2222-4333-8444-555555555555",
        "requestParameters": {
            "tagSpecificationSet": {
                "items": [
                    {
                        "resourceType": "instance",
                        "tags": [
                            {
                                "key": "campaign-run-id",
                                "value": RUN_ID,
                            }
                        ],
                    }
                ]
            }
        },
    }
    unrelated_event = {
        "EventId": unrelated_event_body["eventID"],
        "EventName": "RunInstances",
        "EventSource": "ec2.amazonaws.com",
        "CloudTrailEvent": json.dumps(
            unrelated_event_body,
            separators=(",", ":"),
        ),
    }

    def aws(argv: list[str]) -> object:
        calls.append(argv)
        if argv[1:3] == ["sts", "get-caller-identity"]:
            return {"Account": ACCOUNT}
        if argv[1:3] == ["lambda", "get-layer-version"]:
            support = evidence["support_stack"]
            return {
                "LayerVersionArn": support[
                    "cryptography_layer_version_arn"
                ],
                "CodeSha256": support[
                    "cryptography_layer_code_sha256"
                ],
                "CompatibleRuntimes": ["python3.12"],
                "CompatibleArchitectures": ["x86_64"],
            }
        if argv[1:3] == ["lambda", "get-function"]:
            function_identity = argv[
                argv.index("--function-name") + 1
            ]
            support = evidence["support_stack"]
            code_sha = base64.b64encode(
                bytes.fromhex(
                    support["support_lambda_archive"][
                        "file_sha256"
                    ]
                )
            ).decode("ascii")
            function_name = (
                "keep-glm52-h1g-support-function"
            )
            if not function_identity.startswith("arn:aws:lambda:"):
                assert function_identity == function_name
            function_arn = (
                "arn:aws:lambda:us-west-2:"
                "246813579024:function:"
                "keep-glm52-h1g-support-function"
            )
            version = "$LATEST"
            if function_identity.startswith("arn:aws:lambda:"):
                function_arn = function_identity
                version = "1"
            return {
                "Configuration": {
                    "FunctionName": function_name,
                    "FunctionArn": function_arn,
                    "Version": version,
                    "CodeSha256": code_sha,
                    "Runtime": "python3.12",
                    "Architectures": ["x86_64"],
                    "Handler": (
                        "support_custom_resource_handler.main"
                    ),
                    "MemorySize": 256,
                    "Timeout": 840,
                    "Role": (
                        "arn:aws:iam::246813579024:role/"
                        "keep-glm52-h1g-support-function-role"
                    ),
                    "Environment": {
                        "Variables": {
                            "GLM52_ACTIVATION_ID": ACTIVATION,
                            "GLM52_RUN_ID": RUN_ID,
                        }
                    },
                    "Layers": [
                        {
                            "Arn": support[
                                "cryptography_layer_version_arn"
                            ]
                            }
                        ],
                    },
                    "Concurrency": {
                        "ReservedConcurrentExecutions": 1,
                    },
                }
        if argv[1:3] == ["dynamodb", "query"]:
            return {"Count": 0}
        if argv[1:3] == ["cloudtrail", "lookup-events"]:
            if "--starting-token" not in argv:
                return {
                    "Events": [unrelated_event],
                    "NextToken": "cloudtrail-page-2",
                }
            assert argv[argv.index("--starting-token") + 1] == (
                "cloudtrail-page-2"
            )
            return {"Events": []}
        if argv[1:3] == ["cloudformation", "describe-stacks"]:
            stack_id = argv[argv.index("--stack-name") + 1]
            role_arn = (
                coordinator.FENCE_ROLE_ARN
                if "keep-glm52-h1g-fence/" in stack_id
                else coordinator.DEPLOYMENT_ROLE_ARN
            )
            return {
                "Stacks": [
                    {
                        "StackId": stack_id,
                        "StackStatus": "UPDATE_COMPLETE",
                        "RoleARN": role_arn,
                        "EnableTerminationProtection": True,
                        "Tags": [
                            {
                                "Key": "CampaignRunId",
                                "Value": RUN_ID,
                            },
                            {
                                "Key": "DeploymentState",
                                "Value": "DISABLED",
                            },
                            {
                                "Key": "Task13ActivationId",
                                "Value": ACTIVATION,
                            },
                        ],
                    }
                ]
            }
        if argv[1:3] == ["cloudformation", "get-template"]:
            stack_id = argv[argv.index("--stack-name") + 1]
            kind = (
                "FENCE_TEMPLATE"
                if "keep-glm52-h1g-fence/" in stack_id
                else "SUPPORT_TEMPLATE"
                if "keep-glm52-h1g-support/" in stack_id
                else "RETAINED_TEMPLATE"
            )
            return {"TemplateBody": _fixture_template_body(kind)}
        if argv[1:3] == ["cloudformation", "list-stack-resources"]:
            stack_id = argv[argv.index("--stack-name") + 1]
            if "keep-glm52-h1g-support/" in stack_id:
                return {
                    "StackResourceSummaries": [
                        {
                            "LogicalResourceId": (
                                "SupportFunctionRole"
                            ),
                            "PhysicalResourceId": (
                                "keep-glm52-h1g-"
                                "support-function-role"
                            ),
                            "ResourceType": "AWS::IAM::Role",
                            "ResourceStatus": "CREATE_COMPLETE",
                        },
                        {
                            "LogicalResourceId": "SupportFunction",
                            "PhysicalResourceId": (
                                "keep-glm52-h1g-support-function"
                            ),
                            "ResourceType": (
                                "AWS::Lambda::Function"
                            ),
                            "ResourceStatus": "CREATE_COMPLETE",
                        },
                        {
                            "LogicalResourceId": "SupportVersion",
                            "PhysicalResourceId": (
                                "arn:aws:lambda:us-west-2:"
                                "246813579024:function:"
                                "keep-glm52-h1g-support-function:1"
                            ),
                            "ResourceType": "AWS::Lambda::Version",
                            "ResourceStatus": "CREATE_COMPLETE",
                        },
                    ]
                }
            if "keep-glm52-h1g-fence/" in stack_id:
                return {
                    "StackResourceSummaries": [
                        {
                            "LogicalResourceId": (
                                "H1gProductionFenceBucketPolicy"
                            ),
                            "PhysicalResourceId": (
                                "keep-glm52-models-"
                                "246813579024-us-west-2"
                            ),
                            "ResourceType": "AWS::S3::BucketPolicy",
                            "ResourceStatus": "CREATE_COMPLETE",
                        }
                    ]
                }
            return {"StackResourceSummaries": []}
        if argv[1:3] == ["cloudformation", "list-change-sets"]:
            return {"Summaries": []}
        raise AssertionError(argv)

    monkeypatch.setattr(coordinator, "_aws_json", aws)
    monkeypatch.setattr(
        coordinator,
        "_exact_s3_coordinate",
        lambda request: dict(request),
    )
    monkeypatch.setattr(
        coordinator,
        "_exact_runtime_archive",
        lambda request: dict(request),
    )
    monkeypatch.setattr(coordinator, "_active_p5_ids", lambda: [])

    result = coordinator._dispatch(
        _coordinator_envelope(
            coordinator,
            action="inspect",
            kind="staged-infrastructure-adoption",
            request=evidence,
        )
    )

    assert result["result"] == evidence
    assert calls
    operation_counts = {
        operation: sum(
            tuple(argv[1:3]) == operation for argv in calls
        )
        for operation in {tuple(argv[1:3]) for argv in calls}
    }
    assert operation_counts == {
        ("sts", "get-caller-identity"): 1,
        ("lambda", "get-layer-version"): 1,
        ("lambda", "get-function"): 2,
        ("cloudtrail", "lookup-events"): 2,
        ("dynamodb", "query"): 1,
        ("cloudformation", "describe-stacks"): 3,
        ("cloudformation", "get-template"): 3,
        ("cloudformation", "list-stack-resources"): 3,
        ("cloudformation", "list-change-sets"): 3,
    }


def test_staged_adoption_rejects_live_support_lambda_code_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: deployed Lambda bytes drift outside CloudFormation."""

    coordinator = _coordinator_api()
    evidence = _package()["disabled_deployment"][
        "staged_infrastructure_evidence"
    ]
    observed = {
        "SupportFunction": {
            "PhysicalResourceId": (
                "keep-glm52-h1g-support-function"
            )
        },
        "SupportVersion": {
            "PhysicalResourceId": (
                "arn:aws:lambda:us-west-2:246813579024:function:"
                "keep-glm52-h1g-support-function:1"
            )
        },
    }
    monkeypatch.setattr(
        coordinator,
        "_aws_json",
        lambda argv: {
            "Configuration": {
                "FunctionName": (
                    "keep-glm52-h1g-support-function"
                ),
                "Version": "$LATEST",
                "CodeSha256": base64.b64encode(
                    bytes.fromhex("f" * 64)
                ).decode("ascii"),
                "Runtime": "python3.12",
                "Architectures": ["x86_64"],
                "Handler": "support_custom_resource_handler.main",
                "MemorySize": 256,
                "Timeout": 840,
                "Role": (
                    "arn:aws:iam::246813579024:role/"
                    "keep-glm52-h1g-support-function-role"
                ),
                "Environment": {
                    "Variables": {
                        "GLM52_ACTIVATION_ID": ACTIVATION,
                        "GLM52_RUN_ID": RUN_ID,
                    }
                },
                "VpcConfig": {
                    "SecurityGroupIds": [],
                    "SubnetIds": [],
                },
                "DeadLetterConfig": {"TargetArn": ""},
                "TracingConfig": {"Mode": "PassThrough"},
                "KMSKeyArn": "",
                "EphemeralStorage": {"Size": 512},
                "Layers": [
                    {
                        "Arn": evidence["support_stack"][
                            "cryptography_layer_version_arn"
                        ]
                    }
                ],
            },
            "Concurrency": {
                "ReservedConcurrentExecutions": 1,
            },
        },
    )
    with pytest.raises(
        ValueError,
        match="support Lambda readback drifted",
    ):
        coordinator._exact_support_lambda_readback(
            template_resources=_fixture_template_body(
                "SUPPORT_TEMPLATE"
            )["Resources"],
            observed_resources=observed,
            support_evidence=evidence["support_stack"],
        )


def test_staged_adoption_cloudtrail_counts_exact_campaign_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: raw EC2 history is inferred only from the retained ledger."""

    coordinator = _coordinator_api()
    event_body = {
        "eventSource": "ec2.amazonaws.com",
        "eventName": "RunInstances",
        "awsRegion": REGION,
        "recipientAccountId": ACCOUNT,
        "eventID": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        "requestParameters": {
            "tagSpecificationSet": {
                "items": [
                    {
                        "resourceType": "instance",
                        "tags": [
                            {"key": "RunId", "value": RUN_ID},
                            {
                                "key": "activation-id",
                                "value": ACTIVATION,
                            },
                        ],
                    },
                    {
                        "resourceType": "volume",
                        "tags": [
                            {"key": "RunId", "value": RUN_ID},
                            {
                                "key": "activation-id",
                                "value": ACTIVATION,
                            },
                        ],
                    },
                ]
            }
        },
    }
    monkeypatch.setattr(
        coordinator,
        "_aws_json",
        lambda argv: {
            "Events": [
                {
                    "EventId": event_body["eventID"],
                    "EventName": "RunInstances",
                    "EventSource": "ec2.amazonaws.com",
                    "CloudTrailEvent": json.dumps(
                        event_body,
                        separators=(",", ":"),
                    ),
                }
            ]
        },
    )

    assert coordinator._staged_raw_ec2_launch_count(ACTIVATION) == 1


def test_staged_adoption_cloudtrail_fails_closed_on_malformed_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: malformed CloudTrail pagination is treated as zero."""

    coordinator = _coordinator_api()
    monkeypatch.setattr(
        coordinator,
        "_aws_json",
        lambda argv: {"Events": "not-a-list"},
    )

    with pytest.raises(
        ValueError,
        match="CloudTrail RunInstances page drifted",
    ):
        coordinator._staged_raw_ec2_launch_count(ACTIVATION)


@pytest.mark.parametrize(
    ("action", "kind"),
    [
        ("inspect", "p5-zero"),
        ("inspect", "immutable-marker"),
        ("inspect", "immutable-artifact"),
    ],
)
def test_production_coordinator_dispatches_remaining_runner_routes(
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    kind: str,
) -> None:
    """Break caught: a basic AWS/read route bypasses its exact guarded helper."""

    coordinator = _coordinator_api()
    monkeypatch.setattr(
        coordinator,
        "_current_credential_evidence",
        lambda: {
            "account_id": ACCOUNT,
            "observed_epoch_seconds": 1,
            "expiration_epoch_seconds": 7201,
            "seconds_remaining": 7200,
        },
    )
    monkeypatch.setattr(
        coordinator,
        "_inspect_aws_command",
        lambda request: {"account_id": ACCOUNT},
    )
    monkeypatch.setattr(coordinator, "_active_p5_ids", lambda: [])
    monkeypatch.setattr(
        coordinator,
        "_exact_s3_coordinate",
        lambda request: dict(request),
    )
    monkeypatch.setattr(coordinator, "_aws_json", lambda argv: {})
    request: dict[str, object]
    if kind == "aws-command":
        request = {
            "command_id": "create-fence-disabled-change-set",
            "operation": "cloudformation:CreateChangeSet",
            "account_id": ACCOUNT,
            "profile": PROFILE,
            "region": REGION,
            "mutates_aws": True,
            "execution_state": "BLOCKED_REQUIRES_CONTROLLER",
            "argv": [
                "aws",
                "cloudformation",
                "create-change-set",
                "--stack-name",
                "keep-glm52-h1g-fence",
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--no-cli-pager",
            ],
        }
    else:
        request = {"coordinate": "exact"}
    response = coordinator._dispatch(
        _coordinator_envelope(
            coordinator,
            action=action,
            kind=kind,
            request=request,
        )
    )
    if kind == "p5-zero":
        assert response["result"] == {"active_p5_instance_ids": []}
    elif kind.startswith("immutable-"):
        assert response["result"] == request
    else:
        assert response["result"]["operation"] == request["operation"]


def test_exact_s3_coordinate_recomputes_embedded_body_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: an object's self-declared body hash substitutes for bytes."""

    coordinator = _coordinator_api()
    claimed_body = canonical_sha256({"status": "REVIEWED"})
    raw = (
        canonical_json_bytes(
            {
                "status": "TAMPERED",
                "canonical_identity_sha256": claimed_body,
            }
        )
        + b"\n"
    )
    monkeypatch.setattr(
        coordinator,
        "_get_s3_version",
        lambda **kwargs: raw,
    )
    with pytest.raises(ValueError, match="self-hash drifted"):
        coordinator._exact_s3_coordinate(
            {
                "bucket": (
                    "keep-glm52-models-246813579024-us-west-2"
                ),
                "key": "task13/tampered.json",
                "version_id": "3LgTamperedVersion",
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "body_sha256": claimed_body,
            }
        )


def test_production_coordinator_stepfunctions_reconcile_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: lost start response causes a second StartExecution call."""

    coordinator = _coordinator_api()
    from glm52_enforcement.task10_capacity_reconciliation import (
        build_capacity_reconciliation,
        outcome_key,
    )

    calls: list[list[str]] = []
    execution_arn = (
        "arn:aws:states:us-west-2:246813579024:execution:"
        "keep-glm52-h1g-production:task13-test"
    )
    workflow_arn = (
        "arn:aws:states:us-west-2:246813579024:stateMachine:"
        "keep-glm52-h1g-production:17"
    )
    payload = {
        "status": "SUCCEEDED",
        "executionArn": execution_arn,
        "stateMachineArn": workflow_arn,
        "name": "task13-test",
        "input": '{"closed":true}',
    }
    output_key = (
        f"campaigns/{RUN_ID}/submissions/production/"
        "generations/00000001/workflow/LAUNCH_OUTCOME.json"
    )
    writer_record = build_capacity_reconciliation(
        schema_version=1,
        record_type="glm52_task10_capacity_reconciliation_v1",
        account_id=ACCOUNT,
        region=REGION,
        run_id=RUN_ID,
        activation_id=ACTIVATION,
        generation=1,
        generation_text="00000001",
        classification="WORKER_ALLOCATED",
        execution_arn=execution_arn,
        capacity_outcomes=[
            {
                "attempt": 1,
                "availability_zone": "us-west-2a",
                "outcome": "WORKER_ALLOCATED",
            }
        ],
        instance_id="i-00000000000000001",
        availability_zone="us-west-2a",
        same_token_identity_sha256="1" * 64,
        reserve_identity_sha256="2" * 64,
        spend_authority_identity_sha256="3" * 64,
        action_identity_sha256="4" * 64,
        task9_custody_identity_sha256="5" * 64,
    )
    durable = asdict(writer_record)
    assert outcome_key(writer_record) == output_key
    coordinate = _marker(output_key, "6")

    def aws_json(argv: list[str]) -> dict[str, str]:
        calls.append(argv)
        return dict(payload)

    authority = type(
        "Authority",
        (),
        {
            "expected_execution_arn": execution_arn,
            "workflow_version_arn": workflow_arn,
            "execution_name": "task13-test",
            "activation_id": ACTIVATION,
            "generation": 1,
            "generation_text": "00000001",
            "same_token_identity_sha256": "1" * 64,
            "task8_spend_reserve_identity_sha256": "2" * 64,
            "task8_spend_authority_identity_sha256": "3" * 64,
            "task9_launch_identity_sha256": "4" * 64,
            "task9_custody_identity_sha256": "5" * 64,
        },
    )()
    monkeypatch.setattr(coordinator, "_aws_json", aws_json)
    monkeypatch.setattr(
        coordinator,
        "_read_result_record",
        lambda *, key: (
            copy.deepcopy(durable),
            copy.deepcopy(coordinate),
        )
        if key == output_key
        else (_ for _ in ()).throw(AssertionError(key)),
    )
    observed = coordinator._StepFunctionsBoundary().reconcile(
        authority,
        {"closed": True},
    )
    assert observed["classification"] == "WORKER_ALLOCATED"
    assert observed["workflow_output"] == {
        "bucket": (
            "keep-glm52-models-246813579024-us-west-2"
        ),
        "key": coordinate["key"],
        "version_id": coordinate["version_id"],
        "file_sha256": coordinate["file_sha256"],
        "body_sha256": coordinate["body_sha256"],
    }
    assert observed["observation_identity_sha256"] == canonical_sha256(
        {
            key: observed[key]
            for key in observed
            if key != "observation_identity_sha256"
        }
    )
    assert len(calls) == 1
    assert calls[0][1:3] == ["stepfunctions", "describe-execution"]
    payload["input"] = '{"closed":false}'
    with pytest.raises(ValueError, match="identity"):
        coordinator._StepFunctionsBoundary().reconcile(
            authority,
            {"closed": True},
        )
    payload["input"] = '{"closed":true}'
    payload["status"] = "RUNNING"
    with pytest.raises(ValueError, match="durable reconciliation"):
        coordinator._StepFunctionsBoundary().reconcile(
            authority,
            {"closed": True},
        )


def test_failed_execution_reconcile_proves_start_acceptance_not_terminal_success(
) -> None:
    """Break caught: a failed exact execution is replayed or called terminal success."""

    coordinator = _coordinator_api()
    execution_arn = (
        "arn:aws:states:us-west-2:246813579024:execution:"
        "keep-glm52-h1g-production:task13-test"
    )
    outcome = type(
        "Outcome",
        (),
        {
            "status": "reconciled",
            "detail": {
                "classification": "FAILED",
                "execution_arn": execution_arn,
            },
        },
    )()
    with pytest.raises(ValueError, match="failed"):
        coordinator._runner_launch_result(
            {
                "attempt": 1,
                "availability_zone": "us-west-2a",
                "activation_id": ACTIVATION,
                "execution_authority": {
                    "spend_authority": {
                        "remaining_gpu_seconds": 7200,
                        "ledger_version_id": "3LgSpend",
                        "ledger_body_sha256": "a" * 64,
                    },
                    "action_authority": {
                        "action_kind": "PRODUCTION_SUBMISSION",
                        "action_count": 1,
                        "action_version_id": "3LgAction",
                        "action_body_sha256": "b" * 64,
                    },
                },
                "production_authority_contract": {
                    "required_terminal_markers": [
                        "CAMPAIGN_DRAINED.json",
                        "TERMINAL_VERIFIED.json",
                    ],
                    "monitor_route": (
                        "aws/glm52-gpu/scripts/"
                        "sky_campaign_break_glass.sh status"
                    ),
                },
            },
            outcome,
        )


def test_guarded_launch_starts_once_then_uses_only_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: terminal polling re-enters StartExecution."""

    coordinator = _coordinator_api()
    import glm52_enforcement.task10_production as task10

    calls: list[str] = []

    def run_production(request, *, boundary):
        assert type(boundary).__name__ == "_StepFunctionsBoundary"
        calls.append(request.action)
        return type(
            "Outcome",
            (),
            {
                "status": (
                    "started"
                    if request.action == "start"
                    else "reconciled"
                ),
                "detail": {"classification": "WORKER_ALLOCATED"},
            },
        )()

    monkeypatch.setattr(
        coordinator,
        "_task10_material",
        lambda request: (
            object(),
            type(
                "Authority",
                (),
                {"generation_text": "00000001"},
            )(),
            object(),
            object(),
        ),
    )
    monkeypatch.setattr(task10, "run_production", run_production)
    monkeypatch.setattr(
        coordinator,
        "_runner_launch_result",
        lambda request, outcome: {
            "classification": outcome.detail["classification"]
        },
    )
    sole_sender = {
        "table_name": "keep-glm52-h1g-ledger-v1",
        "partition_key": RUN_ID,
        "sort_key": (
            f"ACTIVATION#{ACTIVATION}#"
            "TASK10_SOLE_SENDER_AUTHORITY#00000001"
        ),
        "authority_identity_sha256": "a" * 64,
        "source_closure_identity_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        coordinator,
        "_read_sole_sender_authority",
        lambda request: copy.deepcopy(sole_sender),
    )

    contract = {
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
    request = {
        "workflow_reconciliation_contract": contract,
        "execution_authority": {
            "sole_sender_authority": copy.deepcopy(sole_sender),
        },
    }
    assert coordinator._guarded_launch(
        request,
        reconcile=False,
    ) == {"classification": "WORKER_ALLOCATED"}
    assert calls == ["start", "reconcile"]

    calls.clear()
    assert coordinator._guarded_launch(
        request,
        reconcile=True,
    ) == {"classification": "WORKER_ALLOCATED"}
    assert calls == ["reconcile"]
    request["workflow_reconciliation_contract"]["record_type"] = "foreign"
    with pytest.raises(ValueError, match="writer contract"):
        coordinator._guarded_launch(request, reconcile=False)


def test_launch_authority_is_bound_to_versioned_durable_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: presend authority is synthesized without an exact S3 version."""

    coordinator = _coordinator_api()
    sole_sender = {
        "table_name": "keep-glm52-h1g-ledger-v1",
        "partition_key": RUN_ID,
        "sort_key": (
            f"ACTIVATION#{ACTIVATION}#"
            "TASK10_SOLE_SENDER_AUTHORITY#00000001"
        ),
        "authority_identity_sha256": "a" * 64,
        "source_closure_identity_sha256": "b" * 64,
    }
    source = {
        "schema_version": 1,
        "record_type": "glm52_task13_launch_authority_v1",
        "account_id": ACCOUNT,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "attempt": 2,
        "availability_zone": "us-west-2b",
        "instance_type": "p5.48xlarge",
        "capacity_type": "ON_DEMAND",
        "same_token_identity_sha256": "1" * 64,
        "reserve_identity_sha256": "6" * 64,
        "spend_authority_identity_sha256": "7" * 64,
        "action_identity_sha256": "8" * 64,
        "task9_custody_identity_sha256": "9" * 64,
        "shared_attempt_counter": 2,
        "immutable_inputs_identity_sha256": "2" * 64,
        "sole_sender_authority": copy.deepcopy(sole_sender),
        "spend_authority": {
            "remaining_gpu_seconds": 7200,
            "remaining_gpu_usd": "110.08",
            "gpu_reserve_seconds": 900,
            "gpu_reserve_usd": "13.76",
            "root_volume_tail_usd_max": "0.01",
            "ledger_version_id": "3LgSpend",
            "ledger_body_sha256": "3" * 64,
        },
        "action_authority": {
            "action_kind": "PRODUCTION_SUBMISSION",
            "action_count": 1,
            "action_version_id": "3LgAction",
            "action_body_sha256": "4" * 64,
        },
        "liability_action": {
            "action_kind": "SAME_TOKEN_COMPLETE",
            "attempt": 2,
            "shared_attempt_counter": 2,
            "action_version_id": "3LgLiability",
            "action_body_sha256": "5" * 64,
        },
        "source_coordinates": {
            "task10_production_authority": {"closed": True},
            "task11_boundary": {"closed": True},
            "task11_inputs": [],
            "approvals": {},
        },
    }
    source["canonical_identity_sha256"] = canonical_sha256(source)
    expected_source = copy.deepcopy(source)
    coordinate = _marker(
        (
            f"campaigns/{RUN_ID}/authorities/task13/{ACTIVATION}/"
            "launch-attempt-02.json"
        ),
        "6",
    )
    monkeypatch.setattr(
        coordinator,
        "_read_result_record",
        lambda *, key: (copy.deepcopy(source), copy.deepcopy(coordinate)),
    )
    task10_authority = type(
        "Task10Authority",
        (),
        {
            "same_token_identity_sha256": "1" * 64,
            "task8_spend_reserve_identity_sha256": "6" * 64,
            "task8_spend_authority_identity_sha256": "7" * 64,
            "task9_launch_identity_sha256": "8" * 64,
            "task9_custody_identity_sha256": "9" * 64,
        },
    )()
    monkeypatch.setattr(
        coordinator,
        "_task10_material",
        lambda request: (object(), task10_authority, object(), object()),
    )
    monkeypatch.setattr(
        coordinator,
        "_build_launch_authority_record",
        lambda request: copy.deepcopy(expected_source),
    )
    monkeypatch.setattr(
        coordinator,
        "_read_sole_sender_authority",
        lambda request: copy.deepcopy(sole_sender),
    )
    request = {
        "account_id": ACCOUNT,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "attempt": 2,
        "availability_zone": "us-west-2b",
        "immutable_inputs_identity_sha256": "2" * 64,
        "sole_sender_authority": copy.deepcopy(sole_sender),
    }
    result = coordinator._read_launch_authority(request)
    assert result["authority_record"] == coordinate
    body = dict(result)
    identity = body.pop("canonical_identity_sha256")
    assert identity == canonical_sha256(body)
    monkeypatch.setattr(
        coordinator,
        "_aws_json",
        lambda argv: (_ for _ in ()).throw(
            AssertionError("existing exact record must be adopted")
        ),
    )
    assert coordinator._execute_launch_authority_materialize(
        request
    ) == result
    source["attempt"] = 3
    with pytest.raises(ValueError, match="positive truth"):
        coordinator._read_launch_authority(request)


def test_launch_authority_materializer_cross_binds_all_task11_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: presend authority trusts opaque Task10 identity strings."""

    coordinator = _coordinator_api()
    from glm52_enforcement.task11_boundary import (
        BOUNDARY_INPUT_KINDS,
        build_task11_boundary_document,
        build_task11_input_coordinate,
    )

    workflow_arn = (
        "arn:aws:states:us-west-2:246813579024:stateMachine:"
        "keep-glm52-h1g-production:17"
    )
    action_key = (
        "ACTIVATION#approved-20260728#ACTION#SKY_POST#00000001"
    )
    coordinates = tuple(
        build_task11_input_coordinate(
            input_kind=kind,
            bucket="task11-source-bucket",
            key=(
                f"campaigns/{RUN_ID}/authorities/task9/{ACTIVATION}/"
                "TASK9_DEPLOYED_IDENTITY.json"
                if kind == "TASK9_DEPLOYED_IDENTITY_COORDINATE"
                else (
                    f"campaigns/{RUN_ID}/authorities/task11/"
                    f"{ACTIVATION}/00000001/{index:02d}-"
                    + kind.lower().replace("_", "-")
                    + ".json"
                )
            ),
            version_id=f"source-version-{index:02d}",
            file_sha256=hashlib.sha256(
                f"file-{index}".encode()
            ).hexdigest(),
            body_sha256=hashlib.sha256(
                f"body-{index}".encode()
            ).hexdigest(),
        )
        for index, kind in enumerate(BOUNDARY_INPUT_KINDS, 1)
    )
    boundary = build_task11_boundary_document(
        activation_id=ACTIVATION,
        generation=1,
        campaign_identity_sha256="a" * 64,
        state_machine_version_arn=workflow_arn,
        action_key=action_key,
        inputs=coordinates,
    )
    boundary_raw = canonical_json_bytes(asdict(boundary)) + b"\n"
    authority = type(
        "Task10Authority",
        (),
        {
            "activation_id": ACTIVATION,
            "generation": 1,
            "generation_text": "00000001",
            "campaign_identity_sha256": "a" * 64,
            "workflow_version_arn": workflow_arn,
            "action_key": action_key,
            "task11_boundary_bucket": "task11-source-bucket",
            "task11_boundary_key": (
                f"campaigns/{RUN_ID}/authorities/task11/"
                f"{ACTIVATION}/00000001.json"
            ),
            "task11_boundary_version_id": "boundary-version",
            "task11_boundary_file_sha256": "b" * 64,
            "task11_boundary_body_sha256": "c" * 64,
            "same_token_identity_sha256": "1" * 64,
            "task8_spend_reserve_identity_sha256": "6" * 64,
            "task8_spend_authority_identity_sha256": "7" * 64,
            "task9_launch_identity_sha256": "8" * 64,
            "task9_custody_identity_sha256": "9" * 64,
        },
    )()
    monkeypatch.setattr(
        coordinator,
        "_task10_material",
        lambda request: (object(), authority, object(), object()),
    )
    monkeypatch.setattr(
        coordinator,
        "_repo_root",
        lambda: Path(__file__).parents[1],
    )
    monkeypatch.setattr(
        coordinator,
        "_exact_s3_coordinate",
        lambda coordinate: dict(coordinate),
    )
    approval_keys = {
        "GPU_SPEND_APPROVAL": "gpu-approval.json",
        "SUPPORT_APPROVAL": "support-approval.json",
        "RESIDUAL_LIABILITY_APPROVAL": "residual-approval.json",
    }
    source_raw = canonical_json_bytes({"source": True}) + b"\n"
    approval_raw = {
        "gpu-approval.json": canonical_json_bytes(
            {
                "remaining_gpu_seconds": 7200,
                "remaining_gpu_usd": "110.08",
            }
        )
        + b"\n",
        "support-approval.json": canonical_json_bytes(
            {"approved": True}
        )
        + b"\n",
        "residual-approval.json": canonical_json_bytes(
            {"approved": True}
        )
        + b"\n",
    }

    def get_s3_version(*, bucket, key, version_id):
        del bucket, version_id
        if key == authority.task11_boundary_key:
            return boundary_raw
        return approval_raw.get(key, source_raw)

    monkeypatch.setattr(coordinator, "_get_s3_version", get_s3_version)
    immutable_inputs = [
        {
            "artifact_kind": kind,
            "bucket": "task11-source-bucket",
            "key": key,
            "version_id": "approved-version",
            "file_sha256": "d" * 64,
            "body_sha256": "e" * 64,
        }
        for kind, key in approval_keys.items()
    ]
    immutable_inputs.append(
        {
            "artifact_kind": "TASK10_PRODUCTION_AUTHORITY",
            "bucket": "task11-source-bucket",
            "key": "task10-authority.json",
            "version_id": "task10-authority-version",
            "file_sha256": "f" * 64,
            "body_sha256": "0" * 64,
        }
    )
    request = {
        "account_id": ACCOUNT,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
        "attempt": 1,
        "availability_zone": "us-west-2a",
        "immutable_inputs_identity_sha256": "2" * 64,
        "immutable_inputs": immutable_inputs,
    }
    sole_sender = {
        "table_name": "keep-glm52-h1g-ledger-v1",
        "partition_key": RUN_ID,
        "sort_key": (
            f"ACTIVATION#{ACTIVATION}#"
            "TASK10_SOLE_SENDER_AUTHORITY#00000001"
        ),
        "authority_identity_sha256": "a" * 64,
        "source_closure_identity_sha256": "b" * 64,
    }
    request["sole_sender_authority"] = sole_sender
    monkeypatch.setattr(
        coordinator,
        "_read_sole_sender_authority",
        lambda request: copy.deepcopy(sole_sender),
    )
    record = coordinator._build_launch_authority_record(request)
    assert len(record["source_coordinates"]["task11_inputs"]) == 14
    assert record["same_token_identity_sha256"] == "1" * 64
    assert record["reserve_identity_sha256"] == "6" * 64
    assert record["action_identity_sha256"] == "8" * 64
    assert record["spend_authority"]["remaining_gpu_seconds"] == 7200
    assert record["canonical_identity_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in record.items()
            if key != "canonical_identity_sha256"
        }
    )
    authority.action_key = "ACTIVATION#foreign#ACTION#SKY_POST#00000001"
    with pytest.raises(ValueError, match="Task11 boundary"):
        coordinator._build_launch_authority_record(request)


def test_sole_sender_coordinator_put_is_conditional_and_get_is_strong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: Task13 overwrites retained authority or adopts stale reads."""

    coordinator = _coordinator_api()
    import glm52_enforcement.task10_authority_materialization as materializer
    from glm52_enforcement.dynamodb import encode_item

    authority = type(
        "Authority",
        (),
        {
            "activation_id": ACTIVATION,
            "generation_text": "00000001",
            "canonical_identity_sha256": "a" * 64,
        },
    )()
    item = {
        "PK": RUN_ID,
        "SK": (
            f"ACTIVATION#{ACTIVATION}#"
            "TASK10_SOLE_SENDER_AUTHORITY#00000001"
        ),
        "canonical_identity_sha256": "a" * 64,
    }
    coordinate = {
        "table_name": "keep-glm52-h1g-ledger-v1",
        "partition_key": RUN_ID,
        "sort_key": item["SK"],
        "authority_identity_sha256": "a" * 64,
        "source_closure_identity_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        coordinator,
        "_build_sole_sender_authority",
        lambda request: (authority, "b" * 64),
    )
    monkeypatch.setattr(
        coordinator,
        "_repo_root",
        lambda: Path(__file__).parents[1],
    )
    monkeypatch.setattr(
        coordinator,
        "_read_sole_sender_authority",
        lambda request: copy.deepcopy(coordinate),
    )
    monkeypatch.setattr(
        materializer,
        "task10_sole_sender_authority_item",
        lambda value: copy.deepcopy(item),
    )
    calls = []

    def aws_json(argv):
        calls.append(list(argv))
        item_path = Path(argv[argv.index("--item") + 1].removeprefix("file://"))
        assert json.loads(item_path.read_text()) == encode_item(item)
        return {}

    monkeypatch.setattr(coordinator, "_aws_json", aws_json)
    assert coordinator._execute_sole_sender_authority_materialize(
        {"activation_id": ACTIVATION}
    ) == coordinate
    assert len(calls) == 1
    command = calls[0]
    assert command[:3] == ["aws", "dynamodb", "put-item"]
    assert command[command.index("--table-name") + 1] == (
        "keep-glm52-h1g-ledger-v1"
    )
    assert command[command.index("--condition-expression") + 1] == (
        "attribute_not_exists(PK) AND attribute_not_exists(SK)"
    )

    observed_item = {
        "PK": RUN_ID,
        "SK": item["SK"],
        "record_type": "retained",
    }
    calls.clear()
    monkeypatch.setattr(
        coordinator,
        "_aws_json",
        lambda argv: (
            calls.append(list(argv))
            or {"Item": encode_item(observed_item)}
        ),
    )
    assert coordinator._read_ddb_item(item["SK"]) == {
        "record_type": "retained"
    }
    assert "--consistent-read" in calls[0]
    assert calls[0][calls[0].index("--return-consumed-capacity") + 1] == (
        "NONE"
    )


def test_sole_sender_builder_cross_binds_six_task8_task9_rows_and_14_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: retained authority is built from opaque hashes alone."""

    coordinator = _coordinator_api()
    import glm52_enforcement.records as records
    from glm52_enforcement.canonical import canonical_sha256
    from glm52_enforcement.launch_custody import (
        LaunchParameterAuthority,
        build_deterministic_client_token,
        build_launch_parameters,
    )
    from glm52_enforcement.task11_boundary import BOUNDARY_INPUT_KINDS

    action_key = (
        f"ACTIVATION#{ACTIVATION}#ACTION#SKY_POST#00000001"
    )
    task10 = type(
        "Task10",
        (),
        {
            "activation_id": ACTIVATION,
            "activation_ordinal": 1,
            "generation": 1,
            "generation_text": "00000001",
            "campaign_identity_sha256": "1" * 64,
            "action_key": action_key,
            "sky_job_name": RUN_ID,
            "task_yaml_sha256": "2" * 64,
            "request_body_sha256": "3" * 64,
            "same_token_identity_sha256": "4" * 64,
            "task8_spend_reserve_identity_sha256": "5" * 64,
            "task8_spend_authority_identity_sha256": "6" * 64,
            "task9_launch_identity_sha256": "7" * 64,
            "task9_custody_identity_sha256": "8" * 64,
            "canonical_identity_sha256": "9" * 64,
        },
    )()
    boundary = type(
        "Boundary",
        (),
        {"canonical_identity_sha256": "a" * 64},
    )()
    source = {
        "approved_ami_id": "ami-00000000000000001",
        "subnet_ids_by_availability_zone": {
            f"us-west-2{letter}": f"subnet-{index:017x}"
            for index, letter in enumerate("abcdef", 1)
        },
        "security_group_id": "sg-00000000000000001",
    }
    source_documents = {
        kind: copy.deepcopy(source) for kind in BOUNDARY_INPUT_KINDS
    }
    source_coordinates = [
        {
            "input_kind": kind,
            "canonical_identity_sha256": f"{index:x}" * 64,
        }
        for index, kind in enumerate(BOUNDARY_INPUT_KINDS, 1)
    ]
    monkeypatch.setattr(
        coordinator,
        "_authenticated_task11_sources",
        lambda request: (
            task10,
            boundary,
            copy.deepcopy(source_documents),
            copy.deepcopy(source_coordinates),
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_repo_root",
        lambda: Path(__file__).parents[1],
    )
    monkeypatch.setattr(
        records,
        "validate_record",
        lambda record_type, value: value,
    )
    rows = {}
    for ordinal in range(1, 7):
        launch_authority = LaunchParameterAuthority(
            account_id=ACCOUNT,
            region=REGION,
            run_id=RUN_ID,
            campaign_identity_sha256="1" * 64,
            activation_id=ACTIVATION,
            activation_ordinal=1,
            generation=1,
            action_key=action_key,
            sky_request_id=f"request-{ordinal}",
            sky_job_name=RUN_ID,
            sky_task_name="glm52-production",
            task_yaml_sha256="2" * 64,
            request_body_sha256="3" * 64,
            approved_ami_id=source["approved_ami_id"],
            subnet_id=source["subnet_ids_by_availability_zone"][
                f"us-west-2{'abcdef'[ordinal - 1]}"
            ],
            security_group_id=source["security_group_id"],
            instance_profile_name="keep-glm52-gpu-worker",
            source_identity_sha256="7" * 64,
        )
        parameters = build_launch_parameters(
            launch_authority,
            allocation_ordinal=ordinal,
        )
        token = build_deterministic_client_token(
            launch_authority,
            ordinal,
            parameters,
        )
        worker = {
            "activation_id": ACTIVATION,
            "activation_ordinal": 1,
            "generation": 1,
            "allocation_ordinal": ordinal,
            "campaign_identity_sha256": "1" * 64,
            "sky_action_key": action_key,
            "sky_request_id": f"request-{ordinal}",
            "sky_job_name": RUN_ID,
            "task_yaml_sha256": "2" * 64,
            "request_body_sha256": "3" * 64,
            "launch_parameters_sha256": (
                parameters.canonical_identity_sha256
            ),
            "ec2_client_token": token,
        }
        worker["canonical_body_sha256"] = canonical_sha256(worker)
        liability = {
            "activation_id": ACTIVATION,
            "allocation_ordinal": ordinal,
            "ec2_client_token": token,
                "state": "WATCHING",
                "current_owner": True,
                "gpu_liability_reserve_ledger_identity_sha256": "5" * 64,
            "gpu_liability_reserve_release_identity_sha256": None,
        }
        liability["canonical_body_sha256"] = canonical_sha256(liability)
        rows[
            f"ACTIVATION#{ACTIVATION}#WORKER_LAUNCH#{ordinal:08d}"
        ] = worker
        rows[
            f"ACTIVATION#{ACTIVATION}#"
            f"WORKER_LAUNCH_LIABILITY#{ordinal:08d}"
        ] = liability
    monkeypatch.setattr(
        coordinator,
        "_read_ddb_item",
        lambda sort_key: copy.deepcopy(rows[sort_key]),
    )
    request = {
        "account_id": ACCOUNT,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": ACTIVATION,
    }
    authority, closure_identity = coordinator._build_sole_sender_authority(
        request
    )
    assert len(authority.attempts) == 6
    assert [attempt.availability_zone for attempt in authority.attempts] == [
        f"us-west-2{letter}" for letter in "abcdef"
    ]
    assert len(closure_identity) == 64
    rows[
        f"ACTIVATION#{ACTIVATION}#WORKER_LAUNCH#00000006"
    ]["ec2_client_token"] = "f" * 64
    with pytest.raises(ValueError, match="Task8/9"):
        coordinator._build_sole_sender_authority(request)


def test_bootstrap_reconcile_reads_one_stack_without_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: first-stack reconciliation creates either stack again."""

    coordinator = _coordinator_api()
    fence_id = (
        "arn:aws:cloudformation:us-west-2:246813579024:"
        "stack/keep-glm52-h1g-fence/"
        "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
    )
    monkeypatch.setattr(
        coordinator,
        "_bootstrap_state",
        lambda request: {
            "status": "BOOTSTRAP_REQUIRED",
            "retained_stack_id": request["retained_stack_id"],
            "absent_stack_names": ["keep-glm52-h1g-support"],
            "existing_stack_ids": {
                "keep-glm52-h1g-fence": fence_id,
            },
        },
    )
    request = {
        "retained_stack_id": (
            "arn:aws:cloudformation:us-west-2:246813579024:stack/"
            "keep-glm52-gpu/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        ),
        "target_stack_name": "keep-glm52-h1g-fence",
        "template": {"body_sha256": "a" * 64},
    }
    assert coordinator._reconcile_bootstrap(request) == {
        "status": "BOOTSTRAPPED_STACK",
        "retained_stack_id": request["retained_stack_id"],
        "stack_name": "keep-glm52-h1g-fence",
        "stack_id": fence_id,
        "termination_protection": True,
        "anchor_logical_id": "ContainerAnchor",
        "template_body_sha256": "a" * 64,
    }


@pytest.mark.parametrize(
    ("target", "expected_role"),
    [
        (
            "keep-glm52-h1g-fence",
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-fence-service",
        ),
        (
            "keep-glm52-h1g-support",
            "arn:aws:iam::246813579024:role/"
            "keep-glm52-h1g-cloudformation-deployment",
        ),
    ],
)
def test_bootstrap_create_uses_stack_specific_service_role(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    expected_role: str,
) -> None:
    """Break caught: the fence anchor was created with deployment authority."""

    coordinator = _coordinator_api()
    retained_id = (
        "arn:aws:cloudformation:us-west-2:246813579024:stack/"
        "keep-glm52-gpu/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    )
    fence_id = (
        "arn:aws:cloudformation:us-west-2:246813579024:stack/"
        "keep-glm52-h1g-fence/bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
    )
    support_id = (
        "arn:aws:cloudformation:us-west-2:246813579024:stack/"
        "keep-glm52-h1g-support/cccccccc-dddd-4eee-8fff-aaaaaaaaaaaa"
    )
    states = iter(
        [
            {
                "status": "BOOTSTRAP_REQUIRED",
                "retained_stack_id": retained_id,
                "absent_stack_names": [target],
                "existing_stack_ids": {},
            },
            {
                "status": "BOOTSTRAPPED",
                "retained_stack_id": retained_id,
                "fence_stack_id": fence_id,
                "support_stack_id": support_id,
                "termination_protection": True,
                "anchor_logical_id": "ContainerAnchor",
                "template_body_sha256": "a" * 64,
            },
        ]
    )
    monkeypatch.setattr(
        coordinator,
        "_bootstrap_state",
        lambda _request: next(states),
    )
    monkeypatch.setattr(
        coordinator,
        "_current_credential_evidence",
        lambda: {"seconds_remaining": 7200},
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        coordinator,
        "_aws_json",
        lambda argv: calls.append(argv) or {},
    )
    request = {
        "activation_id": ACTIVATION,
        "retained_stack_id": retained_id,
        "new_stack_names": [
            "keep-glm52-h1g-fence",
            "keep-glm52-h1g-support",
        ],
        "stack_role_arns": {
            "keep-glm52-h1g-fence": (
                "arn:aws:iam::246813579024:role/"
                "keep-glm52-h1g-fence-service"
            ),
            "keep-glm52-h1g-support": (
                "arn:aws:iam::246813579024:role/"
                "keep-glm52-h1g-cloudformation-deployment"
            ),
        },
        "target_stack_name": target,
        "template": {
            "bucket": "keep-glm52-models-246813579024-us-west-2",
            "key": "task13/templates/container-bootstrap-v1.json",
            "version_id": "bootstrap-v1",
            "body_sha256": "a" * 64,
        },
    }

    coordinator._execute_bootstrap(request)

    create = next(call for call in calls if "create-stack" in call)
    assert create[create.index("--role-arn") + 1] == expected_role


@pytest.mark.parametrize(
    "command_id",
    [
        "read-deployment-cloudformation-role",
        "read-fence-cloudformation-role",
    ],
)
def test_coordinator_inspects_each_role_without_stack_arguments(
    monkeypatch: pytest.MonkeyPatch,
    command_id: str,
) -> None:
    """Break caught: new IAM guards fell through to stack-name parsing."""

    coordinator = _coordinator_api()
    role_name = (
        "keep-glm52-h1g-fence-service"
        if "fence" in command_id
        else "keep-glm52-h1g-cloudformation-deployment"
    )
    role_arn = f"arn:aws:iam::246813579024:role/{role_name}"
    monkeypatch.setattr(
        coordinator,
        "_aws_json",
        lambda _argv: {
            "Role": {"Arn": role_arn, "RoleId": "AROAEXACTROLEID"}
        },
    )

    assert coordinator._inspect_aws_command(
        {
            "command_id": command_id,
            "argv": [
                "aws",
                "iam",
                "get-role",
                "--role-name",
                role_name,
            ],
        }
    ) == {"role_arn": role_arn, "role_id": "AROAEXACTROLEID"}


def test_bootstrap_state_rejects_fence_stack_with_deployment_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: a preexisting fence anchor retained the wrong role."""

    coordinator = _coordinator_api()
    retained_id = (
        "arn:aws:cloudformation:us-west-2:246813579024:stack/"
        "keep-glm52-gpu/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    )
    fence_id = (
        "arn:aws:cloudformation:us-west-2:246813579024:stack/"
        "keep-glm52-h1g-fence/bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
    )
    stacks = {
        "keep-glm52-gpu": {
            "StackName": "keep-glm52-gpu",
            "StackId": retained_id,
        },
        "keep-glm52-h1g-fence": {
            "StackName": "keep-glm52-h1g-fence",
            "StackId": fence_id,
            "RoleARN": (
                "arn:aws:iam::246813579024:role/"
                "keep-glm52-h1g-cloudformation-deployment"
            ),
            "EnableTerminationProtection": True,
            "StackStatus": "CREATE_COMPLETE",
            "Tags": [
                {"Key": "CampaignRunId", "Value": RUN_ID},
                {"Key": "DeploymentState", "Value": "DISABLED"},
                {"Key": "Task13ActivationId", "Value": ACTIVATION},
            ],
        },
        "keep-glm52-h1g-support": None,
    }
    monkeypatch.setattr(
        coordinator,
        "_described_stack",
        lambda name: stacks[name],
    )

    with pytest.raises(ValueError, match="bootstrap stack state drifted"):
        coordinator._bootstrap_state(
            {
                "retained_stack_name": "keep-glm52-gpu",
                "retained_stack_id": retained_id,
                "activation_id": ACTIVATION,
                "new_stack_names": [
                    "keep-glm52-h1g-fence",
                    "keep-glm52-h1g-support",
                ],
                "stack_role_arns": {
                    "keep-glm52-h1g-fence": (
                        "arn:aws:iam::246813579024:role/"
                        "keep-glm52-h1g-fence-service"
                    ),
                    "keep-glm52-h1g-support": (
                        "arn:aws:iam::246813579024:role/"
                        "keep-glm52-h1g-cloudformation-deployment"
                    ),
                },
                "anchor_logical_id": "ContainerAnchor",
                "termination_protection": True,
                "template": {"body_sha256": "a" * 64},
            }
        )


@pytest.mark.parametrize(
    ("operation_kind", "relative_executable", "suffix"),
    [
        (
            "qualification-cache-seed",
            "aws/glm52-gpu/scripts/submit_sky_campaign.py",
            ["cache-seed", "acquire-and-launch"],
        ),
        (
            "h100-qualification",
            "aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh",
            [],
        ),
    ],
)
def test_qualification_driver_manifest_is_versioned_and_shell_free(
    monkeypatch: pytest.MonkeyPatch,
    operation_kind: str,
    relative_executable: str,
    suffix: list[str],
) -> None:
    """Break caught: qualification executes an ambient path or unpinned argv."""

    coordinator = _coordinator_api()
    manifest = {
        "schema_version": 2,
        "record_type": "glm52_task13_repository_driver_v2",
        "activation_id": ACTIVATION,
        "operation_kind": operation_kind,
        "argv": [relative_executable, *suffix],
        "environment": {
            "AWS_PROFILE": PROFILE,
            "AWS_REGION": REGION,
        },
    }
    manifest["canonical_identity_sha256"] = canonical_sha256(manifest)
    monkeypatch.setenv(
        "GLM52_TASK13_REPO_ROOT",
        str(Path(__file__).parents[1]),
    )
    monkeypatch.setattr(
        coordinator,
        "_exact_s3_coordinate",
        lambda coordinate: dict(coordinate),
    )
    monkeypatch.setattr(
        coordinator,
        "_get_s3_version",
        lambda **kwargs: canonical_json_bytes(manifest) + b"\n",
    )
    argv, environment = coordinator._driver_manifest(
        {
            "artifact_kind": (
                "QUALIFICATION_CACHE_SEED_INPUT"
                if operation_kind == "qualification-cache-seed"
                else "H100_QUALIFICATION_INPUT"
            ),
            "bucket": (
                "keep-glm52-models-246813579024-us-west-2"
            ),
            "key": "exact",
            "version_id": "3LgExact",
            "file_sha256": "a" * 64,
            "body_sha256": "b" * 64,
        },
        operation_kind=operation_kind,
        expected_activation_id=ACTIVATION,
    )
    assert argv[0] == str(
        (Path(__file__).parents[1] / relative_executable).resolve()
    )
    assert argv[1:] == suffix
    assert environment == {
        "AWS_PROFILE": PROFILE,
        "AWS_REGION": REGION,
    }


def test_production_coordinator_allows_bounded_cloudformation_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: CloudFormation bootstrap is killed before stack completion."""

    coordinator = _coordinator_api()
    observed_timeout: list[int] = []
    observed_environment: list[dict[str, str]] = []
    monkeypatch.setattr(coordinator.os.path, "isfile", lambda path: True)

    def run(argv: list[str], **kwargs: object) -> object:
        observed_timeout.append(int(kwargs["timeout"]))
        observed_environment.append(dict(kwargs["env"]))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=b"{}\n",
            stderr=b"",
        )

    monkeypatch.setattr(coordinator.subprocess, "run", run)
    coordinator._aws_json(
        [
            "aws",
            "cloudformation",
            "wait",
            "stack-create-complete",
            "--stack-name",
            "keep-glm52-h1g-fence",
            "--profile",
            PROFILE,
            "--region",
            REGION,
        ]
    )
    assert observed_timeout == [30 * 60]
    assert observed_environment == [
        {
            "HOME": coordinator.os.environ.get("HOME", ""),
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
            "AWS_MAX_ATTEMPTS": "1",
            "AWS_RETRY_MODE": "standard",
        }
    ]


def test_production_coordinator_accepts_real_lambda_json_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: valid Lambda JSON is rejected for lacking sorted keys plus LF."""

    coordinator = _coordinator_api()
    raw = b'{"z":1,"a":2}'

    def aws_json(argv: list[str]) -> dict[str, object]:
        destination = Path(argv[argv.index("--payload") + 2])
        destination.write_bytes(raw)
        return {
            "StatusCode": 200,
            "ExecutedVersion": "19",
        }

    monkeypatch.setattr(coordinator, "_aws_json", aws_json)
    result = coordinator._lambda_invoke(
        {
            "function_version_arn": (
                "arn:aws:lambda:us-west-2:246813579024:function:"
                "keep-glm52-h1g-rehearsal-collector:19"
            ),
            "qualifier": "19",
            "event": {"closed": True},
        }
    )
    assert result == {
        "StatusCode": 200,
        "ExecutedVersion": "19",
        "Payload": raw,
    }


def test_version_pinned_gate_readback_preserves_full_canonical_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = _coordinator_api()
    measurements = tuple(
        rehearsal_measurement_from_mapping(_measurement(index))
        for index in range(1, 21)
    )
    raw = build_deployed_gate_document(
        account_id=ACCOUNT,
        region=REGION,
        run_id=RUN_ID,
        activation_id=ACTIVATION,
        deployment_identity_sha256="d" * 64,
        measurements=measurements,
    )
    gate = json.loads(raw)
    payload = {
        "activation_id": ACTIVATION,
        "key": f"rehearsal/gates/{ACTIVATION}/CLOSURE_BUDGET.json",
        "version_id": "gate-version-01",
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": gate["canonical_body_sha256"],
        "checksum_sha256_base64": base64.b64encode(
            hashlib.sha256(raw).digest()
        ).decode("ascii"),
        "status": "CLOSURE_BUDGET_PROVEN",
        "measurement_count": 20,
        "cold_environment_count": 5,
        "measurements_identity_sha256": gate[
            "measurements_identity_sha256"
        ],
    }
    monkeypatch.setattr(
        coordinator,
        "_get_s3_version",
        lambda **_kwargs: raw,
    )

    observed = coordinator._gate_readback(
        {
            "key": payload["key"],
            "version_id": payload["version_id"],
            "file_sha256": payload["file_sha256"],
            "body_sha256": payload["body_sha256"],
            "payload": payload,
        }
    )

    assert observed["content"] == raw
    assert observed["summary"] == {
        field: payload[field]
        for field in (
            "key",
            "version_id",
            "file_sha256",
            "body_sha256",
            "status",
            "measurement_count",
            "cold_environment_count",
            "measurements_identity_sha256",
        )
    }


def test_ambiguous_finalization_reconstructs_exact_payload_from_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = _coordinator_api()
    raw = build_deployed_gate_document(
        account_id=ACCOUNT,
        region=REGION,
        run_id=RUN_ID,
        activation_id=ACTIVATION,
        deployment_identity_sha256="d" * 64,
        measurements=tuple(
            rehearsal_measurement_from_mapping(_measurement(index))
            for index in range(1, 21)
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_unique_current_s3_object",
        lambda **_kwargs: (raw, "gate-version-01"),
    )
    request = {
        "function_version_arn": COLLECTOR_ARN,
        "qualifier": "19",
        "event": {
            "schema_version": 1,
            "record_type": "glm52_task11_finalize_rehearsal_gate_v1",
            "activation_id": ACTIVATION,
        },
    }

    result = coordinator._reconcile_finalize_invoke(request)
    payload = json.loads(result["Payload"])

    assert result["StatusCode"] == 200
    assert result["ExecutedVersion"] == "19"
    assert result["Payload"] == canonical_json_bytes(payload)
    assert payload["version_id"] == "gate-version-01"
    assert payload["file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert payload["collector_function_version_arn"] == COLLECTOR_ARN


def test_production_coordinator_rejects_foreign_account_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: a mutation is attempted with credentials for another account."""

    coordinator = _coordinator_api()
    calls: list[list[str]] = []

    def aws_json(argv: list[str]) -> dict[str, object]:
        calls.append(argv)
        return {
            "Account": "000000000000",
            "Expiration": "2099-01-01T00:00:00Z",
        }

    monkeypatch.setattr(coordinator, "_aws_json", aws_json)
    with pytest.raises(ValueError, match="account identity"):
        coordinator._dispatch(
            _coordinator_envelope(
                coordinator,
                action="execute",
                kind="migration-bootstrap",
                request={
                    "closed": True,
                },
            )
        )
    assert len(calls) == 1
    assert calls[0][1:3] == ["sts", "get-caller-identity"]


def test_production_coordinator_rejects_v1_and_tampering_before_aws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: legacy or request-tampered authority reaches AWS."""

    coordinator = _coordinator_api()
    aws_calls: list[list[str]] = []
    monkeypatch.setattr(
        coordinator,
        "_aws_json",
        lambda argv: aws_calls.append(argv),
    )
    legacy = _coordinator_envelope(
        coordinator,
        action="execute",
        kind="migration-bootstrap",
        request={"closed": True},
    )
    legacy["schema_version"] = 1
    legacy["record_type"] = "glm52_task13_coordinator_request_v1"
    with pytest.raises(ValueError, match="envelope drifted"):
        coordinator._dispatch(legacy)

    tampered = _coordinator_envelope(
        coordinator,
        action="execute",
        kind="migration-bootstrap",
        request={"closed": True},
    )
    tampered["request"] = {"closed": "tampered-after-signing"}
    with pytest.raises(ValueError, match="capability.*drifted"):
        coordinator._dispatch(tampered)
    assert aws_calls == []


def test_production_coordinator_unknown_route_fails_closed() -> None:
    """Break caught: arbitrary operation kinds reach a generic subprocess path."""

    coordinator = _coordinator_api()
    with pytest.raises(ValueError, match="not implemented"):
        coordinator._dispatch(
            _coordinator_envelope(
                coordinator,
                action="execute",
                kind="arbitrary-shell",
                request={"argv": ["/bin/sh", "-c", "false"]},
            )
        )


def test_production_coordinator_aws_cli_is_one_attempt_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: botocore silently retries a possibly sent mutation."""

    coordinator = _coordinator_api()
    observed = {}
    monkeypatch.setattr(
        coordinator.os.path,
        "isfile",
        lambda path: path == "/opt/homebrew/bin/aws",
    )

    def run(argv, **kwargs):
        observed["argv"] = argv
        observed["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            argv,
            0,
            b'{"Account":"246813579024"}\n',
            b"",
        )

    monkeypatch.setattr(coordinator.subprocess, "run", run)
    assert coordinator._aws_json(
        [
            "aws",
            "sts",
            "get-caller-identity",
            "--profile",
            PROFILE,
            "--region",
            REGION,
        ]
    ) == {"Account": ACCOUNT}
    assert observed["env"] == {
        "HOME": os.environ.get("HOME", ""),
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        "AWS_MAX_ATTEMPTS": "1",
        "AWS_RETRY_MODE": "standard",
    }


def test_production_coordinator_aws_mutation_route_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: the finite aws-command kind becomes an arbitrary AWS CLI."""

    coordinator = _coordinator_api()
    monkeypatch.setattr(
        coordinator,
        "_current_credential_evidence",
        lambda: {
            "account_id": ACCOUNT,
            "observed_epoch_seconds": 1,
            "expiration_epoch_seconds": 7201,
            "seconds_remaining": 7200,
        },
    )
    monkeypatch.setattr(coordinator, "_aws_json", lambda argv: {})
    with pytest.raises(ValueError, match="not implemented"):
        coordinator._dispatch(
            _coordinator_envelope(
                coordinator,
                action="execute",
                kind="aws-command",
                request={
                    "command_id": "delete-everything",
                    "operation": "s3:DeleteObject",
                    "account_id": ACCOUNT,
                    "profile": PROFILE,
                    "region": REGION,
                    "mutates_aws": True,
                    "execution_state": "AUTHORIZED",
                    "argv": [
                        "aws",
                        "s3",
                        "rm",
                        "s3://keep-glm52-models/",
                        "--recursive",
                        "--profile",
                        PROFILE,
                        "--region",
                        REGION,
                    ],
                },
            )
        )


def test_deploy_disabled_requires_authority_and_runs_only_finite_manifest() -> None:
    """Break caught: package construction reruns the already committed deploy."""

    api = _api()
    package = _package()
    journal = MemoryJournal()
    services = FiniteServices()
    with pytest.raises(api.CampaignRunnerError, match="controller authority"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="deploy-disabled",
            services=services,
            journal=journal,
            authority=None,
        )
    assert services.calls == []

    result = api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="deploy-disabled",
        services=services,
        journal=journal,
        authority=_authority(api, package, "deploy-disabled"),
    )
    assert result["status"] == "STAGE_COMMITTED"
    assert result["stage"] == "deploy-disabled"
    assert result["mutating_command_count"] == 0
    assert result["staged_journal_sha256"] == package[
        "disabled_deployment"
    ]["staged_infrastructure_evidence"]["staged_journal_sha256"]
    assert not any(call[0] in {"execute", "reconcile"} for call in services.calls)
    inspected = [
        call for call in services.calls if call[0] == "inspect"
    ]
    assert any(
        call[1] == "staged-infrastructure-adoption"
        for call in inspected
    )
    assert any(call[1] == "negative-iam-probe" for call in inspected)
    rendered = json.dumps(services.calls)
    assert "put-object" not in rendered
    assert "run-instances" not in rendered
    assert "request-spot" not in rendered
    assert "capacity-block" not in rendered.lower()


def test_deploy_disabled_rejects_legacy_three_tag_stack_identity() -> None:
    """Break caught: adoption accepts tags outside the canonical stack policy."""

    api = _api()
    package = _package("PREQUALIFICATION")
    package["disabled_deployment"]["tags"] = [
        ["CampaignRunId", RUN_ID],
        ["DeploymentState", "DISABLED"],
        ["Task13ActivationId", ACTIVATION],
    ]
    unsigned = dict(package)
    unsigned.pop("canonical_identity_sha256")
    package["canonical_identity_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(
        api.CampaignRunnerError,
        match="disabled deployment manifest drifted",
    ):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="deploy-disabled",
            services=FiniteServices(),
            journal=MemoryJournal(),
            authority=_authority(api, package, "deploy-disabled"),
        )


def test_deploy_disabled_rejects_opaque_cloudformation_evidence() -> None:
    """Break caught: a generic VERIFIED token substitutes for staged truth."""

    api = _api()
    package = _package()
    with pytest.raises(
        api.CampaignRunnerError,
        match="staged infrastructure exact readback",
    ):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="deploy-disabled",
            services=FiniteServices(opaque_cf_evidence=True),
            journal=MemoryJournal(),
            authority=_authority(api, package, "deploy-disabled"),
        )


@pytest.mark.parametrize(
    "obsolete_operation_id",
    [
        "deploy-disabled:migration-bootstrap:fence",
        "deploy-disabled:migration-bootstrap:support",
        "deploy-disabled:create-fence-disabled-change-set",
        "deploy-disabled:task12-orphan-authority-precreate",
        "deploy-disabled:task12-orphan-authority-postcreate",
    ],
)
def test_deploy_adoption_never_replays_obsolete_mutations(
    obsolete_operation_id: str,
) -> None:
    """Break caught: adoption inherits an old POSSIBLY_SENT mutation route."""

    api = _api()
    package = _package()
    journal = MemoryJournal()
    scope = (
        "support"
        if "support" in obsolete_operation_id
        or "orphan-authority" in obsolete_operation_id
        else "fence"
    )
    journal.for_scope(scope).append(
        {
            "operation_id": obsolete_operation_id,
            "state": "POSSIBLY_SENT",
        }
    )
    obsolete_before = copy.deepcopy(
        journal.for_scope(scope).rows
    )
    services = FiniteServices(
        lost_response_operation=obsolete_operation_id
    )
    result = api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="deploy-disabled",
        services=services,
        journal=journal,
        authority=_authority(api, package, "deploy-disabled"),
    )
    assert result["status"] == "STAGE_COMMITTED"
    assert not any(call[0] in {"execute", "reconcile"} for call in services.calls)
    assert journal.for_scope(scope).rows == obsolete_before

    before = copy.deepcopy(journal.rows)
    again = api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="deploy-disabled",
        services=services,
        journal=journal,
        authority=_authority(api, package, "deploy-disabled"),
    )
    assert again == result
    assert journal.rows == before


def test_first_five_are_concurrent_distinct_cold_and_fault_complete() -> None:
    """Break caught: five flags on one environment replace five cold starts."""

    api = _api()
    package = _package()
    journal = MemoryJournal()
    services = FiniteServices()
    api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="deploy-disabled",
        services=services,
        journal=journal,
        authority=_authority(api, package, "deploy-disabled"),
    )
    result = api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="collect-first-five",
        services=services,
        journal=journal,
        authority=_authority(api, package, "collect-first-five"),
    )
    assert result["status"] == "STAGE_COMMITTED"
    assert result["measurement_count"] == 5
    assert result["cold_environment_count"] == 5
    assert result["failure_kinds"] == [
        "THROTTLING",
        "PAGINATION",
        "NETWORK_AMBIGUITY",
    ]
    assert services.max_active_collect == 5
    invokes = [
        call
        for call in services.calls
        if call[:2] == ("execute", "collector-invoke")
    ]
    assert len(invokes) == 5

    bad_journal = MemoryJournal()
    bad_services = FiniteServices(reused_environment=True)
    api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="deploy-disabled",
        services=bad_services,
        journal=bad_journal,
        authority=_authority(api, package, "deploy-disabled"),
    )
    with pytest.raises(api.CampaignRunnerError, match="distinct cold"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="collect-first-five",
            services=bad_services,
            journal=bad_journal,
            authority=_authority(api, package, "collect-first-five"),
        )


def test_collect_version_mutant_stops_before_finalize() -> None:
    """Break caught: `$LATEST` or FunctionError evidence enters the gate."""

    api = _api()
    package = _package()
    journal = MemoryJournal()
    services = FiniteServices(wrong_version_at=3)
    api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="deploy-disabled",
        services=services,
        journal=journal,
        authority=_authority(api, package, "deploy-disabled"),
    )
    with pytest.raises(api.CampaignRunnerError, match="ExecutedVersion"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="collect-first-five",
            services=services,
            journal=journal,
            authority=_authority(api, package, "collect-first-five"),
        )
    assert not any(call[1] == "finalize-invoke" for call in services.calls)


def _through_finalize(
    api: object,
    package: dict[str, object],
    journal: MemoryJournal,
    services: FiniteServices,
) -> None:
    for stage in (
        "deploy-disabled",
        "collect-first-five",
        "collect-remaining",
        "finalize",
    ):
        result = api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage=stage,
            services=services,
            journal=journal,
            authority=_authority(api, package, stage),
        )
        assert result["status"] == "STAGE_COMMITTED"
        if stage == "finalize":
            finalizer = result["finalizer_payload"]
            readback = result["immutable_gate_readback"]
            assert type(finalizer) is dict
            assert type(readback) is dict
            assert finalizer["version_id"] == result["gate_version_id"]
            assert result["finalizer_function_version_arn"] == COLLECTOR_ARN
            assert result["finalizer_executed_version"] == "19"
            assert readback == {
                field: finalizer[field]
                for field in (
                    "key",
                    "version_id",
                    "file_sha256",
                    "body_sha256",
                    "status",
                    "measurement_count",
                    "cold_environment_count",
                    "measurements_identity_sha256",
                )
            }


def _prepare_finalization(
    api: object,
    package: dict[str, object],
    journal: MemoryJournal,
    services: FiniteServices,
) -> None:
    for stage in (
        "deploy-disabled",
        "collect-first-five",
        "collect-remaining",
    ):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage=stage,
            services=services,
            journal=journal,
            authority=_authority(api, package, stage),
        )


def test_committed_finalization_replays_original_capture_without_new_invoke() -> None:
    api = _api()
    package = _package()
    journal = MemoryJournal()
    services = FiniteServices()
    captures: list[object] = []
    _prepare_finalization(api, package, journal, services)

    for _attempt in range(2):
        result = api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="finalize",
            services=services,
            journal=journal,
            authority=_authority(api, package, "finalize"),
            finalization_capture=captures.append,
        )
        assert result["status"] == "STAGE_COMMITTED"

    assert len(captures) == 2
    assert captures[0] == captures[1]
    capture = cast(Any, captures[0])
    assert capture.status_code == 200
    assert capture.executed_version == "19"
    assert capture.payload_bytes == services.results[
        "finalize:closure-budget"
    ]["Payload"]
    assert capture.gate_bytes == services.gate_raw
    assert sum(
        call[:2] == ("execute", "finalize-invoke")
        for call in services.calls
    ) == 1
    assert sum(
        call[:2] == ("reconcile", "finalize-invoke")
        for call in services.calls
    ) == 1


def test_lost_finalization_response_reconciles_capture_without_reinvoke() -> None:
    api = _api()
    package = _package()
    journal = MemoryJournal()
    services = FiniteServices(
        lost_response_operation="finalize:closure-budget"
    )
    captures: list[object] = []
    _prepare_finalization(api, package, journal, services)

    with pytest.raises(api.CampaignRunnerError, match="POSSIBLY_SENT"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="finalize",
            services=services,
            journal=journal,
            authority=_authority(api, package, "finalize"),
            finalization_capture=captures.append,
        )
    result = api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="finalize",
        services=services,
        journal=journal,
        authority=_authority(api, package, "finalize"),
        finalization_capture=captures.append,
    )

    assert result["status"] == "STAGE_COMMITTED"
    assert len(captures) == 1
    capture = cast(Any, captures[0])
    assert capture.payload_bytes == services.results[
        "finalize:closure-budget"
    ]["Payload"]
    assert capture.gate_bytes == services.gate_raw
    assert sum(
        call[:2] == ("execute", "finalize-invoke")
        for call in services.calls
    ) == 1
    assert sum(
        call[:2] == ("reconcile", "finalize-invoke")
        for call in services.calls
    ) == 1


@pytest.mark.parametrize("drift", ["payload", "version", "deployment"])
def test_finalization_capture_rejects_committed_identity_drift(
    drift: str,
) -> None:
    api = _api()
    package = _package()
    journal = MemoryJournal()
    services = FiniteServices()
    _prepare_finalization(api, package, journal, services)
    api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="finalize",
        services=services,
        journal=journal,
        authority=_authority(api, package, "finalize"),
        finalization_capture=lambda _capture: None,
    )
    committed = services.results["finalize:closure-budget"]
    if drift == "payload":
        committed["Payload"] += b" "
    elif drift == "version":
        committed["ExecutedVersion"] = "20"
    else:
        assert services.gate_raw is not None
        gate = json.loads(services.gate_raw)
        gate["deployment_identity_sha256"] = "e" * 64
        body = dict(gate)
        body.pop("canonical_body_sha256")
        gate["canonical_body_sha256"] = canonical_sha256(body)
        services.gate_raw = canonical_json_bytes(gate)

    with pytest.raises(api.CampaignRunnerError, match="drift|canonical"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="finalize",
            services=services,
            journal=journal,
            authority=_authority(api, package, "finalize"),
            finalization_capture=lambda _capture: None,
        )
    assert sum(
        call[:2] == ("execute", "finalize-invoke")
        for call in services.calls
    ) == 1


def test_remaining_finalize_h100_six_az_launch_and_monitor() -> None:
    """Break caught: qualification, Spot, overlap, or a seventh AZ bypasses gates."""

    api = _api()
    package = _package()
    journal = MemoryJournal()
    services = FiniteServices(launch_accept_at=6)
    _through_finalize(api, package, journal, services)
    collect_executes = [
        call
        for call in services.calls
        if call[:2] == ("execute", "collector-invoke")
    ]
    assert len(collect_executes) == 20
    assert sum(
        call[:2] == ("execute", "finalize-invoke")
        for call in services.calls
    ) == 1

    with pytest.raises(api.CampaignRunnerError, match="required prior stage"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="h100-qualification",
            services=services,
            journal=journal,
            authority=_authority(
                api,
                package,
                "h100-qualification",
            ),
        )
    seeded = api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="qualification-cache-seed",
        services=services,
        journal=journal,
        authority=_authority(
            api,
            package,
            "qualification-cache-seed",
        ),
    )
    assert seeded["status"] == "STAGE_COMMITTED"
    qualified = api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="h100-qualification",
        services=services,
        journal=journal,
        authority=_authority(api, package, "h100-qualification"),
    )
    assert qualified["status"] == "STAGE_COMMITTED"
    assert qualified["h100_resume_ready"] == {
        "key": (
            f"campaigns/{RUN_ID}/qualification/"
            "H100_RESUME_READY.json"
        ),
        "version_id": "3Lg" + "c" * 16,
        "file_sha256": "c" * 64,
        "body_sha256": "c" * 64,
        "canonical_identity_sha256": canonical_sha256(
            {
                "key": (
                    f"campaigns/{RUN_ID}/qualification/"
                    "H100_RESUME_READY.json"
                ),
                "version_id": "3Lg" + "c" * 16,
                "file_sha256": "c" * 64,
                "body_sha256": "c" * 64,
            }
        ),
    }
    launched = api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="launch",
        services=services,
        journal=journal,
        authority=_authority(api, package, "launch"),
    )
    assert launched["status"] == "STAGE_COMMITTED"
    assert launched["accepted_execution_arn"].endswith(":task13-test")
    assert launched["accepted_instance_id"] == "i-00000000000000006"
    assert launched["accepted_availability_zone"] == "us-west-2f"
    assert launched["workflow_invocation_count"] == 1
    assert launched["capacity_outcome_count"] == 6
    assert launched["workflow_reconciliation"]["classification"] == (
        "WORKER_ALLOCATED"
    )
    assert launched["workflow_reconciliation"][
        "observation_identity_sha256"
    ] == canonical_sha256(
        {
            key: launched["workflow_reconciliation"][key]
            for key in launched["workflow_reconciliation"]
            if key != "observation_identity_sha256"
        }
    )
    launch_calls = [
        call
        for call in services.calls
        if call[:2] == ("execute", "guarded-launch")
    ]
    assert len(launch_calls) == 1
    authority_calls = [
        call
        for call in services.calls
        if call[:2]
        == ("execute", "launch-authority-materialize")
    ]
    assert authority_calls == [
        (
            "execute",
            "launch-authority-materialize",
            "launch-authority:attempt-01",
        )
    ]
    assert services.calls.index(authority_calls[0]) < services.calls.index(
        launch_calls[0]
    )
    request = services.launch_requests[0]
    assert request["capacity_plan"]["attempts"] == (
        package["production_retry_plan"]["attempts"]
    )
    assert request["execution_authority"]["liability_action"][
        "shared_attempt_counter"
    ] == 1
    assert request["execution_authority"]["spend_authority"][
        "gpu_reserve_usd"
    ] == "13.76"
    assert request["h100_resume_ready"]["key"] == (
        f"campaigns/{RUN_ID}/qualification/H100_RESUME_READY.json"
    )
    assert request["execution_authority"]["authority_record"]["key"] == (
        f"campaigns/{RUN_ID}/authorities/task13/{ACTIVATION}/"
        "launch-attempt-01.json"
    )
    assert request["immutable_inputs"] == (
        package["production_retry_plan"]["immutable_inputs"]
    )
    assert request["route"] == [
        "aws/glm52-gpu/scripts/submit_sky_campaign.sh",
        "--production",
        "start",
    ]
    assert request["execution_custody"] == package[
        "production_retry_plan"
    ]["execution_custody"]
    resumed = api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="launch",
        services=services,
        journal=journal,
        authority=_authority(api, package, "launch"),
    )
    assert resumed["accepted_execution_arn"] == launched[
        "accepted_execution_arn"
    ]
    assert len(
        [
            call
            for call in services.calls
            if call[:2] == ("execute", "guarded-launch")
        ]
    ) == 1
    p5_checks = [
        call
        for call in services.calls
        if call[:2] == ("inspect", "p5-zero")
        and type(call[2]) is dict
        and call[2].get("workflow_invocation") == 1
    ]
    assert len(p5_checks) == 1
    monitored = api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="monitor",
        services=services,
        journal=journal,
        authority=_authority(api, package, "monitor"),
    )
    assert monitored["status"] == "RUNNING"
    assert monitored["instance_ids"] == ["i-00000000000000006"]


def test_capacity_exhaustion_reconciles_one_workflow_without_replay() -> None:
    """Break caught: six rejected AZs cause a second workflow execution."""

    api = _api()
    package = _package()
    journal = MemoryJournal()
    services = FiniteServices(launch_accept_at=7)
    _through_finalize(api, package, journal, services)
    for stage in ("qualification-cache-seed", "h100-qualification"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage=stage,
            services=services,
            journal=journal,
            authority=_authority(api, package, stage),
        )
    for _ in range(2):
        with pytest.raises(
            api.CampaignRunnerError,
            match="exhausted all six availability zones",
        ):
            api.run_campaign_stage(
                package=package,
                reviewed_artifacts=package["reviewed_artifacts"],
                stage="launch",
                services=services,
                journal=journal,
                authority=_authority(api, package, "launch"),
            )
    assert [
        call[:2]
        for call in services.calls
        if call[1] == "guarded-launch"
    ] == [
        ("execute", "guarded-launch"),
        ("reconcile", "guarded-launch"),
    ]


def test_lost_launch_authority_write_reconciles_before_workflow_start() -> None:
    """Break caught: lost authority response rewrites it or starts too early."""

    api = _api()
    package = _package()
    journal = MemoryJournal()
    operation_id = "launch-authority:attempt-01"
    services = FiniteServices(
        launch_accept_at=1,
        lost_response_operation=operation_id,
    )
    _through_finalize(api, package, journal, services)
    for stage in ("qualification-cache-seed", "h100-qualification"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage=stage,
            services=services,
            journal=journal,
            authority=_authority(api, package, stage),
        )
    with pytest.raises(api.CampaignRunnerError, match="POSSIBLY_SENT"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="launch",
            services=services,
            journal=journal,
            authority=_authority(api, package, "launch"),
        )
    assert not [
        call
        for call in services.calls
        if call[:2] == ("execute", "guarded-launch")
    ]
    launched = api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="launch",
        services=services,
        journal=journal,
        authority=_authority(api, package, "launch"),
    )
    assert launched["status"] == "STAGE_COMMITTED"
    assert [
        call
        for call in services.calls
        if call[1] == "launch-authority-materialize"
    ] == [
        (
            "execute",
            "launch-authority-materialize",
            operation_id,
        ),
        (
            "reconcile",
            "launch-authority-materialize",
            operation_id,
        ),
    ]
    assert sum(
        call[:2] == ("execute", "guarded-launch")
        for call in services.calls
    ) == 1


def test_lost_sole_sender_authority_write_reconciles_before_s3_or_start() -> None:
    """Break caught: ambiguous DDB PutItem permits S3 authority or workflow send."""

    api = _api()
    package = _package()
    journal = MemoryJournal()
    operation_id = "sole-sender-authority:generation-00000001"
    services = FiniteServices(
        launch_accept_at=1,
        lost_response_operation=operation_id,
    )
    _through_finalize(api, package, journal, services)
    for stage in ("qualification-cache-seed", "h100-qualification"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage=stage,
            services=services,
            journal=journal,
            authority=_authority(api, package, stage),
        )
    with pytest.raises(api.CampaignRunnerError, match="POSSIBLY_SENT"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="launch",
            services=services,
            journal=journal,
            authority=_authority(api, package, "launch"),
        )
    assert not [
        call
        for call in services.calls
        if call[1] in {"launch-authority-materialize", "guarded-launch"}
    ]
    launched = api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="launch",
        services=services,
        journal=journal,
        authority=_authority(api, package, "launch"),
    )
    assert launched["status"] == "STAGE_COMMITTED"
    assert [
        call
        for call in services.calls
        if call[1] == "sole-sender-authority-materialize"
    ] == [
        (
            "execute",
            "sole-sender-authority-materialize",
            operation_id,
        ),
        (
            "reconcile",
            "sole-sender-authority-materialize",
            operation_id,
        ),
    ]
    ddb_reconcile = services.calls.index(
        (
            "reconcile",
            "sole-sender-authority-materialize",
            operation_id,
        )
    )
    s3_execute = next(
        index
        for index, call in enumerate(services.calls)
        if call[:2] == ("execute", "launch-authority-materialize")
    )
    start_execute = next(
        index
        for index, call in enumerate(services.calls)
        if call[:2] == ("execute", "guarded-launch")
    )
    assert ddb_reconcile < s3_execute < start_execute


def test_h100_qualification_rejects_foreign_well_formed_resume_marker() -> None:
    """Break caught: a suffix-only H100 marker authorizes production."""

    api = _api()
    package = _package()
    journal = MemoryJournal()
    services = FiniteServices(foreign_h100_marker=True)
    _through_finalize(api, package, journal, services)
    api.run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="qualification-cache-seed",
        services=services,
        journal=journal,
        authority=_authority(api, package, "qualification-cache-seed"),
    )
    with pytest.raises(api.CampaignRunnerError, match="marker coordinate"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="h100-qualification",
            services=services,
            journal=journal,
            authority=_authority(api, package, "h100-qualification"),
        )


def test_cache_seed_requires_exact_immutable_marker_readback() -> None:
    """Break caught: coordinator output substitutes for exact S3 version readback."""

    api = _api()
    package = _package()
    journal = MemoryJournal()
    services = FiniteServices(marker_readback_drift=True)
    _through_finalize(api, package, journal, services)
    with pytest.raises(api.CampaignRunnerError, match="readback"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="qualification-cache-seed",
            services=services,
            journal=journal,
            authority=_authority(
                api,
                package,
                "qualification-cache-seed",
            ),
        )


def test_launch_refuses_bad_presend_spend_or_liability_authority() -> None:
    """Break caught: launch validates spend/action only after the send."""

    api = _api()
    package = _package()
    journal = MemoryJournal()
    services = FiniteServices(bad_launch_authority=True)
    _through_finalize(api, package, journal, services)
    for stage in ("qualification-cache-seed", "h100-qualification"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage=stage,
            services=services,
            journal=journal,
            authority=_authority(api, package, stage),
        )
    with pytest.raises(api.CampaignRunnerError, match="launch authority"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="launch",
            services=services,
            journal=journal,
            authority=_authority(api, package, "launch"),
        )
    assert services.launch_requests == []


def test_launch_exact_reads_every_immutable_input_before_send() -> None:
    """Break caught: launch trusts local coordinates without VersionId reads."""

    api = _api()
    package = _package()
    journal = MemoryJournal()
    services = FiniteServices(artifact_readback_drift=True)
    _through_finalize(api, package, journal, services)
    for stage in ("qualification-cache-seed", "h100-qualification"):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage=stage,
            services=services,
            journal=journal,
            authority=_authority(api, package, stage),
        )
    with pytest.raises(
        api.CampaignRunnerError,
        match="immutable artifact exact VersionId readback",
    ):
        api.run_campaign_stage(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="launch",
            services=services,
            journal=journal,
            authority=_authority(api, package, "launch"),
        )
    assert services.launch_requests == []


def test_terminal_stage_requires_drained_markers_zero_billable_and_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: SUCCEEDED status substitutes for drain and cost proof."""

    api = _api()

    def validate_terminal_fixture(
        proof: object,
        *,
        terminal_verified_value: object,
        campaign_drained_value: object,
        spend_ledger_raw: bytes,
    ) -> dict[str, object]:
        assert terminal_verified_value == {
            "fixture": "terminal-verified"
        }
        assert campaign_drained_value == {
            "fixture": "campaign-drained"
        }
        assert spend_ledger_raw == b"fixture-spend-ledger\n"
        assert type(proof) is dict
        if proof["ec2_billable_instance_ids"]:
            raise ValueError("terminal proof billable instances drifted")
        return {
            "activation_id": proof["activation_id"],
            "sky_state": proof["sky_state"],
            "campaign_drained": proof["campaign_drained"],
            "terminal_verified": proof["terminal_verified"],
            "settlement": {
                "gpu_spend_ledger": {
                    "bucket": (
                        "keep-glm52-models-"
                        "246813579024-us-west-2"
                    ),
                    "key": (
                        f"campaigns/{RUN_ID}/runtime/"
                        "GPU_SPEND_LEDGER.jsonl"
                    ),
                    "version_id": "spend-ledger-version",
                    "file_sha256": "a" * 64,
                    "body_sha256": "b" * 64,
                    "head_record_sha256": "c" * 64,
                    "canonical_identity_sha256": "d" * 64,
                },
                "total_gpu_seconds": proof["total_gpu_seconds"],
                "total_gpu_cost_usd": proof["total_gpu_cost_usd"],
                "remaining_gpu_seconds": proof[
                    "remaining_gpu_seconds"
                ],
                "remaining_gpu_usd": proof["remaining_gpu_usd"],
                "retained_costs": proof["retained_costs"],
            },
        }

    monkeypatch.setattr(
        api,
        "validate_task13_terminal_proof",
        validate_terminal_fixture,
    )
    package = _package()
    for drift, accepted in ((False, True), (True, False)):
        journal = MemoryJournal()
        services = FiniteServices(terminal_proof_drift=drift)
        _through_finalize(api, package, journal, services)
        for stage in (
            "qualification-cache-seed",
            "h100-qualification",
            "launch",
        ):
            api.run_campaign_stage(
                package=package,
                reviewed_artifacts=package["reviewed_artifacts"],
                stage=stage,
                services=services,
                journal=journal,
                authority=_authority(api, package, stage),
            )
        if accepted:
            result = api.run_campaign_stage(
                package=package,
                reviewed_artifacts=package["reviewed_artifacts"],
                stage="terminal",
                services=services,
                journal=journal,
                authority=_authority(api, package, "terminal"),
            )
            assert result["status"] == "STAGE_COMMITTED"
            assert result["total_gpu_seconds"] == 4500
            assert result["remaining_gpu_usd"] == "1252.16"
        else:
            with pytest.raises(
                api.CampaignRunnerError,
                match="terminal proof",
            ):
                api.run_campaign_stage(
                    package=package,
                    reviewed_artifacts=package["reviewed_artifacts"],
                    stage="terminal",
                    services=services,
                    journal=journal,
                    authority=_authority(api, package, "terminal"),
                )
