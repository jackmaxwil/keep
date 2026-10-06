"""Canonical 14-step staged-infrastructure fixture construction."""

from __future__ import annotations

import base64
import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.cloudformation_stacks import (
    STACK_TAGS,
    BootstrapCoordinate,
    MigrationBootstrapResult,
    MigrationBundle,
    MigrationEvidence,
    MigrationExecutionAuthority,
    StackIdentity,
    StackKind,
    TemplateArtifact,
    initial_migration_execution_state,
    migration_execution_state_projection,
)
from glm52_enforcement.task13_migration_adapter import (
    MIGRATION_TEMPLATE_KEYS,
    execution_authority_projection,
    sealed_migration_projection,
)

ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
RUN_ID = "glm52-sky-20260724"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
DEPLOYMENT_ROLE_ARN = (
    "arn:aws:iam::246813579024:role/"
    "keep-glm52-h1g-cloudformation-deployment"
)
STEPS = (
    "ACCOUNT_LIVE_BASELINE",
    "BOOTSTRAP_STACK_MIGRATION",
    "MATERIALIZE_PRE_SUPPORT",
    "UPDATE_RETAINED_PRE_SUPPORT",
    "MATERIALIZE_FULL_SUPPORT_INPUTS",
    "BUILD_PUBLISH_SUPPORT",
    "MATERIALIZE_PUBLISH_STACK_MIGRATION",
    "CAPTURE_PRECREATE_ORPHAN_AUTHORITY",
    "EXECUTE_STACK_MIGRATION",
    "MATERIALIZE_POSTCREATE_FRAGMENT",
    "UPDATE_RETAINED_FINAL",
    "POSTPUBLICATION_AUTHORITY",
    "FINAL_EXACT_READBACK",
    "PROVE_NO_WORKER_ACTIVATION",
)


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _published_runtime_coordinate(
    *,
    key_kind: str,
    seed: bytes,
    version_id: str,
    size_bytes: int,
) -> dict[str, object]:
    identity = hashlib.sha256(seed).hexdigest()
    return {
        "bucket": BUCKET,
        "key": f"task13/artifacts/{key_kind}/{identity}.zip",
        "version_id": version_id,
        "size_bytes": size_bytes,
        "file_sha256": identity,
    }


def _set_coordinate_fields(
    evidence: dict[str, object],
    *,
    prefix: str,
    coordinate: Mapping[str, object],
) -> None:
    for field in (
        "bucket",
        "key",
        "version_id",
        "file_sha256",
        "body_sha256",
    ):
        evidence[f"{prefix}_{field}"] = coordinate[field]


def _identity(
    kind: StackKind,
    *,
    retained_stack_id: str,
    fence_stack_id: str,
    support_stack_id: str,
) -> StackIdentity:
    values = {
        StackKind.RETAINED: ("keep-glm52-gpu", retained_stack_id),
        StackKind.FENCE: ("keep-glm52-h1g-fence", fence_stack_id),
        StackKind.SUPPORT: ("keep-glm52-h1g-support", support_stack_id),
    }
    name, stack_id = values[kind]
    return StackIdentity(
        kind=kind,
        name=name,
        stack_id=stack_id,
        account_id=ACCOUNT_ID,
        region=REGION,
        termination_protection=True,
        tags=STACK_TAGS,
    )


def _migration_artifact(
    *,
    stage: str,
    kind: StackKind,
    template: Mapping[str, object],
    stack_id: str,
    version_id: str,
) -> TemplateArtifact:
    canonical = canonical_json_bytes(template)
    body = canonical + b"\n"
    key = MIGRATION_TEMPLATE_KEYS[stage]
    return TemplateArtifact(
        stage=stage,
        stack_id=stack_id,
        template_url=(
            f"https://{BUCKET}.s3.{REGION}.amazonaws.com/"
            f"{key}?versionId={version_id}"
        ),
        version_id=version_id,
        body=body,
        sha256=hashlib.sha256(body).hexdigest(),
        template_body_sha256=hashlib.sha256(canonical).hexdigest(),
        policy_sha256=None,
        template=copy.deepcopy(dict(template)),
    )


def _rechain(
    committed: Mapping[str, Mapping[str, object]],
    *,
    request_identity_sha256: str,
) -> tuple[list[dict[str, object]], bytes]:
    records: list[dict[str, object]] = []
    previous: str | None = None
    for sequence, step in enumerate(STEPS, 1):
        row: dict[str, object] = {
            "schema_version": 1,
            "record_type": (
                "glm52_task13_staged_deployment_journal_v1"
            ),
            "sequence": sequence,
            "request_identity_sha256": request_identity_sha256,
            "step": step,
            "state": "COMMITTED",
            "evidence": copy.deepcopy(dict(committed[step])),
            "previous_record_sha256": previous,
        }
        row["record_sha256"] = _sha(row)
        previous = str(row["record_sha256"])
        records.append(row)
    raw = b"".join(
        canonical_json_bytes(row) + b"\n" for row in records
    )
    return records, raw


def build_staged_infrastructure_evidence(
    artifacts: Sequence[Mapping[str, object]],
    *,
    activation_id: str,
    retained_stack_id: str,
    fence_stack_id: str,
    support_stack_id: str,
    output_directory: str = "/tmp/glm52-task13-staged-fixture",
    journal_path: str | None = None,
) -> dict[str, object]:
    """Build one self-hashed coordinator-backed staged evidence document."""

    source_rows = {
        str(row["artifact_kind"]): row for row in artifacts
    }
    by_kind = {
        kind: copy.deepcopy(dict(row))
        for kind, row in source_rows.items()
    }
    support_lambda = _published_runtime_coordinate(
        key_kind="support-lambda",
        seed=b"support-lambda-archive",
        version_id="3LgSupportLambdaArchiveVersion",
        size_bytes=4096,
    )
    cryptography_layer = _published_runtime_coordinate(
        key_kind="cryptography-layer-python312-x86_64",
        seed=b"cryptography-layer-archive",
        version_id="3LgCryptographyLayerArchiveVersion",
        size_bytes=8192,
    )
    cryptography_layer_version_arn = (
        "arn:aws:lambda:us-west-2:246813579024:layer:"
        "keep-glm52-h1g-cryptography-py312-x86-64:7"
    )
    final_fence_template = {
        "Resources": {
            "H1gProductionFenceBucketPolicy": {
                "Type": "AWS::S3::BucketPolicy"
            }
        }
    }
    disabled_support_template = {
        "Resources": {
            "SupportFunctionRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "RoleName": "keep-glm52-h1g-support-function-role",
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
                    "FunctionName": "keep-glm52-h1g-support-function",
                    "Code": {
                        "S3Bucket": support_lambda["bucket"],
                        "S3Key": support_lambda["key"],
                        "S3ObjectVersion": support_lambda["version_id"],
                    },
                    "Handler": "support_custom_resource_handler.main",
                    "Runtime": "python3.12",
                    "Architectures": ["x86_64"],
                    "MemorySize": 256,
                    "Timeout": 840,
                    "ReservedConcurrentExecutions": 1,
                    "Role": {
                        "Fn::GetAtt": ["SupportFunctionRole", "Arn"]
                    },
                    "Environment": {
                        "Variables": {
                            "GLM52_ACTIVATION_ID": activation_id,
                            "GLM52_RUN_ID": RUN_ID,
                        }
                    },
                    "Layers": [cryptography_layer_version_arn],
                },
            },
            "SupportVersion": {
                "Type": "AWS::Lambda::Version",
                "Properties": {
                    "FunctionName": {"Ref": "SupportFunction"},
                    "Description": (
                        "code=" + str(support_lambda["file_sha256"])
                    ),
                },
            },
        }
    }
    for kind, key, template in (
        (
            "FENCE_TEMPLATE",
            MIGRATION_TEMPLATE_KEYS["fence-transfer"],
            final_fence_template,
        ),
        (
            "SUPPORT_TEMPLATE",
            "task13/templates/support-disabled.yaml",
            disabled_support_template,
        ),
    ):
        canonical = canonical_json_bytes(template)
        coordinate = by_kind[kind]
        coordinate["key"] = key
        coordinate["file_sha256"] = hashlib.sha256(
            canonical + b"\n"
        ).hexdigest()
        coordinate["body_sha256"] = hashlib.sha256(
            canonical
        ).hexdigest()
        mutable = source_rows[kind]
        if type(mutable) is dict:
            mutable.update(coordinate)

    staged_kinds = {
        "BOOTSTRAP_TEMPLATE",
        "FENCE_TEMPLATE",
        "RETAINED_FOUNDATION_TEMPLATE",
        "RETAINED_PRE_SUPPORT_TEMPLATE",
        "RETAINED_TEMPLATE",
        "SUPPORT_TEMPLATE",
        "SUPPORT_INPUTS",
    }
    fixed = [by_kind[kind] for kind in sorted(staged_kinds)]
    foundation = {
        "stack_id": retained_stack_id,
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
    }
    support_publication_body = {
        "schema_version": 1,
        "record_type": (
            "glm52_task13_support_artifact_publication_v1"
        ),
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "runtime": "python3.12",
        "architecture": "x86_64",
        "source_manifest_identity_sha256": hashlib.sha256(
            b"support-artifact-manifest"
        ).hexdigest(),
        "support_lambda_archive": support_lambda,
        "cryptography_layer_archive": cryptography_layer,
    }
    support_publication = {
        **support_publication_body,
        "canonical_identity_sha256": _sha(
            support_publication_body
        ),
    }

    bootstrap = by_kind["BOOTSTRAP_TEMPLATE"]
    bootstrap_url = (
        f"https://{BUCKET}.s3.{REGION}.amazonaws.com/"
        f"{bootstrap['key']}?versionId={bootstrap['version_id']}"
    )
    authority = MigrationExecutionAuthority(
        deployment_role_arn=DEPLOYMENT_ROLE_ARN,
        deployment_role_id="AROAEXACTRETAINEDROLEID",
        fence_bootstrap=BootstrapCoordinate(
            kind=StackKind.FENCE,
            template_url=bootstrap_url,
            version_id=str(bootstrap["version_id"]),
        ),
        support_bootstrap=BootstrapCoordinate(
            kind=StackKind.SUPPORT,
            template_url=bootstrap_url,
            version_id=str(bootstrap["version_id"]),
        ),
        import_change_set_name="glm52-task13-import-fence-v1",
        action_identity_sha256="8" * 64,
    )
    seed_body = {
        "schema_version": 1,
        "record_type": "glm52_task13_stack_migration_seed_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "authority": execution_authority_projection(authority),
        "initial_state": migration_execution_state_projection(
            initial_migration_execution_state(authority)
        ),
        "publication_plan": [
            {
                "stage": stage,
                "bucket": BUCKET,
                "key": key,
            }
            for stage, key in MIGRATION_TEMPLATE_KEYS.items()
        ],
    }
    stack_migration_seed = {
        **seed_body,
        "canonical_identity_sha256": _sha(seed_body),
    }
    production_request = {
        "schema_version": 1,
        "record_type": "glm52_task13_production_operations_v1",
        "activation_id": activation_id,
        "output_directory": output_directory,
        "retained_stack_id": retained_stack_id,
        "stack_migration_seed": stack_migration_seed,
        "bootstrap_template": bootstrap,
        "pre_support_runtime_inputs": {"fixture": "pre-support"},
        "support_input_materialization_request": {
            "fixture": "support-materialization"
        },
        "support_price_card": {"fixture": "price-card"},
        "orphan_precreate": {
            "path": str(
                Path(output_directory)
                / "PRECREATE_ORPHAN_BASELINE.json"
            ),
            "expected_retained": [],
        },
        "retained_foundation_evidence": copy.deepcopy(foundation),
        "fixed_artifacts": [
            by_kind["RETAINED_FOUNDATION_TEMPLATE"]
        ],
        "support_artifact_publication": support_publication,
    }
    template_path = (
        Path(journal_path)
        if journal_path is not None
        else (
            Path(__file__).resolve().parent
            / "fixtures/glm52_task13_staged_deployment_journal_v1.jsonl"
        )
    )
    staged_request = {
        "schema_version": 1,
        "record_type": "glm52_task13_staged_deployment_request_v1",
        "activation_id": activation_id,
        "journal_path": str(template_path),
        "production_request": production_request,
    }
    request_identity = _sha(staged_request)

    identities = {
        kind: _identity(
            kind,
            retained_stack_id=retained_stack_id,
            fence_stack_id=fence_stack_id,
            support_stack_id=support_stack_id,
        )
        for kind in StackKind
    }
    bootstrap_result = MigrationBootstrapResult(
        fence=identities[StackKind.FENCE],
        support=identities[StackKind.SUPPORT],
        action_identity_sha256=authority.action_identity_sha256,
        state_revision=1,
        reconciled_creates=(),
    )
    templates = {
        "retention-only": {
            "Resources": {"ModelBucket": {"Type": "AWS::S3::Bucket"}}
        },
        "post-retain": {
            "Resources": {"ModelBucket": {"Type": "AWS::S3::Bucket"}}
        },
        "fence-import": final_fence_template,
        "fence-transfer": final_fence_template,
        "disabled-support": disabled_support_template,
    }
    kinds = {
        "retention-only": StackKind.RETAINED,
        "post-retain": StackKind.RETAINED,
        "fence-import": StackKind.FENCE,
        "fence-transfer": StackKind.FENCE,
        "disabled-support": StackKind.SUPPORT,
    }
    bundle = MigrationBundle(
        artifacts=tuple(
            _migration_artifact(
                stage=stage,
                kind=kinds[stage],
                template=template,
                stack_id=identities[kinds[stage]].stack_id,
                version_id=(
                    str(by_kind["FENCE_TEMPLATE"]["version_id"])
                    if stage == "fence-transfer"
                    else (
                        str(
                            by_kind["SUPPORT_TEMPLATE"]["version_id"]
                        )
                        if stage == "disabled-support"
                        else stage + "-version-1"
                    )
                ),
            )
            for stage, template in templates.items()
        ),
        manifest={
            "record_type": "glm52_h1g_stack_migration_manifest_v1"
        },
    )
    policy = {"Version": "2012-10-17", "Statement": []}
    migration_evidence = MigrationEvidence(
        retained=identities[StackKind.RETAINED],
        fence=identities[StackKind.FENCE],
        support=identities[StackKind.SUPPORT],
        current_policy_logical_id="CurrentPolicy",
        current_policy_physical_id=BUCKET,
        current_policy_stack_id=retained_stack_id,
        bucket_name=BUCKET,
        import_identifier=(("Bucket", BUCKET),),
        live_policy=policy,
        direct_policy_readbacks=(
            policy,
            copy.deepcopy(policy),
            copy.deepcopy(policy),
        ),
        retained_deployment_role_id=authority.deployment_role_id,
        retained_resource_physical_ids=(BUCKET,),
        retained_export_names=(),
        support_export_names=(),
    )
    durable_state = replace(
        initial_migration_execution_state(authority),
        revision=1,
    )
    sealed = dict(
        sealed_migration_projection(
            authority=authority,
            evidence=migration_evidence,
            bundle=bundle,
            bootstrap_result=bootstrap_result,
            durable_state=migration_execution_state_projection(
                durable_state
            ),
        )
    )
    migration_artifacts = {
        artifact.stage: artifact for artifact in bundle.artifacts
    }
    common = {
        "worker_activation_attempts": 0,
        "raw_ec2_launch_calls": 0,
    }
    pre_fragment = "b" * 64
    final_fragment = "a" * 64
    committed: dict[str, dict[str, object]] = {
        "ACCOUNT_LIVE_BASELINE": {
            **common,
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "profile": PROFILE,
            "model_bucket_name": BUCKET,
            "retained_stack_id": retained_stack_id,
            "retained_role_arn": DEPLOYMENT_ROLE_ARN,
            "retained_termination_protection": True,
            "credential_observed_at": "2026-07-29T20:00:00Z",
            "credential_expiration": "2026-07-29T22:00:00Z",
            "credential_seconds_remaining": 7200,
            "exact": True,
        },
        "BOOTSTRAP_STACK_MIGRATION": {
            **common,
            "fence_stack_id": fence_stack_id,
            "support_stack_id": support_stack_id,
            "deployment_role_arn": DEPLOYMENT_ROLE_ARN,
            "fence_anchor_inert": True,
            "support_anchor_inert": True,
            "fence_stack_status": "CREATE_COMPLETE",
            "support_stack_status": "CREATE_COMPLETE",
            "state_revision": 1,
            "reconciled_creates": [],
            "action_identity_sha256": authority.action_identity_sha256,
            "durable_state_sha256": _sha(
                migration_execution_state_projection(durable_state)
            ),
        },
        "MATERIALIZE_PRE_SUPPORT": {
            **common,
            "narrow_inputs": True,
            "support_build_inputs_sha256": "a" * 64,
            "retained_fragment_sha256": pre_fragment,
        },
        "UPDATE_RETAINED_PRE_SUPPORT": {
            **common,
            "stack_id": retained_stack_id,
            "stack_status": "UPDATE_COMPLETE",
            "change_set_type": "UPDATE",
            "role_arn": DEPLOYMENT_ROLE_ARN,
            "phase": "pre-support",
            "source_fragment_sha256": pre_fragment,
            "change_set_id": "retained-pre-change-set",
            "change_set_name": "glm52-task13-retained-pre-support-v1",
            "readback_sha256": "a" * 64,
        },
        "MATERIALIZE_FULL_SUPPORT_INPUTS": {
            **common,
            "complete": True,
            "support_inputs_sha256": "a" * 64,
            "terminal_v2_export_present": True,
            "support_lambda_archive": support_lambda,
            "cryptography_layer_archive": cryptography_layer,
            "cryptography_layer_version_arn": (
                cryptography_layer_version_arn
            ),
            "cryptography_layer_code_sha256": base64.b64encode(
                bytes.fromhex(
                    str(cryptography_layer["file_sha256"])
                )
            ).decode("ascii"),
        },
        "BUILD_PUBLISH_SUPPORT": {
            **common,
            "immutable": True,
            "support_template_version_id": by_kind[
                "SUPPORT_TEMPLATE"
            ]["version_id"],
            "support_inputs_version_id": by_kind[
                "SUPPORT_INPUTS"
            ]["version_id"],
            "support_template_sha256": by_kind[
                "SUPPORT_TEMPLATE"
            ]["body_sha256"],
            "support_inputs_sha256": by_kind[
                "SUPPORT_INPUTS"
            ]["body_sha256"],
        },
        "MATERIALIZE_PUBLISH_STACK_MIGRATION": {
            **common,
            "sealed_migration": sealed,
            "sealed_migration_identity_sha256": sealed[
                "canonical_identity_sha256"
            ],
            "action_identity_sha256": authority.action_identity_sha256,
            "fence_stack_id": fence_stack_id,
            "support_stack_id": support_stack_id,
            "final_fence_template_sha256": migration_artifacts[
                "fence-transfer"
            ].template_body_sha256,
            "support_template_sha256": migration_artifacts[
                "disabled-support"
            ].template_body_sha256,
            "worker_activation_enabled": False,
            "worker_activation_evidence_sha256": "a" * 64,
            "complete": True,
        },
        "CAPTURE_PRECREATE_ORPHAN_AUTHORITY": {
            **common,
            "phase": "PRECREATE",
            "authority_sha256": "a" * 64,
            "complete": True,
        },
        "EXECUTE_STACK_MIGRATION": {
            **common,
            "fence_stack_id": fence_stack_id,
            "support_stack_id": support_stack_id,
            "import_change_set_id": "import-change-set-id",
            "operation_count": 7,
            "reconciled_creates": [],
            "direct_policy_sha256": ["a" * 64] * 3,
            "action_identity_sha256": authority.action_identity_sha256,
            "bundle_identity_sha256": sealed[
                "canonical_identity_sha256"
            ],
            "complete": True,
        },
        "MATERIALIZE_POSTCREATE_FRAGMENT": {
            **common,
            "physical_truth_exact": True,
            "retained_fragment_sha256": final_fragment,
            "support_stack_id": support_stack_id,
        },
        "UPDATE_RETAINED_FINAL": {
            **common,
            "stack_id": retained_stack_id,
            "stack_status": "UPDATE_COMPLETE",
            "change_set_type": "UPDATE",
            "role_arn": DEPLOYMENT_ROLE_ARN,
            "phase": "final",
            "source_fragment_sha256": final_fragment,
            "change_set_id": "retained-final-change-set",
            "change_set_name": "glm52-task13-retained-final-v1",
            "readback_sha256": "b" * 64,
        },
        "POSTPUBLICATION_AUTHORITY": {
            **common,
            "phase": "POSTPUBLICATION",
            "authority_sha256": "b" * 64,
            "complete": True,
        },
        "FINAL_EXACT_READBACK": {
            **common,
            "exact": True,
            "retained_stack_id": retained_stack_id,
            "fence_stack_id": fence_stack_id,
            "support_stack_id": support_stack_id,
            "retained_stack_status": "UPDATE_COMPLETE",
            "fence_stack_status": "UPDATE_COMPLETE",
            "support_stack_status": "UPDATE_COMPLETE",
            "pending_change_sets": 0,
            "retained_template_sha256": by_kind[
                "RETAINED_TEMPLATE"
            ]["body_sha256"],
            "fence_template_sha256": by_kind[
                "FENCE_TEMPLATE"
            ]["body_sha256"],
            "support_template_sha256": by_kind[
                "SUPPORT_TEMPLATE"
            ]["body_sha256"],
        },
        "PROVE_NO_WORKER_ACTIVATION": {
            **common,
            "worker_count": 0,
            "gpu_instance_ids": [],
            "evidence_window_start": "2026-07-29T20:00:00Z",
            "evidence_window_end": "2026-07-29T20:05:00Z",
            "gpu_inventory_sha256": "a" * 64,
            "cloudtrail_launch_events_sha256": "b" * 64,
            "complete": True,
        },
    }
    _set_coordinate_fields(
        committed["UPDATE_RETAINED_PRE_SUPPORT"],
        prefix="template",
        coordinate=by_kind["RETAINED_PRE_SUPPORT_TEMPLATE"],
    )
    _set_coordinate_fields(
        committed["UPDATE_RETAINED_FINAL"],
        prefix="template",
        coordinate=by_kind["RETAINED_TEMPLATE"],
    )
    _set_coordinate_fields(
        committed["BUILD_PUBLISH_SUPPORT"],
        prefix="support_template",
        coordinate=by_kind["SUPPORT_TEMPLATE"],
    )
    _set_coordinate_fields(
        committed["BUILD_PUBLISH_SUPPORT"],
        prefix="support_inputs",
        coordinate=by_kind["SUPPORT_INPUTS"],
    )
    records, journal_raw = _rechain(
        committed,
        request_identity_sha256=request_identity,
    )
    migration_execution = committed["EXECUTE_STACK_MIGRATION"]
    pre = committed["UPDATE_RETAINED_PRE_SUPPORT"]
    final = committed["UPDATE_RETAINED_FINAL"]
    readback = committed["FINAL_EXACT_READBACK"]
    no_workers = committed["PROVE_NO_WORKER_ACTIVATION"]
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": (
            "glm52_task13_staged_infrastructure_evidence_v1"
        ),
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "activation_id": activation_id,
        "staged_request": staged_request,
        "staged_request_identity_sha256": request_identity,
        "staged_journal_path": str(template_path),
        "staged_journal_size_bytes": len(journal_raw),
        "staged_journal_sha256": hashlib.sha256(
            journal_raw
        ).hexdigest(),
        "staged_journal_records": records,
        "completed_steps": list(STEPS),
        "retained_foundation": foundation,
        "fence": {
            "stack_id": fence_stack_id,
            "stack_status": "UPDATE_COMPLETE",
            "change_set_type": "IMPORT",
            "policy_logical_id": "H1gProductionFenceBucketPolicy",
            "template_coordinate": by_kind["FENCE_TEMPLATE"],
            "template_body_sha256": by_kind["FENCE_TEMPLATE"][
                "body_sha256"
            ],
            "readback_sha256": _sha(migration_execution),
        },
        "pre_support": {
            "stack_id": pre["stack_id"],
            "stack_status": pre["stack_status"],
            "change_set_type": pre["change_set_type"],
            "role_arn": pre["role_arn"],
            "template_coordinate": by_kind[
                "RETAINED_PRE_SUPPORT_TEMPLATE"
            ],
            "retained_fragment_sha256": pre[
                "source_fragment_sha256"
            ],
            "readback_sha256": pre["readback_sha256"],
        },
        "support_stack": {
            "stack_id": support_stack_id,
            "stack_status": "UPDATE_COMPLETE",
            "change_set_type": "UPDATE",
            "worker_activation_enabled": False,
            "template_coordinate": by_kind["SUPPORT_TEMPLATE"],
            "inputs_coordinate": by_kind["SUPPORT_INPUTS"],
            "support_lambda_archive": support_lambda,
            "cryptography_layer_archive": cryptography_layer,
            "cryptography_layer_version_arn": committed[
                "MATERIALIZE_FULL_SUPPORT_INPUTS"
            ]["cryptography_layer_version_arn"],
            "cryptography_layer_code_sha256": committed[
                "MATERIALIZE_FULL_SUPPORT_INPUTS"
            ]["cryptography_layer_code_sha256"],
            "readback_sha256": _sha(migration_execution),
        },
        "postcreate_final": {
            "retained_stack_id": retained_stack_id,
            "support_stack_id": support_stack_id,
            "stack_status": final["stack_status"],
            "change_set_type": final["change_set_type"],
            "role_arn": final["role_arn"],
            "template_coordinate": by_kind["RETAINED_TEMPLATE"],
            "retained_fragment_sha256": final[
                "source_fragment_sha256"
            ],
            "postpublication_authority_sha256": committed[
                "POSTPUBLICATION_AUTHORITY"
            ]["authority_sha256"],
            "readback_sha256": final["readback_sha256"],
        },
        "fixed_artifacts": fixed,
        "fixed_artifacts_identity_sha256": _sha(fixed),
        "final_readback": {
            "retained_stack_id": retained_stack_id,
            "fence_stack_id": fence_stack_id,
            "support_stack_id": support_stack_id,
            "retained_stack_status": readback[
                "retained_stack_status"
            ],
            "fence_stack_status": readback["fence_stack_status"],
            "support_stack_status": readback[
                "support_stack_status"
            ],
            "retained_template_sha256": readback[
                "retained_template_sha256"
            ],
            "fence_template_sha256": readback[
                "fence_template_sha256"
            ],
            "support_template_sha256": readback[
                "support_template_sha256"
            ],
            "pending_change_sets": 0,
            "active_p5_instance_ids": no_workers["gpu_instance_ids"],
            "worker_activation_attempts": 0,
            "raw_ec2_launch_calls": 0,
        },
    }
    return {
        **body,
        "canonical_identity_sha256": _sha(body),
    }
