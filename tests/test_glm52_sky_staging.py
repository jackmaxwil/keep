from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mlx_vq.quality.glm52_s3_artifact_audit import (
    build_s3_artifact_inventory,
)
from mlx_vq.quality.glm52_sky_campaign import (
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
)
from mlx_vq.quality.glm52_sky_must_start import (
    build_must_start_controller_observation,
    build_must_start_job_binding,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "aws/glm52-gpu/scripts"
ACTIVATION_HELPER = SCRIPTS / "validate_must_start_activation.py"
RUN_ID = "glm52-sky-20260724"
MODE = "cache-seed"
JOB_NAME = f"{RUN_ID}-cache-seed"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
DESCRIPTOR_KEY = (
    f"campaigns/{RUN_ID}/submissions/seed/campaign-descriptor-v2.json"
)
DESCRIPTOR_FILE_SHA = "a" * 64
DESCRIPTOR_BODY_SHA = "b" * 64
SUBMISSION_BODY_SHA = "c" * 64
SUBMISSION_KEY = (
    f"campaigns/{RUN_ID}/monitor/submission-locks/"
    f"{DESCRIPTOR_FILE_SHA}-{MODE}.json"
)
SUBMITTED_AT = "2026-07-26T03:51:05.495390Z"
DEADLINE = "2026-07-26T14:27:42Z"
CONTROLLER_ID = "i-0511af4e31aa5406a"
CONTROLLER_TYPE = "c6a.xlarge"
CONTROLLER_PROFILE = (
    "arn:aws:iam::246813579024:instance-profile/"
    "keep-glm52-skypilot-controller"
)
CONTROLLER_CLUSTER = "sky-jobs-controller-9d9f31a9-9d9f31a9"
CODE_VERSION_ID = "3/L4kqtJlcpXroDTDmJ+rmSpXd3dIbrHY+MTRCxf3vjVBH40="
STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:"
    "stack/keep-glm52-gpu/11111111-2222-3333-4444-555555555555"
)
ACTIVE_CHANGE_SET_ARN = (
    "arn:aws:cloudformation:us-west-2:246813579024:"
    "changeSet/awscli-cloudformation-package-deploy-1785000001/"
    "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
)
FOUNDATION_CHANGE_SET_ARN = (
    "arn:aws:cloudformation:us-west-2:246813579024:"
    "changeSet/awscli-cloudformation-package-deploy-1785000002/"
    "11111111-aaaa-4bbb-8ccc-222222222222"
)
TEMPLATE_UPLOAD_KEY = "6af6b08ca0136c7f9d1a12f317f4f35b.template"
WORKER_V2_DISABLED_PARAMETERS = {
    "EnableSkyWorkerStartV2Coordinator": "false",
    "SkyWorkerStartV2FoundationRevision": "versioned-code-v1",
    "SkyWorkerStartV2CodeSha256": "disabled",
    "SkyWorkerStartV2CodeVersionId": "disabled",
    "SkyWorkerStartV2DescriptorRelativeKey": "disabled",
    "SkyWorkerStartV2DescriptorFileSha256": "disabled",
    "SkyWorkerStartV2IntentFileSha256": "disabled",
    "SkyWorkerStartV2IntentBodySha256": "disabled",
    "SkyWorkerStartV2ControllerInstanceId": "disabled",
    "SkyWorkerStartV2ControllerClusterName": "disabled",
}
WORKER_V2_CURRENT_ENV = {
    "EnableSkyWorkerStartV2Coordinator": (
        "FAKE_CURRENT_WORKER_START_V2_ENABLE"
    ),
    "SkyWorkerStartV2FoundationRevision": (
        "FAKE_CURRENT_WORKER_START_V2_FOUNDATION_REVISION"
    ),
    "SkyWorkerStartV2CodeSha256": (
        "FAKE_CURRENT_WORKER_START_V2_CODE_SHA256"
    ),
    "SkyWorkerStartV2CodeVersionId": (
        "FAKE_CURRENT_WORKER_START_V2_CODE_VERSION_ID"
    ),
    "SkyWorkerStartV2DescriptorRelativeKey": (
        "FAKE_CURRENT_WORKER_START_V2_DESCRIPTOR_RELATIVE_KEY"
    ),
    "SkyWorkerStartV2DescriptorFileSha256": (
        "FAKE_CURRENT_WORKER_START_V2_DESCRIPTOR_FILE_SHA256"
    ),
    "SkyWorkerStartV2IntentFileSha256": (
        "FAKE_CURRENT_WORKER_START_V2_INTENT_FILE_SHA256"
    ),
    "SkyWorkerStartV2IntentBodySha256": (
        "FAKE_CURRENT_WORKER_START_V2_INTENT_BODY_SHA256"
    ),
    "SkyWorkerStartV2ControllerInstanceId": (
        "FAKE_CURRENT_WORKER_START_V2_CONTROLLER_INSTANCE_ID"
    ),
    "SkyWorkerStartV2ControllerClusterName": (
        "FAKE_CURRENT_WORKER_START_V2_CONTROLLER_CLUSTER_NAME"
    ),
}
WORKER_V2_CHANGE_SET_ENV = {
    parameter: environment.replace("FAKE_CURRENT_", "FAKE_CHANGE_SET_")
    for parameter, environment in WORKER_V2_CURRENT_ENV.items()
}
WORKER_V2_DISABLED_OVERRIDES = tuple(
    f"{parameter}={value}"
    for parameter, value in WORKER_V2_DISABLED_PARAMETERS.items()
)


def _change_detail(
    *,
    evaluation: str,
    source: str,
    target: str,
    recreation: str = "Never",
    causing_entity: str | None = None,
) -> dict[str, object]:
    detail: dict[str, object] = {
        "ChangeSource": source,
        "Evaluation": evaluation,
        "Target": {
            "Attribute": "Properties",
            "Name": target,
            "RequiresRecreation": recreation,
        },
    }
    if causing_entity is not None:
        detail["CausingEntity"] = causing_entity
    return detail


def _foundation_resource_change(
    logical_id: str,
    resource_type: str,
    details: list[dict[str, object]],
    *,
    replacement: str = "False",
) -> dict[str, object]:
    return {
        "Type": "Resource",
        "ResourceChange": {
            "Action": "Modify",
            "Details": details,
            "LogicalResourceId": logical_id,
            "PhysicalResourceId": f"fixture-{logical_id}",
            "Replacement": replacement,
            "ResourceType": resource_type,
            "Scope": ["Properties"],
        },
    }


def _foundation_change_graph() -> list[dict[str, object]]:
    role_details = [
        _change_detail(
            evaluation="Dynamic",
            source="ResourceAttribute",
            target="Policies",
            causing_entity="ModelBucket.Arn",
        ),
        _change_detail(
            evaluation="Static",
            source="ParameterReference",
            target="Policies",
            causing_entity="SkyMustStartSubmissionBodySha256",
        ),
        _change_detail(
            evaluation="Static",
            source="ParameterReference",
            target="Policies",
            causing_entity="SkyMustStartManagedMode",
        ),
        _change_detail(
            evaluation="Dynamic",
            source="DirectModification",
            target="Policies",
        ),
    ]
    return [
        _foundation_resource_change(
            "InstanceRole",
            "AWS::IAM::Role",
            role_details,
        ),
        _foundation_resource_change(
            "ModelBucket",
            "AWS::S3::Bucket",
            [
                _change_detail(
                    evaluation="Static",
                    source="DirectModification",
                    target="VersioningConfiguration",
                )
            ],
        ),
        _foundation_resource_change(
            "S3BatchChecksumRole",
            "AWS::IAM::Role",
            [
                _change_detail(
                    evaluation="Dynamic",
                    source="ResourceAttribute",
                    target="Policies",
                    causing_entity="ModelBucket.Arn",
                )
            ],
        ),
        _foundation_resource_change(
            "SkyPilotControllerRole",
            "AWS::IAM::Role",
            [
                role_details[0],
                _change_detail(
                    evaluation="Dynamic",
                    source="ResourceAttribute",
                    target="Policies",
                    causing_entity="SkyPilotWorkerRole.Arn",
                ),
                *role_details[1:],
            ],
        ),
        _foundation_resource_change(
            "SkyPilotWorkerRole",
            "AWS::IAM::Role",
            role_details,
        ),
        _foundation_resource_change(
            "SkyWatchdogFunction",
            "AWS::Lambda::Function",
            [
                _change_detail(
                    evaluation="Dynamic",
                    source="ResourceAttribute",
                    target="Role",
                    causing_entity="SkyWatchdogRole.Arn",
                )
            ],
        ),
        _foundation_resource_change(
            "SkyWatchdogInvokePermission",
            "AWS::Lambda::Permission",
            [
                _change_detail(
                    evaluation="Dynamic",
                    source="ResourceAttribute",
                    target="SourceArn",
                    recreation="Always",
                    causing_entity="SkyWatchdogRule.Arn",
                )
            ],
            replacement="Conditional",
        ),
        _foundation_resource_change(
            "SkyWatchdogRole",
            "AWS::IAM::Role",
            role_details,
        ),
        _foundation_resource_change(
            "SkyWatchdogRule",
            "AWS::Events::Rule",
            [
                _change_detail(
                    evaluation="Dynamic",
                    source="ResourceAttribute",
                    target="Targets",
                    causing_entity="SkyWatchdogFunction.Arn",
                )
            ],
        ),
    ]


def _activation_records() -> tuple[dict[str, object], dict[str, object]]:
    observation = build_must_start_controller_observation(
        run_id=RUN_ID,
        managed_mode=MODE,
        account_id="246813579024",
        region="us-west-2",
        bucket=BUCKET,
        descriptor_body_sha256=DESCRIPTOR_BODY_SHA,
        submission_body_sha256=SUBMISSION_BODY_SHA,
        sky_job_name=JOB_NAME,
        must_start_by=DEADLINE,
        target_job_id=3,
        workspace="default",
        controller_instance_id=CONTROLLER_ID,
        controller_instance_type=CONTROLLER_TYPE,
        controller_profile_arn=CONTROLLER_PROFILE,
        controller_cluster_name=CONTROLLER_CLUSTER,
        status="PENDING",
        schedule_state="LAUNCHING",
        submitted_at=SUBMITTED_AT,
        start_at=None,
        worker_cluster_name=None,
        recovery_count=0,
        observed_at="2026-07-26T07:20:00Z",
    )
    binding = build_must_start_job_binding(
        run_id=RUN_ID,
        managed_mode=MODE,
        account_id="246813579024",
        region="us-west-2",
        bucket=BUCKET,
        descriptor_body_sha256=DESCRIPTOR_BODY_SHA,
        submission_body_sha256=SUBMISSION_BODY_SHA,
        sky_job_name=JOB_NAME,
        must_start_by=DEADLINE,
        descriptor_key=DESCRIPTOR_KEY,
        descriptor_file_sha256=DESCRIPTOR_FILE_SHA,
        submission_key=SUBMISSION_KEY,
        submission_submitted_at=SUBMITTED_AT,
        target_job_id=3,
        workspace="default",
        controller_instance_id=CONTROLLER_ID,
        controller_instance_type=CONTROLLER_TYPE,
        controller_profile_arn=CONTROLLER_PROFILE,
        controller_cluster_name=CONTROLLER_CLUSTER,
        observation_body_sha256=str(observation["observation_body_sha256"]),
        bound_at=str(observation["observed_at"]),
    )
    return binding, observation


def _deploy_change_set_output(change_set_id: str) -> str:
    return (
        "Changeset created successfully. Run the following command to "
        "review changes:\n"
        "aws cloudformation describe-change-set "
        f"--change-set-name {change_set_id}\n"
    )


def _activation_args(
    action: str,
    binding_path: Path,
    observation_path: Path | None = None,
    *,
    descriptor_key: str = DESCRIPTOR_KEY,
) -> list[str]:
    args = [
        "python3",
        str(ACTIVATION_HELPER),
        action,
        "--job-binding",
        str(binding_path),
        "--account-id",
        "246813579024",
        "--region",
        "us-west-2",
        "--bucket",
        BUCKET,
        "--run-id",
        RUN_ID,
        "--managed-mode",
        MODE,
        "--descriptor-key",
        descriptor_key,
        "--descriptor-file-sha256",
        DESCRIPTOR_FILE_SHA,
        "--descriptor-body-sha256",
        DESCRIPTOR_BODY_SHA,
        "--submission-key",
        SUBMISSION_KEY,
        "--submission-body-sha256",
        SUBMISSION_BODY_SHA,
        "--submission-submitted-at",
        SUBMITTED_AT,
        "--target-job-id",
        "3",
        "--workspace",
        "default",
        "--job-name",
        JOB_NAME,
        "--must-start-by",
        DEADLINE,
        "--controller-instance-id",
        CONTROLLER_ID,
        "--controller-instance-type",
        CONTROLLER_TYPE,
        "--controller-profile-arn",
        CONTROLLER_PROFILE,
        "--controller-cluster-name",
        CONTROLLER_CLUSTER,
    ]
    if observation_path is not None:
        args.extend(["--observation", str(observation_path)])
    return args


def _write_fake_aws(
    tmp_path: Path,
    *,
    binding_path: Path,
    observation_path: Path,
    remote_code_path: Path,
    code_key: str,
) -> tuple[Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "aws-calls.jsonl"
    fake = fake_bin / "aws"
    fake.write_text(
        """#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

args = sys.argv[1:]
with Path(os.environ["FAKE_AWS_LOG"]).open("a") as handle:
    handle.write(json.dumps(args) + "\\n")

if args[:2] == ["sts", "get-caller-identity"]:
    print(
        json.dumps(
            {
                "Account": "246813579024",
                "Arn": "arn:aws:iam::246813579024:user/fake",
                "UserId": "fake",
            }
        )
    )
elif args[:2] == ["s3api", "get-bucket-versioning"]:
    print(os.environ["FAKE_BUCKET_VERSIONING_STATUS"])
elif args[:2] == ["s3api", "get-object"]:
    key = args[args.index("--key") + 1]
    destination = Path(args[-1])
    if key == os.environ["FAKE_TEMPLATE_UPLOAD_KEY"]:
        assert (
            args[args.index("--bucket") + 1]
            == os.environ["FAKE_TEMPLATE_UPLOAD_BUCKET"]
        )
        source = Path(os.environ["FAKE_REMOTE_TEMPLATE"])
    elif key.endswith("/JOB_BINDING.json"):
        source = Path(os.environ["FAKE_JOB_BINDING"])
    elif "/observations/" in key:
        source = Path(os.environ["FAKE_OBSERVATION"])
    elif key == os.environ["FAKE_CODE_KEY"]:
        assert args[args.index("--version-id") + 1] == os.environ["FAKE_CODE_VERSION"]
        source = Path(os.environ["FAKE_REMOTE_CODE"])
    else:
        raise SystemExit(f"unexpected fake S3 key: {key}")
    shutil.copyfile(source, destination)
    if key == os.environ["FAKE_TEMPLATE_UPLOAD_KEY"]:
        source_bytes = source.read_bytes()
        response = {
            "ContentLength": len(source_bytes),
            "ETag": (
                '"'
                + hashlib.md5(
                    source_bytes,
                    usedforsecurity=False,
                ).hexdigest()
                + '"'
            ),
        }
    else:
        response = {"ETag": "fake"}
    if key == os.environ["FAKE_CODE_KEY"]:
        response["VersionId"] = os.environ["FAKE_CODE_VERSION"]
    print(json.dumps(response))
elif args[:2] == ["s3api", "head-object"]:
    key = args[args.index("--key") + 1]
    if key == os.environ["FAKE_TEMPLATE_UPLOAD_KEY"]:
        assert (
            args[args.index("--bucket") + 1]
            == os.environ["FAKE_TEMPLATE_UPLOAD_BUCKET"]
        )
        source_bytes = Path(os.environ["FAKE_REMOTE_TEMPLATE"]).read_bytes()
        print(
            json.dumps(
                {
                    "ContentLength": len(source_bytes),
                    "ETag": (
                        '"'
                        + hashlib.md5(
                            source_bytes,
                            usedforsecurity=False,
                        ).hexdigest()
                        + '"'
                    ),
                }
            )
        )
    else:
        assert key == os.environ["FAKE_CODE_KEY"]
        assert args[args.index("--version-id") + 1] == os.environ["FAKE_CODE_VERSION"]
        if "--output" in args and args[args.index("--output") + 1] == "text":
            print(os.environ["FAKE_CODE_VERSION"])
        else:
            print(
                json.dumps(
                    {
                        "ETag": "fake",
                        "VersionId": os.environ["FAKE_CODE_VERSION"],
                        "ContentLength": Path(os.environ["FAKE_REMOTE_CODE"]).stat().st_size,
                    }
                )
            )
elif args[:2] == ["cloudformation", "deploy"]:
    Path(os.environ["FAKE_STACK_CREATED_MARKER"]).touch()
    if "--no-execute-changeset" in args:
        assert "--fail-on-empty-changeset" in args
        assert "--no-fail-on-empty-changeset" not in args
        assert "--changeset-name" not in args
        assert "--change-set-name" not in args
        if "FAKE_DEPLOY_OUTPUT" in os.environ:
            output = os.environ["FAKE_DEPLOY_OUTPUT"]
            sys.stdout.write(output)
            if output and not output.endswith("\\n"):
                sys.stdout.write("\\n")
        else:
            is_active = any(
                value.startswith("SkyMustStartCancelCodeSha256=")
                for value in args
            )
            change_set_arn = (
                os.environ["FAKE_ACTIVE_CHANGE_SET_ARN"]
                if is_active
                else os.environ["FAKE_FOUNDATION_CHANGE_SET_ARN"]
            )
            print(
                "Changeset created successfully. Run the following command "
                "to review changes:"
            )
            print(
                "aws cloudformation describe-change-set "
                f"--change-set-name {change_set_arn}"
            )
    else:
        assert "--no-fail-on-empty-changeset" in args
        print("fake stack deployed")
elif args[:2] == ["cloudformation", "describe-stacks"]:
    stack_exists = (
        os.environ["FAKE_STACK_EXISTS"] == "true"
        or Path(os.environ["FAKE_STACK_CREATED_MARKER"]).exists()
    )
    if not stack_exists:
        print(
            "An error occurred (ValidationError) when calling the "
            "DescribeStacks operation: Stack with id keep-glm52-gpu "
            "does not exist",
            file=sys.stderr,
        )
        raise SystemExit(255)
    query = args[args.index("--query") + 1]
    output = args[args.index("--output") + 1]
    if query == "Stacks[0].StackId" and output == "text":
        print(
            os.environ.get(
                "FAKE_CURRENT_STACK_ID_AFTER_DEPLOY",
                os.environ["FAKE_CURRENT_STACK_ID"],
            )
            if Path(os.environ["FAKE_STACK_CREATED_MARKER"]).exists()
            else os.environ["FAKE_CURRENT_STACK_ID"]
        )
    elif query == "Stacks[0].[StackId,StackStatus]" and output == "text":
        after_deploy = Path(
            os.environ["FAKE_STACK_CREATED_MARKER"]
        ).exists()
        print(
            (
                os.environ.get(
                    "FAKE_CURRENT_STACK_ID_AFTER_DEPLOY",
                    os.environ["FAKE_CURRENT_STACK_ID"],
                )
                if after_deploy
                else os.environ["FAKE_CURRENT_STACK_ID"]
            )
            + "\\t"
            + (
                os.environ.get(
                    "FAKE_CURRENT_STACK_STATUS_AFTER_DEPLOY",
                    os.environ["FAKE_CURRENT_STACK_STATUS"],
                )
                if after_deploy
                else os.environ["FAKE_CURRENT_STACK_STATUS"]
            )
        )
    elif "EnableSkyPilotSupport" in query and output == "text":
        values = [
            os.environ["FAKE_CURRENT_SUPPORT"],
            os.environ["FAKE_CURRENT_OBSERVE"],
            os.environ["FAKE_CURRENT_CANCEL"],
            os.environ["FAKE_CURRENT_RUN_ID"],
            os.environ["FAKE_CURRENT_DESCRIPTOR_KEY"],
            os.environ["FAKE_CURRENT_WATCHDOG_CODE_S3_KEY"],
        ]
        if "SkyMustStartFoundationRevision" in query:
            values.append(os.environ["FAKE_FOUNDATION_REVISION"])
        if "EnableSkyWorkerStartV2Coordinator" in query:
            values.extend(
                [
                    os.environ["FAKE_CURRENT_WORKER_START_V2_ENABLE"],
                    os.environ[
                        "FAKE_CURRENT_WORKER_START_V2_FOUNDATION_REVISION"
                    ],
                    os.environ["FAKE_CURRENT_WORKER_START_V2_CODE_SHA256"],
                    os.environ[
                        "FAKE_CURRENT_WORKER_START_V2_CODE_VERSION_ID"
                    ],
                    os.environ[
                        "FAKE_CURRENT_WORKER_START_V2_DESCRIPTOR_RELATIVE_KEY"
                    ],
                    os.environ[
                        "FAKE_CURRENT_WORKER_START_V2_DESCRIPTOR_FILE_SHA256"
                    ],
                    os.environ[
                        "FAKE_CURRENT_WORKER_START_V2_INTENT_FILE_SHA256"
                    ],
                    os.environ[
                        "FAKE_CURRENT_WORKER_START_V2_INTENT_BODY_SHA256"
                    ],
                    os.environ[
                        "FAKE_CURRENT_WORKER_START_V2_CONTROLLER_INSTANCE_ID"
                    ],
                    os.environ[
                        "FAKE_CURRENT_WORKER_START_V2_CONTROLLER_CLUSTER_NAME"
                    ],
                ]
            )
        print("\\t".join(values))
    elif "CampaignAlertTopicArn" in query and output == "text":
        print("arn:aws:sns:us-west-2:246813579024:keep-glm52-alerts")
    elif "StackStatus" in query and output == "json":
        print(json.dumps(["UPDATE_COMPLETE", []]))
    else:
        raise SystemExit(f"unexpected fake describe-stacks call: {args}")
elif args[:2] == ["cloudformation", "list-stack-resources"]:
    query = args[args.index("--query") + 1]
    if "SkyWorkerStartV2" in query:
        print(os.environ["FAKE_WORKER_START_V2_RESOURCES"])
    else:
        print(os.environ["FAKE_MUST_START_RESOURCES"])
elif args[:2] == ["cloudformation", "describe-change-set"]:
    requested_change_set = args[args.index("--change-set-name") + 1]
    if "--output" in args and args[args.index("--output") + 1] == "json":
        query = args[args.index("--query") + 1]
        if (
            "ChangeSetId" in query
            and "StackId" in query
            and "Status" in query
            and "ExecutionStatus" in query
        ):
            assert "ChangeSetType" not in query
            identity = {
                "ChangeSetId": os.environ.get(
                    "FAKE_RETURNED_CHANGE_SET_ID",
                    requested_change_set,
                ),
                "StackId": os.environ.get(
                    "FAKE_RETURNED_STACK_ID",
                    os.environ["FAKE_CURRENT_STACK_ID"],
                ),
                "StackName": os.environ.get(
                    "FAKE_RETURNED_STACK_NAME",
                    "keep-glm52-gpu",
                ),
                "Status": os.environ.get(
                    "FAKE_RETURNED_CHANGE_SET_STATUS",
                    "CREATE_COMPLETE",
                ),
                "ExecutionStatus": os.environ.get(
                    "FAKE_RETURNED_EXECUTION_STATUS",
                    "AVAILABLE",
                ),
            }
            omitted = os.environ.get("FAKE_IDENTITY_OMIT_FIELD")
            if omitted is not None:
                identity.pop(omitted)
            nulled = os.environ.get("FAKE_IDENTITY_NULL_FIELD")
            if nulled is not None:
                identity[nulled] = None
            print(json.dumps(identity))
        elif query == "Changes":
            print(Path(os.environ["FAKE_FOUNDATION_CHANGE_GRAPH"]).read_text())
        else:
            raise SystemExit(f"unexpected fake JSON change-set query: {query}")
    elif "--output" in args and args[args.index("--output") + 1] == "text":
        query = args[args.index("--query") + 1]
        if "ResourceChange.LogicalResourceId" in query:
            graph = json.loads(
                Path(os.environ["FAKE_FOUNDATION_CHANGE_GRAPH"]).read_text()
            )
            print(
                "\\t".join(
                    change["ResourceChange"]["LogicalResourceId"]
                    for change in graph
                )
            )
        elif "SkyMustStartActivationJobBindingSha256" in query:
            values = [
                "CREATE_COMPLETE",
                "AVAILABLE",
                os.environ["FAKE_CONTROLLER_BASELINE_SHA"],
                os.environ["FAKE_BINDING_SHA"],
                os.environ["FAKE_OBSERVATION_SHA"],
                os.environ["FAKE_CODE_SHA"],
                os.environ["FAKE_CODE_VERSION"],
            ]
            if "EnableSkyWorkerStartV2Coordinator" in query:
                values.extend(
                    [
                        os.environ["FAKE_CHANGE_SET_WORKER_START_V2_ENABLE"],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_FOUNDATION_REVISION"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_CODE_SHA256"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_CODE_VERSION_ID"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_DESCRIPTOR_RELATIVE_KEY"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_DESCRIPTOR_FILE_SHA256"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_INTENT_FILE_SHA256"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_INTENT_BODY_SHA256"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_CONTROLLER_INSTANCE_ID"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_CONTROLLER_CLUSTER_NAME"
                        ],
                    ]
                )
            print("\\t".join(values))
        elif "EnableSkyPilotSupport" in query:
            values = [
                "CREATE_COMPLETE",
                "AVAILABLE",
                "true",
                "false",
                "false",
                os.environ["RUN_ID"],
                os.environ["DESCRIPTOR_KEY"],
                os.environ["WATCHDOG_CODE_S3_KEY"],
                "versioned-code-v1",
            ]
            if "EnableSkyWorkerStartV2Coordinator" in query:
                values.extend(
                    [
                        os.environ["FAKE_CHANGE_SET_WORKER_START_V2_ENABLE"],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_FOUNDATION_REVISION"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_CODE_SHA256"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_CODE_VERSION_ID"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_DESCRIPTOR_RELATIVE_KEY"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_DESCRIPTOR_FILE_SHA256"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_INTENT_FILE_SHA256"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_INTENT_BODY_SHA256"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_CONTROLLER_INSTANCE_ID"
                        ],
                        os.environ[
                            "FAKE_CHANGE_SET_WORKER_START_V2_CONTROLLER_CLUSTER_NAME"
                        ],
                    ]
                )
            print("\\t".join(values))
        else:
            raise SystemExit(f"unexpected fake change-set query: {query}")
    else:
        print(json.dumps({"Status": "CREATE_COMPLETE", "ExecutionStatus": "AVAILABLE"}))
elif args[:2] == ["cloudformation", "get-template"]:
    assert args[args.index("--template-stage") + 1] == "Original"
    assert args[args.index("--output") + 1] == "json"
    template_body = (
        os.environ["FAKE_CHANGE_SET_TEMPLATE_BODY"]
        if "FAKE_CHANGE_SET_TEMPLATE_BODY" in os.environ
        else Path(os.environ["FAKE_TEMPLATE_PATH"]).read_text().replace("—", "?")
    )
    print(
        json.dumps(
            {
                "TemplateBody": template_body,
                "StagesAvailable": ["Original", "Processed"],
            }
        )
    )
elif args[:2] == ["cloudformation", "execute-change-set"]:
    print(json.dumps({"executed": True}))
elif args[:2] == ["sns", "list-subscriptions-by-topic"]:
    topic = args[args.index("--topic-arn") + 1]
    print(
        json.dumps(
            {
                "Subscriptions": [
                    {
                        "TopicArn": topic,
                        "Protocol": "email",
                        "Endpoint": "operator@example.com",
                        "SubscriptionArn": (
                            "arn:aws:sns:us-west-2:246813579024:"
                            "keep-glm52-alerts:subscription"
                        ),
                    }
                ]
            }
        )
    )
else:
    raise SystemExit(f"unexpected fake AWS call: {args}")
"""
    )
    fake.chmod(0o755)
    return fake_bin, log_path


def _active_environment(
    tmp_path: Path,
    *,
    remote_code: bytes | None = None,
) -> tuple[dict[str, str], Path]:
    binding, observation = _activation_records()
    binding_path = tmp_path / "JOB_BINDING.json"
    observation_path = tmp_path / "observation.json"
    code_path = tmp_path / "sky-must-start.zip"
    remote_code_path = tmp_path / "remote-sky-must-start.zip"
    remote_template_path = tmp_path / "remote-template-upload.template"
    foundation_graph_path = tmp_path / "foundation-change-graph.json"
    binding_path.write_text(json.dumps(binding))
    observation_path.write_text(json.dumps(observation))
    code_path.write_bytes(b"exact package bytes")
    remote_code_path.write_bytes(
        code_path.read_bytes() if remote_code is None else remote_code
    )
    remote_template_path.write_bytes(
        (ROOT / "aws/glm52-gpu/cfn/gpu-teacher-stack.yaml").read_bytes()
    )
    foundation_graph_path.write_text(json.dumps(_foundation_change_graph()))
    code_sha = hashlib.sha256(code_path.read_bytes()).hexdigest()
    code_key = f"lambda/sky-must-start-cancel/{code_sha}.zip"
    fake_bin, log_path = _write_fake_aws(
        tmp_path,
        binding_path=binding_path,
        observation_path=observation_path,
        remote_code_path=remote_code_path,
        code_key=code_key,
    )
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "AWS_PROFILE": "fake-rnd",
        "RUN_ID": RUN_ID,
        "DESCRIPTOR_KEY": DESCRIPTOR_KEY,
        "WATCHDOG_CODE_S3_KEY": "lambda/sky-watchdog/exact.zip",
        "ENABLE_MUST_START_OBSERVE": "true",
        "MUST_START_CANCEL_CODE_ZIP": str(code_path),
        "MUST_START_CANCEL_CODE_SHA256": code_sha,
        "MUST_START_CANCEL_CODE_VERSION_ID": CODE_VERSION_ID,
        "MUST_START_SUBMISSION_KEY": SUBMISSION_KEY,
        "MUST_START_SUBMISSION_BODY_SHA256": SUBMISSION_BODY_SHA,
        "MUST_START_CONTROLLER_BASELINE_BODY_SHA256": "2" * 64,
        "MUST_START_TARGET_JOB_ID": "3",
        "MUST_START_MANAGED_MODE": MODE,
        "MUST_START_JOB_NAME": JOB_NAME,
        "MUST_START_BY": DEADLINE,
        "MUST_START_DESCRIPTOR_FILE_SHA256": DESCRIPTOR_FILE_SHA,
        "MUST_START_DESCRIPTOR_BODY_SHA256": DESCRIPTOR_BODY_SHA,
        "MUST_START_SUBMISSION_SUBMITTED_AT": SUBMITTED_AT,
        "MUST_START_CONTROLLER_INSTANCE_ID": CONTROLLER_ID,
        "MUST_START_CONTROLLER_INSTANCE_TYPE": CONTROLLER_TYPE,
        "MUST_START_CONTROLLER_PROFILE_ARN": CONTROLLER_PROFILE,
        "MUST_START_CONTROLLER_CLUSTER_NAME": CONTROLLER_CLUSTER,
        "MUST_START_CHANGE_SET_HANDOFF_FILE": str(
            tmp_path / "active-change-set.json"
        ),
        "MUST_START_FOUNDATION_CHANGE_SET_HANDOFF_FILE": str(
            tmp_path / "foundation-change-set.json"
        ),
        "MUST_START_EXECUTE_APPROVAL": (
            "execute-exact-reviewed-must-start-change-set"
        ),
        "MUST_START_FOUNDATION_EXECUTE_APPROVAL": (
            "execute-exact-reviewed-must-start-foundation-change-set"
        ),
        "FAKE_AWS_LOG": str(log_path),
        "FAKE_JOB_BINDING": str(binding_path),
        "FAKE_OBSERVATION": str(observation_path),
        "FAKE_REMOTE_CODE": str(remote_code_path),
        "FAKE_REMOTE_TEMPLATE": str(remote_template_path),
        "FAKE_CODE_KEY": code_key,
        "FAKE_TEMPLATE_UPLOAD_KEY": TEMPLATE_UPLOAD_KEY,
        "FAKE_TEMPLATE_UPLOAD_BUCKET": BUCKET,
        "FAKE_BINDING_SHA": str(binding["job_binding_body_sha256"]),
        "FAKE_CONTROLLER_BASELINE_SHA": "2" * 64,
        "FAKE_OBSERVATION_SHA": str(
            observation["observation_body_sha256"]
        ),
        "FAKE_CODE_SHA": code_sha,
        "FAKE_CODE_VERSION": CODE_VERSION_ID,
        "FAKE_CURRENT_SUPPORT": "true",
        "FAKE_CURRENT_RUN_ID": RUN_ID,
        "FAKE_CURRENT_DESCRIPTOR_KEY": DESCRIPTOR_KEY,
        "FAKE_CURRENT_WATCHDOG_CODE_S3_KEY": (
            "lambda/sky-watchdog/exact.zip"
        ),
        "FAKE_CURRENT_OBSERVE": "true",
        "FAKE_CURRENT_CANCEL": "false",
        "FAKE_FOUNDATION_REVISION": "versioned-code-v1",
        "FAKE_CURRENT_WORKER_START_V2_ENABLE": "false",
        "FAKE_CURRENT_WORKER_START_V2_FOUNDATION_REVISION": (
            "versioned-code-v1"
        ),
        "FAKE_CURRENT_WORKER_START_V2_CODE_SHA256": "disabled",
        "FAKE_CURRENT_WORKER_START_V2_CODE_VERSION_ID": "disabled",
        "FAKE_CURRENT_WORKER_START_V2_DESCRIPTOR_RELATIVE_KEY": "disabled",
        "FAKE_CURRENT_WORKER_START_V2_DESCRIPTOR_FILE_SHA256": "disabled",
        "FAKE_CURRENT_WORKER_START_V2_INTENT_FILE_SHA256": "disabled",
        "FAKE_CURRENT_WORKER_START_V2_INTENT_BODY_SHA256": "disabled",
        "FAKE_CURRENT_WORKER_START_V2_CONTROLLER_INSTANCE_ID": "disabled",
        "FAKE_CURRENT_WORKER_START_V2_CONTROLLER_CLUSTER_NAME": "disabled",
        "FAKE_CHANGE_SET_WORKER_START_V2_ENABLE": "false",
        "FAKE_CHANGE_SET_WORKER_START_V2_FOUNDATION_REVISION": (
            "versioned-code-v1"
        ),
        "FAKE_CHANGE_SET_WORKER_START_V2_CODE_SHA256": "disabled",
        "FAKE_CHANGE_SET_WORKER_START_V2_CODE_VERSION_ID": "disabled",
        "FAKE_CHANGE_SET_WORKER_START_V2_DESCRIPTOR_RELATIVE_KEY": (
            "disabled"
        ),
        "FAKE_CHANGE_SET_WORKER_START_V2_DESCRIPTOR_FILE_SHA256": (
            "disabled"
        ),
        "FAKE_CHANGE_SET_WORKER_START_V2_INTENT_FILE_SHA256": "disabled",
        "FAKE_CHANGE_SET_WORKER_START_V2_INTENT_BODY_SHA256": "disabled",
        "FAKE_CHANGE_SET_WORKER_START_V2_CONTROLLER_INSTANCE_ID": (
            "disabled"
        ),
        "FAKE_CHANGE_SET_WORKER_START_V2_CONTROLLER_CLUSTER_NAME": (
            "disabled"
        ),
        "FAKE_BUCKET_VERSIONING_STATUS": "Enabled",
        "FAKE_STACK_EXISTS": "true",
        "FAKE_STACK_CREATED_MARKER": str(tmp_path / "stack-created"),
        "FAKE_MUST_START_RESOURCES": "None",
        "FAKE_WORKER_START_V2_RESOURCES": "None",
        "FAKE_CURRENT_STACK_ID": STACK_ID,
        "FAKE_CURRENT_STACK_STATUS": "UPDATE_COMPLETE",
        "FAKE_ACTIVE_CHANGE_SET_ARN": ACTIVE_CHANGE_SET_ARN,
        "FAKE_FOUNDATION_CHANGE_SET_ARN": FOUNDATION_CHANGE_SET_ARN,
        "FAKE_FOUNDATION_CHANGE_GRAPH": str(foundation_graph_path),
        "FAKE_TEMPLATE_PATH": str(
            ROOT / "aws/glm52-gpu/cfn/gpu-teacher-stack.yaml"
        ),
    }
    return environment, log_path


def _foundation_environment(
    tmp_path: Path,
) -> tuple[dict[str, str], Path]:
    environment, log_path = _active_environment(tmp_path)
    environment["FAKE_CURRENT_OBSERVE"] = "false"
    environment["FAKE_CURRENT_CANCEL"] = "false"
    environment["FAKE_FOUNDATION_REVISION"] = "None"
    for environment_name in WORKER_V2_CURRENT_ENV.values():
        environment[environment_name] = "None"
    environment["FAKE_BUCKET_VERSIONING_STATUS"] = "None"
    return environment, log_path


def _set_post_core_state(environment: dict[str, str]) -> None:
    environment["FAKE_STACK_EXISTS"] = "true"
    environment["FAKE_CURRENT_SUPPORT"] = "false"
    environment["FAKE_CURRENT_OBSERVE"] = "false"
    environment["FAKE_CURRENT_CANCEL"] = "false"
    environment["FAKE_CURRENT_RUN_ID"] = "disabled"
    environment["FAKE_CURRENT_DESCRIPTOR_KEY"] = ""
    environment["FAKE_CURRENT_WATCHDOG_CODE_S3_KEY"] = (
        "disabled/not-deployed-sky-watchdog.zip"
    )
    environment["FAKE_FOUNDATION_REVISION"] = "versioned-code-v1"
    environment["FAKE_BUCKET_VERSIONING_STATUS"] = "Enabled"


def _aws_calls(log_path: Path) -> list[list[str]]:
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line.strip()
    ]


def _original_frozen_template_text() -> str:
    template_text = (
        ROOT / "aws/glm52-gpu/cfn/gpu-teacher-stack.yaml"
    ).read_text()
    assert hashlib.sha256(template_text.encode()).hexdigest() == (
        "b4c0bd8c725c3c4db951fccadd32707a0b2f8064c2519d264abfbc2f8d2924a4"
    )
    return template_text


def _copied_deploy_gate_fixture(
    tmp_path: Path,
    *,
    template_text: str,
) -> tuple[Path, Path, dict[str, str]]:
    copied_scripts = tmp_path / "repo/aws/glm52-gpu/scripts"
    copied_template = (
        tmp_path / "repo/aws/glm52-gpu/cfn/gpu-teacher-stack.yaml"
    )
    copied_scripts.mkdir(parents=True)
    copied_template.parent.mkdir(parents=True)
    copied_deploy = copied_scripts / "deploy_sky_control_plane.sh"
    copied_deploy.write_bytes(
        (SCRIPTS / "deploy_sky_control_plane.sh").read_bytes()
    )
    copied_deploy.chmod(0o755)
    copied_template.write_text(template_text)
    account_guard_called = tmp_path / "account-guard-called"
    copied_account_guard = copied_scripts / "assert_rnd_aws_account.sh"
    copied_account_guard.write_text(
        "#!/usr/bin/env bash\n"
        'touch "$TEMPLATE_GATE_ACCOUNT_GUARD_CALLED"\n'
        "exit 79\n"
    )
    copied_account_guard.chmod(0o755)
    environment = {
        **os.environ,
        "AWS_PROFILE": "fake-rnd",
        "TEMPLATE_GATE_ACCOUNT_GUARD_CALLED": str(account_guard_called),
    }
    return copied_deploy, account_guard_called, environment


def test_deploy_accepts_exact_frozen_utf8_template_before_account_guard(
    tmp_path: Path,
) -> None:
    copied_deploy, account_guard_called, environment = (
        _copied_deploy_gate_fixture(
            tmp_path,
            template_text=_original_frozen_template_text(),
        )
    )
    result = subprocess.run(
        [str(copied_deploy), "prepare-foundation"],
        cwd=tmp_path / "repo",
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 79
    assert account_guard_called.exists()


def test_deploy_rejects_unknown_non_ascii_before_account_or_aws(
    tmp_path: Path,
) -> None:
    copied_deploy, account_guard_called, environment = (
        _copied_deploy_gate_fixture(
            tmp_path,
            template_text=(
                _original_frozen_template_text()
                + "# unknown non-ASCII: é\n"
            ),
        )
    )

    result = subprocess.run(
        [str(copied_deploy), "prepare-foundation"],
        cwd=tmp_path / "repo",
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 65
    assert (
        "frozen CloudFormation template does not match exact UTF-8 upload"
        in result.stderr
    )
    assert not account_guard_called.exists()


def test_deploy_rejects_server_projection_and_hyphen_variant_before_aws(
    tmp_path: Path,
) -> None:
    original = _original_frozen_template_text()
    for index, separator in enumerate(("?", "-")):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        copied_deploy, account_guard_called, environment = (
            _copied_deploy_gate_fixture(
                case_path,
                template_text=original.replace("—", separator),
            )
        )
        result = subprocess.run(
            [str(copied_deploy), "prepare-foundation"],
            cwd=case_path / "repo",
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 65
        assert (
            "frozen CloudFormation template does not match exact UTF-8 upload"
            in result.stderr
        )
        assert not account_guard_called.exists()


def test_bundle_builder_is_local_secret_free_and_descriptor_last() -> None:
    source = (SCRIPTS / "build_sky_campaign_bundle.sh").read_text()
    assert "build_sky_repository_tar.sh" in source
    assert "build_gpu_spend_approval.py" in source
    assert "build_sky_training_config.py" in source
    assert "build_s3_artifact_inventory.py" in source
    assert "build_sky_campaign_descriptor.py" in source
    assert "package_sky_watchdog_lambda.sh" in source
    assert "REPO_TAR_SOURCE" in source
    assert "cp \"$REPO_TAR_SOURCE\" \"$REPO_TAR\"" in source
    assert 'inventories/artifact-inventory-$INVENTORY_SHA.json' in source
    assert "descriptor_body_sha256" in source
    assert "bundle_manifest_body_sha256" in source
    assert "aws s3" not in source

    tar_builder = (SCRIPTS / "build_sky_repository_tar.sh").read_text()
    for excluded in (
        ".git",
        "runs",
        ".venv",
        ".env",
        "*.safetensors",
    ):
        assert excluded in tar_builder
    assert "COPYFILE_DISABLE=1" in tar_builder


def test_stager_is_account_guarded_audits_before_descriptor_and_marker_last() -> None:
    source = (SCRIPTS / "stage_sky_campaign_bundle.py").read_text()
    assert "assert_rnd_aws_account.sh" in source
    assert "--if-none-match" in source
    assert "--checksum-authority" in source
    assert "audit_s3_campaign_artifacts.py" in source
    assert source.index("audit_s3_campaign_artifacts.py") < source.index(
        'staged["descriptor"]'
    )
    assert "STAGED_CONTROL_PLANE_READY.json" in source
    assert "PurchaseCapacityBlock" not in source
    assert "run-instances" not in source


def _load_bundle_stager():
    path = SCRIPTS / "stage_sky_campaign_bundle.py"
    specification = importlib.util.spec_from_file_location(
        "_task_3pe_bundle_stager",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_immutable_put_recovers_generic_ambiguous_committed_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_bundle_stager()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"ambiguous committed bytes")
    calls: list[list[str]] = []
    committed = False

    def fake_run(command: list[str], **_kwargs: object):
        nonlocal committed
        calls.append(command)
        operation = command[1:3]
        if operation == ["s3api", "head-object"]:
            if not committed:
                return subprocess.CompletedProcess(command, 1, "", "404 Not Found")
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "ContentLength": payload.stat().st_size,
                        "VersionId": "ambiguous-version-1",
                    }
                ),
                "",
            )
        if operation == ["s3api", "put-object"]:
            committed = True
            return subprocess.CompletedProcess(
                command,
                1,
                "",
                "transport reset after commit",
            )
        if operation == ["s3api", "get-object"]:
            assert command[command.index("--version-id") + 1] == ("ambiguous-version-1")
            Path(command[-1]).write_bytes(payload.read_bytes())
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"VersionId": "ambiguous-version-1"}),
                "",
            )
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    staged = module._put_immutable(
        profile="fake",
        region="us-west-2",
        bucket="fake-bucket",
        key="campaigns/test/payload.bin",
        path=payload,
        run_id="test",
    )

    assert staged.disposition == "reused-after-ambiguous-put"
    assert staged.version_id == "ambiguous-version-1"
    assert sum(call[1:3] == ["s3api", "put-object"] for call in calls) == 1


@pytest.mark.parametrize(
    ("version_case", "version_id"),
    (
        ("missing", "unused"),
        ("empty", ""),
        ("json_null", None),
        ("literal_null", "null"),
    ),
)
def test_immutable_put_rejects_success_without_non_null_version_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version_case: str,
    version_id: object,
) -> None:
    module = _load_bundle_stager()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"successful bytes without version identity")

    def fake_run(command: list[str], **_kwargs: object):
        if command[1:3] == ["s3api", "head-object"]:
            return subprocess.CompletedProcess(command, 1, "", "NoSuchKey")
        if command[1:3] == ["s3api", "put-object"]:
            response: dict[str, object] = {}
            if version_case != "missing":
                response["VersionId"] = version_id
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(response),
                "",
            )
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="VersionId"):
        module._put_immutable(
            profile="fake",
            region="us-west-2",
            bucket="fake-bucket",
            key="campaigns/test/payload.bin",
            path=payload,
            run_id="test",
        )


def test_immutable_put_ambiguous_absence_is_terminal_after_one_put(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_bundle_stager()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"never committed")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object):
        calls.append(command)
        if command[1:3] == ["s3api", "head-object"]:
            return subprocess.CompletedProcess(command, 1, "", "NoSuchKey")
        if command[1:3] == ["s3api", "put-object"]:
            return subprocess.CompletedProcess(command, 1, "", "connection lost")
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="connection lost"):
        module._put_immutable(
            profile="fake",
            region="us-west-2",
            bucket="fake-bucket",
            key="campaigns/test/payload.bin",
            path=payload,
            run_id="test",
        )

    assert sum(call[1:3] == ["s3api", "put-object"] for call in calls) == 1
    assert sum(call[1:3] == ["s3api", "head-object"] for call in calls) == 2


def test_immutable_put_ambiguous_incompatible_object_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_bundle_stager()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"intended bytes")
    foreign = b"x" * payload.stat().st_size
    assert foreign != payload.read_bytes()
    calls: list[list[str]] = []
    committed = False

    def fake_run(command: list[str], **_kwargs: object):
        nonlocal committed
        calls.append(command)
        if command[1:3] == ["s3api", "head-object"]:
            if not committed:
                return subprocess.CompletedProcess(command, 1, "", "NoSuchKey")
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "ContentLength": len(foreign),
                        "VersionId": "foreign-version",
                    }
                ),
                "",
            )
        if command[1:3] == ["s3api", "put-object"]:
            committed = True
            return subprocess.CompletedProcess(command, 1, "", "connection lost")
        if command[1:3] == ["s3api", "get-object"]:
            assert command[command.index("--version-id") + 1] == "foreign-version"
            Path(command[-1]).write_bytes(foreign)
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"VersionId": "foreign-version"}),
                "",
            )
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(
        RuntimeError,
        match="connection lost.*incompatible exact S3 version",
    ):
        module._put_immutable(
            profile="fake",
            region="us-west-2",
            bucket="fake-bucket",
            key="campaigns/test/payload.bin",
            path=payload,
            run_id="test",
        )

    assert sum(call[1:3] == ["s3api", "put-object"] for call in calls) == 1
    assert sum(call[1:3] == ["s3api", "head-object"] for call in calls) == 2


@pytest.mark.parametrize(
    ("version_case", "version_id"),
    (
        ("missing", "unused"),
        ("empty", ""),
        ("json_null", None),
        ("literal_null", "null"),
    ),
)
def test_immutable_put_existing_reuse_requires_non_null_version_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version_case: str,
    version_id: object,
) -> None:
    module = _load_bundle_stager()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"existing exact bytes")

    def fake_run(command: list[str], **_kwargs: object):
        assert command[1:3] == ["s3api", "head-object"]
        response: dict[str, object] = {"ContentLength": payload.stat().st_size}
        if version_case != "missing":
            response["VersionId"] = version_id
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="VersionId"):
        module._put_immutable(
            profile="fake",
            region="us-west-2",
            bucket="fake-bucket",
            key="campaigns/test/payload.bin",
            path=payload,
            run_id="test",
        )


def test_immutable_put_authenticates_captured_version_not_changed_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_bundle_stager()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"captured exact bytes")
    latest_version = "captured-v1"

    def fake_run(command: list[str], **_kwargs: object):
        nonlocal latest_version
        if command[1:3] == ["s3api", "head-object"]:
            captured = latest_version
            result = subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "ContentLength": payload.stat().st_size,
                        "VersionId": captured,
                    }
                ),
                "",
            )
            latest_version = "changed-v2"
            return result
        if command[1:3] == ["s3api", "get-object"]:
            assert command[command.index("--version-id") + 1] == "captured-v1"
            Path(command[-1]).write_bytes(payload.read_bytes())
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"VersionId": "captured-v1"}),
                "",
            )
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    staged = module._put_immutable(
        profile="fake",
        region="us-west-2",
        bucket="fake-bucket",
        key="campaigns/test/payload.bin",
        path=payload,
        run_id="test",
    )

    assert staged.disposition == "reused"
    assert staged.version_id == "captured-v1"
    assert latest_version == "changed-v2"


def test_immutable_put_never_overwrites_incompatible_existing_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_bundle_stager()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"intended")
    foreign = b"foreign!"
    assert len(foreign) == payload.stat().st_size
    assert foreign != payload.read_bytes()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object):
        calls.append(command)
        if command[1:3] == ["s3api", "head-object"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "ContentLength": len(foreign),
                        "VersionId": "foreign-v1",
                    }
                ),
                "",
            )
        if command[1:3] == ["s3api", "get-object"]:
            assert command[command.index("--version-id") + 1] == "foreign-v1"
            Path(command[-1]).write_bytes(foreign)
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"VersionId": "foreign-v1"}),
                "",
            )
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="incompatible"):
        module._put_immutable(
            profile="fake",
            region="us-west-2",
            bucket="fake-bucket",
            key="campaigns/test/payload.bin",
            path=payload,
            run_id="test",
        )

    assert not any(call[1:3] == ["s3api", "put-object"] for call in calls)


class _FakeVersionedS3:
    def __init__(self) -> None:
        self.objects: dict[str, list[tuple[object, bytes]]] = {}
        self.metadata: dict[tuple[str, object], dict[str, str]] = {}
        self.head_overrides: dict[str, dict[str, object]] = {}
        self.head_calls: list[str] = []
        self.get_calls: list[tuple[str, str]] = []
        self.put_calls: list[str] = []
        self.put_hook = None
        self._next_version = 1

    def add_version(
        self,
        key: str,
        raw: bytes,
        *,
        version_id: object | None = None,
        metadata: dict[str, str] | None = None,
    ) -> object:
        if version_id is None:
            version_id = f"fake-version-{self._next_version}"
            self._next_version += 1
        self.objects.setdefault(key, []).append((version_id, raw))
        self.metadata[(key, version_id)] = dict(metadata or {})
        return version_id

    def head_object(self, key: str) -> dict[str, object] | None:
        self.head_calls.append(key)
        if key in self.head_overrides:
            return dict(self.head_overrides[key])
        versions = self.objects.get(key)
        if not versions:
            return None
        version_id, raw = versions[-1]
        return {
            "ContentLength": len(raw),
            "VersionId": version_id,
        }

    def get_object(
        self,
        *,
        key: str,
        version_id: str,
        destination: Path,
    ) -> dict[str, object]:
        self.get_calls.append((key, version_id))
        for candidate, raw in self.objects.get(key, ()):
            if candidate == version_id:
                destination.write_bytes(raw)
                return {
                    "VersionId": candidate,
                    "Metadata": self.metadata[(key, candidate)],
                }
        raise RuntimeError(f"missing fake exact version: {key}#{version_id}")

    def put_object(
        self,
        *,
        key: str,
        path: Path,
        run_id: str,
        expected_sha: str,
        metadata: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.put_calls.append(key)
        if self.put_hook is not None:
            result = self.put_hook(
                key=key,
                path=path,
                run_id=run_id,
                expected_sha=expected_sha,
                metadata=metadata,
            )
            if result is not None:
                return result
        if self.objects.get(key):
            return subprocess.CompletedProcess(
                ["fake-aws"],
                1,
                "",
                "412 PreconditionFailed",
            )
        version_id = self.add_version(
            key,
            path.read_bytes(),
            metadata=metadata,
        )
        return subprocess.CompletedProcess(
            ["fake-aws"],
            0,
            json.dumps({"VersionId": version_id}),
            "",
        )


class _AuditRecorder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, inventory_path: Path, output: Path) -> None:
        self.calls += 1
        inventory = json.loads(inventory_path.read_bytes())
        audit = {
            "schema_version": 1,
            "record_type": "glm52_s3_artifact_audit_v1",
            "audit_pass": True,
            "run_id": inventory["run_id"],
            "bucket": inventory["bucket"],
            "inventory_body_sha256": inventory["inventory_body_sha256"],
            "object_count": len(inventory["objects"]),
            "object_bytes": sum(int(item["size"]) for item in inventory["objects"]),
            "safetensors_object_count": 0,
            "safetensors_tensor_count": 0,
        }
        output.write_bytes(_staging_json_bytes(audit))


class _ClockRecorder:
    def __init__(self, value: datetime) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> datetime:
        self.calls += 1
        return self.value


def _staging_canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _staging_json_bytes(value: object) -> bytes:
    return _staging_canonical(value) + b"\n"


def _staging_sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_staging_json(path: Path, value: object) -> None:
    path.write_bytes(_staging_json_bytes(value))


_STAGED_READY_V2_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "descriptor_key",
    "descriptor_sha256",
    "descriptor_body_sha256",
    "campaign_identity_sha256",
    "bundle_manifest_key",
    "bundle_manifest_file_sha256",
    "bundle_manifest_body_sha256",
    "bundle_manifest_version_id",
    "staged_object_version_ids",
    "artifact_audit_key",
    "artifact_audit_sha256",
    "staged_at",
    "ready_body_sha256",
}
_STAGED_OBJECT_VERSION_ROLES = {
    "repository_tar",
    "approval",
    "training_config",
    "watchdog",
    "artifact_inventory",
    "artifact_audit",
    "descriptor",
}
_STAGING_RECEIPT_ROLES = _STAGED_OBJECT_VERSION_ROLES | {
    "bundle_manifest",
    "ready",
}


def _task_3pe_bundle(tmp_path: Path) -> dict[str, object]:
    run_id = "glm52-sky-stage-replay"
    bucket = "keep-glm52-stage-replay"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    payloads = {
        "repository_tar": ("repo.tar.gz", b"exact repository tar\n"),
        "training_config": ("training-config.json", b'{"batch_size":1}\n'),
        "watchdog": ("watchdog.zip", b"exact watchdog zip bytes\n"),
    }
    for _role, (name, raw) in payloads.items():
        (bundle / name).write_bytes(raw)

    approval = build_gpu_spend_approval(
        ingested_at=datetime(2026, 7, 24, 1, 5, tzinfo=UTC),
        slack_permalink=None,
    )
    approval_path = bundle / "GPU_SPEND_APPROVAL.json"
    _write_staging_json(approval_path, approval)
    repo_path = bundle / "repo.tar.gz"
    repo_sha = _staging_sha(repo_path.read_bytes())
    approval_sha = _staging_sha(approval_path.read_bytes())
    repo_key = f"campaigns/{run_id}/repository/repo-{repo_sha}.tar.gz"
    approval_key = (
        f"campaigns/{run_id}/authorities/GPU_SPEND_APPROVAL-{approval_sha}.json"
    )
    training_path = bundle / "training-config.json"
    training_sha = _staging_sha(training_path.read_bytes())
    training_key = f"campaigns/{run_id}/authorities/training-config-{training_sha}.json"
    watchdog_path = bundle / "watchdog.zip"
    watchdog_sha = _staging_sha(watchdog_path.read_bytes())
    watchdog_key = f"campaigns/{run_id}/lambda/watchdog-{watchdog_sha}.zip"

    inventory = build_s3_artifact_inventory(
        run_id=run_id,
        bucket=bucket,
        objects=[
            {
                "key": repo_key,
                "size": repo_path.stat().st_size,
                "sha256": repo_sha,
                "kind": "repository_tar",
                "safetensors": False,
                "run_scope": run_id,
            }
        ],
    )
    inventory_path = bundle / "artifact-inventory-v1.json"
    _write_staging_json(inventory_path, inventory)
    inventory_sha = _staging_sha(inventory_path.read_bytes())
    inventory_key = (
        f"campaigns/{run_id}/inventories/artifact-inventory-{inventory_sha}.json"
    )
    descriptor_key = f"campaigns/{run_id}/submissions/test/campaign-descriptor-v2.json"
    descriptor = build_sky_campaign_descriptor(
        run_id=run_id,
        must_start_by=datetime(2026, 7, 27, 1, 5, tzinfo=UTC),
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
        worker_identity=("arn:aws:iam::246813579024:role/keep-glm52-gpu-worker"),
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket=bucket,
        jobs_bucket=bucket,
        repo_tar_key=repo_key,
        repo_tar_sha256=repo_sha,
        campaign_descriptor_key=descriptor_key,
        approval_key=approval_key,
        approval_sha256=approval_sha,
        artifacts={
            "source_snapshot_prefix": "source-snapshot/",
            "source_snapshot_sha256": "2" * 64,
            "non_vq_prefix": "non-vq-package/",
            "non_vq_package_sha256": "3" * 64,
            "teich_pack_key": "teich-pack/pack.json",
            "teich_pack_sha256": "4" * 64,
            "frozen_prompt_pack_key": "quality/frozen.json",
            "frozen_prompt_pack_sha256": "5" * 64,
            "training_baseline_prefix": "training-baseline/",
            "training_baseline_sha256": "6" * 64,
            "training_config_key": training_key,
            "training_config_sha256": training_sha,
            "artifact_inventory_key": inventory_key,
            "artifact_inventory_sha256": inventory_sha,
            "qualification_cache_prefix": (
                f"qualification-cache/seeds/{run_id}/{'9' * 64}/"
            ),
            "qualification_cache_manifest_sha256": "9" * 64,
        },
    )
    descriptor_path = bundle / "campaign-descriptor-v2.json"
    _write_staging_json(descriptor_path, descriptor)

    role_specs = (
        ("repository_tar", repo_path, repo_key, 10),
        ("approval", approval_path, approval_key, 20),
        ("training_config", training_path, training_key, 30),
        ("watchdog", watchdog_path, watchdog_key, 40),
        ("artifact_inventory", inventory_path, inventory_key, 50),
        ("descriptor", descriptor_path, descriptor_key, 60),
    )
    manifest_files = [
        {
            "local_name": path.name,
            "key": key,
            "role": role,
            "stage_order": order,
            "size": path.stat().st_size,
            "sha256": _staging_sha(path.read_bytes()),
        }
        for role, path, key, order in role_specs
    ]
    manifest_body = {
        "schema_version": 1,
        "record_type": "glm52_sky_campaign_bundle_v1",
        "run_id": run_id,
        "bucket": bucket,
        "files": manifest_files,
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
    }
    manifest = {
        **manifest_body,
        "bundle_manifest_body_sha256": _staging_sha(_staging_canonical(manifest_body)),
    }
    _write_staging_json(bundle / "bundle-manifest-v1.json", manifest)
    ready_key = f"campaigns/{run_id}/submissions/test/STAGED_CONTROL_PLANE_READY.json"
    return {
        "bundle": bundle,
        "run_id": run_id,
        "bucket": bucket,
        "ready_key": ready_key,
        "descriptor_key": descriptor_key,
        "manifest": manifest,
    }


def _remote_manifest_key(fixture: dict[str, object]) -> str:
    return (
        f"campaigns/{fixture['run_id']}/submissions/test/bundle-manifests/"
        f"{fixture['manifest']['bundle_manifest_body_sha256']}/"
        "bundle-manifest-v1.json"
    )


def _rewrite_local_manifest(
    fixture: dict[str, object],
    mutate,
    *,
    canonical: bool = True,
) -> None:
    path = Path(fixture["bundle"]) / "bundle-manifest-v1.json"
    value = json.loads(path.read_bytes())
    mutate(value)
    body = dict(value)
    body.pop("bundle_manifest_body_sha256", None)
    value["bundle_manifest_body_sha256"] = _staging_sha(
        _staging_canonical(body)
    )
    if canonical:
        path.write_bytes(_staging_json_bytes(value))
    else:
        path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _run_fake_stage(
    module,
    fixture: dict[str, object],
    *,
    s3: _FakeVersionedS3,
    audit: _AuditRecorder,
    clock: _ClockRecorder,
) -> dict[str, object]:
    return module._stage_bundle(
        bundle=fixture["bundle"],
        s3=s3,
        audit_output=Path(fixture["bundle"]) / "artifact-audit-v1.json",
        run_audit=audit,
        clock=clock,
    )


def _replace_ready(
    raw: bytes,
    **changes: object,
) -> bytes:
    value = json.loads(raw)
    value.update(changes)
    body = dict(value)
    body.pop("ready_body_sha256")
    value["ready_body_sha256"] = _staging_sha(_staging_canonical(body))
    return _staging_json_bytes(value)


def _replace_audit(
    raw: bytes,
    *,
    canonical: bool = True,
    **changes: object,
) -> bytes:
    value = json.loads(raw)
    value.update(changes)
    if canonical:
        return _staging_json_bytes(value)
    return json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"


def _bind_ready_to_audit(
    *,
    fixture: dict[str, object],
    ready_raw: bytes,
    audit_raw: bytes,
    audit_version_id: str,
) -> tuple[str, bytes]:
    audit_sha = _staging_sha(audit_raw)
    audit_key = f"campaigns/{fixture['run_id']}/audits/artifact-audit-{audit_sha}.json"
    versions = dict(json.loads(ready_raw)["staged_object_version_ids"])
    versions["artifact_audit"] = audit_version_id
    return (
        audit_key,
        _replace_ready(
            ready_raw,
            artifact_audit_key=audit_key,
            artifact_audit_sha256=audit_sha,
            staged_object_version_ids=versions,
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "noncanonical",
        "unsafe_local_name",
        "duplicate_local_name",
        "unsafe_key",
        "duplicate_key",
        "bool_size",
        "bad_sha",
        "duplicate_role",
        "duplicate_stage_order",
        "bool_stage_order",
        "descriptor_not_last",
        "extra_field",
    ),
)
def test_stager_rejects_invalid_local_manifest_before_remote_or_audit_side_effect(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)

    def mutate(value: dict[str, object]) -> None:
        files = value["files"]
        assert isinstance(files, list)
        if mutation == "unsafe_local_name":
            files[0]["local_name"] = "../repo.tar.gz"
        elif mutation == "duplicate_local_name":
            files[1]["local_name"] = files[0]["local_name"]
        elif mutation == "unsafe_key":
            files[0]["key"] = "../repository.tar.gz"
        elif mutation == "duplicate_key":
            files[1]["key"] = files[0]["key"]
        elif mutation == "bool_size":
            files[0]["size"] = True
        elif mutation == "bad_sha":
            files[0]["sha256"] = "A" * 64
        elif mutation == "duplicate_role":
            files[1]["role"] = files[0]["role"]
        elif mutation == "duplicate_stage_order":
            files[1]["stage_order"] = files[0]["stage_order"]
        elif mutation == "bool_stage_order":
            files[0]["stage_order"] = True
        elif mutation == "descriptor_not_last":
            files[-1]["stage_order"] = 0
        elif mutation == "extra_field":
            value["unexpected"] = "forbidden"

    _rewrite_local_manifest(
        fixture,
        mutate,
        canonical=mutation != "noncanonical",
    )
    s3 = _FakeVersionedS3()
    audit = _AuditRecorder()
    clock = _ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC))

    with pytest.raises((TypeError, ValueError)):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=audit,
            clock=clock,
        )

    assert s3.head_calls == []
    assert s3.get_calls == []
    assert s3.put_calls == []
    assert audit.calls == 0
    assert clock.calls == 0


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("run_id", 7),
        ("run_id", True),
        ("run_id", None),
        ("run_id", ["nested"]),
        ("run_id", "-leading-hyphen"),
        ("run_id", "contains whitespace"),
        ("run_id", "x" * 129),
        ("bucket", 7),
        ("bucket", True),
        ("bucket", None),
        ("bucket", {"nested": "value"}),
        ("bucket", "UPPERCASE"),
        ("bucket", "-leading-hyphen"),
        ("bucket", "trailing-hyphen-"),
        ("bucket", "two..dots"),
        ("bucket", "ab"),
    ),
)
def test_stager_rejects_invalid_manifest_run_and_bucket_before_side_effect(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    _rewrite_local_manifest(
        fixture,
        lambda value: value.__setitem__(field, invalid),
    )
    manifest_raw = (
        Path(fixture["bundle"]) / "bundle-manifest-v1.json"
    ).read_bytes()

    with pytest.raises(ValueError, match=field):
        module._validate_manifest_bytes(manifest_raw)

    s3 = _FakeVersionedS3()
    audit = _AuditRecorder()
    clock = _ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC))
    with pytest.raises(ValueError, match=field):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=audit,
            clock=clock,
        )
    assert s3.head_calls == []
    assert s3.get_calls == []
    assert s3.put_calls == []
    assert audit.calls == 0
    assert clock.calls == 0


@pytest.mark.parametrize(
    "unsafe",
    (
        "campaigns/run/watchdog\npayload.zip",
        "campaigns/run/watchdog\tpayload.zip",
        "campaigns/run/watchdog\x00payload.zip",
        "campaigns/run/watchdog\x7fpayload.zip",
        r"campaigns/run/watchdog\payload.zip",
    ),
)
def test_stager_rejects_non_printable_or_backslash_watchdog_key_before_side_effect(
    tmp_path: Path,
    unsafe: str,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)

    def mutate(value: dict[str, object]) -> None:
        files = value["files"]
        assert isinstance(files, list)
        watchdog = next(item for item in files if item["role"] == "watchdog")
        watchdog["key"] = unsafe

    _rewrite_local_manifest(fixture, mutate)
    manifest_raw = (
        Path(fixture["bundle"]) / "bundle-manifest-v1.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="safe|ASCII"):
        module._validate_manifest_bytes(manifest_raw)

    s3 = _FakeVersionedS3()
    audit = _AuditRecorder()
    clock = _ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC))
    with pytest.raises(ValueError, match="safe|ASCII"):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=audit,
            clock=clock,
        )
    assert s3.head_calls == []
    assert s3.get_calls == []
    assert s3.put_calls == []
    assert audit.calls == 0
    assert clock.calls == 0


@pytest.mark.parametrize(
    "unsafe",
    (
        "campaigns/run/watchdog\npayload.zip",
        "campaigns/run/watchdog\tpayload.zip",
        "campaigns/run/watchdog\x00payload.zip",
        "campaigns/run/watchdog\x7fpayload.zip",
        r"campaigns/run/watchdog\payload.zip",
    ),
)
def test_stager_remote_manifest_replay_parser_rejects_unsafe_watchdog_key(
    tmp_path: Path,
    unsafe: str,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    context = module._prepare_bundle(Path(fixture["bundle"]))
    manifest = json.loads(context.manifest_raw)
    watchdog = next(
        item for item in manifest["files"] if item["role"] == "watchdog"
    )
    watchdog["key"] = unsafe
    body = dict(manifest)
    body.pop("bundle_manifest_body_sha256")
    manifest["bundle_manifest_body_sha256"] = _staging_sha(
        _staging_canonical(body)
    )
    manifest_raw = _staging_json_bytes(manifest)
    manifest_key = (
        str(context.descriptor_item["key"]).removesuffix(
            "campaign-descriptor-v2.json"
        )
        + "bundle-manifests/"
        + str(manifest["bundle_manifest_body_sha256"])
        + "/bundle-manifest-v1.json"
    )
    s3 = _FakeVersionedS3()
    version = str(s3.add_version(manifest_key, manifest_raw))
    ready = {
        "bundle_manifest_key": manifest_key,
        "bundle_manifest_file_sha256": _staging_sha(manifest_raw),
        "bundle_manifest_body_sha256": manifest[
            "bundle_manifest_body_sha256"
        ],
        "bundle_manifest_version_id": version,
    }

    with pytest.raises(ValueError, match="safe|ASCII"):
        module._authenticate_remote_manifest(
            s3=s3,
            context=context,
            ready=ready,
        )

    assert s3.get_calls == [(manifest_key, version)]
    assert s3.put_calls == []


def test_stager_cli_completes_bundle_validation_before_account_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    _rewrite_local_manifest(fixture, lambda _value: None, canonical=False)
    subprocess_calls: list[list[str]] = []

    def fake_process(command: list[str], **_kwargs: object):
        subprocess_calls.append(command)
        raise AssertionError("invalid local authority reached an external side effect")

    monkeypatch.setattr(module.subprocess, "run", fake_process)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPTS / "stage_sky_campaign_bundle.py"),
            "--bundle",
            str(fixture["bundle"]),
            "--profile",
            "fake-profile",
            "--checksum-authority",
            str(tmp_path / "unused.json"),
        ],
    )

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2
    assert subprocess_calls == []


def test_stager_fresh_publishes_remote_manifest_and_readiness_v2(
    tmp_path: Path,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()

    receipt = _run_fake_stage(
        module,
        fixture,
        s3=s3,
        audit=_AuditRecorder(),
        clock=_ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC)),
    )

    manifest_raw = (
        Path(fixture["bundle"]) / "bundle-manifest-v1.json"
    ).read_bytes()
    manifest_key = _remote_manifest_key(fixture)
    ready_raw = s3.objects[str(fixture["ready_key"])][-1][1]
    ready = json.loads(ready_raw)

    assert s3.objects[manifest_key][-1][1] == manifest_raw
    assert set(ready) == _STAGED_READY_V2_FIELDS
    assert ready["schema_version"] == 2
    assert ready["record_type"] == "glm52_staged_control_plane_ready_v2"
    assert ready["bundle_manifest_key"] == manifest_key
    assert ready["bundle_manifest_file_sha256"] == _staging_sha(manifest_raw)
    assert (
        ready["bundle_manifest_version_id"]
        == receipt["s3_version_ids"]["bundle_manifest"]
    )
    assert set(ready["staged_object_version_ids"]) == (
        _STAGED_OBJECT_VERSION_ROLES
    )
    assert ready["staged_object_version_ids"] == {
        role: receipt["s3_version_ids"][role]
        for role in _STAGED_OBJECT_VERSION_ROLES
    }
    ready_body = dict(ready)
    ready_body_sha = ready_body.pop("ready_body_sha256")
    assert ready_body_sha == _staging_sha(_staging_canonical(ready_body))
    assert set(receipt["staged"]) == _STAGING_RECEIPT_ROLES
    assert set(receipt["s3_version_ids"]) == _STAGING_RECEIPT_ROLES
    puts = s3.put_calls
    audit_index = puts.index(str(receipt["artifact_audit_key"]))
    manifest_index = puts.index(manifest_key)
    descriptor_index = puts.index(str(fixture["descriptor_key"]))
    ready_index = puts.index(str(fixture["ready_key"]))
    assert audit_index < manifest_index < descriptor_index < ready_index


def test_stager_replay_uses_first_readiness_winner_before_clock_audit_or_put(
    tmp_path: Path,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    first_audit = _AuditRecorder()
    first_clock = _ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC))

    first = _run_fake_stage(
        module,
        fixture,
        s3=s3,
        audit=first_audit,
        clock=first_clock,
    )
    puts_after_first = len(s3.put_calls)
    gets_before_replay = len(s3.get_calls)
    second_audit = _AuditRecorder()
    second_clock = _ClockRecorder(datetime(2026, 7, 26, 13, 0, tzinfo=UTC))
    second = _run_fake_stage(
        module,
        fixture,
        s3=s3,
        audit=second_audit,
        clock=second_clock,
    )

    expected_receipt_fields = {
        "run_id",
        "bucket",
        "descriptor_key",
        "ready_key",
        "artifact_audit_key",
        "staged",
        "staged_readiness_file_sha256",
        "staged_readiness_body_sha256",
        "staged_readiness_version_id",
        "s3_version_ids",
    }
    expected_roles = _STAGING_RECEIPT_ROLES
    assert set(first) == expected_receipt_fields
    assert set(first["s3_version_ids"]) == expected_roles
    assert all(
        isinstance(version, str) and version not in {"", "null"}
        for version in first["s3_version_ids"].values()
    )
    assert (
        first["staged_readiness_file_sha256"] == second["staged_readiness_file_sha256"]
    )
    assert (
        first["staged_readiness_body_sha256"] == second["staged_readiness_body_sha256"]
    )
    assert first["staged_readiness_version_id"] == second["staged_readiness_version_id"]
    assert first["s3_version_ids"]["ready"] == second["s3_version_ids"]["ready"]
    ready_versions = s3.objects[str(fixture["ready_key"])]
    assert len(ready_versions) == 1
    assert json.loads(ready_versions[0][1])["staged_at"] == ("2026-07-26T12:00:00Z")
    assert len(s3.put_calls) == puts_after_first
    assert first_audit.calls == 1
    assert first_clock.calls == 1
    assert second_audit.calls == 0
    assert second_clock.calls == 0
    replay_reads = s3.get_calls[gets_before_replay:]
    assert {key for key, _version in replay_reads} == {
        item["key"] for item in fixture["manifest"]["files"]
    } | {
        first["artifact_audit_key"],
        _remote_manifest_key(fixture),
        fixture["ready_key"],
    }
    assert second["staged"] == {role: "reused" for role in expected_roles}


def test_stager_replay_uses_only_marker_pinned_versions_after_latest_changes(
    tmp_path: Path,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    first = _run_fake_stage(
        module,
        fixture,
        s3=s3,
        audit=_AuditRecorder(),
        clock=_ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC)),
    )
    ready_raw = s3.objects[str(fixture["ready_key"])][-1][1]
    ready = json.loads(ready_raw)
    manifest_key = _remote_manifest_key(fixture)
    pinned_versions = dict(first["s3_version_ids"])

    for item in fixture["manifest"]["files"]:
        s3.add_version(str(item["key"]), b"x" * int(item["size"]))
    s3.add_version(str(first["artifact_audit_key"]), b"foreign latest audit\n")
    s3.add_version(manifest_key, b"foreign latest manifest\n")
    gets_before = len(s3.get_calls)

    replay = _run_fake_stage(
        module,
        fixture,
        s3=s3,
        audit=_AuditRecorder(),
        clock=_ClockRecorder(datetime(2026, 7, 26, 13, 0, tzinfo=UTC)),
    )

    replay_gets = s3.get_calls[gets_before:]
    expected_exact_reads = {
        (str(fixture["ready_key"]), pinned_versions["ready"]),
        (manifest_key, str(ready["bundle_manifest_version_id"])),
        (
            str(first["artifact_audit_key"]),
            str(ready["staged_object_version_ids"]["artifact_audit"]),
        ),
    } | {
        (
            str(item["key"]),
            str(ready["staged_object_version_ids"][str(item["role"])]),
        )
        for item in fixture["manifest"]["files"]
    }
    assert set(replay_gets) == expected_exact_reads
    assert replay["s3_version_ids"] == pinned_versions


def test_stager_conditional_readiness_race_accepts_different_valid_winner_once(
    tmp_path: Path,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    winner_time = "2026-07-26T12:00:00Z"

    def race_put(*, key: str, path: Path, **_kwargs: object):
        if key != fixture["ready_key"]:
            return None
        winner_raw = _replace_ready(path.read_bytes(), staged_at=winner_time)
        s3.add_version(
            key,
            winner_raw,
            version_id="first-winner-ready-version",
        )
        return subprocess.CompletedProcess(
            ["fake-aws"],
            1,
            "",
            "transport reset after concurrent winner",
        )

    s3.put_hook = race_put
    receipt = _run_fake_stage(
        module,
        fixture,
        s3=s3,
        audit=_AuditRecorder(),
        clock=_ClockRecorder(datetime(2026, 7, 26, 13, 0, tzinfo=UTC)),
    )
    winner_raw = s3.objects[str(fixture["ready_key"])][-1][1]

    assert json.loads(winner_raw)["staged_at"] == winner_time
    assert receipt["staged_readiness_file_sha256"] == _staging_sha(winner_raw)
    assert (
        receipt["staged_readiness_body_sha256"]
        == json.loads(winner_raw)["ready_body_sha256"]
    )
    assert receipt["staged_readiness_version_id"] == ("first-winner-ready-version")
    assert receipt["s3_version_ids"]["ready"] == "first-winner-ready-version"
    assert receipt["staged"]["ready"] == "reused-after-race"
    assert s3.put_calls.count(str(fixture["ready_key"])) == 1
    assert len(s3.objects[str(fixture["ready_key"])]) == 1


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("corrupt_hash", "body SHA-256"),
        ("noncanonical", "canonical"),
        ("foreign_run", "run_id"),
        ("foreign_descriptor", "descriptor"),
        ("foreign_manifest", "manifest"),
        ("foreign_audit", "audit"),
        ("unsafe_time", "staged_at"),
        ("fractional_time", "staged_at"),
        ("mixed_schema", "schema"),
    ),
)
def test_stager_replay_rejects_corrupt_or_foreign_readiness_before_put(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    first = _run_fake_stage(
        module,
        fixture,
        s3=s3,
        audit=_AuditRecorder(),
        clock=_ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC)),
    )
    ready_key = str(fixture["ready_key"])
    original = s3.objects[ready_key][-1][1]
    if mutation == "corrupt_hash":
        value = json.loads(original)
        value["ready_body_sha256"] = "f" * 64
        changed = _staging_json_bytes(value)
    elif mutation == "noncanonical":
        changed = json.dumps(json.loads(original), indent=2).encode() + b"\n"
    elif mutation == "foreign_run":
        changed = _replace_ready(original, run_id="foreign-run")
    elif mutation == "foreign_descriptor":
        changed = _replace_ready(
            original,
            descriptor_key=(
                f"campaigns/{fixture['run_id']}/submissions/foreign/"
                "campaign-descriptor-v2.json"
            ),
        )
    elif mutation == "foreign_manifest":
        changed = _replace_ready(
            original,
            bundle_manifest_body_sha256="f" * 64,
        )
    elif mutation == "foreign_audit":
        changed = _replace_ready(
            original,
            artifact_audit_key=(
                f"campaigns/{fixture['run_id']}/audits/artifact-audit-{'f' * 64}.json"
            ),
        )
    elif mutation == "unsafe_time":
        changed = _replace_ready(
            original,
            staged_at="2026-07-26T12:00:00+00:00",
        )
    elif mutation == "fractional_time":
        changed = _replace_ready(
            original,
            staged_at="2026-07-26T12:00:00.123456Z",
        )
    else:
        changed = _replace_ready(
            original,
            schema_version=1,
            record_type="glm52_staged_control_plane_ready_v1",
        )
    s3.add_version(ready_key, changed)
    puts_before = len(s3.put_calls)

    with pytest.raises((RuntimeError, TypeError, ValueError), match=error):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=_AuditRecorder(),
            clock=_ClockRecorder(datetime(2026, 7, 26, 13, 0, tzinfo=UTC)),
        )

    assert len(s3.put_calls) == puts_before
    assert first["ready_key"] == ready_key


def test_stager_rejects_historical_v1_readiness_before_mutation(
    tmp_path: Path,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    historical_body = {
        "schema_version": 1,
        "record_type": "glm52_staged_control_plane_ready_v1",
        "run_id": fixture["run_id"],
        "descriptor_key": fixture["descriptor_key"],
        "descriptor_sha256": "a" * 64,
        "descriptor_body_sha256": "b" * 64,
        "campaign_identity_sha256": "c" * 64,
        "bundle_manifest_body_sha256": (
            fixture["manifest"]["bundle_manifest_body_sha256"]
        ),
        "artifact_audit_key": (
            f"campaigns/{fixture['run_id']}/audits/"
            f"artifact-audit-{'d' * 64}.json"
        ),
        "artifact_audit_sha256": "d" * 64,
        "staged_at": "2026-07-26T12:00:00Z",
    }
    historical = {
        **historical_body,
        "ready_body_sha256": _staging_sha(_staging_canonical(historical_body)),
    }
    s3.add_version(
        str(fixture["ready_key"]),
        _staging_json_bytes(historical),
        version_id="historical-v1",
    )
    audit = _AuditRecorder()
    clock = _ClockRecorder(datetime(2026, 7, 26, 13, 0, tzinfo=UTC))

    with pytest.raises(ValueError, match="schema"):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=audit,
            clock=clock,
        )

    assert audit.calls == 0
    assert clock.calls == 0
    assert s3.put_calls == []


@pytest.mark.parametrize(
    ("mutation", "version_value"),
    (
        ("missing_role", "unused"),
        ("extra_role", "unused"),
        ("missing_value", "unused"),
        ("empty", ""),
        ("json_null", None),
        ("literal_null", "null"),
        ("non_string", 7),
    ),
)
def test_stager_replay_rejects_invalid_pinned_version_map_before_mutation(
    tmp_path: Path,
    mutation: str,
    version_value: object,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    _run_fake_stage(
        module,
        fixture,
        s3=s3,
        audit=_AuditRecorder(),
        clock=_ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC)),
    )
    ready_key = str(fixture["ready_key"])
    ready = json.loads(s3.objects[ready_key][-1][1])
    versions = dict(ready["staged_object_version_ids"])
    if mutation == "missing_role":
        versions.pop("repository_tar")
    elif mutation == "extra_role":
        versions["unexpected"] = "foreign"
    elif mutation == "missing_value":
        versions.pop("artifact_audit")
    else:
        versions["repository_tar"] = version_value
    s3.add_version(
        ready_key,
        _replace_ready(
            s3.objects[ready_key][-1][1],
            staged_object_version_ids=versions,
        ),
    )
    puts_before = len(s3.put_calls)

    with pytest.raises((RuntimeError, TypeError, ValueError), match="VersionId|version"):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=_AuditRecorder(),
            clock=_ClockRecorder(datetime(2026, 7, 26, 13, 0, tzinfo=UTC)),
        )

    assert len(s3.put_calls) == puts_before


@pytest.mark.parametrize(
    ("version_case", "version_id"),
    (
        ("missing", "unused"),
        ("empty", ""),
        ("json_null", None),
        ("literal_null", "null"),
    ),
)
def test_stager_replay_rejects_missing_empty_or_null_readiness_version(
    tmp_path: Path,
    version_case: str,
    version_id: object,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    _run_fake_stage(
        module,
        fixture,
        s3=s3,
        audit=_AuditRecorder(),
        clock=_ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC)),
    )
    ready_key = str(fixture["ready_key"])
    ready_raw = s3.objects[ready_key][-1][1]
    s3.head_overrides[ready_key] = {"ContentLength": len(ready_raw)}
    if version_case != "missing":
        s3.head_overrides[ready_key]["VersionId"] = version_id
    puts_before = len(s3.put_calls)

    with pytest.raises(ValueError, match="VersionId"):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=_AuditRecorder(),
            clock=_ClockRecorder(datetime(2026, 7, 26, 13, 0, tzinfo=UTC)),
        )

    assert len(s3.put_calls) == puts_before


@pytest.mark.parametrize(
    ("version_case", "version_id"),
    (
        ("missing", "unused"),
        ("empty", ""),
        ("json_null", None),
        ("literal_null", "null"),
    ),
)
def test_stager_successful_readiness_put_without_non_null_version_has_no_receipt(
    tmp_path: Path,
    version_case: str,
    version_id: object,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()

    def readiness_put(*, key: str, path: Path, **_kwargs: object):
        if key != fixture["ready_key"]:
            return None
        s3.add_version(
            key,
            path.read_bytes(),
            version_id="committed-ready-version",
        )
        response: dict[str, object] = {}
        if version_case != "missing":
            response["VersionId"] = version_id
        return subprocess.CompletedProcess(
            ["fake-aws"],
            0,
            json.dumps(response),
            "",
        )

    s3.put_hook = readiness_put
    with pytest.raises(ValueError, match="VersionId"):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=_AuditRecorder(),
            clock=_ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC)),
        )

    ready_key = str(fixture["ready_key"])
    assert s3.put_calls.count(ready_key) == 1
    assert not any(key == ready_key for key, _version in s3.get_calls)


@pytest.mark.parametrize(
    ("version_case", "version_id"),
    (
        ("missing", "unused"),
        ("empty", ""),
        ("json_null", None),
        ("literal_null", "null"),
    ),
)
def test_stager_replay_rejects_invalid_manifest_version_id(
    tmp_path: Path,
    version_case: str,
    version_id: object,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    _run_fake_stage(
        module,
        fixture,
        s3=s3,
        audit=_AuditRecorder(),
        clock=_ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC)),
    )
    ready_key = str(fixture["ready_key"])
    value = json.loads(s3.objects[ready_key][-1][1])
    if version_case == "missing":
        value.pop("bundle_manifest_version_id")
    else:
        value["bundle_manifest_version_id"] = version_id
    body = dict(value)
    body.pop("ready_body_sha256")
    value["ready_body_sha256"] = _staging_sha(_staging_canonical(body))
    s3.add_version(ready_key, _staging_json_bytes(value))
    puts_before = len(s3.put_calls)

    with pytest.raises(ValueError, match="schema|VersionId"):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=_AuditRecorder(),
            clock=_ClockRecorder(datetime(2026, 7, 26, 13, 0, tzinfo=UTC)),
        )

    assert len(s3.put_calls) == puts_before


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("missing", "manifest"),
        ("corrupt", "manifest"),
        ("noncanonical", "manifest"),
        ("wrong_key", "manifest"),
        ("wrong_file_sha", "manifest"),
        ("wrong_body_sha", "manifest"),
    ),
)
def test_stager_replay_rejects_invalid_remote_manifest_authority(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    _run_fake_stage(
        module,
        fixture,
        s3=s3,
        audit=_AuditRecorder(),
        clock=_ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC)),
    )
    ready_key = str(fixture["ready_key"])
    original_ready = s3.objects[ready_key][-1][1]
    ready = json.loads(original_ready)
    manifest_key = str(ready["bundle_manifest_key"])
    manifest_version = str(ready["bundle_manifest_version_id"])
    if mutation == "missing":
        s3.objects[manifest_key] = [
            pair
            for pair in s3.objects[manifest_key]
            if pair[0] != manifest_version
        ]
    elif mutation == "corrupt":
        s3.objects[manifest_key] = [
            (
                candidate,
                b"corrupt remote manifest\n" if candidate == manifest_version else raw,
            )
            for candidate, raw in s3.objects[manifest_key]
        ]
    elif mutation == "noncanonical":
        remote = json.loads(s3.objects[manifest_key][-1][1])
        s3.objects[manifest_key][-1] = (
            manifest_version,
            json.dumps(remote, indent=2, sort_keys=True).encode() + b"\n",
        )
    elif mutation == "wrong_key":
        ready_raw = _replace_ready(
            original_ready,
            bundle_manifest_key=(
                f"campaigns/{fixture['run_id']}/submissions/test/"
                f"bundle-manifests/{'f' * 64}/bundle-manifest-v1.json"
            ),
        )
        s3.add_version(ready_key, ready_raw)
    elif mutation == "wrong_file_sha":
        s3.add_version(
            ready_key,
            _replace_ready(
                original_ready,
                bundle_manifest_file_sha256="f" * 64,
            ),
        )
    else:
        s3.add_version(
            ready_key,
            _replace_ready(
                original_ready,
                bundle_manifest_body_sha256="f" * 64,
            ),
        )
    puts_before = len(s3.put_calls)

    with pytest.raises((RuntimeError, TypeError, ValueError), match=error):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=_AuditRecorder(),
            clock=_ClockRecorder(datetime(2026, 7, 26, 13, 0, tzinfo=UTC)),
        )

    assert len(s3.put_calls) == puts_before


@pytest.mark.parametrize("mutation", ("absent", "drift"))
def test_stager_replay_rejects_referenced_audit_absence_or_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    first = _run_fake_stage(
        module,
        fixture,
        s3=s3,
        audit=_AuditRecorder(),
        clock=_ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC)),
    )
    audit_key = str(first["artifact_audit_key"])
    if mutation == "absent":
        del s3.objects[audit_key]
    else:
        pinned_version = str(first["s3_version_ids"]["artifact_audit"])
        s3.objects[audit_key] = [
            (
                candidate,
                b"drifted audit bytes\n" if candidate == pinned_version else raw,
            )
            for candidate, raw in s3.objects[audit_key]
        ]
    puts_before = len(s3.put_calls)

    with pytest.raises((RuntimeError, ValueError), match="audit"):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=_AuditRecorder(),
            clock=_ClockRecorder(datetime(2026, 7, 26, 13, 0, tzinfo=UTC)),
        )

    assert len(s3.put_calls) == puts_before


def test_stager_replay_rejects_foreign_inventory_audit_before_clock_audit_or_put(
    tmp_path: Path,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    first = _run_fake_stage(
        module,
        fixture,
        s3=s3,
        audit=_AuditRecorder(),
        clock=_ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC)),
    )
    original_audit_key = str(first["artifact_audit_key"])
    original_audit = s3.objects[original_audit_key][-1][1]
    foreign_audit = _replace_audit(
        original_audit,
        inventory_body_sha256="f" * 64,
    )
    foreign_audit_sha = _staging_sha(foreign_audit)
    foreign_audit_key = (
        f"campaigns/{fixture['run_id']}/audits/"
        f"artifact-audit-{foreign_audit_sha}.json"
    )
    foreign_audit_version = str(
        s3.add_version(foreign_audit_key, foreign_audit)
    )
    _foreign_audit_key, foreign_ready = _bind_ready_to_audit(
        fixture=fixture,
        ready_raw=s3.objects[str(fixture["ready_key"])][-1][1],
        audit_raw=foreign_audit,
        audit_version_id=foreign_audit_version,
    )
    assert _foreign_audit_key == foreign_audit_key
    s3.add_version(str(fixture["ready_key"]), foreign_ready)
    puts_before = len(s3.put_calls)
    replay_audit = _AuditRecorder()
    replay_clock = _ClockRecorder(datetime(2026, 7, 26, 13, 0, tzinfo=UTC))

    with pytest.raises(ValueError, match="inventory"):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=replay_audit,
            clock=replay_clock,
        )

    assert json.loads(foreign_audit)["run_id"] == fixture["run_id"]
    assert json.loads(foreign_audit)["bucket"] == fixture["bucket"]
    assert len(s3.put_calls) == puts_before
    assert replay_audit.calls == 0
    assert replay_clock.calls == 0


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("extra_field", "schema"),
        ("noncanonical", "canonical"),
        ("truthy_pass", "pass status"),
        ("wrong_count", "object_count"),
        ("wrong_bytes", "object_bytes"),
        ("wrong_safetensors", "safetensors_object_count"),
        ("wrong_tensor_count", "safetensors_tensor_count"),
    ),
)
def test_stager_replay_rejects_self_consistent_invalid_audit_authority(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    first = _run_fake_stage(
        module,
        fixture,
        s3=s3,
        audit=_AuditRecorder(),
        clock=_ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC)),
    )
    original_audit = s3.objects[str(first["artifact_audit_key"])][-1][1]
    if mutation == "extra_field":
        changed_audit = _replace_audit(original_audit, unexpected=True)
    elif mutation == "noncanonical":
        changed_audit = _replace_audit(original_audit, canonical=False)
    elif mutation == "truthy_pass":
        changed_audit = _replace_audit(original_audit, audit_pass=1)
    elif mutation == "wrong_bytes":
        changed_audit = _replace_audit(original_audit, object_bytes=999)
    elif mutation == "wrong_safetensors":
        changed_audit = _replace_audit(
            original_audit,
            safetensors_object_count=1,
        )
    elif mutation == "wrong_tensor_count":
        changed_audit = _replace_audit(
            original_audit,
            safetensors_tensor_count=1,
        )
    else:
        changed_audit = _replace_audit(original_audit, object_count=2)
    changed_audit_sha = _staging_sha(changed_audit)
    changed_audit_key = (
        f"campaigns/{fixture['run_id']}/audits/"
        f"artifact-audit-{changed_audit_sha}.json"
    )
    changed_audit_version = str(
        s3.add_version(changed_audit_key, changed_audit)
    )
    _changed_audit_key, changed_ready = _bind_ready_to_audit(
        fixture=fixture,
        ready_raw=s3.objects[str(fixture["ready_key"])][-1][1],
        audit_raw=changed_audit,
        audit_version_id=changed_audit_version,
    )
    assert _changed_audit_key == changed_audit_key
    s3.add_version(str(fixture["ready_key"]), changed_ready)
    puts_before = len(s3.put_calls)

    with pytest.raises(ValueError, match=error):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=_AuditRecorder(),
            clock=_ClockRecorder(datetime(2026, 7, 26, 13, 0, tzinfo=UTC)),
        )

    assert len(s3.put_calls) == puts_before


def test_stager_fresh_publication_rejects_foreign_inventory_audit_before_authority_puts(
    tmp_path: Path,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    audit = _AuditRecorder()
    clock = _ClockRecorder(datetime(2026, 7, 26, 12, 0, tzinfo=UTC))

    def foreign_inventory_audit(inventory_path: Path, output: Path) -> None:
        audit(inventory_path, output)
        output.write_bytes(
            _replace_audit(
                output.read_bytes(),
                inventory_body_sha256="f" * 64,
            )
        )

    with pytest.raises(ValueError, match="inventory"):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=foreign_inventory_audit,
            clock=clock,
        )

    assert audit.calls == 1
    assert clock.calls == 0
    assert not any("/audits/" in key for key in s3.put_calls)
    assert str(fixture["descriptor_key"]) not in s3.put_calls
    assert str(fixture["ready_key"]) not in s3.put_calls


def test_stager_readiness_race_rejects_foreign_inventory_audit_winner_once(
    tmp_path: Path,
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    audit = _AuditRecorder()
    clock = _ClockRecorder(datetime(2026, 7, 26, 13, 0, tzinfo=UTC))

    def race_put(*, key: str, path: Path, **_kwargs: object):
        if key != fixture["ready_key"]:
            return None
        candidate = json.loads(path.read_bytes())
        candidate_audit = s3.objects[str(candidate["artifact_audit_key"])][-1][1]
        foreign_audit = _replace_audit(
            candidate_audit,
            inventory_body_sha256="f" * 64,
        )
        foreign_audit_sha = _staging_sha(foreign_audit)
        foreign_audit_key = (
            f"campaigns/{fixture['run_id']}/audits/"
            f"artifact-audit-{foreign_audit_sha}.json"
        )
        foreign_audit_version = str(
            s3.add_version(foreign_audit_key, foreign_audit)
        )
        _foreign_audit_key, foreign_ready = _bind_ready_to_audit(
            fixture=fixture,
            ready_raw=path.read_bytes(),
            audit_raw=foreign_audit,
            audit_version_id=foreign_audit_version,
        )
        assert _foreign_audit_key == foreign_audit_key
        s3.add_version(
            key,
            foreign_ready,
            version_id="foreign-inventory-race-winner",
        )
        return subprocess.CompletedProcess(
            ["fake-aws"],
            1,
            "",
            "transport reset after concurrent winner",
        )

    s3.put_hook = race_put
    with pytest.raises(RuntimeError, match="inventory"):
        _run_fake_stage(
            module,
            fixture,
            s3=s3,
            audit=audit,
            clock=clock,
        )

    assert audit.calls == 1
    assert clock.calls == 1
    assert s3.put_calls.count(str(fixture["ready_key"])) == 1
    assert len(s3.objects[str(fixture["ready_key"])]) == 1


def test_stager_main_stdout_reports_authoritative_nine_role_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_bundle_stager()
    fixture = _task_3pe_bundle(tmp_path)
    s3 = _FakeVersionedS3()
    audit = _AuditRecorder()

    def fake_process(command: list[str], **_kwargs: object):
        if command[0].endswith("assert_rnd_aws_account.sh"):
            return subprocess.CompletedProcess(command, 0)
        if command[0].endswith("audit_s3_campaign_artifacts.py"):
            audit(
                Path(command[command.index("--inventory") + 1]),
                Path(command[command.index("--output") + 1]),
            )
            return subprocess.CompletedProcess(command, 0)
        raise AssertionError(command)

    monkeypatch.setattr(module, "_AwsCliS3", lambda **_kwargs: s3)
    monkeypatch.setattr(module.subprocess, "run", fake_process)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPTS / "stage_sky_campaign_bundle.py"),
            "--bundle",
            str(fixture["bundle"]),
            "--profile",
            "fake-profile",
            "--checksum-authority",
            str(tmp_path / "fake-checksum-authority.json"),
        ],
    )

    assert module.main() == 0
    uploaded = json.loads(capsys.readouterr().out)
    assert set(uploaded["s3_version_ids"]) == _STAGING_RECEIPT_ROLES
    assert (
        uploaded["staged_readiness_version_id"] == uploaded["s3_version_ids"]["ready"]
    )

    assert module.main() == 0
    replayed = json.loads(capsys.readouterr().out)
    assert (
        replayed["staged_readiness_file_sha256"]
        == uploaded["staged_readiness_file_sha256"]
    )
    assert (
        replayed["staged_readiness_body_sha256"]
        == uploaded["staged_readiness_body_sha256"]
    )
    assert replayed["s3_version_ids"] == uploaded["s3_version_ids"]
    assert audit.calls == 1


def test_clean_rehearsal_fetches_exact_staged_objects_and_stops_before_cuda() -> None:
    source = (SCRIPTS / "rehearse_sky_control_plane.sh").read_text()
    for required in (
        "assert_rnd_aws_account.sh",
        'CHECKSUM_AUTHORITY="${CHECKSUM_AUTHORITY:?set CHECKSUM_AUTHORITY',
        "campaign-descriptor-v2.json",
        "repo.tar.gz",
        "GPU_SPEND_APPROVAL.json",
        "artifact-inventory-v1.json",
        "audit_s3_campaign_artifacts.py",
        '"$WORK/repo/aws/glm52-gpu/scripts/audit_s3_campaign_artifacts.py"',
        '--checksum-authority "$CHECKSUM_AUTHORITY"',
        "validate_skypilot_control_plane.py",
        "GLM52_BOOTSTRAP_REHEARSAL=1",
        "passed_before_cuda_h100_boundary",
        "/opt/keep-campaign/repo",
        "/mnt/nvme/glm52-campaign",
    ):
        assert required in source


def test_new_staging_shell_scripts_have_valid_syntax() -> None:
    for name in (
        "build_sky_repository_tar.sh",
        "build_sky_campaign_bundle.sh",
        "rehearse_sky_control_plane.sh",
        "deploy_sky_control_plane.sh",
    ):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPTS / name)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"


def test_control_plane_deploy_has_observe_and_explicit_active_changeset_phases() -> None:
    source = (SCRIPTS / "deploy_sky_control_plane.sh").read_text()
    assert "assert_rnd_aws_account.sh" in source
    for phase in (
        "core",
        "enable",
        "prepare-foundation",
        "inspect-foundation",
        "execute-foundation",
        "prepare-active",
        "inspect-active",
        "execute-active",
    ):
        assert phase in source
    assert "EnableCampaignController=false" in source
    assert "EnableSkyPilotSupport=false" in source
    assert "EnableSkyPilotSupport=true" in source
    assert "EnableSkyMustStartCancel=false" in source
    assert "EnableSkyMustStartObserve=false" in source
    assert 'ENABLE_MUST_START_CANCEL:-false' in source
    assert 'ENABLE_MUST_START_OBSERVE:-false' in source
    assert "MUST_START_OBSERVATION_APPROVED" not in source
    assert "--no-execute-changeset" in source
    assert "cloudformation execute-change-set" in source
    assert "validate_must_start_activation.py" in source
    assert "CAPABILITY_NAMED_IAM" in source
    assert "assert_sns_email_confirmed.sh" in source
    assert "PurchaseCapacityBlock" not in source


def test_control_plane_has_only_exact_disabled_worker_v2_authority() -> None:
    source = (SCRIPTS / "deploy_sky_control_plane.sh").read_text()
    for parameter, override in zip(
        WORKER_V2_DISABLED_PARAMETERS,
        WORKER_V2_DISABLED_OVERRIDES,
        strict=True,
    ):
        assert override in source
        assert source.count(f"ParameterKey=='{parameter}'") == 4
    for forbidden in (
        "prepare-worker-v2",
        "execute-worker-v2",
        "WORKER_START_V2_CODE_ZIP",
        "WORKER_START_V2_CODE_S3_KEY",
        "SkyWorkerStartV2CodeS3Key",
    ):
        assert forbidden not in source


def test_must_start_coordinator_can_be_packaged_and_deployed_without_rebuilding_authorities() -> None:
    deploy = (SCRIPTS / "deploy_sky_control_plane.sh").read_text()
    package = SCRIPTS / "package_sky_must_start_cancel_lambda.sh"
    assert package.is_file()
    assert "MUST_START_CANCEL_CODE_S3_KEY" not in deploy
    assert "SkyMustStartCancelCodeS3Key" not in deploy
    assert "MUST_START_CANCEL_CODE_ZIP" in deploy
    assert "MUST_START_CANCEL_CODE_SHA256" in deploy
    assert "MUST_START_CANCEL_CODE_VERSION_ID" in deploy
    assert "SkyMustStartCancelCodeSha256" in deploy
    assert "SkyMustStartCancelCodeVersionId" in deploy
    assert "SkyMustStartSubmissionKey" not in deploy
    assert "SkyMustStartDescriptorRelativeKey" in deploy
    assert "SkyMustStartSubmissionBodySha256" in deploy
    assert "SkyMustStartTargetJobId" in deploy
    assert "MUST_START_TARGET_JOB_ID" in deploy
    assert "SkyMustStartManagedMode" in deploy
    assert "SkyMustStartJobName" in deploy
    assert "SkyMustStartBy" in deploy
    assert "SkyMustStartScheduleAt" not in deploy
    assert "SkyMustStartDescriptorFileSha256" in deploy
    assert "SkyMustStartSubmissionSubmittedAt" in deploy
    assert "SkyMustStartControllerInstanceId" in deploy
    assert "SkyMustStartControllerInstanceType" in deploy
    assert "SkyMustStartControllerProfileArn" in deploy
    assert "SkyMustStartControllerClusterName" in deploy
    assert "build_sky_repository_tar.sh" not in deploy
    assert "build_sky_campaign_descriptor.py" not in deploy


def test_scheduler_primary_wake_is_one_minute_before_canonical_deadline() -> None:
    helper = SCRIPTS / "derive_must_start_schedule_at.py"
    valid = subprocess.run(
        ["python3", str(helper), "2026-07-26T14:27:42Z"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert valid.returncode == 0
    assert valid.stdout.strip() == "2026-07-26T14:26:42Z"
    for invalid in (
        "2026-07-26T14:27:42.000000Z",
        "2026-07-26T07:27:42-07:00",
        "2026-07-26T14:27Z",
        "2026-02-30T14:27:42Z",
    ):
        result = subprocess.run(
            ["python3", str(helper), invalid],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
    deploy = (SCRIPTS / "deploy_sky_control_plane.sh").read_text()
    assert "derive_must_start_schedule_at.py" in deploy
    assert "MUST_START_SCHEDULE_AT:-" not in deploy
    assert "MUST_START_PRIMARY_WAKE_AT:-" not in deploy
    assert '"SkyMustStartPrimaryWakeAt=$PRIMARY_WAKE_AT"' in deploy


def test_activation_proof_authenticates_exact_binding_and_observation(
    tmp_path: Path,
) -> None:
    binding, observation = _activation_records()
    binding_path = tmp_path / "JOB_BINDING.json"
    observation_path = tmp_path / "observation.json"
    binding_path.write_text(json.dumps(binding))
    observation_path.write_text(json.dumps(observation))
    expected_key = (
        f"campaigns/{RUN_ID}/monitor/must-start/{MODE}/"
        f"{SUBMISSION_BODY_SHA}/observations/"
        f"{observation['observation_body_sha256']}.json"
    )

    binding_result = subprocess.run(
        _activation_args("observation-key", binding_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert binding_result.returncode == 0, binding_result.stderr
    assert binding_result.stdout.strip() == expected_key

    validation = subprocess.run(
        _activation_args("validate", binding_path, observation_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validation.returncode == 0, validation.stderr
    assert json.loads(validation.stdout) == {
        "job_binding_body_sha256": binding["job_binding_body_sha256"],
        "observation_body_sha256": observation["observation_body_sha256"],
        "observation_key": expected_key,
    }


def test_activation_proof_rejects_foreign_observation_and_mixed_run_key(
    tmp_path: Path,
) -> None:
    binding, observation = _activation_records()
    binding_path = tmp_path / "JOB_BINDING.json"
    observation_path = tmp_path / "observation.json"
    binding_path.write_text(json.dumps(binding))
    foreign = dict(observation)
    foreign["observation_body_sha256"] = "d" * 64
    observation_path.write_text(json.dumps(foreign))

    wrong_observation = subprocess.run(
        _activation_args("validate", binding_path, observation_path),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert wrong_observation.returncode != 0

    mixed_prefix = subprocess.run(
        _activation_args(
            "observation-key",
            binding_path,
            descriptor_key=(
                "campaigns/other-run/submissions/seed/"
                "campaign-descriptor-v2.json"
            ),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert mixed_prefix.returncode != 0


def test_active_prepare_authenticates_proof_and_only_creates_changeset(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = _aws_calls(log_path)
    prepare = next(
        call for call in calls if call[:2] == ["cloudformation", "deploy"]
    )
    assert "--no-execute-changeset" in prepare
    assert "--fail-on-empty-changeset" in prepare
    assert "--no-fail-on-empty-changeset" not in prepare
    assert "--changeset-name" not in prepare
    assert "--change-set-name" not in prepare
    assert not any(
        call[:2] == ["cloudformation", "execute-change-set"]
        for call in calls
    )
    assert sum(
        call[:2] == ["s3api", "get-object"] for call in calls
    ) == 4
    code_get = next(
        call
        for call in calls
        if call[:2] == ["s3api", "get-object"]
        and call[call.index("--key") + 1]
        == environment["FAKE_CODE_KEY"]
    )
    code_head = next(
        call
        for call in calls
        if call[:2] == ["s3api", "head-object"]
        and call[call.index("--key") + 1]
        == environment["FAKE_CODE_KEY"]
    )
    for call in (code_head, code_get):
        assert call[call.index("--version-id") + 1] == CODE_VERSION_ID
    overrides = prepare[prepare.index("--parameter-overrides") + 1 :]
    assert (
        f"SkyMustStartCancelCodeVersionId={CODE_VERSION_ID}" in overrides
    )
    expected_template_sha = hashlib.sha256(
        (ROOT / "aws/glm52-gpu/cfn/gpu-teacher-stack.yaml").read_bytes()
    ).hexdigest()
    handoff_path = Path(environment["MUST_START_CHANGE_SET_HANDOFF_FILE"])
    assert json.loads(handoff_path.read_text()) == {
        "account_id": "246813579024",
        "change_set_arn": ACTIVE_CHANGE_SET_ARN,
        "change_set_type": "UPDATE",
        "kind": "active",
        "region": "us-west-2",
        "schema_version": 2,
        "stack_id": STACK_ID,
        "stack_name": "keep-glm52-gpu",
        "stack_status": "UPDATE_COMPLETE",
            "template_projection_sha256": (
                "9e57301564efb325f16caecdb4d8a5f"
                "f372a7a306e823e3d4ac87aa502c7ad69"
        ),
        "template_sha256": expected_template_sha,
        "template_upload_bucket": BUCKET,
        "template_upload_key": TEMPLATE_UPLOAD_KEY,
    }
    assert handoff_path.read_text().endswith("\n")
    assert f"MUST_START_CHANGE_SET_ID={ACTIVE_CHANGE_SET_ARN}" in result.stdout
    assert (
        f"MUST_START_CHANGE_SET_TEMPLATE_SHA256={expected_template_sha}"
        in result.stdout
    )
    exact_change_set_calls = [
        call
        for call in calls
        if call[:2] in (
            ["cloudformation", "describe-change-set"],
            ["cloudformation", "get-template"],
        )
    ]
    assert exact_change_set_calls
    assert all(
        call[call.index("--change-set-name") + 1]
        == ACTIVE_CHANGE_SET_ARN
        for call in exact_change_set_calls
    )
    deploy_index = calls.index(prepare)
    stack_id_call_indexes = [
        index
        for index, call in enumerate(calls)
        if call[:2] == ["cloudformation", "describe-stacks"]
        and call[call.index("--query") + 1]
        == "Stacks[0].[StackId,StackStatus]"
    ]
    assert any(index < deploy_index for index in stack_id_call_indexes)
    assert any(index > deploy_index for index in stack_id_call_indexes)
    assert not any(
        call[:2] == ["cloudformation", "describe-stacks"]
        and call[call.index("--query") + 1] == "Stacks[0].StackId"
        for call in calls
    )
    identity_call = next(
        call
        for call in calls
        if call[:2] == ["cloudformation", "describe-change-set"]
        and "--output" in call
        and call[call.index("--output") + 1] == "json"
    )
    identity_query = identity_call[identity_call.index("--query") + 1]
    assert "ChangeSetType" not in identity_query
    for field in (
        "ChangeSetId",
        "StackId",
        "StackName",
        "Status",
        "ExecutionStatus",
    ):
        assert field in identity_query


def test_active_code_refresh_prepare_and_execute_accept_exact_active_state(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    environment["FAKE_CURRENT_CANCEL"] = "true"

    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr

    executed = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "execute-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert executed.returncode == 0, executed.stderr
    calls = _aws_calls(log_path)
    assert sum(
        call[:2] == ["cloudformation", "deploy"] for call in calls
    ) == 1
    assert sum(
        call[:2] == ["cloudformation", "execute-change-set"]
        for call in calls
    ) == 1


def test_active_code_refresh_rejects_flag_or_authority_drift(
    tmp_path: Path,
) -> None:
    drifts = (
        ("FAKE_CURRENT_SUPPORT", "false"),
        ("FAKE_CURRENT_OBSERVE", "false"),
        ("FAKE_CURRENT_CANCEL", "None"),
        ("FAKE_CURRENT_RUN_ID", "glm52-sky-foreign"),
        (
            "FAKE_CURRENT_DESCRIPTOR_KEY",
            "campaigns/glm52-sky-foreign/campaign-descriptor-v2.json",
        ),
        (
            "FAKE_CURRENT_WATCHDOG_CODE_S3_KEY",
            "lambda/sky-watchdog/foreign.zip",
        ),
        ("FAKE_FOUNDATION_REVISION", "None"),
    )
    for index, (field, value) in enumerate(drifts):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _active_environment(case_path)
        environment["FAKE_CURRENT_CANCEL"] = "true"
        environment[field] = value

        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode != 0, field
        assert "exact observed Sky authority" in result.stderr
        assert not any(
            call[:2]
            in (
                ["s3api", "get-bucket-versioning"],
                ["s3api", "head-object"],
                ["s3api", "get-object"],
                ["cloudformation", "deploy"],
                ["cloudformation", "execute-change-set"],
            )
            for call in _aws_calls(log_path)
        )


def test_pinned_fake_aws_cli_deploy_help_has_no_caller_changeset_name(
    tmp_path: Path,
) -> None:
    fake_aws = tmp_path / "aws"
    fake_aws.write_text(
        "#!/bin/sh\n"
        'test "$1 $2 $3" = "cloudformation deploy help" || exit 64\n'
        "printf '%s\\n' --no-execute-changeset --fail-on-empty-changeset\n"
    )
    fake_aws.chmod(0o755)
    result = subprocess.run(
        [str(fake_aws), "cloudformation", "deploy", "help"],
        cwd=ROOT,
        env={"AWS_PAGER": ""},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rendered = result.stdout.replace("\b", "")
    assert "--no-execute-changeset" in rendered
    assert "--fail-on-empty-changeset" in rendered
    assert "--changeset-name" not in rendered
    assert "--change-set-name" not in rendered


def test_prepare_rejects_missing_malformed_duplicate_or_legacy_change_set_id(
    tmp_path: Path,
) -> None:
    outputs = (
        "Changeset created successfully without review command\n",
        _deploy_change_set_output("glm52-must-start-job3-active"),
        _deploy_change_set_output(ACTIVE_CHANGE_SET_ARN)
        + _deploy_change_set_output(ACTIVE_CHANGE_SET_ARN),
        _deploy_change_set_output(ACTIVE_CHANGE_SET_ARN).rstrip()
        + " --output json\n",
    )
    for index, output in enumerate(outputs):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _active_environment(case_path)
        environment["FAKE_DEPLOY_OUTPUT"] = output
        environment["MUST_START_CHANGE_SET_NAME"] = (
            "glm52-must-start-job3-active"
        )
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "exactly one generated change-set ARN" in result.stderr
        assert output.strip() in result.stdout
        assert not Path(
            environment["MUST_START_CHANGE_SET_HANDOFF_FILE"]
        ).exists()
        calls = _aws_calls(log_path)
        assert not any(
            call[:2] in (
                ["cloudformation", "describe-change-set"],
                ["cloudformation", "get-template"],
                ["cloudformation", "execute-change-set"],
            )
            for call in calls
        )


def test_prepare_rejects_foreign_or_non_generated_change_set_arns(
    tmp_path: Path,
) -> None:
    invalid_arns = (
        ACTIVE_CHANGE_SET_ARN.replace(":us-west-2:", ":us-east-1:"),
        ACTIVE_CHANGE_SET_ARN.replace(
            ":246813579024:", ":111122223333:"
        ),
        ACTIVE_CHANGE_SET_ARN.replace(
            "awscli-cloudformation-package-deploy-1785000001",
            "operator-chosen-name",
        ),
        ACTIVE_CHANGE_SET_ARN.rsplit("/", 1)[0] + "/not-a-uuid",
    )
    for index, invalid_arn in enumerate(invalid_arns):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _active_environment(case_path)
        environment["FAKE_DEPLOY_OUTPUT"] = _deploy_change_set_output(
            invalid_arn
        )
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "exactly one generated change-set ARN" in result.stderr
        assert invalid_arn in result.stdout
        assert not Path(
            environment["MUST_START_CHANGE_SET_HANDOFF_FILE"]
        ).exists()
        assert not any(
            call[:2] == ["cloudformation", "describe-change-set"]
            for call in _aws_calls(log_path)
        )


def test_prepare_rejects_wrong_server_change_set_identity(
    tmp_path: Path,
) -> None:
    identity_drifts = (
        (
            "FAKE_RETURNED_CHANGE_SET_ID",
            FOUNDATION_CHANGE_SET_ARN,
        ),
        (
            "FAKE_RETURNED_STACK_ID",
            STACK_ID.replace(
                "11111111-2222-3333-4444-555555555555",
                "99999999-2222-3333-4444-555555555555",
            ),
        ),
        ("FAKE_RETURNED_STACK_NAME", "foreign-stack"),
        ("FAKE_RETURNED_CHANGE_SET_STATUS", "FAILED"),
        ("FAKE_RETURNED_EXECUTION_STATUS", "UNAVAILABLE"),
    )
    for index, (name, value) in enumerate(identity_drifts):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _active_environment(case_path)
        environment[name] = value
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "change-set identity does not match" in result.stderr
        assert ACTIVE_CHANGE_SET_ARN in result.stdout
        assert not Path(
            environment["MUST_START_CHANGE_SET_HANDOFF_FILE"]
        ).exists()
        assert not any(
            call[:2] in (
                ["cloudformation", "get-template"],
                ["cloudformation", "execute-change-set"],
            )
            for call in _aws_calls(log_path)
        )


def test_prepare_rejects_missing_or_null_real_change_set_identity_fields(
    tmp_path: Path,
) -> None:
    required_fields = (
        "ChangeSetId",
        "StackId",
        "StackName",
        "Status",
        "ExecutionStatus",
    )
    for index, (mode, field) in enumerate(
        (mode, field)
        for mode in ("omit", "null")
        for field in required_fields
    ):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _active_environment(case_path)
        environment[
            (
                "FAKE_IDENTITY_OMIT_FIELD"
                if mode == "omit"
                else "FAKE_IDENTITY_NULL_FIELD"
            )
        ] = field
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "change-set identity does not match" in result.stderr
        assert ACTIVE_CHANGE_SET_ARN in result.stdout
        assert not Path(
            environment["MUST_START_CHANGE_SET_HANDOFF_FILE"]
        ).exists()
        assert not any(
            call[:2] in (
                ["cloudformation", "get-template"],
                ["cloudformation", "execute-change-set"],
            )
            for call in _aws_calls(log_path)
        )


def test_prepare_rejects_same_name_stack_recreation_during_create(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    replacement_stack_id = STACK_ID.replace(
        "11111111-2222-3333-4444-555555555555",
        "99999999-2222-3333-4444-555555555555",
    )
    environment["FAKE_CURRENT_STACK_ID_AFTER_DEPLOY"] = replacement_stack_id
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "stack identity changed while preparing change set" in result.stderr
    assert ACTIVE_CHANGE_SET_ARN in result.stdout
    assert not Path(
        environment["MUST_START_CHANGE_SET_HANDOFF_FILE"]
    ).exists()
    calls = _aws_calls(log_path)
    assert not any(
        call[:2] in (
            ["cloudformation", "describe-change-set"],
            ["cloudformation", "get-template"],
            ["cloudformation", "execute-change-set"],
        )
        for call in calls
    )


def test_prepare_rejects_review_in_progress_same_stack_id_before_deploy(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    environment["FAKE_CURRENT_STACK_STATUS"] = "REVIEW_IN_PROGRESS"
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "exact existing UPDATE_COMPLETE stack" in result.stderr
    assert not Path(
        environment["MUST_START_CHANGE_SET_HANDOFF_FILE"]
    ).exists()
    assert not any(
        call[:2] in (
            ["cloudformation", "deploy"],
            ["cloudformation", "describe-change-set"],
            ["cloudformation", "get-template"],
            ["cloudformation", "execute-change-set"],
        )
        for call in _aws_calls(log_path)
    )


def test_prepare_rejects_same_stack_id_status_drift_after_create(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    environment[
        "FAKE_CURRENT_STACK_STATUS_AFTER_DEPLOY"
    ] = "REVIEW_IN_PROGRESS"
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "exact existing UPDATE_COMPLETE stack" in result.stderr
    assert ACTIVE_CHANGE_SET_ARN in result.stdout
    assert not Path(
        environment["MUST_START_CHANGE_SET_HANDOFF_FILE"]
    ).exists()
    assert not any(
        call[:2] in (
            ["cloudformation", "describe-change-set"],
            ["cloudformation", "get-template"],
            ["cloudformation", "execute-change-set"],
        )
        for call in _aws_calls(log_path)
    )


def test_inspect_and_execute_reject_review_in_progress_same_stack_id(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    environment["FAKE_CURRENT_STACK_STATUS"] = "REVIEW_IN_PROGRESS"
    for mode in ("inspect-active", "execute-active"):
        before = len(_aws_calls(log_path))
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), mode],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "exact existing UPDATE_COMPLETE stack" in result.stderr
        assert not any(
            call[:2] in (
                ["cloudformation", "describe-change-set"],
                ["cloudformation", "get-template"],
                ["cloudformation", "execute-change-set"],
            )
            for call in _aws_calls(log_path)[before:]
        )


def test_prepare_rejects_change_set_template_drift_and_preserves_cli_output(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    environment["FAKE_CHANGE_SET_TEMPLATE_BODY"] = (
        "AWSTemplateFormatVersion: '2010-09-09'\nResources: {}\n"
    )
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "change-set template does not match frozen server projection" in result.stderr
    assert ACTIVE_CHANGE_SET_ARN in result.stdout
    assert not Path(
        environment["MUST_START_CHANGE_SET_HANDOFF_FILE"]
    ).exists()
    assert not any(
        call[:2] == ["cloudformation", "execute-change-set"]
        for call in _aws_calls(log_path)
    )


def test_prepare_rejects_uploaded_template_object_drift_before_handoff(
    tmp_path: Path,
) -> None:
    environment, log_path = _foundation_environment(tmp_path)
    drifted_template = tmp_path / "drifted-upload.template"
    drifted_template.write_bytes(b"unrelated uploaded template bytes")
    environment["FAKE_REMOTE_TEMPLATE"] = str(drifted_template)

    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "uploaded CloudFormation template object does not match frozen local template"
        in result.stderr
    )
    assert FOUNDATION_CHANGE_SET_ARN in result.stdout
    assert not Path(
        environment["MUST_START_FOUNDATION_CHANGE_SET_HANDOFF_FILE"]
    ).exists()
    template_calls = [
        call
        for call in _aws_calls(log_path)
        if call[:2] in (
            ["s3api", "head-object"],
            ["s3api", "get-object"],
        )
        and call[call.index("--key") + 1] == TEMPLATE_UPLOAD_KEY
    ]
    assert [call[:2] for call in template_calls] == [
        ["s3api", "head-object"],
        ["s3api", "get-object"],
    ]


def test_execute_rejects_uploaded_template_object_drift_before_execution(
    tmp_path: Path,
) -> None:
    environment, log_path = _foundation_environment(tmp_path)
    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    drifted_template = tmp_path / "drifted-before-execute.template"
    drifted_template.write_bytes(b"drifted after handoff")
    environment["FAKE_REMOTE_TEMPLATE"] = str(drifted_template)
    before = len(_aws_calls(log_path))

    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "execute-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "uploaded CloudFormation template object does not match frozen local template"
        in result.stderr
    )
    assert not any(
        call[:2] == ["cloudformation", "execute-change-set"]
        for call in _aws_calls(log_path)[before:]
    )


def test_prepare_refuses_unsafe_handoff_target_before_cloudformation_deploy(
    tmp_path: Path,
) -> None:
    for index, target_kind in enumerate(("relative", "existing", "symlink")):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _active_environment(case_path)
        if target_kind == "relative":
            environment["MUST_START_CHANGE_SET_HANDOFF_FILE"] = (
                "active-change-set.json"
            )
        elif target_kind == "existing":
            handoff = Path(
                environment["MUST_START_CHANGE_SET_HANDOFF_FILE"]
            )
            handoff.write_text("must not be overwritten\n")
        else:
            handoff = Path(
                environment["MUST_START_CHANGE_SET_HANDOFF_FILE"]
            )
            handoff.symlink_to(case_path / "elsewhere.json")
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "unused absolute non-symlink path" in result.stderr
        assert not any(
            call[:2] == ["cloudformation", "deploy"]
            for call in _aws_calls(log_path)
        )


def test_observe_only_deploy_passes_exact_code_version_without_active_proof(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "enable"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = _aws_calls(log_path)
    deploy = next(
        call for call in calls if call[:2] == ["cloudformation", "deploy"]
    )
    overrides = deploy[deploy.index("--parameter-overrides") + 1 :]
    for exact in (
        "EnableSkyMustStartObserve=true",
        "EnableSkyMustStartCancel=false",
        f"SkyMustStartCancelCodeSha256={environment['FAKE_CODE_SHA']}",
        f"SkyMustStartCancelCodeVersionId={CODE_VERSION_ID}",
    ):
        assert exact in overrides
    assert not any(
        value.startswith("SkyMustStartActivation")
        for value in overrides
    )
    code_calls = [
        call
        for call in calls
        if call[:2] in (
            ["s3api", "head-object"],
            ["s3api", "get-object"],
        )
        and "--key" in call
        and call[call.index("--key") + 1] == environment["FAKE_CODE_KEY"]
    ]
    assert len(code_calls) == 2
    assert all(
        call[call.index("--version-id") + 1] == CODE_VERSION_ID
        for call in code_calls
    )


def test_active_inspect_then_explicit_execute_never_directly_deploys(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    before_inspect = len(_aws_calls(log_path))
    inspect = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "inspect-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert inspect.returncode == 0, inspect.stderr
    inspect_calls = _aws_calls(log_path)[before_inspect:]
    assert any(
        call[:2] == ["cloudformation", "get-template"]
        for call in inspect_calls
    )
    assert all(
        call[call.index("--change-set-name") + 1]
        == ACTIVE_CHANGE_SET_ARN
        for call in inspect_calls
        if call[:2] in (
            ["cloudformation", "describe-change-set"],
            ["cloudformation", "get-template"],
        )
    )
    before_execute = len(_aws_calls(log_path))

    no_approval_environment = dict(environment)
    no_approval_environment.pop("MUST_START_EXECUTE_APPROVAL")
    refused = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "execute-active"],
        cwd=ROOT,
        env=no_approval_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode != 0
    assert len(_aws_calls(log_path)) == before_execute + 1

    executed = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "execute-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert executed.returncode == 0, executed.stderr
    execute_calls = _aws_calls(log_path)[before_execute + 1 :]
    assert any(
        call[:2] == ["cloudformation", "describe-change-set"]
        for call in execute_calls
    )
    assert any(
        call[:2] == ["cloudformation", "execute-change-set"]
        for call in execute_calls
    )
    assert all(
        call[call.index("--change-set-name") + 1]
        == ACTIVE_CHANGE_SET_ARN
        for call in execute_calls
        if call[:2] in (
            ["cloudformation", "describe-change-set"],
            ["cloudformation", "get-template"],
            ["cloudformation", "execute-change-set"],
        )
    )
    assert not any(
        call[:2] == ["cloudformation", "deploy"]
        for call in execute_calls
    )


def test_inspect_and_execute_reject_server_template_drift(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    drifted = dict(environment)
    drifted["FAKE_CHANGE_SET_TEMPLATE_BODY"] = (
        "AWSTemplateFormatVersion: '2010-09-09'\nResources: {}\n"
    )
    for mode in ("inspect-active", "execute-active"):
        before = len(_aws_calls(log_path))
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), mode],
            cwd=ROOT,
            env=drifted,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert (
            "change-set template does not match frozen server projection"
            in result.stderr
        )
        calls = _aws_calls(log_path)[before:]
        assert any(
            call[:2] == ["cloudformation", "get-template"]
            for call in calls
        )
        assert not any(
            call[:2] == ["cloudformation", "execute-change-set"]
            for call in calls
        )


def test_inspect_rejects_tampered_template_pin_before_change_set_read(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    handoff_path = Path(environment["MUST_START_CHANGE_SET_HANDOFF_FILE"])
    handoff = json.loads(handoff_path.read_text())
    handoff["template_sha256"] = "0" * 64
    handoff_path.write_text(
        json.dumps(handoff, sort_keys=True, separators=(",", ":")) + "\n"
    )
    before = len(_aws_calls(log_path))
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "inspect-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "local template no longer matches pinned template SHA-256" in result.stderr
    assert not any(
        call[:2] in (
            ["cloudformation", "describe-change-set"],
            ["cloudformation", "get-template"],
        )
        for call in _aws_calls(log_path)[before:]
    )


def test_inspect_and_execute_reject_wrong_kind_handoff_before_execution(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    handoff_path = Path(environment["MUST_START_CHANGE_SET_HANDOFF_FILE"])
    handoff = json.loads(handoff_path.read_text())
    handoff["kind"] = "foundation"
    handoff_path.write_text(
        json.dumps(handoff, sort_keys=True, separators=(",", ":")) + "\n"
    )
    for mode in ("inspect-active", "execute-active"):
        before = len(_aws_calls(log_path))
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), mode],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "change-set handoff is invalid" in result.stderr
        assert not any(
            call[:2] in (
                ["cloudformation", "describe-change-set"],
                ["cloudformation", "get-template"],
                ["cloudformation", "execute-change-set"],
            )
            for call in _aws_calls(log_path)[before:]
        )


def test_inspect_rejects_recreated_stack_or_wrong_returned_identity(
    tmp_path: Path,
) -> None:
    for index, drift in enumerate(("current-stack", "change-set-stack")):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _active_environment(case_path)
        prepared = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert prepared.returncode == 0, prepared.stderr
        replacement_stack_id = STACK_ID.replace(
            "11111111-2222-3333-4444-555555555555",
            "99999999-2222-3333-4444-555555555555",
        )
        if drift == "current-stack":
            environment["FAKE_CURRENT_STACK_ID"] = replacement_stack_id
        else:
            environment["FAKE_RETURNED_STACK_ID"] = replacement_stack_id
        before = len(_aws_calls(log_path))
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "inspect-active"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert (
            "stack identity no longer matches prepared change set"
            in result.stderr
            or "change-set identity does not match" in result.stderr
        )
        assert not any(
            call[:2] == ["cloudformation", "execute-change-set"]
            for call in _aws_calls(log_path)[before:]
        )


def test_legacy_change_set_name_cannot_replace_exact_handoff(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    environment["MUST_START_CHANGE_SET_NAME"] = (
        "glm52-must-start-job3-active"
    )
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "inspect-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "MUST_START_CHANGE_SET_HANDOFF_FILE" in result.stderr
    assert not any(
        call[:2] == ["cloudformation", "describe-change-set"]
        for call in _aws_calls(log_path)
    )


def test_code_object_hash_drift_stops_before_cloudformation_mutation(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(
        tmp_path,
        remote_code=b"foreign package bytes",
    )
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "S3 code object SHA-256 mismatch" in result.stderr
    calls = _aws_calls(log_path)
    assert any(
        call[:2] == ["cloudformation", "describe-stacks"] for call in calls
    )
    assert not any(
        call[:2] in (
            ["cloudformation", "deploy"],
            ["cloudformation", "execute-change-set"],
        )
        for call in calls
    )


def test_missing_code_version_stops_before_s3_or_cloudformation(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    environment.pop("MUST_START_CANCEL_CODE_VERSION_ID")
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "MUST_START_CANCEL_CODE_VERSION_ID" in result.stderr
    assert not any(
        call[0] in {"s3api", "cloudformation"}
        for call in _aws_calls(log_path)
    )


def test_invalid_code_versions_stop_before_s3_or_cloudformation(
    tmp_path: Path,
) -> None:
    for index, invalid in enumerate(
        (
            "disabled",
            "null",
            "None",
            "../version",
            "version/../id",
            "bad version",
            "*",
        )
    ):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _active_environment(case_path)
        environment["MUST_START_CANCEL_CODE_VERSION_ID"] = invalid
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "exact S3 object VersionId" in result.stderr
        assert not any(
            call[0] in {"s3api", "cloudformation"}
            for call in _aws_calls(log_path)
        )


def test_existing_deployment_modes_bind_exact_worker_v2_disabled_authority(
    tmp_path: Path,
) -> None:
    modes = (
        ("core", False),
        ("enable", False),
        ("prepare-foundation", True),
        ("prepare-active", False),
    )
    for index, (mode, legacy_worker_state) in enumerate(modes):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        if legacy_worker_state:
            environment, log_path = _foundation_environment(case_path)
        else:
            environment, log_path = _active_environment(case_path)
        if mode == "core":
            environment["FAKE_STACK_EXISTS"] = "false"
        for parameter, value in WORKER_V2_DISABLED_PARAMETERS.items():
            environment[parameter] = (
                "true" if value == "false" else f"preloaded-{value}"
            )

        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), mode],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, (mode, result.stderr)
        calls = _aws_calls(log_path)
        deploy = next(
            call
            for call in calls
            if call[:2] == ["cloudformation", "deploy"]
        )
        overrides = deploy[deploy.index("--parameter-overrides") + 1 :]
        assert [
            value
            for value in overrides
            if value.split("=", 1)[0] in WORKER_V2_DISABLED_PARAMETERS
        ] == list(WORKER_V2_DISABLED_OVERRIDES)
        assert not any(
            call[:2] in (
                ["s3api", "head-object"],
                ["s3api", "get-object"],
            )
            and "worker-start-v2"
            in call[call.index("--key") + 1].lower()
            for call in calls
        )


def test_worker_v2_consumers_reject_every_nondefault_current_authority(
    tmp_path: Path,
) -> None:
    for index, (parameter, environment_name) in enumerate(
        WORKER_V2_CURRENT_ENV.items()
    ):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _active_environment(case_path)
        if parameter == "EnableSkyWorkerStartV2Coordinator":
            environment[environment_name] = "true"
        elif parameter == "SkyWorkerStartV2FoundationRevision":
            environment[environment_name] = "None"
        else:
            environment[environment_name] = "preloaded"

        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode != 0, parameter
        assert "reviewed must-start foundation is not installed" in result.stderr
        assert not any(
            call[:2] in (
                ["s3api", "head-object"],
                ["s3api", "get-object"],
                ["cloudformation", "deploy"],
                ["cloudformation", "execute-change-set"],
            )
            for call in _aws_calls(log_path)
        )


def test_every_worker_v2_consumer_rejects_pre_task_3o_foundation(
    tmp_path: Path,
) -> None:
    for index, mode in enumerate(("enable", "prepare-active")):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _active_environment(case_path)
        for environment_name in WORKER_V2_CURRENT_ENV.values():
            environment[environment_name] = "None"

        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), mode],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode != 0, mode
        assert "reviewed must-start foundation is not installed" in result.stderr
        assert not any(
            call[:2] in (
                ["s3api", "head-object"],
                ["s3api", "get-object"],
                ["cloudformation", "deploy"],
                ["cloudformation", "execute-change-set"],
            )
            for call in _aws_calls(log_path)
        )

    execute_path = tmp_path / "execute"
    execute_path.mkdir()
    environment, log_path = _active_environment(execute_path)
    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    for environment_name in WORKER_V2_CURRENT_ENV.values():
        environment[environment_name] = "None"
    before = len(_aws_calls(log_path))
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "execute-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "reviewed must-start foundation is not installed" in result.stderr
    assert not any(
        call[:2]
        in (
            ["s3api", "head-object"],
            ["s3api", "get-object"],
            ["cloudformation", "execute-change-set"],
        )
        for call in _aws_calls(log_path)[before:]
    )


def test_worker_v2_current_state_rejects_missing_extra_or_reordered_fields(
    tmp_path: Path,
) -> None:
    drifts = (
        ("FAKE_CURRENT_WORKER_START_V2_CODE_SHA256", ""),
        (
            "FAKE_CURRENT_WORKER_START_V2_CODE_VERSION_ID",
            "disabled\textra",
        ),
        ("FAKE_CURRENT_WORKER_START_V2_ENABLE", "versioned-code-v1"),
    )
    for index, (name, value) in enumerate(drifts):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _active_environment(case_path)
        environment[name] = value
        if index == 2:
            environment[
                "FAKE_CURRENT_WORKER_START_V2_FOUNDATION_REVISION"
            ] = "false"
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "reviewed must-start foundation is not installed" in result.stderr
        assert not any(
            call[:2] == ["cloudformation", "deploy"]
            for call in _aws_calls(log_path)
        )


def test_coordinator_consumers_require_installed_foundation_before_code_or_mutation(
    tmp_path: Path,
) -> None:
    for index, mode in enumerate(("enable", "prepare-active", "execute-active")):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _active_environment(case_path)
        environment["FAKE_FOUNDATION_REVISION"] = "None"
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), mode],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "reviewed must-start foundation is not installed" in result.stderr
        calls = _aws_calls(log_path)
        assert not any(
            call[:2] in (
                ["s3api", "head-object"],
                ["s3api", "get-object"],
                ["cloudformation", "deploy"],
                ["cloudformation", "execute-change-set"],
            )
            for call in calls
        )


def test_coordinator_consumers_require_enabled_bucket_versioning(
    tmp_path: Path,
) -> None:
    for index, mode in enumerate(("enable", "prepare-active", "execute-active")):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _active_environment(case_path)
        environment["FAKE_BUCKET_VERSIONING_STATUS"] = "None"
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), mode],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "model bucket versioning must be Enabled" in result.stderr
        calls = _aws_calls(log_path)
        assert any(
            call[:2] == ["s3api", "get-bucket-versioning"]
            for call in calls
        )
        assert not any(
            call[:2] in (
                ["s3api", "head-object"],
                ["s3api", "get-object"],
                ["cloudformation", "deploy"],
                ["cloudformation", "execute-change-set"],
            )
            for call in calls
        )


def test_enable_without_observe_cannot_apply_pending_foundation(
    tmp_path: Path,
) -> None:
    environment, log_path = _foundation_environment(tmp_path)
    environment["ENABLE_MUST_START_OBSERVE"] = "false"
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "enable"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "reviewed must-start foundation is not installed" in result.stderr
    assert not any(
        call[:2] == ["cloudformation", "deploy"]
        for call in _aws_calls(log_path)
    )


def test_core_refuses_existing_stack_before_deploy(tmp_path: Path) -> None:
    environment, log_path = _active_environment(tmp_path)
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "core"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "core is creation-only" in result.stderr
    assert not any(
        call[:2] == ["cloudformation", "deploy"]
        for call in _aws_calls(log_path)
    )


def test_core_allows_strongly_proved_new_stack_and_installs_foundation(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    environment["FAKE_STACK_EXISTS"] = "false"
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "core"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = _aws_calls(log_path)
    first_describe = next(
        index
        for index, call in enumerate(calls)
        if call[:2] == ["cloudformation", "describe-stacks"]
    )
    deploy_index = next(
        index
        for index, call in enumerate(calls)
        if call[:2] == ["cloudformation", "deploy"]
    )
    assert first_describe < deploy_index
    deploy = calls[deploy_index]
    overrides = deploy[deploy.index("--parameter-overrides") + 1 :]
    for exact in (
        "EnableSkyPilotSupport=false",
        "EnableSkyMustStartObserve=false",
        "EnableSkyMustStartCancel=false",
        "SkyMustStartFoundationRevision=versioned-code-v1",
    ):
        assert exact in overrides


def test_new_stack_core_can_then_establish_sky_authority_without_observation(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    environment["FAKE_STACK_EXISTS"] = "false"
    core = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "core"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert core.returncode == 0, core.stderr

    _set_post_core_state(environment)
    environment["ENABLE_MUST_START_OBSERVE"] = "false"
    enabled = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "enable"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert enabled.returncode == 0, enabled.stderr
    calls = _aws_calls(log_path)
    deploys = [
        call
        for call in calls
        if call[:2] == ["cloudformation", "deploy"]
    ]
    assert len(deploys) == 2
    overrides = deploys[1][deploys[1].index("--parameter-overrides") + 1 :]
    for exact in (
        "EnableSkyPilotSupport=true",
        f"SkyCampaignRunId={RUN_ID}",
        f"SkyCampaignDescriptorKey={DESCRIPTOR_KEY}",
        "SkyWatchdogCodeS3Key=lambda/sky-watchdog/exact.zip",
        "EnableSkyMustStartObserve=false",
        "EnableSkyMustStartCancel=false",
    ):
        assert exact in overrides
    assert any(
        call[:2] == ["s3api", "get-bucket-versioning"]
        for call in calls
    )
    assert not any(
        call[:2] in (
            ["s3api", "head-object"],
            ["s3api", "get-object"],
        )
        for call in calls
    )


def test_post_core_state_cannot_jump_directly_to_observation(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    _set_post_core_state(environment)
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "enable"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "exact prior Sky authority" in result.stderr
    calls = _aws_calls(log_path)
    assert not any(
        call[:2] in (
            ["s3api", "get-bucket-versioning"],
            ["s3api", "head-object"],
            ["s3api", "get-object"],
            ["cloudformation", "deploy"],
        )
        for call in calls
    )


def test_post_core_none_descriptor_is_rejected_as_missing_authority(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    _set_post_core_state(environment)
    environment["FAKE_CURRENT_DESCRIPTOR_KEY"] = "None"
    environment["ENABLE_MUST_START_OBSERVE"] = "false"
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "enable"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "exact prior Sky authority" in result.stderr
    calls = _aws_calls(log_path)
    assert not any(
        call[:2] in (
            ["s3api", "get-bucket-versioning"],
            ["s3api", "head-object"],
            ["s3api", "get-object"],
            ["cloudformation", "deploy"],
        )
        for call in calls
    )


def test_post_core_authority_bootstrap_rejects_mixed_default_state(
    tmp_path: Path,
) -> None:
    drifts = (
        ("FAKE_CURRENT_RUN_ID", RUN_ID),
        ("FAKE_CURRENT_DESCRIPTOR_KEY", DESCRIPTOR_KEY),
        (
            "FAKE_CURRENT_WATCHDOG_CODE_S3_KEY",
            "lambda/sky-watchdog/exact.zip",
        ),
    )
    for index, (name, value) in enumerate(drifts):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _active_environment(case_path)
        _set_post_core_state(environment)
        environment["ENABLE_MUST_START_OBSERVE"] = "false"
        environment[name] = value
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "enable"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "exact prior Sky authority" in result.stderr
        assert not any(
            call[:2] == ["cloudformation", "deploy"]
            for call in _aws_calls(log_path)
        )


def test_foundation_change_set_preserves_sky_authority_and_cannot_create_coordinator(
    tmp_path: Path,
) -> None:
    environment, log_path = _foundation_environment(tmp_path)
    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    calls = _aws_calls(log_path)
    prepare = next(
        call for call in calls if call[:2] == ["cloudformation", "deploy"]
    )
    assert "--no-execute-changeset" in prepare
    assert "--fail-on-empty-changeset" in prepare
    assert "--no-fail-on-empty-changeset" not in prepare
    assert "--changeset-name" not in prepare
    assert "--change-set-name" not in prepare
    overrides = prepare[prepare.index("--parameter-overrides") + 1 :]
    for exact in (
        "EnableSkyPilotSupport=true",
        f"SkyCampaignRunId={RUN_ID}",
        f"SkyCampaignDescriptorKey={DESCRIPTOR_KEY}",
        "SkyWatchdogCodeS3Key=lambda/sky-watchdog/exact.zip",
        "EnableSkyMustStartObserve=false",
        "EnableSkyMustStartCancel=false",
        "SkyMustStartFoundationRevision=versioned-code-v1",
    ):
        assert exact in overrides
    assert not any("SkyMustStartCancelCode" in value for value in overrides)
    template_calls = [
        call
        for call in calls
        if call[:2] in (
            ["s3api", "head-object"],
            ["s3api", "get-object"],
        )
        and call[call.index("--key") + 1] == TEMPLATE_UPLOAD_KEY
    ]
    assert [call[:2] for call in template_calls] == [
        ["s3api", "head-object"],
        ["s3api", "get-object"],
    ]
    assert any(
        call[:2] == ["cloudformation", "describe-stacks"]
        for call in calls
    )
    expected_template_sha = hashlib.sha256(
        (ROOT / "aws/glm52-gpu/cfn/gpu-teacher-stack.yaml").read_bytes()
    ).hexdigest()
    handoff_path = Path(
        environment["MUST_START_FOUNDATION_CHANGE_SET_HANDOFF_FILE"]
    )
    assert json.loads(handoff_path.read_text()) == {
        "account_id": "246813579024",
        "change_set_arn": FOUNDATION_CHANGE_SET_ARN,
        "change_set_type": "UPDATE",
        "kind": "foundation",
        "region": "us-west-2",
        "schema_version": 2,
        "stack_id": STACK_ID,
        "stack_name": "keep-glm52-gpu",
        "stack_status": "UPDATE_COMPLETE",
            "template_projection_sha256": (
                "9e57301564efb325f16caecdb4d8a5f"
                "f372a7a306e823e3d4ac87aa502c7ad69"
        ),
        "template_sha256": expected_template_sha,
        "template_upload_bucket": BUCKET,
        "template_upload_key": TEMPLATE_UPLOAD_KEY,
    }
    assert (
        f"MUST_START_FOUNDATION_CHANGE_SET_ID={FOUNDATION_CHANGE_SET_ARN}"
        in prepared.stdout
    )

    inspected = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "inspect-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert inspected.returncode == 0, inspected.stderr

    no_approval = dict(environment)
    no_approval.pop("MUST_START_FOUNDATION_EXECUTE_APPROVAL")
    refused = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "execute-foundation"],
        cwd=ROOT,
        env=no_approval,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode != 0

    executed = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "execute-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert executed.returncode == 0, executed.stderr
    execute_calls = _aws_calls(log_path)
    assert any(
        call[:2] == ["cloudformation", "describe-change-set"]
        and "--query" in call
        and call[call.index("--query") + 1] == "Changes"
        and call[call.index("--output") + 1] == "json"
        for call in execute_calls
    )
    assert any(
        call[:2] == ["cloudformation", "execute-change-set"]
        for call in execute_calls
    )


def test_foundation_execute_rechecks_every_worker_v2_change_set_parameter(
    tmp_path: Path,
) -> None:
    environment, log_path = _foundation_environment(tmp_path)
    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr

    for parameter, environment_name in WORKER_V2_CHANGE_SET_ENV.items():
        original = environment[environment_name]
        environment[environment_name] = (
            "true"
            if parameter == "EnableSkyWorkerStartV2Coordinator"
            else "preloaded"
        )
        before = len(_aws_calls(log_path))
        result = subprocess.run(
            [
                str(SCRIPTS / "deploy_sky_control_plane.sh"),
                "execute-foundation",
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        environment[environment_name] = original
        assert result.returncode != 0, parameter
        assert (
            "reviewed foundation change set no longer preserves exact disabled coordinator authority"
            in result.stderr
        )
        new_calls = _aws_calls(log_path)[before:]
        assert not any(
            call[:2] == ["cloudformation", "execute-change-set"]
            for call in new_calls
        )
        assert not any(
            call[:2] == ["cloudformation", "describe-change-set"]
            and call[call.index("--query") + 1] == "Changes"
            for call in new_calls
        )


def test_active_execute_rechecks_every_worker_v2_change_set_parameter(
    tmp_path: Path,
) -> None:
    environment, log_path = _active_environment(tmp_path)
    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-active"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr

    for parameter, environment_name in WORKER_V2_CHANGE_SET_ENV.items():
        original = environment[environment_name]
        environment[environment_name] = (
            "true"
            if parameter == "EnableSkyWorkerStartV2Coordinator"
            else "preloaded"
        )
        before = len(_aws_calls(log_path))
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "execute-active"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        environment[environment_name] = original
        assert result.returncode != 0, parameter
        assert (
            "reviewed change set no longer binds exact baseline/activation/code/worker-v2 proof"
            in result.stderr
        )
        assert not any(
            call[:2] == ["cloudformation", "execute-change-set"]
            for call in _aws_calls(log_path)[before:]
        )


def test_foundation_execute_rejects_any_causal_graph_drift(
    tmp_path: Path,
) -> None:
    environment, log_path = _foundation_environment(tmp_path)
    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    graph_path = Path(environment["FAKE_FOUNDATION_CHANGE_GRAPH"])
    pristine = _foundation_change_graph()

    def change(logical_id: str) -> dict[str, object]:
        return next(
            item["ResourceChange"]
            for item in graph
            if item["ResourceChange"]["LogicalResourceId"] == logical_id
        )

    cases = (
        ("extra-id",),
        ("missing-id",),
        ("top-level-type",),
        ("action",),
        ("remove",),
        ("resource-type",),
        ("replacement",),
        ("scope",),
        ("evaluation",),
        ("source",),
        ("causing-entity",),
        ("target",),
        ("recreation",),
        ("missing-detail",),
        ("extra-detail",),
        ("extra-deny-detail",),
        ("bucket-notification-detail",),
        ("worker-function-add",),
        ("worker-rule-modify",),
        ("worker-role-remove",),
        ("worker-queue-add",),
        ("worker-bucket-policy-modify",),
        ("worker-alarm-remove",),
    )
    for (case,) in cases:
        graph = copy.deepcopy(pristine)
        if case == "extra-id":
            rogue = copy.deepcopy(graph[0])
            rogue["ResourceChange"]["LogicalResourceId"] = "UnexpectedRole"
            graph.append(rogue)
        elif case == "missing-id":
            graph.pop()
        elif case == "top-level-type":
            graph[0]["Type"] = "Parameter"
        elif case == "action":
            change("SkyWatchdogRule")["Action"] = "Add"
        elif case == "remove":
            change("SkyWatchdogRule")["Action"] = "Remove"
        elif case == "resource-type":
            change("SkyWatchdogRule")["ResourceType"] = "AWS::SNS::Topic"
        elif case == "replacement":
            change("SkyWatchdogRule")["Replacement"] = "Conditional"
        elif case == "scope":
            change("SkyWatchdogRule")["Scope"] = ["Metadata"]
        elif case == "evaluation":
            change("SkyWatchdogRule")["Details"][0]["Evaluation"] = "Static"
        elif case == "source":
            change("SkyWatchdogRule")["Details"][0][
                "ChangeSource"
            ] = "ResourceReference"
        elif case == "causing-entity":
            change("SkyWatchdogRule")["Details"][0][
                "CausingEntity"
            ] = "ForeignFunction.Arn"
        elif case == "target":
            change("SkyWatchdogRule")["Details"][0]["Target"][
                "Name"
            ] = "State"
        elif case == "recreation":
            change("SkyWatchdogInvokePermission")["Details"][0]["Target"][
                "RequiresRecreation"
            ] = "Never"
        elif case == "missing-detail":
            change("SkyPilotControllerRole")["Details"].pop()
        elif case == "extra-detail":
            change("SkyWatchdogRule")["Details"].append(
                copy.deepcopy(change("SkyWatchdogRule")["Details"][0])
            )
        elif case == "extra-deny-detail":
            change("InstanceRole")["Details"].append(
                _change_detail(
                    evaluation="Static",
                    source="ParameterReference",
                    target="Policies",
                    causing_entity="SkyWorkerStartV2IntentBodySha256",
                )
            )
        elif case == "bucket-notification-detail":
            change("ModelBucket")["Details"].append(
                _change_detail(
                    evaluation="Static",
                    source="DirectModification",
                    target="NotificationConfiguration",
                )
            )
        else:
            logical_id, resource_type, action = {
                "worker-function-add": (
                    "SkyWorkerStartV2Function",
                    "AWS::Lambda::Function",
                    "Add",
                ),
                "worker-rule-modify": (
                    "SkyWorkerStartV2Rule",
                    "AWS::Events::Rule",
                    "Modify",
                ),
                "worker-role-remove": (
                    "SkyWorkerStartV2Role",
                    "AWS::IAM::Role",
                    "Remove",
                ),
                "worker-queue-add": (
                    "SkyWorkerStartV2DeadLetterQueue",
                    "AWS::SQS::Queue",
                    "Add",
                ),
                "worker-bucket-policy-modify": (
                    "SkyWorkerStartV2EvidenceBucketPolicy",
                    "AWS::S3::BucketPolicy",
                    "Modify",
                ),
                "worker-alarm-remove": (
                    "SkyWorkerStartV2ErrorsAlarm",
                    "AWS::CloudWatch::Alarm",
                    "Remove",
                ),
            }[case]
            rogue = copy.deepcopy(graph[0])
            rogue["ResourceChange"].update(
                {
                    "Action": action,
                    "LogicalResourceId": logical_id,
                    "ResourceType": resource_type,
                }
            )
            graph.append(rogue)
        graph_path.write_text(json.dumps(graph))
        before = len(_aws_calls(log_path))
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "execute-foundation"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0, case
        assert "foundation change set causal graph is not exact" in result.stderr
        assert not any(
            call[:2] == ["cloudformation", "execute-change-set"]
            for call in _aws_calls(log_path)[before:]
        )


def test_foundation_refuses_to_change_current_sky_run_authority(
    tmp_path: Path,
) -> None:
    environment, log_path = _foundation_environment(tmp_path)
    environment["FAKE_CURRENT_RUN_ID"] = "glm52-sky-other-run"
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "does not preserve current Sky authority" in result.stderr
    assert not any(
        call[:2] == ["cloudformation", "deploy"]
        for call in _aws_calls(log_path)
    )


def test_foundation_accepts_only_all_absent_legacy_worker_v2_state(
    tmp_path: Path,
) -> None:
    passing_path = tmp_path / "passing"
    passing_path.mkdir()
    environment, log_path = _foundation_environment(passing_path)
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    worker_resource_proof = [
        call
        for call in _aws_calls(log_path)
        if call[:2] == ["cloudformation", "list-stack-resources"]
        and "SkyWorkerStartV2" in call[call.index("--query") + 1]
    ]
    assert len(worker_resource_proof) == 1

    for index, (name, value) in enumerate(
        (
            (
                "FAKE_CURRENT_WORKER_START_V2_CODE_SHA256",
                "disabled",
            ),
            (
                "FAKE_CURRENT_WORKER_START_V2_FOUNDATION_REVISION",
                "versioned-code-v1",
            ),
            (
                "FAKE_CURRENT_WORKER_START_V2_ENABLE",
                "false",
            ),
        )
    ):
        case_path = tmp_path / f"partial-{index}"
        case_path.mkdir()
        environment, log_path = _foundation_environment(case_path)
        environment[name] = value
        result = subprocess.run(
            [
                str(SCRIPTS / "deploy_sky_control_plane.sh"),
                "prepare-foundation",
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert (
            "exact worker-v2 disabled authority"
            in result.stderr
        )
        assert not any(
            call[:2] == ["cloudformation", "deploy"]
            for call in _aws_calls(log_path)
        )


def test_foundation_rejects_legacy_worker_v2_resource_and_rechecks_at_execute(
    tmp_path: Path,
) -> None:
    resource_path = tmp_path / "resource"
    resource_path.mkdir()
    environment, log_path = _foundation_environment(resource_path)
    environment["FAKE_WORKER_START_V2_RESOURCES"] = (
        "SkyWorkerStartV2Function"
    )
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "legacy stack still has worker-v2 coordinator resources" in result.stderr
    assert not any(
        call[:2] == ["cloudformation", "deploy"]
        for call in _aws_calls(log_path)
    )

    race_path = tmp_path / "race"
    race_path.mkdir()
    environment, log_path = _foundation_environment(race_path)
    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    environment["FAKE_WORKER_START_V2_RESOURCES"] = (
        "SkyWorkerStartV2Rule"
    )
    before = len(_aws_calls(log_path))
    executed = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "execute-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert executed.returncode != 0
    assert (
        "legacy stack still has worker-v2 coordinator resources"
        in executed.stderr
    )
    new_calls = _aws_calls(log_path)[before:]
    assert not any(
        call[:2]
        in (
            ["cloudformation", "describe-change-set"],
            ["cloudformation", "execute-change-set"],
        )
        for call in new_calls
    )


def test_foundation_accepts_legacy_absent_flags_only_without_coordinator_resources(
    tmp_path: Path,
) -> None:
    environment, log_path = _foundation_environment(tmp_path)
    environment["FAKE_CURRENT_OBSERVE"] = "None"
    environment["FAKE_CURRENT_CANCEL"] = "None"

    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    executed = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "execute-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert executed.returncode == 0, executed.stderr
    calls = _aws_calls(log_path)
    resource_proofs = [
        call
        for call in calls
        if call[:2] == ["cloudformation", "list-stack-resources"]
    ]
    assert len(resource_proofs) == 4
    for prefix in ("SkyMustStart", "SkyWorkerStartV2"):
        assert (
            sum(
                f"starts_with(LogicalResourceId, '{prefix}')"
                in call[call.index("--query") + 1]
                for call in resource_proofs
            )
            == 2
        )


def test_foundation_rejects_legacy_absence_with_coordinator_resource(
    tmp_path: Path,
) -> None:
    environment, log_path = _foundation_environment(tmp_path)
    environment["FAKE_CURRENT_OBSERVE"] = "None"
    environment["FAKE_CURRENT_CANCEL"] = "None"
    environment["FAKE_MUST_START_RESOURCES"] = "SkyMustStartCancelFunction"

    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "legacy stack still has must-start coordinator resources" in result.stderr
    assert not any(
        call[:2] == ["cloudformation", "deploy"]
        for call in _aws_calls(log_path)
    )


def test_foundation_execute_revalidates_legacy_resource_absence(
    tmp_path: Path,
) -> None:
    environment, log_path = _foundation_environment(tmp_path)
    environment["FAKE_CURRENT_OBSERVE"] = "None"
    environment["FAKE_CURRENT_CANCEL"] = "None"
    prepared = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-foundation"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr

    drifted = dict(environment)
    drifted["FAKE_MUST_START_RESOURCES"] = "SkyMustStartCancelRule"
    result = subprocess.run(
        [str(SCRIPTS / "deploy_sky_control_plane.sh"), "execute-foundation"],
        cwd=ROOT,
        env=drifted,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "legacy stack still has must-start coordinator resources" in result.stderr
    assert not any(
        call[:2] == ["cloudformation", "execute-change-set"]
        for call in _aws_calls(log_path)
    )


def test_foundation_rejects_partial_or_enabled_coordinator_flags(
    tmp_path: Path,
) -> None:
    for index, (observe, cancel) in enumerate(
        (
            ("None", "false"),
            ("false", "None"),
            ("true", "false"),
            ("false", "true"),
            ("true", "true"),
        )
    ):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        environment, log_path = _foundation_environment(case_path)
        environment["FAKE_CURRENT_OBSERVE"] = observe
        environment["FAKE_CURRENT_CANCEL"] = cancel
        result = subprocess.run(
            [str(SCRIPTS / "deploy_sky_control_plane.sh"), "prepare-foundation"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "does not preserve current Sky authority" in result.stderr
        assert not any(
            call[:2] == ["cloudformation", "deploy"]
            for call in _aws_calls(log_path)
        )
