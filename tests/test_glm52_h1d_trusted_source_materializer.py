from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "aws/glm52-gpu/scripts/materialize_h1d_trusted_sources.py"
)
ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
PROFILE = "keep-gpu"
TABLE = "keep-glm52-h1g-ledger-v1"
PK = f"RUN#{RUN_ID}"
SK = "H1D_TRUSTED_SOURCE"
DEPLOYMENT_ROLE_ID = "AROAEXACTRETAINEDROLEID"


def _load_script():
    specification = importlib.util.spec_from_file_location(
        "_materialize_h1d_trusted_sources",
        SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _source_payloads() -> dict[str, object]:
    inventory_body = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_postcreate_inventory_v1",
        "stack_resources": [],
        "support_template_body_sha256": canonical_sha256(
            {"Resources": {}}
        ),
    }
    inventory = {
        **inventory_body,
        "canonical_body_sha256": canonical_sha256(inventory_body),
    }
    postcreate_body = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_postcreate_manifest_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": "act-20260729-0001",
        "support_template_body_sha256": canonical_sha256(
            {"Resources": {}}
        ),
        "support_postcreate_inventory_sha256": inventory[
            "canonical_body_sha256"
        ],
    }
    postcreate = {
        **postcreate_body,
        "canonical_body_sha256": canonical_sha256(postcreate_body),
    }
    templates_body = {
        "schema_version": 1,
        "record_type": "glm52_h1d_task6_task7_template_bundle_v1",
        "task6_templates": {},
        "task7_support_template": {"Resources": {}},
        "h1d_specs_identity_sha256": "f" * 64,
    }
    templates = {
        **templates_body,
        "canonical_body_sha256": canonical_sha256(templates_body),
    }
    return {
        "task6_manifest": {
            "schema_version": 1,
            "record_type": "glm52_h1g_stack_migration_manifest_v1",
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "run_id": RUN_ID,
            "bucket_name": (
                "keep-glm52-models-246813579024-us-west-2"
            ),
            "stacks": [],
            "artifacts": [],
        },
        "task6_templates": templates,
        "task7_postcreate_manifest": postcreate,
        "task7_inventory": inventory,
    }


def _source_raw() -> dict[str, bytes]:
    return {
        name: canonical_json_bytes(payload) + b"\n"
        for name, payload in _three_stack_documents().items()
    }


def _sources() -> dict[str, object]:
    raw = _source_raw()
    return {
        name: {
            "key": f"authority/{name}.json",
            "version_id": f"{name}-version",
            "file_sha256": hashlib.sha256(raw[name]).hexdigest(),
        }
        for name in raw
    }


def _input_raw() -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": 1,
            "record_type": "glm52_h1d_trusted_source_input_v1",
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "run_id": RUN_ID,
            "authority_sources": _sources(),
        }
    ) + b"\n"


def _ddb_items() -> list[dict[str, object]]:
    def item(
        sk: str,
        record_type: str,
        **attributes: object,
    ) -> dict[str, object]:
        return {
            "PK": {"S": PK},
            "SK": {"S": sk},
            "record_type": {"S": record_type},
            **{
                name: {"S": str(value)}
                for name, value in attributes.items()
            },
        }

    activation = "act-20260729-0001"
    return [
        item(
            "ACTIVATION_INDEX",
            "glm52_production_activation_index",
            current_activation_id=activation,
        ),
        item(
            f"ACTIVATION#{activation}#CONTROL",
            "glm52_production_control",
        ),
        item(
            (
                f"ACTIVATION#{activation}#"
                "ACTION#00000001#SKY#00000001"
            ),
            "glm52_production_action",
        ),
    ]


def _three_stack_documents() -> dict[str, object]:
    documents = deepcopy(_source_payloads())
    documents["task6_manifest"][
        "retained_deployment_role_id"
    ] = DEPLOYMENT_ROLE_ID
    stack_specs = (
        (
            "retained",
            "keep-glm52-gpu",
            "post-retain",
            "RetainedBucket",
            "AWS::S3::Bucket",
        ),
        (
            "fence",
            "keep-glm52-h1g-fence",
            "final-fence",
            "FencePolicy",
            "AWS::S3::BucketPolicy",
        ),
        (
            "support",
            "keep-glm52-h1g-support",
            "task7-support",
            "SupportRole",
            "AWS::IAM::Role",
        ),
    )
    task6_templates: dict[str, object] = {}
    artifacts = []
    stacks = []
    for kind, name, stage, logical_id, resource_type in stack_specs:
        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Resources": {
                logical_id: {
                    "Type": resource_type,
                    "Properties": {},
                }
            },
        }
        stack_id = (
            f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:"
            f"stack/{name}/{kind}-uuid"
        )
        stacks.append(
            {
                "kind": kind,
                "name": name,
                "stack_id": stack_id,
                "termination_protection": True,
                "tags": {
                    "campaign-run-id": RUN_ID,
                    "glm52-owner": "keep-ramp",
                },
            }
        )
        if stage == "task7-support":
            documents["task6_templates"]["task7_support_template"] = template
        else:
            task6_templates[stage] = template
            artifacts.append(
                {
                    "stage": stage,
                    "stack_id": stack_id,
                    "template_body_sha256": canonical_sha256(template),
                }
            )
    documents["task6_manifest"]["stacks"] = stacks
    documents["task6_manifest"]["artifacts"] = artifacts
    templates_body = dict(documents["task6_templates"])
    templates_body.pop("canonical_body_sha256")
    templates_body["task6_templates"] = task6_templates
    documents["task6_templates"] = {
        **templates_body,
        "canonical_body_sha256": canonical_sha256(templates_body),
    }
    support_template = documents["task6_templates"][
        "task7_support_template"
    ]
    inventory_body = dict(documents["task7_inventory"])
    inventory_body.pop("canonical_body_sha256")
    inventory_body["support_template_body_sha256"] = canonical_sha256(
        support_template
    )
    documents["task7_inventory"] = {
        **inventory_body,
        "canonical_body_sha256": canonical_sha256(inventory_body),
    }
    postcreate_body = dict(documents["task7_postcreate_manifest"])
    postcreate_body.pop("canonical_body_sha256")
    postcreate_body["support_template_body_sha256"] = canonical_sha256(
        support_template
    )
    postcreate_body["support_postcreate_inventory_sha256"] = documents[
        "task7_inventory"
    ]["canonical_body_sha256"]
    documents["task7_postcreate_manifest"] = {
        **postcreate_body,
        "canonical_body_sha256": canonical_sha256(postcreate_body),
    }
    return documents


class _ThreeStackRunner:
    def __init__(
        self,
        documents: dict[str, object],
        *,
        mutation: str | None = None,
    ) -> None:
        self.documents = documents
        self.mutation = mutation
        self.operations: list[tuple[str, ...]] = []
        self.role_arn = (
            f"arn:aws:iam::{ACCOUNT_ID}:role/"
            "keep-glm52-h1g-cloudformation-deployment"
        )
        self.role_id = DEPLOYMENT_ROLE_ID

    def run_json(
        self,
        operation: tuple[str, ...],
        *,
        timeout_seconds: int,
    ) -> dict[str, object]:
        assert timeout_seconds == 5
        self.operations.append(operation)
        if operation[:2] == ("iam", "get-role"):
            role_name = operation[operation.index("--role-name") + 1]
            assert role_name in {
                "keep-glm52-h1g-cloudformation-deployment",
                "attacker-deployment",
            }
            role_arn = (
                f"arn:aws:iam::{ACCOUNT_ID}:role/attacker-deployment"
                if self.mutation == "shared-attacker-role"
                else self.role_arn
            )
            role_id = (
                "AROAATTACKERROLEIDENTITY"
                if self.mutation == "shared-attacker-role"
                else self.role_id
            )
            return {
                "Role": {
                    "Arn": role_arn,
                    "RoleId": role_id,
                    "RoleName": role_name,
                    "Path": "/",
                },
                "ResponseMetadata": {
                    "RequestId": "iam-get-role-request"
                },
            }
        stack_name = operation[operation.index("--stack-name") + 1]
        stack = next(
            item
            for item in self.documents["task6_manifest"]["stacks"]
            if item["stack_id"] == stack_name
        )
        kind = stack["kind"]
        if kind == "retained":
            template = self.documents["task6_templates"][
                "task6_templates"
            ]["post-retain"]
        elif kind == "fence":
            template = self.documents["task6_templates"][
                "task6_templates"
            ]["final-fence"]
        else:
            template = self.documents["task6_templates"][
                "task7_support_template"
            ]
        logical_id, resource = next(iter(template["Resources"].items()))
        if operation[:2] == ("cloudformation", "describe-stacks"):
            role = (
                f"arn:aws:iam::{ACCOUNT_ID}:role/foreign"
                if self.mutation == "role" and kind == "retained"
                else (
                    f"arn:aws:iam::{ACCOUNT_ID}:"
                    "role/attacker-deployment"
                    if self.mutation
                    in {
                        "shared-attacker-role",
                        "shared-attacker-substitution",
                    }
                    else self.role_arn
                )
            )
            tags = [
                {"Key": key, "Value": value}
                for key, value in stack["tags"].items()
            ]
            if self.mutation == "foreign-tag" and kind == "retained":
                tags[0]["Value"] = "foreign"
            if self.mutation == "missing-tag" and kind == "retained":
                tags.pop()
            return {
                "Stacks": [
                    {
                        "StackId": stack["stack_id"],
                        "StackName": stack["name"],
                        "StackStatus": (
                            "UPDATE_ROLLBACK_COMPLETE"
                            if self.mutation == "stack-rollback"
                            and kind == "retained"
                            else "UPDATE_COMPLETE"
                        ),
                        "Parameters": [],
                        "RoleARN": role,
                        "Tags": tags,
                    }
                ],
                "ResponseMetadata": {
                    "RequestId": f"describe-{kind}"
                },
            }
        if operation[:2] == ("cloudformation", "get-template"):
            return {
                "TemplateBody": deepcopy(template),
                "ResponseMetadata": {"RequestId": f"template-{kind}"},
            }
        if operation[:2] == ("cloudformation", "list-stack-resources"):
            physical_id = (
                ""
                if self.mutation == "empty-physical"
                and kind == "retained"
                else (
                    " \t "
                    if self.mutation == "whitespace-physical"
                    and kind == "retained"
                    else f"{kind}-physical"
                )
            )
            rows = [] if self.mutation == "missing-resource" else [
                {
                    "LogicalResourceId": logical_id,
                    "PhysicalResourceId": physical_id,
                    "ResourceType": resource["Type"],
                    **(
                        {}
                        if self.mutation == "missing-resource-status"
                        and kind == "retained"
                        else {
                            "ResourceStatus": (
                                {
                                    "resource-failed": "UPDATE_FAILED",
                                    "resource-rollback": (
                                        "UPDATE_ROLLBACK_COMPLETE"
                                    ),
                                    "resource-in-progress": (
                                        "UPDATE_IN_PROGRESS"
                                    ),
                                }.get(
                                    self.mutation,
                                    "CREATE_COMPLETE",
                                )
                            )
                        }
                    ),
                }
            ]
            return {
                "StackResourceSummaries": rows,
                "ResponseMetadata": {"RequestId": f"resources-{kind}"},
            }
        if operation[:2] == (
            "cloudformation",
            "describe-termination-protection",
        ):
            return {
                "EnableTerminationProtection": True,
                "ResponseMetadata": {"RequestId": f"termination-{kind}"},
            }
        raise AssertionError(operation)


def test_materializer_seals_exact_three_stack_cloudformation_truth() -> None:
    module = _load_script()
    documents = _three_stack_documents()
    runner = _ThreeStackRunner(documents)
    items, role = module._read_cloudformation_contract(runner, documents)
    assert len(items) == 3
    assert {item["stack_id"] for item in items} == {
        stack["stack_id"]
        for stack in documents["task6_manifest"]["stacks"]
    }
    assert {item["service_role_arn"] for item in items} == {
        runner.role_arn
    }
    assert role == {
        "role_arn": runner.role_arn,
        "role_id": DEPLOYMENT_ROLE_ID,
        "request_id": "iam-get-role-request",
    }
    assert all(item["status"] == "UPDATE_COMPLETE" for item in items)
    assert all(
        item["stack_tags"]
        == [
            {"key": key, "value": value}
            for key, value in sorted(
                next(
                    stack["tags"]
                    for stack in documents["task6_manifest"]["stacks"]
                    if stack["stack_id"] == item["stack_id"]
                ).items()
            )
        ]
        for item in items
    )
    assert all(item["termination_protection"] is True for item in items)
    retained = next(
        item
        for item in items
        if "/keep-glm52-gpu/" in item["stack_id"]
    )
    assert retained["resources_sha256"] == canonical_sha256(
        [
            {
                "logical_id": "RetainedBucket",
                "resource_type": "AWS::S3::Bucket",
                "physical_id": "retained-physical",
                "resource_status": "CREATE_COMPLETE",
            }
        ]
    )
    assert len(runner.operations) == 13
    assert all("--no-paginate" in operation for operation in runner.operations)


@pytest.mark.parametrize(
    "mutation",
    (
        "role",
        "shared-attacker-role",
        "shared-attacker-substitution",
        "missing-resource",
        "stack-rollback",
        "foreign-tag",
        "missing-tag",
        "resource-failed",
        "resource-rollback",
        "resource-in-progress",
        "missing-resource-status",
    ),
)
def test_materializer_rejects_foreign_or_noncomplete_stack_truth(
    mutation: str,
) -> None:
    module = _load_script()
    documents = _three_stack_documents()
    with pytest.raises(
        ValueError,
        match="role|resource|status|tag",
    ):
        module._read_cloudformation_contract(
            _ThreeStackRunner(documents, mutation=mutation),
            documents,
        )


@pytest.mark.parametrize(
    "mutation",
    ("empty-physical", "whitespace-physical"),
)
def test_materializer_rejects_empty_physical_resource_identity_mutant(
    mutation: str,
) -> None:
    module = _load_script()
    documents = _three_stack_documents()
    with pytest.raises(ValueError, match="physical|resource"):
        module._read_cloudformation_contract(
            _ThreeStackRunner(documents, mutation=mutation),
            documents,
        )


def test_materializer_conditionally_creates_fixed_sealed_record_and_reads_back(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    raw = _input_raw()
    input_path = tmp_path / "trusted-source-input.json"
    input_path.write_bytes(raw)
    source_raw = _source_raw()

    class Runner:
        def __init__(self, **kwargs: object) -> None:
            self.operations: list[tuple[str, ...]] = []
            self.object_reads: list[dict[str, object]] = []
            self.item: dict[str, object] | None = None
            self.cloudformation = _ThreeStackRunner(
                _three_stack_documents()
            )

        def get_s3_object(
            self,
            *,
            bucket: str,
            key: str,
            version_id: str,
            timeout_seconds: int,
        ) -> tuple[dict[str, object], bytes]:
            self.object_reads.append(
                {
                    "bucket": bucket,
                    "key": key,
                    "version_id": version_id,
                    "timeout_seconds": timeout_seconds,
                }
            )
            name = key.removeprefix("authority/").removesuffix(".json")
            return {"VersionId": version_id}, source_raw[name]

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            self.operations.append(operation)
            if operation == ("sts", "get-caller-identity"):
                return {
                    "Account": ACCOUNT_ID,
                    "Arn": (
                        f"arn:aws:sts::{ACCOUNT_ID}:"
                        "assumed-role/keep-glm52-h1g-decision/session"
                    ),
                    "UserId": "AROATEST:session",
                }
            if operation[:2] == ("dynamodb", "query"):
                return {"Items": _ddb_items()}
            if operation[0] in {"cloudformation", "iam"}:
                return self.cloudformation.run_json(
                    operation,
                    timeout_seconds=timeout_seconds,
                )
            if operation[:2] == ("dynamodb", "put-item"):
                self.item = json.loads(
                    operation[operation.index("--item") + 1]
                )
                return {}
            if operation[:2] == ("dynamodb", "get-item"):
                return {"Item": self.item}
            raise AssertionError(operation)

    holder: dict[str, Runner] = {}

    def runner_factory(**kwargs: object) -> Runner:
        holder["runner"] = Runner(**kwargs)
        return holder["runner"]

    assert (
        module.run(
            [
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--input",
                str(input_path),
                "--input-sha256",
                hashlib.sha256(raw).hexdigest(),
            ],
            runner_factory=runner_factory,
        )
        == 0
    )
    expected_item = holder["runner"].item
    assert expected_item is not None
    record = json.loads(
        expected_item["canonical_body_json"]["S"]
    )
    assert set(record) == {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "PK",
        "SK",
        "state",
        "authority_sources",
        "source_contract_sha256",
        "cloudformation_deployment_role",
        "dynamodb_expected_items",
        "cloudformation_expected_items",
        "canonical_body_sha256",
    }
    assert record["source_contract_sha256"] == module._source_contract_sha256(
        _three_stack_documents()
    )
    assert record["dynamodb_expected_items"] == module._ddb_expected_items(
        _ddb_items(),
        activation_id="act-20260729-0001",
    )
    assert record["cloudformation_deployment_role"] == {
        "role_arn": holder["runner"].cloudformation.role_arn,
        "role_id": DEPLOYMENT_ROLE_ID,
        "request_id": "iam-get-role-request",
    }
    assert "h1d_specs" not in record
    item_json = canonical_json_bytes(expected_item).decode("utf-8")
    key_json = canonical_json_bytes(
        {"PK": {"S": PK}, "SK": {"S": SK}}
    ).decode("utf-8")
    assert holder["runner"].operations == [
        ("sts", "get-caller-identity"),
        *holder["runner"].cloudformation.operations,
        (
            "dynamodb",
            "query",
            "--table-name",
            TABLE,
            "--key-condition-expression",
            "PK = :pk",
            "--expression-attribute-values",
            canonical_json_bytes({":pk": {"S": PK}}).decode("utf-8"),
            "--consistent-read",
            "--no-paginate",
        ),
        (
            "dynamodb",
            "put-item",
            "--table-name",
            TABLE,
            "--item",
            item_json,
            "--condition-expression",
            "attribute_not_exists(PK) AND attribute_not_exists(SK)",
            "--return-consumed-capacity",
            "NONE",
        ),
        (
            "dynamodb",
            "get-item",
            "--table-name",
            TABLE,
            "--key",
            key_json,
            "--consistent-read",
            "--return-consumed-capacity",
            "NONE",
        ),
    ]
    assert holder["runner"].object_reads == [
        {
            "bucket": "keep-glm52-models-246813579024-us-west-2",
            "key": source["key"],
            "version_id": source["version_id"],
            "timeout_seconds": 5,
        }
        for source in _sources().values()
    ]
    assert json.loads(capsys.readouterr().out) == {
        "account_id": ACCOUNT_ID,
        "record_identity_sha256": expected_item[
            "canonical_body_sha256"
        ]["S"],
        "region": REGION,
        "run_id": RUN_ID,
        "status": "SEALED_TRUSTED_SOURCE_CREATED",
    }


def test_materializer_wrong_account_stops_before_input_read(
    tmp_path: Path,
) -> None:
    module = _load_script()

    class Runner:
        def __init__(self, **kwargs: object) -> None:
            self.operations: list[tuple[str, ...]] = []

        def run_json(
            self, operation: tuple[str, ...], *, timeout_seconds: int
        ) -> dict[str, object]:
            self.operations.append(operation)
            return {
                "Account": "000000000000",
                "Arn": "arn:aws:iam::000000000000:user/foreign",
                "UserId": "foreign",
            }

    holder: dict[str, Runner] = {}

    def runner_factory(**kwargs: object) -> Runner:
        holder["runner"] = Runner(**kwargs)
        return holder["runner"]

    with pytest.raises(ValueError, match="account"):
        module.run(
            [
                "--profile",
                PROFILE,
                "--region",
                REGION,
                "--input",
                str(tmp_path / "must-not-be-read.json"),
                "--input-sha256",
                "1" * 64,
            ],
            runner_factory=runner_factory,
        )
    assert holder["runner"].operations == [
        ("sts", "get-caller-identity")
    ]


def test_materializer_runner_has_one_attempt_and_no_hidden_retry() -> None:
    module = _load_script()
    calls: list[dict[str, object]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess:
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "Account": ACCOUNT_ID,
                    "Arn": f"arn:aws:iam::{ACCOUNT_ID}:user/test",
                    "UserId": "test",
                }
            ),
            "",
        )

    runner = module.AwsCliCommandRunner(subprocess_run=fake_run)
    runner.run_json(("sts", "get-caller-identity"), timeout_seconds=5)
    assert len(calls) == 1
    assert calls[0]["env"]["AWS_MAX_ATTEMPTS"] == "1"
    assert calls[0]["env"]["AWS_RETRY_MODE"] == "standard"
    assert calls[0]["timeout"] == 5


def test_materializer_production_runner_is_one_attempt_native_sdk() -> None:
    module = _load_script()
    configs: list[dict[str, object]] = []
    client_calls: list[tuple[str, dict[str, object]]] = []

    class StsClient:
        def get_caller_identity(self, **kwargs: object) -> dict[str, object]:
            client_calls.append(("get_caller_identity", kwargs))
            return {
                "Account": ACCOUNT_ID,
                "Arn": f"arn:aws:iam::{ACCOUNT_ID}:user/test",
                "UserId": "test",
                "ResponseMetadata": {"RequestId": "sdk-sts-request"},
            }

    class IamClient:
        def get_role(self, **kwargs: object) -> dict[str, object]:
            client_calls.append(("get_role", kwargs))
            return {
                "Role": {
                    "Arn": (
                        f"arn:aws:iam::{ACCOUNT_ID}:role/"
                        "keep-glm52-h1g-cloudformation-deployment"
                    ),
                    "RoleId": DEPLOYMENT_ROLE_ID,
                    "RoleName": (
                        "keep-glm52-h1g-cloudformation-deployment"
                    ),
                    "Path": "/",
                },
                "ResponseMetadata": {"RequestId": "sdk-iam-request"},
            }

    class Session:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs == {
                "profile_name": PROFILE,
                "region_name": REGION,
            }

        def client(
            self,
            service: str,
            *,
            config: object,
        ) -> object:
            assert config is configs[0]
            if service == "sts":
                return StsClient()
            if service == "iam":
                return IamClient()
            raise AssertionError(service)

    def config_factory(**kwargs: object) -> object:
        configs.append(kwargs)
        return kwargs

    runner = module.AwsSdkCommandRunner(
        session_factory=Session,
        config_factory=config_factory,
    )
    assert module._PRODUCTION_RUNNER is module.AwsSdkCommandRunner
    assert runner.run_json(
        ("sts", "get-caller-identity"),
        timeout_seconds=5,
    )["Account"] == ACCOUNT_ID
    assert runner.run_json(
        (
            "iam",
            "get-role",
            "--role-name",
            "keep-glm52-h1g-cloudformation-deployment",
            "--no-paginate",
        ),
        timeout_seconds=5,
    )["Role"]["RoleId"] == DEPLOYMENT_ROLE_ID
    assert configs == [
        {
            "connect_timeout": 1,
            "read_timeout": 2,
            "retries": {
                "total_max_attempts": 1,
                "mode": "standard",
            },
        }
    ]
    assert client_calls == [
        ("get_caller_identity", {}),
        (
            "get_role",
            {
                "RoleName": (
                    "keep-glm52-h1g-cloudformation-deployment"
                )
            },
        ),
    ]


def test_source_contract_ignores_rehashed_caller_h1d_spec_claim() -> None:
    module = _load_script()
    original = _source_payloads()
    forged = deepcopy(original)
    template_body = dict(forged["task6_templates"])
    template_body.pop("canonical_body_sha256")
    template_body["h1d_specs_identity_sha256"] = "1" * 64
    forged["task6_templates"] = {
        **template_body,
        "canonical_body_sha256": canonical_sha256(template_body),
    }
    assert module._source_contract_sha256(
        forged
    ) == module._source_contract_sha256(original)


def test_source_contract_rejects_rehashed_inventory_template_drift() -> None:
    module = _load_script()
    forged = deepcopy(_source_payloads())
    inventory_body = dict(forged["task7_inventory"])
    inventory_body.pop("canonical_body_sha256")
    inventory_body["support_template_body_sha256"] = "1" * 64
    forged["task7_inventory"] = {
        **inventory_body,
        "canonical_body_sha256": canonical_sha256(inventory_body),
    }
    postcreate_body = dict(forged["task7_postcreate_manifest"])
    postcreate_body.pop("canonical_body_sha256")
    postcreate_body["support_postcreate_inventory_sha256"] = forged[
        "task7_inventory"
    ]["canonical_body_sha256"]
    forged["task7_postcreate_manifest"] = {
        **postcreate_body,
        "canonical_body_sha256": canonical_sha256(postcreate_body),
    }
    with pytest.raises(ValueError, match="semantic contract"):
        module._source_contract_sha256(forged)
