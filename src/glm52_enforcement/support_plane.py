"""Pure H.1g ephemeral support-plane generation and validation contracts."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from .canonical import canonical_json_bytes, canonical_sha256
from .decision_closure import (
    AUTHORITY_AUDIT_KINDS,
    CLOSURE_DEADLINE_SECONDS,
    CLOSURE_PHASE_CEILINGS,
    CLOSURE_STEPS,
    HANDOFF_AUDIT_KIND,
    LAMBDA_CONFIGURED_TIMEOUT_SECONDS,
    MIN_REMAINING_BEFORE_TOKEN_SECONDS,
    STEP_FUNCTION_TASK_TIMEOUT_SECONDS,
    SUFFIX_PHASE_CEILINGS,
    build_task11_workflow_definition,
)
from .task13_support_input_materialization import (
    SupportBuildInputs,
    support_build_inputs_from_mapping,  # noqa: F401
    support_build_inputs_identity,
    support_build_inputs_projection,  # noqa: F401
)

ACCOUNT_ID = "246813579024"
ORGANIZATIONS_MANAGEMENT_ACCOUNT_ID = "008316604477"
ORGANIZATION_ID = "o-08ddnqdzd3"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
PINNED_RELAY_PORTS = (18443, 18444, 18445, 18446)
SUPPORT_STACK_NAME = "keep-glm52-h1g-support"
FENCE_STACK_NAME = "keep-glm52-h1g-fence"
RETAINED_STACK_NAME = "keep-glm52-gpu"
MODEL_BUCKET_NAME = "keep-glm52-models-246813579024-us-west-2"
SUPPORT_DELETION_ROLE_ARN = (
    f"arn:aws:iam::{ACCOUNT_ID}:role/keep-glm52-h1g-support-deletion"
)
TLS_HANDLER_ROLE_ARN = (
    f"arn:aws:iam::{ACCOUNT_ID}:role/keep-glm52-h1g-support-tls-handler"
)
SUPPORT_OPERATOR_TOPIC_ARN = (
    f"arn:aws:sns:{REGION}:{ACCOUNT_ID}:keep-glm52-h1g-operator-alerts"
)
SUPPORT_CIDRS = (
    "10.20.101.0/24",
    "10.20.102.0/24",
    "10.20.103.0/24",
)
SECRET_LOGICAL_IDS = (
    "RawSkyTokenSecret",
    "SkyBootstrapSecret",
    "AttestationClientTlsSecret",
    "LaunchAdmissionClientTlsSecret",
    "NumericBindingClientTlsSecret",
    "RetainedCancellationClientTlsSecret",
    "CombinedHostTlsSecret",
    "CaIssuanceSecret",
)
_SECRET_PURPOSES = {
    "RawSkyTokenSecret": "raw-sky-token",
    "SkyBootstrapSecret": "sky-bootstrap-hash",
    "AttestationClientTlsSecret": "attestation-client-tls",
    "LaunchAdmissionClientTlsSecret": "launch-admission-client-tls",
    "NumericBindingClientTlsSecret": "numeric-binding-client-tls",
    "RetainedCancellationClientTlsSecret": "retained-cancellation-client-tls",
    "CombinedHostTlsSecret": "combined-host-server-tls",
    "CaIssuanceSecret": "ca-issuance",
}
CLIENT_PATHS: Mapping[str, Mapping[str, object]] = {
    "attestation": {
        "client_security_group": "AttestationClientSecurityGroup",
        "egress_logical_id": "AttestationClientEgress",
        "ingress_logical_id": "AttestationHostIngress",
        "port": lambda inputs: inputs.attestation_port,
    },
    "launch_admission": {
        "client_security_group": "LaunchAdmissionClientSecurityGroup",
        "egress_logical_id": "LaunchAdmissionClientEgress",
        "ingress_logical_id": "LaunchAdmissionHostIngress",
        "port": lambda inputs: inputs.launch_admission_port,
    },
    "numeric_binding": {
        "client_security_group": "NumericBindingClientSecurityGroup",
        "egress_logical_id": "NumericBindingClientEgress",
        "ingress_logical_id": "NumericBindingHostIngress",
        "port": lambda inputs: inputs.numeric_binding_port,
    },
    "retained_cancellation": {
        "client_security_group": "RetainedCancellationClientSecurityGroup",
        "egress_logical_id": "RetainedCancellationClientEgress",
        "ingress_logical_id": "RetainedCancellationHostIngress",
        "port": lambda inputs: inputs.retained_cancellation_port,
    },
}
VPC_SECRET_READERS: Mapping[str, tuple[str, str, tuple[str, ...]]] = {
    "Attestation": (
        "AttestationRole",
        "AttestationClientSecurityGroup",
        ("RawSkyTokenSecret", "AttestationClientTlsSecret"),
    ),
    "LaunchAdmission": (
        "LaunchAdmissionRole",
        "LaunchAdmissionClientSecurityGroup",
        ("RawSkyTokenSecret", "LaunchAdmissionClientTlsSecret"),
    ),
    "NumericBinding": (
        "NumericBindingRole",
        "NumericBindingClientSecurityGroup",
        ("RawSkyTokenSecret", "NumericBindingClientTlsSecret"),
    ),
    "RetainedCancellation": (
        "RetainedCancellationRole",
        "RetainedCancellationClientSecurityGroup",
        ("RawSkyTokenSecret", "RetainedCancellationClientTlsSecret"),
    ),
}
REHEARSAL_PROBE_SECRET_IDS = (
    "RawSkyTokenSecret",
    "AttestationClientTlsSecret",
)
REHEARSAL_TASK11_READ_SIDS = frozenset(
    {
        "WalkExactCampaignBucket",
        "ReadWriteExactCampaignObjects",
        "DecryptExactCampaignObjectsViaS3Only",
        "ReadExactCampaignBucketState",
        "ReadExactFenceStack",
        "AuthenticateClosurePrincipal",
        "ReadExactH1dStacks",
        "ReadExactH1dFunctions",
        "ReadExactH1dRoles",
        "ReadExactH1dInstanceProfile",
        "ReadExactH1dStateMachines",
        "ReadExactH1dEventRule",
        "ReadExactH1dSchedules",
        "ReadExactH1dQueue",
        "ReadExactH1dOperatorTopic",
        "ReadOnlyUnscopableH1dIndexes",
    }
)
_SECRET_ENV_PREFIX = {
    "RawSkyTokenSecret": "RAW_SKY_TOKEN",
    "SkyBootstrapSecret": "SKY_BOOTSTRAP",
    "AttestationClientTlsSecret": "ATTESTATION_CLIENT_TLS",
    "LaunchAdmissionClientTlsSecret": "LAUNCH_ADMISSION_CLIENT_TLS",
    "NumericBindingClientTlsSecret": "NUMERIC_BINDING_CLIENT_TLS",
    "RetainedCancellationClientTlsSecret": "RETAINED_CANCELLATION_CLIENT_TLS",
    "CombinedHostTlsSecret": "COMBINED_HOST_TLS",
    "CaIssuanceSecret": "CA_ISSUANCE",
}
_SECRET_VERSION_SOURCE = {
    logical_id: (
        "SkyBootstrapCustomResource"
        if logical_id in SECRET_LOGICAL_IDS[:2]
        else "TlsBundleCustomResource"
    )
    for logical_id in SECRET_LOGICAL_IDS
}
RUNTIME_FUNCTIONS: Mapping[str, Mapping[str, object]] = {
    "BudgetGate": {
        "role": "BudgetGateRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 120,
        "memory": 256,
        "handler": "support_budget_gate_handler.main",
    },
    "RehearsalCollector": {
        "role": "RehearsalCollectorRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 120,
        "memory": 256,
        "handler": "support_rehearsal_collector_handler.main",
    },
    "RehearsalProbe": {
        "role": "RehearsalProbeRole",
        "vpc_attached": True,
        "client_security_group": "AttestationClientSecurityGroup",
        "timeout": 15,
        "memory": 256,
        "handler": "support_rehearsal_probe_handler.main",
    },
    "Decision": {
        "role": "DecisionRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 840,
        "memory": 512,
        "handler": "support_decision_handler.main",
    },
    "SourceGpuSpend": {
        "role": "SourceGpuSpendRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 5,
        "memory": 256,
        "handler": "support_source_publisher_handler.main",
    },
    "SourceSubmissionIntent": {
        "role": "SourceSubmissionIntentRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 5,
        "memory": 256,
        "handler": "support_source_publisher_handler.main",
    },
    "SourceControllerBaseline": {
        "role": "SourceControllerBaselineRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 5,
        "memory": 256,
        "handler": "support_source_publisher_handler.main",
    },
    "SourceControlPlaneReadiness": {
        "role": "SourceControlPlaneReadinessRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 5,
        "memory": 256,
        "handler": "support_source_publisher_handler.main",
    },
    "SourceSubmissionAcquisition": {
        "role": "SourceSubmissionAcquisitionRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 5,
        "memory": 256,
        "handler": "support_source_publisher_handler.main",
    },
    "FenceExecutor": {
        "role": "FenceExecutorRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 12,
        "memory": 256,
        "handler": "support_fence_handler.main",
    },
    "FenceSuccessor": {
        "role": "FenceSuccessorRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 8,
        "memory": 256,
        "handler": "support_effect_writer_handler.main",
    },
    "ClaimWriter": {
        "role": "ClaimWriterRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 8,
        "memory": 256,
        "handler": "support_effect_writer_handler.main",
    },
    "DecisionWriter": {
        "role": "DecisionWriterRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 8,
        "memory": 256,
        "handler": "support_effect_writer_handler.main",
    },
    "TerminalV1Writer": {
        "role": "TerminalV1WriterRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 8,
        "memory": 256,
        "handler": "support_effect_writer_handler.main",
    },
    "ClosureHandoff": {
        "role": "ClosureHandoffRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 8,
        "memory": 256,
        "handler": "support_effect_writer_handler.main",
    },
    "Attestation": {
        "role": "AttestationRole",
        "vpc_attached": True,
        "client_security_group": "AttestationClientSecurityGroup",
        "timeout": 15,
        "memory": 256,
        "handler": "support_attestation_handler.main",
    },
    "LaunchAdmission": {
        "role": "LaunchAdmissionRole",
        "vpc_attached": True,
        "client_security_group": "LaunchAdmissionClientSecurityGroup",
        "timeout": 25,
        "memory": 256,
        "handler": "support_launchadmission_handler.main",
    },
    "NumericBinding": {
        "role": "NumericBindingRole",
        "vpc_attached": True,
        "client_security_group": "NumericBindingClientSecurityGroup",
        "timeout": 20,
        "memory": 256,
        "handler": "support_numericbinding_handler.main",
    },
    "RetainedCancellation": {
        "role": "RetainedCancellationRole",
        "vpc_attached": True,
        "client_security_group": "RetainedCancellationClientSecurityGroup",
        "timeout": 20,
        "memory": 256,
        "handler": "support_retainedcancellation_handler.main",
    },
    "SupportDeadline": {
        "role": "SupportDeadlineRole",
        "vpc_attached": False,
        "client_security_group": None,
        "timeout": 540,
        "memory": 256,
        "handler": "support_supportdeadline_handler.main",
    },
}
_TASK11_CLOSURE_CALLEES = frozenset(
    {
        "SourceGpuSpend",
        "SourceSubmissionIntent",
        "SourceControllerBaseline",
        "SourceControlPlaneReadiness",
        "SourceSubmissionAcquisition",
        "FenceExecutor",
        "FenceSuccessor",
        "ClaimWriter",
        "DecisionWriter",
        "TerminalV1Writer",
        "ClosureHandoff",
        "Attestation",
        "LaunchAdmission",
        "NumericBinding",
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EC2_ID = re.compile(r"(?:vpc|subnet|vpce)-[0-9a-f]{17}\Z")
_AMI_ID = re.compile(r"ami-[0-9a-f]{17}\Z")
_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
_ACTIVATION_ID = re.compile(r"[a-z0-9](?:[a-z0-9-]{1,62}[a-z0-9])?\Z")
_BUCKET = re.compile(
    r"(?=.{3,63}\Z)(?![0-9]+(?:\.[0-9]+){3}\Z)"
    r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\Z"
)
_VERSION_ID = re.compile(r"[A-Za-z0-9._+=/-]{1,1024}\Z")
_SECRET_VERSION_ID = re.compile(r"[A-Za-z0-9-]{32,64}\Z")
_PRIVATE_MATERIAL_PATTERNS = (
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
        re.IGNORECASE,
    ),
    re.compile(
        r"-----BEGIN (?:TRUSTED )?(?:CERTIFICATE|CERTIFICATE REQUEST|"
        r"PUBLIC KEY|SSH2 PUBLIC KEY|PGP PUBLIC KEY BLOCK|PKCS7|CMS|"
        r"PKCS #7 SIGNED DATA)-----",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[\s;])(?:export\s+)?"
        r"[A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PRIVATE_KEY|ACCESS_KEY|"
        r"CERTIFICATE|PUBLIC_KEY|CA_BUNDLE|TRUST_BUNDLE)"
        r"[A-Z0-9_]*\s*=",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"(?:^|\s)(?:ssh-rsa|ssh-ed25519|ecdsa-sha2-[A-Za-z0-9-]+)"
        r"\s+[A-Za-z0-9+/=]{16,}(?:\s|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"(?:^|\s)[A-Za-z0-9@._+-]+-cert-v01@openssh[.]com"
        r"\s+[A-Za-z0-9+/=]{16,}(?:\s|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
        r"[A-Za-z0-9_-]{8,}\b"
    ),
)

_INPUT_FIELDS = (
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "run_id",
    "activation_id",
    "nat_gateway_id",
    "support_stack_id",
    "fence_stack_name",
    "fence_stack_id",
    "fence_service_role_arn",
    "fence_manifest_coordinate",
    "fence_template_inventory",
    "task11_writer_bindings",
    "organizations_id",
    "retained_stack_id",
    "retained_vpc_id",
    "retained_vpc_cidr",
    "existing_subnet_cidrs",
    "existing_secondary_cidrs",
    "primary_az",
    "alternate_az",
    "primary_public_subnet_id",
    "retained_public_s3_endpoint_id",
    "retained_kms_key_arn",
    "retained_kms_key_id",
    "ledger_table_name",
    "ledger_table_arn",
    "model_bucket_name",
    "model_bucket_arn",
    "model_prefix",
    "host_ami_id",
    "host_private_ip",
    "host_user_data",
    "host_user_data_sha256",
    "host_boot_identity_sha256",
    "root_volume_gib",
    "root_volume_type",
    "root_volume_iops",
    "root_volume_throughput_mibps",
    "cryptography_layer_arn",
    "cryptography_layer_sha256",
    "lambda_code_bucket",
    "lambda_code_key",
    "lambda_code_version_id",
    "lambda_code_sha256",
    "attestation_port",
    "launch_admission_port",
    "numeric_binding_port",
    "retained_cancellation_port",
    "activation_started_at",
    "runtime_credential_cutoff_at",
    "support_deletion_inventory",
    "price_card_identity_sha256",
    "retained_function_version_bindings",
    "retained_export_names",
)
_POSTCREATE_INPUT_FIELDS = frozenset(
    {
        "nat_gateway_id",
        "support_stack_id",
        "support_deletion_inventory",
    }
)
_BUILD_INPUT_FIELDS = tuple(SupportBuildInputs.__dataclass_fields__)
_PRE_SUPPORT_RUNTIME_INPUT_FIELDS = (
    "schema_version",
    "record_type",
    "account_id",
    "region",
    "run_id",
    "activation_id",
    "retained_stack_id",
    "retained_foundation_readback_sha256",
    "fence_manifest_coordinate",
    "future_support_version_absence",
    "retained_kms_key_arn",
    "ledger_table_arn",
    "model_bucket_name",
    "model_bucket_arn",
    "lambda_code_bucket",
    "lambda_code_key",
    "lambda_code_version_id",
    "lambda_code_sha256",
    "retained_export_names",
)
_PRE_SUPPORT_REQUIRED_RETAINED_EXPORT_NAMES = frozenset(
    {
        "KeepGlm52VpcId",
        "KeepGlm52PrimaryPublicSubnetId",
        "KeepGlm52CampaignKmsKeyArn",
        "KeepGlm52H1gLedgerArn",
    }
)
_REQUIRED_RETAINED_EXPORT_NAMES = frozenset(
    {
        *_PRE_SUPPORT_REQUIRED_RETAINED_EXPORT_NAMES,
        "KeepGlm52Task12TerminalV2VersionArn",
        "KeepGlm52Task12WorkerDrainVersionArn",
        "KeepGlm52Task9LiabilityWatcherVersionArn",
    }
)
_RETAINED_FUNCTION_VERSION_SPECS = (
    (
        "Task12TerminalV2VersionArn",
        "KeepGlm52Task12TerminalV2VersionArn",
        "keep-glm52-h1g-terminal-v2-writer",
    ),
    (
        "Task12WorkerDrainVersionArn",
        "KeepGlm52Task12WorkerDrainVersionArn",
        "keep-glm52-h1g-worker-drain-signal",
    ),
    (
        "Task9LiabilityWatcherVersionArn",
        "KeepGlm52Task9LiabilityWatcherVersionArn",
        "keep-glm52-h1g-worker-launch-custody",
    ),
)
_SUPPORT_DELETION_ACTIONS = {
    "ec2:CreateSnapshot",
    "ec2:DeleteNatGateway",
    "ec2:DeleteNetworkInterface",
    "ec2:DeleteRoute",
    "ec2:DeleteRouteTable",
    "ec2:DisassociateRouteTable",
    "ec2:RevokeSecurityGroupEgress",
    "ec2:RevokeSecurityGroupIngress",
    "ec2:DeleteSecurityGroup",
    "ec2:DeleteSubnet",
    "ec2:DeleteVolume",
    "ec2:DetachVolume",
    "ec2:DeleteVpcEndpoints",
    "ec2:ReleaseAddress",
    "ec2:TerminateInstances",
    "lambda:InvokeFunction",
    "lambda:DeleteFunction",
    "lambda:DeleteFunctionEventInvokeConfig",
    "lambda:RemovePermission",
    "iam:DeleteRolePolicy",
    "iam:DeleteRole",
    "iam:RemoveRoleFromInstanceProfile",
    "iam:DeleteInstanceProfile",
    "s3:ListBucket",
    "s3:DeleteObject",
    "s3:DeleteObjectVersion",
    "s3:GetBucketPolicy",
    "s3:DeleteBucketPolicy",
    "s3:DeleteBucket",
    "logs:DeleteLogGroup",
    "cloudwatch:DeleteAlarms",
    "sqs:DeleteQueue",
    "secretsmanager:DeleteResourcePolicy",
    "secretsmanager:DeleteSecret",
    "states:DeleteStateMachine",
    "states:DeleteStateMachineVersion",
    "scheduler:DeleteSchedule",
    "events:RemoveTargets",
    "events:DeleteRule",
    "kms:CreateGrant",
    "kms:RetireGrant",
    "kms:RevokeGrant",
    "ec2:DescribeAddresses",
    "ec2:DescribeInstances",
    "ec2:DescribeNatGateways",
    "ec2:DescribeNetworkInterfaces",
    "ec2:DescribeRouteTables",
    "ec2:DescribeSecurityGroups",
    "ec2:DescribeSnapshots",
    "ec2:DescribeSubnets",
    "ec2:DescribeVolumes",
    "ec2:DescribeVpcEndpoints",
    "lambda:GetFunction",
    "lambda:ListVersionsByFunction",
    "iam:GetRole",
    "iam:GetInstanceProfile",
    "iam:ListRolePolicies",
    "s3:GetBucketLocation",
    "s3:ListBucketVersions",
    "logs:DescribeLogGroups",
    "cloudwatch:DescribeAlarms",
    "sqs:GetQueueAttributes",
    "secretsmanager:DescribeSecret",
    "states:DescribeStateMachine",
    "scheduler:GetSchedule",
    "events:DescribeRule",
    "events:ListTargetsByRule",
}
_UNSCOPED_SUPPORT_DELETION_READ_ACTIONS = {
    "ec2:DescribeAddresses",
    "ec2:DescribeInstances",
    "ec2:DescribeNatGateways",
    "ec2:DescribeNetworkInterfaces",
    "ec2:DescribeRouteTables",
    "ec2:DescribeSecurityGroups",
    "ec2:DescribeSnapshots",
    "ec2:DescribeSubnets",
    "ec2:DescribeVolumes",
    "ec2:DescribeVpcEndpoints",
    "logs:DescribeLogGroups",
    "cloudwatch:DescribeAlarms",
}
_SUPPORT_FUNCTION_NAMES = {
    "keep-glm52-h1g-support-tls-handler",
    "keep-glm52-h1g-support-decision",
    "keep-glm52-h1g-support-attestation",
    "keep-glm52-h1g-support-launch-admission",
    "keep-glm52-h1g-support-numeric-binding",
    "keep-glm52-h1g-support-retained-cancellation",
    "keep-glm52-h1g-support-deadline",
}
_SUPPORT_ROLE_NAMES = {
    "keep-glm52-h1g-support-combined-host",
    "keep-glm52-h1g-support-decision",
    "keep-glm52-h1g-support-attestation",
    "keep-glm52-h1g-support-launch-admission",
    "keep-glm52-h1g-support-numeric-binding",
    "keep-glm52-h1g-support-retained-cancellation",
    "keep-glm52-h1g-support-tls-handler",
    "keep-glm52-h1g-support-workflow",
    "keep-glm52-h1g-support-deadline",
    "keep-glm52-h1g-support-schedule-invoke",
}


@dataclass(frozen=True)
class RetainedFenceBootstrapInputs:
    """Manifest-independent authority for the retained fence bootstrap runtime."""

    schema_version: int
    record_type: str
    activation_id: str
    retained_stack_id: str
    model_bucket_arn: str
    retained_kms_key_arn: str
    lambda_code_bucket: str
    lambda_code_key: str
    lambda_code_version_id: str
    lambda_code_sha256: str
    stage_operator_role_arn: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 2
            or self.record_type != "glm52_h1g_retained_fence_bootstrap_inputs_v2"
            or self.activation_id != "glm52-v2-amber-quartz"
            or not self.retained_stack_id.startswith(
                f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/"
                f"{RETAINED_STACK_NAME}/"
            )
            or self.model_bucket_arn != f"arn:aws:s3:::{MODEL_BUCKET_NAME}"
            or not self.retained_kms_key_arn.startswith(
                f"arn:aws:kms:{REGION}:{ACCOUNT_ID}:key/"
            )
            or self.lambda_code_bucket != MODEL_BUCKET_NAME
            or not self.lambda_code_key.startswith("task13/artifacts/support-lambda/")
            or ".." in self.lambda_code_key
            or _VERSION_ID.fullmatch(self.lambda_code_version_id) is None
            or _SHA256.fullmatch(self.lambda_code_sha256) is None
            or not self.stage_operator_role_arn.startswith(
                f"arn:aws:iam::{ACCOUNT_ID}:role/"
            )
        ):
            raise ValueError("retained fence bootstrap inputs are not exact")


def retained_fence_bootstrap_inputs_from_mapping(
    value: object,
) -> RetainedFenceBootstrapInputs:
    fields = {
        "schema_version",
        "record_type",
        "activation_id",
        "retained_stack_id",
        "model_bucket_arn",
        "retained_kms_key_arn",
        "lambda_code_bucket",
        "lambda_code_key",
        "lambda_code_version_id",
        "lambda_code_sha256",
        "stage_operator_role_arn",
    }
    if type(value) is not dict or set(value) != fields:
        raise ValueError("retained fence bootstrap input fields are not exact")
    return RetainedFenceBootstrapInputs(**value)


@dataclass(frozen=True)
class RetainedFenceRuntimeInputs:
    """Manifest-bound inputs for retained execution and settlement identities."""

    schema_version: int
    record_type: str
    activation_id: str
    retained_stack_id: str
    fence_stack_id: str
    fence_service_role_arn: str
    bootstrap_manifest_coordinate: Mapping[str, object]
    model_bucket_arn: str
    retained_kms_key_arn: str
    ledger_table_arn: str
    lambda_code_bucket: str
    lambda_code_key: str
    lambda_code_version_id: str
    lambda_code_sha256: str

    def __post_init__(self) -> None:
        from .fence_artifacts import (
            BOOTSTRAP_PUBLICATION_ORDER,
            parse_artifact_coordinate,
        )

        coordinate = parse_artifact_coordinate(
            json.loads(canonical_json_bytes(self.bootstrap_manifest_coordinate))
        )
        if (
            self.schema_version != 2
            or self.record_type != "glm52_h1g_retained_fence_runtime_inputs_v2"
            or self.activation_id != "glm52-v2-amber-quartz"
            or not self.retained_stack_id.startswith(
                f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/"
                f"{RETAINED_STACK_NAME}/"
            )
            or not self.fence_stack_id.startswith(
                f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/"
                f"{FENCE_STACK_NAME}/"
            )
            or self.fence_service_role_arn
            != f"arn:aws:iam::{ACCOUNT_ID}:role/keep-glm52-h1g-fence-service"
            or coordinate.bucket != MODEL_BUCKET_NAME
            or coordinate.key != BOOTSTRAP_PUBLICATION_ORDER[-1]
            or self.model_bucket_arn != f"arn:aws:s3:::{MODEL_BUCKET_NAME}"
            or not self.retained_kms_key_arn.startswith(
                f"arn:aws:kms:{REGION}:{ACCOUNT_ID}:key/"
            )
            or self.ledger_table_arn
            != (
                f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/keep-glm52-h1g-ledger-v1"
            )
            or self.lambda_code_bucket != MODEL_BUCKET_NAME
            or not self.lambda_code_key.startswith("task13/artifacts/support-lambda/")
            or ".." in self.lambda_code_key
            or _VERSION_ID.fullmatch(self.lambda_code_version_id) is None
            or _SHA256.fullmatch(self.lambda_code_sha256) is None
        ):
            raise ValueError("retained fence runtime inputs are not exact")
        object.__setattr__(
            self,
            "bootstrap_manifest_coordinate",
            MappingProxyType(coordinate.to_dict()),
        )


def retained_fence_runtime_inputs_from_mapping(
    value: object,
) -> RetainedFenceRuntimeInputs:
    fields = {
        "schema_version",
        "record_type",
        "activation_id",
        "retained_stack_id",
        "fence_stack_id",
        "fence_service_role_arn",
        "bootstrap_manifest_coordinate",
        "model_bucket_arn",
        "retained_kms_key_arn",
        "ledger_table_arn",
        "lambda_code_bucket",
        "lambda_code_key",
        "lambda_code_version_id",
        "lambda_code_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise ValueError("retained fence runtime input fields are not exact")
    return RetainedFenceRuntimeInputs(**value)


_RETAINED_FENCE_RUNTIME_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "activation_id",
        "retained_stack_id",
        "fence_service_role_arn",
        "model_bucket_arn",
        "retained_kms_key_arn",
        "ledger_table_arn",
        "lambda_code_bucket",
        "lambda_code_key",
        "lambda_code_version_id",
        "lambda_code_sha256",
    }
)


def retained_fence_runtime_authority_from_mapping(
    value: object,
) -> Mapping[str, object]:
    """Validate manifest-independent retained-runtime intent."""

    if type(value) is not dict or set(value) != _RETAINED_FENCE_RUNTIME_AUTHORITY_FIELDS:
        raise ValueError("retained fence runtime authority fields are not exact")
    if value.get("record_type") != "glm52_h1g_retained_fence_runtime_authority_v2":
        raise ValueError("retained fence runtime authority identity is not exact")
    probe = {
        **dict(value),
        "record_type": "glm52_h1g_retained_fence_runtime_inputs_v2",
        "fence_stack_id": (
            f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/"
            f"{FENCE_STACK_NAME}/00000000-0000-0000-0000-000000000000"
        ),
        "bootstrap_manifest_coordinate": {
            "bucket": MODEL_BUCKET_NAME,
            "key": (
                f"campaigns/{RUN_ID}/authorities/fence/manifests/"
                "glm52-v2-amber-quartz/00000001/FENCE_BOOTSTRAP_MANIFEST.json"
            ),
            "version_id": "authority-validation",
            "file_sha256": "0" * 64,
            "canonical_identity_sha256": "1" * 64,
        },
    }
    retained_fence_runtime_inputs_from_mapping(probe)
    return MappingProxyType(dict(value))


def materialize_retained_fence_runtime_inputs(
    *,
    authority: object,
    fence_stack_id: str,
    bootstrap_manifest_coordinate: Mapping[str, object],
) -> dict[str, object]:
    """Bind retained-runtime intent to exact migration/publication readbacks."""

    parsed = retained_fence_runtime_authority_from_mapping(authority)
    result = {
        **dict(parsed),
        "record_type": "glm52_h1g_retained_fence_runtime_inputs_v2",
        "fence_stack_id": fence_stack_id,
        "bootstrap_manifest_coordinate": json.loads(
            canonical_json_bytes(bootstrap_manifest_coordinate)
        ),
    }
    retained_fence_runtime_inputs_from_mapping(result)
    return result


@dataclass(frozen=True)
class PreSupportRuntimeInputs:
    """Authenticated retained/fence inputs available before TerminalV2 exists."""

    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    activation_id: str
    retained_stack_id: str
    retained_foundation_readback_sha256: str
    fence_manifest_coordinate: Mapping[str, object]
    future_support_version_absence: Mapping[str, object]
    retained_kms_key_arn: str
    ledger_table_arn: str
    model_bucket_name: str
    model_bucket_arn: str
    lambda_code_bucket: str
    lambda_code_key: str
    lambda_code_version_id: str
    lambda_code_sha256: str
    retained_export_names: tuple[str, ...]


@dataclass(frozen=True)
class SupportInputs:
    """Authenticated exact environment inputs for one support activation."""

    schema_version: int
    record_type: str
    account_id: str
    region: str
    run_id: str
    activation_id: str
    nat_gateway_id: str
    support_stack_id: str
    fence_stack_name: str
    fence_stack_id: str
    fence_service_role_arn: str
    fence_manifest_coordinate: Mapping[str, object]
    fence_template_inventory: tuple[Mapping[str, object], ...]
    task11_writer_bindings: tuple[Mapping[str, object], ...]
    organizations_id: str
    retained_stack_id: str
    retained_vpc_id: str
    retained_vpc_cidr: str
    existing_subnet_cidrs: tuple[str, ...]
    existing_secondary_cidrs: tuple[str, ...]
    primary_az: str
    alternate_az: str
    primary_public_subnet_id: str
    retained_public_s3_endpoint_id: str
    retained_kms_key_arn: str
    retained_kms_key_id: str
    ledger_table_name: str
    ledger_table_arn: str
    model_bucket_name: str
    model_bucket_arn: str
    model_prefix: str
    host_ami_id: str
    host_private_ip: str
    host_user_data: str
    host_user_data_sha256: str
    host_boot_identity_sha256: str
    root_volume_gib: int
    root_volume_type: str
    root_volume_iops: int
    root_volume_throughput_mibps: int
    cryptography_layer_arn: str
    cryptography_layer_sha256: str
    lambda_code_bucket: str
    lambda_code_key: str
    lambda_code_version_id: str
    lambda_code_sha256: str
    attestation_port: int
    launch_admission_port: int
    numeric_binding_port: int
    retained_cancellation_port: int
    activation_started_at: str
    runtime_credential_cutoff_at: str
    support_deletion_inventory: Mapping[str, object]
    price_card_identity_sha256: str
    retained_function_version_bindings: tuple[Mapping[str, object], ...]
    retained_export_names: tuple[str, ...]


@dataclass(frozen=True)
class PriceTerm:
    term: str
    unit: str
    quantity: str
    unit_price_usd: str
    estimated_usd: str


@dataclass(frozen=True)
class SupportPriceCard:
    schema_version: int
    record_type: str
    region: str
    effective_date: str
    price_card_identity_sha256: str
    currency: str
    support_terms: tuple[PriceTerm, ...]
    retained_terms: tuple[PriceTerm, ...]
    estimated_support_total_usd: str


@dataclass(frozen=True)
class SupportPlaneBundle:
    inputs: SupportInputs
    price_card: SupportPriceCard
    support_template: Mapping[str, object]
    retained_augmentation: Mapping[str, object]
    spend_descriptor: Mapping[str, object]
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class SupportPrecreateBundle:
    """Logical phase artifacts produced before any support resource exists."""

    inputs: SupportBuildInputs
    price_card: SupportPriceCard
    support_template: Mapping[str, object]
    retained_runtime_fragment: Mapping[str, object]
    spend_descriptor: Mapping[str, object]
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class SupportMaterializationServices:
    """Injected read-only AWS boundaries for post-create materialization."""

    cloudformation: object
    ec2: object


@dataclass(frozen=True)
class SupportPostcreateBundle:
    """Retained augmentation bound only after authenticated materialization."""

    inputs: SupportInputs
    postcreate_retained_fragment: Mapping[str, object]
    retained_augmentation: Mapping[str, object]
    manifest: Mapping[str, object]
    task9_deployed_identity: Mapping[str, object]


_SUPPORT_RESOURCE_ACTION_FAMILIES = {
    "AWS::CloudWatch::Alarm": (
        "cloudwatch:DescribeAlarms",
        "cloudwatch:DeleteAlarms",
    ),
    "AWS::EC2::EIP": ("ec2:DescribeAddresses", "ec2:ReleaseAddress"),
    "AWS::EC2::Instance": (
        "ec2:DescribeInstances",
        "ec2:TerminateInstances",
    ),
    "AWS::EC2::NatGateway": (
        "ec2:DescribeNatGateways",
        "ec2:DeleteNatGateway",
    ),
    "AWS::EC2::Route": ("ec2:DescribeRouteTables", "ec2:DeleteRoute"),
    "AWS::EC2::RouteTable": (
        "ec2:DescribeRouteTables",
        "ec2:DisassociateRouteTable",
        "ec2:DeleteRouteTable",
    ),
    "AWS::EC2::SecurityGroup": (
        "ec2:DescribeSecurityGroups",
        "ec2:RevokeSecurityGroupEgress",
        "ec2:RevokeSecurityGroupIngress",
        "ec2:DeleteSecurityGroup",
    ),
    "AWS::EC2::SecurityGroupEgress": (
        "ec2:DescribeSecurityGroups",
        "ec2:RevokeSecurityGroupEgress",
    ),
    "AWS::EC2::SecurityGroupIngress": (
        "ec2:DescribeSecurityGroups",
        "ec2:RevokeSecurityGroupIngress",
    ),
    "AWS::EC2::Subnet": ("ec2:DescribeSubnets", "ec2:DeleteSubnet"),
    "AWS::EC2::SubnetRouteTableAssociation": (
        "ec2:DescribeRouteTables",
        "ec2:DisassociateRouteTable",
    ),
    "AWS::EC2::VPCEndpoint": (
        "ec2:DescribeVpcEndpoints",
        "ec2:DeleteVpcEndpoints",
    ),
    "AWS::EC2::Volume": (
        "ec2:DescribeVolumes",
        "ec2:DetachVolume",
        "ec2:DeleteVolume",
    ),
    "AWS::EC2::VolumeAttachment": (
        "ec2:DescribeInstances",
        "ec2:DescribeVolumes",
        "ec2:DetachVolume",
    ),
    "AWS::Events::Rule": (
        "events:DescribeRule",
        "events:ListTargetsByRule",
        "events:RemoveTargets",
        "events:DeleteRule",
    ),
    "AWS::IAM::InstanceProfile": (
        "iam:GetInstanceProfile",
        "iam:RemoveRoleFromInstanceProfile",
        "iam:DeleteInstanceProfile",
    ),
    "AWS::IAM::Role": (
        "iam:GetRole",
        "iam:ListRolePolicies",
        "iam:DeleteRolePolicy",
        "iam:DeleteRole",
    ),
    "AWS::Lambda::EventInvokeConfig": (
        "lambda:GetFunction",
        "lambda:DeleteFunctionEventInvokeConfig",
    ),
    "AWS::Lambda::Function": (
        "lambda:GetFunction",
        "lambda:ListVersionsByFunction",
        "lambda:DeleteFunction",
    ),
    "AWS::Lambda::Permission": (
        "lambda:GetFunction",
        "lambda:RemovePermission",
    ),
    "AWS::Lambda::Version": (
        "lambda:GetFunction",
        "lambda:DeleteFunction",
    ),
    "AWS::Logs::LogGroup": (
        "logs:DescribeLogGroups",
        "logs:DeleteLogGroup",
    ),
    "AWS::S3::Bucket": (
        "s3:GetBucketLocation",
        "s3:ListBucketVersions",
        "s3:ListBucket",
        "s3:DeleteObject",
        "s3:DeleteObjectVersion",
        "s3:DeleteBucket",
    ),
    "AWS::S3::BucketPolicy": (
        "s3:GetBucketPolicy",
        "s3:DeleteBucketPolicy",
    ),
    "AWS::SQS::Queue": ("sqs:GetQueueAttributes", "sqs:DeleteQueue"),
    "AWS::Scheduler::Schedule": (
        "scheduler:GetSchedule",
        "scheduler:DeleteSchedule",
    ),
    "AWS::SecretsManager::ResourcePolicy": (
        "secretsmanager:DescribeSecret",
        "secretsmanager:DeleteResourcePolicy",
    ),
    "AWS::SecretsManager::Secret": (
        "secretsmanager:DescribeSecret",
        "secretsmanager:DeleteSecret",
    ),
    "AWS::StepFunctions::StateMachine": (
        "states:DescribeStateMachine",
        "states:DeleteStateMachine",
    ),
    "AWS::StepFunctions::StateMachineVersion": (
        "states:DescribeStateMachine",
        "states:DeleteStateMachineVersion",
    ),
    "Custom::H1gSkyBootstrap": ("lambda:InvokeFunction",),
    "Custom::H1gTlsBundle": ("lambda:InvokeFunction",),
}


class SupportLifecycleState(Enum):
    ACTIVE = "ACTIVE"
    WORK_STOPPED = "WORK_STOPPED"
    DELETE_REQUESTED = "DELETE_REQUESTED"
    DELETE_DEADLINE_MISSED = "DELETE_DEADLINE_MISSED"
    ABSENT = "ABSENT"


@dataclass(frozen=True)
class SupportLifecycleDecision:
    state: SupportLifecycleState
    elapsed_seconds: int
    host_stopped: bool
    egress_disabled: bool
    delete_stack_requested: bool
    page_operator: bool
    incident: str | None
    read_only_stack_reconciliations: int
    direct_child_deletions: tuple[str, ...]


@dataclass(frozen=True)
class SupportEgressDecision:
    host_non_endpoint_bytes: int
    nat_processed_bytes: int
    warning: bool
    drain: bool
    egress_disabled: bool
    authority_exceeded: bool
    work_authorized: bool
    bootstrap_grace_complete: bool
    bootstrap_grace_deadline: str
    reason: str


@dataclass(frozen=True)
class RetainedDeleteDecision:
    mutation: str | None
    mutation_parameters: Mapping[str, object] | None
    reads: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class NatScheduleRetirementDecision:
    mutation: str | None
    mutation_parameters: Mapping[str, object] | None
    reads: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class NatAccumulatorState:
    activation_id: str
    nat_gateway_id: str
    counter_table_name: str
    counter_partition_key: str
    counter_id: str
    support_counter_contract_sha256: str
    last_window_end: str
    cumulative_nat_processed_bytes: int
    accepted_windows: int
    last_observation_sha256: str | None


_SUPPORT_RUNTIME_EXPECTED_FIELDS = frozenset(
    {
        "authority_class",
        "function_logical_id",
        "role_logical_id",
        "function_code_sha256",
        "execution_role_arn",
        "role_trust_policy_sha256",
        "role_permission_policy_sha256",
        "expected_attachment_identity_sha256",
        "disabled_support_profile_sha256",
        "canonical_identity_sha256",
    }
)
_SUPPORT_RUNTIME_LIVE_FIELDS = frozenset(
    {
        "function_version_arn",
        "function_code_sha256",
        "execution_role_arn",
        "execution_role_id",
        "role_trust_policy_sha256",
        "role_permission_policy_sha256",
        "stack_id",
        "stack_template_sha256",
        "resource_policy_sha256",
        "event_source_state_sha256",
        "function_url_state_sha256",
        "expected_attachment_identity_sha256",
        "observed_attachment_identity_sha256",
        "disabled_support_profile_sha256",
    }
)
_SUPPORT_RUNTIME_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "run_id",
        "activation_id",
        "generation",
        "bootstrap_manifest_coordinate",
        "bootstrap_expected_contract_sha256",
        "expected_contract",
        "live_identity",
        "canonical_identity_sha256",
    }
)
_SUPPORT_RUNTIME_FUNCTION_VERSION_ARN = re.compile(
    rf"^arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:"
    r"keep-glm52-h1g-fence-executor:[1-9][0-9]*$"
)
_SUPPORT_RUNTIME_ROLE_ARN = (
    f"arn:aws:iam::{ACCOUNT_ID}:role/keep-glm52-h1g-fence-executor"
)
_SUPPORT_RUNTIME_ROLE_ID = re.compile(r"^AROA[A-Z0-9]{16,}$")
_SUPPORT_RUNTIME_TABLE_NAME = re.compile(r"^[A-Za-z0-9_.-]{3,255}$")
_SUPPORT_RUNTIME_ACTIVATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _support_runtime_freeze(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {str(key): _support_runtime_freeze(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_support_runtime_freeze(item) for item in value)
    return value


def _support_runtime_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _support_runtime_thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_support_runtime_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class SupportRuntimeIdentity:
    """One immutable post-operation-7 support runtime identity."""

    _value: Mapping[str, object]

    @property
    def activation_id(self) -> str:
        return str(self._value["activation_id"])

    @property
    def generation(self) -> int:
        return int(self._value["generation"])

    @property
    def canonical_identity_sha256(self) -> str:
        return str(self._value["canonical_identity_sha256"])

    @property
    def expected_contract(self) -> Mapping[str, object]:
        return self._value["expected_contract"]  # type: ignore[return-value]

    @property
    def live_identity(self) -> Mapping[str, object]:
        return self._value["live_identity"]  # type: ignore[return-value]

    def to_dict(self) -> dict[str, object]:
        return _support_runtime_thaw(self._value)  # type: ignore[return-value]


def _support_runtime_sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} is not one lowercase SHA-256")
    return value


def _support_runtime_expected_contract(
    value: object,
) -> Mapping[str, object]:
    if type(value) is not dict or set(value) != _SUPPORT_RUNTIME_EXPECTED_FIELDS:
        raise ValueError("support runtime expected contract field set drifted")
    contract = json.loads(canonical_json_bytes(value))
    identity = contract.pop("canonical_identity_sha256")
    if (
        contract["authority_class"] != "SUPPORT_RUNTIME"
        or contract["function_logical_id"] != "FenceExecutorFunction"
        or contract["role_logical_id"] != "FenceExecutorRole"
        or contract["execution_role_arn"] != _SUPPORT_RUNTIME_ROLE_ARN
        or canonical_sha256(contract) != identity
    ):
        raise ValueError("support runtime expected contract identity drifted")
    for field in (
        "function_code_sha256",
        "role_trust_policy_sha256",
        "role_permission_policy_sha256",
        "expected_attachment_identity_sha256",
        "disabled_support_profile_sha256",
    ):
        _support_runtime_sha(contract[field], f"support runtime {field}")
    contract["canonical_identity_sha256"] = identity
    return _support_runtime_freeze(contract)  # type: ignore[return-value]


def _support_runtime_live_identity(
    value: object,
    *,
    expected: Mapping[str, object],
) -> Mapping[str, object]:
    if type(value) is not dict or set(value) != _SUPPORT_RUNTIME_LIVE_FIELDS:
        raise ValueError("support runtime live identity field set drifted")
    live = json.loads(canonical_json_bytes(value))
    for field in (
        "function_code_sha256",
        "role_trust_policy_sha256",
        "role_permission_policy_sha256",
        "stack_template_sha256",
        "resource_policy_sha256",
        "event_source_state_sha256",
        "function_url_state_sha256",
        "expected_attachment_identity_sha256",
        "observed_attachment_identity_sha256",
        "disabled_support_profile_sha256",
    ):
        _support_runtime_sha(live[field], f"support runtime live {field}")
    if (
        type(live["function_version_arn"]) is not str
        or _SUPPORT_RUNTIME_FUNCTION_VERSION_ARN.fullmatch(live["function_version_arn"])
        is None
        or live["execution_role_arn"] != _SUPPORT_RUNTIME_ROLE_ARN
        or type(live["execution_role_id"]) is not str
        or _SUPPORT_RUNTIME_ROLE_ID.fullmatch(live["execution_role_id"]) is None
        or type(live["stack_id"]) is not str
        or not live["stack_id"].startswith(
            f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/{SUPPORT_STACK_NAME}/"
        )
    ):
        raise ValueError("support runtime live AWS identity drifted")
    for field in (
        "function_code_sha256",
        "execution_role_arn",
        "role_trust_policy_sha256",
        "role_permission_policy_sha256",
        "expected_attachment_identity_sha256",
        "disabled_support_profile_sha256",
    ):
        if live[field] != expected[field]:
            raise ValueError("support runtime expected/live equality drifted")
    if (
        live["observed_attachment_identity_sha256"]
        != expected["expected_attachment_identity_sha256"]
    ):
        raise ValueError("support runtime attachment equality drifted")
    return _support_runtime_freeze(live)  # type: ignore[return-value]


def build_support_runtime_identity(
    *,
    activation_id: str,
    generation: int,
    bootstrap_manifest_coordinate: object,
    expected_contract: object,
    live_readback: object,
) -> SupportRuntimeIdentity:
    """Build the closed identity only from expected and fresh live truth."""

    from .fence_artifacts import parse_artifact_coordinate

    if (
        type(activation_id) is not str
        or _SUPPORT_RUNTIME_ACTIVATION.fullmatch(activation_id) is None
        or type(generation) is not int
        or generation <= 0
    ):
        raise ValueError("support runtime activation coordinate drifted")
    coordinate = parse_artifact_coordinate(bootstrap_manifest_coordinate)
    expected_key = (
        f"campaigns/{RUN_ID}/authorities/fence/manifests/"
        f"{activation_id}/{generation:08d}/FENCE_BOOTSTRAP_MANIFEST.json"
    )
    if coordinate.key != expected_key:
        raise ValueError("support runtime bootstrap manifest coordinate drifted")
    expected = _support_runtime_expected_contract(expected_contract)
    live = _support_runtime_live_identity(live_readback, expected=expected)
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_runtime_identity_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": activation_id,
        "generation": generation,
        "bootstrap_manifest_coordinate": coordinate.to_dict(),
        "bootstrap_expected_contract_sha256": expected["canonical_identity_sha256"],
        "expected_contract": _support_runtime_thaw(expected),
        "live_identity": _support_runtime_thaw(live),
    }
    body["canonical_identity_sha256"] = canonical_sha256(body)
    return SupportRuntimeIdentity(
        _support_runtime_freeze(body)  # type: ignore[arg-type]
    )


def parse_support_runtime_identity(value: object) -> SupportRuntimeIdentity:
    """Parse one closed support-runtime identity artifact."""

    if type(value) is not dict or set(value) != _SUPPORT_RUNTIME_FIELDS:
        raise ValueError("support runtime identity field set drifted")
    body = json.loads(canonical_json_bytes(value))
    identity = body.pop("canonical_identity_sha256")
    if (
        body["schema_version"] != 1
        or body["record_type"] != "glm52_h1g_support_runtime_identity_v1"
        or body["account_id"] != ACCOUNT_ID
        or body["region"] != REGION
        or body["run_id"] != RUN_ID
        or canonical_sha256(body) != identity
    ):
        raise ValueError("support runtime canonical identity drifted")
    rebuilt = build_support_runtime_identity(
        activation_id=body["activation_id"],
        generation=body["generation"],
        bootstrap_manifest_coordinate=body["bootstrap_manifest_coordinate"],
        expected_contract=body["expected_contract"],
        live_readback=body["live_identity"],
    )
    if (
        rebuilt.canonical_identity_sha256 != identity
        or body["bootstrap_expected_contract_sha256"]
        != rebuilt.expected_contract["canonical_identity_sha256"]
    ):
        raise ValueError("support runtime bootstrap contract binding drifted")
    return rebuilt


def require_live_support_runtime_identity(
    identity: SupportRuntimeIdentity,
    *,
    expected_contract: object,
    live_readback: object,
) -> SupportRuntimeIdentity:
    """Require current support truth to equal the committed immutable record."""

    if type(identity) is not SupportRuntimeIdentity:
        raise TypeError("support runtime identity must be exact and typed")
    expected = _support_runtime_expected_contract(expected_contract)
    live = _support_runtime_live_identity(live_readback, expected=expected)
    if expected != identity.expected_contract or live != identity.live_identity:
        raise ValueError("fresh support runtime live equality drifted")
    return identity


def _support_runtime_item(
    identity: SupportRuntimeIdentity,
    *,
    coordinate: object,
) -> dict[str, object]:
    from .fence_artifacts import parse_support_runtime_identity_coordinate

    if type(identity) is not SupportRuntimeIdentity:
        raise TypeError("support runtime identity must be exact and typed")
    parsed = parse_support_runtime_identity_coordinate(coordinate)
    expected_key = (
        f"campaigns/{RUN_ID}/authorities/fence/runtime/"
        f"{identity.activation_id}/{identity.generation:08d}/"
        "SUPPORT_RUNTIME_IDENTITY.json"
    )
    if (
        parsed.key != expected_key
        or parsed.canonical_identity_sha256 != identity.canonical_identity_sha256
    ):
        raise ValueError("support runtime artifact coordinate drifted")
    return {
        "PK": RUN_ID,
        "SK": (
            f"ACTIVATION#{identity.activation_id}#"
            f"SUPPORT_RUNTIME_IDENTITY#{identity.generation:08d}"
        ),
        **identity.to_dict(),
        "artifact_coordinate": parsed.to_dict(),
    }


def _support_runtime_metadata(response: object, label: str) -> None:
    if (
        type(response) is not dict
        or type(response.get("ResponseMetadata")) is not dict
        or response["ResponseMetadata"].get("HTTPStatusCode") != 200
        or type(response["ResponseMetadata"].get("RequestId")) is not str
        or not response["ResponseMetadata"]["RequestId"]
        or response["ResponseMetadata"].get("RetryAttempts") != 0
    ):
        raise ValueError(label + " response is unauthenticated")


def commit_support_runtime_identity(
    identity: SupportRuntimeIdentity,
    *,
    coordinate: object,
    dynamodb: object,
    table_name: str,
) -> Mapping[str, object]:
    """Conditionally commit once and adopt only an equal lost response."""

    from .dynamodb import decode_item, encode_item

    if (
        type(table_name) is not str
        or _SUPPORT_RUNTIME_TABLE_NAME.fullmatch(table_name) is None
    ):
        raise ValueError("support runtime ledger table name drifted")
    expected = _support_runtime_item(identity, coordinate=coordinate)
    outcome = "WRITTEN"
    try:
        response = dynamodb.put_item(
            TableName=table_name,
            Item=encode_item(expected),
            ConditionExpression=(
                "attribute_not_exists(PK) AND attribute_not_exists(SK)"
            ),
            ReturnConsumedCapacity="NONE",
        )
        _support_runtime_metadata(
            response,
            "support runtime identity PutItem",
        )
    except Exception:
        outcome = "RECONCILED"
    response = dynamodb.get_item(
        TableName=table_name,
        Key=encode_item({"PK": expected["PK"], "SK": expected["SK"]}),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _support_runtime_metadata(response, "support runtime identity GetItem")
    if (
        type(response.get("Item")) is not dict
        or decode_item(response["Item"]) != expected
    ):
        raise ValueError("different or foreign support runtime identity already exists")
    return MappingProxyType(
        {
            "outcome": outcome,
            "partition_key": expected["PK"],
            "sort_key": expected["SK"],
            "canonical_identity_sha256": (identity.canonical_identity_sha256),
            "artifact_coordinate": expected["artifact_coordinate"],
        }
    )


def publish_support_runtime_identity(
    identity: SupportRuntimeIdentity,
    *,
    services: object,
    kms_key_arn: str,
) -> object:
    """Publish the one fixed runtime-identity artifact and return its coordinate."""

    from .fence_artifacts import SupportRuntimeIdentityCoordinate
    from .task13_fixed_artifacts import publish_fixed_key_bytes

    if type(identity) is not SupportRuntimeIdentity:
        raise TypeError("support runtime identity must be exact and typed")
    key = (
        f"campaigns/{RUN_ID}/authorities/fence/runtime/"
        f"{identity.activation_id}/{identity.generation:08d}/"
        "SUPPORT_RUNTIME_IDENTITY.json"
    )
    raw = canonical_json_bytes(identity.to_dict()) + b"\n"
    version_id = publish_fixed_key_bytes(
        services=services,
        bucket=f"keep-glm52-models-{ACCOUNT_ID}-{REGION}",
        key=key,
        raw=raw,
        record_type="glm52_h1g_support_runtime_identity_v1",
        sse_kms_key_id=kms_key_arn,
    )
    return SupportRuntimeIdentityCoordinate(
        bucket=f"keep-glm52-models-{ACCOUNT_ID}-{REGION}",
        key=key,
        version_id=version_id,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_identity_sha256=identity.canonical_identity_sha256,
    )


def load_support_runtime_identity(
    *,
    dynamodb: object,
    table_name: str,
    activation_id: str,
    generation: int,
) -> tuple[SupportRuntimeIdentity, object]:
    """Strongly load the one committed identity and its exact S3 coordinate."""

    from .dynamodb import decode_item, encode_item
    from .fence_artifacts import parse_support_runtime_identity_coordinate

    if (
        type(table_name) is not str
        or _SUPPORT_RUNTIME_TABLE_NAME.fullmatch(table_name) is None
        or type(activation_id) is not str
        or _SUPPORT_RUNTIME_ACTIVATION.fullmatch(activation_id) is None
        or type(generation) is not int
        or generation <= 0
    ):
        raise ValueError("support runtime ledger coordinate drifted")
    key = {
        "PK": RUN_ID,
        "SK": (f"ACTIVATION#{activation_id}#SUPPORT_RUNTIME_IDENTITY#{generation:08d}"),
    }
    response = dynamodb.get_item(
        TableName=table_name,
        Key=encode_item(key),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _support_runtime_metadata(response, "support runtime identity GetItem")
    if type(response.get("Item")) is not dict:
        raise ValueError("support runtime identity is absent")
    decoded = decode_item(response["Item"])
    if (
        type(decoded) is not dict
        or decoded.get("PK") != key["PK"]
        or decoded.get("SK") != key["SK"]
    ):
        raise ValueError("support runtime identity ledger key drifted")
    identity_value = {
        field: decoded[field] for field in _SUPPORT_RUNTIME_FIELDS if field in decoded
    }
    if set(identity_value) != _SUPPORT_RUNTIME_FIELDS:
        raise ValueError("support runtime identity ledger body is incomplete")
    identity = parse_support_runtime_identity(identity_value)
    coordinate = parse_support_runtime_identity_coordinate(
        decoded.get("artifact_coordinate")
    )
    expected = _support_runtime_item(identity, coordinate=coordinate.to_dict())
    if decoded != expected:
        raise ValueError("support runtime identity ledger body drifted")
    return identity, coordinate


def _exact_string(value: object, label: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise ValueError(f"{label} must be one nonempty ASCII string")
    return value


def assert_no_private_material(value: object, label: str) -> str:
    """Reject closed high-signal private credential/key encodings."""

    text = _exact_string(value, label)
    if "\x00" in text or any(
        pattern.search(text) for pattern in _PRIVATE_MATERIAL_PATTERNS
    ):
        raise ValueError(f"{label} contains private material")
    return text


def _exact_string_tuple(value: object, label: str) -> tuple[str, ...]:
    if (
        type(value) is not list
        or not value
        or any(
            type(item) is not str or not item or not item.isascii() for item in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{label} must be one unique nonempty ASCII array")
    return tuple(value)


def _retained_export_names(value: object) -> tuple[str, ...]:
    exports = _exact_string_tuple(value, "retained export names")
    if not _REQUIRED_RETAINED_EXPORT_NAMES.issubset(exports):
        missing = sorted(_REQUIRED_RETAINED_EXPORT_NAMES - set(exports))
        raise ValueError(
            "retained export inventory lacks required exports: " + ", ".join(missing)
        )
    return exports


def _retained_function_version_bindings(
    value: object,
    *,
    lambda_code_sha256: object,
    retained_export_names: tuple[str, ...],
) -> tuple[Mapping[str, object], ...]:
    if (
        type(lambda_code_sha256) is not str
        or _SHA256.fullmatch(lambda_code_sha256) is None
        or type(value) is not list
        or len(value) != len(_RETAINED_FUNCTION_VERSION_SPECS)
    ):
        raise ValueError("retained function version bindings are not exact")
    expected_fields = {
        "output_key",
        "export_name",
        "version_arn",
        "function_name",
        "version",
        "code_sha256",
    }
    result = []
    for row, (output_key, export_name, function_name) in zip(
        value,
        _RETAINED_FUNCTION_VERSION_SPECS,
        strict=True,
    ):
        if type(row) is not dict or set(row) != expected_fields:
            raise ValueError("retained function binding schema is not exact")
        version = row.get("version")
        expected_version_arn = (
            f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{function_name}:{version}"
        )
        if (
            row.get("output_key") != output_key
            or row.get("export_name") != export_name
            or row.get("function_name") != function_name
            or type(version) is not str
            or re.fullmatch(r"[1-9][0-9]*", version) is None
            or row.get("version_arn") != expected_version_arn
            or row.get("code_sha256") != lambda_code_sha256
            or export_name not in retained_export_names
        ):
            raise ValueError("retained function binding identity is not exact")
        result.append(deepcopy(row))
    return tuple(result)


def _stack_id(value: object, name: str) -> str:
    stack_id = _exact_string(value, f"{name} stack ID")
    prefix = f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/{name}/"
    suffix = stack_id.removeprefix(prefix)
    if not stack_id.startswith(prefix) or _UUID.fullmatch(suffix) is None:
        raise ValueError(f"{name} stack ID is not exact")
    return stack_id


def _fence_coordinates(
    value: Mapping[str, object],
) -> tuple[str, str, str]:
    fence_stack_name = _exact_string(value["fence_stack_name"], "fence stack name")
    if fence_stack_name != FENCE_STACK_NAME:
        raise ValueError("fence stack name is not the approved exact name")
    fence_stack_id = _stack_id(value["fence_stack_id"], fence_stack_name)
    fence_service_role_arn = _exact_string(
        value["fence_service_role_arn"], "fence service role ARN"
    )
    expected_service_role = (
        f"arn:aws:iam::{ACCOUNT_ID}:role/keep-glm52-h1g-fence-service"
    )
    if fence_service_role_arn != expected_service_role:
        raise ValueError("fence service role ARN is not exact")
    return fence_stack_name, fence_stack_id, fence_service_role_arn


def _organization_coordinate(value: Mapping[str, object]) -> str:
    organization_id = _exact_string(value["organizations_id"], "Organizations ID")
    if organization_id != ORGANIZATION_ID:
        raise ValueError("Organizations identity is not exact")
    return organization_id


_FENCE_TEMPLATE_SLOTS = (
    "PREPARE_GENESIS_LIVE_STATE",
    "BATCH_FIVE_SOURCE_ACTIVATION",
    "RESERVATION_ONLY",
    "CLOSED_SOURCE",
    "TERMINAL",
)


def _fence_change_set_name(activation_id: str, slot: str) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "run_id": RUN_ID,
                "activation_id": activation_id,
                "generation": 1,
                "slot": slot,
            }
        )
    ).hexdigest()[:16]
    return "glm52-h1g-00000001-" + slot.lower().replace("_", "-") + "-" + digest


def _rehearsal_session_name(activation_id: str) -> str:
    return (
        "h1g-rehearsal-"
        + hashlib.sha256(activation_id.encode("ascii")).hexdigest()[:16]
    )


def _fence_manifest_coordinate(
    value: object,
    *,
    activation_id: str,
    model_bucket_name: str,
) -> Mapping[str, object]:
    coordinate = value
    expected_manifest_key = (
        f"campaigns/{RUN_ID}/authorities/fence/manifests/"
        f"{activation_id}/00000001/FENCE_TEMPLATE_MANIFEST.json"
    )
    if (
        type(coordinate) is not dict
        or set(coordinate)
        != {
            "input_kind",
            "bucket",
            "key",
            "version_id",
            "file_sha256",
            "body_sha256",
            "canonical_identity_sha256",
        }
        or coordinate["input_kind"] != "FENCE_EXECUTION_REQUEST"
        or coordinate["bucket"] != model_bucket_name
        or coordinate["key"] != expected_manifest_key
        or type(coordinate["version_id"]) is not str
        or _VERSION_ID.fullmatch(coordinate["version_id"]) is None
        or type(coordinate["file_sha256"]) is not str
        or _SHA256.fullmatch(coordinate["file_sha256"]) is None
        or type(coordinate["body_sha256"]) is not str
        or _SHA256.fullmatch(coordinate["body_sha256"]) is None
        or coordinate["canonical_identity_sha256"]
        != hashlib.sha256(
            canonical_json_bytes(
                {
                    key: coordinate[key]
                    for key in (
                        "input_kind",
                        "bucket",
                        "key",
                        "version_id",
                        "file_sha256",
                        "body_sha256",
                    )
                }
            )
        ).hexdigest()
    ):
        raise ValueError("fence template manifest coordinate is not exact")
    return deepcopy(coordinate)


def _fence_inventory(
    value: Mapping[str, object],
    *,
    activation_id: str,
) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]:
    bucket = value["model_bucket_name"]
    coordinate = _fence_manifest_coordinate(
        value["fence_manifest_coordinate"],
        activation_id=activation_id,
        model_bucket_name=bucket,
    )
    inventory = value["fence_template_inventory"]
    required_fields = {
        "slot",
        "template_key",
        "template_url",
        "version_id",
        "file_sha256",
        "body_sha256",
        "policy_sha256",
        "change_set_name",
        "change_set_arn",
    }
    if (
        type(inventory) is not list
        or len(inventory) != len(_FENCE_TEMPLATE_SLOTS)
        or any(type(item) is not dict for item in inventory)
        or tuple(item.get("slot") for item in inventory) != _FENCE_TEMPLATE_SLOTS
    ):
        raise ValueError("fence template inventory is not the finite slot set")
    result = []
    for item in inventory:
        slot = item["slot"]
        expected_key = (
            f"campaigns/{RUN_ID}/authorities/fence/templates/"
            f"{activation_id}/00000001/{slot}.json"
        )
        expected_name = _fence_change_set_name(activation_id, slot)
        expected_url = (
            f"https://{bucket}.s3.{REGION}.amazonaws.com/"
            f"{expected_key}?versionId={item.get('version_id')}"
        )
        change_set_prefix = (
            f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:changeSet/{expected_name}/"
        )
        change_set_arn = item.get("change_set_arn")
        change_set_id = (
            change_set_arn.removeprefix(change_set_prefix)
            if type(change_set_arn) is str
            else ""
        )
        if (
            set(item) != required_fields
            or item["template_key"] != expected_key
            or item["template_url"] != expected_url
            or type(item["version_id"]) is not str
            or _VERSION_ID.fullmatch(item["version_id"]) is None
            or item["change_set_name"] != expected_name
            or type(change_set_arn) is not str
            or not change_set_arn.startswith(change_set_prefix)
            or _UUID.fullmatch(change_set_id) is None
            or any(
                type(item[field]) is not str or _SHA256.fullmatch(item[field]) is None
                for field in (
                    "file_sha256",
                    "body_sha256",
                    "policy_sha256",
                )
            )
        ):
            raise ValueError("fence template entry is not finite and exact")
        result.append(deepcopy(item))
    return deepcopy(coordinate), tuple(result)


_TASK11_WRITER_IDS = (
    "SourceGpuSpend",
    "SourceSubmissionIntent",
    "SourceControllerBaseline",
    "SourceControlPlaneReadiness",
    "SourceSubmissionAcquisition",
    "FenceSuccessor",
    "ClaimWriter",
    "DecisionWriter",
    "TerminalV1Writer",
    "ClosureHandoff",
)


def _task11_writer_bindings(
    value: Mapping[str, object],
    *,
    activation_id: str,
    fence_manifest_key: str,
) -> tuple[Mapping[str, object], ...]:
    bindings = value["task11_writer_bindings"]
    fields = {
        "role_id",
        "activation_id",
        "generation",
        "index",
        "action_key",
        "input_key",
        "output_key",
        "read_keys",
        "ledger_leading_keys",
    }
    if (
        type(bindings) is not list
        or len(bindings) != len(_TASK11_WRITER_IDS)
        or any(type(item) is not dict for item in bindings)
        or tuple(item.get("role_id") for item in bindings) != _TASK11_WRITER_IDS
    ):
        raise ValueError("Task 11 writer binding inventory is not exact")
    sha = r"[0-9a-f]{64}"
    base = rf"campaigns/{re.escape(RUN_ID)}/"
    output_patterns = {
        "SourceGpuSpend": (base + rf"spend-snapshots/{sha}/GPU_SPEND_SNAPSHOT[.]json"),
        "SourceSubmissionIntent": (
            base + rf"submissions/production/intents/{sha}/"
            r"SKYPILOT_SUBMISSION_INTENT[.]json"
        ),
        "SourceControllerBaseline": (
            base + rf"production/controller-baselines/{sha}/"
            r"CONTROLLER_BASELINE[.]json"
        ),
        "SourceControlPlaneReadiness": (
            base
            + rf"monitor/must-start/production/{sha}/"
            + rf"control-plane-ready/{sha}/CONTROL_PLANE_READY[.]json"
        ),
        "SourceSubmissionAcquisition": (
            base + rf"submissions/production/acquisitions/{sha}/"
            r"SUBMISSION_ACQUIRED[.]json"
        ),
        "FenceSuccessor": (
            base + rf"authorities/fence/successors/{sha}/"
            r"FENCE_SUCCESSOR[.]json"
        ),
        "ClaimWriter": (
            base + r"submissions/production/generations/00000001/"
            r"GENERATION_CLAIM[.]json"
        ),
        "DecisionWriter": (
            base + r"submissions/production/generations/00000001/"
            r"START_DECISION[.]json"
        ),
        "TerminalV1Writer": (
            base + r"submissions/production/generations/00000001/"
            r"GENERATION_TERMINAL[.]json"
        ),
        "ClosureHandoff": (
            base + r"submissions/production/generations/00000001/handoff/"
            r"SKY_POST_HANDOFF[.]json"
        ),
    }
    result = []
    for index, item in enumerate(bindings):
        role_id = _TASK11_WRITER_IDS[index]
        input_key = (
            f"campaigns/{RUN_ID}/authorities/task11/{activation_id}/"
            f"00000001/{index:02d}-{role_id}/REQUEST.json"
        )
        action_key = f"ACTIVATION#{activation_id}#TASK11#{index:02d}#{role_id}"
        if (
            set(item) != fields
            or item["activation_id"] != activation_id
            or item["generation"] != 1
            or item["index"] != index
            or item["action_key"] != action_key
            or item["input_key"] != input_key
            or type(item["output_key"]) is not str
            or re.fullmatch(output_patterns[role_id], item["output_key"]) is None
            or item["read_keys"] != [input_key, fence_manifest_key]
            or item["ledger_leading_keys"] != [f"RUN#{RUN_ID}"]
        ):
            raise ValueError("Task 11 writer binding drifted")
        result.append(deepcopy(item))
    if len({item["output_key"] for item in result}) != len(result):
        raise ValueError("Task 11 writer outputs are not disjoint")
    return tuple(result)


def _parse_utc(value: object, label: str) -> str:
    text = _exact_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not an RFC3339 UTC instant") from exc
    if (
        not text.endswith("Z")
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
        or parsed.microsecond
    ):
        raise ValueError(f"{label} is not an exact whole-second UTC instant")
    return text


def _materialization_response(response: object, operation: str) -> Mapping[str, object]:
    if type(response) is not dict:
        raise ValueError(f"{operation} readback is not one object")
    metadata = response.get("ResponseMetadata")
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
    ):
        raise ValueError(f"{operation} lacks authenticated success metadata")
    return response


def _materialization_method(service: object, name: str) -> object:
    method = getattr(service, name, None)
    if not callable(method):
        raise ValueError(f"postcreate materializer lacks {name}")
    return method


def _postcreate_action_resource(
    *,
    logical_id: str,
    resource_type: str,
    action: str,
    physical_id: str,
    template: Mapping[str, object],
    physical_by_logical: Mapping[str, str],
) -> tuple[str, ...]:
    if action in _UNSCOPED_SUPPORT_DELETION_READ_ACTIONS:
        return ("*",)
    properties = template["Resources"][logical_id].get("Properties", {})

    def referenced_physical(property_name: str) -> str | None:
        value = properties.get(property_name)
        if type(value) is dict and set(value) == {"Ref"}:
            return physical_by_logical.get(value["Ref"])
        return None

    if action.startswith("ec2:"):
        kind_by_type = {
            "AWS::EC2::EIP": "elastic-ip",
            "AWS::EC2::Instance": "instance",
            "AWS::EC2::NatGateway": "natgateway",
            "AWS::EC2::RouteTable": "route-table",
            "AWS::EC2::SecurityGroup": "security-group",
            "AWS::EC2::Subnet": "subnet",
            "AWS::EC2::VPCEndpoint": "vpc-endpoint",
            "AWS::EC2::Volume": "volume",
        }
        target_ids = []
        kind = kind_by_type.get(resource_type)
        if resource_type == "AWS::EC2::Route":
            kind = "route-table"
            target_ids = [referenced_physical("RouteTableId")]
        elif resource_type == "AWS::EC2::SubnetRouteTableAssociation":
            if action == "ec2:DisassociateRouteTable":
                return tuple(
                    f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:{target_kind}/{target}"
                    for target_kind, target in (
                        ("route-table", referenced_physical("RouteTableId")),
                        ("subnet", referenced_physical("SubnetId")),
                    )
                    if target is not None
                )
        elif resource_type in {
            "AWS::EC2::SecurityGroupEgress",
            "AWS::EC2::SecurityGroupIngress",
        }:
            kind = "security-group"
            target_ids = [referenced_physical("GroupId")]
        elif resource_type == "AWS::EC2::VolumeAttachment":
            targets = (
                ("instance", referenced_physical("InstanceId")),
                ("volume", referenced_physical("VolumeId")),
            )
            return tuple(
                f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:{target_kind}/{target}"
                for target_kind, target in targets
                if target is not None
            )
        if kind is None:
            raise ValueError(f"unclassified EC2 deletion resource: {logical_id}")
        if not target_ids:
            target_ids = [physical_id]
        return tuple(
            f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:{kind}/{target}"
            for target in target_ids
            if target is not None
        )
    if action.startswith("lambda:"):
        if resource_type.startswith("Custom::"):
            target = physical_by_logical.get("TlsHandlerVersion")
        elif resource_type in {
            "AWS::Lambda::Permission",
            "AWS::Lambda::EventInvokeConfig",
        }:
            target = referenced_physical("FunctionName")
        else:
            target = physical_id
        if type(target) is not str:
            raise ValueError("Lambda deletion resource is unresolved")
        arn = (
            target
            if target.startswith(f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:")
            else f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{target}"
        )
        return (arn,)
    if action.startswith("iam:"):
        if resource_type == "AWS::IAM::InstanceProfile":
            profile = (
                physical_id
                if physical_id.startswith("arn:aws:iam::")
                else f"arn:aws:iam::{ACCOUNT_ID}:instance-profile/{physical_id}"
            )
            if action == "iam:RemoveRoleFromInstanceProfile":
                roles = tuple(
                    f"arn:aws:iam::{ACCOUNT_ID}:role/{candidate}"
                    for candidate_logical, candidate in physical_by_logical.items()
                    if template["Resources"][candidate_logical]["Type"]
                    == "AWS::IAM::Role"
                    and not candidate.startswith("arn:aws:iam::")
                )
                return roles + (profile,)
            return (profile,)
        role = (
            physical_id
            if physical_id.startswith("arn:aws:iam::")
            else f"arn:aws:iam::{ACCOUNT_ID}:role/{physical_id}"
        )
        return (role,)
    if action.startswith("s3:"):
        bucket_id = (
            referenced_physical("Bucket")
            if resource_type == "AWS::S3::BucketPolicy"
            else physical_id
        )
        if type(bucket_id) is not str:
            raise ValueError("S3 deletion resource is unresolved")
        bucket = f"arn:aws:s3:::{bucket_id}"
        return (
            (f"{bucket}/*",)
            if action in {"s3:DeleteObject", "s3:DeleteObjectVersion"}
            else (bucket,)
        )
    if action.startswith("logs:"):
        return (f"arn:aws:logs:{REGION}:{ACCOUNT_ID}:log-group:{physical_id}:*",)
    if action.startswith("cloudwatch:"):
        return (f"arn:aws:cloudwatch:{REGION}:{ACCOUNT_ID}:alarm:{physical_id}",)
    if action.startswith("sqs:"):
        queue_name = physical_id.rsplit("/", 1)[-1]
        return (f"arn:aws:sqs:{REGION}:{ACCOUNT_ID}:{queue_name}",)
    if action.startswith("secretsmanager:"):
        if resource_type == "AWS::SecretsManager::ResourcePolicy":
            target = referenced_physical("SecretId")
        else:
            target = physical_id
        if type(target) is not str:
            raise ValueError("secret deletion resource is unresolved")
        return (target,)
    if action.startswith("states:"):
        return (physical_id,)
    if action.startswith("scheduler:"):
        schedule_name = physical_id.rsplit("/", 1)[-1]
        return (
            f"arn:aws:scheduler:{REGION}:{ACCOUNT_ID}:schedule/default/{schedule_name}",
        )
    if action.startswith("events:"):
        rule_name = physical_id.rsplit("/", 1)[-1]
        return (f"arn:aws:events:{REGION}:{ACCOUNT_ID}:rule/{rule_name}",)
    raise ValueError(f"unclassified support deletion action: {action}")


def _materialize_support_postcreate(
    *,
    template: Mapping[str, object],
    expected_support_stack_id: str,
    services: SupportMaterializationServices,
) -> Mapping[str, object]:
    """Read every physical coordinate from CFN, then derive service children."""

    if (
        type(services) is not SupportMaterializationServices
        or type(template) is not dict
        or type(template.get("Resources")) is not dict
    ):
        raise TypeError("postcreate materialization requires exact services/template")
    resources = template["Resources"]
    if any(
        type(logical_id) is not str
        or type(resource) is not dict
        or resource.get("Type") not in _SUPPORT_RESOURCE_ACTION_FAMILIES
        for logical_id, resource in resources.items()
    ):
        raise ValueError("template contains an unclassified deletion resource")
    if any(
        resource.get("Type") == "AWS::EC2::Snapshot" for resource in resources.values()
    ):
        raise ValueError("deletion-time snapshot cannot exist at postcreate")

    describe_stacks = _materialization_method(
        services.cloudformation, "describe_stacks"
    )
    raw_stack = _materialization_response(
        describe_stacks(StackName=expected_support_stack_id),
        "DescribeStacks",
    )
    stacks = raw_stack.get("Stacks")
    if type(stacks) is not list or len(stacks) != 1 or type(stacks[0]) is not dict:
        raise ValueError("support stack readback is absent or duplicated")
    stack = stacks[0]
    stack_id = _stack_id(stack.get("StackId"), SUPPORT_STACK_NAME)
    if (
        stack_id != expected_support_stack_id
        or stack.get("StackName") != SUPPORT_STACK_NAME
        or stack.get("StackStatus")
        not in {
            "CREATE_COMPLETE",
            "UPDATE_COMPLETE",
        }
    ):
        raise ValueError("support stack is not one stable created stack")
    output_rows = stack.get("Outputs")
    expected_output_keys = set(template.get("Outputs", {}))
    if type(output_rows) is not list or any(
        type(row) is not dict or set(row) != {"OutputKey", "OutputValue"}
        for row in output_rows
    ):
        raise ValueError("support stack TLS callback outputs are absent")
    stack_outputs = {row["OutputKey"]: row["OutputValue"] for row in output_rows}
    if set(stack_outputs) != expected_output_keys or len(stack_outputs) != len(
        output_rows
    ):
        raise ValueError("support stack TLS callback outputs drifted")

    list_stack_resources = _materialization_method(
        services.cloudformation, "list_stack_resources"
    )
    summaries = []
    list_response_identities = []
    next_token: str | None = None
    while True:
        request: dict[str, object] = {"StackName": stack_id}
        if next_token is not None:
            request["NextToken"] = next_token
        raw_page = _materialization_response(
            list_stack_resources(**request),
            "ListStackResources",
        )
        page = raw_page.get("StackResourceSummaries")
        if type(page) is not list or any(type(row) is not dict for row in page):
            raise ValueError("CFN resource summary page is malformed")
        summaries.extend(page)
        list_response_identities.append(
            hashlib.sha256(canonical_json_bytes(raw_page)).hexdigest()
        )
        token = raw_page.get("NextToken")
        if token is None:
            break
        next_token = _exact_string(token, "CFN resource page token")
        if len(list_response_identities) > 100:
            raise ValueError("CFN resource pagination is unbounded")
    summaries_by_logical: dict[str, Mapping[str, object]] = {}
    for summary in summaries:
        logical_id = summary.get("LogicalResourceId")
        if type(logical_id) is not str or logical_id in summaries_by_logical:
            raise ValueError("CFN resource summary is duplicated")
        summaries_by_logical[logical_id] = summary
    if set(summaries_by_logical) != set(resources):
        raise ValueError("CFN resource summary is incomplete or has extras")

    describe_stack_resource = _materialization_method(
        services.cloudformation, "describe_stack_resource"
    )
    materialized_rows = []
    detail_by_logical: dict[str, Mapping[str, object]] = {}
    for logical_id in sorted(resources):
        expected_type = resources[logical_id]["Type"]
        raw_detail = _materialization_response(
            describe_stack_resource(
                StackName=stack_id,
                LogicalResourceId=logical_id,
            ),
            "DescribeStackResource",
        )
        detail = raw_detail.get("StackResourceDetail")
        summary = summaries_by_logical[logical_id]
        physical_id = detail.get("PhysicalResourceId") if type(detail) is dict else None
        if (
            type(detail) is not dict
            or detail.get("StackId") != stack_id
            or detail.get("StackName") != SUPPORT_STACK_NAME
            or detail.get("LogicalResourceId") != logical_id
            or detail.get("ResourceType") != expected_type
            or type(detail.get("PhysicalResourceId")) is not str
            or not detail["PhysicalResourceId"]
            or type(detail.get("ResourceStatus")) is not str
            or not detail["ResourceStatus"].endswith("_COMPLETE")
            or detail["ResourceStatus"].startswith("DELETE_")
            or summary.get("LogicalResourceId") != logical_id
            or summary.get("ResourceType") != expected_type
            or summary.get("PhysicalResourceId") != detail["PhysicalResourceId"]
            or summary.get("ResourceStatus") != detail["ResourceStatus"]
        ):
            raise ValueError(f"CFN detail is not exact for {logical_id}")
        properties = resources[logical_id].get("Properties", {})
        expected_physical: str | None = None
        if expected_type == "AWS::S3::Bucket":
            expected_physical = properties.get("BucketName")
        elif expected_type == "AWS::IAM::Role":
            expected_physical = properties.get("RoleName")
        elif expected_type == "AWS::CloudWatch::Alarm":
            expected_physical = properties.get("AlarmName")
        if expected_physical is not None and (
            type(expected_physical) is not str or physical_id != expected_physical
        ):
            raise ValueError(f"CFN physical identity was substituted for {logical_id}")
        if expected_type == "AWS::SQS::Queue":
            queue_name = properties.get("QueueName")
            if (
                type(queue_name) is not str
                or type(physical_id) is not str
                or not physical_id.endswith(f"/{queue_name}")
            ):
                raise ValueError(
                    f"CFN physical identity was substituted for {logical_id}"
                )
        if expected_type == "AWS::SecretsManager::Secret":
            secret_name = properties.get("Name")
            if (
                type(secret_name) is not str
                or type(physical_id) is not str
                or re.fullmatch(
                    rf"arn:aws:secretsmanager:{REGION}:{ACCOUNT_ID}:"
                    rf"secret:{re.escape(secret_name)}-[A-Za-z0-9]{{6}}",
                    physical_id,
                )
                is None
            ):
                raise ValueError(
                    f"CFN physical identity was substituted for {logical_id}"
                )
        detail_by_logical[logical_id] = detail
        materialized_rows.append(
            {
                "logical_id": logical_id,
                "resource_type": expected_type,
                "physical_id": detail["PhysicalResourceId"],
                "source_api": "DescribeStackResource",
                "source_response_sha256": hashlib.sha256(
                    canonical_json_bytes(raw_detail)
                ).hexdigest(),
                "action_families": list(
                    _SUPPORT_RESOURCE_ACTION_FAMILIES[expected_type]
                ),
            }
        )

    for logical_id, detail in detail_by_logical.items():
        resource = resources[logical_id]
        resource_type = resource["Type"]
        physical_id = detail["PhysicalResourceId"]
        properties = resource.get("Properties", {})
        if type(physical_id) is not str or type(properties) is not dict:
            raise ValueError("postcreate physical identity is malformed")
        if physical_id.startswith("arn:") and (
            re.match(
                rf"arn:aws:[a-z0-9-]+:{REGION}:{ACCOUNT_ID}:",
                physical_id,
            )
            is None
            and re.match(
                rf"arn:aws:iam::{ACCOUNT_ID}:",
                physical_id,
            )
            is None
            and re.match(
                rf"arn:aws:s3:::{re.escape(physical_id[13:])}\Z",
                physical_id,
            )
            is None
        ):
            raise ValueError(
                f"CFN physical identity has a foreign account for {logical_id}"
            )
        if resource_type == "AWS::Lambda::Function":
            expected_name = properties.get("FunctionName")
            if expected_name is not None and (
                type(expected_name) is not str or physical_id != expected_name
            ):
                raise ValueError(
                    f"Lambda function physical identity was substituted for {logical_id}"
                )
        elif resource_type == "AWS::Lambda::Version":
            function_ref = properties.get("FunctionName")
            function_logical_id = (
                function_ref.get("Ref") if type(function_ref) is dict else None
            )
            function_physical = (
                detail_by_logical.get(function_logical_id, {}).get("PhysicalResourceId")
                if type(function_logical_id) is str
                else None
            )
            if (
                type(function_physical) is not str
                or re.fullmatch(
                    rf"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:"
                    rf"{re.escape(function_physical)}:[1-9][0-9]*",
                    physical_id,
                )
                is None
            ):
                raise ValueError(
                    f"Lambda callback/version identity was substituted for {logical_id}"
                )
        elif resource_type == "AWS::StepFunctions::StateMachine":
            expected_name = properties.get("StateMachineName")
            if (
                expected_name is not None
                and (
                    type(expected_name) is not str
                    or physical_id
                    != (
                        f"arn:aws:states:{REGION}:{ACCOUNT_ID}:"
                        f"stateMachine:{expected_name}"
                    )
                )
            ) or re.fullmatch(
                rf"arn:aws:states:{REGION}:{ACCOUNT_ID}:"
                r"stateMachine:[A-Za-z0-9_-]{1,80}",
                physical_id,
            ) is None:
                raise ValueError(
                    f"state-machine physical identity was substituted for {logical_id}"
                )
        elif resource_type == "AWS::StepFunctions::StateMachineVersion":
            machine_ref = properties.get("StateMachineArn")
            machine_logical_id = (
                machine_ref.get("Ref") if type(machine_ref) is dict else None
            )
            machine_physical = (
                detail_by_logical.get(machine_logical_id, {}).get("PhysicalResourceId")
                if type(machine_logical_id) is str
                else None
            )
            if (
                type(machine_physical) is not str
                or re.fullmatch(
                    rf"{re.escape(machine_physical)}:[1-9][0-9]*",
                    physical_id,
                )
                is None
            ):
                raise ValueError(
                    f"state-machine version identity was substituted for {logical_id}"
                )

    nat_gateway_id = detail_by_logical["NatGateway"]["PhysicalResourceId"]
    if (
        type(nat_gateway_id) is not str
        or re.fullmatch(r"nat-[0-9a-f]{17}", nat_gateway_id) is None
    ):
        raise ValueError("NAT identity is not authenticated CFN readback")
    instance_id = detail_by_logical["CombinedHost"]["PhysicalResourceId"]
    if (
        type(instance_id) is not str
        or re.fullmatch(r"i-[0-9a-f]{17}", instance_id) is None
    ):
        raise ValueError("combined-host identity is not exact CFN readback")
    describe_instances = _materialization_method(services.ec2, "describe_instances")
    raw_instance = _materialization_response(
        describe_instances(InstanceIds=[instance_id]),
        "DescribeInstances",
    )
    reservations = raw_instance.get("Reservations")
    if (
        type(reservations) is not list
        or len(reservations) != 1
        or type(reservations[0]) is not dict
        or type(reservations[0].get("Instances")) is not list
        or len(reservations[0]["Instances"]) != 1
        or type(reservations[0]["Instances"][0]) is not dict
    ):
        raise ValueError("combined-host root-volume readback is not exact")
    instance = reservations[0]["Instances"][0]
    root_device_name = instance.get("RootDeviceName")
    mappings = instance.get("BlockDeviceMappings")
    root_mappings = (
        [
            row
            for row in mappings
            if type(row) is dict and row.get("DeviceName") == root_device_name
        ]
        if type(mappings) is list
        else []
    )
    if (
        instance.get("InstanceId") != instance_id
        or instance.get("ImageId")
        != template["Resources"]["CombinedHost"]["Properties"]["ImageId"]
        or instance.get("PrivateIpAddress")
        != template["Resources"]["CombinedHost"]["Properties"]["NetworkInterfaces"][0][
            "PrivateIpAddress"
        ]
        or instance.get("State", {}).get("Name") != "running"
        or instance.get("IamInstanceProfile", {}).get("Arn")
        != (
            f"arn:aws:iam::{ACCOUNT_ID}:instance-profile/"
            + str(detail_by_logical["CombinedHostProfile"]["PhysicalResourceId"])
        )
        or type(root_device_name) is not str
        or len(root_mappings) != 1
        or type(root_mappings[0].get("Ebs")) is not dict
        or re.fullmatch(
            r"vol-[0-9a-f]{17}",
            str(root_mappings[0]["Ebs"].get("VolumeId")),
        )
        is None
    ):
        raise ValueError("root volume is not authenticated instance readback")
    root_volume_id = root_mappings[0]["Ebs"]["VolumeId"]

    endpoint_id = detail_by_logical["SecretsManagerEndpoint"]["PhysicalResourceId"]
    if (
        type(endpoint_id) is not str
        or re.fullmatch(r"vpce-[0-9a-f]{17}", endpoint_id) is None
    ):
        raise ValueError("interface endpoint identity is not exact CFN readback")
    describe_vpc_endpoints = _materialization_method(
        services.ec2, "describe_vpc_endpoints"
    )
    raw_endpoint = _materialization_response(
        describe_vpc_endpoints(VpcEndpointIds=[endpoint_id]),
        "DescribeVpcEndpoints",
    )
    endpoints = raw_endpoint.get("VpcEndpoints")
    if (
        type(endpoints) is not list
        or len(endpoints) != 1
        or type(endpoints[0]) is not dict
        or endpoints[0].get("VpcEndpointId") != endpoint_id
        or type(endpoints[0].get("NetworkInterfaceIds")) is not list
        or len(endpoints[0]["NetworkInterfaceIds"]) != 2
        or len(set(endpoints[0]["NetworkInterfaceIds"])) != 2
        or any(
            type(eni) is not str or re.fullmatch(r"eni-[0-9a-f]{17}", eni) is None
            for eni in endpoints[0]["NetworkInterfaceIds"]
        )
    ):
        raise ValueError("interface endpoint ENIs are not authenticated readback")
    instance_identity = hashlib.sha256(canonical_json_bytes(raw_instance)).hexdigest()
    endpoint_identity = hashlib.sha256(canonical_json_bytes(raw_endpoint)).hexdigest()
    derived_rows = [
        {
            "resource_kind": "ROOT_VOLUME",
            "physical_id": root_volume_id,
            "parent_logical_id": "CombinedHost",
            "source_api": "DescribeInstances",
            "source_response_sha256": instance_identity,
            "action_families": [
                "ec2:DescribeVolumes",
                "ec2:DeleteVolume",
            ],
        },
        *[
            {
                "resource_kind": "INTERFACE_ENDPOINT_ENI",
                "physical_id": eni,
                "parent_logical_id": "SecretsManagerEndpoint",
                "source_api": "DescribeVpcEndpoints",
                "source_response_sha256": endpoint_identity,
                "action_families": [
                    "ec2:DescribeNetworkInterfaces",
                    "ec2:DeleteNetworkInterface",
                ],
            }
            for eni in sorted(endpoints[0]["NetworkInterfaceIds"])
        ],
    ]
    physical_by_logical = {
        logical_id: str(detail["PhysicalResourceId"])
        for logical_id, detail in detail_by_logical.items()
    }
    action_resource_sets = {action: set() for action in _SUPPORT_DELETION_ACTIONS}
    for row in materialized_rows:
        for action in row["action_families"]:
            action_resource_sets[action].update(
                _postcreate_action_resource(
                    logical_id=row["logical_id"],
                    resource_type=row["resource_type"],
                    action=action,
                    physical_id=row["physical_id"],
                    template=template,
                    physical_by_logical=physical_by_logical,
                )
            )
    data_volume_id = physical_by_logical["CombinedHostDataVolume"]
    data_volume_arn = f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:volume/{data_volume_id}"
    root_volume_arn = f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:volume/{root_volume_id}"
    action_resource_sets["ec2:CreateSnapshot"].add(data_volume_arn)
    action_resource_sets["ec2:DeleteVolume"].add(root_volume_arn)
    action_resource_sets["ec2:DescribeSnapshots"].add("*")
    action_resource_sets["ec2:DescribeNetworkInterfaces"].add("*")
    for eni in endpoints[0]["NetworkInterfaceIds"]:
        action_resource_sets["ec2:DeleteNetworkInterface"].add(
            f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:network-interface/{eni}"
        )
    for action in ("kms:CreateGrant", "kms:RetireGrant", "kms:RevokeGrant"):
        action_resource_sets[action].add(
            template["Resources"]["CombinedHostDataVolume"]["Properties"]["KmsKeyId"]
        )
    if set(action_resource_sets) != _SUPPORT_DELETION_ACTIONS or any(
        not values for values in action_resource_sets.values()
    ):
        raise ValueError("full support deletion action inventory is incomplete")
    action_resources = {
        action: sorted(values)
        for action, values in sorted(action_resource_sets.items())
    }
    snapshot_contract = {
        "state": "CAPTURE_AT_DELETION",
        "source_logical_id": "CombinedHostDataVolume",
        "source_volume_id": detail_by_logical["CombinedHostDataVolume"][
            "PhysicalResourceId"
        ],
        "kms_key_id": template["Resources"]["CombinedHostDataVolume"]["Properties"][
            "KmsKeyId"
        ],
        "required_tags": {
            "ActivationId": template["Metadata"]["ActivationId"],
            "Purpose": "forensic-data-volume-deletion-snapshot",
            "RunId": RUN_ID,
        },
        "snapshot_id": None,
    }
    callback_version = detail_by_logical["TlsHandlerVersion"]["PhysicalResourceId"]
    callback_bindings = []
    for logical_id in (
        "SkyBootstrapCustomResource",
        "TlsBundleCustomResource",
    ):
        if template["Resources"][logical_id]["Properties"].get("ServiceToken") != {
            "Ref": "TlsHandlerVersion"
        }:
            raise ValueError("custom-resource delete callback was substituted")
        callback_bindings.append(
            {
                "logical_id": logical_id,
                "handler_logical_id": "TlsHandlerVersion",
                "handler_physical_id": callback_version,
                "action": "lambda:InvokeFunction",
            }
        )
    inventory: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_postcreate_inventory_v1",
        "support_stack_id": stack_id,
        "support_template_body_sha256": hashlib.sha256(
            canonical_json_bytes(template)
        ).hexdigest(),
        "stack_resources": materialized_rows,
        "derived_resources": derived_rows,
        "custom_resource_delete_callbacks": callback_bindings,
        "snapshot_contract": snapshot_contract,
        "action_resources": action_resources,
        "resource_arns": sorted(
            {
                resource
                for values in action_resources.values()
                for resource in values
                if resource != "*"
            }
        ),
        "resource_ids": sorted(
            {row["physical_id"] for row in materialized_rows + derived_rows}
        ),
        "resource_names": sorted(
            {row["physical_id"].rsplit("/", 1)[-1] for row in materialized_rows}
        ),
        "describe_stacks_response_sha256": hashlib.sha256(
            canonical_json_bytes(raw_stack)
        ).hexdigest(),
        "list_stack_resources_response_sha256": (list_response_identities),
        "tls_callback_outputs": stack_outputs,
        "combined_host_identity": {
            "instance_id": instance_id,
            "private_ip": instance["PrivateIpAddress"],
            "ami_id": instance["ImageId"],
            "instance_profile_arn": instance["IamInstanceProfile"]["Arn"],
            "describe_instances_response_sha256": instance_identity,
        },
    }
    inventory["canonical_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(inventory)
    ).hexdigest()
    return {
        "support_stack_id": stack_id,
        "nat_gateway_id": nat_gateway_id,
        "support_template_body_sha256": inventory["support_template_body_sha256"],
        "stack_resources": tuple(materialized_rows),
        "derived_resources": tuple(derived_rows),
        "snapshot_contract": snapshot_contract,
        "deletion_inventory": inventory,
        "stack_outputs": stack_outputs,
        "combined_host_identity": inventory["combined_host_identity"],
    }


def _validate_support_resource_readback(
    value: object,
    evidence: object,
    *,
    activation_id: str,
    support_stack_id: str,
    resource_arns: object,
    resource_ids: object,
    resource_names: object,
    action_resources: object,
) -> Mapping[str, Mapping[str, object]]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "record_type",
        "activation_id",
        "support_stack_id",
        "observed_at",
        "resources",
        "deletion_inventory_projection",
        "canonical_body_sha256",
    }:
        raise ValueError("support resource readback schema is not exact")
    readback_body = dict(value)
    readback_identity = readback_body.pop("canonical_body_sha256")
    if (
        type(readback_identity) is not str
        or _SHA256.fullmatch(readback_identity) is None
        or hashlib.sha256(canonical_json_bytes(readback_body)).hexdigest()
        != readback_identity
        or readback_body["schema_version"] != 1
        or readback_body["record_type"] != "glm52_h1g_support_resource_readback_v1"
        or readback_body["activation_id"] != activation_id
        or readback_body["support_stack_id"] != support_stack_id
    ):
        raise ValueError("support resource readback identity is foreign")
    observed_at = _parse_utc(
        readback_body["observed_at"], "support resource readback time"
    )
    try:
        expected_projection = {
            field: hashlib.sha256(canonical_json_bytes(source)).hexdigest()
            for field, source in (
                ("resource_arns_sha256", resource_arns),
                ("resource_ids_sha256", resource_ids),
                ("resource_names_sha256", resource_names),
                ("action_resources_sha256", action_resources),
            )
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("support deletion projection is not canonical") from exc
    if readback_body["deletion_inventory_projection"] != expected_projection:
        raise ValueError("support resource readback does not bind deletion inventory")
    if type(evidence) is not dict or set(evidence) != {
        "schema_version",
        "record_type",
        "support_stack_id",
        "list_stack_resources_request_id",
        "list_stack_resources_response_sha256",
        "describe_instances_request_id",
        "describe_instances_response_sha256",
        "describe_snapshots_request_id",
        "describe_snapshots_response_sha256",
        "observed_at",
        "resource_readback_identity_sha256",
        "canonical_body_sha256",
    }:
        raise ValueError("support resource readback evidence schema is not exact")
    evidence_body = dict(evidence)
    evidence_identity = evidence_body.pop("canonical_body_sha256")
    if (
        type(evidence_identity) is not str
        or _SHA256.fullmatch(evidence_identity) is None
        or hashlib.sha256(canonical_json_bytes(evidence_body)).hexdigest()
        != evidence_identity
        or evidence_body["schema_version"] != 1
        or evidence_body["record_type"]
        != "glm52_h1g_support_resource_readback_evidence_v1"
        or evidence_body["support_stack_id"] != support_stack_id
        or evidence_body["observed_at"] != observed_at
        or evidence_body["resource_readback_identity_sha256"] != readback_identity
        or any(
            type(evidence_body[field]) is not str
            or _UUID.fullmatch(evidence_body[field]) is None
            for field in (
                "list_stack_resources_request_id",
                "describe_instances_request_id",
                "describe_snapshots_request_id",
            )
        )
        or any(
            type(evidence_body[field]) is not str
            or _SHA256.fullmatch(evidence_body[field]) is None
            for field in (
                "list_stack_resources_response_sha256",
                "describe_instances_response_sha256",
                "describe_snapshots_response_sha256",
            )
        )
    ):
        raise ValueError("support resource readback evidence is foreign")
    expected: dict[str, tuple[str, str]] = {
        "CombinedHostRootVolume": (
            "AWS::EC2::Volume",
            "DescribeInstances",
        ),
        "CombinedHostDataVolume": (
            "AWS::EC2::Volume",
            "ListStackResources",
        ),
        "CombinedHostDataVolumeDeletionSnapshot": (
            "AWS::EC2::Snapshot",
            "DescribeSnapshots",
        ),
        "NatGateway": (
            "AWS::EC2::NatGateway",
            "ListStackResources",
        ),
        "TlsHandlerVersion": (
            "AWS::Lambda::Version",
            "ListStackResources",
        ),
    }
    expected.update(
        {
            logical_id: (
                "AWS::SecretsManager::Secret",
                "ListStackResources",
            )
            for logical_id in SECRET_LOGICAL_IDS
        }
    )
    rows = readback_body["resources"]
    if (
        type(rows) is not list
        or len(rows) != len(expected)
        or any(type(row) is not dict for row in rows)
    ):
        raise ValueError("support resource readback inventory is incomplete")
    by_logical_id: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if set(row) != {
            "LogicalResourceId",
            "ResourceType",
            "PhysicalResourceId",
            "ResourceArn",
            "SourceApi",
        }:
            raise ValueError("support resource readback row is not exact")
        logical_id = row["LogicalResourceId"]
        spec = expected.get(logical_id)
        if (
            type(logical_id) is not str
            or logical_id in by_logical_id
            or spec is None
            or (row["ResourceType"], row["SourceApi"]) != spec
            or type(row["PhysicalResourceId"]) is not str
            or type(row["ResourceArn"]) is not str
        ):
            raise ValueError("support resource readback row is foreign")
        physical_id = row["PhysicalResourceId"]
        resource_arn = row["ResourceArn"]
        if logical_id in {
            "CombinedHostRootVolume",
            "CombinedHostDataVolume",
        }:
            match = re.fullmatch(
                rf"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:volume/"
                r"(vol-[0-9a-f]{17})",
                resource_arn,
            )
            valid = match is not None and match.group(1) == physical_id
        elif logical_id == "CombinedHostDataVolumeDeletionSnapshot":
            match = re.fullmatch(
                rf"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:snapshot/"
                r"(snap-[0-9a-f]{17})",
                resource_arn,
            )
            valid = match is not None and match.group(1) == physical_id
        elif logical_id == "NatGateway":
            match = re.fullmatch(
                rf"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:natgateway/"
                r"(nat-[0-9a-f]{17})",
                resource_arn,
            )
            valid = match is not None and match.group(1) == physical_id
        elif logical_id == "TlsHandlerVersion":
            valid = (
                physical_id == resource_arn
                and re.fullmatch(
                    rf"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:"
                    r"keep-glm52-h1g-support-tls-handler:[1-9][0-9]*",
                    resource_arn,
                )
                is not None
            )
        else:
            purpose = _SECRET_PURPOSES[logical_id]
            valid = (
                physical_id == resource_arn
                and re.fullmatch(
                    rf"arn:aws:secretsmanager:{REGION}:{ACCOUNT_ID}:secret:"
                    rf"/keep/glm52/{RUN_ID}/{re.escape(activation_id)}/"
                    rf"{re.escape(purpose)}-[A-Za-z0-9]{{6}}",
                    resource_arn,
                )
                is not None
            )
        if not valid:
            raise ValueError("support resource physical identity is foreign")
        by_logical_id[logical_id] = row
    if set(by_logical_id) != set(expected):
        raise ValueError("support resource readback logical inventory drifted")
    return by_logical_id


def _validate_support_deletion_inventory(
    value: object,
    *,
    activation_id: str,
    support_stack_id: str,
    retained_kms_key_arn: str,
) -> Mapping[str, object]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "record_type",
        "activation_id",
        "support_stack_id",
        "resource_arns",
        "resource_ids",
        "resource_names",
        "action_resources",
        "stack_resource_readback",
        "stack_resource_readback_evidence",
        "custom_resource_delete_callbacks",
        "canonical_body_sha256",
    }:
        raise ValueError("support deletion inventory schema is not exact")
    body = dict(value)
    identity = body.pop("canonical_body_sha256")
    if (
        type(identity) is not str
        or _SHA256.fullmatch(identity) is None
        or hashlib.sha256(canonical_json_bytes(body)).hexdigest() != identity
        or body["schema_version"] != 1
        or body["record_type"] != "glm52_h1g_support_deletion_inventory_v1"
        or body["activation_id"] != activation_id
        or body["support_stack_id"] != support_stack_id
    ):
        raise ValueError("support deletion inventory identity is foreign")
    resource_arns = body["resource_arns"]
    resource_ids = body["resource_ids"]
    resource_names = body["resource_names"]
    action_resources = body["action_resources"]
    callbacks = body["custom_resource_delete_callbacks"]
    readback = _validate_support_resource_readback(
        body["stack_resource_readback"],
        body["stack_resource_readback_evidence"],
        activation_id=activation_id,
        support_stack_id=support_stack_id,
        resource_arns=resource_arns,
        resource_ids=resource_ids,
        resource_names=resource_names,
        action_resources=action_resources,
    )
    if (
        type(resource_arns) is not list
        or not resource_arns
        or any(
            type(item) is not str
            or not item.isascii()
            or item == "*"
            or not item.startswith("arn:aws:")
            for item in resource_arns
        )
        or len(resource_arns) != len(set(resource_arns))
        or type(resource_ids) is not list
        or not resource_ids
        or any(
            type(item) is not str
            or re.fullmatch(
                r"(?:i|vol|snap|nat|eni|rtb|subnet|sg|vpce|eipalloc)-"
                r"[0-9a-f]{17}",
                item,
            )
            is None
            for item in resource_ids
        )
        or len(resource_ids) != len(set(resource_ids))
        or type(resource_names) is not list
        or not resource_names
        or any(
            type(item) is not str
            or re.fullmatch(r"[A-Za-z0-9+=,.@_/-]{1,256}", item) is None
            for item in resource_names
        )
        or len(resource_names) != len(set(resource_names))
        or type(action_resources) is not dict
        or set(action_resources) != _SUPPORT_DELETION_ACTIONS
    ):
        raise ValueError("support deletion resource allowlists are not exact")
    referenced = []
    for action, resources in action_resources.items():
        unscoped = action in _UNSCOPED_SUPPORT_DELETION_READ_ACTIONS
        if (
            type(action) is not str
            or type(resources) is not list
            or not resources
            or any(
                type(resource) is not str
                or (unscoped and resource != "*")
                or (not unscoped and (resource == "*" or resource not in resource_arns))
                for resource in resources
            )
            or len(resources) != len(set(resources))
            or (unscoped and resources != ["*"])
        ):
            raise ValueError("support deletion action resources are not exact")
        referenced.extend(resource for resource in resources if resource != "*")
    if set(referenced) != set(resource_arns):
        raise ValueError("support deletion resource inventory is unused or missing")
    if (
        type(callbacks) is not list
        or len(callbacks) != 2
        or any(type(callback) is not dict for callback in callbacks)
        or callbacks
        != [
            {
                "LogicalResourceId": "SkyBootstrapCustomResource",
                "HandlerVersionArn": callbacks[0].get("HandlerVersionArn"),
            },
            {
                "LogicalResourceId": "TlsBundleCustomResource",
                "HandlerVersionArn": callbacks[0].get("HandlerVersionArn"),
            },
        ]
    ):
        raise ValueError("custom-resource deletion callback inventory is not exact")
    handler_version = callbacks[0]["HandlerVersionArn"]
    readback_handler_version = readback["TlsHandlerVersion"]["ResourceArn"]
    readback_secrets = {
        readback[logical_id]["ResourceArn"] for logical_id in SECRET_LOGICAL_IDS
    }
    data_volume_arn = readback["CombinedHostDataVolume"]["ResourceArn"]
    root_volume_arn = readback["CombinedHostRootVolume"]["ResourceArn"]
    nat_gateway_arn = readback["NatGateway"]["ResourceArn"]
    ec2_arn_ids = {
        match.group(1)
        for resource in resource_arns
        for match in [
            re.fullmatch(
                rf"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:[^/]+/"
                r"((?:i|vol|nat|eni|rtb|subnet|sg|vpce|eipalloc)-"
                r"[0-9a-f]{17})",
                resource,
            )
        ]
        if match is not None
    }
    expected_resource_ids = ec2_arn_ids | {
        readback["CombinedHostDataVolumeDeletionSnapshot"]["PhysicalResourceId"]
    }
    expected_names = (
        _SUPPORT_FUNCTION_NAMES
        | _SUPPORT_ROLE_NAMES
        | {_closure_role_name(activation_id)}
        | {
            "keep-glm52-h1g-support-combined-host",
            f"keep-glm52-h1g-rehearsal-{ACCOUNT_ID}-{REGION}",
            f"keep-glm52-h1g-support-dlq-{activation_id}",
        }
    )
    allowed_role_arns = {
        f"arn:aws:iam::{ACCOUNT_ID}:role/{name}"
        for name in (_SUPPORT_ROLE_NAMES | {_closure_role_name(activation_id)})
    }
    profile_arn = (
        f"arn:aws:iam::{ACCOUNT_ID}:instance-profile/"
        "keep-glm52-h1g-support-combined-host"
    )
    if (
        type(handler_version) is not str
        or handler_version != readback_handler_version
        or action_resources["lambda:InvokeFunction"] != [handler_version]
        or set(resource_ids) != expected_resource_ids
        or set(resource_names) != expected_names
        or action_resources["ec2:CreateSnapshot"] != [data_volume_arn]
        or action_resources["ec2:DeleteVolume"] != [data_volume_arn, root_volume_arn]
        or action_resources["ec2:DeleteNatGateway"] != [nat_gateway_arn]
        or any(
            set(action_resources[action]) != readback_secrets
            for action in (
                "secretsmanager:DeleteResourcePolicy",
                "secretsmanager:DeleteSecret",
                "secretsmanager:DescribeSecret",
            )
        )
        or any(
            set(action_resources[action]) != allowed_role_arns
            for action in (
                "iam:DeleteRolePolicy",
                "iam:DeleteRole",
                "iam:GetRole",
                "iam:ListRolePolicies",
            )
        )
        or set(action_resources["iam:RemoveRoleFromInstanceProfile"])
        != allowed_role_arns | {profile_arn}
        or action_resources["iam:DeleteInstanceProfile"] != [profile_arn]
        or action_resources["iam:GetInstanceProfile"] != [profile_arn]
        or any(
            action_resources[action] != [retained_kms_key_arn]
            for action in (
                "kms:CreateGrant",
                "kms:RetireGrant",
                "kms:RevokeGrant",
            )
        )
    ):
        raise ValueError("support deletion callback or KMS authority is foreign")
    for action in (
        "lambda:DeleteFunction",
        "lambda:DeleteFunctionEventInvokeConfig",
        "lambda:RemovePermission",
        "lambda:GetFunction",
        "lambda:ListVersionsByFunction",
    ):
        for resource in action_resources[action]:
            match = re.fullmatch(
                rf"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:"
                r"([A-Za-z0-9-_]+)(?::[1-9][0-9]*)?",
                resource,
            )
            if match is None or match.group(1) not in _SUPPORT_FUNCTION_NAMES:
                raise ValueError("support deletion Lambda identity is foreign")
    return deepcopy(value)


def _future_support_version_absence(
    value: object,
) -> Mapping[str, object]:
    fields = {
        "schema_version",
        "record_type",
        "account_id",
        "region",
        "support_stack_name",
        "observed_at",
        "lambda_function_names",
        "state_machine_names",
        "lambda_get_function_absent",
        "states_describe_state_machine_absent",
        "lambda_readback_sha256",
        "states_readback_sha256",
        "canonical_body_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise ValueError("future support version absence evidence is malformed")
    body = dict(value)
    identity = body.pop("canonical_body_sha256")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["record_type"] != "glm52_h1g_future_support_version_absence_v1"
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["support_stack_name"] != SUPPORT_STACK_NAME
        or value["lambda_function_names"]
        != [
            "keep-glm52-h1g-numeric-binding",
            "keep-glm52-h1g-retained-cancellation",
        ]
        or value["state_machine_names"] != ["keep-glm52-h1g-retained-lifecycle"]
        or value["lambda_get_function_absent"] is not True
        or value["states_describe_state_machine_absent"] is not True
        or type(value["observed_at"]) is not str
        or _parse_utc(
            value["observed_at"],
            "future support absence observation",
        )
        != value["observed_at"]
        or any(
            type(value[field]) is not str or _SHA256.fullmatch(value[field]) is None
            for field in (
                "lambda_readback_sha256",
                "states_readback_sha256",
            )
        )
        or type(identity) is not str
        or _SHA256.fullmatch(identity) is None
        or hashlib.sha256(canonical_json_bytes(body)).hexdigest() != identity
    ):
        raise ValueError("future support version absence evidence is not exact")
    return deepcopy(value)


def pre_support_runtime_inputs_from_mapping(
    value: object,
) -> PreSupportRuntimeInputs:
    """Parse the retained/fence authority available before TerminalV2."""

    if type(value) is not dict or set(value) != set(_PRE_SUPPORT_RUNTIME_INPUT_FIELDS):
        missing = sorted(
            set(_PRE_SUPPORT_RUNTIME_INPUT_FIELDS) - set(value)
            if type(value) is dict
            else ()
        )
        unknown = sorted(
            set(value) - set(_PRE_SUPPORT_RUNTIME_INPUT_FIELDS)
            if type(value) is dict
            else ()
        )
        raise ValueError(
            "pre-support runtime input schema mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["record_type"] != "glm52_h1g_pre_support_runtime_inputs_v1"
        or value["account_id"] != ACCOUNT_ID
        or value["region"] != REGION
        or value["run_id"] != RUN_ID
    ):
        raise ValueError("pre-support runtime input identity is not exact")
    activation_id = _exact_string(value["activation_id"], "activation ID")
    if _ACTIVATION_ID.fullmatch(activation_id) is None:
        raise ValueError("activation ID has unsafe grammar")
    retained_stack_id = _stack_id(value["retained_stack_id"], RETAINED_STACK_NAME)
    foundation_identity = _exact_string(
        value["retained_foundation_readback_sha256"],
        "retained foundation readback SHA-256",
    )
    if _SHA256.fullmatch(foundation_identity) is None:
        raise ValueError("retained foundation readback identity is not exact")
    model_bucket_name = _exact_string(value["model_bucket_name"], "model bucket")
    model_bucket_arn = _exact_string(value["model_bucket_arn"], "model bucket ARN")
    if (
        _BUCKET.fullmatch(model_bucket_name) is None
        or ".." in model_bucket_name
        or model_bucket_arn != f"arn:aws:s3:::{model_bucket_name}"
    ):
        raise ValueError("model bucket identity is not exact")
    fence_manifest_coordinate = _fence_manifest_coordinate(
        value["fence_manifest_coordinate"],
        activation_id=activation_id,
        model_bucket_name=model_bucket_name,
    )
    future_support_version_absence = _future_support_version_absence(
        value["future_support_version_absence"]
    )
    retained_kms_key_arn = _exact_string(
        value["retained_kms_key_arn"], "retained KMS key ARN"
    )
    if (
        re.fullmatch(
            rf"arn:aws:kms:{REGION}:{ACCOUNT_ID}:key/{_UUID.pattern}",
            retained_kms_key_arn,
        )
        is None
    ):
        raise ValueError("retained KMS key identity is not exact")
    ledger_table_arn = _exact_string(value["ledger_table_arn"], "ledger table ARN")
    if ledger_table_arn != (
        f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/keep-glm52-h1g-ledger-v1"
    ):
        raise ValueError("retained ledger identity is not exact")
    lambda_code_bucket = _exact_string(
        value["lambda_code_bucket"], "Lambda code bucket"
    )
    if _BUCKET.fullmatch(lambda_code_bucket) is None or ".." in lambda_code_bucket:
        raise ValueError("Lambda code bucket is not exact")
    lambda_code_key = _exact_string(value["lambda_code_key"], "Lambda code key")
    if lambda_code_key.startswith("/") or ".." in lambda_code_key:
        raise ValueError("Lambda code key is unsafe")
    lambda_code_version = _exact_string(
        value["lambda_code_version_id"], "Lambda code VersionId"
    )
    if _VERSION_ID.fullmatch(lambda_code_version) is None:
        raise ValueError("Lambda code VersionId is not exact")
    lambda_code_sha256 = _exact_string(
        value["lambda_code_sha256"], "Lambda code SHA-256"
    )
    if _SHA256.fullmatch(lambda_code_sha256) is None:
        raise ValueError("Lambda code SHA-256 is not exact")
    retained_exports = _exact_string_tuple(
        value["retained_export_names"], "retained export names"
    )
    missing_exports = _PRE_SUPPORT_REQUIRED_RETAINED_EXPORT_NAMES - set(
        retained_exports
    )
    if missing_exports:
        raise ValueError(
            "pre-support retained export inventory lacks required exports: "
            + ", ".join(sorted(missing_exports))
        )
    if {
        "KeepGlm52Task12TerminalV2VersionArn",
        "KeepGlm52Task12WorkerDrainVersionArn",
        "KeepGlm52Task9LiabilityWatcherVersionArn",
    } & set(retained_exports):
        raise ValueError(
            "pre-support retained export inventory already contains Task 12"
        )
    return PreSupportRuntimeInputs(
        schema_version=1,
        record_type="glm52_h1g_pre_support_runtime_inputs_v1",
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        activation_id=activation_id,
        retained_stack_id=retained_stack_id,
        retained_foundation_readback_sha256=foundation_identity,
        fence_manifest_coordinate=fence_manifest_coordinate,
        future_support_version_absence=future_support_version_absence,
        retained_kms_key_arn=retained_kms_key_arn,
        ledger_table_arn=ledger_table_arn,
        model_bucket_name=model_bucket_name,
        model_bucket_arn=model_bucket_arn,
        lambda_code_bucket=lambda_code_bucket,
        lambda_code_key=lambda_code_key,
        lambda_code_version_id=lambda_code_version,
        lambda_code_sha256=lambda_code_sha256,
        retained_export_names=retained_exports,
    )


def pre_support_runtime_inputs_projection(
    inputs: PreSupportRuntimeInputs,
) -> Mapping[str, object]:
    """Return the exact canonical narrow pre-support authority."""

    if type(inputs) is not PreSupportRuntimeInputs:
        raise TypeError("pre-support runtime projection requires exact inputs")
    return {
        field: (
            list(getattr(inputs, field))
            if field == "retained_export_names"
            else deepcopy(getattr(inputs, field))
        )
        for field in _PRE_SUPPORT_RUNTIME_INPUT_FIELDS
    }


def pre_support_runtime_inputs_identity(
    inputs: PreSupportRuntimeInputs,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(pre_support_runtime_inputs_projection(inputs))
    ).hexdigest()


def support_inputs_from_mapping(value: object) -> SupportInputs:
    """Parse and fail-close one exact support-input projection."""

    if type(value) is not dict or set(value) != set(_INPUT_FIELDS):
        missing = sorted(set(_INPUT_FIELDS) - set(value) if type(value) is dict else ())
        unknown = sorted(set(value) - set(_INPUT_FIELDS) if type(value) is dict else ())
        raise ValueError(
            f"support input schema mismatch: missing={missing}, unknown={unknown}"
        )
    if value["schema_version"] != 1 or type(value["schema_version"]) is not int:
        raise ValueError("support input schema version is not exact")
    if value["record_type"] != "glm52_h1g_support_inputs_v1":
        raise ValueError("support input record type is not exact")
    for field, expected in (
        ("account_id", ACCOUNT_ID),
        ("region", REGION),
        ("run_id", RUN_ID),
    ):
        if type(value[field]) is not str or value[field] != expected:
            raise ValueError(f"{field} is not the frozen campaign coordinate")
    activation_id = _exact_string(value["activation_id"], "activation ID")
    if _ACTIVATION_ID.fullmatch(activation_id) is None:
        raise ValueError("activation ID has unsafe grammar")
    support_stack_id = _stack_id(value["support_stack_id"], SUPPORT_STACK_NAME)
    (
        fence_stack_name,
        fence_stack_id,
        fence_service_role_arn,
    ) = _fence_coordinates(value)
    fence_manifest_coordinate, fence_template_inventory = _fence_inventory(
        value,
        activation_id=activation_id,
    )
    task11_writer_bindings = _task11_writer_bindings(
        value,
        activation_id=activation_id,
        fence_manifest_key=fence_manifest_coordinate["key"],
    )
    organizations_id = _organization_coordinate(value)
    retained_stack_id = _stack_id(value["retained_stack_id"], RETAINED_STACK_NAME)
    retained_vpc_id = _exact_string(value["retained_vpc_id"], "retained VPC ID")
    primary_subnet_id = _exact_string(
        value["primary_public_subnet_id"], "primary public subnet ID"
    )
    retained_s3_endpoint = _exact_string(
        value["retained_public_s3_endpoint_id"],
        "retained public S3 endpoint ID",
    )
    if any(
        _EC2_ID.fullmatch(item) is None
        for item in (retained_vpc_id, primary_subnet_id, retained_s3_endpoint)
    ):
        raise ValueError("retained EC2 identities are not exact")
    try:
        vpc_cidr = ipaddress.ip_network(value["retained_vpc_cidr"], strict=True)
        existing_subnets = tuple(
            str(ipaddress.ip_network(item, strict=True))
            for item in _exact_string_tuple(
                value["existing_subnet_cidrs"], "existing subnet CIDRs"
            )
        )
        secondary_cidrs = tuple(
            str(ipaddress.ip_network(item, strict=True))
            for item in _exact_string_tuple(
                value["existing_secondary_cidrs"], "existing secondary CIDRs"
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("retained network CIDRs are not exact") from exc
    if (
        vpc_cidr.version != 4
        or any(ipaddress.ip_network(item).version != 4 for item in existing_subnets)
        or any(ipaddress.ip_network(item).version != 4 for item in secondary_cidrs)
    ):
        raise ValueError("support network requires exact IPv4 CIDRs")
    primary_az = _exact_string(value["primary_az"], "primary AZ")
    alternate_az = _exact_string(value["alternate_az"], "alternate AZ")
    if (
        primary_az == alternate_az
        or re.fullmatch(r"us-west-2[a-z]", primary_az) is None
        or re.fullmatch(r"us-west-2[a-z]", alternate_az) is None
    ):
        raise ValueError("primary and alternate AZ identities are not exact")
    retained_kms_key_arn = _exact_string(
        value["retained_kms_key_arn"], "retained KMS key ARN"
    )
    retained_kms_key_id = _exact_string(
        value["retained_kms_key_id"], "retained KMS key ID"
    )
    if (
        _UUID.fullmatch(retained_kms_key_id) is None
        or retained_kms_key_arn
        != f"arn:aws:kms:{REGION}:{ACCOUNT_ID}:key/{retained_kms_key_id}"
    ):
        raise ValueError("retained KMS key identity is not exact")
    deletion_inventory = _validate_support_deletion_inventory(
        value["support_deletion_inventory"],
        activation_id=activation_id,
        support_stack_id=support_stack_id,
        retained_kms_key_arn=retained_kms_key_arn,
    )
    nat_gateway_id = _exact_string(value["nat_gateway_id"], "NAT gateway ID")
    nat_readbacks = [
        row
        for row in deletion_inventory["stack_resource_readback"]["resources"]
        if type(row) is dict and row.get("LogicalResourceId") == "NatGateway"
    ]
    if (
        re.fullmatch(r"nat-[0-9a-f]{17}", nat_gateway_id) is None
        or len(nat_readbacks) != 1
        or nat_readbacks[0].get("PhysicalResourceId") != nat_gateway_id
    ):
        raise ValueError("NAT gateway identity is not authenticated resource readback")
    ledger_name = _exact_string(value["ledger_table_name"], "ledger table name")
    ledger_arn = _exact_string(value["ledger_table_arn"], "ledger table ARN")
    if (
        ledger_name != "keep-glm52-h1g-ledger-v1"
        or ledger_arn != f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/{ledger_name}"
    ):
        raise ValueError("retained ledger identity is not exact")
    model_bucket_name = _exact_string(value["model_bucket_name"], "model bucket")
    model_bucket_arn = _exact_string(value["model_bucket_arn"], "model bucket ARN")
    if (
        _BUCKET.fullmatch(model_bucket_name) is None
        or ".." in model_bucket_name
        or model_bucket_arn != f"arn:aws:s3:::{model_bucket_name}"
    ):
        raise ValueError("model bucket identity is not exact")
    model_prefix = _exact_string(value["model_prefix"], "model prefix")
    if (
        model_prefix != f"campaigns/{RUN_ID}/"
        or model_prefix.startswith("/")
        or ".." in model_prefix
    ):
        raise ValueError("model prefix is not exact")
    host_ami_id = _exact_string(value["host_ami_id"], "host AMI")
    if _AMI_ID.fullmatch(host_ami_id) is None:
        raise ValueError("host AMI is not an exact ID")
    try:
        host_ip = ipaddress.ip_address(value["host_private_ip"])
    except (TypeError, ValueError) as exc:
        raise ValueError("host private address is not exact") from exc
    if host_ip.version != 4 or not host_ip.is_private:
        raise ValueError("host address must be private IPv4")
    user_data = assert_no_private_material(value["host_user_data"], "host user data")
    user_data_sha256 = _exact_string(
        value["host_user_data_sha256"], "host user-data SHA-256"
    )
    if (
        _SHA256.fullmatch(user_data_sha256) is None
        or hashlib.sha256(user_data.encode("utf-8")).hexdigest() != user_data_sha256
    ):
        raise ValueError("host user-data digest mismatch")
    for field in (
        "host_boot_identity_sha256",
        "cryptography_layer_sha256",
        "lambda_code_sha256",
        "price_card_identity_sha256",
    ):
        if type(value[field]) is not str or _SHA256.fullmatch(value[field]) is None:
            raise ValueError(f"{field} is not an exact SHA-256")
    for field in (
        "root_volume_gib",
        "root_volume_iops",
        "root_volume_throughput_mibps",
    ):
        if type(value[field]) is not int or value[field] <= 0:
            raise ValueError(f"{field} must be one positive integer")
    if value["root_volume_type"] != "gp3":
        raise ValueError("root volume type must be gp3")
    layer_arn = _exact_string(value["cryptography_layer_arn"], "cryptography layer ARN")
    if (
        re.fullmatch(
            rf"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:"
            r"layer/[A-Za-z0-9-_]+:[1-9][0-9]*",
            layer_arn,
        )
        is None
    ):
        raise ValueError("cryptography layer ARN is not an exact version")
    lambda_bucket = _exact_string(value["lambda_code_bucket"], "Lambda code bucket")
    if _BUCKET.fullmatch(lambda_bucket) is None or ".." in lambda_bucket:
        raise ValueError("Lambda code bucket is not exact")
    lambda_key = _exact_string(value["lambda_code_key"], "Lambda code key")
    if lambda_key.startswith("/") or ".." in lambda_key:
        raise ValueError("Lambda code key is unsafe")
    lambda_version = _exact_string(
        value["lambda_code_version_id"], "Lambda code VersionId"
    )
    if _VERSION_ID.fullmatch(lambda_version) is None:
        raise ValueError("Lambda code VersionId is not exact")
    ports = tuple(
        value[field]
        for field in (
            "attestation_port",
            "launch_admission_port",
            "numeric_binding_port",
            "retained_cancellation_port",
        )
    )
    if ports != PINNED_RELAY_PORTS:
        raise ValueError("relay ports drifted from the pinned Task 9 identity")
    retained_exports = _retained_export_names(value["retained_export_names"])
    retained_function_version_bindings = _retained_function_version_bindings(
        value["retained_function_version_bindings"],
        lambda_code_sha256=value["lambda_code_sha256"],
        retained_export_names=retained_exports,
    )
    activation_started_at = _parse_utc(
        value["activation_started_at"],
        "activation start",
    )
    runtime_credential_cutoff_at = _parse_utc(
        value["runtime_credential_cutoff_at"],
        "runtime credential cutoff",
    )
    if datetime.fromisoformat(
        runtime_credential_cutoff_at.replace("Z", "+00:00")
    ) <= datetime.fromisoformat(activation_started_at.replace("Z", "+00:00")):
        raise ValueError("runtime credential cutoff must follow activation start")
    return SupportInputs(
        schema_version=1,
        record_type="glm52_h1g_support_inputs_v1",
        account_id=ACCOUNT_ID,
        region=REGION,
        run_id=RUN_ID,
        activation_id=activation_id,
        nat_gateway_id=nat_gateway_id,
        support_stack_id=support_stack_id,
        fence_stack_name=fence_stack_name,
        fence_stack_id=fence_stack_id,
        fence_service_role_arn=fence_service_role_arn,
        fence_manifest_coordinate=fence_manifest_coordinate,
        fence_template_inventory=fence_template_inventory,
        task11_writer_bindings=task11_writer_bindings,
        organizations_id=organizations_id,
        retained_stack_id=retained_stack_id,
        retained_vpc_id=retained_vpc_id,
        retained_vpc_cidr=str(vpc_cidr),
        existing_subnet_cidrs=existing_subnets,
        existing_secondary_cidrs=secondary_cidrs,
        primary_az=primary_az,
        alternate_az=alternate_az,
        primary_public_subnet_id=primary_subnet_id,
        retained_public_s3_endpoint_id=retained_s3_endpoint,
        retained_kms_key_arn=retained_kms_key_arn,
        retained_kms_key_id=retained_kms_key_id,
        ledger_table_name=ledger_name,
        ledger_table_arn=ledger_arn,
        model_bucket_name=model_bucket_name,
        model_bucket_arn=model_bucket_arn,
        model_prefix=model_prefix,
        host_ami_id=host_ami_id,
        host_private_ip=str(host_ip),
        host_user_data=user_data,
        host_user_data_sha256=user_data_sha256,
        host_boot_identity_sha256=value["host_boot_identity_sha256"],
        root_volume_gib=value["root_volume_gib"],
        root_volume_type="gp3",
        root_volume_iops=value["root_volume_iops"],
        root_volume_throughput_mibps=value["root_volume_throughput_mibps"],
        cryptography_layer_arn=layer_arn,
        cryptography_layer_sha256=value["cryptography_layer_sha256"],
        lambda_code_bucket=lambda_bucket,
        lambda_code_key=lambda_key,
        lambda_code_version_id=lambda_version,
        lambda_code_sha256=value["lambda_code_sha256"],
        attestation_port=ports[0],
        launch_admission_port=ports[1],
        numeric_binding_port=ports[2],
        retained_cancellation_port=ports[3],
        activation_started_at=activation_started_at,
        runtime_credential_cutoff_at=runtime_credential_cutoff_at,
        support_deletion_inventory=deletion_inventory,
        price_card_identity_sha256=value["price_card_identity_sha256"],
        retained_function_version_bindings=(retained_function_version_bindings),
        retained_export_names=retained_exports,
    )


def support_inputs_projection(inputs: SupportInputs) -> Mapping[str, object]:
    """Return the exact canonical JSON projection for authenticated inputs."""

    if type(inputs) is not SupportInputs:
        raise TypeError("support input projection requires exact SupportInputs")
    return {
        field: (
            [deepcopy(item) for item in getattr(inputs, field)]
            if field
            in {
                "fence_template_inventory",
                "task11_writer_bindings",
                "retained_function_version_bindings",
            }
            else list(getattr(inputs, field))
            if field
            in {
                "existing_subnet_cidrs",
                "existing_secondary_cidrs",
                "retained_export_names",
            }
            else deepcopy(getattr(inputs, field))
        )
        for field in _INPUT_FIELDS
    }


def support_inputs_identity(inputs: SupportInputs) -> str:
    """Hash exactly the canonical support-input projection."""

    return hashlib.sha256(
        canonical_json_bytes(support_inputs_projection(inputs))
    ).hexdigest()


_SUPPORT_PRICE_TERMS = (
    "combined_host_hours",
    "root_ebs",
    "data_ebs",
    "nat_gateway_hours",
    "nat_processed_gib",
    "eip_hours",
    "secrets_eight",
    "secrets_api_calls",
    "interface_endpoint_two_eni_hours",
    "cross_az_bytes",
    "lambda",
    "step_functions",
    "logs_ingestion",
    "logs_storage",
    "rehearsal_s3",
)
_RETAINED_PRICE_TERMS = (
    "retained_s3",
    "retained_ledger",
    "retained_kms",
    "liability_watcher",
    "snapshot_cleanup",
    "snapshot_retention",
)


def _mechanical_price_shape(
    inputs: SupportInputs,
) -> tuple[
    Mapping[str, tuple[str, str]],
    Mapping[str, tuple[str, str]],
]:
    support = {
        "combined_host_hours": ("instance-hour", "72.000000"),
        "root_ebs": (
            "gib-hour",
            f"{inputs.root_volume_gib * 72}.000000",
        ),
        "data_ebs": ("gib-hour", "3600.000000"),
        "nat_gateway_hours": ("gateway-hour", "72.000000"),
        "nat_processed_gib": ("gib", "10.000000"),
        "eip_hours": ("ipv4-hour", "72.000000"),
        "secrets_eight": ("secret-hour", "576.000000"),
        "secrets_api_calls": ("api-call", "20000.000000"),
        "interface_endpoint_two_eni_hours": ("eni-hour", "144.000000"),
        "cross_az_bytes": ("gib", "10.000000"),
        "lambda": ("invocation", "10000.000000"),
        "step_functions": ("state-transition", "12000.000000"),
        "logs_ingestion": ("gib", "0.500000"),
        "logs_storage": ("gib-day", "7.000000"),
        "rehearsal_s3": ("gib-day", "3.000000"),
    }
    retained = {
        "retained_s3": ("gib-day", "30.000000"),
        "retained_ledger": ("request", "129600.000000"),
        "retained_kms": ("key-month", "1.000000"),
        "liability_watcher": ("scan", "43200.000000"),
        "snapshot_cleanup": ("delete-call", "12.000000"),
        "snapshot_retention": ("gib-day", "350.000000"),
    }
    return support, retained


def _money(value: object, label: str) -> Decimal:
    if (
        type(value) is not str
        or re.fullmatch(r"(?:0|[1-9][0-9]*)\.[0-9]{2}", value) is None
    ):
        raise ValueError(f"{label} must be one nonnegative two-decimal USD string")
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{label} is not decimal") from exc


def _positive_decimal(value: object, label: str) -> Decimal:
    if (
        type(value) is not str
        or re.fullmatch(r"(?:0|[1-9][0-9]*)\.[0-9]{1,12}", value) is None
    ):
        raise ValueError(f"{label} must be one fixed-point decimal")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{label} is not decimal") from exc
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _price_terms(
    value: object,
    expected: tuple[str, ...],
    shape: Mapping[str, tuple[str, str]],
    label: str,
) -> tuple[PriceTerm, ...]:
    if type(value) is not list or len(value) != len(expected):
        raise ValueError(f"{label} price terms are not exact")
    result = []
    for index, item in enumerate(value):
        if type(item) is not dict or set(item) != {
            "term",
            "unit",
            "quantity",
            "unit_price_usd",
            "estimated_usd",
        }:
            raise ValueError(f"{label} price term schema is not exact")
        if item["term"] != expected[index]:
            raise ValueError(f"{label} price terms are not in frozen order")
        expected_unit, expected_quantity = shape[item["term"]]
        unit = _exact_string(item["unit"], f"{label} term unit")
        quantity = item["quantity"]
        if (
            type(quantity) is not str
            or re.fullmatch(r"(?:0|[1-9][0-9]*)\.[0-9]{6}", quantity) is None
            or Decimal(quantity) <= 0
            or unit != expected_unit
            or quantity != expected_quantity
        ):
            raise ValueError(f"{label} unit/quantity is not the mechanical shape")
        unit_price = _positive_decimal(item["unit_price_usd"], f"{label} unit price")
        estimate = _money(item["estimated_usd"], f"{label} estimate")
        if (Decimal(quantity) * unit_price).quantize(Decimal("0.01")) != estimate:
            raise ValueError(f"{label} extended price is inconsistent")
        result.append(
            PriceTerm(
                term=item["term"],
                unit=unit,
                quantity=quantity,
                unit_price_usd=item["unit_price_usd"],
                estimated_usd=item["estimated_usd"],
            )
        )
    return tuple(result)


def _support_input_identity(
    inputs: object,
) -> str:
    if type(inputs) is SupportInputs:
        return support_inputs_identity(inputs)
    if type(inputs) is SupportBuildInputs:
        return support_build_inputs_identity(inputs)
    raise TypeError("support input identity requires an exact input type")


def support_price_card_from_mapping(
    value: object, *, inputs: object
) -> SupportPriceCard:
    """Parse the dated exact regional price card and enforce the $25 gate."""

    fields = {
        "schema_version",
        "record_type",
        "region",
        "effective_date",
        "price_card_identity_sha256",
        "currency",
        "support_terms",
        "retained_terms",
        "estimated_support_total_usd",
    }
    if type(value) is not dict or set(value) != fields:
        raise ValueError("support price-card schema is not exact")
    if type(inputs) not in {SupportInputs, SupportBuildInputs}:
        raise TypeError("price-card validation requires exact support inputs")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["record_type"] != "glm52_h1g_support_price_card_v1"
        or value["region"] != REGION
        or value["currency"] != "USD"
        or type(value["effective_date"]) is not str
        or re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}", value["effective_date"])
        is None
    ):
        raise ValueError("support price-card identity is not exact")
    identity = value.get("price_card_identity_sha256")
    unsigned = dict(value)
    unsigned.pop("price_card_identity_sha256", None)
    if (
        type(identity) is not str
        or _SHA256.fullmatch(identity) is None
        or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != identity
        or identity != inputs.price_card_identity_sha256
    ):
        raise ValueError("support price-card bytes are not authenticated")
    support_shape, retained_shape = _mechanical_price_shape(inputs)
    support_terms = _price_terms(
        value["support_terms"],
        _SUPPORT_PRICE_TERMS,
        support_shape,
        "support",
    )
    retained_terms = _price_terms(
        value["retained_terms"],
        _RETAINED_PRICE_TERMS,
        retained_shape,
        "retained",
    )
    total = sum(
        (_money(term.estimated_usd, "support estimate") for term in support_terms),
        Decimal("0.00"),
    )
    declared = _money(value["estimated_support_total_usd"], "declared support total")
    if declared != total or declared > Decimal("25.00"):
        raise ValueError("support estimate is inconsistent or exceeds $25.00")
    return SupportPriceCard(
        schema_version=1,
        record_type="glm52_h1g_support_price_card_v1",
        region=REGION,
        effective_date=value["effective_date"],
        price_card_identity_sha256=identity,
        currency="USD",
        support_terms=support_terms,
        retained_terms=retained_terms,
        estimated_support_total_usd=value["estimated_support_total_usd"],
    )


def validate_support_network_inputs(inputs: object) -> bool:
    """Prove the three fixed support CIDRs fit the authenticated retained VPC."""

    if type(inputs) not in {SupportInputs, SupportBuildInputs}:
        raise TypeError("network validation requires exact support inputs")
    vpc = ipaddress.ip_network(inputs.retained_vpc_cidr, strict=True)
    support = tuple(ipaddress.ip_network(item, strict=True) for item in SUPPORT_CIDRS)
    existing = tuple(
        ipaddress.ip_network(item, strict=True)
        for item in inputs.existing_subnet_cidrs + inputs.existing_secondary_cidrs
    )
    if any(not item.subnet_of(vpc) for item in support):
        raise ValueError("support CIDR is outside the authenticated retained VPC")
    for index, network in enumerate(support):
        if any(network.overlaps(other) for other in support[index + 1 :]):
            raise ValueError("support CIDRs overlap")
        if any(network.overlaps(other) for other in existing):
            raise ValueError("support CIDR overlaps an authenticated retained CIDR")
    if ipaddress.ip_address(inputs.host_private_ip) not in support[0]:
        raise ValueError("fixed host address is outside the host-egress subnet")
    return True


def _tags(inputs: SupportInputs, purpose: str) -> tuple[Mapping[str, str], ...]:
    values = {
        "ActivationId": inputs.activation_id,
        "Authority": "H1g",
        "Campaign": "GLM-5.2",
        "ManagedBy": "CloudFormation",
        "Project": "KEEP",
        "Purpose": purpose,
        "RunId": RUN_ID,
    }
    return tuple({"Key": key, "Value": value} for key, value in sorted(values.items()))


def secret_stage(activation_identity: str, logical_id: str) -> str:
    """Return the sole immutable stage for one activation/secret family."""

    if (
        type(activation_identity) is not str
        or _ACTIVATION_ID.fullmatch(activation_identity) is None
        or logical_id not in SECRET_LOGICAL_IDS
    ):
        raise ValueError("secret stage coordinates are not exact")
    return f"h1g-{activation_identity}-{_SECRET_PURPOSES[logical_id]}-v1"


def _secret_stage(inputs: SupportInputs, logical_id: str) -> str:
    return secret_stage(inputs.activation_id, logical_id)


def _secret_reader_policy(
    *,
    inputs: SupportInputs,
    allowed: tuple[str, ...],
    policy_name: str,
) -> Mapping[str, object]:
    denied = tuple(item for item in SECRET_LOGICAL_IDS if item not in allowed)
    allow_statements = []
    for logical_id in allowed:
        allow_statements.extend(
            [
                {
                    "Sid": f"Read{logical_id}ExactStage",
                    "Effect": "Allow",
                    "Action": "secretsmanager:GetSecretValue",
                    "Resource": {"Ref": logical_id},
                    "Condition": {
                        "StringEquals": {
                            "secretsmanager:VersionStage": _secret_stage(
                                inputs, logical_id
                            )
                        }
                    },
                },
                {
                    "Sid": f"List{logical_id}VersionsExactSecret",
                    "Effect": "Allow",
                    "Action": "secretsmanager:ListSecretVersionIds",
                    "Resource": {"Ref": logical_id},
                },
            ]
        )
    return {
        "PolicyName": policy_name,
        "PolicyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                *allow_statements,
                {
                    "Sid": "DenyOtherSecretFamilies",
                    "Effect": "Deny",
                    "Action": "secretsmanager:*",
                    "Resource": [{"Ref": item} for item in denied],
                },
                {
                    "Sid": "DenySecretMutation",
                    "Effect": "Deny",
                    "Action": [
                        "secretsmanager:PutSecretValue",
                        "secretsmanager:UpdateSecret",
                        "secretsmanager:UpdateSecretVersionStage",
                        "secretsmanager:RotateSecret",
                    ],
                    "Resource": "*",
                },
            ],
        },
    }


def _service_role(
    *, service: str, policies: tuple[Mapping[str, object], ...]
) -> Mapping[str, object]:
    return {
        "AssumeRolePolicyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": service},
                    "Action": "sts:AssumeRole",
                }
            ],
        },
        "Policies": list(policies),
    }


def _lambda_vpc_network_policy() -> Mapping[str, object]:
    """Permit only the EC2 calls Lambda requires to manage VPC ENIs."""

    return {
        "PolicyName": "ManageLambdaVpcNetworkInterfacesOnly",
        "PolicyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "ManageLambdaVpcNetworkInterfacesInRegion",
                    "Effect": "Allow",
                    "Action": [
                        "ec2:CreateNetworkInterface",
                        "ec2:DescribeNetworkInterfaces",
                        "ec2:DeleteNetworkInterface",
                    ],
                    "Resource": "*",
                    "Condition": {"StringEquals": {"aws:RequestedRegion": REGION}},
                }
            ],
        },
    }


def _campaign_s3_decrypt_policy(
    inputs: object,
    *,
    policy_name: str,
    sid: str,
    object_arns: object,
) -> Mapping[str, object]:
    """Bind SSE-KMS decrypt to the campaign key and S3 object context."""

    return {
        "PolicyName": policy_name,
        "PolicyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": sid,
                    "Effect": "Allow",
                    "Action": "kms:Decrypt",
                    "Resource": inputs.retained_kms_key_arn,
                    "Condition": {
                        "StringEquals": {
                            "kms:ViaService": f"s3.{REGION}.amazonaws.com"
                        },
                        "StringLike": {"kms:EncryptionContext:aws:s3:arn": object_arns},
                    },
                }
            ],
        },
    }


def _task9_identity_reader_policy(
    inputs: SupportInputs,
) -> Mapping[str, object]:
    return {
        "PolicyName": "ReadExactTask9DeployedIdentity",
        "PolicyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "ReadExactVersionedTask9Identity",
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObjectVersion",
                        "s3:GetObjectVersionAttributes",
                    ],
                    "Resource": (
                        inputs.model_bucket_arn
                        + "/campaigns/"
                        + RUN_ID
                        + "/authorities/task9/"
                        + inputs.activation_id
                        + "/TASK9_DEPLOYED_IDENTITY.json"
                    ),
                }
            ],
        },
    }


def _closure_role_name(inputs: object) -> str:
    activation_id = inputs if type(inputs) is str else inputs.activation_id
    activation_sha = hashlib.sha256(activation_id.encode("ascii")).hexdigest()[:16]
    return "keep-glm52-h1g-closure-session-" + activation_sha


def _runtime_cutoff_denies(inputs: object) -> list[Mapping[str, object]]:
    cutoff = inputs.runtime_credential_cutoff_at
    return [
        {
            "Sid": "DenyPreCutoffRuntimeSessionAfterCutoff",
            "Effect": "Deny",
            "Action": "*",
            "Resource": "*",
            "Condition": {
                "DateGreaterThanEquals": {"aws:CurrentTime": cutoff},
                "DateLessThan": {"aws:TokenIssueTime": cutoff},
            },
        },
        {
            "Sid": "DenyMissingRuntimeTokenIssueTimeAfterCutoff",
            "Effect": "Deny",
            "Action": "*",
            "Resource": "*",
            "Condition": {
                "DateGreaterThanEquals": {"aws:CurrentTime": cutoff},
                "Null": {"aws:TokenIssueTime": "true"},
            },
        },
    ]


def _secret_targets(
    inputs: SupportInputs, logical_ids: tuple[str, ...]
) -> Mapping[str, object]:
    return {
        logical_id: {
            "SecretId": {"Ref": logical_id},
            "VersionStage": _secret_stage(inputs, logical_id),
        }
        for logical_id in logical_ids
    }


def _host_user_data(
    inputs: SupportInputs,
) -> Mapping[str, object]:
    first_line, separator, rest = inputs.host_user_data.partition("\n")
    if first_line != "#!/bin/sh" or not separator:
        raise ValueError("host user data must begin with the exact POSIX shell")
    binding_lines = (
        "export GLM52_BOOTSTRAP_RESOURCE_ID='${Glm52SkyBootstrapSecretId}'\n"
        "export GLM52_BOOTSTRAP_VERSION_ID='${Glm52SkyBootstrapVersionId}'\n"
        "export GLM52_BOOTSTRAP_VERSION_STAGE='${Glm52SkyBootstrapVersionStage}'\n"
        "export GLM52_HOST_TLS_RESOURCE_ID='${Glm52CombinedHostTlsSecretId}'\n"
        "export GLM52_HOST_TLS_VERSION_ID='${Glm52CombinedHostTlsVersionId}'\n"
        "export GLM52_HOST_TLS_VERSION_STAGE='${Glm52CombinedHostTlsVersionStage}'\n"
    )
    return {
        "Fn::Base64": {
            "Fn::Sub": [
                f"{first_line}\n{binding_lines}{rest}",
                {
                    "Glm52SkyBootstrapSecretId": {"Ref": "SkyBootstrapSecret"},
                    "Glm52SkyBootstrapVersionId": {
                        "Fn::GetAtt": [
                            "SkyBootstrapCustomResource",
                            "SkyBootstrapSecretVersionId",
                        ]
                    },
                    "Glm52SkyBootstrapVersionStage": _secret_stage(
                        inputs, "SkyBootstrapSecret"
                    ),
                    "Glm52CombinedHostTlsSecretId": {"Ref": "CombinedHostTlsSecret"},
                    "Glm52CombinedHostTlsVersionId": {
                        "Fn::GetAtt": [
                            "TlsBundleCustomResource",
                            "CombinedHostTlsSecretVersionId",
                        ]
                    },
                    "Glm52CombinedHostTlsVersionStage": _secret_stage(
                        inputs, "CombinedHostTlsSecret"
                    ),
                },
            ]
        }
    }


def _support_fence_manifest_coordinate(
    inputs: object,
) -> Mapping[str, object]:
    coordinate = getattr(inputs, "bootstrap_manifest_coordinate", None)
    if coordinate is None:
        coordinate = getattr(inputs, "fence_manifest_coordinate", None)
    if not isinstance(coordinate, Mapping):
        raise ValueError("support fence bootstrap manifest coordinate is absent")
    return coordinate


def _support_fence_executor_environment(
    inputs: object,
) -> dict[str, object]:
    return {
        "GLM52_ACCOUNT_ID": ACCOUNT_ID,
        "GLM52_REGION": REGION,
        "GLM52_RUN_ID": RUN_ID,
        "GLM52_ACTIVATION_ID": inputs.activation_id,
        "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
        "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_name,
        "GLM52_AUTHORITY_CLASS": "SUPPORT_RUNTIME",
        "GLM52_FENCE_EXECUTOR_ROLE_ARN": {"Fn::GetAtt": ["FenceExecutorRole", "Arn"]},
        "GLM52_BOOTSTRAP_MANIFEST_COORDINATE": canonical_json_bytes(
            _support_fence_manifest_coordinate(inputs)
        ).decode("ascii"),
        "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (inputs.lambda_code_sha256),
    }


def _build_support_template(inputs: object) -> dict[str, object]:
    validate_support_network_inputs(inputs)
    endpoint_policy_statements = []
    for prefix, (role, _client_sg, secret_ids) in VPC_SECRET_READERS.items():
        for logical_id in secret_ids:
            endpoint_policy_statements.extend(
                [
                    {
                        "Sid": f"{prefix}{logical_id}GetExactStage",
                        "Effect": "Allow",
                        "Principal": {"AWS": {"Fn::GetAtt": [role, "Arn"]}},
                        "Action": "secretsmanager:GetSecretValue",
                        "Resource": {"Ref": logical_id},
                        "Condition": {
                            "StringEquals": {
                                "secretsmanager:VersionStage": _secret_stage(
                                    inputs, logical_id
                                )
                            }
                        },
                    },
                    {
                        "Sid": f"{prefix}{logical_id}ListExactSecret",
                        "Effect": "Allow",
                        "Principal": {"AWS": {"Fn::GetAtt": [role, "Arn"]}},
                        "Action": "secretsmanager:ListSecretVersionIds",
                        "Resource": {"Ref": logical_id},
                    },
                ]
            )
    for logical_id in REHEARSAL_PROBE_SECRET_IDS:
        endpoint_policy_statements.extend(
            [
                {
                    "Sid": (f"RehearsalProbe{logical_id}GetExactStage"),
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": {
                            "Fn::GetAtt": [
                                "RehearsalProbeRole",
                                "Arn",
                            ]
                        }
                    },
                    "Action": "secretsmanager:GetSecretValue",
                    "Resource": {"Ref": logical_id},
                    "Condition": {
                        "StringEquals": {
                            "secretsmanager:VersionStage": _secret_stage(
                                inputs, logical_id
                            )
                        }
                    },
                },
                {
                    "Sid": (f"RehearsalProbe{logical_id}ListExactSecret"),
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": {
                            "Fn::GetAtt": [
                                "RehearsalProbeRole",
                                "Arn",
                            ]
                        }
                    },
                    "Action": ("secretsmanager:ListSecretVersionIds"),
                    "Resource": {"Ref": logical_id},
                },
            ]
        )
    endpoint_policy_statements.append(
        {
            "Sid": "DenyCrossRunSecretAuthority",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "secretsmanager:*",
            "NotResource": [{"Ref": logical_id} for logical_id in SECRET_LOGICAL_IDS],
        }
    )
    resources: dict[str, object] = {
        "HostEgressSubnet": {
            "Type": "AWS::EC2::Subnet",
            "Properties": {
                "AvailabilityZone": inputs.primary_az,
                "CidrBlock": SUPPORT_CIDRS[0],
                "MapPublicIpOnLaunch": False,
                "VpcId": inputs.retained_vpc_id,
            },
        },
        "PrimaryIsolatedSubnet": {
            "Type": "AWS::EC2::Subnet",
            "Properties": {
                "AvailabilityZone": inputs.primary_az,
                "CidrBlock": SUPPORT_CIDRS[1],
                "MapPublicIpOnLaunch": False,
                "VpcId": inputs.retained_vpc_id,
            },
        },
        "AlternateIsolatedSubnet": {
            "Type": "AWS::EC2::Subnet",
            "Properties": {
                "AvailabilityZone": inputs.alternate_az,
                "CidrBlock": SUPPORT_CIDRS[2],
                "MapPublicIpOnLaunch": False,
                "VpcId": inputs.retained_vpc_id,
            },
        },
        "HostRouteTable": {
            "Type": "AWS::EC2::RouteTable",
            "Properties": {"VpcId": inputs.retained_vpc_id},
        },
        "PrimaryIsolatedRouteTable": {
            "Type": "AWS::EC2::RouteTable",
            "Properties": {"VpcId": inputs.retained_vpc_id},
        },
        "AlternateIsolatedRouteTable": {
            "Type": "AWS::EC2::RouteTable",
            "Properties": {"VpcId": inputs.retained_vpc_id},
        },
        "HostRouteAssociation": {
            "Type": "AWS::EC2::SubnetRouteTableAssociation",
            "Properties": {
                "RouteTableId": {"Ref": "HostRouteTable"},
                "SubnetId": {"Ref": "HostEgressSubnet"},
            },
        },
        "PrimaryIsolatedRouteAssociation": {
            "Type": "AWS::EC2::SubnetRouteTableAssociation",
            "Properties": {
                "RouteTableId": {"Ref": "PrimaryIsolatedRouteTable"},
                "SubnetId": {"Ref": "PrimaryIsolatedSubnet"},
            },
        },
        "AlternateIsolatedRouteAssociation": {
            "Type": "AWS::EC2::SubnetRouteTableAssociation",
            "Properties": {
                "RouteTableId": {"Ref": "AlternateIsolatedRouteTable"},
                "SubnetId": {"Ref": "AlternateIsolatedSubnet"},
            },
        },
        "NatEip": {
            "Type": "AWS::EC2::EIP",
            "Properties": {"Domain": "vpc", "Tags": list(_tags(inputs, "support-nat"))},
        },
        "NatGateway": {
            "Type": "AWS::EC2::NatGateway",
            "Properties": {
                "AllocationId": {"Fn::GetAtt": ["NatEip", "AllocationId"]},
                "SubnetId": inputs.primary_public_subnet_id,
                "Tags": list(_tags(inputs, "support-nat")),
            },
        },
        "HostDefaultRoute": {
            "Type": "AWS::EC2::Route",
            "DependsOn": "NatGateway",
            "Properties": {
                "DestinationCidrBlock": "0.0.0.0/0",
                "NatGatewayId": {"Ref": "NatGateway"},
                "RouteTableId": {"Ref": "HostRouteTable"},
            },
        },
        "S3GatewayEndpoint": {
            "Type": "AWS::EC2::VPCEndpoint",
            "Properties": {
                "PolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Sid": "ExactCampaignPrefixOnly",
                            "Effect": "Allow",
                            "Principal": "*",
                            "Action": ["s3:GetObject", "s3:PutObject"],
                            "Resource": (
                                f"{inputs.model_bucket_arn}/{inputs.model_prefix}*"
                            ),
                        },
                        {
                            "Sid": "DenyCrossRun",
                            "Effect": "Deny",
                            "Principal": "*",
                            "Action": "s3:*",
                            "NotResource": [
                                inputs.model_bucket_arn,
                                (f"{inputs.model_bucket_arn}/{inputs.model_prefix}*"),
                            ],
                        },
                    ],
                },
                "RouteTableIds": [
                    {"Ref": "HostRouteTable"},
                    {"Ref": "PrimaryIsolatedRouteTable"},
                    {"Ref": "AlternateIsolatedRouteTable"},
                ],
                "ServiceName": f"com.amazonaws.{REGION}.s3",
                "VpcEndpointType": "Gateway",
                "VpcId": inputs.retained_vpc_id,
            },
        },
        "DynamoDbGatewayEndpoint": {
            "Type": "AWS::EC2::VPCEndpoint",
            "Properties": {
                "PolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Sid": "ExactRetainedLedgerOnly",
                            "Effect": "Allow",
                            "Principal": "*",
                            "Action": [
                                "dynamodb:GetItem",
                                "dynamodb:PutItem",
                                "dynamodb:UpdateItem",
                                "dynamodb:TransactGetItems",
                                "dynamodb:TransactWriteItems",
                            ],
                            "Resource": inputs.ledger_table_arn,
                        }
                    ],
                },
                "RouteTableIds": [
                    {"Ref": "HostRouteTable"},
                    {"Ref": "PrimaryIsolatedRouteTable"},
                    {"Ref": "AlternateIsolatedRouteTable"},
                ],
                "ServiceName": f"com.amazonaws.{REGION}.dynamodb",
                "VpcEndpointType": "Gateway",
                "VpcId": inputs.retained_vpc_id,
            },
        },
        "SecretsEndpointSecurityGroup": {
            "Type": "AWS::EC2::SecurityGroup",
            "Properties": {
                "GroupDescription": "H1g Secrets Manager endpoint only",
                "SecurityGroupEgress": [],
                "VpcId": inputs.retained_vpc_id,
            },
        },
        "SecretsManagerEndpoint": {
            "Type": "AWS::EC2::VPCEndpoint",
            "Properties": {
                "PolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": endpoint_policy_statements,
                },
                "PrivateDnsEnabled": True,
                "SecurityGroupIds": [{"Ref": "SecretsEndpointSecurityGroup"}],
                "ServiceName": f"com.amazonaws.{REGION}.secretsmanager",
                "SubnetIds": [
                    {"Ref": "PrimaryIsolatedSubnet"},
                    {"Ref": "AlternateIsolatedSubnet"},
                ],
                "VpcEndpointType": "Interface",
                "VpcId": inputs.retained_vpc_id,
            },
        },
        "CombinedHostSecurityGroup": {
            "Type": "AWS::EC2::SecurityGroup",
            "Properties": {
                "GroupDescription": "H1g four fixed mTLS host listeners",
                "SecurityGroupEgress": [
                    {
                        "CidrIp": "0.0.0.0/0",
                        "Description": "host-only bounded NAT egress",
                        "IpProtocol": "-1",
                    }
                ],
                "Tags": list(_tags(inputs, "combined-host-egress")),
                "VpcId": inputs.retained_vpc_id,
            },
        },
    }
    for path in CLIENT_PATHS.values():
        client_sg = path["client_security_group"]
        if type(client_sg) is not str:
            raise AssertionError("client security-group contract is malformed")
        port_fn = path["port"]
        if not callable(port_fn):
            raise AssertionError("client port contract is malformed")
        port = port_fn(inputs)
        resources[client_sg] = {
            "Type": "AWS::EC2::SecurityGroup",
            "Properties": {
                "GroupDescription": f"{client_sg} one relay only",
                "SecurityGroupEgress": [],
                "VpcId": inputs.retained_vpc_id,
            },
        }
        resources[path["egress_logical_id"]] = {
            "Type": "AWS::EC2::SecurityGroupEgress",
            "Properties": {
                "DestinationSecurityGroupId": {"Ref": "CombinedHostSecurityGroup"},
                "FromPort": port,
                "GroupId": {"Ref": client_sg},
                "IpProtocol": "tcp",
                "ToPort": port,
            },
        }
        resources[path["ingress_logical_id"]] = {
            "Type": "AWS::EC2::SecurityGroupIngress",
            "Properties": {
                "FromPort": port,
                "GroupId": {"Ref": "CombinedHostSecurityGroup"},
                "IpProtocol": "tcp",
                "SourceSecurityGroupId": {"Ref": client_sg},
                "ToPort": port,
            },
        }
    for prefix, (_role, client_sg, _secret_ids) in VPC_SECRET_READERS.items():
        resources[f"{prefix}SecretsEndpointEgress"] = {
            "Type": "AWS::EC2::SecurityGroupEgress",
            "Properties": {
                "DestinationSecurityGroupId": {"Ref": "SecretsEndpointSecurityGroup"},
                "FromPort": 443,
                "GroupId": {"Ref": client_sg},
                "IpProtocol": "tcp",
                "ToPort": 443,
            },
        }
        resources[f"{prefix}SecretsEndpointIngress"] = {
            "Type": "AWS::EC2::SecurityGroupIngress",
            "Properties": {
                "FromPort": 443,
                "GroupId": {"Ref": "SecretsEndpointSecurityGroup"},
                "IpProtocol": "tcp",
                "SourceSecurityGroupId": {"Ref": client_sg},
                "ToPort": 443,
            },
        }
    for logical_id in SECRET_LOGICAL_IDS:
        purpose = _SECRET_PURPOSES[logical_id]
        stage = _secret_stage(inputs, logical_id)
        resources[logical_id] = {
            "Type": "AWS::SecretsManager::Secret",
            "Properties": {
                "Description": f"H1g immutable {purpose}",
                "KmsKeyId": inputs.retained_kms_key_arn,
                "Name": (f"/keep/glm52/{RUN_ID}/{inputs.activation_id}/{purpose}"),
                "Tags": list(_tags(inputs, purpose))
                + [{"Key": "ImmutableVersionStage", "Value": stage}],
            },
        }
        resources[f"{logical_id}Policy"] = {
            "Type": "AWS::SecretsManager::ResourcePolicy",
            "Properties": {
                "BlockPublicPolicy": True,
                "ResourcePolicy": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Sid": "DenyUnstagedRead",
                            "Effect": "Deny",
                            "Principal": "*",
                            "Action": "secretsmanager:GetSecretValue",
                            "Resource": "*",
                            "Condition": {
                                "StringNotEquals": {
                                    "secretsmanager:VersionStage": stage
                                }
                            },
                        }
                    ],
                },
                "SecretId": {"Ref": logical_id},
            },
        }
    resources.update(
        {
            "CombinedHostRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    **_service_role(
                        service="ec2.amazonaws.com",
                        policies=(
                            _secret_reader_policy(
                                inputs=inputs,
                                allowed=(
                                    "SkyBootstrapSecret",
                                    "CombinedHostTlsSecret",
                                ),
                                policy_name="ReadBootstrapAndHostTlsOnly",
                            ),
                        ),
                    )
                },
            },
            "BudgetGateRole": {
                "Type": "AWS::IAM::Role",
                "Properties": _service_role(
                    service="lambda.amazonaws.com",
                    policies=(
                        {
                            "PolicyName": "ReadExactVersionedClosureGateOnly",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [
                                    {
                                        "Sid": "ListExactClosureGateVersions",
                                        "Effect": "Allow",
                                        "Action": "s3:ListBucketVersions",
                                        "Resource": {
                                            "Fn::GetAtt": [
                                                "RehearsalBucket",
                                                "Arn",
                                            ]
                                        },
                                        "Condition": {
                                            "StringLike": {
                                                "s3:prefix": (
                                                    "rehearsal/gates/"
                                                    + inputs.activation_id
                                                    + "/CLOSURE_BUDGET.json"
                                                )
                                            }
                                        },
                                    },
                                    {
                                        "Sid": "ReadExactClosureGateVersion",
                                        "Effect": "Allow",
                                        "Action": [
                                            "s3:GetObjectVersion",
                                            "s3:GetObjectVersionAttributes",
                                        ],
                                        "Resource": {
                                            "Fn::Sub": (
                                                "${RehearsalBucket.Arn}/"
                                                "rehearsal/gates/"
                                                + inputs.activation_id
                                                + "/CLOSURE_BUDGET.json"
                                            )
                                        },
                                    },
                                    {
                                        "Sid": "DecryptExactRehearsalObject",
                                        "Effect": "Allow",
                                        "Action": "kms:Decrypt",
                                        "Resource": (inputs.retained_kms_key_arn),
                                        "Condition": {
                                            "StringEquals": {
                                                "kms:ViaService": (
                                                    "s3.us-west-2.amazonaws.com"
                                                )
                                            },
                                            "StringLike": {
                                                "kms:EncryptionContext:aws:s3:arn": {
                                                    "Fn::Sub": (
                                                        "${RehearsalBucket.Arn}/"
                                                        "rehearsal/gates/"
                                                        + inputs.activation_id
                                                        + "/*"
                                                    )
                                                }
                                            },
                                        },
                                    },
                                ],
                            },
                        },
                    ),
                ),
            },
            "RehearsalCollectorRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    **_service_role(
                        service="lambda.amazonaws.com",
                        policies=(
                            {
                                "PolicyName": ("RehearsalCollectorCanaryOnly"),
                                "PolicyDocument": {
                                    "Version": "2012-10-17",
                                    "Statement": [
                                        {
                                            "Sid": ("WriteExactCollectorLogs"),
                                            "Effect": "Allow",
                                            "Action": [
                                                "logs:CreateLogStream",
                                                "logs:PutLogEvents",
                                            ],
                                            "Resource": (
                                                "arn:aws:logs:"
                                                + REGION
                                                + ":"
                                                + ACCOUNT_ID
                                                + ":log-group:/aws/lambda/"
                                                "keep-glm52-h1g-"
                                                "rehearsal-collector:*"
                                            ),
                                        },
                                        {
                                            "Sid": ("ListExactRehearsalVersions"),
                                            "Effect": "Allow",
                                            "Action": ("s3:ListBucketVersions"),
                                            "Resource": {
                                                "Fn::GetAtt": [
                                                    "RehearsalBucket",
                                                    "Arn",
                                                ]
                                            },
                                            "Condition": {
                                                "StringLike": {
                                                    "s3:prefix": [
                                                        (
                                                            "rehearsal/"
                                                            "canary/"
                                                            + inputs.activation_id
                                                            + "/"
                                                            + inputs.lambda_code_sha256
                                                            + "/*"
                                                        ),
                                                        (
                                                            "rehearsal/"
                                                            "measurements/"
                                                            + inputs.activation_id
                                                            + "/"
                                                            + inputs.lambda_code_sha256
                                                            + "/*"
                                                        ),
                                                        (
                                                            "rehearsal/gates/"
                                                            + inputs.activation_id
                                                            + "/*"
                                                        ),
                                                    ]
                                                }
                                            },
                                        },
                                        {
                                            "Sid": ("ReadExactRehearsalBucketPolicy"),
                                            "Effect": "Allow",
                                            "Action": "s3:GetBucketPolicy",
                                            "Resource": {
                                                "Fn::GetAtt": [
                                                    "RehearsalBucket",
                                                    "Arn",
                                                ]
                                            },
                                        },
                                        {
                                            "Sid": ("ReadWriteExactRehearsalObjects"),
                                            "Effect": "Allow",
                                            "Action": [
                                                "s3:GetObjectVersion",
                                                ("s3:GetObjectVersionAttributes"),
                                                "s3:PutObject",
                                            ],
                                            "Resource": [
                                                {
                                                    "Fn::Sub": (
                                                        "${RehearsalBucket.Arn}/"
                                                        "rehearsal/canary/"
                                                        + inputs.activation_id
                                                        + "/"
                                                        + inputs.lambda_code_sha256
                                                        + "/*"
                                                    )
                                                },
                                                {
                                                    "Fn::Sub": (
                                                        "${RehearsalBucket.Arn}/"
                                                        "rehearsal/measurements/"
                                                        + inputs.activation_id
                                                        + "/"
                                                        + inputs.lambda_code_sha256
                                                        + "/*"
                                                    )
                                                },
                                                {
                                                    "Fn::Sub": (
                                                        "${RehearsalBucket.Arn}/"
                                                        "rehearsal/gates/"
                                                        + inputs.activation_id
                                                        + "/*"
                                                    )
                                                },
                                            ],
                                        },
                                        {
                                            "Sid": (
                                                "UseExactRehearsal"
                                                "EncryptionContextsViaS3"
                                            ),
                                            "Effect": "Allow",
                                            "Action": [
                                                "kms:Decrypt",
                                                "kms:Encrypt",
                                                "kms:GenerateDataKey",
                                            ],
                                            "Resource": (inputs.retained_kms_key_arn),
                                            "Condition": {
                                                "StringEquals": {
                                                    "kms:ViaService": (
                                                        "s3.us-west-2.amazonaws.com"
                                                    )
                                                },
                                                "StringLike": {
                                                    (
                                                        "kms:"
                                                        "EncryptionContext:"
                                                        "aws:s3:arn"
                                                    ): [
                                                        {
                                                            "Fn::Sub": (
                                                                "${RehearsalBucket.Arn}/"
                                                                "rehearsal/canary/"
                                                                + inputs.activation_id
                                                                + "/"
                                                                + inputs.lambda_code_sha256
                                                                + "/*"
                                                            )
                                                        },
                                                        {
                                                            "Fn::Sub": (
                                                                "${RehearsalBucket.Arn}/"
                                                                "rehearsal/measurements/"
                                                                + inputs.activation_id
                                                                + "/"
                                                                + inputs.lambda_code_sha256
                                                                + "/*"
                                                            )
                                                        },
                                                        {
                                                            "Fn::Sub": (
                                                                "${RehearsalBucket.Arn}/"
                                                                "rehearsal/gates/"
                                                                + inputs.activation_id
                                                                + "/*"
                                                            )
                                                        },
                                                    ]
                                                },
                                            },
                                        },
                                        {
                                            "Sid": ("InvokeExactReadOnlyProbeVersion"),
                                            "Effect": "Allow",
                                            "Action": ("lambda:InvokeFunction"),
                                            "Resource": {
                                                "Ref": ("RehearsalProbeVersion")
                                            },
                                        },
                                        {
                                            "Sid": ("DenyNonProbeLambdaInvocation"),
                                            "Effect": "Deny",
                                            "Action": ("lambda:InvokeFunction"),
                                            "NotResource": {
                                                "Ref": ("RehearsalProbeVersion")
                                            },
                                        },
                                        {
                                            "Sid": (
                                                "DenyCampaignAndModelBucketMutation"
                                            ),
                                            "Effect": "Deny",
                                            "Action": [
                                                "s3:AbortMultipartUpload",
                                                "s3:DeleteObject*",
                                                "s3:PutObject*",
                                            ],
                                            "Resource": [
                                                inputs.model_bucket_arn,
                                                (inputs.model_bucket_arn + "/*"),
                                            ],
                                        },
                                        {
                                            "Sid": ("DenyProductionMutations"),
                                            "Effect": "Deny",
                                            "Action": [
                                                "cloudformation:Create*",
                                                "cloudformation:Delete*",
                                                "cloudformation:Execute*",
                                                "cloudformation:Update*",
                                                "dynamodb:BatchWriteItem",
                                                "dynamodb:DeleteItem",
                                                "dynamodb:PutItem",
                                                "dynamodb:TransactWriteItems",
                                                "dynamodb:UpdateItem",
                                                "ec2:AllocateAddress",
                                                "ec2:Associate*",
                                                "ec2:Attach*",
                                                "ec2:Authorize*",
                                                "ec2:Create*",
                                                "ec2:Delete*",
                                                "ec2:Detach*",
                                                "ec2:Disassociate*",
                                                "ec2:Modify*",
                                                "ec2:ReleaseAddress",
                                                "ec2:Replace*",
                                                "ec2:Reset*",
                                                "ec2:Revoke*",
                                                "ec2:Run*",
                                                "ec2:Start*",
                                                "ec2:Stop*",
                                                "ec2:Terminate*",
                                                "events:PutEvents",
                                                "execute-api:Invoke",
                                                "iam:Add*",
                                                "iam:Attach*",
                                                "iam:Create*",
                                                "iam:Delete*",
                                                "iam:Detach*",
                                                "iam:PassRole",
                                                "iam:Put*",
                                                "iam:Remove*",
                                                "iam:Set*",
                                                "iam:Tag*",
                                                "iam:Untag*",
                                                "iam:Update*",
                                                "secretsmanager:*",
                                                "sns:Publish",
                                                "sts:AssumeRole",
                                                "sts:TagSession",
                                            ],
                                            "Resource": "*",
                                        },
                                    ],
                                },
                            },
                        ),
                    ),
                    "RoleName": ("keep-glm52-h1g-rehearsal-collector"),
                },
            },
            "RehearsalProbeRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    **_service_role(
                        service="lambda.amazonaws.com",
                        policies=(
                            _secret_reader_policy(
                                inputs=inputs,
                                allowed=(
                                    "RawSkyTokenSecret",
                                    "AttestationClientTlsSecret",
                                ),
                                policy_name=("ReadRehearsalProbeSecretsOnly"),
                            ),
                            {
                                "PolicyName": ("ExecuteReadOnlyRehearsalProbe"),
                                "PolicyDocument": {
                                    "Version": "2012-10-17",
                                    "Statement": [
                                        {
                                            "Sid": "WriteExactProbeLogs",
                                            "Effect": "Allow",
                                            "Action": [
                                                "logs:CreateLogStream",
                                                "logs:PutLogEvents",
                                            ],
                                            "Resource": (
                                                "arn:aws:logs:"
                                                + REGION
                                                + ":"
                                                + ACCOUNT_ID
                                                + ":log-group:/aws/lambda/"
                                                "keep-glm52-h1g-"
                                                "rehearsal-probe:*"
                                            ),
                                        },
                                        {
                                            "Sid": ("DenyProbeProductionAuthority"),
                                            "Effect": "Deny",
                                            "Action": [
                                                "cloudformation:*",
                                                "dynamodb:*",
                                                "ec2:Run*",
                                                "ec2:Start*",
                                                "ec2:Stop*",
                                                "ec2:Terminate*",
                                                "events:PutEvents",
                                                "execute-api:Invoke",
                                                "iam:*",
                                                "lambda:InvokeFunction",
                                                "s3:*",
                                                "sns:Publish",
                                                "sts:*",
                                            ],
                                            "Resource": "*",
                                        },
                                    ],
                                },
                            },
                            _lambda_vpc_network_policy(),
                        ),
                    ),
                    "RoleName": "keep-glm52-h1g-rehearsal-probe",
                },
            },
            "RehearsalExecutorRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "RoleName": "keep-glm52-h1g-rehearsal-executor",
                    "MaxSessionDuration": 900,
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "AWS": {
                                        "Fn::GetAtt": [
                                            "FenceExecutorRole",
                                            "Arn",
                                        ]
                                    }
                                },
                                "Action": "sts:AssumeRole",
                                "Condition": {
                                    "ArnEquals": {
                                        "aws:PrincipalArn": {
                                            "Fn::GetAtt": [
                                                "FenceExecutorRole",
                                                "Arn",
                                            ]
                                        }
                                    },
                                    "StringEquals": {
                                        "sts:RoleSessionName": (
                                            _rehearsal_session_name(
                                                inputs.activation_id
                                            )
                                        )
                                    },
                                },
                            }
                        ],
                    },
                    "Policies": [
                        {
                            "PolicyName": ("InvokeExactRehearsalCollectorVersionOnly"),
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [
                                    {
                                        "Effect": "Allow",
                                        "Action": "lambda:InvokeFunction",
                                        "Resource": {
                                            "Ref": ("RehearsalCollectorVersion")
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            },
            "DecisionRole": {
                "Type": "AWS::IAM::Role",
                "Properties": _service_role(
                    service="lambda.amazonaws.com",
                    policies=(
                        {
                            "PolicyName": "AssumeExactClosureSessionOnly",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [
                                    {
                                        "Sid": "WriteExactDecisionLogs",
                                        "Effect": "Allow",
                                        "Action": [
                                            "logs:CreateLogStream",
                                            "logs:PutLogEvents",
                                        ],
                                        "Resource": (
                                            "arn:aws:logs:"
                                            + REGION
                                            + ":"
                                            + ACCOUNT_ID
                                            + ":log-group:/aws/lambda/"
                                            "keep-glm52-h1g-decision:*"
                                        ),
                                    },
                                    {
                                        "Sid": "AssumeExactClosureSession",
                                        "Effect": "Allow",
                                        "Action": "sts:AssumeRole",
                                        "Resource": {
                                            "Fn::GetAtt": [
                                                "ClosureSessionRole",
                                                "Arn",
                                            ]
                                        },
                                    },
                                    *_runtime_cutoff_denies(inputs),
                                ],
                            },
                        },
                    ),
                ),
            },
            "ClosureSessionRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "RoleName": _closure_role_name(inputs),
                    "MaxSessionDuration": 3600,
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "AWS": {
                                        "Fn::GetAtt": [
                                            "DecisionRole",
                                            "Arn",
                                        ]
                                    }
                                },
                                "Action": "sts:AssumeRole",
                                "Condition": {
                                    "ArnEquals": {
                                        "aws:PrincipalArn": {
                                            "Fn::GetAtt": [
                                                "DecisionRole",
                                                "Arn",
                                            ]
                                        }
                                    },
                                    "StringEquals": {
                                        "sts:ExternalId": (inputs.lambda_code_sha256)
                                    },
                                    "NumericEquals": {"sts:DurationSeconds": 900},
                                    "StringLike": {
                                        "sts:RoleSessionName": (
                                            "h1g-decision-"
                                            + hashlib.sha256(
                                                inputs.activation_id.encode("ascii")
                                            ).hexdigest()[:16]
                                            + "-*"
                                        )
                                    },
                                },
                            }
                        ],
                    },
                    "Policies": [
                        {
                            "PolicyName": "ExecuteExactPrivateClosureOnly",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [
                                    {
                                        "Sid": "ReadWriteExactLedger",
                                        "Effect": "Allow",
                                        "Action": [
                                            "dynamodb:GetItem",
                                            "dynamodb:Query",
                                            "dynamodb:TransactGetItems",
                                            "dynamodb:TransactWriteItems",
                                        ],
                                        "Resource": inputs.ledger_table_arn,
                                    },
                                    {
                                        "Sid": "WalkExactCampaignBucket",
                                        "Effect": "Allow",
                                        "Action": "s3:ListBucketVersions",
                                        "Resource": inputs.model_bucket_arn,
                                        "Condition": {
                                            "StringLike": {
                                                "s3:prefix": (
                                                    "campaigns/" + RUN_ID + "/*"
                                                )
                                            }
                                        },
                                    },
                                    {
                                        "Sid": "ReadWriteExactCampaignObjects",
                                        "Effect": "Allow",
                                        "Action": [
                                            "s3:GetObjectVersion",
                                            "s3:GetObjectVersionAttributes",
                                        ],
                                        "Resource": (
                                            inputs.model_bucket_arn
                                            + "/campaigns/"
                                            + RUN_ID
                                            + "/*"
                                        ),
                                    },
                                    {
                                        "Sid": "DecryptExactCampaignObjectsViaS3Only",
                                        "Effect": "Allow",
                                        "Action": "kms:Decrypt",
                                        "Resource": inputs.retained_kms_key_arn,
                                        "Condition": {
                                            "StringEquals": {
                                                "kms:ViaService": (
                                                    "s3.us-west-2.amazonaws.com"
                                                )
                                            },
                                            "StringLike": {
                                                "kms:EncryptionContext:aws:s3:arn": (
                                                    inputs.model_bucket_arn
                                                    + "/campaigns/"
                                                    + RUN_ID
                                                    + "/*"
                                                )
                                            },
                                        },
                                    },
                                    {
                                        "Sid": "ReadExactCampaignBucketState",
                                        "Effect": "Allow",
                                        "Action": [
                                            "s3:GetBucketVersioning",
                                            "s3:GetBucketPolicy",
                                            "s3:GetLifecycleConfiguration",
                                            "s3:GetReplicationConfiguration",
                                        ],
                                        "Resource": inputs.model_bucket_arn,
                                    },
                                    {
                                        "Sid": "ReadExactFenceStack",
                                        "Effect": "Allow",
                                        "Action": [
                                            "cloudformation:DescribeChangeSet",
                                            "cloudformation:DescribeStacks",
                                            "cloudformation:GetTemplate",
                                            "cloudformation:ListChangeSets",
                                        ],
                                        "Resource": inputs.fence_stack_id,
                                    },
                                    {
                                        "Sid": "InvokeExactSupportCallees",
                                        "Effect": "Allow",
                                        "Action": "lambda:InvokeFunction",
                                        "Resource": [
                                            {"Ref": "SourceGpuSpendVersion"},
                                            {"Ref": ("SourceSubmissionIntentVersion")},
                                            {
                                                "Ref": (
                                                    "SourceControllerBaselineVersion"
                                                )
                                            },
                                            {
                                                "Ref": (
                                                    "SourceControlPlaneReadinessVersion"
                                                )
                                            },
                                            {
                                                "Ref": (
                                                    "SourceSubmissionAcquisitionVersion"
                                                )
                                            },
                                            {"Ref": "FenceExecutorVersion"},
                                            {"Ref": "FenceSuccessorVersion"},
                                            {"Ref": "ClaimWriterVersion"},
                                            {"Ref": "DecisionWriterVersion"},
                                            {"Ref": "TerminalV1WriterVersion"},
                                            {"Ref": "ClosureHandoffVersion"},
                                            {"Ref": "AttestationVersion"},
                                            {"Ref": "LaunchAdmissionVersion"},
                                            {"Ref": "NumericBindingVersion"},
                                        ],
                                    },
                                    {
                                        "Sid": "AuthenticateClosurePrincipal",
                                        "Effect": "Allow",
                                        "Action": "sts:GetCallerIdentity",
                                        "Resource": "*",
                                    },
                                    {
                                        "Sid": "ReadExactH1dStacks",
                                        "Effect": "Allow",
                                        "Action": [
                                            "cloudformation:DescribeStacks",
                                            "cloudformation:ListStackResources",
                                            "cloudformation:GetTemplate",
                                        ],
                                        "Resource": [
                                            inputs.retained_stack_id,
                                            inputs.fence_stack_id,
                                            {"Ref": "AWS::StackId"},
                                        ],
                                    },
                                    {
                                        "Sid": "ReadExactH1dFunctions",
                                        "Effect": "Allow",
                                        "Action": [
                                            "lambda:GetFunction",
                                            "lambda:GetPolicy",
                                        ],
                                        "Resource": [
                                            {
                                                "Fn::GetAtt": [
                                                    name + "Function",
                                                    "Arn",
                                                ]
                                            }
                                            for name in RUNTIME_FUNCTIONS
                                        ]
                                        + [
                                            {"Ref": name + "Version"}
                                            for name in RUNTIME_FUNCTIONS
                                        ]
                                        + [
                                            {
                                                "Fn::GetAtt": [
                                                    "TlsHandlerFunction",
                                                    "Arn",
                                                ]
                                            }
                                        ],
                                    },
                                    {
                                        "Sid": "ReadExactH1dRoles",
                                        "Effect": "Allow",
                                        "Action": [
                                            "iam:GetRole",
                                            "iam:GetRolePolicy",
                                            "iam:ListAttachedRolePolicies",
                                            "iam:ListInstanceProfilesForRole",
                                            "iam:ListRolePolicies",
                                        ],
                                        "Resource": [
                                            {
                                                "Fn::GetAtt": [
                                                    role_id,
                                                    "Arn",
                                                ]
                                            }
                                            for role_id in (
                                                "BudgetGateRole",
                                                "RehearsalCollectorRole",
                                                "RehearsalProbeRole",
                                                "RehearsalExecutorRole",
                                                "DecisionRole",
                                                "ClosureSessionRole",
                                                "CombinedHostRole",
                                                "AttestationRole",
                                                "LaunchAdmissionRole",
                                                "NumericBindingRole",
                                                "RetainedCancellationRole",
                                                *(
                                                    name + "Role"
                                                    for name in (
                                                        *_TASK11_WRITER_IDS,
                                                        "FenceExecutor",
                                                    )
                                                ),
                                                "SupportDeadlineRole",
                                                "SupportScheduleInvokeRole",
                                                "SupportWorkflowRole",
                                                "TlsHandlerRole",
                                            )
                                        ],
                                    },
                                    {
                                        "Sid": "ReadExactH1dInstanceProfile",
                                        "Effect": "Allow",
                                        "Action": "iam:GetInstanceProfile",
                                        "Resource": {
                                            "Fn::GetAtt": [
                                                "CombinedHostProfile",
                                                "Arn",
                                            ]
                                        },
                                    },
                                    {
                                        "Sid": "ReadExactH1dStateMachines",
                                        "Effect": "Allow",
                                        "Action": [
                                            "states:DescribeStateMachine",
                                            "states:ListStateMachineVersions",
                                        ],
                                        "Resource": [
                                            {"Ref": "SupportStateMachine"},
                                            {
                                                "Fn::Sub": (
                                                    "arn:${AWS::Partition}:states:"
                                                    "${AWS::Region}:${AWS::AccountId}:"
                                                    "stateMachine:"
                                                    "keep-glm52-h1g-retained-lifecycle"
                                                )
                                            },
                                        ],
                                    },
                                    {
                                        "Sid": "ReadExactH1dEventRule",
                                        "Effect": "Allow",
                                        "Action": [
                                            "events:ListTargetsByRule",
                                            "events:DescribeRule",
                                        ],
                                        "Resource": {
                                            "Fn::GetAtt": [
                                                "SupportEgressAlarmRule",
                                                "Arn",
                                            ]
                                        },
                                    },
                                    {
                                        "Sid": "ReadExactH1dSchedules",
                                        "Effect": "Allow",
                                        "Action": "scheduler:GetSchedule",
                                        "Resource": [
                                            {
                                                "Fn::GetAtt": [
                                                    logical_id,
                                                    "Arn",
                                                ]
                                            }
                                            for logical_id in (
                                                "WorkStopSchedule",
                                                "DeleteRequestSchedule",
                                                "AbsenceDeadlineSchedule",
                                            )
                                        ],
                                    },
                                    {
                                        "Sid": "ReadExactH1dQueue",
                                        "Effect": "Allow",
                                        "Action": "sqs:GetQueueAttributes",
                                        "Resource": {
                                            "Fn::GetAtt": [
                                                "SupportAsyncDlq",
                                                "Arn",
                                            ]
                                        },
                                    },
                                    {
                                        "Sid": "ReadExactH1dOperatorTopic",
                                        "Effect": "Allow",
                                        "Action": ("sns:ListSubscriptionsByTopic"),
                                        "Resource": (SUPPORT_OPERATOR_TOPIC_ARN),
                                    },
                                    {
                                        "Sid": "ReadOnlyUnscopableH1dIndexes",
                                        "Effect": "Allow",
                                        "Action": [
                                            "lambda:ListEventSourceMappings",
                                            "lambda:ListFunctions",
                                            "events:ListRules",
                                            "scheduler:ListSchedules",
                                            "sqs:ListQueues",
                                            "iam:ListInstanceProfiles",
                                            "iam:ListRoles",
                                            "states:ListStateMachines",
                                            "ec2:DescribeIamInstanceProfileAssociations",
                                            "ec2:DescribeInstances",
                                            "ssm:DescribeInstanceInformation",
                                            "cloudwatch:DescribeAlarms",
                                            "cloudwatch:GetMetricData",
                                            "logs:DescribeLogGroups",
                                        ],
                                        "Resource": "*",
                                    },
                                    *_runtime_cutoff_denies(inputs),
                                ],
                            },
                        },
                    ],
                },
            },
            "AttestationRole": {
                "Type": "AWS::IAM::Role",
                "Properties": _service_role(
                    service="lambda.amazonaws.com",
                    policies=(
                        _secret_reader_policy(
                            inputs=inputs,
                            allowed=(
                                "RawSkyTokenSecret",
                                "AttestationClientTlsSecret",
                            ),
                            policy_name="ReadAttestationSecretsOnly",
                        ),
                        _task9_identity_reader_policy(inputs),
                        {
                            "PolicyName": "ConsumeExactAttestationNonceOnly",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [
                                    {
                                        "Sid": "PutExactAttestationNonce",
                                        "Effect": "Allow",
                                        "Action": "dynamodb:PutItem",
                                        "Resource": inputs.ledger_table_arn,
                                    }
                                ],
                            },
                        },
                        _lambda_vpc_network_policy(),
                    ),
                ),
            },
            "LaunchAdmissionRole": {
                "Type": "AWS::IAM::Role",
                "Properties": _service_role(
                    service="lambda.amazonaws.com",
                    policies=(
                        _secret_reader_policy(
                            inputs=inputs,
                            allowed=(
                                "RawSkyTokenSecret",
                                "LaunchAdmissionClientTlsSecret",
                            ),
                            policy_name="ReadLaunchAdmissionSecretsOnly",
                        ),
                        {
                            "PolicyName": "OwnAndAuditExactLaunchAdmission",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [
                                    {
                                        "Sid": "ReadWriteExactAdmissionLedger",
                                        "Effect": "Allow",
                                        "Action": [
                                            "dynamodb:GetItem",
                                            "dynamodb:Query",
                                            "dynamodb:TransactGetItems",
                                            "dynamodb:TransactWriteItems",
                                        ],
                                        "Resource": inputs.ledger_table_arn,
                                    },
                                    {
                                        "Sid": "WalkExactAdmissionAuthority",
                                        "Effect": "Allow",
                                        "Action": "s3:ListBucketVersions",
                                        "Resource": inputs.model_bucket_arn,
                                        "Condition": {
                                            "StringLike": {
                                                "s3:prefix": (
                                                    "campaigns/" + RUN_ID + "/*"
                                                )
                                            }
                                        },
                                    },
                                    {
                                        "Sid": "ReadExactAdmissionAuthority",
                                        "Effect": "Allow",
                                        "Action": [
                                            "s3:GetObjectVersion",
                                            "s3:GetObjectVersionAttributes",
                                        ],
                                        "Resource": (
                                            inputs.model_bucket_arn
                                            + "/campaigns/"
                                            + RUN_ID
                                            + "/*"
                                        ),
                                    },
                                ],
                            },
                        },
                        _campaign_s3_decrypt_policy(
                            inputs,
                            policy_name=("DecryptExactLaunchAdmissionObjectsOnly"),
                            sid=("DecryptExactLaunchAdmissionObjectsViaS3Only"),
                            object_arns=(
                                inputs.model_bucket_arn + "/campaigns/" + RUN_ID + "/*"
                            ),
                        ),
                        _lambda_vpc_network_policy(),
                    ),
                ),
            },
            "NumericBindingRole": {
                "Type": "AWS::IAM::Role",
                "Properties": _service_role(
                    service="lambda.amazonaws.com",
                    policies=(
                        _secret_reader_policy(
                            inputs=inputs,
                            allowed=(
                                "RawSkyTokenSecret",
                                "NumericBindingClientTlsSecret",
                            ),
                            policy_name="ReadNumericBindingSecretsOnly",
                        ),
                        _task9_identity_reader_policy(inputs),
                        _lambda_vpc_network_policy(),
                    ),
                ),
            },
            "RetainedCancellationRole": {
                "Type": "AWS::IAM::Role",
                "Properties": _service_role(
                    service="lambda.amazonaws.com",
                    policies=(
                        _secret_reader_policy(
                            inputs=inputs,
                            allowed=(
                                "RawSkyTokenSecret",
                                "RetainedCancellationClientTlsSecret",
                            ),
                            policy_name="ReadCancellationSecretsOnly",
                        ),
                        _task9_identity_reader_policy(inputs),
                        _lambda_vpc_network_policy(),
                    ),
                ),
            },
            "CombinedHostProfile": {
                "Type": "AWS::IAM::InstanceProfile",
                "Properties": {"Roles": [{"Ref": "CombinedHostRole"}]},
            },
            "CombinedHost": {
                "Type": "AWS::EC2::Instance",
                "DependsOn": [
                    "SkyBootstrapCustomResource",
                    "TlsBundleCustomResource",
                ],
                "Properties": {
                    "BlockDeviceMappings": [
                        {
                            "DeviceName": "/dev/sda1",
                            "Ebs": {
                                "DeleteOnTermination": True,
                                "Encrypted": True,
                                "Iops": inputs.root_volume_iops,
                                "KmsKeyId": inputs.retained_kms_key_arn,
                                "Throughput": inputs.root_volume_throughput_mibps,
                                "VolumeSize": inputs.root_volume_gib,
                                "VolumeType": inputs.root_volume_type,
                            },
                        }
                    ],
                    "IamInstanceProfile": {"Ref": "CombinedHostProfile"},
                    "ImageId": inputs.host_ami_id,
                    "InstanceType": "c6a.xlarge",
                    "MetadataOptions": {
                        "HttpEndpoint": "enabled",
                        "HttpTokens": "required",
                    },
                    "NetworkInterfaces": [
                        {
                            "AssociatePublicIpAddress": False,
                            "DeviceIndex": "0",
                            "GroupSet": [{"Ref": "CombinedHostSecurityGroup"}],
                            "PrivateIpAddress": inputs.host_private_ip,
                            "SubnetId": {"Ref": "HostEgressSubnet"},
                        }
                    ],
                    "Tags": list(_tags(inputs, "combined-host")),
                    "UserData": _host_user_data(inputs),
                },
            },
            "CombinedHostDataVolume": {
                "Type": "AWS::EC2::Volume",
                "DeletionPolicy": "Snapshot",
                "UpdateReplacePolicy": "Snapshot",
                "Properties": {
                    "AvailabilityZone": inputs.primary_az,
                    "Encrypted": True,
                    "Iops": 3000,
                    "KmsKeyId": inputs.retained_kms_key_arn,
                    "Size": 50,
                    "Throughput": 125,
                    "VolumeType": "gp3",
                    "Tags": list(_tags(inputs, "forensic-data-volume")),
                },
            },
            "CombinedHostDataVolumeAttachment": {
                "Type": "AWS::EC2::VolumeAttachment",
                "Properties": {
                    "Device": "/dev/sdf",
                    "InstanceId": {"Ref": "CombinedHost"},
                    "VolumeId": {"Ref": "CombinedHostDataVolume"},
                },
            },
            "TlsHandlerRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    **_service_role(
                        service="lambda.amazonaws.com",
                        policies=(
                            {
                                "PolicyName": "WriteTlsAndCaMaterialOnly",
                                "PolicyDocument": {
                                    "Version": "2012-10-17",
                                    "Statement": [
                                        *[
                                            {
                                                "Sid": (f"Write{logical_id}ExactStage"),
                                                "Effect": "Allow",
                                                "Action": (
                                                    "secretsmanager:PutSecretValue"
                                                ),
                                                "Resource": {"Ref": logical_id},
                                                "Condition": {
                                                    "StringEquals": {
                                                        (
                                                            "secretsmanager:"
                                                            "VersionStage"
                                                        ): _secret_stage(
                                                            inputs,
                                                            logical_id,
                                                        )
                                                    }
                                                },
                                            }
                                            for logical_id in SECRET_LOGICAL_IDS
                                        ],
                                        *[
                                            {
                                                "Sid": (f"List{logical_id}Versions"),
                                                "Effect": "Allow",
                                                "Action": (
                                                    "secretsmanager:"
                                                    "ListSecretVersionIds"
                                                ),
                                                "Resource": {"Ref": logical_id},
                                            }
                                            for logical_id in SECRET_LOGICAL_IDS
                                        ],
                                        {
                                            "Sid": "ReadCaIssuanceExactStage",
                                            "Effect": "Allow",
                                            "Action": ("secretsmanager:GetSecretValue"),
                                            "Resource": {"Ref": "CaIssuanceSecret"},
                                            "Condition": {
                                                "StringEquals": {
                                                    (
                                                        "secretsmanager:VersionStage"
                                                    ): _secret_stage(
                                                        inputs,
                                                        "CaIssuanceSecret",
                                                    )
                                                }
                                            },
                                        },
                                        {
                                            "Sid": "UseRetainedCampaignKey",
                                            "Effect": "Allow",
                                            "Action": [
                                                "kms:Decrypt",
                                                "kms:Encrypt",
                                                "kms:GenerateDataKey",
                                            ],
                                            "Resource": inputs.retained_kms_key_arn,
                                            "Condition": {
                                                "StringLike": {
                                                    "kms:EncryptionContext:SecretARN": (
                                                        "arn:aws:secretsmanager:"
                                                        f"{REGION}:{ACCOUNT_ID}:"
                                                        "secret:/keep/glm52/*"
                                                    )
                                                }
                                            },
                                        },
                                        {
                                            "Sid": "CreateExactTlsHandlerGrant",
                                            "Effect": "Allow",
                                            "Action": "kms:CreateGrant",
                                            "Resource": inputs.retained_kms_key_arn,
                                            "Condition": {
                                                "StringEquals": {
                                                    "kms:GranteePrincipal": (
                                                        TLS_HANDLER_ROLE_ARN
                                                    ),
                                                    "kms:RetiringPrincipal": (
                                                        SUPPORT_DELETION_ROLE_ARN
                                                    ),
                                                    (
                                                        "kms:EncryptionContext:RunId"
                                                    ): RUN_ID,
                                                    (
                                                        "kms:"
                                                        "EncryptionContext:"
                                                        "ActivationId"
                                                    ): inputs.activation_id,
                                                },
                                                "ForAllValues:StringEquals": {
                                                    "kms:GrantOperations": [
                                                        "Decrypt",
                                                        "Encrypt",
                                                        "GenerateDataKey",
                                                    ]
                                                },
                                            },
                                        },
                                        {
                                            "Sid": ("PersistExactDirectGrantEvidence"),
                                            "Effect": "Allow",
                                            "Action": [
                                                "dynamodb:GetItem",
                                                "dynamodb:PutItem",
                                            ],
                                            "Resource": inputs.ledger_table_arn,
                                        },
                                        {
                                            "Sid": "ReconcileExactTlsHandlerGrant",
                                            "Effect": "Allow",
                                            "Action": [
                                                "kms:ListGrants",
                                                "kms:RetireGrant",
                                                "kms:RevokeGrant",
                                            ],
                                            "Resource": inputs.retained_kms_key_arn,
                                        },
                                    ],
                                },
                            },
                        ),
                    ),
                    "RoleName": "keep-glm52-h1g-support-tls-handler",
                },
            },
            "TlsHandlerFunction": {
                "Type": "AWS::Lambda::Function",
                "Properties": {
                    "Code": {
                        "S3Bucket": inputs.lambda_code_bucket,
                        "S3Key": inputs.lambda_code_key,
                        "S3ObjectVersion": inputs.lambda_code_version_id,
                    },
                    "Environment": {
                        "Variables": {
                            "GLM52_ACTIVATION_ID": inputs.activation_id,
                            "GLM52_COMBINED_HOST_PRIVATE_IP": (inputs.host_private_ip),
                            "GLM52_RETAINED_KMS_KEY_ARN": (inputs.retained_kms_key_arn),
                            "GLM52_LEDGER_TABLE_NAME": (inputs.ledger_table_name),
                            **{
                                f"GLM52_{logical_id.upper()}_ID": {"Ref": logical_id}
                                for logical_id in SECRET_LOGICAL_IDS
                            },
                            **{
                                f"GLM52_{logical_id.upper()}_STAGE": (
                                    _secret_stage(inputs, logical_id)
                                )
                                for logical_id in SECRET_LOGICAL_IDS
                            },
                        }
                    },
                    "Handler": "support_custom_resource_handler.main",
                    "Layers": [inputs.cryptography_layer_arn],
                    "MemorySize": 256,
                    "ReservedConcurrentExecutions": 1,
                    "Role": {"Fn::GetAtt": ["TlsHandlerRole", "Arn"]},
                    "Runtime": "python3.12",
                    "Timeout": 840,
                },
            },
            "TlsHandlerVersion": {
                "Type": "AWS::Lambda::Version",
                "Properties": {
                    "Description": (
                        f"code={inputs.lambda_code_sha256};"
                        f"cryptography={inputs.cryptography_layer_sha256}"
                    ),
                    "FunctionName": {"Ref": "TlsHandlerFunction"},
                },
            },
            "TlsHandlerPermission": {
                "Type": "AWS::Lambda::Permission",
                "Properties": {
                    "Action": "lambda:InvokeFunction",
                    "FunctionName": {"Ref": "TlsHandlerVersion"},
                    "Principal": "cloudformation.amazonaws.com",
                    "SourceAccount": ACCOUNT_ID,
                    "SourceArn": {"Ref": "AWS::StackId"},
                },
            },
            "SkyBootstrapCustomResource": {
                "Type": "Custom::H1gSkyBootstrap",
                "DependsOn": [
                    "TlsHandlerRole",
                    "TlsHandlerVersion",
                    "TlsHandlerPermission",
                    "RawSkyTokenSecret",
                    "SkyBootstrapSecret",
                ],
                "Properties": {
                    "ServiceToken": {"Ref": "TlsHandlerVersion"},
                    "ActivationIdentity": inputs.activation_id,
                    "ResourceKind": "SKY_BOOTSTRAP",
                    "MaterialIdentitySha256": inputs.host_boot_identity_sha256,
                    "SecretTargets": _secret_targets(inputs, SECRET_LOGICAL_IDS[:2]),
                },
            },
            "TlsBundleCustomResource": {
                "Type": "Custom::H1gTlsBundle",
                "DependsOn": [
                    "TlsHandlerRole",
                    "TlsHandlerVersion",
                    "TlsHandlerPermission",
                    *SECRET_LOGICAL_IDS[2:],
                ],
                "Properties": {
                    "ServiceToken": {"Ref": "TlsHandlerVersion"},
                    "ActivationIdentity": inputs.activation_id,
                    "ResourceKind": "TLS_BUNDLE",
                    "MaterialIdentitySha256": inputs.cryptography_layer_sha256,
                    "SecretTargets": _secret_targets(inputs, SECRET_LOGICAL_IDS[2:]),
                },
            },
            "SupportWorkflowRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"Service": "states.amazonaws.com"},
                                "Action": "sts:AssumeRole",
                            }
                        ],
                    },
                    "Policies": [],
                },
            },
            "SupportStateMachine": {
                "Type": "AWS::StepFunctions::StateMachine",
                "Properties": {
                    "Definition": build_task11_workflow_definition(),
                    "RoleArn": {"Fn::GetAtt": ["SupportWorkflowRole", "Arn"]},
                    "StateMachineType": "STANDARD",
                },
            },
            "SupportStateMachineVersion": {
                "Type": "AWS::StepFunctions::StateMachineVersion",
                "Properties": {
                    "Description": ("Exact Task11 private decision-to-POST version"),
                    "StateMachineArn": {"Ref": "SupportStateMachine"},
                },
            },
            "RehearsalBucket": {
                "Type": "AWS::S3::Bucket",
                "Properties": {
                    "BucketEncryption": {
                        "ServerSideEncryptionConfiguration": [
                            {
                                "BucketKeyEnabled": True,
                                "ServerSideEncryptionByDefault": {
                                    "KMSMasterKeyID": inputs.retained_kms_key_arn,
                                    "SSEAlgorithm": "aws:kms",
                                },
                            }
                        ]
                    },
                    "BucketName": (f"keep-glm52-h1g-rehearsal-{ACCOUNT_ID}-{REGION}"),
                    "PublicAccessBlockConfiguration": {
                        "BlockPublicAcls": True,
                        "BlockPublicPolicy": True,
                        "IgnorePublicAcls": True,
                        "RestrictPublicBuckets": True,
                    },
                    "VersioningConfiguration": {"Status": "Enabled"},
                },
            },
            "RehearsalBucketPolicy": {
                "Type": "AWS::S3::BucketPolicy",
                "Properties": {
                    "Bucket": {"Ref": "RehearsalBucket"},
                    "PolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Sid": ("DenyNamedProductionPrincipalRehearsalRead"),
                                "Effect": "Deny",
                                "Principal": "*",
                                "Action": [
                                    "s3:GetObject",
                                    "s3:GetObjectVersion",
                                    "s3:GetObjectVersionAttributes",
                                    "s3:ListBucket",
                                    "s3:ListBucketVersions",
                                ],
                                "Resource": [
                                    {
                                        "Fn::GetAtt": [
                                            "RehearsalBucket",
                                            "Arn",
                                        ]
                                    },
                                    {"Fn::Sub": ("${RehearsalBucket.Arn}/rehearsal/*")},
                                ],
                                "Condition": {
                                    "ArnEquals": {
                                        "aws:PrincipalArn": [
                                            {
                                                "Fn::GetAtt": [
                                                    "ClosureSessionRole",
                                                    "Arn",
                                                ]
                                            },
                                            {
                                                "Fn::GetAtt": [
                                                    "DecisionRole",
                                                    "Arn",
                                                ]
                                            },
                                            {
                                                "Fn::GetAtt": [
                                                    "LaunchAdmissionRole",
                                                    "Arn",
                                                ]
                                            },
                                            {
                                                "Fn::GetAtt": [
                                                    "CombinedHostRole",
                                                    "Arn",
                                                ]
                                            },
                                            (
                                                "arn:aws:iam::" + ACCOUNT_ID + ":role/"
                                                "keep-glm52-gpu-worker"
                                            ),
                                            (
                                                "arn:aws:iam::" + ACCOUNT_ID + ":role/"
                                                "keep-glm52-h1g-"
                                                "worker-drain-signal"
                                            ),
                                        ]
                                    }
                                },
                            },
                            {
                                "Sid": ("DenyEveryOtherPrincipalRehearsalRead"),
                                "Effect": "Deny",
                                "Principal": "*",
                                "Action": [
                                    "s3:GetObject",
                                    "s3:GetObjectVersion",
                                    "s3:GetObjectVersionAttributes",
                                    "s3:ListBucket",
                                    "s3:ListBucketVersions",
                                ],
                                "Resource": [
                                    {
                                        "Fn::GetAtt": [
                                            "RehearsalBucket",
                                            "Arn",
                                        ]
                                    },
                                    {"Fn::Sub": ("${RehearsalBucket.Arn}/rehearsal/*")},
                                ],
                                "Condition": {
                                    "ArnNotEquals": {
                                        "aws:PrincipalArn": [
                                            {
                                                "Fn::GetAtt": [
                                                    "RehearsalCollectorRole",
                                                    "Arn",
                                                ]
                                            },
                                            {
                                                "Fn::GetAtt": [
                                                    "BudgetGateRole",
                                                    "Arn",
                                                ]
                                            },
                                        ]
                                    }
                                },
                            },
                            {
                                "Sid": ("DenyEveryNonCollectorRehearsalMutation"),
                                "Effect": "Deny",
                                "Principal": "*",
                                "Action": [
                                    "s3:AbortMultipartUpload",
                                    "s3:DeleteObject",
                                    "s3:DeleteObjectVersion",
                                    "s3:PutObject",
                                ],
                                "Resource": {
                                    "Fn::Sub": ("${RehearsalBucket.Arn}/rehearsal/*")
                                },
                                "Condition": {
                                    "ArnNotEquals": {
                                        "aws:PrincipalArn": {
                                            "Fn::GetAtt": [
                                                "RehearsalCollectorRole",
                                                "Arn",
                                            ]
                                        }
                                    }
                                },
                            },
                        ],
                    },
                },
            },
        }
    )
    task11_writer_bindings = {
        binding["role_id"]: binding for binding in inputs.task11_writer_bindings
    }

    def source_worker_policy(
        source_name: str,
    ) -> Mapping[str, object]:
        binding = task11_writer_bindings[source_name]
        object_arn = inputs.model_bucket_arn + "/" + binding["output_key"]
        read_object_arns = [
            inputs.model_bucket_arn + "/" + key for key in binding["read_keys"]
        ]
        encryption_context_key = "kms:EncryptionContext:aws:s3:arn"
        return {
            "PolicyName": source_name + "ExactCoordinatesOnly",
            "PolicyDocument": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "ReadWriteExactActivationLedger",
                        "Effect": "Allow",
                        "Action": [
                            "dynamodb:GetItem",
                            "dynamodb:TransactGetItems",
                            "dynamodb:TransactWriteItems",
                        ],
                        "Resource": inputs.ledger_table_arn,
                        "Condition": {
                            "ForAllValues:StringEquals": {
                                "dynamodb:LeadingKeys": (binding["ledger_leading_keys"])
                            }
                        },
                    },
                    {
                        "Sid": "ReadOnlyBoundTask11Inputs",
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObjectVersion",
                            "s3:GetObjectVersionAttributes",
                        ],
                        "Resource": read_object_arns,
                    },
                    {
                        "Sid": "WriteOwnExactOutput",
                        "Effect": "Allow",
                        "Action": "s3:PutObject",
                        "Resource": object_arn,
                    },
                    {
                        "Sid": "DecryptRequiredTask11ObjectsViaS3Only",
                        "Effect": "Allow",
                        "Action": "kms:Decrypt",
                        "Resource": inputs.retained_kms_key_arn,
                        "Condition": {
                            "StringEquals": {
                                "kms:ViaService": ("s3.us-west-2.amazonaws.com"),
                                encryption_context_key: read_object_arns,
                            },
                        },
                    },
                    {
                        "Sid": "EncryptOwnExactKeyFamilyViaS3Only",
                        "Effect": "Allow",
                        "Action": "kms:GenerateDataKey",
                        "Resource": inputs.retained_kms_key_arn,
                        "Condition": {
                            "StringEquals": {
                                "kms:ViaService": ("s3.us-west-2.amazonaws.com"),
                                encryption_context_key: object_arn,
                            },
                        },
                    },
                ],
            },
        }

    for source_name in task11_writer_bindings:
        source_role = _service_role(
            service="lambda.amazonaws.com",
            policies=(source_worker_policy(source_name),),
        )
        source_role_name = (
            "keep-glm52-h1g-"
            + re.sub(
                r"([a-z0-9])([A-Z])",
                r"\1-\2",
                source_name,
            ).lower()
        )
        source_role["RoleName"] = source_role_name
        resources[f"{source_name}Role"] = {
            "Type": "AWS::IAM::Role",
            "Properties": source_role,
        }
    fence_name = "FenceExecutor"
    fence_entries = inputs.fence_template_inventory
    fence_manifest = _support_fence_manifest_coordinate(inputs)
    fence_object_arns = [
        inputs.model_bucket_arn + "/" + fence_manifest["key"],
        *[
            inputs.model_bucket_arn + "/" + entry["template_key"]
            for entry in fence_entries
        ],
    ]
    resources[f"{fence_name}Role"] = {
        "Type": "AWS::IAM::Role",
        "Properties": _service_role(
            service="lambda.amazonaws.com",
            policies=(
                {
                    "PolicyName": fence_name + "ExactTransitionOnly",
                    "PolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Sid": ("CreateOnlyAuthenticatedFenceChangeSets"),
                                "Effect": "Allow",
                                "Action": "cloudformation:CreateChangeSet",
                                "Resource": inputs.fence_stack_id,
                                "Condition": {
                                    "ArnEquals": {
                                        "cloudformation:RoleArn": (
                                            inputs.fence_service_role_arn
                                        )
                                    },
                                    "ForAllValues:StringEquals": {
                                        "cloudformation:ResourceTypes": [
                                            "AWS::S3::BucketPolicy"
                                        ]
                                    },
                                },
                            },
                            {
                                "Sid": ("ExecuteOnlyAuthenticatedFenceChangeSets"),
                                "Effect": "Allow",
                                "Action": [
                                    "cloudformation:DescribeChangeSet",
                                    "cloudformation:ExecuteChangeSet",
                                ],
                                "Resource": inputs.fence_stack_id,
                            },
                            {
                                "Sid": "ReadExactFenceStackState",
                                "Effect": "Allow",
                                "Action": [
                                    "cloudformation:DescribeStacks",
                                    "cloudformation:GetTemplate",
                                ],
                                "Resource": inputs.fence_stack_id,
                            },
                            {
                                "Sid": "ReadExactFenceInventory",
                                "Effect": "Allow",
                                "Action": [
                                    "s3:GetObjectVersion",
                                    "s3:GetObjectVersionAttributes",
                                ],
                                "Resource": fence_object_arns,
                            },
                            {
                                "Sid": "ReadExactCampaignBucketPolicy",
                                "Effect": "Allow",
                                "Action": "s3:GetBucketPolicy",
                                "Resource": inputs.model_bucket_arn,
                            },
                            {
                                "Sid": "ReadWriteExactFenceActions",
                                "Effect": "Allow",
                                "Action": [
                                    "dynamodb:GetItem",
                                    "dynamodb:TransactGetItems",
                                    "dynamodb:TransactWriteItems",
                                ],
                                "Resource": inputs.ledger_table_arn,
                            },
                            {
                                "Sid": "DecryptExactFenceObjectsViaS3Only",
                                "Effect": "Allow",
                                "Action": "kms:Decrypt",
                                "Resource": inputs.retained_kms_key_arn,
                                "Condition": {
                                    "StringEquals": {
                                        "kms:ViaService": ("s3.us-west-2.amazonaws.com")
                                    },
                                    "StringLike": {
                                        "kms:EncryptionContext:aws:s3:arn": (
                                            fence_object_arns
                                        )
                                    },
                                },
                            },
                            {
                                "Sid": "PassExactFenceServiceRole",
                                "Effect": "Allow",
                                "Action": "iam:PassRole",
                                "Resource": inputs.fence_service_role_arn,
                                "Condition": {
                                    "StringEquals": {
                                        "iam:PassedToService": (
                                            "cloudformation.amazonaws.com"
                                        )
                                    }
                                },
                            },
                            {
                                "Sid": "ReadExactFenceStabilizationRoles",
                                "Effect": "Allow",
                                "Action": [
                                    "iam:GetRole",
                                    "iam:GetRolePolicy",
                                    "iam:ListAttachedRolePolicies",
                                    "iam:ListRolePolicies",
                                ],
                                "Resource": [
                                    inputs.fence_service_role_arn,
                                    {
                                        "Fn::Sub": (
                                            "arn:${AWS::Partition}:iam::"
                                            "${AWS::AccountId}:role/"
                                            "keep-glm52-h1g-fence-executor"
                                        )
                                    },
                                    {
                                        "Fn::Sub": (
                                            "arn:${AWS::Partition}:iam::"
                                            "${AWS::AccountId}:role/"
                                            "keep-glm52-h1g-fence-successor"
                                        )
                                    },
                                    *[
                                        {
                                            "Fn::Sub": (
                                                "arn:${AWS::Partition}:iam::"
                                                "${AWS::AccountId}:role/"
                                                "keep-glm52-h1g-"
                                                + re.sub(
                                                    r"([a-z0-9])([A-Z])",
                                                    r"\1-\2",
                                                    source_name,
                                                ).lower()
                                            )
                                        }
                                        for source_name in task11_writer_bindings
                                        if source_name.startswith("Source")
                                    ],
                                ],
                            },
                            {
                                "Sid": "ReadStabilizationManagedPolicyVersions",
                                "Effect": "Allow",
                                "Action": [
                                    "iam:GetPolicy",
                                    "iam:GetPolicyVersion",
                                ],
                                "Resource": "arn:aws:iam::*:policy/*",
                            },
                        ],
                    },
                },
            ),
        ),
    }
    resources[f"{fence_name}Role"]["Properties"]["RoleName"] = (
        "keep-glm52-h1g-fence-executor"
    )
    closure_policy_statements = resources["ClosureSessionRole"]["Properties"][
        "Policies"
    ][0]["PolicyDocument"]["Statement"]
    collector_policy_statements = resources["RehearsalCollectorRole"]["Properties"][
        "Policies"
    ][0]["PolicyDocument"]["Statement"]
    for statement in closure_policy_statements:
        sid = statement.get("Sid")
        if sid in REHEARSAL_TASK11_READ_SIDS:
            projected = deepcopy(statement)
            projected["Sid"] = "Rehearsal" + str(sid)
            collector_policy_statements.append(projected)
    resources["SupportDeadlineRole"] = {
        "Type": "AWS::IAM::Role",
        "Properties": _service_role(
            service="lambda.amazonaws.com",
            policies=(
                {
                    "PolicyName": "EnforceExactSupportDeadlinesOnly",
                    "PolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Sid": "StopExactCombinedHost",
                                "Effect": "Allow",
                                "Action": "ec2:StopInstances",
                                "Resource": {
                                    "Fn::Sub": (
                                        f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:"
                                        "instance/${CombinedHost}"
                                    )
                                },
                                "Condition": {
                                    "StringEquals": {
                                        "ec2:ResourceTag/ActivationId": (
                                            inputs.activation_id
                                        ),
                                        "ec2:ResourceTag/RunId": RUN_ID,
                                        "ec2:ResourceTag/Purpose": ("combined-host"),
                                    }
                                },
                            },
                            {
                                "Sid": "DisableExactHostEgress",
                                "Effect": "Allow",
                                "Action": ("ec2:RevokeSecurityGroupEgress"),
                                "Resource": {
                                    "Fn::Sub": (
                                        f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:"
                                        "security-group/"
                                        "${CombinedHostSecurityGroup}"
                                    )
                                },
                                "Condition": {
                                    "StringEquals": {
                                        "ec2:ResourceTag/ActivationId": (
                                            inputs.activation_id
                                        ),
                                        "ec2:ResourceTag/RunId": RUN_ID,
                                        "ec2:ResourceTag/Purpose": (
                                            "combined-host-egress"
                                        ),
                                    }
                                },
                            },
                            {
                                "Sid": "ReadExactSupportStackOnly",
                                "Effect": "Allow",
                                "Action": "cloudformation:DescribeStacks",
                                "Resource": {"Ref": "AWS::StackId"},
                            },
                            {
                                "Sid": "WriteAndReadExactDeadlineControl",
                                "Effect": "Allow",
                                "Action": [
                                    "dynamodb:PutItem",
                                    "dynamodb:GetItem",
                                    "dynamodb:UpdateItem",
                                    "dynamodb:Query",
                                ],
                                "Resource": inputs.ledger_table_arn,
                                "Condition": {
                                    "ForAllValues:StringEquals": {
                                        "dynamodb:LeadingKeys": [
                                            (
                                                "SUPPORT_DEADLINE#"
                                                f"{inputs.activation_id}"
                                            ),
                                            "RUN#" + RUN_ID,
                                        ]
                                    }
                                },
                            },
                            {
                                "Sid": "ReadExactContinuationObjects",
                                "Effect": "Allow",
                                "Action": [
                                    "s3:ListBucketVersions",
                                    "s3:GetObjectVersion",
                                    "s3:GetObjectVersionAttributes",
                                ],
                                "Resource": [
                                    inputs.model_bucket_arn,
                                    (
                                        inputs.model_bucket_arn
                                        + "/campaigns/"
                                        + RUN_ID
                                        + "/*"
                                    ),
                                    (
                                        inputs.model_bucket_arn + "/task13/production/"
                                        "task10-worker-descriptor.json"
                                    ),
                                ],
                            },
                            {
                                "Sid": ("DecryptExactContinuationObjectsViaS3Only"),
                                "Effect": "Allow",
                                "Action": "kms:Decrypt",
                                "Resource": inputs.retained_kms_key_arn,
                                "Condition": {
                                    "StringEquals": {
                                        "kms:ViaService": ("s3.us-west-2.amazonaws.com")
                                    },
                                    "StringLike": {
                                        ("kms:EncryptionContext:aws:s3:arn"): [
                                            (
                                                inputs.model_bucket_arn
                                                + "/campaigns/"
                                                + RUN_ID
                                                + "/*"
                                            ),
                                            (
                                                inputs.model_bucket_arn
                                                + "/task13/production/"
                                                "task10-worker-descriptor.json"
                                            ),
                                        ]
                                    },
                                },
                            },
                            {
                                "Sid": "InvokeExactContinuationCallees",
                                "Effect": "Allow",
                                "Action": "lambda:InvokeFunction",
                                "Resource": [
                                    {"Ref": "NumericBindingVersion"},
                                    {"Ref": ("RetainedCancellationVersion")},
                                    {
                                        "Fn::ImportValue": (
                                            "KeepGlm52Task12WorkerDrainVersionArn"
                                        )
                                    },
                                    {
                                        "Fn::ImportValue": (
                                            "KeepGlm52Task12TerminalV2VersionArn"
                                        )
                                    },
                                    {
                                        "Fn::ImportValue": (
                                            "KeepGlm52Task9LiabilityWatcherVersionArn"
                                        )
                                    },
                                ],
                            },
                            {
                                "Sid": "RequestExactRetainedLifecycle",
                                "Effect": "Allow",
                                "Action": "events:PutEvents",
                                "Resource": (
                                    f"arn:aws:events:{REGION}:{ACCOUNT_ID}:"
                                    "event-bus/default"
                                ),
                            },
                            {
                                "Sid": "ReadActivationCompute",
                                "Effect": "Allow",
                                "Action": "ec2:DescribeInstances",
                                "Resource": "*",
                            },
                            {
                                "Sid": "ReadControllerAndWorkerSsmStatus",
                                "Effect": "Allow",
                                "Action": ("ssm:DescribeInstanceInformation"),
                                "Resource": "*",
                            },
                            {
                                "Sid": ("ReadExactRetainedLifecycleExecutionHistory"),
                                "Effect": "Allow",
                                "Action": "states:GetExecutionHistory",
                                "Resource": (
                                    f"arn:aws:states:{REGION}:{ACCOUNT_ID}:"
                                    "execution:keep-glm52-h1g-retainedlifecycle:*"
                                ),
                            },
                            {
                                "Sid": "DenyUnguardedInstanceTermination",
                                "Effect": "Deny",
                                "Action": "ec2:TerminateInstances",
                                "Resource": "*",
                            },
                            {
                                "Sid": "PageExactOperatorTopic",
                                "Effect": "Allow",
                                "Action": "sns:Publish",
                                "Resource": SUPPORT_OPERATOR_TOPIC_ARN,
                            },
                        ],
                    },
                },
            ),
        ),
    }
    resources["SupportAsyncDlq"] = {
        "Type": "AWS::SQS::Queue",
        "Properties": {
            "KmsMasterKeyId": inputs.retained_kms_key_arn,
            "MessageRetentionPeriod": 1209600,
            "QueueName": (f"keep-glm52-h1g-support-dlq-{inputs.activation_id}"),
        },
    }
    resources["TlsHandlerLogGroup"] = {
        "Type": "AWS::Logs::LogGroup",
        "Properties": {
            "LogGroupName": {"Fn::Sub": "/aws/lambda/${TlsHandlerFunction}"},
            "RetentionInDays": 14,
        },
    }
    resources["TlsHandlerErrorAlarm"] = {
        "Type": "AWS::CloudWatch::Alarm",
        "Properties": {
            "AlarmName": (f"keep-glm52-h1g-{inputs.activation_id}-tls-handler-errors"),
            "ComparisonOperator": "GreaterThanThreshold",
            "Dimensions": [
                {
                    "Name": "FunctionName",
                    "Value": {"Ref": "TlsHandlerFunction"},
                }
            ],
            "EvaluationPeriods": 1,
            "MetricName": "Errors",
            "Namespace": "AWS/Lambda",
            "Period": 60,
            "Statistic": "Sum",
            "Threshold": 0,
            "TreatMissingData": "breaching",
        },
    }
    for name, contract in RUNTIME_FUNCTIONS.items():
        environment_variables: dict[str, object] = {
            "GLM52_ACTIVATION_ID": inputs.activation_id,
            "GLM52_RUN_ID": RUN_ID,
        }
        reader = VPC_SECRET_READERS.get(name)
        if reader is not None:
            environment_variables.update(
                {
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
                    "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_name,
                    "GLM52_COMBINED_HOST_PRIVATE_IP": {
                        "Fn::GetAtt": ["CombinedHost", "PrivateIp"]
                    },
                    "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (
                        inputs.lambda_code_sha256
                    ),
                }
            )
            for logical_id in reader[2]:
                prefix = _SECRET_ENV_PREFIX[logical_id]
                environment_variables[f"GLM52_{prefix}_SECRET_ID"] = {"Ref": logical_id}
                environment_variables[f"GLM52_{prefix}_VERSION_ID"] = {
                    "Fn::GetAtt": [
                        _SECRET_VERSION_SOURCE[logical_id],
                        f"{logical_id}VersionId",
                    ]
                }
                environment_variables[f"GLM52_{prefix}_VERSION_STAGE"] = _secret_stage(
                    inputs, logical_id
                )
        if name == "Attestation":
            environment_variables["GLM52_RELAY_PORT"] = str(inputs.attestation_port)
        if name == "LaunchAdmission":
            environment_variables["GLM52_RELAY_PORT"] = str(
                inputs.launch_admission_port
            )
        if name == "NumericBinding":
            environment_variables["GLM52_RELAY_PORT"] = str(inputs.numeric_binding_port)
        if name == "RetainedCancellation":
            environment_variables["GLM52_RELAY_PORT"] = str(
                inputs.retained_cancellation_port
            )
        if name == "BudgetGate":
            environment_variables.update(
                {
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_REHEARSAL_BUCKET": {"Ref": "RehearsalBucket"},
                    "GLM52_CLOSURE_GATE_KEY": (
                        "rehearsal/gates/"
                        + inputs.activation_id
                        + "/CLOSURE_BUDGET.json"
                    ),
                    "GLM52_EXPECTED_BUCKET_OWNER": ACCOUNT_ID,
                    "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (
                        inputs.lambda_code_sha256
                    ),
                }
            )
        if name == "RehearsalCollector":
            environment_variables.update(
                {
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
                    "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_name,
                    "GLM52_MODEL_BUCKET": inputs.model_bucket_name,
                    "GLM52_MODEL_PREFIX": inputs.model_prefix,
                    "GLM52_FENCE_STACK_ID": inputs.fence_stack_id,
                    "GLM52_SUPPORT_STACK_ID": {"Ref": "AWS::StackId"},
                    "GLM52_CLOSURE_ROLE_ARN": {
                        "Fn::GetAtt": ["ClosureSessionRole", "Arn"]
                    },
                    "GLM52_ATTESTATION_VERSION_ARN": {"Ref": "AttestationVersion"},
                    "GLM52_LAUNCH_ADMISSION_VERSION_ARN": {
                        "Ref": "LaunchAdmissionVersion"
                    },
                    "GLM52_NUMERIC_BINDING_VERSION_ARN": {
                        "Ref": "NumericBindingVersion"
                    },
                    "GLM52_SOURCE_GPU_SPEND_VERSION_ARN": {
                        "Ref": "SourceGpuSpendVersion"
                    },
                    "GLM52_SOURCE_SUBMISSION_INTENT_VERSION_ARN": {
                        "Ref": "SourceSubmissionIntentVersion"
                    },
                    "GLM52_SOURCE_CONTROLLER_BASELINE_VERSION_ARN": {
                        "Ref": "SourceControllerBaselineVersion"
                    },
                    "GLM52_SOURCE_CONTROL_PLANE_READINESS_VERSION_ARN": {
                        "Ref": "SourceControlPlaneReadinessVersion"
                    },
                    "GLM52_SOURCE_SUBMISSION_ACQUISITION_VERSION_ARN": {
                        "Ref": "SourceSubmissionAcquisitionVersion"
                    },
                    "GLM52_FENCE_EXECUTOR_VERSION_ARN": {"Ref": "FenceExecutorVersion"},
                    "GLM52_FENCE_SUCCESSOR_VERSION_ARN": {
                        "Ref": "FenceSuccessorVersion"
                    },
                    "GLM52_CLAIM_WRITER_VERSION_ARN": {"Ref": "ClaimWriterVersion"},
                    "GLM52_DECISION_WRITER_VERSION_ARN": {
                        "Ref": "DecisionWriterVersion"
                    },
                    "GLM52_TERMINAL_V1_WRITER_VERSION_ARN": {
                        "Ref": "TerminalV1WriterVersion"
                    },
                    "GLM52_CLOSURE_HANDOFF_VERSION_ARN": {
                        "Ref": "ClosureHandoffVersion"
                    },
                    "GLM52_DECISION_VERSION_ARN": {"Ref": "DecisionVersion"},
                    "GLM52_REHEARSAL_CONTROLLER_ROLE_ARN": {
                        "Fn::GetAtt": ["FenceExecutorRole", "Arn"]
                    },
                    "GLM52_REHEARSAL_EXECUTOR_ROLE_ARN": {
                        "Fn::GetAtt": ["RehearsalExecutorRole", "Arn"]
                    },
                    "GLM52_REHEARSAL_BUCKET": {"Ref": "RehearsalBucket"},
                    "GLM52_EXPECTED_BUCKET_OWNER": ACCOUNT_ID,
                    "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (
                        inputs.lambda_code_sha256
                    ),
                    "GLM52_REHEARSAL_PROBE_VERSION_ARN": {
                        "Ref": "RehearsalProbeVersion"
                    },
                }
            )
        if name == "RehearsalProbe":
            environment_variables.update(
                {
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_REHEARSAL_BUCKET": {"Ref": "RehearsalBucket"},
                    "GLM52_COMBINED_HOST_PRIVATE_IP": {
                        "Fn::GetAtt": ["CombinedHost", "PrivateIp"]
                    },
                    "GLM52_RELAY_PORT": str(inputs.attestation_port),
                    "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (
                        inputs.lambda_code_sha256
                    ),
                    "GLM52_ATTESTATION_CLIENT_DER_SHA256": {
                        "Fn::GetAtt": [
                            "TlsBundleCustomResource",
                            "AttestationClientDerSha256",
                        ]
                    },
                    "GLM52_ATTESTATION_SERVER_DER_SHA256": {
                        "Fn::GetAtt": [
                            "TlsBundleCustomResource",
                            "AttestationServerDerSha256",
                        ]
                    },
                    **{
                        f"GLM52_{_SECRET_ENV_PREFIX[logical_id]}_SECRET_ID": {
                            "Ref": logical_id
                        }
                        for logical_id in (
                            "RawSkyTokenSecret",
                            "AttestationClientTlsSecret",
                        )
                    },
                    **{
                        f"GLM52_{_SECRET_ENV_PREFIX[logical_id]}_VERSION_ID": {
                            "Fn::GetAtt": [
                                _SECRET_VERSION_SOURCE[logical_id],
                                f"{logical_id}VersionId",
                            ]
                        }
                        for logical_id in (
                            "RawSkyTokenSecret",
                            "AttestationClientTlsSecret",
                        )
                    },
                    **{
                        f"GLM52_{_SECRET_ENV_PREFIX[logical_id]}"
                        "_VERSION_STAGE": _secret_stage(inputs, logical_id)
                        for logical_id in (
                            "RawSkyTokenSecret",
                            "AttestationClientTlsSecret",
                        )
                    },
                }
            )
        if name == "Decision":
            environment_variables.update(
                {
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
                    "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_name,
                    "GLM52_MODEL_BUCKET": inputs.model_bucket_name,
                    "GLM52_MODEL_PREFIX": inputs.model_prefix,
                    "GLM52_FENCE_STACK_ID": inputs.fence_stack_id,
                    "GLM52_SUPPORT_STACK_ID": {"Ref": "AWS::StackId"},
                    "GLM52_CLOSURE_ROLE_ARN": {
                        "Fn::GetAtt": ["ClosureSessionRole", "Arn"]
                    },
                    "GLM52_ATTESTATION_VERSION_ARN": {"Ref": "AttestationVersion"},
                    "GLM52_LAUNCH_ADMISSION_VERSION_ARN": {
                        "Ref": "LaunchAdmissionVersion"
                    },
                    "GLM52_NUMERIC_BINDING_VERSION_ARN": {
                        "Ref": "NumericBindingVersion"
                    },
                    "GLM52_SOURCE_GPU_SPEND_VERSION_ARN": {
                        "Ref": "SourceGpuSpendVersion"
                    },
                    "GLM52_SOURCE_SUBMISSION_INTENT_VERSION_ARN": {
                        "Ref": "SourceSubmissionIntentVersion"
                    },
                    "GLM52_SOURCE_CONTROLLER_BASELINE_VERSION_ARN": {
                        "Ref": "SourceControllerBaselineVersion"
                    },
                    "GLM52_SOURCE_CONTROL_PLANE_READINESS_VERSION_ARN": {
                        "Ref": "SourceControlPlaneReadinessVersion"
                    },
                    "GLM52_SOURCE_SUBMISSION_ACQUISITION_VERSION_ARN": {
                        "Ref": "SourceSubmissionAcquisitionVersion"
                    },
                    "GLM52_FENCE_EXECUTOR_VERSION_ARN": {"Ref": "FenceExecutorVersion"},
                    "GLM52_FENCE_SUCCESSOR_VERSION_ARN": {
                        "Ref": "FenceSuccessorVersion"
                    },
                    "GLM52_CLAIM_WRITER_VERSION_ARN": {"Ref": "ClaimWriterVersion"},
                    "GLM52_DECISION_WRITER_VERSION_ARN": {
                        "Ref": "DecisionWriterVersion"
                    },
                    "GLM52_TERMINAL_V1_WRITER_VERSION_ARN": {
                        "Ref": "TerminalV1WriterVersion"
                    },
                    "GLM52_CLOSURE_HANDOFF_VERSION_ARN": {
                        "Ref": "ClosureHandoffVersion"
                    },
                    "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (
                        inputs.lambda_code_sha256
                    ),
                }
            )
        if name.startswith("Source"):
            environment_variables.update(
                {
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
                    "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_name,
                    "GLM52_CLOSURE_ROLE_ARN": {
                        "Fn::GetAtt": ["ClosureSessionRole", "Arn"]
                    },
                    "GLM52_PUBLISHER_ROLE_ARN": {"Fn::GetAtt": [f"{name}Role", "Arn"]},
                    "GLM52_SOURCE_KIND": name,
                    "GLM52_TASK11_ACTION_KEY": (
                        task11_writer_bindings[name]["action_key"]
                    ),
                    "GLM52_TASK11_INPUT_KEY": (
                        task11_writer_bindings[name]["input_key"]
                    ),
                    "GLM52_TASK11_OUTPUT_KEY": (
                        task11_writer_bindings[name]["output_key"]
                    ),
                    "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (
                        inputs.lambda_code_sha256
                    ),
                }
            )
        if name in {
            "FenceSuccessor",
            "ClaimWriter",
            "DecisionWriter",
            "TerminalV1Writer",
            "ClosureHandoff",
        }:
            environment_variables.update(
                {
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
                    "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_name,
                    "GLM52_CLOSURE_ROLE_ARN": {
                        "Fn::GetAtt": ["ClosureSessionRole", "Arn"]
                    },
                    "GLM52_WRITER_ROLE_ARN": {"Fn::GetAtt": [f"{name}Role", "Arn"]},
                    "GLM52_EFFECT_WRITER_KIND": name,
                    "GLM52_TASK11_ACTION_KEY": (
                        task11_writer_bindings[name]["action_key"]
                    ),
                    "GLM52_TASK11_INPUT_KEY": (
                        task11_writer_bindings[name]["input_key"]
                    ),
                    "GLM52_TASK11_OUTPUT_KEY": (
                        task11_writer_bindings[name]["output_key"]
                    ),
                    "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (
                        inputs.lambda_code_sha256
                    ),
                }
            )
        if name == "FenceExecutor":
            environment_variables.clear()
            environment_variables.update(_support_fence_executor_environment(inputs))
        if name == "SupportDeadline":
            environment_variables.update(
                {
                    "GLM52_SUPPORT_STACK_ID": {"Ref": "AWS::StackId"},
                    "GLM52_SUPPORT_HOST_INSTANCE_ID": {"Ref": "CombinedHost"},
                    "GLM52_COMBINED_HOST_INSTANCE_ID": {"Ref": "CombinedHost"},
                    "GLM52_SUPPORT_HOST_SECURITY_GROUP_ID": {
                        "Ref": "CombinedHostSecurityGroup"
                    },
                    "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_name,
                    "GLM52_EXPECTED_BUCKET_OWNER": ACCOUNT_ID,
                    "GLM52_NUMERIC_BINDING_VERSION_ARN": {
                        "Ref": "NumericBindingVersion"
                    },
                    "GLM52_RETAINED_CANCELLATION_VERSION_ARN": {
                        "Ref": "RetainedCancellationVersion"
                    },
                    "GLM52_WORKER_DRAIN_VERSION_ARN": {
                        "Fn::ImportValue": ("KeepGlm52Task12WorkerDrainVersionArn")
                    },
                    "GLM52_TASK9_LIABILITY_WATCHER_VERSION_ARN": {
                        "Fn::ImportValue": ("KeepGlm52Task9LiabilityWatcherVersionArn")
                    },
                    "GLM52_TERMINAL_V2_VERSION_ARN": {
                        "Fn::ImportValue": ("KeepGlm52Task12TerminalV2VersionArn")
                    },
                    "GLM52_SUPPORT_QUIESCENCE_SECONDS": "60",
                    "GLM52_OPERATOR_TOPIC_ARN": SUPPORT_OPERATOR_TOPIC_ARN,
                    "GLM52_WORK_STOP_HOURS": "68",
                    "GLM52_DELETE_REQUEST_HOURS": "71",
                    "GLM52_ABSENCE_EXPECTED_HOURS": "72",
                    "GLM52_HOST_EGRESS_WARNING_BYTES": str(5 * 1024**3),
                    "GLM52_HOST_EGRESS_DRAIN_BYTES": str(6 * 1024**3),
                    "GLM52_NAT_AUTHORITY_BYTES": str(10 * 1024**3),
                    "GLM52_NAT_GATEWAY_ID": {"Ref": "NatGateway"},
                    "GLM52_DIRECT_CHILD_DELETE_ACTIONS": "FORBIDDEN",
                    "GLM52_POST_72H_RECONCILIATIONS": "1",
                    "GLM52_DELETE_STACK_OWNER": "RETAINED_LIFECYCLE_ONLY",
                }
            )
        properties: dict[str, object] = {
            "Code": {
                "S3Bucket": inputs.lambda_code_bucket,
                "S3Key": inputs.lambda_code_key,
                "S3ObjectVersion": inputs.lambda_code_version_id,
            },
            "Environment": {"Variables": environment_variables},
            "FunctionName": (
                "keep-glm52-h1g-"
                + re.sub(
                    r"([a-z0-9])([A-Z])",
                    r"\1-\2",
                    name,
                ).lower()
            ),
            "Handler": contract["handler"],
            "MemorySize": contract["memory"],
            "ReservedConcurrentExecutions": (
                5 if name in {"RehearsalCollector", "RehearsalProbe"} else 1
            ),
            "Role": {"Fn::GetAtt": [contract["role"], "Arn"]},
            "Runtime": "python3.12",
            "Timeout": contract["timeout"],
        }
        if contract["vpc_attached"]:
            properties["VpcConfig"] = {
                "SecurityGroupIds": [{"Ref": contract["client_security_group"]}],
                "SubnetIds": [
                    {"Ref": "PrimaryIsolatedSubnet"},
                    {"Ref": "AlternateIsolatedSubnet"},
                ],
            }
        function_resource: dict[str, object] = {
            "Type": "AWS::Lambda::Function",
            "Properties": properties,
        }
        if contract["vpc_attached"]:
            function_resource["DependsOn"] = [
                "SkyBootstrapCustomResource",
                "TlsBundleCustomResource",
            ]
        resources[f"{name}Function"] = function_resource
        resources[f"{name}Version"] = {
            "Type": "AWS::Lambda::Version",
            "Properties": {
                "Description": f"code={inputs.lambda_code_sha256}",
                "FunctionName": {"Ref": f"{name}Function"},
            },
        }
        if name in _TASK11_CLOSURE_CALLEES:
            resources[f"{name}ClosureInvokePermission"] = {
                "Type": "AWS::Lambda::Permission",
                "Properties": {
                    "Action": "lambda:InvokeFunction",
                    "FunctionName": {"Ref": f"{name}Version"},
                    "Principal": {"Fn::GetAtt": ["ClosureSessionRole", "Arn"]},
                    "SourceAccount": ACCOUNT_ID,
                },
            }
        if name == "RehearsalCollector":
            resources["RehearsalCollectorExecutorInvokePermission"] = {
                "Type": "AWS::Lambda::Permission",
                "Properties": {
                    "Action": "lambda:InvokeFunction",
                    "FunctionName": {"Ref": "RehearsalCollectorVersion"},
                    "Principal": {"Fn::GetAtt": ["RehearsalExecutorRole", "Arn"]},
                    "SourceAccount": ACCOUNT_ID,
                },
            }
        if name == "RehearsalProbe":
            resources["RehearsalProbeCollectorInvokePermission"] = {
                "Type": "AWS::Lambda::Permission",
                "Properties": {
                    "Action": "lambda:InvokeFunction",
                    "FunctionName": {"Ref": "RehearsalProbeVersion"},
                    "Principal": {
                        "Fn::GetAtt": [
                            "RehearsalCollectorRole",
                            "Arn",
                        ]
                    },
                    "SourceAccount": ACCOUNT_ID,
                },
            }
        resources[f"{name}LogGroup"] = {
            "Type": "AWS::Logs::LogGroup",
            "Properties": {
                "LogGroupName": {"Fn::Sub": (f"/aws/lambda/${{{name}Function}}")},
                "RetentionInDays": 14,
            },
        }
        resources[f"{name}ErrorAlarm"] = {
            "Type": "AWS::CloudWatch::Alarm",
            "Properties": {
                "AlarmName": (
                    f"keep-glm52-h1g-{inputs.activation_id}-{name.lower()}-errors"
                ),
                "ComparisonOperator": "GreaterThanThreshold",
                "Dimensions": [
                    {
                        "Name": "FunctionName",
                        "Value": {"Ref": f"{name}Function"},
                    }
                ],
                "EvaluationPeriods": 1,
                "MetricName": "Errors",
                "Namespace": "AWS/Lambda",
                "Period": 60,
                "Statistic": "Sum",
                "Threshold": 0,
                "TreatMissingData": "breaching",
            },
        }
    resources["SupportDeadlineEventInvokeConfig"] = {
        "Type": "AWS::Lambda::EventInvokeConfig",
        "Properties": {
            "DestinationConfig": {
                "OnFailure": {"Destination": {"Fn::GetAtt": ["SupportAsyncDlq", "Arn"]}}
            },
            "FunctionName": {"Ref": "SupportDeadlineFunction"},
            "MaximumEventAgeInSeconds": 60,
            "MaximumRetryAttempts": 0,
            "Qualifier": {"Fn::GetAtt": ["SupportDeadlineVersion", "Version"]},
        },
    }
    resources["SupportScheduleInvokeRole"] = {
        "Type": "AWS::IAM::Role",
        "Properties": _service_role(
            service="scheduler.amazonaws.com",
            policies=(
                {
                    "PolicyName": "InvokeExactSupportDeadlineVersionOnly",
                    "PolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Action": "lambda:InvokeFunction",
                                "Resource": {"Ref": "SupportDeadlineVersion"},
                            }
                        ],
                    },
                },
            ),
        ),
    }
    start = datetime.fromisoformat(inputs.activation_started_at.replace("Z", "+00:00"))
    for logical_id, hours, payload in (
        ("WorkStopSchedule", 68, "WORK_STOP"),
        ("DeleteRequestSchedule", 71, "DELETE_REQUEST"),
        ("AbsenceDeadlineSchedule", 72, "ABSENCE_EXPECTED"),
    ):
        at = (start + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
        resources[logical_id] = {
            "Type": "AWS::Scheduler::Schedule",
            "Properties": {
                "FlexibleTimeWindow": {"Mode": "OFF"},
                "ScheduleExpression": f"at({at})",
                "ScheduleExpressionTimezone": "UTC",
                "State": "ENABLED",
                "Target": {
                    "Arn": {"Ref": "SupportDeadlineVersion"},
                    "Input": (
                        '{"activation_id":"'
                        f'{inputs.activation_id}","deadline":"{payload}"'
                        "}"
                    ),
                    "RoleArn": {"Fn::GetAtt": ["SupportScheduleInvokeRole", "Arn"]},
                },
            },
        }
    host_metric = {
        "Dimensions": [
            {
                "Name": "ActivationId",
                "Value": inputs.activation_id,
            }
        ],
        "MetricName": "HostNonEndpointBytes",
        "Namespace": "GLM52/H1g",
        "Period": 60,
        "Statistic": "Maximum",
    }
    cumulative_nat_metric = {
        "Dimensions": [
            {
                "Name": "ActivationId",
                "Value": inputs.activation_id,
            },
            {
                "Name": "NatGatewayId",
                "Value": {"Ref": "NatGateway"},
            },
        ],
        "MetricName": "ActivationCumulativeNatProcessedBytes",
        "Namespace": "GLM52/H1g",
        "Period": 60,
        "Statistic": "Maximum",
    }
    for logical_id, threshold in (
        ("HostEgressWarningAlarm", 5 * 1024**3),
        ("HostEgressDrainAlarm", 6 * 1024**3),
    ):
        resources[logical_id] = {
            "Type": "AWS::CloudWatch::Alarm",
            "Properties": {
                "AlarmName": (
                    f"keep-glm52-h1g-{inputs.activation_id}-{logical_id.lower()}"
                ),
                "ComparisonOperator": "GreaterThanOrEqualToThreshold",
                "EvaluationPeriods": 1,
                **host_metric,
                "Threshold": threshold,
                "TreatMissingData": "notBreaching",
            },
        }
    for logical_id, threshold in (
        ("NatEgressWarningAlarm", 5 * 1024**3),
        ("NatEgressDrainAlarm", 6 * 1024**3),
        ("NatEgressAuthorityAlarm", 10 * 1024**3),
    ):
        resources[logical_id] = {
            "Type": "AWS::CloudWatch::Alarm",
            "Properties": {
                "AlarmName": (
                    f"keep-glm52-h1g-{inputs.activation_id}-{logical_id.lower()}"
                ),
                "ComparisonOperator": "GreaterThanOrEqualToThreshold",
                "EvaluationPeriods": 1,
                **cumulative_nat_metric,
                "Threshold": threshold,
                "TreatMissingData": "notBreaching",
            },
        }
    resources["SupportEgressAlarmRule"] = {
        "Type": "AWS::Events::Rule",
        "Properties": {
            "EventPattern": {
                "detail": {
                    "alarmName": [
                        {"prefix": (f"keep-glm52-h1g-{inputs.activation_id}-")}
                    ],
                    "state": {"value": ["ALARM"]},
                },
                "detail-type": ["CloudWatch Alarm State Change"],
                "source": ["aws.cloudwatch"],
            },
            "State": "ENABLED",
            "Targets": [
                {
                    "Arn": {"Ref": "SupportDeadlineVersion"},
                    "Id": "ExactSupportDeadlineVersion",
                    "RetryPolicy": {
                        "MaximumEventAgeInSeconds": 60,
                        "MaximumRetryAttempts": 0,
                    },
                }
            ],
        },
    }
    resources["SupportEgressAlarmInvokePermission"] = {
        "Type": "AWS::Lambda::Permission",
        "Properties": {
            "Action": "lambda:InvokeFunction",
            "FunctionName": {"Ref": "SupportDeadlineVersion"},
            "Principal": "events.amazonaws.com",
            "SourceArn": {"Fn::GetAtt": ["SupportEgressAlarmRule", "Arn"]},
        },
    }
    resources["SupportWorkflowRole"]["Properties"]["Policies"] = [
        {
            "PolicyName": "InvokeExactPublishedSupportVersionsOnly",
            "PolicyDocument": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "lambda:InvokeFunction",
                        "Resource": [
                            {"Ref": "BudgetGateVersion"},
                            {"Ref": "DecisionVersion"},
                            {"Ref": "SupportDeadlineVersion"},
                        ],
                    }
                ],
            },
        }
    ]
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "GLM-5.2 H.1g bounded ephemeral support plane",
        "Metadata": {
            "ActivationId": inputs.activation_id,
            "HostBootIdentitySha256": inputs.host_boot_identity_sha256,
            "HostUserDataSha256": inputs.host_user_data_sha256,
            "InputIdentitySha256": _support_input_identity(inputs),
            "NoAwsMutation": True,
            "RetainedPublicS3EndpointId": inputs.retained_public_s3_endpoint_id,
        },
        "Resources": resources,
        "Outputs": {
            output_name: {
                "Value": {"Fn::GetAtt": ["TlsBundleCustomResource", attribute]}
            }
            for output_name, attribute in (
                ("IssuanceId", "issuance_id"),
                ("BundleSha256", "bundle_sha256"),
                ("TlsCaDerSha256", "TlsCaDerSha256"),
                ("AttestationClientDerSha256", "AttestationClientDerSha256"),
                ("AttestationServerDerSha256", "AttestationServerDerSha256"),
                (
                    "LaunchAdmissionClientDerSha256",
                    "LaunchAdmissionClientDerSha256",
                ),
                (
                    "LaunchAdmissionServerDerSha256",
                    "LaunchAdmissionServerDerSha256",
                ),
                ("NumericBindingClientDerSha256", "NumericBindingClientDerSha256"),
                ("NumericBindingServerDerSha256", "NumericBindingServerDerSha256"),
                (
                    "RetainedCancellationClientDerSha256",
                    "RetainedCancellationClientDerSha256",
                ),
                (
                    "RetainedCancellationServerDerSha256",
                    "RetainedCancellationServerDerSha256",
                ),
                ("TlsNotValidBefore", "TlsNotValidBefore"),
                ("TlsNotValidAfter", "TlsNotValidAfter"),
                (
                    "AttestationClientTlsSecretVersionId",
                    "AttestationClientTlsSecretVersionId",
                ),
                (
                    "LaunchAdmissionClientTlsSecretVersionId",
                    "LaunchAdmissionClientTlsSecretVersionId",
                ),
                (
                    "NumericBindingClientTlsSecretVersionId",
                    "NumericBindingClientTlsSecretVersionId",
                ),
                (
                    "RetainedCancellationClientTlsSecretVersionId",
                    "RetainedCancellationClientTlsSecretVersionId",
                ),
                (
                    "CombinedHostTlsSecretVersionId",
                    "CombinedHostTlsSecretVersionId",
                ),
                (
                    "DirectGrantEvidenceCoordinate",
                    "DirectGrantEvidenceCoordinate",
                ),
                (
                    "DirectGrantEvidenceSha256",
                    "DirectGrantEvidenceSha256",
                ),
            )
        },
    }


def support_resource_cardinality(
    template: Mapping[str, object],
) -> Mapping[str, int]:
    if type(template) is not dict or type(template.get("Resources")) is not dict:
        raise ValueError("support template lacks exact resources")
    counts: dict[str, int] = {}
    for resource in template["Resources"].values():
        if type(resource) is not dict or type(resource.get("Type")) is not str:
            raise ValueError("support resource is malformed")
        resource_type = resource["Type"]
        counts[resource_type] = counts.get(resource_type, 0) + 1
    return counts


def validate_task11_runtime_role_authority(
    template: Mapping[str, object], inputs: object
) -> bool:
    """Prove the finite production IAM needed by Task 11 remains exact."""

    if type(template) is not dict or type(template.get("Resources")) is not dict:
        raise ValueError("Task 11 runtime authority requires exact resources")
    resources = template["Resources"]

    def statements(role_id: str) -> list[Mapping[str, object]]:
        role = resources.get(role_id, {}).get("Properties", {})
        policies = role.get("Policies")
        if type(policies) is not list:
            raise ValueError(f"{role_id} inline policy inventory is malformed")
        result = [
            statement
            for policy in policies
            for statement in policy.get("PolicyDocument", {}).get("Statement", [])
        ]
        if any(type(statement) is not dict for statement in result):
            raise ValueError(f"{role_id} contains a malformed statement")
        return result

    collector_properties = resources.get("RehearsalCollectorRole", {}).get(
        "Properties", {}
    )
    collector_policies = collector_properties.get("Policies")
    if (
        collector_properties.get("RoleName") != "keep-glm52-h1g-rehearsal-collector"
        or collector_properties.get("AssumeRolePolicyDocument")
        != {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        or type(collector_policies) is not list
        or len(collector_policies) != 1
        or collector_policies[0].get("PolicyName") != "RehearsalCollectorCanaryOnly"
    ):
        raise ValueError("rehearsal collector role identity drifted")
    collector_statements = {
        statement.get("Sid"): statement
        for statement in statements("RehearsalCollectorRole")
    }
    if set(collector_statements) != {
        "WriteExactCollectorLogs",
        "ListExactRehearsalVersions",
        "ReadExactRehearsalBucketPolicy",
        "ReadWriteExactRehearsalObjects",
        "UseExactRehearsalEncryptionContextsViaS3",
        "InvokeExactReadOnlyProbeVersion",
        "DenyNonProbeLambdaInvocation",
        "DenyCampaignAndModelBucketMutation",
        "DenyProductionMutations",
        *{"Rehearsal" + sid for sid in REHEARSAL_TASK11_READ_SIDS},
    }:
        raise ValueError("rehearsal collector statement inventory drifted")
    collector_prefixes = [
        (
            "rehearsal/canary/"
            + inputs.activation_id
            + "/"
            + inputs.lambda_code_sha256
            + "/*"
        ),
        (
            "rehearsal/measurements/"
            + inputs.activation_id
            + "/"
            + inputs.lambda_code_sha256
            + "/*"
        ),
        "rehearsal/gates/" + inputs.activation_id + "/*",
    ]
    collector_object_arns = [
        {"Fn::Sub": "${RehearsalBucket.Arn}/" + prefix} for prefix in collector_prefixes
    ]
    if (
        collector_statements["WriteExactCollectorLogs"]
        != {
            "Sid": "WriteExactCollectorLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            "Resource": (
                "arn:aws:logs:" + REGION + ":" + ACCOUNT_ID + ":log-group:/aws/lambda/"
                "keep-glm52-h1g-rehearsal-collector:*"
            ),
        }
        or collector_statements["ListExactRehearsalVersions"]
        != {
            "Sid": "ListExactRehearsalVersions",
            "Effect": "Allow",
            "Action": "s3:ListBucketVersions",
            "Resource": {"Fn::GetAtt": ["RehearsalBucket", "Arn"]},
            "Condition": {"StringLike": {"s3:prefix": collector_prefixes}},
        }
        or collector_statements["ReadExactRehearsalBucketPolicy"]
        != {
            "Sid": "ReadExactRehearsalBucketPolicy",
            "Effect": "Allow",
            "Action": "s3:GetBucketPolicy",
            "Resource": {"Fn::GetAtt": ["RehearsalBucket", "Arn"]},
        }
        or collector_statements["ReadWriteExactRehearsalObjects"]
        != {
            "Sid": "ReadWriteExactRehearsalObjects",
            "Effect": "Allow",
            "Action": [
                "s3:GetObjectVersion",
                "s3:GetObjectVersionAttributes",
                "s3:PutObject",
            ],
            "Resource": collector_object_arns,
        }
        or collector_statements["UseExactRehearsalEncryptionContextsViaS3"]
        != {
            "Sid": "UseExactRehearsalEncryptionContextsViaS3",
            "Effect": "Allow",
            "Action": [
                "kms:Decrypt",
                "kms:Encrypt",
                "kms:GenerateDataKey",
            ],
            "Resource": inputs.retained_kms_key_arn,
            "Condition": {
                "StringEquals": {"kms:ViaService": "s3.us-west-2.amazonaws.com"},
                "StringLike": {
                    "kms:EncryptionContext:aws:s3:arn": (collector_object_arns)
                },
            },
        }
        or collector_statements["InvokeExactReadOnlyProbeVersion"]
        != {
            "Sid": "InvokeExactReadOnlyProbeVersion",
            "Effect": "Allow",
            "Action": "lambda:InvokeFunction",
            "Resource": {"Ref": "RehearsalProbeVersion"},
        }
        or collector_statements["DenyNonProbeLambdaInvocation"]
        != {
            "Sid": "DenyNonProbeLambdaInvocation",
            "Effect": "Deny",
            "Action": "lambda:InvokeFunction",
            "NotResource": {"Ref": "RehearsalProbeVersion"},
        }
        or collector_statements["DenyCampaignAndModelBucketMutation"]
        != {
            "Sid": "DenyCampaignAndModelBucketMutation",
            "Effect": "Deny",
            "Action": [
                "s3:AbortMultipartUpload",
                "s3:DeleteObject*",
                "s3:PutObject*",
            ],
            "Resource": [
                inputs.model_bucket_arn,
                inputs.model_bucket_arn + "/*",
            ],
        }
        or collector_statements["DenyProductionMutations"]
        != {
            "Sid": "DenyProductionMutations",
            "Effect": "Deny",
            "Action": [
                "cloudformation:Create*",
                "cloudformation:Delete*",
                "cloudformation:Execute*",
                "cloudformation:Update*",
                "dynamodb:BatchWriteItem",
                "dynamodb:DeleteItem",
                "dynamodb:PutItem",
                "dynamodb:TransactWriteItems",
                "dynamodb:UpdateItem",
                "ec2:AllocateAddress",
                "ec2:Associate*",
                "ec2:Attach*",
                "ec2:Authorize*",
                "ec2:Create*",
                "ec2:Delete*",
                "ec2:Detach*",
                "ec2:Disassociate*",
                "ec2:Modify*",
                "ec2:ReleaseAddress",
                "ec2:Replace*",
                "ec2:Reset*",
                "ec2:Revoke*",
                "ec2:Run*",
                "ec2:Start*",
                "ec2:Stop*",
                "ec2:Terminate*",
                "events:PutEvents",
                "execute-api:Invoke",
                "iam:Add*",
                "iam:Attach*",
                "iam:Create*",
                "iam:Delete*",
                "iam:Detach*",
                "iam:PassRole",
                "iam:Put*",
                "iam:Remove*",
                "iam:Set*",
                "iam:Tag*",
                "iam:Untag*",
                "iam:Update*",
                "secretsmanager:*",
                "sns:Publish",
                "sts:AssumeRole",
                "sts:TagSession",
            ],
            "Resource": "*",
        }
    ):
        raise ValueError("rehearsal collector authority drifted")
    closure_by_sid = {
        statement.get("Sid"): statement
        for statement in statements("ClosureSessionRole")
    }
    for sid in REHEARSAL_TASK11_READ_SIDS:
        expected = deepcopy(closure_by_sid.get(sid))
        if type(expected) is not dict:
            raise ValueError("shared Task 11 read authority is absent")
        expected["Sid"] = "Rehearsal" + sid
        if collector_statements.get("Rehearsal" + sid) != expected:
            raise ValueError("shared Task 11 read authority drifted")

    probe_properties = resources.get("RehearsalProbeRole", {}).get("Properties", {})
    probe_policies = probe_properties.get("Policies")
    if (
        probe_properties.get("RoleName") != "keep-glm52-h1g-rehearsal-probe"
        or probe_properties.get("AssumeRolePolicyDocument")
        != {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        or type(probe_policies) is not list
        or [policy.get("PolicyName") for policy in probe_policies]
        != [
            "ReadRehearsalProbeSecretsOnly",
            "ExecuteReadOnlyRehearsalProbe",
            "ManageLambdaVpcNetworkInterfacesOnly",
        ]
    ):
        raise ValueError("rehearsal probe role identity drifted")
    probe_statements = {
        statement.get("Sid"): statement
        for statement in statements("RehearsalProbeRole")
    }
    if probe_statements.get("WriteExactProbeLogs") != {
        "Sid": "WriteExactProbeLogs",
        "Effect": "Allow",
        "Action": [
            "logs:CreateLogStream",
            "logs:PutLogEvents",
        ],
        "Resource": (
            "arn:aws:logs:" + REGION + ":" + ACCOUNT_ID + ":log-group:/aws/lambda/"
            "keep-glm52-h1g-rehearsal-probe:*"
        ),
    } or probe_statements.get("DenyProbeProductionAuthority") != {
        "Sid": "DenyProbeProductionAuthority",
        "Effect": "Deny",
        "Action": [
            "cloudformation:*",
            "dynamodb:*",
            "ec2:Run*",
            "ec2:Start*",
            "ec2:Stop*",
            "ec2:Terminate*",
            "events:PutEvents",
            "execute-api:Invoke",
            "iam:*",
            "lambda:InvokeFunction",
            "s3:*",
            "sns:Publish",
            "sts:*",
        ],
        "Resource": "*",
    }:
        raise ValueError("rehearsal probe authority drifted")

    executor_properties = resources.get("RehearsalExecutorRole", {}).get(
        "Properties", {}
    )
    if executor_properties != {
        "RoleName": "keep-glm52-h1g-rehearsal-executor",
        "MaxSessionDuration": 900,
        "AssumeRolePolicyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": {
                            "Fn::GetAtt": [
                                "FenceExecutorRole",
                                "Arn",
                            ]
                        }
                    },
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "ArnEquals": {
                            "aws:PrincipalArn": {
                                "Fn::GetAtt": [
                                    "FenceExecutorRole",
                                    "Arn",
                                ]
                            }
                        },
                        "StringEquals": {
                            "sts:RoleSessionName": (
                                _rehearsal_session_name(inputs.activation_id)
                            )
                        },
                    },
                }
            ],
        },
        "Policies": [
            {
                "PolicyName": ("InvokeExactRehearsalCollectorVersionOnly"),
                "PolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": "lambda:InvokeFunction",
                            "Resource": {"Ref": "RehearsalCollectorVersion"},
                        }
                    ],
                },
            }
        ],
    }:
        raise ValueError("rehearsal executor authority drifted")

    expected_collector_permission = {
        "Type": "AWS::Lambda::Permission",
        "Properties": {
            "Action": "lambda:InvokeFunction",
            "FunctionName": {"Ref": "RehearsalCollectorVersion"},
            "Principal": {"Fn::GetAtt": ["RehearsalExecutorRole", "Arn"]},
            "SourceAccount": ACCOUNT_ID,
        },
    }
    expected_probe_permission = {
        "Type": "AWS::Lambda::Permission",
        "Properties": {
            "Action": "lambda:InvokeFunction",
            "FunctionName": {"Ref": "RehearsalProbeVersion"},
            "Principal": {"Fn::GetAtt": ["RehearsalCollectorRole", "Arn"]},
            "SourceAccount": ACCOUNT_ID,
        },
    }
    closure_invokes = [
        statement
        for statement in statements("ClosureSessionRole")
        if statement.get("Sid") == "InvokeExactSupportCallees"
    ]
    if (
        resources.get("RehearsalCollectorExecutorInvokePermission")
        != expected_collector_permission
        or resources.get("RehearsalProbeCollectorInvokePermission")
        != expected_probe_permission
        or resources.get("RehearsalCollectorClosureInvokePermission") is not None
        or resources.get("RehearsalProbeClosureInvokePermission") is not None
        or len(closure_invokes) != 1
        or {"Ref": "RehearsalCollectorVersion"}
        in closure_invokes[0].get("Resource", [])
        or {"Ref": "RehearsalProbeVersion"} in closure_invokes[0].get("Resource", [])
    ):
        raise ValueError("rehearsal invocation partition drifted")

    rehearsal_policy = resources.get("RehearsalBucketPolicy", {}).get("Properties", {})
    bucket_statements = rehearsal_policy.get("PolicyDocument", {}).get("Statement")
    if (
        rehearsal_policy.get("Bucket") != {"Ref": "RehearsalBucket"}
        or type(bucket_statements) is not list
        or len(bucket_statements) != 3
    ):
        raise ValueError("rehearsal bucket policy inventory drifted")
    bucket_by_sid = {
        statement.get("Sid"): statement
        for statement in bucket_statements
        if type(statement) is dict
    }
    read_actions = [
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:GetObjectVersionAttributes",
        "s3:ListBucket",
        "s3:ListBucketVersions",
    ]
    read_resources = [
        {"Fn::GetAtt": ["RehearsalBucket", "Arn"]},
        {"Fn::Sub": "${RehearsalBucket.Arn}/rehearsal/*"},
    ]
    named_principals = [
        {"Fn::GetAtt": ["ClosureSessionRole", "Arn"]},
        {"Fn::GetAtt": ["DecisionRole", "Arn"]},
        {"Fn::GetAtt": ["LaunchAdmissionRole", "Arn"]},
        {"Fn::GetAtt": ["CombinedHostRole", "Arn"]},
        ("arn:aws:iam::" + ACCOUNT_ID + ":role/keep-glm52-gpu-worker"),
        ("arn:aws:iam::" + ACCOUNT_ID + ":role/keep-glm52-h1g-worker-drain-signal"),
    ]
    if (
        bucket_by_sid.get("DenyNamedProductionPrincipalRehearsalRead")
        != {
            "Sid": "DenyNamedProductionPrincipalRehearsalRead",
            "Effect": "Deny",
            "Principal": "*",
            "Action": read_actions,
            "Resource": read_resources,
            "Condition": {"ArnEquals": {"aws:PrincipalArn": named_principals}},
        }
        or bucket_by_sid.get("DenyEveryOtherPrincipalRehearsalRead")
        != {
            "Sid": "DenyEveryOtherPrincipalRehearsalRead",
            "Effect": "Deny",
            "Principal": "*",
            "Action": read_actions,
            "Resource": read_resources,
            "Condition": {
                "ArnNotEquals": {
                    "aws:PrincipalArn": [
                        {
                            "Fn::GetAtt": [
                                "RehearsalCollectorRole",
                                "Arn",
                            ]
                        },
                        {
                            "Fn::GetAtt": [
                                "BudgetGateRole",
                                "Arn",
                            ]
                        },
                    ]
                }
            },
        }
        or bucket_by_sid.get("DenyEveryNonCollectorRehearsalMutation")
        != {
            "Sid": "DenyEveryNonCollectorRehearsalMutation",
            "Effect": "Deny",
            "Principal": "*",
            "Action": [
                "s3:AbortMultipartUpload",
                "s3:DeleteObject",
                "s3:DeleteObjectVersion",
                "s3:PutObject",
            ],
            "Resource": {"Fn::Sub": "${RehearsalBucket.Arn}/rehearsal/*"},
            "Condition": {
                "ArnNotEquals": {
                    "aws:PrincipalArn": {
                        "Fn::GetAtt": [
                            "RehearsalCollectorRole",
                            "Arn",
                        ]
                    }
                }
            },
        }
    ):
        raise ValueError("production rehearsal readability drifted")

    eni_actions = {
        "ec2:CreateNetworkInterface",
        "ec2:DescribeNetworkInterfaces",
        "ec2:DeleteNetworkInterface",
    }
    expected_eni = {
        "Sid": "ManageLambdaVpcNetworkInterfacesInRegion",
        "Effect": "Allow",
        "Action": [
            "ec2:CreateNetworkInterface",
            "ec2:DescribeNetworkInterfaces",
            "ec2:DeleteNetworkInterface",
        ],
        "Resource": "*",
        "Condition": {"StringEquals": {"aws:RequestedRegion": REGION}},
    }
    vpc_roles = {
        "RehearsalProbeRole",
        "AttestationRole",
        "LaunchAdmissionRole",
        "NumericBindingRole",
        "RetainedCancellationRole",
    }
    observed_eni_roles = set()
    for logical_id, resource in resources.items():
        if resource.get("Type") != "AWS::IAM::Role":
            continue
        matches = [
            statement
            for statement in statements(logical_id)
            if eni_actions & set(_statement_actions(statement))
        ]
        if matches:
            observed_eni_roles.add(logical_id)
            if matches != [expected_eni]:
                raise ValueError("Lambda VPC ENI authority drifted")
    if observed_eni_roles != vpc_roles:
        raise ValueError("Lambda VPC ENI authority role set drifted")

    campaign_objects = inputs.model_bucket_arn + "/campaigns/" + RUN_ID + "/*"
    expected_decrypt_contexts = {
        "ClosureSessionRole": campaign_objects,
        "LaunchAdmissionRole": campaign_objects,
        "FenceExecutorRole": [
            inputs.model_bucket_arn
            + "/"
            + _support_fence_manifest_coordinate(inputs)["key"],
            *[
                inputs.model_bucket_arn + "/" + entry["template_key"]
                for entry in inputs.fence_template_inventory
            ],
        ],
    }
    for logical_id in expected_decrypt_contexts:
        decrypts = [
            statement
            for statement in statements(logical_id)
            if "kms:Decrypt" in _statement_actions(statement)
        ]
        if len(decrypts) != 1:
            raise ValueError("campaign KMS decrypt role set drifted")
        decrypt = decrypts[0]
        if decrypt.get("Resource") != inputs.retained_kms_key_arn or decrypt.get(
            "Condition"
        ) != {
            "StringEquals": {"kms:ViaService": f"s3.{REGION}.amazonaws.com"},
            "StringLike": {
                "kms:EncryptionContext:aws:s3:arn": (
                    expected_decrypt_contexts[logical_id]
                )
            },
        }:
            raise ValueError("campaign KMS decrypt authority drifted")
    if any(
        "kms:Decrypt" in _statement_actions(statement)
        and statement.get("Resource") != inputs.retained_kms_key_arn
        for logical_id, resource in resources.items()
        if resource.get("Type") == "AWS::IAM::Role"
        for statement in statements(logical_id)
    ):
        raise ValueError("wildcard or foreign KMS decrypt authority exists")

    pass_roles = [
        statement
        for statement in statements("FenceExecutorRole")
        if "iam:PassRole" in _statement_actions(statement)
    ]
    if pass_roles != [
        {
            "Sid": "PassExactFenceServiceRole",
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": inputs.fence_service_role_arn,
            "Condition": {
                "StringEquals": {"iam:PassedToService": "cloudformation.amazonaws.com"}
            },
        }
    ]:
        raise ValueError("fence executor PassRole authority drifted")

    fence_entries = inputs.fence_template_inventory
    fence_object_arns = [
        (
            inputs.model_bucket_arn
            + "/"
            + _support_fence_manifest_coordinate(inputs)["key"]
        ),
        *[
            inputs.model_bucket_arn + "/" + entry["template_key"]
            for entry in fence_entries
        ],
    ]
    finite_fence_statements = {
        statement.get("Sid"): statement for statement in statements("FenceExecutorRole")
    }
    expected_finite_fence = {
        "CreateOnlyAuthenticatedFenceChangeSets": {
            "Sid": "CreateOnlyAuthenticatedFenceChangeSets",
            "Effect": "Allow",
            "Action": "cloudformation:CreateChangeSet",
            "Resource": inputs.fence_stack_id,
            "Condition": {
                "ArnEquals": {
                    "cloudformation:RoleArn": (inputs.fence_service_role_arn)
                },
                "ForAllValues:StringEquals": {
                    "cloudformation:ResourceTypes": ["AWS::S3::BucketPolicy"]
                },
            },
        },
        "ExecuteOnlyAuthenticatedFenceChangeSets": {
            "Sid": "ExecuteOnlyAuthenticatedFenceChangeSets",
            "Effect": "Allow",
            "Action": [
                "cloudformation:DescribeChangeSet",
                "cloudformation:ExecuteChangeSet",
            ],
            "Resource": inputs.fence_stack_id,
        },
        "ReadExactFenceStackState": {
            "Sid": "ReadExactFenceStackState",
            "Effect": "Allow",
            "Action": [
                "cloudformation:DescribeStacks",
                "cloudformation:GetTemplate",
            ],
            "Resource": inputs.fence_stack_id,
        },
        "ReadExactFenceInventory": {
            "Sid": "ReadExactFenceInventory",
            "Effect": "Allow",
            "Action": [
                "s3:GetObjectVersion",
                "s3:GetObjectVersionAttributes",
            ],
            "Resource": fence_object_arns,
        },
    }
    if any(
        finite_fence_statements.get(sid) != expected
        for sid, expected in expected_finite_fence.items()
    ):
        raise ValueError("finite fence inventory authority drifted")
    source_role_ids = [
        binding["role_id"]
        for binding in inputs.task11_writer_bindings
        if binding["role_id"].startswith("Source")
    ]
    source_role_arns = [
        {
            "Fn::Sub": (
                "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/"
                "keep-glm52-h1g-"
                + re.sub(
                    r"([a-z0-9])([A-Z])",
                    r"\1-\2",
                    role_id,
                ).lower()
            )
        }
        for role_id in source_role_ids
    ]
    fence_executor_role_arn = {
        "Fn::Sub": (
            "arn:${AWS::Partition}:iam::${AWS::AccountId}:"
            "role/keep-glm52-h1g-fence-executor"
        )
    }
    exact_fence_stabilization = {
        "ReadExactFenceStabilizationRoles": {
            "Sid": "ReadExactFenceStabilizationRoles",
            "Effect": "Allow",
            "Action": [
                "iam:GetRole",
                "iam:GetRolePolicy",
                "iam:ListAttachedRolePolicies",
                "iam:ListRolePolicies",
            ],
            "Resource": [
                inputs.fence_service_role_arn,
                fence_executor_role_arn,
                {
                    "Fn::Sub": (
                        "arn:${AWS::Partition}:iam::${AWS::AccountId}:"
                        "role/keep-glm52-h1g-fence-successor"
                    )
                },
                *source_role_arns,
            ],
        },
        "ReadStabilizationManagedPolicyVersions": {
            "Sid": "ReadStabilizationManagedPolicyVersions",
            "Effect": "Allow",
            "Action": [
                "iam:GetPolicy",
                "iam:GetPolicyVersion",
            ],
            "Resource": "arn:aws:iam::*:policy/*",
        },
    }
    if any(
        finite_fence_statements.get(sid) != expected
        for sid, expected in exact_fence_stabilization.items()
    ):
        raise ValueError("fence stabilization IAM authority drifted")
    fence_executor_properties = resources["FenceExecutorRole"]["Properties"]
    fence_executor_policies = fence_executor_properties.get("Policies")
    fence_executor_statements = statements("FenceExecutorRole")
    expected_fence_statement_sids = (
        set(expected_finite_fence)
        | set(exact_fence_stabilization)
        | {
            "ReadExactCampaignBucketPolicy",
            "ReadWriteExactFenceActions",
            "DecryptExactFenceObjectsViaS3Only",
            "PassExactFenceServiceRole",
        }
    )
    if (
        set(fence_executor_properties)
        != {"AssumeRolePolicyDocument", "Policies", "RoleName"}
        or fence_executor_properties.get("AssumeRolePolicyDocument")
        != {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        or type(fence_executor_policies) is not list
        or len(fence_executor_policies) != 1
        or fence_executor_policies[0].get("PolicyName")
        != "FenceExecutorExactTransitionOnly"
        or fence_executor_policies[0].get("PolicyDocument", {}).get("Version")
        != "2012-10-17"
        or len(fence_executor_statements) != len(expected_fence_statement_sids)
        or {statement.get("Sid") for statement in fence_executor_statements}
        != expected_fence_statement_sids
    ):
        raise ValueError("fence stabilization IAM authority drifted")
    if (
        resources["FenceExecutorRole"]["Properties"].get("RoleName")
        != "keep-glm52-h1g-fence-executor"
    ):
        raise ValueError("fence executor role identity drifted")
    expected_lambda_trust = {
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }
    for binding in inputs.task11_writer_bindings:
        role_id = binding["role_id"]
        properties = resources[role_id + "Role"]["Properties"]
        expected_role_name = (
            "keep-glm52-h1g-"
            + re.sub(
                r"([a-z0-9])([A-Z])",
                r"\1-\2",
                role_id,
            ).lower()
        )
        expected_trust = [expected_lambda_trust]
        if properties.get("RoleName") != expected_role_name or properties.get(
            "AssumeRolePolicyDocument"
        ) != {
            "Version": "2012-10-17",
            "Statement": expected_trust,
        }:
            raise ValueError("Task 11 publisher trust identity drifted")
    if any(
        wildcard in str(statements("FenceExecutorRole"))
        for wildcard in (
            "authorities/task11/*",
            "authorities/fence/*",
            "stack/keep-glm52-h1g-fence/*",
        )
    ):
        raise ValueError("fence executor retains prefix authority")

    writer_bindings = {
        binding["role_id"]: binding for binding in inputs.task11_writer_bindings
    }
    writer_outputs = {
        role_id: inputs.model_bucket_arn + "/" + binding["output_key"]
        for role_id, binding in writer_bindings.items()
    }
    for role_id, binding in writer_bindings.items():
        role_statements = {
            statement.get("Sid"): statement
            for statement in statements(role_id + "Role")
        }
        read_arns = [
            inputs.model_bucket_arn + "/" + key for key in binding["read_keys"]
        ]
        ledger = role_statements.get("ReadWriteExactActivationLedger", {})
        read = role_statements.get("ReadOnlyBoundTask11Inputs", {})
        write = role_statements.get("WriteOwnExactOutput", {})
        decrypt = role_statements.get("DecryptRequiredTask11ObjectsViaS3Only", {})
        encrypt = role_statements.get("EncryptOwnExactKeyFamilyViaS3Only", {})
        if (
            ledger.get("Resource") != inputs.ledger_table_arn
            or ledger.get("Condition")
            != {
                "ForAllValues:StringEquals": {
                    "dynamodb:LeadingKeys": (binding["ledger_leading_keys"])
                }
            }
            or read.get("Resource") != read_arns
            or write.get("Resource") != writer_outputs[role_id]
            or decrypt.get("Resource") != inputs.retained_kms_key_arn
            or decrypt.get("Condition", {}).get("StringEquals")
            != {
                "kms:ViaService": f"s3.{REGION}.amazonaws.com",
                "kms:EncryptionContext:aws:s3:arn": read_arns,
            }
            or encrypt.get("Resource") != inputs.retained_kms_key_arn
            or encrypt.get("Condition", {}).get("StringEquals")
            != {
                "kms:ViaService": f"s3.{REGION}.amazonaws.com",
                "kms:EncryptionContext:aws:s3:arn": (writer_outputs[role_id]),
            }
            or "*" in str(statements(role_id + "Role"))
            or any(
                foreign_output in str(statements(role_id + "Role"))
                for foreign_id, foreign_output in writer_outputs.items()
                if foreign_id != role_id
            )
        ):
            raise ValueError("Task 11 writer authority is not disjoint")
        environment = resources[role_id + "Function"]["Properties"]["Environment"][
            "Variables"
        ]
        if any(
            environment.get(variable) != binding[field]
            for variable, field in (
                ("GLM52_TASK11_ACTION_KEY", "action_key"),
                ("GLM52_TASK11_INPUT_KEY", "input_key"),
                ("GLM52_TASK11_OUTPUT_KEY", "output_key"),
            )
        ):
            raise ValueError("Task 11 writer runtime binding drifted")

    resource_capable_h1d = {
        "cloudformation:DescribeStacks",
        "cloudformation:ListStackResources",
        "cloudformation:GetTemplate",
        "lambda:GetFunction",
        "lambda:GetPolicy",
        "iam:GetRole",
        "iam:GetInstanceProfile",
        "iam:GetRolePolicy",
        "iam:ListAttachedRolePolicies",
        "iam:ListInstanceProfilesForRole",
        "iam:ListRolePolicies",
        "states:DescribeStateMachine",
        "states:ListStateMachineVersions",
        "events:ListTargetsByRule",
        "events:DescribeRule",
        "scheduler:GetSchedule",
        "sqs:GetQueueAttributes",
        "sns:ListSubscriptionsByTopic",
    }
    unscopable_h1d = {
        "lambda:ListEventSourceMappings",
        "lambda:ListFunctions",
        "events:ListRules",
        "scheduler:ListSchedules",
        "sqs:ListQueues",
        "iam:ListInstanceProfiles",
        "iam:ListRoles",
        "states:ListStateMachines",
        "ec2:DescribeIamInstanceProfileAssociations",
        "ec2:DescribeInstances",
        "ssm:DescribeInstanceInformation",
        "cloudwatch:DescribeAlarms",
        "cloudwatch:GetMetricData",
        "logs:DescribeLogGroups",
    }
    closure_statements = statements("ClosureSessionRole")
    wildcard_h1d = {
        action
        for statement in closure_statements
        if statement.get("Resource") == "*"
        for action in _statement_actions(statement)
    }
    if wildcard_h1d & resource_capable_h1d:
        raise ValueError("resource-capable H1d read has wildcard authority")
    unscopable_statement = [
        statement
        for statement in closure_statements
        if statement.get("Sid") == "ReadOnlyUnscopableH1dIndexes"
    ]
    if (
        len(unscopable_statement) != 1
        or unscopable_statement[0].get("Resource") != "*"
        or set(_statement_actions(unscopable_statement[0])) != unscopable_h1d
        or any(
            not any(
                action in _statement_actions(statement)
                and statement.get("Resource") != "*"
                for statement in closure_statements
            )
            for action in resource_capable_h1d
        )
    ):
        raise ValueError("H1d read partition is not exact")

    for role_id, sid in (
        ("ClosureSessionRole", "ReadExactFenceStack"),
        (
            "FenceExecutorRole",
            "CreateOnlyAuthenticatedFenceChangeSets",
        ),
    ):
        fence_statements = [
            statement
            for statement in statements(role_id)
            if statement.get("Sid") == sid
        ]
        if (
            len(fence_statements) != 1
            or fence_statements[0].get("Resource") != inputs.fence_stack_id
        ):
            raise ValueError("approved fence stack binding drifted")
    decision_environment = resources["DecisionFunction"]["Properties"]["Environment"][
        "Variables"
    ]
    if decision_environment.get("GLM52_FENCE_STACK_ID") != inputs.fence_stack_id:
        raise ValueError("runtime fence stack binding drifted")
    executor_environment = resources["FenceExecutorFunction"]["Properties"][
        "Environment"
    ]["Variables"]
    expected_executor_environment = _support_fence_executor_environment(inputs)
    if executor_environment != expected_executor_environment:
        raise ValueError("runtime fence request-coordinate binding drifted")
    if b"keep-glm52-gpu-fence" in canonical_json_bytes(template):
        raise ValueError("stale GPU fence name remains executable")
    return True


def validate_support_template(
    template: Mapping[str, object], inputs: object
) -> Mapping[str, object]:
    """Validate exact deterministic support ownership, topology, and shape."""

    if type(inputs) not in {SupportInputs, SupportBuildInputs}:
        raise TypeError("template validation requires exact support inputs")
    validate_support_network_inputs(inputs)
    if (
        type(template) is not dict
        or set(template)
        != {
            "AWSTemplateFormatVersion",
            "Description",
            "Metadata",
            "Resources",
            "Outputs",
        }
        or template.get("AWSTemplateFormatVersion") != "2010-09-09"
        or template.get("Metadata", {}).get("InputIdentitySha256")
        != _support_input_identity(inputs)
        or type(template.get("Outputs")) is not dict
        or len(template["Outputs"]) != 20
    ):
        raise ValueError("support template envelope is not exact")
    counts = support_resource_cardinality(template)
    expected = {
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
        "AWS::S3::BucketPolicy": 1,
        "AWS::StepFunctions::StateMachine": 1,
        "AWS::StepFunctions::StateMachineVersion": 1,
    }
    if any(counts.get(kind) != count for kind, count in expected.items()):
        raise ValueError("support resource cardinality is not exact")
    if {
        "AWS::KMS::Key",
        "AWS::KMS::Alias",
        "AWS::DynamoDB::Table",
    } & set(counts):
        raise ValueError("support template contains retained/fence authority")
    validate_support_security_graph(template, inputs)
    validate_support_lifecycle_resources(template, inputs)
    validate_task11_runtime_role_authority(template, inputs)
    _validate_secret_resource_shapes(template, inputs)
    return template


def validate_support_runtime_inventory(
    template: Mapping[str, object], inputs: object
) -> bool:
    """Validate exact published functions, VPC split, logs, alarms, and DLQ."""

    if type(template) is not dict or type(template.get("Resources")) is not dict:
        raise ValueError("runtime inventory requires an exact support template")
    resources = template["Resources"]
    code = {
        "S3Bucket": inputs.lambda_code_bucket,
        "S3Key": inputs.lambda_code_key,
        "S3ObjectVersion": inputs.lambda_code_version_id,
    }
    task11_writer_bindings = {
        binding["role_id"]: binding for binding in inputs.task11_writer_bindings
    }
    for name, contract in RUNTIME_FUNCTIONS.items():
        function = resources.get(f"{name}Function", {})
        properties = function.get("Properties", {})
        variables: dict[str, object] = {
            "GLM52_ACTIVATION_ID": inputs.activation_id,
            "GLM52_RUN_ID": RUN_ID,
        }
        reader = VPC_SECRET_READERS.get(name)
        if reader is not None:
            variables.update(
                {
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
                    "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_name,
                    "GLM52_COMBINED_HOST_PRIVATE_IP": {
                        "Fn::GetAtt": ["CombinedHost", "PrivateIp"]
                    },
                    "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (
                        inputs.lambda_code_sha256
                    ),
                }
            )
            for logical_id in reader[2]:
                prefix = _SECRET_ENV_PREFIX[logical_id]
                variables[f"GLM52_{prefix}_SECRET_ID"] = {"Ref": logical_id}
                variables[f"GLM52_{prefix}_VERSION_ID"] = {
                    "Fn::GetAtt": [
                        _SECRET_VERSION_SOURCE[logical_id],
                        f"{logical_id}VersionId",
                    ]
                }
                variables[f"GLM52_{prefix}_VERSION_STAGE"] = _secret_stage(
                    inputs, logical_id
                )
        if name == "Attestation":
            variables["GLM52_RELAY_PORT"] = str(inputs.attestation_port)
        if name == "LaunchAdmission":
            variables["GLM52_RELAY_PORT"] = str(inputs.launch_admission_port)
        if name == "NumericBinding":
            variables["GLM52_RELAY_PORT"] = str(inputs.numeric_binding_port)
        if name == "RetainedCancellation":
            variables["GLM52_RELAY_PORT"] = str(inputs.retained_cancellation_port)
        if name == "BudgetGate":
            variables.update(
                {
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_REHEARSAL_BUCKET": {"Ref": "RehearsalBucket"},
                    "GLM52_CLOSURE_GATE_KEY": (
                        "rehearsal/gates/"
                        + inputs.activation_id
                        + "/CLOSURE_BUDGET.json"
                    ),
                    "GLM52_EXPECTED_BUCKET_OWNER": ACCOUNT_ID,
                    "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (
                        inputs.lambda_code_sha256
                    ),
                }
            )
        if name == "RehearsalCollector":
            variables.update(
                {
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
                    "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_name,
                    "GLM52_MODEL_BUCKET": inputs.model_bucket_name,
                    "GLM52_MODEL_PREFIX": inputs.model_prefix,
                    "GLM52_FENCE_STACK_ID": inputs.fence_stack_id,
                    "GLM52_SUPPORT_STACK_ID": {"Ref": "AWS::StackId"},
                    "GLM52_CLOSURE_ROLE_ARN": {
                        "Fn::GetAtt": ["ClosureSessionRole", "Arn"]
                    },
                    "GLM52_ATTESTATION_VERSION_ARN": {"Ref": "AttestationVersion"},
                    "GLM52_LAUNCH_ADMISSION_VERSION_ARN": {
                        "Ref": "LaunchAdmissionVersion"
                    },
                    "GLM52_NUMERIC_BINDING_VERSION_ARN": {
                        "Ref": "NumericBindingVersion"
                    },
                    "GLM52_SOURCE_GPU_SPEND_VERSION_ARN": {
                        "Ref": "SourceGpuSpendVersion"
                    },
                    "GLM52_SOURCE_SUBMISSION_INTENT_VERSION_ARN": {
                        "Ref": "SourceSubmissionIntentVersion"
                    },
                    "GLM52_SOURCE_CONTROLLER_BASELINE_VERSION_ARN": {
                        "Ref": "SourceControllerBaselineVersion"
                    },
                    "GLM52_SOURCE_CONTROL_PLANE_READINESS_VERSION_ARN": {
                        "Ref": "SourceControlPlaneReadinessVersion"
                    },
                    "GLM52_SOURCE_SUBMISSION_ACQUISITION_VERSION_ARN": {
                        "Ref": "SourceSubmissionAcquisitionVersion"
                    },
                    "GLM52_FENCE_EXECUTOR_VERSION_ARN": {"Ref": "FenceExecutorVersion"},
                    "GLM52_FENCE_SUCCESSOR_VERSION_ARN": {
                        "Ref": "FenceSuccessorVersion"
                    },
                    "GLM52_CLAIM_WRITER_VERSION_ARN": {"Ref": "ClaimWriterVersion"},
                    "GLM52_DECISION_WRITER_VERSION_ARN": {
                        "Ref": "DecisionWriterVersion"
                    },
                    "GLM52_TERMINAL_V1_WRITER_VERSION_ARN": {
                        "Ref": "TerminalV1WriterVersion"
                    },
                    "GLM52_CLOSURE_HANDOFF_VERSION_ARN": {
                        "Ref": "ClosureHandoffVersion"
                    },
                    "GLM52_DECISION_VERSION_ARN": {"Ref": "DecisionVersion"},
                    "GLM52_REHEARSAL_CONTROLLER_ROLE_ARN": {
                        "Fn::GetAtt": ["FenceExecutorRole", "Arn"]
                    },
                    "GLM52_REHEARSAL_EXECUTOR_ROLE_ARN": {
                        "Fn::GetAtt": ["RehearsalExecutorRole", "Arn"]
                    },
                    "GLM52_REHEARSAL_BUCKET": {"Ref": "RehearsalBucket"},
                    "GLM52_EXPECTED_BUCKET_OWNER": ACCOUNT_ID,
                    "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (
                        inputs.lambda_code_sha256
                    ),
                    "GLM52_REHEARSAL_PROBE_VERSION_ARN": {
                        "Ref": "RehearsalProbeVersion"
                    },
                }
            )
        if name == "RehearsalProbe":
            variables.update(
                {
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_REHEARSAL_BUCKET": {"Ref": "RehearsalBucket"},
                    "GLM52_COMBINED_HOST_PRIVATE_IP": {
                        "Fn::GetAtt": ["CombinedHost", "PrivateIp"]
                    },
                    "GLM52_RELAY_PORT": str(inputs.attestation_port),
                    "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (
                        inputs.lambda_code_sha256
                    ),
                    "GLM52_ATTESTATION_CLIENT_DER_SHA256": {
                        "Fn::GetAtt": [
                            "TlsBundleCustomResource",
                            "AttestationClientDerSha256",
                        ]
                    },
                    "GLM52_ATTESTATION_SERVER_DER_SHA256": {
                        "Fn::GetAtt": [
                            "TlsBundleCustomResource",
                            "AttestationServerDerSha256",
                        ]
                    },
                    **{
                        f"GLM52_{_SECRET_ENV_PREFIX[logical_id]}_SECRET_ID": {
                            "Ref": logical_id
                        }
                        for logical_id in REHEARSAL_PROBE_SECRET_IDS
                    },
                    **{
                        f"GLM52_{_SECRET_ENV_PREFIX[logical_id]}_VERSION_ID": {
                            "Fn::GetAtt": [
                                _SECRET_VERSION_SOURCE[logical_id],
                                f"{logical_id}VersionId",
                            ]
                        }
                        for logical_id in REHEARSAL_PROBE_SECRET_IDS
                    },
                    **{
                        f"GLM52_{_SECRET_ENV_PREFIX[logical_id]}"
                        "_VERSION_STAGE": _secret_stage(inputs, logical_id)
                        for logical_id in REHEARSAL_PROBE_SECRET_IDS
                    },
                }
            )
        if name == "Decision":
            variables.update(
                {
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
                    "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_name,
                    "GLM52_MODEL_BUCKET": inputs.model_bucket_name,
                    "GLM52_MODEL_PREFIX": inputs.model_prefix,
                    "GLM52_FENCE_STACK_ID": inputs.fence_stack_id,
                    "GLM52_SUPPORT_STACK_ID": {"Ref": "AWS::StackId"},
                    "GLM52_CLOSURE_ROLE_ARN": {
                        "Fn::GetAtt": ["ClosureSessionRole", "Arn"]
                    },
                    "GLM52_ATTESTATION_VERSION_ARN": {"Ref": "AttestationVersion"},
                    "GLM52_LAUNCH_ADMISSION_VERSION_ARN": {
                        "Ref": "LaunchAdmissionVersion"
                    },
                    "GLM52_NUMERIC_BINDING_VERSION_ARN": {
                        "Ref": "NumericBindingVersion"
                    },
                    "GLM52_SOURCE_GPU_SPEND_VERSION_ARN": {
                        "Ref": "SourceGpuSpendVersion"
                    },
                    "GLM52_SOURCE_SUBMISSION_INTENT_VERSION_ARN": {
                        "Ref": "SourceSubmissionIntentVersion"
                    },
                    "GLM52_SOURCE_CONTROLLER_BASELINE_VERSION_ARN": {
                        "Ref": "SourceControllerBaselineVersion"
                    },
                    "GLM52_SOURCE_CONTROL_PLANE_READINESS_VERSION_ARN": {
                        "Ref": "SourceControlPlaneReadinessVersion"
                    },
                    "GLM52_SOURCE_SUBMISSION_ACQUISITION_VERSION_ARN": {
                        "Ref": "SourceSubmissionAcquisitionVersion"
                    },
                    "GLM52_FENCE_EXECUTOR_VERSION_ARN": {"Ref": "FenceExecutorVersion"},
                    "GLM52_FENCE_SUCCESSOR_VERSION_ARN": {
                        "Ref": "FenceSuccessorVersion"
                    },
                    "GLM52_CLAIM_WRITER_VERSION_ARN": {"Ref": "ClaimWriterVersion"},
                    "GLM52_DECISION_WRITER_VERSION_ARN": {
                        "Ref": "DecisionWriterVersion"
                    },
                    "GLM52_TERMINAL_V1_WRITER_VERSION_ARN": {
                        "Ref": "TerminalV1WriterVersion"
                    },
                    "GLM52_CLOSURE_HANDOFF_VERSION_ARN": {
                        "Ref": "ClosureHandoffVersion"
                    },
                    "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (
                        inputs.lambda_code_sha256
                    ),
                }
            )
        if name.startswith("Source"):
            variables.update(
                {
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
                    "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_name,
                    "GLM52_CLOSURE_ROLE_ARN": {
                        "Fn::GetAtt": ["ClosureSessionRole", "Arn"]
                    },
                    "GLM52_PUBLISHER_ROLE_ARN": {"Fn::GetAtt": [f"{name}Role", "Arn"]},
                    "GLM52_SOURCE_KIND": name,
                    "GLM52_TASK11_ACTION_KEY": (
                        task11_writer_bindings[name]["action_key"]
                    ),
                    "GLM52_TASK11_INPUT_KEY": (
                        task11_writer_bindings[name]["input_key"]
                    ),
                    "GLM52_TASK11_OUTPUT_KEY": (
                        task11_writer_bindings[name]["output_key"]
                    ),
                    "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (
                        inputs.lambda_code_sha256
                    ),
                }
            )
        if name in {
            "FenceSuccessor",
            "ClaimWriter",
            "DecisionWriter",
            "TerminalV1Writer",
            "ClosureHandoff",
        }:
            variables.update(
                {
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
                    "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_name,
                    "GLM52_CLOSURE_ROLE_ARN": {
                        "Fn::GetAtt": ["ClosureSessionRole", "Arn"]
                    },
                    "GLM52_WRITER_ROLE_ARN": {"Fn::GetAtt": [f"{name}Role", "Arn"]},
                    "GLM52_EFFECT_WRITER_KIND": name,
                    "GLM52_TASK11_ACTION_KEY": (
                        task11_writer_bindings[name]["action_key"]
                    ),
                    "GLM52_TASK11_INPUT_KEY": (
                        task11_writer_bindings[name]["input_key"]
                    ),
                    "GLM52_TASK11_OUTPUT_KEY": (
                        task11_writer_bindings[name]["output_key"]
                    ),
                    "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": (
                        inputs.lambda_code_sha256
                    ),
                }
            )
        if name == "FenceExecutor":
            variables.clear()
            variables.update(_support_fence_executor_environment(inputs))
        if name == "SupportDeadline":
            variables.update(
                {
                    "GLM52_SUPPORT_STACK_ID": {"Ref": "AWS::StackId"},
                    "GLM52_SUPPORT_HOST_INSTANCE_ID": {"Ref": "CombinedHost"},
                    "GLM52_COMBINED_HOST_INSTANCE_ID": {"Ref": "CombinedHost"},
                    "GLM52_SUPPORT_HOST_SECURITY_GROUP_ID": {
                        "Ref": "CombinedHostSecurityGroup"
                    },
                    "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
                    "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                    "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_name,
                    "GLM52_EXPECTED_BUCKET_OWNER": ACCOUNT_ID,
                    "GLM52_NUMERIC_BINDING_VERSION_ARN": {
                        "Ref": "NumericBindingVersion"
                    },
                    "GLM52_RETAINED_CANCELLATION_VERSION_ARN": {
                        "Ref": "RetainedCancellationVersion"
                    },
                    "GLM52_WORKER_DRAIN_VERSION_ARN": {
                        "Fn::ImportValue": ("KeepGlm52Task12WorkerDrainVersionArn")
                    },
                    "GLM52_TASK9_LIABILITY_WATCHER_VERSION_ARN": {
                        "Fn::ImportValue": ("KeepGlm52Task9LiabilityWatcherVersionArn")
                    },
                    "GLM52_TERMINAL_V2_VERSION_ARN": {
                        "Fn::ImportValue": ("KeepGlm52Task12TerminalV2VersionArn")
                    },
                    "GLM52_SUPPORT_QUIESCENCE_SECONDS": "60",
                    "GLM52_OPERATOR_TOPIC_ARN": SUPPORT_OPERATOR_TOPIC_ARN,
                    "GLM52_WORK_STOP_HOURS": "68",
                    "GLM52_DELETE_REQUEST_HOURS": "71",
                    "GLM52_ABSENCE_EXPECTED_HOURS": "72",
                    "GLM52_HOST_EGRESS_WARNING_BYTES": str(5 * 1024**3),
                    "GLM52_HOST_EGRESS_DRAIN_BYTES": str(6 * 1024**3),
                    "GLM52_NAT_AUTHORITY_BYTES": str(10 * 1024**3),
                    "GLM52_NAT_GATEWAY_ID": {"Ref": "NatGateway"},
                    "GLM52_DIRECT_CHILD_DELETE_ACTIONS": "FORBIDDEN",
                    "GLM52_POST_72H_RECONCILIATIONS": "1",
                    "GLM52_DELETE_STACK_OWNER": ("RETAINED_LIFECYCLE_ONLY"),
                }
            )
        expected_keys = {
            "Code",
            "Environment",
            "FunctionName",
            "Handler",
            "MemorySize",
            "ReservedConcurrentExecutions",
            "Role",
            "Runtime",
            "Timeout",
        }
        if contract["vpc_attached"]:
            expected_keys.add("VpcConfig")
        expected_depends = (
            ["SkyBootstrapCustomResource", "TlsBundleCustomResource"]
            if contract["vpc_attached"]
            else None
        )
        if (
            function.get("Type") != "AWS::Lambda::Function"
            or set(properties) != expected_keys
            or properties.get("Code") != code
            or properties.get("Environment") != {"Variables": variables}
            or properties.get("FunctionName")
            != (
                "keep-glm52-h1g-"
                + re.sub(
                    r"([a-z0-9])([A-Z])",
                    r"\1-\2",
                    name,
                ).lower()
            )
            or properties.get("Handler") != contract["handler"]
            or properties.get("MemorySize") != contract["memory"]
            or properties.get("ReservedConcurrentExecutions")
            != (5 if name in {"RehearsalCollector", "RehearsalProbe"} else 1)
            or properties.get("Role") != {"Fn::GetAtt": [contract["role"], "Arn"]}
            or properties.get("Runtime") != "python3.12"
            or properties.get("Timeout") != contract["timeout"]
            or function.get("DependsOn") != expected_depends
            and (expected_depends is not None or "DependsOn" in function)
        ):
            raise ValueError(f"support runtime function changed: {name}")
        if contract["vpc_attached"]:
            if properties.get("VpcConfig") != {
                "SecurityGroupIds": [{"Ref": contract["client_security_group"]}],
                "SubnetIds": [
                    {"Ref": "PrimaryIsolatedSubnet"},
                    {"Ref": "AlternateIsolatedSubnet"},
                ],
            }:
                raise ValueError("sensitive client function VPC path changed")
        elif "VpcConfig" in properties:
            raise ValueError("non-VPC function acquired a support network path")
        if resources.get(f"{name}Version") != {
            "Type": "AWS::Lambda::Version",
            "Properties": {
                "Description": f"code={inputs.lambda_code_sha256}",
                "FunctionName": {"Ref": f"{name}Function"},
            },
        }:
            raise ValueError(f"support runtime version changed: {name}")
        permission = resources.get(f"{name}ClosureInvokePermission")
        expected_permission = (
            {
                "Type": "AWS::Lambda::Permission",
                "Properties": {
                    "Action": "lambda:InvokeFunction",
                    "FunctionName": {"Ref": f"{name}Version"},
                    "Principal": {"Fn::GetAtt": ["ClosureSessionRole", "Arn"]},
                    "SourceAccount": ACCOUNT_ID,
                },
            }
            if name in _TASK11_CLOSURE_CALLEES
            else None
        )
        if permission != expected_permission:
            raise ValueError(f"support runtime resource policy changed: {name}")
        if resources.get(f"{name}LogGroup") != {
            "Type": "AWS::Logs::LogGroup",
            "Properties": {
                "LogGroupName": {"Fn::Sub": f"/aws/lambda/${{{name}Function}}"},
                "RetentionInDays": 14,
            },
        }:
            raise ValueError(f"support runtime log group changed: {name}")
        alarm = resources.get(f"{name}ErrorAlarm", {})
        alarm_properties = alarm.get("Properties", {})
        if (
            alarm.get("Type") != "AWS::CloudWatch::Alarm"
            or alarm_properties.get("Dimensions")
            != [
                {
                    "Name": "FunctionName",
                    "Value": {"Ref": f"{name}Function"},
                }
            ]
            or alarm_properties.get("MetricName") != "Errors"
            or alarm_properties.get("Namespace") != "AWS/Lambda"
            or alarm_properties.get("Threshold") != 0
            or alarm_properties.get("TreatMissingData") != "breaching"
        ):
            raise ValueError(f"support runtime alarm changed: {name}")
    if resources.get("RehearsalCollectorExecutorInvokePermission") != {
        "Type": "AWS::Lambda::Permission",
        "Properties": {
            "Action": "lambda:InvokeFunction",
            "FunctionName": {"Ref": "RehearsalCollectorVersion"},
            "Principal": {"Fn::GetAtt": ["RehearsalExecutorRole", "Arn"]},
            "SourceAccount": ACCOUNT_ID,
        },
    } or resources.get("RehearsalProbeCollectorInvokePermission") != {
        "Type": "AWS::Lambda::Permission",
        "Properties": {
            "Action": "lambda:InvokeFunction",
            "FunctionName": {"Ref": "RehearsalProbeVersion"},
            "Principal": {"Fn::GetAtt": ["RehearsalCollectorRole", "Arn"]},
            "SourceAccount": ACCOUNT_ID,
        },
    }:
        raise ValueError("rehearsal runtime resource policy changed")
    tls = resources.get("TlsHandlerFunction", {})
    tls_properties = tls.get("Properties", {})
    tls_variables = {
        "GLM52_ACTIVATION_ID": inputs.activation_id,
        "GLM52_COMBINED_HOST_PRIVATE_IP": inputs.host_private_ip,
        "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
        "GLM52_RETAINED_KMS_KEY_ARN": inputs.retained_kms_key_arn,
        **{
            f"GLM52_{logical_id.upper()}_ID": {"Ref": logical_id}
            for logical_id in SECRET_LOGICAL_IDS
        },
        **{
            f"GLM52_{logical_id.upper()}_STAGE": _secret_stage(inputs, logical_id)
            for logical_id in SECRET_LOGICAL_IDS
        },
    }
    if (
        tls.get("Type") != "AWS::Lambda::Function"
        or set(tls_properties)
        != {
            "Code",
            "Environment",
            "Handler",
            "Layers",
            "MemorySize",
            "ReservedConcurrentExecutions",
            "Role",
            "Runtime",
            "Timeout",
        }
        or tls_properties.get("Code") != code
        or tls_properties.get("Environment") != {"Variables": tls_variables}
        or tls_properties.get("Handler") != "support_custom_resource_handler.main"
        or tls_properties.get("Layers") != [inputs.cryptography_layer_arn]
        or tls_properties.get("MemorySize") != 256
        or tls_properties.get("ReservedConcurrentExecutions") != 1
        or tls_properties.get("Role") != {"Fn::GetAtt": ["TlsHandlerRole", "Arn"]}
        or tls_properties.get("Runtime") != "python3.12"
        or tls_properties.get("Timeout") != 840
        or "VpcConfig" in tls_properties
    ):
        raise ValueError("custom-resource callback handler must remain non-VPC")
    if resources.get("TlsHandlerVersion") != {
        "Type": "AWS::Lambda::Version",
        "Properties": {
            "Description": (
                f"code={inputs.lambda_code_sha256};"
                f"cryptography={inputs.cryptography_layer_sha256}"
            ),
            "FunctionName": {"Ref": "TlsHandlerFunction"},
        },
    }:
        raise ValueError("custom-resource version is not exact")
    tls_log = resources.get("TlsHandlerLogGroup", {})
    tls_alarm = resources.get("TlsHandlerErrorAlarm", {})
    if (
        tls_log.get("Type") != "AWS::Logs::LogGroup"
        or tls_log.get("Properties")
        != {
            "LogGroupName": {"Fn::Sub": "/aws/lambda/${TlsHandlerFunction}"},
            "RetentionInDays": 14,
        }
        or tls_alarm.get("Type") != "AWS::CloudWatch::Alarm"
        or tls_alarm.get("Properties", {}).get("Dimensions")
        != [
            {
                "Name": "FunctionName",
                "Value": {"Ref": "TlsHandlerFunction"},
            }
        ]
        or tls_alarm.get("Properties", {}).get("TreatMissingData") != "breaching"
    ):
        raise ValueError("custom-resource runtime evidence is not exact")
    if resources.get("SupportAsyncDlq") != {
        "Type": "AWS::SQS::Queue",
        "Properties": {
            "KmsMasterKeyId": inputs.retained_kms_key_arn,
            "MessageRetentionPeriod": 1209600,
            "QueueName": (f"keep-glm52-h1g-support-dlq-{inputs.activation_id}"),
        },
    } or resources.get("SupportDeadlineEventInvokeConfig") != {
        "Type": "AWS::Lambda::EventInvokeConfig",
        "Properties": {
            "DestinationConfig": {
                "OnFailure": {"Destination": {"Fn::GetAtt": ["SupportAsyncDlq", "Arn"]}}
            },
            "FunctionName": {"Ref": "SupportDeadlineFunction"},
            "MaximumEventAgeInSeconds": 60,
            "MaximumRetryAttempts": 0,
            "Qualifier": {"Fn::GetAtt": ["SupportDeadlineVersion", "Version"]},
        },
    }:
        raise ValueError("support async failure path is not exact")
    workflow_role = resources.get("SupportWorkflowRole", {}).get("Properties", {})
    if (
        workflow_role.get("AssumeRolePolicyDocument", {})
        .get("Statement", [{}])[0]
        .get("Principal")
        != {"Service": "states.amazonaws.com"}
        or workflow_role.get("Policies", [{}])[0]
        .get("PolicyDocument", {})
        .get("Statement", [{}])[0]
        .get("Resource")
        != [
            {"Ref": "BudgetGateVersion"},
            {"Ref": "DecisionVersion"},
            {"Ref": "SupportDeadlineVersion"},
        ]
        or resources.get("SupportStateMachine", {})
        .get("Properties", {})
        .get("Definition")
        != build_task11_workflow_definition()
        or resources.get("SupportStateMachineVersion")
        != {
            "Type": "AWS::StepFunctions::StateMachineVersion",
            "Properties": {
                "Description": ("Exact Task11 private decision-to-POST version"),
                "StateMachineArn": {"Ref": "SupportStateMachine"},
            },
        }
    ):
        raise ValueError("support workflow version or role is not exact")
    return True


def _utc_instant(value: object, label: str) -> datetime:
    text = _parse_utc(value, label)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed.astimezone(UTC)


def evaluate_support_lifecycle(
    *,
    activation_started_at: str,
    observed_at: str,
    stack_exists: bool,
) -> SupportLifecycleDecision:
    """Evaluate the monotonic 68h stop, 71h delete, and 72h incident envelope."""

    if type(stack_exists) is not bool:
        raise ValueError("stack existence must be one exact observation")
    start = _utc_instant(activation_started_at, "activation start")
    observed = _utc_instant(observed_at, "lifecycle observation")
    elapsed = int((observed - start).total_seconds())
    if elapsed < 0:
        raise ValueError("lifecycle observation predates activation")
    if not stack_exists:
        return SupportLifecycleDecision(
            state=SupportLifecycleState.ABSENT,
            elapsed_seconds=elapsed,
            host_stopped=True,
            egress_disabled=True,
            delete_stack_requested=False,
            page_operator=False,
            incident=None,
            read_only_stack_reconciliations=0,
            direct_child_deletions=(),
        )
    if elapsed >= 72 * 3600:
        return SupportLifecycleDecision(
            state=SupportLifecycleState.DELETE_DEADLINE_MISSED,
            elapsed_seconds=elapsed,
            host_stopped=True,
            egress_disabled=True,
            delete_stack_requested=True,
            page_operator=True,
            incident="SUPPORT_DELETE_DEADLINE_MISSED",
            read_only_stack_reconciliations=1,
            direct_child_deletions=(),
        )
    if elapsed >= 71 * 3600:
        return SupportLifecycleDecision(
            state=SupportLifecycleState.DELETE_REQUESTED,
            elapsed_seconds=elapsed,
            host_stopped=True,
            egress_disabled=True,
            delete_stack_requested=True,
            page_operator=False,
            incident=None,
            read_only_stack_reconciliations=0,
            direct_child_deletions=(),
        )
    if elapsed >= 68 * 3600:
        return SupportLifecycleDecision(
            state=SupportLifecycleState.WORK_STOPPED,
            elapsed_seconds=elapsed,
            host_stopped=True,
            egress_disabled=True,
            delete_stack_requested=False,
            page_operator=False,
            incident=None,
            read_only_stack_reconciliations=0,
            direct_child_deletions=(),
        )
    return SupportLifecycleDecision(
        state=SupportLifecycleState.ACTIVE,
        elapsed_seconds=elapsed,
        host_stopped=False,
        egress_disabled=False,
        delete_stack_requested=False,
        page_operator=False,
        incident=None,
        read_only_stack_reconciliations=0,
        direct_child_deletions=(),
    )


def evaluate_retained_delete_protocol(
    *,
    inputs: SupportInputs,
    finalization_state: str,
    stack_exists: bool,
    termination_protection: bool | None,
    update_submissions: int,
    delete_submissions: int,
    mutation_outcome: str,
) -> RetainedDeleteDecision:
    """Select at most one protected-stack mutation between exact readbacks."""

    if (
        type(inputs) is not SupportInputs
        or finalization_state not in {"SUPPORT_FINALIZING", "SUPPORT_FINALIZED"}
        or type(stack_exists) is not bool
        or (
            termination_protection is not None
            and type(termination_protection) is not bool
        )
        or type(update_submissions) is not int
        or update_submissions not in {0, 1}
        or type(delete_submissions) is not int
        or delete_submissions not in {0, 1}
        or delete_submissions > update_submissions
        or mutation_outcome not in {"NONE", "SUCCESS", "AMBIGUOUS"}
    ):
        raise ValueError("retained delete protocol observation is not exact")
    if not stack_exists:
        return RetainedDeleteDecision(
            mutation=None,
            mutation_parameters=None,
            reads=(),
            complete=True,
        )
    if finalization_state != "SUPPORT_FINALIZED":
        return RetainedDeleteDecision(
            mutation=None,
            mutation_parameters=None,
            reads=("GetItem",),
            complete=False,
        )
    if mutation_outcome == "AMBIGUOUS":
        return RetainedDeleteDecision(
            mutation=None,
            mutation_parameters=None,
            reads=("DescribeStacks",),
            complete=False,
        )
    if termination_protection is True and update_submissions == 0:
        return RetainedDeleteDecision(
            mutation="UpdateTerminationProtection",
            mutation_parameters={
                "StackName": inputs.support_stack_id,
                "EnableTerminationProtection": False,
            },
            reads=(),
            complete=False,
        )
    if termination_protection is not False:
        return RetainedDeleteDecision(
            mutation=None,
            mutation_parameters=None,
            reads=("DescribeStacks",),
            complete=False,
        )
    if update_submissions != 1:
        return RetainedDeleteDecision(
            mutation=None,
            mutation_parameters=None,
            reads=("DescribeStacks",),
            complete=False,
        )
    if delete_submissions == 0:
        return RetainedDeleteDecision(
            mutation="DeleteStack",
            mutation_parameters={
                "StackName": inputs.support_stack_id,
                "RoleARN": SUPPORT_DELETION_ROLE_ARN,
            },
            reads=(),
            complete=False,
        )
    return RetainedDeleteDecision(
        mutation=None,
        mutation_parameters=None,
        reads=("DescribeStacks",),
        complete=False,
    )


_BOOTSTRAP_GRACE_SECONDS = 15 * 60
_NAT_ACCUMULATOR_MAX_WINDOWS = 72 * 60


def bootstrap_grace_contract(inputs: SupportInputs) -> Mapping[str, object]:
    """Derive one immutable grace window from the authenticated activation."""

    if type(inputs) is not SupportInputs:
        raise ValueError("bootstrap grace requires exact support inputs")
    started = datetime.fromisoformat(
        inputs.activation_started_at.replace("Z", "+00:00")
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_bootstrap_grace_v1",
        "activation_id": inputs.activation_id,
        "support_inputs_identity_sha256": support_inputs_identity(inputs),
        "grace_started_at": inputs.activation_started_at,
        "grace_deadline": (
            started + timedelta(seconds=_BOOTSTRAP_GRACE_SECONDS)
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "grace_duration_seconds": _BOOTSTRAP_GRACE_SECONDS,
        "work_authorization": "METRICS_OK_ONLY",
        "missing_metric_after_grace": "DRAIN",
    }
    body["canonical_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    return body


def validate_bootstrap_grace_contract(value: object, inputs: SupportInputs) -> bool:
    """Reject a replayed, caller-extended, or otherwise foreign grace window."""

    if type(value) is not dict or set(value) != {
        "schema_version",
        "record_type",
        "activation_id",
        "support_inputs_identity_sha256",
        "grace_started_at",
        "grace_deadline",
        "grace_duration_seconds",
        "work_authorization",
        "missing_metric_after_grace",
        "canonical_body_sha256",
    }:
        raise ValueError("bootstrap grace schema is not exact")
    body = dict(value)
    identity = body.pop("canonical_body_sha256")
    started = datetime.fromisoformat(
        inputs.activation_started_at.replace("Z", "+00:00")
    )
    deadline = (started + timedelta(seconds=_BOOTSTRAP_GRACE_SECONDS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    if (
        type(identity) is not str
        or _SHA256.fullmatch(identity) is None
        or hashlib.sha256(canonical_json_bytes(body)).hexdigest() != identity
        or body
        != {
            "schema_version": 1,
            "record_type": "glm52_h1g_bootstrap_grace_v1",
            "activation_id": inputs.activation_id,
            "support_inputs_identity_sha256": support_inputs_identity(inputs),
            "grace_started_at": inputs.activation_started_at,
            "grace_deadline": deadline,
            "grace_duration_seconds": _BOOTSTRAP_GRACE_SECONDS,
            "work_authorization": "METRICS_OK_ONLY",
            "missing_metric_after_grace": "DRAIN",
        }
    ):
        raise ValueError("bootstrap grace identity or bounds are foreign")
    return True


def nat_metric_data_query_contract(
    inputs: SupportInputs,
) -> Mapping[str, object]:
    """Build the exact four-query AWS/NATGateway GetMetricData projection."""

    if type(inputs) is not SupportInputs:
        raise ValueError("NAT metric query contract requires exact inputs")
    metric_names = (
        "BytesInFromSource",
        "BytesOutToDestination",
        "BytesInFromDestination",
        "BytesOutToSource",
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_nat_metric_query_contract_v1",
        "activation_id": inputs.activation_id,
        "nat_gateway_id": inputs.nat_gateway_id,
        "period_seconds": 60,
        "queries": [
            {
                "Id": f"nat_counter_{index}",
                "MetricStat": {
                    "Metric": {
                        "Dimensions": [
                            {
                                "Name": "NatGatewayId",
                                "Value": inputs.nat_gateway_id,
                            }
                        ],
                        "MetricName": metric_name,
                        "Namespace": "AWS/NATGateway",
                    },
                    "Period": 60,
                    "Stat": "Sum",
                },
                "ReturnData": True,
            }
            for index, metric_name in enumerate(metric_names, start=1)
        ],
    }
    body["canonical_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    return body


def support_counter_contract(inputs: SupportInputs) -> Mapping[str, object]:
    """Bind NAT accounting to its dedicated Task 7 counter store."""

    if type(inputs) is not SupportInputs:
        raise ValueError("support counter contract requires exact inputs")
    metric_query = nat_metric_data_query_contract(inputs)
    table_name = f"keep-glm52-h1g-task7-counter-{inputs.activation_id}"
    counter_id = f"ACTIVATION#{inputs.activation_id}#NAT_PROCESSED_BYTES"
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_task7_support_counter_contract_v1",
        "activation_id": inputs.activation_id,
        "nat_gateway_id": inputs.nat_gateway_id,
        "support_inputs_identity_sha256": support_inputs_identity(inputs),
        "counter_table_name": table_name,
        "counter_table_arn": (
            f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/{table_name}"
        ),
        "counter_partition_key": "CounterId",
        "counter_id": counter_id,
        "counter_owner": "TASK7_NAT_ACCUMULATOR_ONLY",
        "frozen_ledger_access": "FORBIDDEN",
        "nat_metric_query_contract_sha256": metric_query["canonical_body_sha256"],
        "nat_counter_metrics": [
            "BytesInFromSource",
            "BytesOutToDestination",
            "BytesInFromDestination",
            "BytesOutToSource",
        ],
        "max_windows": _NAT_ACCUMULATOR_MAX_WINDOWS,
    }
    body["canonical_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    return body


def initial_nat_accumulator_state(
    inputs: SupportInputs,
) -> NatAccumulatorState:
    """Return the empty activation-bound dedicated counter state."""

    if type(inputs) is not SupportInputs:
        raise ValueError("NAT accumulator requires exact support inputs")
    counter_contract = support_counter_contract(inputs)
    return NatAccumulatorState(
        activation_id=inputs.activation_id,
        nat_gateway_id=inputs.nat_gateway_id,
        counter_table_name=str(counter_contract["counter_table_name"]),
        counter_partition_key=str(counter_contract["counter_partition_key"]),
        counter_id=str(counter_contract["counter_id"]),
        support_counter_contract_sha256=str(counter_contract["canonical_body_sha256"]),
        last_window_end=inputs.activation_started_at,
        cumulative_nat_processed_bytes=0,
        accepted_windows=0,
        last_observation_sha256=None,
    )


def accumulate_nat_window(
    *,
    inputs: SupportInputs,
    state: NatAccumulatorState,
    window_started_at: str,
    window_ended_at: str,
    nat_gateway_id: str,
    nat_counter_bytes: tuple[int, int, int, int],
) -> NatAccumulatorState:
    """Accumulate one consecutive 60-second four-counter NAT observation."""

    counter_contract = (
        support_counter_contract(inputs) if type(inputs) is SupportInputs else None
    )
    if (
        type(inputs) is not SupportInputs
        or type(state) is not NatAccumulatorState
        or state.activation_id != inputs.activation_id
        or nat_gateway_id != inputs.nat_gateway_id
        or state.nat_gateway_id != inputs.nat_gateway_id
        or state.counter_table_name
        != f"keep-glm52-h1g-task7-counter-{inputs.activation_id}"
        or state.counter_partition_key != "CounterId"
        or state.counter_id != f"ACTIVATION#{inputs.activation_id}#NAT_PROCESSED_BYTES"
        or counter_contract is None
        or state.support_counter_contract_sha256
        != counter_contract["canonical_body_sha256"]
        or type(state.cumulative_nat_processed_bytes) is not int
        or state.cumulative_nat_processed_bytes < 0
        or type(state.accepted_windows) is not int
        or not 0 <= state.accepted_windows <= _NAT_ACCUMULATOR_MAX_WINDOWS
        or (
            state.last_observation_sha256 is not None
            and (
                type(state.last_observation_sha256) is not str
                or _SHA256.fullmatch(state.last_observation_sha256) is None
            )
        )
        or type(nat_counter_bytes) is not tuple
        or len(nat_counter_bytes) != 4
        or any(type(item) is not int or item < 0 for item in nat_counter_bytes)
    ):
        raise ValueError("NAT accumulator state or observation is not exact")
    start_text = _parse_utc(window_started_at, "NAT window start")
    end_text = _parse_utc(window_ended_at, "NAT window end")
    activation = datetime.fromisoformat(
        inputs.activation_started_at.replace("Z", "+00:00")
    )
    expected_state_end = activation + timedelta(seconds=60 * state.accepted_windows)
    state_end = datetime.fromisoformat(
        _parse_utc(state.last_window_end, "NAT accumulator end").replace("Z", "+00:00")
    )
    if (
        state_end != expected_state_end
        or (
            state.accepted_windows == 0
            and (
                state.cumulative_nat_processed_bytes != 0
                or state.last_observation_sha256 is not None
            )
        )
        or (state.accepted_windows > 0 and state.last_observation_sha256 is None)
    ):
        raise ValueError("NAT accumulator state continuity is foreign")
    start = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_text.replace("Z", "+00:00"))
    if end - start != timedelta(seconds=60):
        raise ValueError("NAT accumulator windows must be exactly 60 seconds")
    observation = {
        "activation_id": inputs.activation_id,
        "nat_gateway_id": inputs.nat_gateway_id,
        "support_counter_contract_sha256": (state.support_counter_contract_sha256),
        "window_started_at": start_text,
        "window_ended_at": end_text,
        "nat_counter_bytes": list(nat_counter_bytes),
    }
    identity = hashlib.sha256(canonical_json_bytes(observation)).hexdigest()
    if identity == state.last_observation_sha256:
        if end != state_end:
            raise ValueError("NAT accumulator replay does not match state")
        return state
    if start != state_end or state.accepted_windows >= _NAT_ACCUMULATOR_MAX_WINDOWS:
        raise ValueError("NAT accumulator window is gapped or out of bounds")
    return NatAccumulatorState(
        activation_id=inputs.activation_id,
        nat_gateway_id=inputs.nat_gateway_id,
        counter_table_name=state.counter_table_name,
        counter_partition_key=state.counter_partition_key,
        counter_id=state.counter_id,
        support_counter_contract_sha256=(state.support_counter_contract_sha256),
        last_window_end=end_text,
        cumulative_nat_processed_bytes=(
            state.cumulative_nat_processed_bytes + sum(nat_counter_bytes)
        ),
        accepted_windows=state.accepted_windows + 1,
        last_observation_sha256=identity,
    )


def evaluate_support_egress(
    *,
    inputs: SupportInputs,
    observed_at: str,
    host_non_endpoint_bytes: int,
    nat_cumulative_bytes: int,
    nat_gateway_id: str,
    metric_state: str,
) -> SupportEgressDecision:
    """Evaluate activation-cumulative egress with a derived bootstrap grace."""

    if (
        type(inputs) is not SupportInputs
        or type(host_non_endpoint_bytes) is not int
        or host_non_endpoint_bytes < 0
        or type(nat_cumulative_bytes) is not int
        or nat_cumulative_bytes < 0
        or nat_gateway_id != inputs.nat_gateway_id
        or metric_state not in {"OK", "MISSING", "DELAYED", "INSUFFICIENT_DATA"}
    ):
        raise ValueError("support egress observation is not exact")
    observed_text = _parse_utc(observed_at, "support egress observation")
    observed = datetime.fromisoformat(observed_text.replace("Z", "+00:00"))
    started = datetime.fromisoformat(
        inputs.activation_started_at.replace("Z", "+00:00")
    )
    grace = bootstrap_grace_contract(inputs)
    deadline_text = grace["grace_deadline"]
    deadline = datetime.fromisoformat(str(deadline_text).replace("Z", "+00:00"))
    if observed < started:
        raise ValueError("support egress observation predates activation")
    grace_complete = observed >= deadline
    gib = 1024**3
    bad_metrics = metric_state != "OK"
    warning = host_non_endpoint_bytes >= 5 * gib or nat_cumulative_bytes >= 5 * gib
    drain = (
        host_non_endpoint_bytes >= 6 * gib
        or nat_cumulative_bytes >= 6 * gib
        or (bad_metrics and grace_complete)
    )
    authority_exceeded = nat_cumulative_bytes > 10 * gib
    work_authorized = not bad_metrics and not drain
    if authority_exceeded:
        reason = "NAT_AUTHORITY_EXCEEDED"
    elif bad_metrics and grace_complete:
        reason = f"NAT_METRIC_{metric_state}"
    elif bad_metrics:
        reason = "BOOTSTRAP_METRICS_PENDING"
    elif drain:
        reason = "EGRESS_DRAIN_THRESHOLD"
    elif warning:
        reason = "EGRESS_WARNING_THRESHOLD"
    else:
        reason = "WITHIN_ENVELOPE"
    return SupportEgressDecision(
        host_non_endpoint_bytes=host_non_endpoint_bytes,
        nat_processed_bytes=nat_cumulative_bytes,
        warning=warning,
        drain=drain,
        egress_disabled=drain,
        authority_exceeded=authority_exceeded,
        work_authorized=work_authorized,
        bootstrap_grace_complete=grace_complete,
        bootstrap_grace_deadline=str(deadline_text),
        reason=reason,
    )


def _nat_accumulator_schedule_name(inputs: SupportInputs) -> str:
    return f"keep-glm52-h1g-{inputs.activation_id}-nat-accumulator"


def _nat_accumulator_schedule_end_date(inputs: SupportInputs) -> str:
    started = datetime.fromisoformat(
        inputs.activation_started_at.replace("Z", "+00:00")
    )
    return (started + timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _nat_accumulator_schedule_arn(inputs: SupportInputs) -> str:
    return (
        f"arn:aws:scheduler:{REGION}:{ACCOUNT_ID}:schedule/default/"
        f"{_nat_accumulator_schedule_name(inputs)}"
    )


def evaluate_nat_accumulator_schedule_retirement(
    *,
    inputs: SupportInputs,
    observed_at: str,
    schedule_exists: bool,
    schedule_state: str | None,
    disable_submissions: int,
    delete_submissions: int,
) -> NatScheduleRetirementDecision:
    """Evaluate the bounded disable/read/delete/read schedule sequence."""

    if (
        type(inputs) is not SupportInputs
        or type(schedule_exists) is not bool
        or schedule_state not in {None, "ENABLED", "DISABLED"}
        or type(disable_submissions) is not int
        or disable_submissions not in {0, 1}
        or type(delete_submissions) is not int
        or delete_submissions not in {0, 1}
        or delete_submissions > disable_submissions
    ):
        raise ValueError("NAT schedule retirement state is not exact")
    observed_text = _parse_utc(observed_at, "NAT schedule retirement observation")
    observed = datetime.fromisoformat(observed_text.replace("Z", "+00:00"))
    end_date = _nat_accumulator_schedule_end_date(inputs)
    deadline = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    if observed < deadline:
        return NatScheduleRetirementDecision(
            mutation=None,
            mutation_parameters=None,
            reads=(),
            complete=False,
        )
    if not schedule_exists:
        if schedule_state is not None:
            raise ValueError("absent NAT schedule has a state")
        return NatScheduleRetirementDecision(
            mutation=None,
            mutation_parameters=None,
            reads=(),
            complete=True,
        )
    if schedule_state is None:
        return NatScheduleRetirementDecision(
            mutation=None,
            mutation_parameters=None,
            reads=("GetSchedule",),
            complete=False,
        )
    if schedule_state == "ENABLED":
        if disable_submissions == 0:
            return NatScheduleRetirementDecision(
                mutation="UpdateSchedule",
                mutation_parameters={
                    "Name": _nat_accumulator_schedule_name(inputs),
                    "GroupName": "default",
                    "State": "DISABLED",
                    "ScheduleExpression": "rate(1 minute)",
                    "EndDate": end_date,
                    "FlexibleTimeWindow": {"Mode": "OFF"},
                    "BoundTarget": {
                        "activation_id": inputs.activation_id,
                        "nat_gateway_id": inputs.nat_gateway_id,
                    },
                },
                reads=(),
                complete=False,
            )
        return NatScheduleRetirementDecision(
            mutation=None,
            mutation_parameters=None,
            reads=("GetSchedule",),
            complete=False,
        )
    if disable_submissions != 1:
        return NatScheduleRetirementDecision(
            mutation=None,
            mutation_parameters=None,
            reads=("GetSchedule",),
            complete=False,
        )
    if delete_submissions == 0:
        return NatScheduleRetirementDecision(
            mutation="DeleteSchedule",
            mutation_parameters={
                "Name": _nat_accumulator_schedule_name(inputs),
                "GroupName": "default",
            },
            reads=(),
            complete=False,
        )
    return NatScheduleRetirementDecision(
        mutation=None,
        mutation_parameters=None,
        reads=("GetSchedule",),
        complete=False,
    )


SUPPORT_USAGE_CEILINGS: Mapping[str, int] = {
    "support_lambda_invocations": 10000,
    "lambda_reserved_concurrency_per_function": 1,
    "support_execution_count": 1,
    "workflow_history_events": 12000,
    "retained_finalization_executions": 12,
    "active_finalization_owners": 1,
    "runtime_observations": 432,
    "transient_log_retention_days": 14,
    "application_log_ingestion_bytes": 512 * 1024 * 1024,
    "liability_scan_interval_seconds": 60,
    "liability_nominal_scans": 43200,
    "liability_transitions_per_no_match_scan": 4,
    "liability_reads_per_no_match_scan": 3,
    "liability_lambda_memory_mib": 256,
    "liability_lambda_timeout_seconds": 20,
    "liability_event_accelerators": 1000,
    "liability_log_bytes": 1024**3,
    "custom_resource_creates": 2,
    "custom_resource_identity_updates": 2,
    "custom_resource_deletes": 2,
    "retained_sky_cancel_actions": 8,
    "snapshot_schedule_count": 1,
    "snapshot_delete_calls": 12,
    "forensic_snapshot_retention_days": 7,
    "support_host_count": 1,
    "support_nat_count": 1,
    "support_eip_count": 1,
    "support_data_volume_count": 1,
    "support_interface_endpoint_az_set_count": 1,
    "support_workflow_count": 1,
    "nat_processed_bytes": 10 * 1024**3,
}
_EXACT_USAGE_FIELDS = {
    "lambda_reserved_concurrency_per_function",
    "transient_log_retention_days",
    "liability_scan_interval_seconds",
    "liability_lambda_memory_mib",
    "liability_lambda_timeout_seconds",
    "snapshot_schedule_count",
    "forensic_snapshot_retention_days",
    "support_host_count",
    "support_nat_count",
    "support_eip_count",
    "support_data_volume_count",
    "support_interface_endpoint_az_set_count",
    "support_workflow_count",
}


def validate_support_usage(usage: object) -> bool:
    """Validate every fixed or maximum support/retained lifecycle counter."""

    if type(usage) is not dict or set(usage) != set(SUPPORT_USAGE_CEILINGS):
        raise ValueError("support usage field set is not exact")
    for field, ceiling in SUPPORT_USAGE_CEILINGS.items():
        value = usage[field]
        if (
            type(value) is not int
            or value < 0
            or (field in _EXACT_USAGE_FIELDS and value != ceiling)
            or (field not in _EXACT_USAGE_FIELDS and value > ceiling)
        ):
            raise ValueError(f"{field} exceeds or changes its exact ceiling")
    return True


def _validate_secret_resource_shapes(
    template: Mapping[str, object], inputs: SupportInputs
) -> Mapping[str, str]:
    if type(template) is not dict or type(template.get("Resources")) is not dict:
        raise ValueError("secret inventory requires an exact support template")
    resources = template["Resources"]
    secret_ids = tuple(
        logical_id
        for logical_id, resource in resources.items()
        if type(resource) is dict
        and resource.get("Type") == "AWS::SecretsManager::Secret"
    )
    if secret_ids != SECRET_LOGICAL_IDS:
        raise ValueError("support template must contain exactly eight ordered secrets")
    result: dict[str, str] = {}
    for logical_id in SECRET_LOGICAL_IDS:
        resource = resources[logical_id]
        properties = resource.get("Properties")
        if (
            type(properties) is not dict
            or properties.get("KmsKeyId") != inputs.retained_kms_key_arn
            or set(properties) != {"Description", "KmsKeyId", "Name", "Tags"}
            or type(properties["Tags"]) is not list
        ):
            raise ValueError("secret resource shape or retained KMS binding is wrong")
        tag_map = {
            item["Key"]: item["Value"]
            for item in properties["Tags"]
            if type(item) is dict and set(item) == {"Key", "Value"}
        }
        stage = tag_map.get("ImmutableVersionStage")
        if stage != _secret_stage(inputs, logical_id):
            raise ValueError("secret immutable version stage is missing or changed")
        policy = resources.get(f"{logical_id}Policy")
        if (
            type(policy) is not dict
            or policy.get("Type") != "AWS::SecretsManager::ResourcePolicy"
            or policy.get("Properties", {}).get("SecretId") != {"Ref": logical_id}
        ):
            raise ValueError("secret resource policy is absent or misbound")
        result[logical_id] = stage
    if len(set(result.values())) != 8:
        raise ValueError("each secret requires one unique immutable stage")
    return result


def secret_inventory(
    template: Mapping[str, object],
    inputs: SupportInputs,
    *,
    version_observations: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    """Validate exact ListSecretVersionIds observations for all eight secrets."""

    stages = _validate_secret_resource_shapes(template, inputs)
    if (
        type(version_observations) is not dict
        or tuple(version_observations) != SECRET_LOGICAL_IDS
    ):
        raise ValueError("secret version observation families are not exact")
    result: dict[str, Mapping[str, object]] = {}
    for logical_id in SECRET_LOGICAL_IDS:
        observation = version_observations[logical_id]
        if type(observation) is not dict or set(observation) != {
            "secret_arn",
            "version_id",
            "version_stage",
            "list_versions_identity_sha256",
            "versions",
        }:
            raise ValueError("secret version observation schema is not exact")
        secret_arn = observation["secret_arn"]
        expected_prefix = (
            f"arn:aws:secretsmanager:{REGION}:{ACCOUNT_ID}:secret:"
            f"/keep/glm52/{RUN_ID}/{inputs.activation_id}/"
            f"{_SECRET_PURPOSES[logical_id]}-"
        )
        version_id = observation["version_id"]
        stage = observation["version_stage"]
        versions = observation["versions"]
        if (
            type(secret_arn) is not str
            or not secret_arn.startswith(expected_prefix)
            or re.fullmatch(
                re.escape(expected_prefix) + r"[A-Za-z0-9-]{1,64}",
                secret_arn,
            )
            is None
            or type(version_id) is not str
            or _SECRET_VERSION_ID.fullmatch(version_id) is None
            or stage != stages[logical_id]
            or type(observation["list_versions_identity_sha256"]) is not str
            or _SHA256.fullmatch(observation["list_versions_identity_sha256"]) is None
            or type(versions) is not tuple
            or len(versions) != 1
            or type(versions[0]) is not dict
            or set(versions[0]) != {"version_id", "version_stages"}
            or versions[0]["version_id"] != version_id
            or versions[0]["version_stages"] != (stage,)
        ):
            raise ValueError("secret does not have one exact staged version")
        result[logical_id] = {
            "purpose": _SECRET_PURPOSES[logical_id],
            "secret_arn": secret_arn,
            "version_id": version_id,
            "version_stage": stage,
            "authorized_version_count": len(versions),
            "list_versions_identity_sha256": observation[
                "list_versions_identity_sha256"
            ],
            "version_id_source": (
                "Custom::H1gSkyBootstrap"
                if logical_id in SECRET_LOGICAL_IDS[:2]
                else "Custom::H1gTlsBundle"
            ),
        }
    return result


_SECRET_READER_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "ATTESTATION": (
        "RawSkyTokenSecret",
        "AttestationClientTlsSecret",
    ),
    "LAUNCH_ADMISSION": (
        "RawSkyTokenSecret",
        "LaunchAdmissionClientTlsSecret",
    ),
    "NUMERIC_BINDING": (
        "RawSkyTokenSecret",
        "NumericBindingClientTlsSecret",
    ),
    "RETAINED_CANCELLATION": (
        "RawSkyTokenSecret",
        "RetainedCancellationClientTlsSecret",
    ),
    "COMBINED_HOST_FIRST_BOOT": (
        "SkyBootstrapSecret",
        "CombinedHostTlsSecret",
    ),
    "TLS_HANDLER": ("CaIssuanceSecret",),
}


def validate_secret_read_result(
    *,
    inventory: Mapping[str, Mapping[str, object]],
    caller: str,
    secret_logical_id: str,
    requested_version_id: str,
    requested_version_stage: str,
    returned_secret_arn: str,
    returned_version_id: str,
    returned_version_stages: tuple[str, ...],
) -> bool:
    """Require each exact reader to request and verify both version coordinates."""

    if (
        type(inventory) is not dict
        or tuple(inventory) != SECRET_LOGICAL_IDS
        or caller not in _SECRET_READER_FAMILIES
        or secret_logical_id not in _SECRET_READER_FAMILIES[caller]
        or type(returned_version_stages) is not tuple
    ):
        raise ValueError("secret caller or inventory is not exact")
    expected = inventory[secret_logical_id]
    if (
        set(expected)
        != {
            "purpose",
            "secret_arn",
            "version_id",
            "version_stage",
            "authorized_version_count",
            "list_versions_identity_sha256",
            "version_id_source",
        }
        or expected["authorized_version_count"] != 1
        or requested_version_id != expected["version_id"]
        or requested_version_stage != expected["version_stage"]
        or returned_secret_arn != expected["secret_arn"]
        or returned_version_id != expected["version_id"]
        or returned_version_stages != (expected["version_stage"],)
    ):
        raise ValueError("secret read result differs from the pinned version")
    return True


def validate_secret_reader_partition(
    template: Mapping[str, object], inputs: SupportInputs
) -> bool:
    """Prove raw/bootstrap/TLS/CA readers are disjoint and mutation-closed."""

    _validate_secret_resource_shapes(template, inputs)
    resources = template["Resources"]
    expected_readers = {
        "CombinedHostRole": (
            "ec2.amazonaws.com",
            ("SkyBootstrapSecret", "CombinedHostTlsSecret"),
            "ReadBootstrapAndHostTlsOnly",
        ),
        "AttestationRole": (
            "lambda.amazonaws.com",
            ("RawSkyTokenSecret", "AttestationClientTlsSecret"),
            "ReadAttestationSecretsOnly",
        ),
        "RehearsalProbeRole": (
            "lambda.amazonaws.com",
            REHEARSAL_PROBE_SECRET_IDS,
            "ReadRehearsalProbeSecretsOnly",
        ),
        "LaunchAdmissionRole": (
            "lambda.amazonaws.com",
            ("RawSkyTokenSecret", "LaunchAdmissionClientTlsSecret"),
            "ReadLaunchAdmissionSecretsOnly",
        ),
        "NumericBindingRole": (
            "lambda.amazonaws.com",
            ("RawSkyTokenSecret", "NumericBindingClientTlsSecret"),
            "ReadNumericBindingSecretsOnly",
        ),
        "RetainedCancellationRole": (
            "lambda.amazonaws.com",
            ("RawSkyTokenSecret", "RetainedCancellationClientTlsSecret"),
            "ReadCancellationSecretsOnly",
        ),
    }
    for role_id, (service, allowed, policy_name) in expected_readers.items():
        properties = resources.get(role_id, {}).get("Properties", {})
        expected_property_keys = {
            "AssumeRolePolicyDocument",
            "Policies",
        }
        if role_id == "RehearsalProbeRole":
            expected_property_keys.add("RoleName")
        if (
            set(properties) != expected_property_keys
            or (
                role_id == "RehearsalProbeRole"
                and properties.get("RoleName") != "keep-glm52-h1g-rehearsal-probe"
            )
            or properties.get("AssumeRolePolicyDocument")
            != {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": service},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }
            or type(properties.get("Policies")) is not list
            or not properties["Policies"]
        ):
            raise ValueError(f"{role_id} trust or policy inventory changed")
        secret_policies = [
            item
            for item in properties["Policies"]
            if item.get("PolicyName") == policy_name
        ]
        if len(secret_policies) != 1:
            raise ValueError(f"{role_id} secret policy inventory changed")
        policy = secret_policies[0]
        statements = policy.get("PolicyDocument", {}).get("Statement")
        denied = tuple(item for item in SECRET_LOGICAL_IDS if item not in allowed)
        expected_statements = [
            *[
                statement
                for logical_id in allowed
                for statement in (
                    {
                        "Sid": f"Read{logical_id}ExactStage",
                        "Effect": "Allow",
                        "Action": "secretsmanager:GetSecretValue",
                        "Resource": {"Ref": logical_id},
                        "Condition": {
                            "StringEquals": {
                                "secretsmanager:VersionStage": _secret_stage(
                                    inputs, logical_id
                                )
                            }
                        },
                    },
                    {
                        "Sid": f"List{logical_id}VersionsExactSecret",
                        "Effect": "Allow",
                        "Action": "secretsmanager:ListSecretVersionIds",
                        "Resource": {"Ref": logical_id},
                    },
                )
            ],
            {
                "Sid": "DenyOtherSecretFamilies",
                "Effect": "Deny",
                "Action": "secretsmanager:*",
                "Resource": [{"Ref": item} for item in denied],
            },
            {
                "Sid": "DenySecretMutation",
                "Effect": "Deny",
                "Action": [
                    "secretsmanager:PutSecretValue",
                    "secretsmanager:UpdateSecret",
                    "secretsmanager:UpdateSecretVersionStage",
                    "secretsmanager:RotateSecret",
                ],
                "Resource": "*",
            },
        ]
        if (
            policy.get("PolicyName") != policy_name
            or policy.get("PolicyDocument", {}).get("Version") != "2012-10-17"
            or statements != expected_statements
        ):
            raise ValueError(f"{role_id} secret-reader partition changed")
        for extra in properties["Policies"]:
            if extra is policy:
                continue
            actions = {
                action
                for statement in extra.get("PolicyDocument", {}).get(
                    "Statement",
                    [],
                )
                for action in (
                    [statement.get("Action")]
                    if type(statement.get("Action")) is str
                    else statement.get("Action", [])
                )
            }
            if any(
                type(action) is str and action.startswith("secretsmanager:")
                for action in actions
            ):
                raise ValueError(f"{role_id} extra policy crosses secret partition")
    decision = resources.get("DecisionRole", {}).get("Properties", {})
    decision_policies = decision.get("Policies")
    decision_actions = {
        action
        for policy in (decision_policies if type(decision_policies) is list else [])
        for statement in policy.get("PolicyDocument", {}).get(
            "Statement",
            [],
        )
        for action in (
            [statement.get("Action")]
            if type(statement.get("Action")) is str
            else statement.get("Action", [])
        )
    }
    if (
        decision.get("AssumeRolePolicyDocument", {})
        .get("Statement", [{}])[0]
        .get("Principal")
        != {"Service": "lambda.amazonaws.com"}
        or type(decision_policies) is not list
        or len(decision_policies) != 1
        or any(
            type(action) is str and action.startswith("secretsmanager:")
            for action in decision_actions
        )
    ):
        raise ValueError("decision role may not read any secret")
    tls_properties = resources.get("TlsHandlerRole", {}).get("Properties", {})
    if (
        tls_properties.get("AssumeRolePolicyDocument", {})
        .get("Statement", [{}])[0]
        .get("Principal")
        != {"Service": "lambda.amazonaws.com"}
        or type(tls_properties.get("Policies")) is not list
        or len(tls_properties["Policies"]) != 1
    ):
        raise ValueError("TLS handler role inventory changed")
    tls_statements = (
        tls_properties["Policies"][0].get("PolicyDocument", {}).get("Statement")
    )
    if type(tls_statements) is not list:
        raise ValueError("TLS handler statements are absent")
    put_bindings = {
        (
            statement.get("Resource", {}).get("Ref"),
            statement.get("Condition", {})
            .get("StringEquals", {})
            .get("secretsmanager:VersionStage"),
        )
        for statement in tls_statements
        if statement.get("Action") == "secretsmanager:PutSecretValue"
    }
    list_targets = {
        statement.get("Resource", {}).get("Ref")
        for statement in tls_statements
        if statement.get("Action") == "secretsmanager:ListSecretVersionIds"
    }
    ca_reads = [
        statement
        for statement in tls_statements
        if statement.get("Action") == "secretsmanager:GetSecretValue"
    ]
    kms = [
        statement
        for statement in tls_statements
        if statement.get("Action")
        == ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey"]
    ]
    create_grant = [
        statement
        for statement in tls_statements
        if statement.get("Sid") == "CreateExactTlsHandlerGrant"
    ]
    reconcile_grant = [
        statement
        for statement in tls_statements
        if statement.get("Sid") == "ReconcileExactTlsHandlerGrant"
    ]
    direct_grant_evidence = [
        statement
        for statement in tls_statements
        if statement.get("Sid") == "PersistExactDirectGrantEvidence"
    ]
    if (
        len(tls_statements) != 21
        or put_bindings
        != {
            (logical_id, _secret_stage(inputs, logical_id))
            for logical_id in SECRET_LOGICAL_IDS
        }
        or list_targets != set(SECRET_LOGICAL_IDS)
        or ca_reads
        != [
            {
                "Sid": "ReadCaIssuanceExactStage",
                "Effect": "Allow",
                "Action": "secretsmanager:GetSecretValue",
                "Resource": {"Ref": "CaIssuanceSecret"},
                "Condition": {
                    "StringEquals": {
                        "secretsmanager:VersionStage": _secret_stage(
                            inputs, "CaIssuanceSecret"
                        )
                    }
                },
            }
        ]
        or len(kms) != 1
        or kms[0].get("Resource") != inputs.retained_kms_key_arn
        or kms[0].get("Condition")
        != {
            "StringLike": {
                "kms:EncryptionContext:SecretARN": (
                    f"arn:aws:secretsmanager:{REGION}:{ACCOUNT_ID}:secret:/keep/glm52/*"
                )
            }
        }
        or create_grant
        != [
            {
                "Sid": "CreateExactTlsHandlerGrant",
                "Effect": "Allow",
                "Action": "kms:CreateGrant",
                "Resource": inputs.retained_kms_key_arn,
                "Condition": {
                    "StringEquals": {
                        "kms:GranteePrincipal": TLS_HANDLER_ROLE_ARN,
                        "kms:RetiringPrincipal": SUPPORT_DELETION_ROLE_ARN,
                        "kms:EncryptionContext:RunId": RUN_ID,
                        "kms:EncryptionContext:ActivationId": (inputs.activation_id),
                    },
                    "ForAllValues:StringEquals": {
                        "kms:GrantOperations": [
                            "Decrypt",
                            "Encrypt",
                            "GenerateDataKey",
                        ]
                    },
                },
            }
        ]
        or reconcile_grant
        != [
            {
                "Sid": "ReconcileExactTlsHandlerGrant",
                "Effect": "Allow",
                "Action": [
                    "kms:ListGrants",
                    "kms:RetireGrant",
                    "kms:RevokeGrant",
                ],
                "Resource": inputs.retained_kms_key_arn,
            }
        ]
        or direct_grant_evidence
        != [
            {
                "Sid": "PersistExactDirectGrantEvidence",
                "Effect": "Allow",
                "Action": [
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                ],
                "Resource": inputs.ledger_table_arn,
            }
        ]
        or any(statement.get("Effect") != "Allow" for statement in tls_statements)
    ):
        raise ValueError("TLS handler secret authority is not exact")
    host_statements = resources["CombinedHostRole"]["Properties"]["Policies"][0][
        "PolicyDocument"
    ]["Statement"]
    for statement in host_statements:
        actions = statement["Action"]
        action_set = {actions} if type(actions) is str else set(actions)
        if (
            statement["Effect"] == "Allow"
            and "secretsmanager:PutSecretValue" in action_set
        ):
            raise ValueError("combined host may not mutate secrets")
    return True


_GRANT_PROJECTION_FIELDS = (
    "grant_id",
    "name",
    "grantee",
    "retiring_principal",
    "operations",
    "encryption_context",
    "revocable",
)
_KMS_OPERATIONS = {
    "Decrypt",
    "Encrypt",
    "GenerateDataKey",
    "GenerateDataKeyWithoutPlaintext",
}
_DIRECT_GRANT_OPERATIONS = ("Decrypt", "Encrypt", "GenerateDataKey")
_SERVICE_GRANT_OPERATIONS: Mapping[str, tuple[str, ...]] = {
    "CombinedHostRootVolume": (
        "Decrypt",
        "Encrypt",
        "GenerateDataKeyWithoutPlaintext",
    ),
    "CombinedHostDataVolume": (
        "Decrypt",
        "Encrypt",
        "GenerateDataKeyWithoutPlaintext",
    ),
    "CombinedHostDataVolumeDeletionSnapshot": (
        "Decrypt",
        "Encrypt",
        "GenerateDataKeyWithoutPlaintext",
    ),
    **{
        logical_id: ("Decrypt", "Encrypt", "GenerateDataKey")
        for logical_id in SECRET_LOGICAL_IDS
    },
}
_SERVICE_GRANT_SUFFIX = {
    "CombinedHostRootVolume": "ebs-root",
    "CombinedHostDataVolume": "ebs-data",
    "CombinedHostDataVolumeDeletionSnapshot": "ebs-data-deletion-snapshot",
    **{logical_id: _SECRET_PURPOSES[logical_id] for logical_id in SECRET_LOGICAL_IDS},
}


def _validate_grant_projection(
    grant: Mapping[str, object], *, label: str
) -> Mapping[str, object]:
    if type(grant) is not dict or set(grant) != set(_GRANT_PROJECTION_FIELDS):
        raise ValueError(f"{label} KMS grant projection is not exact")
    grant_id = grant["grant_id"]
    name = grant["name"]
    grantee = grant["grantee"]
    retiring = grant["retiring_principal"]
    operations = grant["operations"]
    context = grant["encryption_context"]
    if (
        type(grant_id) is not str
        or re.fullmatch(r"[A-Za-z0-9+=,.@_/-]{8,256}", grant_id) is None
        or type(name) is not str
        or re.fullmatch(r"[A-Za-z0-9+=,.@_/-]{1,256}", name) is None
        or type(grantee) is not str
        or (
            re.fullmatch(
                rf"arn:aws:iam::{ACCOUNT_ID}:role/[A-Za-z0-9+=,.@_/-]+",
                grantee,
            )
            is None
            and re.fullmatch(
                r"[a-z0-9-]+\." + re.escape(REGION) + r"\.amazonaws\.com",
                grantee,
            )
            is None
        )
        or type(retiring) is not str
        or re.fullmatch(
            rf"arn:aws:iam::{ACCOUNT_ID}:role/[A-Za-z0-9+=,.@_/-]+",
            retiring,
        )
        is None
        or type(operations) is not tuple
        or not operations
        or len(operations) != len(set(operations))
        or not set(operations).issubset(_KMS_OPERATIONS)
        or type(context) is not tuple
        or not context
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not str
            or not item[1]
            or not item[0].isascii()
            or not item[1].isascii()
            for item in context
        )
        or len({item[0] for item in context}) != len(context)
        or grant["revocable"] is not True
    ):
        raise ValueError(f"{label} KMS grant is malformed or unrevocable")
    return grant


def _grant_projection(grant: Mapping[str, object]) -> Mapping[str, object]:
    return {field: grant[field] for field in _GRANT_PROJECTION_FIELDS}


def _validate_unique_grant_ids(
    grants: tuple[Mapping[str, object], ...], *, label: str
) -> None:
    identities = tuple(grant["grant_id"] for grant in grants)
    if len(identities) != len(set(identities)):
        raise ValueError(f"{label} KMS grant identities are duplicated")


def _authenticated_kms_origin_identity(
    inputs: SupportInputs, logical_id: str
) -> str | None:
    readback = inputs.support_deletion_inventory["stack_resource_readback"]["resources"]
    matches = [
        row
        for row in readback
        if type(row) is dict and row.get("LogicalResourceId") == logical_id
    ]
    if len(matches) != 1:
        return None
    row = matches[0]
    if logical_id in {
        "CombinedHostRootVolume",
        "CombinedHostDataVolume",
        "CombinedHostDataVolumeDeletionSnapshot",
    }:
        physical_id = row.get("PhysicalResourceId")
        return physical_id if type(physical_id) is str else None
    if logical_id in SECRET_LOGICAL_IDS:
        resource_arn = row.get("ResourceArn")
        return resource_arn if type(resource_arn) is str else None
    return None


def validate_kms_grant_inventory(
    *,
    inputs: SupportInputs,
    baseline: tuple[Mapping[str, object], ...],
    active: tuple[Mapping[str, object], ...],
    h1g_created: tuple[Mapping[str, object], ...],
    service_created: tuple[Mapping[str, object], ...],
    restored: tuple[Mapping[str, object], ...],
) -> bool:
    """Validate direct/service provenance and exact retained-baseline restoration."""

    if type(inputs) is not SupportInputs:
        raise ValueError("KMS grant inventory requires exact support inputs")
    for value in (baseline, active, h1g_created, service_created, restored):
        if type(value) is not tuple:
            raise ValueError("KMS grant inventories must be exact tuples")
    for grant in baseline:
        _validate_grant_projection(grant, label="baseline")
    for grant in active:
        _validate_grant_projection(grant, label="active")
    for grant in restored:
        _validate_grant_projection(grant, label="restored")
    _validate_unique_grant_ids(baseline, label="baseline")
    _validate_unique_grant_ids(active, label="active")
    _validate_unique_grant_ids(restored, label="restored")
    if restored != baseline:
        raise ValueError("retained KMS grant baseline was not restored exactly")
    if len(h1g_created) != 1:
        raise ValueError("one exact direct H1g KMS grant is required")
    direct_projections = []
    for grant in h1g_created:
        if set(grant) != {
            *_GRANT_PROJECTION_FIELDS,
            "request_identity_sha256",
            "response_identity_sha256",
        }:
            raise ValueError("H1g-created grant provenance is not exact")
        projection = _grant_projection(grant)
        _validate_grant_projection(projection, label="H1g-created")
        if (
            any(
                type(grant[field]) is not str or _SHA256.fullmatch(grant[field]) is None
                for field in (
                    "request_identity_sha256",
                    "response_identity_sha256",
                )
            )
            or grant["operations"] != _DIRECT_GRANT_OPERATIONS
            or grant["name"] != f"h1g-{inputs.activation_id}-secrets"
            or grant["grantee"] != TLS_HANDLER_ROLE_ARN
            or grant["retiring_principal"] != SUPPORT_DELETION_ROLE_ARN
            or grant["encryption_context"]
            != (
                ("RunId", RUN_ID),
                ("ActivationId", inputs.activation_id),
            )
            or "snapshot-recovery" in canonical_json_bytes(grant).decode("ascii")
        ):
            raise ValueError("H1g-created grant is overbroad or unattributed")
        direct_projections.append(projection)
    service_projections = []
    for grant in service_created:
        if set(grant) != {
            *_GRANT_PROJECTION_FIELDS,
            "baseline_identity_sha256",
            "diff_identity_sha256",
            "list_grants_identity_sha256",
            "cloudtrail_event_identity_sha256",
            "cloudtrail_request_identity_sha256",
            "cloudtrail_response_identity_sha256",
            "originating_resource",
            "originating_resource_identity",
            "originating_service",
            "settling_window_seconds",
        }:
            raise ValueError("service-created grant provenance is not exact")
        projection = _grant_projection(grant)
        _validate_grant_projection(projection, label="service-created")
        originating_resource = grant["originating_resource"]
        originating_identity = grant["originating_resource_identity"]
        expected_operations = _SERVICE_GRANT_OPERATIONS.get(originating_resource)
        is_secret = originating_resource in SECRET_LOGICAL_IDS
        expected_service = (
            f"secretsmanager.{REGION}.amazonaws.com"
            if is_secret
            else f"ec2.{REGION}.amazonaws.com"
        )
        context_key = "SecretARN" if is_secret else "aws:ebs:id"
        expected_identity = _authenticated_kms_origin_identity(
            inputs, originating_resource
        )
        if (
            any(
                type(grant[field]) is not str or _SHA256.fullmatch(grant[field]) is None
                for field in (
                    "baseline_identity_sha256",
                    "diff_identity_sha256",
                    "list_grants_identity_sha256",
                    "cloudtrail_event_identity_sha256",
                    "cloudtrail_request_identity_sha256",
                    "cloudtrail_response_identity_sha256",
                )
            )
            or expected_operations is None
            or grant["operations"] != expected_operations
            or grant["name"]
            != (
                f"h1g-{inputs.activation_id}-"
                f"{_SERVICE_GRANT_SUFFIX.get(originating_resource, '')}"
            )
            or grant["grantee"] != expected_service
            or grant["originating_service"] != expected_service
            or grant["retiring_principal"] != SUPPORT_DELETION_ROLE_ARN
            or type(originating_identity) is not str
            or originating_identity != expected_identity
            or grant["encryption_context"]
            != (
                ("RunId", RUN_ID),
                ("ActivationId", inputs.activation_id),
                (context_key, originating_identity),
            )
            or type(grant["settling_window_seconds"]) is not int
            or not 1 <= grant["settling_window_seconds"] <= 900
        ):
            raise ValueError("service-created grant evidence is missing")
        service_projections.append(projection)
    originating_resources = tuple(
        grant["originating_resource"] for grant in service_created
    )
    if len(originating_resources) != len(set(originating_resources)):
        raise ValueError("service-created grant origin is duplicated")
    expected_active = baseline + tuple(direct_projections) + tuple(service_projections)
    _validate_unique_grant_ids(expected_active, label="attributed")
    if active != expected_active:
        raise ValueError(
            "active KMS grants are missing, unknown, duplicated, or unattributed"
        )
    return True


def _term_map(terms: tuple[PriceTerm, ...]) -> Mapping[str, str]:
    return {term.term: term.estimated_usd for term in terms}


_SUPPORT_ENVELOPE: Mapping[str, object] = {
    "accepted_support_lambda_invocations": 10000,
    "application_log_ingestion_mib_max": 512,
    "custom_resource_creates_max": 2,
    "custom_resource_deletes_max": 2,
    "custom_resource_identity_updates_max": 2,
    "delete_request_deadline_hours": 71,
    "forensic_snapshot_count": 1,
    "forensic_snapshot_retention_days": 7,
    "host_local_egress_drain_gib": 6,
    "host_local_egress_warning_gib": 5,
    "host_root_volume_count": 1,
    "host_workflow_count": 1,
    "interface_endpoint_eni_count": 2,
    "lambda_reserved_concurrency_per_function": 1,
    "nat_processed_gib_max": 10,
    "retained_finalization_execution_max": 12,
    "retained_sky_cancel_actions_max": 8,
    "runtime_observations_max": 432,
    "secret_count": 8,
    "secret_versions_per_secret": 1,
    "snapshot_delete_calls_max": 12,
    "absence_expected_hours": 72,
    "support_data_volume_count": 1,
    "support_data_volume_gib": 50,
    "support_eip_count": 1,
    "support_host_count": 1,
    "support_interface_endpoint_az_set_count": 1,
    "support_nat_count": 1,
    "support_work_stop_hours": 68,
    "transient_log_retention_days": 14,
    "workflow_history_events_max": 12000,
}


def build_support_spend_descriptor(
    *, inputs: object, price_card: SupportPriceCard
) -> dict[str, object]:
    if (
        type(inputs) not in {SupportInputs, SupportBuildInputs}
        or type(price_card) is not SupportPriceCard
        or price_card.price_card_identity_sha256 != inputs.price_card_identity_sha256
    ):
        raise ValueError("support spend inputs are not exact and mutually bound")
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_spend_descriptor_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": inputs.activation_id,
        "support_inputs_identity_sha256": _support_input_identity(inputs),
        "price_card_effective_date": price_card.effective_date,
        "price_card_identity_sha256": price_card.price_card_identity_sha256,
        "estimated_support_total_usd": price_card.estimated_support_total_usd,
        "estimated_support_ceiling_usd": "25.00",
        "support_costs": dict(_term_map(price_card.support_terms)),
        "retained_costs": dict(_term_map(price_card.retained_terms)),
        "envelope": dict(_SUPPORT_ENVELOPE),
        "gpu_residual_reserve_included": False,
        "worker_root_volume_tail_included": False,
    }
    body["canonical_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    return body


def validate_support_spend_descriptor(
    descriptor: Mapping[str, object], inputs: object
) -> Mapping[str, object]:
    if type(descriptor) is not dict:
        raise ValueError("support spend descriptor must be one exact object")
    body = dict(descriptor)
    identity = body.pop("canonical_body_sha256", None)
    if (
        type(identity) is not str
        or _SHA256.fullmatch(identity) is None
        or hashlib.sha256(canonical_json_bytes(body)).hexdigest() != identity
        or body.get("support_inputs_identity_sha256") != _support_input_identity(inputs)
        or body.get("price_card_identity_sha256") != inputs.price_card_identity_sha256
        or body.get("estimated_support_ceiling_usd") != "25.00"
        or _money(
            body.get("estimated_support_total_usd"),
            "support descriptor total",
        )
        > Decimal("25.00")
        or body.get("envelope") != _SUPPORT_ENVELOPE
        or set(body.get("support_costs", {})) != set(_SUPPORT_PRICE_TERMS)
        or set(body.get("retained_costs", {})) != set(_RETAINED_PRICE_TERMS)
        or set(body["support_costs"]) & set(body["retained_costs"])
        or body.get("gpu_residual_reserve_included") is not False
        or body.get("worker_root_volume_tail_included") is not False
    ):
        raise ValueError("support spend descriptor is not exact")
    return descriptor


def _deletion_statement_sid(action: str) -> str:
    return "Exact" + "".join(
        part[:1].upper() + part[1:]
        for part in re.split(r"[^A-Za-z0-9]+", action)
        if part
    )


def _support_deletion_inventory_metadata(
    inventory: Mapping[str, object],
) -> Mapping[str, object]:
    if type(inventory) is not dict:
        raise ValueError("support deletion inventory is not exact")
    if inventory.get("record_type") == ("glm52_h1g_support_deletion_inventory_v1"):
        return {
            "SupportDeletionInventorySha256": inventory["canonical_body_sha256"],
            "SupportResourceReadbackSha256": inventory["stack_resource_readback"][
                "canonical_body_sha256"
            ],
            "SupportResourceReadbackEvidenceSha256": inventory[
                "stack_resource_readback_evidence"
            ]["canonical_body_sha256"],
            "SupportDeletionResourceArns": list(inventory["resource_arns"]),
            "SupportDeletionResourceIds": list(inventory["resource_ids"]),
            "SupportDeletionResourceNames": list(inventory["resource_names"]),
            "CustomResourceDeleteCallbacks": deepcopy(
                inventory["custom_resource_delete_callbacks"]
            ),
        }
    if inventory.get("record_type") != ("glm52_h1g_support_postcreate_inventory_v1"):
        raise ValueError("support deletion inventory record type is foreign")
    body = dict(inventory)
    identity = body.pop("canonical_body_sha256", None)
    rows = body.get("stack_resources")
    if (
        type(identity) is not str
        or _SHA256.fullmatch(identity) is None
        or hashlib.sha256(canonical_json_bytes(body)).hexdigest() != identity
        or type(rows) is not list
        or len(rows) != 202
        or type(body.get("action_resources")) is not dict
        or set(body["action_resources"]) != _SUPPORT_DELETION_ACTIONS
    ):
        raise ValueError("postcreate support deletion inventory is foreign")
    evidence_identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "describe_stacks_response_sha256": body[
                    "describe_stacks_response_sha256"
                ],
                "list_stack_resources_response_sha256": body[
                    "list_stack_resources_response_sha256"
                ],
                "detail_response_sha256": [
                    row["source_response_sha256"] for row in rows
                ],
            }
        )
    ).hexdigest()
    return {
        "SupportDeletionInventorySha256": identity,
        "SupportResourceReadbackSha256": identity,
        "SupportResourceReadbackEvidenceSha256": evidence_identity,
        "SupportDeletionResourceArns": list(body["resource_arns"]),
        "SupportDeletionResourceIds": list(body["resource_ids"]),
        "SupportDeletionResourceNames": list(body["resource_names"]),
        "CustomResourceDeleteCallbacks": deepcopy(
            body["custom_resource_delete_callbacks"]
        ),
    }


def _support_deletion_policy_statements(
    inputs: SupportInputs,
) -> tuple[Mapping[str, object], ...]:
    action_resources = inputs.support_deletion_inventory["action_resources"]
    statements = []
    for action in sorted(_SUPPORT_DELETION_ACTIONS):
        resources = action_resources[action]
        statement: dict[str, object] = {
            "Sid": _deletion_statement_sid(action),
            "Effect": "Allow",
            "Action": action,
            "Resource": resources[0] if len(resources) == 1 else list(resources),
        }
        if action == "ec2:CreateSnapshot":
            statement["Condition"] = {
                "StringEquals": {
                    "aws:RequestTag/ActivationId": inputs.activation_id,
                    "aws:RequestTag/RunId": RUN_ID,
                    "aws:RequestTag/Purpose": (
                        "forensic-data-volume-deletion-snapshot"
                    ),
                },
                "ForAllValues:StringEquals": {
                    "aws:TagKeys": ["ActivationId", "Purpose", "RunId"]
                },
            }
        elif action == "kms:CreateGrant":
            statement["Condition"] = {
                "Bool": {"kms:GrantIsForAWSResource": "true"},
                "StringEquals": {
                    "kms:EncryptionContext:ActivationId": (inputs.activation_id),
                    "kms:EncryptionContext:RunId": RUN_ID,
                },
            }
        elif action in _UNSCOPED_SUPPORT_DELETION_READ_ACTIONS:
            statement["Condition"] = {"StringEquals": {"aws:RequestedRegion": REGION}}
        statements.append(statement)
    return tuple(statements)


def _task12_support_plane_inputs(inputs: object) -> object:
    from .task10_worker import GRACEFUL_STOP_DOCUMENT
    from .task12_support_plane import Task12SupportPlaneInputs

    return Task12SupportPlaneInputs(
        activation_id=inputs.activation_id,
        lambda_code_sha256=inputs.lambda_code_sha256,
        lambda_code_bucket=inputs.lambda_code_bucket,
        lambda_code_key=inputs.lambda_code_key,
        lambda_code_version=inputs.lambda_code_version_id,
        ledger_table_arn=inputs.ledger_table_arn,
        retained_kms_key_arn=inputs.retained_kms_key_arn,
        model_bucket_arn=inputs.model_bucket_arn,
        worker_drain_document_arn=(
            f"arn:aws:ssm:{REGION}:{ACCOUNT_ID}:document/{GRACEFUL_STOP_DOCUMENT}"
        ),
    )


def _task12_authenticated_dispatch_bindings(
    inputs: object,
) -> Mapping[str, str]:
    if type(inputs) not in {
        PreSupportRuntimeInputs,
        SupportBuildInputs,
        SupportInputs,
    }:
        raise TypeError("Task 12 dispatch binding requires exact authenticated inputs")
    activation_ordinal = "1"
    generation = "1"
    generation_text = "00000001"
    dispatch_identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "GLM52_TASK12_RETAINED_DISPATCH_BINDING_V1",
                "account_id": ACCOUNT_ID,
                "region": REGION,
                "run_id": RUN_ID,
                "activation_id": inputs.activation_id,
                "activation_ordinal": 1,
                "generation": 1,
                "generation_text": generation_text,
                "fence_manifest_identity_sha256": (
                    _support_fence_manifest_coordinate(inputs)[
                        "canonical_identity_sha256"
                    ]
                ),
                "lambda_code_sha256": inputs.lambda_code_sha256,
            }
        )
    ).hexdigest()
    return {
        "Task12ActivationOrdinal": activation_ordinal,
        "Task12Generation": generation,
        "Task12GenerationText": generation_text,
        "Task12DispatchIdentitySha256": dispatch_identity,
    }


def _task12_future_support_bindings() -> Mapping[str, str]:
    state_machine_arn = (
        f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:"
        "keep-glm52-h1g-retained-lifecycle"
    )
    return {
        "NumericBindingVersion": (
            f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:"
            "keep-glm52-h1g-numeric-binding:1"
        ),
        "RetainedCancellationVersion": (
            f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:"
            "keep-glm52-h1g-retained-cancellation:1"
        ),
        "H1gRetainedLifecycleStateMachine": state_machine_arn,
        "H1gRetainedLifecycleStateMachineVersion": (state_machine_arn + ":1"),
    }


def _bind_task12_dispatch_parameters(
    value: object,
    *,
    bindings: Mapping[str, str],
) -> object:
    if type(value) is dict and set(value) == {"Ref"} and value["Ref"] in bindings:
        return bindings[value["Ref"]]
    if type(value) is dict:
        return {
            key: _bind_task12_dispatch_parameters(
                item,
                bindings=bindings,
            )
            for key, item in value.items()
        }
    if type(value) is list:
        return [
            _bind_task12_dispatch_parameters(
                item,
                bindings=bindings,
            )
            for item in value
        ]
    return deepcopy(value)


def _build_task12_retained_fragment(
    inputs: object,
) -> Mapping[str, object]:
    """Render Task 12 from the same authenticated retained coordinates."""

    from .task12_support_plane import render_task12_support_plane_fragment

    rendered = render_task12_support_plane_fragment(
        inputs=_task12_support_plane_inputs(inputs)
    )
    dispatch_bindings = _task12_authenticated_dispatch_bindings(inputs)
    future_support_bindings = _task12_future_support_bindings()
    bindings = {
        **dispatch_bindings,
        **future_support_bindings,
    }
    bound = _bind_task12_dispatch_parameters(
        rendered,
        bindings=bindings,
    )
    if type(bound) is not dict:
        raise TypeError("Task 12 retained fragment is malformed")
    parameters = bound.pop("Parameters", None)
    if type(parameters) is not dict or set(parameters) != set(dispatch_bindings):
        raise ValueError("Task 12 dispatch parameter inventory is not exact")
    if any(
        canonical_json_bytes({"Ref": name}) in canonical_json_bytes(bound)
        for name in bindings
    ):
        raise ValueError("Task 12 dispatch parameter remains unresolved")
    bound["Metadata"]["task12_authenticated_dispatch_binding"] = {
        "activation_ordinal": 1,
        "generation": 1,
        "generation_text": "00000001",
        "dispatch_identity_sha256": dispatch_bindings["Task12DispatchIdentitySha256"],
        "source": "AUTHENTICATED_RETAINED_FOUNDATION_AND_FENCE_INPUTS",
    }
    bound["Metadata"]["task12_future_support_bindings"] = {
        "absence_prerequisite": ("AUTHENTICATED_AWS_NOT_FOUND_READBACK"),
        "first_publication_version": 1,
        "logical_bindings": dict(future_support_bindings),
        "postcreate_exact_readback_required": True,
    }
    return bound


def _build_task10_retained_fragment(inputs: object) -> Mapping[str, object]:
    from .task10_support_plane import (
        Task10SupportInputs,
        render_task10_support_plane_fragment,
    )

    return render_task10_support_plane_fragment(
        inputs=Task10SupportInputs(
            lambda_code_bucket=inputs.lambda_code_bucket,
            lambda_code_key=inputs.lambda_code_key,
            lambda_code_version=inputs.lambda_code_version_id,
            lambda_code_sha256=inputs.lambda_code_sha256,
            ledger_table_arn=inputs.ledger_table_arn,
            campaign_bucket_arn=inputs.model_bucket_arn,
        )
    )


_COMPOSABLE_TEMPLATE_KEYS = frozenset(
    {
        "AWSTemplateFormatVersion",
        "Conditions",
        "Description",
        "Mappings",
        "Metadata",
        "Outputs",
        "Parameters",
        "Resources",
        "Rules",
        "Transform",
    }
)
_ADDITIVE_TEMPLATE_SECTIONS = (
    "Parameters",
    "Mappings",
    "Rules",
    "Conditions",
    "Resources",
    "Outputs",
)
_SECTION_COLLISION_LABEL = {
    "Parameters": "parameter",
    "Mappings": "mapping",
    "Rules": "rule",
    "Conditions": "condition",
    "Resources": "resource",
    "Outputs": "output",
}


def _validate_composable_template(
    value: object,
    *,
    label: str,
) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} is a malformed template")
    unknown = set(value) - _COMPOSABLE_TEMPLATE_KEYS
    if unknown:
        raise ValueError(f"{label} is a malformed template: unknown={sorted(unknown)}")
    if value.get("AWSTemplateFormatVersion") != "2010-09-09":
        raise ValueError(f"{label} template format is not exact")
    description = value.get("Description")
    if description is not None and type(description) is not str:
        raise ValueError(f"{label} is a malformed template")
    transform = value.get("Transform")
    if transform is not None and not (
        type(transform) is str
        or (
            type(transform) is list
            and transform
            and all(type(item) is str for item in transform)
        )
    ):
        raise ValueError(f"{label} is a malformed template")
    if "Resources" not in value:
        raise ValueError(f"{label} is a malformed template")
    for section in (*_ADDITIVE_TEMPLATE_SECTIONS, "Metadata"):
        section_value = value.get(section)
        if section_value is not None and type(section_value) is not dict:
            raise ValueError(f"{label} has malformed {section}")
    for logical_id, resource in value["Resources"].items():
        if (
            type(logical_id) is not str
            or not logical_id
            or type(resource) is not dict
            or type(resource.get("Type")) is not str
            or not resource["Type"]
        ):
            raise ValueError(f"{label} contains a malformed resource")
    for name, parameter in value.get("Parameters", {}).items():
        if (
            type(name) is not str
            or not name
            or type(parameter) is not dict
            or type(parameter.get("Type")) is not str
        ):
            raise ValueError(f"{label} contains a malformed parameter")
    for name, output in value.get("Outputs", {}).items():
        if (
            type(name) is not str
            or not name
            or type(output) is not dict
            or "Value" not in output
            or (
                "Export" in output
                and (
                    type(output["Export"]) is not dict
                    or set(output["Export"]) != {"Name"}
                )
            )
        ):
            raise ValueError(f"{label} contains a malformed output")
    for section in ("Mappings", "Rules"):
        for name, item in value.get(section, {}).items():
            if type(name) is not str or not name or type(item) is not dict:
                raise ValueError(f"{label} contains malformed {section}")
    if any(
        type(name) is not str or not name or expression is None
        for name, expression in value.get("Conditions", {}).items()
    ):
        raise ValueError(f"{label} contains malformed Conditions")
    export_owner: dict[bytes, str] = {}
    for output_name, output in value.get("Outputs", {}).items():
        if "Export" not in output:
            continue
        identity = canonical_json_bytes(output["Export"]["Name"])
        if identity in export_owner:
            raise ValueError(
                f"output export collision: {export_owner[identity]}, {output_name}"
            )
        export_owner[identity] = output_name
    return value


def _merge_additive_metadata(
    target: dict[str, object],
    incoming: Mapping[str, object],
    *,
    path: str = "Metadata",
) -> None:
    for key, value in incoming.items():
        if type(key) is not str or not key:
            raise ValueError("metadata conflict at malformed key")
        if key not in target:
            target[key] = deepcopy(value)
            continue
        current = target[key]
        if type(current) is dict and type(value) is dict:
            _merge_additive_metadata(
                current,
                value,
                path=f"{path}.{key}",
            )
        elif current != value:
            raise ValueError(f"metadata conflict at {path}.{key}")


def compose_complete_retained_template(
    base_template: Mapping[str, object],
    *fragments: Mapping[str, object],
) -> Mapping[str, object]:
    """Deep-preserve one base template and merge additive retained fragments."""

    base = _validate_composable_template(
        base_template,
        label="base",
    )
    base_snapshot = deepcopy(base)
    result: dict[str, object] = deepcopy(base)
    for index, fragment_value in enumerate(fragments, start=1):
        fragment = _validate_composable_template(
            fragment_value,
            label=f"fragment {index}",
        )
        for scalar in (
            "AWSTemplateFormatVersion",
            "Description",
            "Transform",
        ):
            if scalar not in fragment:
                continue
            if scalar in result and result[scalar] != fragment[scalar]:
                name = (
                    "format" if scalar == "AWSTemplateFormatVersion" else scalar.lower()
                )
                raise ValueError(f"template {name} conflict")
            result.setdefault(scalar, deepcopy(fragment[scalar]))
        for section in _ADDITIVE_TEMPLATE_SECTIONS:
            incoming = fragment.get(section, {})
            if not incoming:
                continue
            destination = result.setdefault(section, {})
            if type(destination) is not dict:
                raise ValueError(f"base has malformed {section}")
            collisions = set(destination) & set(incoming)
            if collisions:
                label = _SECTION_COLLISION_LABEL[section]
                raise ValueError(f"{label} collision: {sorted(collisions)}")
            destination.update(deepcopy(incoming))
        metadata = fragment.get("Metadata", {})
        if metadata:
            destination_metadata = result.setdefault("Metadata", {})
            if type(destination_metadata) is not dict:
                raise ValueError("base has malformed Metadata")
            _merge_additive_metadata(destination_metadata, metadata)
    if base != base_snapshot:
        raise RuntimeError("base template was mutated")
    _validate_composable_template(result, label="composed")
    return result


def build_retained_fence_bootstrap_fragment(
    inputs: RetainedFenceBootstrapInputs,
) -> Mapping[str, object]:
    """Build the manifest-independent runtime that alone may publish the seed."""

    from .fence_artifacts import BOOTSTRAP_PUBLICATION_ORDER

    if type(inputs) is not RetainedFenceBootstrapInputs:
        raise TypeError("retained fence bootstrap requires exact inputs")

    materializer_name = "keep-glm52-h1g-fence-bootstrap-materializer"
    publisher_name = "keep-glm52-h1g-fence-bootstrap-artifact-publisher"
    invoker_name = "keep-glm52-h1g-fence-bootstrap-invoker"
    deployment_role_name = "keep-glm52-h1g-cloudformation-deployment"
    gpu_ami_parameter_arn = (
        f"arn:aws:ssm:{REGION}::parameter/aws/service/deeplearning/ami/"
        "x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
    )
    materializer_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{materializer_name}"
    publisher_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{publisher_name}"
    invoker_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{invoker_name}"
    object_arns = [
        inputs.model_bucket_arn + "/" + key for key in BOOTSTRAP_PUBLICATION_ORDER
    ]
    log_group_name = f"/aws/lambda/{materializer_name}"
    log_group_arn = f"arn:aws:logs:{REGION}:{ACCOUNT_ID}:log-group:{log_group_name}:*"

    def retained(resource: Mapping[str, object]) -> dict[str, object]:
        return {
            **deepcopy(resource),
            "DeletionPolicy": "Retain",
            "UpdateReplacePolicy": "Retain",
        }

    def role_trust(role_arn: str) -> Mapping[str, object]:
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": role_arn},
                    "Action": "sts:AssumeRole",
                    "Condition": {"ArnEquals": {"aws:PrincipalArn": role_arn}},
                }
            ],
        }

    publisher = retained(
        {
            "Type": "AWS::IAM::Role",
            "Properties": {
                "RoleName": publisher_name,
                "AssumeRolePolicyDocument": role_trust(materializer_arn),
                "Policies": [
                    {
                        "PolicyName": "CreateOnlyExactBootstrapArtifacts",
                        "PolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Sid": "CreateOnlyExactBootstrapArtifacts",
                                    "Effect": "Allow",
                                    "Action": "s3:PutObject",
                                    "Resource": object_arns,
                                },
                                {
                                    "Sid": "EncryptOnlyExactBootstrapArtifacts",
                                    "Effect": "Allow",
                                    "Action": [
                                        "kms:Encrypt",
                                        "kms:GenerateDataKey",
                                    ],
                                    "Resource": inputs.retained_kms_key_arn,
                                    "Condition": {
                                        "StringEquals": {
                                            "kms:ViaService": (
                                                "s3.us-west-2.amazonaws.com"
                                            )
                                        },
                                        "ForAnyValue:StringEquals": {
                                            "kms:EncryptionContext:aws:s3:arn": (
                                                object_arns
                                            )
                                        },
                                    },
                                },
                            ],
                        },
                    }
                ],
            },
        }
    )
    publisher["DependsOn"] = "FenceBootstrapMaterializerRole"
    materializer = retained(
        {
            "Type": "AWS::IAM::Role",
            "Properties": {
                "RoleName": materializer_name,
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "lambda.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                },
                "Policies": [
                    {
                        "PolicyName": "MaterializeOnlyBootstrapArtifacts",
                        "PolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Sid": "WriteOnlyOwnLogs",
                                    "Effect": "Allow",
                                    "Action": [
                                        "logs:CreateLogStream",
                                        "logs:PutLogEvents",
                                    ],
                                    "Resource": log_group_arn,
                                },
                                {
                                    "Sid": "ReadOnlyExactBootstrapArtifacts",
                                    "Effect": "Allow",
                                    "Action": [
                                        "s3:GetObjectVersion",
                                        "s3:GetObjectVersionAttributes",
                                        "s3:GetObjectTagging",
                                    ],
                                    "Resource": object_arns,
                                },
                                {
                                    "Sid": "InventoryOnlyBootstrapHistory",
                                    "Effect": "Allow",
                                    "Action": "s3:ListBucketVersions",
                                    "Resource": inputs.model_bucket_arn,
                                    "Condition": {
                                        "StringLike": {
                                            "s3:prefix": [
                                                key
                                                for key in (BOOTSTRAP_PUBLICATION_ORDER)
                                            ]
                                        }
                                    },
                                },
                                {
                                    "Sid": "DecryptOnlyBootstrapArtifacts",
                                    "Effect": "Allow",
                                    "Action": "kms:Decrypt",
                                    "Resource": inputs.retained_kms_key_arn,
                                    "Condition": {
                                        "StringEquals": {
                                            "kms:ViaService": (
                                                "s3.us-west-2.amazonaws.com"
                                            )
                                        },
                                        "ForAnyValue:StringEquals": {
                                            "kms:EncryptionContext:aws:s3:arn": (
                                                object_arns
                                            )
                                        },
                                    },
                                },
                                {
                                    "Sid": "AssumeOnlyBootstrapPublisher",
                                    "Effect": "Allow",
                                    "Action": "sts:AssumeRole",
                                    "Resource": publisher_arn,
                                },
                            ],
                        },
                    }
                ],
            },
        }
    )
    invoker = retained(
        {
            "Type": "AWS::IAM::Role",
            "Properties": {
                "RoleName": invoker_name,
                "AssumeRolePolicyDocument": role_trust(inputs.stage_operator_role_arn),
                "Policies": [
                    {
                        "PolicyName": "InvokeOnlyBootstrapMaterializer",
                        "PolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Action": "lambda:InvokeFunction",
                                    "Resource": {
                                        "Ref": "FenceBootstrapMaterializerVersion"
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        }
    )
    resources = {
        "FenceBootstrapInvokerRole": invoker,
        "FenceBootstrapMaterializerLogGroup": retained(
            {
                "Type": "AWS::Logs::LogGroup",
                "Properties": {
                    "LogGroupName": log_group_name,
                    "RetentionInDays": 30,
                },
            }
        ),
        "FenceBootstrapMaterializerRole": materializer,
        "FenceBootstrapMaterializerFunction": retained(
            {
                "Type": "AWS::Lambda::Function",
                "DependsOn": [
                    "FenceBootstrapMaterializerLogGroup",
                    "FenceBootstrapArtifactPublisherRole",
                    "FenceBootstrapDeploymentRoleBootstrapAuthorityPolicy",
                ],
                "Properties": {
                    "FunctionName": materializer_name,
                    "Runtime": "python3.13",
                    "Handler": ("support_fence_handler.bootstrap_publication_main"),
                    "Role": {
                        "Fn::GetAtt": [
                            "FenceBootstrapMaterializerRole",
                            "Arn",
                        ]
                    },
                    "Code": {
                        "S3Bucket": inputs.lambda_code_bucket,
                        "S3Key": inputs.lambda_code_key,
                        "S3ObjectVersion": inputs.lambda_code_version_id,
                    },
                    "Timeout": 30,
                    "MemorySize": 256,
                    "ReservedConcurrentExecutions": 1,
                    "Environment": {
                        "Variables": {
                            "GLM52_ACCOUNT_ID": ACCOUNT_ID,
                            "GLM52_REGION": REGION,
                            "GLM52_RUN_ID": RUN_ID,
                            "GLM52_ACTIVATION_ID": inputs.activation_id,
                            "GLM52_BOOTSTRAP_PUBLISHER_ROLE_ARN": (publisher_arn),
                            "GLM52_MODEL_BUCKET_ARN": (inputs.model_bucket_arn),
                            "GLM52_RETAINED_KMS_KEY_ARN": (inputs.retained_kms_key_arn),
                        }
                    },
                },
            }
        ),
        "FenceBootstrapMaterializerVersion": retained(
            {
                "Type": "AWS::Lambda::Version",
                "Properties": {
                    "FunctionName": {"Ref": "FenceBootstrapMaterializerFunction"},
                    "CodeSha256": base64.b64encode(
                        bytes.fromhex(inputs.lambda_code_sha256)
                    ).decode("ascii"),
                },
            }
        ),
        "FenceBootstrapMaterializerInvokePermission": retained(
            {
                "Type": "AWS::Lambda::Permission",
                "DependsOn": "FenceBootstrapInvokerRole",
                "Properties": {
                    "Action": "lambda:InvokeFunction",
                    "FunctionName": {"Ref": "FenceBootstrapMaterializerVersion"},
                    "Principal": invoker_arn,
                },
            }
        ),
        "FenceBootstrapArtifactPublisherRole": publisher,
        "FenceBootstrapDeploymentRoleBootstrapAuthorityPolicy": retained(
            {
                "Type": "AWS::IAM::Policy",
                "Properties": {
                    "PolicyName": "keep-glm52-h1g-retained-bootstrap-authority",
                    "Roles": [deployment_role_name],
                    "PolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Sid": "ResolveExactGpuAmiParameter",
                                "Effect": "Allow",
                                "Action": "ssm:GetParameters",
                                "Resource": gpu_ami_parameter_arn,
                            },
                            {
                                "Sid": "PassExactBootstrapMaterializerRole",
                                "Effect": "Allow",
                                "Action": "iam:PassRole",
                                "Resource": materializer_arn,
                                "Condition": {
                                    "StringEquals": {
                                        "iam:PassedToService": ("lambda.amazonaws.com")
                                    }
                                },
                            },
                        ],
                    },
                },
            }
        ),
    }
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Metadata": {
            "RecordType": "glm52_h1g_retained_fence_bootstrap_runtime_v2",
            "ActivationId": inputs.activation_id,
            "RetainedStackId": inputs.retained_stack_id,
        },
        "Outputs": {
            "FenceBootstrapInvokerRoleArn": {
                "Value": {"Fn::GetAtt": ["FenceBootstrapInvokerRole", "Arn"]}
            },
            "FenceBootstrapMaterializerVersionArn": {
                "Value": {"Ref": "FenceBootstrapMaterializerVersion"}
            },
            "FenceBootstrapPublisherRoleArn": {
                "Value": {
                    "Fn::GetAtt": [
                        "FenceBootstrapArtifactPublisherRole",
                        "Arn",
                    ]
                }
            },
        },
        "Resources": resources,
    }


def _build_h1g_retained_fence_fragment(
    inputs: object,
) -> Mapping[str, object]:
    """Build pre-support execution and source-settlement retained identities."""

    from .fence_artifacts import (
        BOOTSTRAP_PUBLICATION_ORDER,
        SOURCE_SETTLED_PUBLICATION_ORDER,
    )
    from .fence_source_settlement import (
        SOURCE_SETTLEMENT_EVIDENCE_READER_NAMES,
    )

    required = (
        "activation_id",
        "fence_stack_id",
        "fence_service_role_arn",
        "ledger_table_arn",
        "model_bucket_arn",
        "retained_kms_key_arn",
        "lambda_code_bucket",
        "lambda_code_key",
        "lambda_code_version_id",
        "lambda_code_sha256",
    )
    if any(not hasattr(inputs, field) for field in required):
        return {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Metadata": {"RecordType": "glm52_h1g_retained_fence_runtime_absent_v1"},
            "Resources": {},
        }
    manifest_coordinate = getattr(
        inputs,
        "bootstrap_manifest_coordinate",
        getattr(inputs, "fence_manifest_coordinate", None),
    )
    if not isinstance(manifest_coordinate, Mapping):
        raise ValueError("retained fence bootstrap coordinate is absent")
    manifest_coordinate = dict(manifest_coordinate)
    model_bucket_arn = inputs.model_bucket_arn
    bootstrap_objects = [
        model_bucket_arn + "/" + key for key in BOOTSTRAP_PUBLICATION_ORDER
    ]
    source_settled_objects = [
        model_bucket_arn + "/" + key for key in SOURCE_SETTLED_PUBLICATION_ORDER
    ]
    source_evidence_objects = [
        (
            model_bucket_arn
            + f"/campaigns/{RUN_ID}/authorities/fence/evidence/"
            + f"{inputs.activation_id}/00000001/{reader_name}.json"
        )
        for reader_name in SOURCE_SETTLEMENT_EVIDENCE_READER_NAMES
    ]
    source_materializer_read_objects = [
        *bootstrap_objects,
        *source_evidence_objects,
        model_bucket_arn + f"/campaigns/{RUN_ID}/spend-snapshots/*",
        model_bucket_arn + f"/campaigns/{RUN_ID}/submissions/*",
        model_bucket_arn + f"/campaigns/{RUN_ID}/production/controller-baselines/*",
        model_bucket_arn + f"/campaigns/{RUN_ID}/monitor/must-start/production/*",
    ]
    all_fence_objects = bootstrap_objects + source_settled_objects
    pre_executor_role_name = "keep-glm52-h1g-pre-support-fence-executor"
    source_materializer_role_name = "keep-glm52-h1g-fence-source-settlement"
    source_publisher_role_name = (
        "keep-glm52-h1g-fence-source-settled-artifact-publisher"
    )
    materializer_role_arn = (
        f"arn:aws:iam::{ACCOUNT_ID}:role/{source_materializer_role_name}"
    )
    source_publisher_role_arn = (
        f"arn:aws:iam::{ACCOUNT_ID}:role/{source_publisher_role_name}"
    )

    def retained(resource: Mapping[str, object]) -> dict[str, object]:
        return {
            **deepcopy(resource),
            "DeletionPolicy": "Retain",
            "UpdateReplacePolicy": "Retain",
        }

    def lambda_trust() -> Mapping[str, object]:
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }

    def exact_role_trust(role_arn: str) -> Mapping[str, object]:
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": role_arn},
                    "Action": "sts:AssumeRole",
                    "Condition": {"ArnEquals": {"aws:PrincipalArn": role_arn}},
                }
            ],
        }

    def publisher_role(
        *,
        role_name: str,
        trusted_role_arn: str,
        object_arns: list[str],
    ) -> Mapping[str, object]:
        return retained(
            {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "RoleName": role_name,
                    "AssumeRolePolicyDocument": exact_role_trust(trusted_role_arn),
                    "Policies": [
                        {
                            "PolicyName": "WriteOnlyExactFenceArtifacts",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [
                                    {
                                        "Sid": "CreateOnlyExactFenceArtifacts",
                                        "Effect": "Allow",
                                        "Action": "s3:PutObject",
                                        "Resource": object_arns,
                                    },
                                    {
                                        "Sid": "EncryptOnlyExactFenceArtifacts",
                                        "Effect": "Allow",
                                        "Action": [
                                            "kms:Encrypt",
                                            "kms:GenerateDataKey",
                                        ],
                                        "Resource": inputs.retained_kms_key_arn,
                                        "Condition": {
                                            "StringEquals": {
                                                "kms:ViaService": (
                                                    "s3.us-west-2.amazonaws.com"
                                                )
                                            },
                                            "ForAnyValue:StringEquals": {
                                                "kms:EncryptionContext:aws:s3:arn": (
                                                    object_arns
                                                )
                                            },
                                        },
                                    },
                                ],
                            },
                        }
                    ],
                },
            }
        )

    artifact_read_resources = [
        *all_fence_objects,
        (model_bucket_arn + f"/campaigns/{RUN_ID}/authorities/fence/templates/*"),
        (model_bucket_arn + f"/campaigns/{RUN_ID}/authorities/fence/requests/*"),
        (model_bucket_arn + f"/campaigns/{RUN_ID}/authorities/fence/runtime/*"),
    ]
    pre_executor_statements = [
        {
            "Sid": "ReadOnlyPinnedFenceArtifacts",
            "Effect": "Allow",
            "Action": [
                "s3:GetObjectVersion",
                "s3:GetObjectVersionAttributes",
            ],
            "Resource": artifact_read_resources,
        },
        {
            "Sid": "InventoryOnlyFenceArtifactHistory",
            "Effect": "Allow",
            "Action": "s3:ListBucketVersions",
            "Resource": model_bucket_arn,
            "Condition": {
                "StringLike": {"s3:prefix": [f"campaigns/{RUN_ID}/authorities/fence/*"]}
            },
        },
        {
            "Sid": "CreateAndExecuteOnlyFenceStackTransitions",
            "Effect": "Allow",
            "Action": [
                "cloudformation:CreateChangeSet",
                "cloudformation:ExecuteChangeSet",
                "cloudformation:DescribeChangeSet",
                "cloudformation:ListChangeSets",
                "cloudformation:DescribeStacks",
                "cloudformation:GetTemplate",
            ],
            "Resource": inputs.fence_stack_id,
        },
        {
            "Sid": "PassOnlyFenceServiceRole",
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": inputs.fence_service_role_arn,
            "Condition": {
                "StringEquals": {"iam:PassedToService": "cloudformation.amazonaws.com"}
            },
        },
        {
            "Sid": "ReadAndConditionallyCommitFenceAuthority",
            "Effect": "Allow",
            "Action": [
                "dynamodb:GetItem",
                "dynamodb:PutItem",
                "dynamodb:TransactGetItems",
                "dynamodb:TransactWriteItems",
            ],
            "Resource": inputs.ledger_table_arn,
        },
    ]
    materializer_statements = [
        {
            "Sid": "ReadOnlyPinnedBootstrapAndSourceEvidence",
            "Effect": "Allow",
            "Action": [
                "s3:GetObjectVersion",
                "s3:GetObjectVersionAttributes",
            ],
            "Resource": source_materializer_read_objects,
        },
        {
            "Sid": "InventoryOnlySourceAndFenceHistory",
            "Effect": "Allow",
            "Action": "s3:ListBucketVersions",
            "Resource": model_bucket_arn,
            "Condition": {
                "StringLike": {
                    "s3:prefix": [
                        f"campaigns/{RUN_ID}/authorities/fence/*",
                        f"campaigns/{RUN_ID}/spend-snapshots/*",
                        f"campaigns/{RUN_ID}/submissions/*",
                        (f"campaigns/{RUN_ID}/production/controller-baselines/*"),
                        (f"campaigns/{RUN_ID}/monitor/must-start/production/*"),
                    ]
                }
            },
        },
        {
            "Sid": "DecryptOnlyPinnedBootstrapAndSourceEvidence",
            "Effect": "Allow",
            "Action": "kms:Decrypt",
            "Resource": inputs.retained_kms_key_arn,
            "Condition": {
                "StringEquals": {"kms:ViaService": "s3.us-west-2.amazonaws.com"},
                "ForAnyValue:StringLike": {
                    "kms:EncryptionContext:aws:s3:arn": (
                        source_materializer_read_objects
                    )
                },
            },
        },
        {
            "Sid": "ReadOnlySourceTerminalAuthority",
            "Effect": "Allow",
            "Action": [
                "dynamodb:GetItem",
                "dynamodb:Query",
                "dynamodb:TransactGetItems",
            ],
            "Resource": inputs.ledger_table_arn,
        },
        {
            "Sid": "AssumeOnlySourceSettledArtifactPublisher",
            "Effect": "Allow",
            "Action": "sts:AssumeRole",
            "Resource": source_publisher_role_arn,
        },
    ]
    code = {
        "S3Bucket": inputs.lambda_code_bucket,
        "S3Key": inputs.lambda_code_key,
        "S3ObjectVersion": inputs.lambda_code_version_id,
    }
    common_environment = {
        "GLM52_ACCOUNT_ID": ACCOUNT_ID,
        "GLM52_REGION": REGION,
        "GLM52_RUN_ID": RUN_ID,
        "GLM52_ACTIVATION_ID": inputs.activation_id,
        "GLM52_MODEL_BUCKET_ARN": model_bucket_arn,
        "GLM52_RETAINED_KMS_KEY_ARN": inputs.retained_kms_key_arn,
        "GLM52_BOOTSTRAP_MANIFEST_COORDINATE": canonical_json_bytes(
            manifest_coordinate
        ).decode("ascii"),
        "GLM52_SUPPORT_DEPLOYMENT_IDENTITY_SHA256": inputs.lambda_code_sha256,
    }
    resources = {
        "PreSupportFenceExecutorRole": retained(
            {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "RoleName": pre_executor_role_name,
                    "AssumeRolePolicyDocument": lambda_trust(),
                    "Policies": [
                        {
                            "PolicyName": "ExecuteOnlyRetainedFenceFailover",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": pre_executor_statements,
                            },
                        }
                    ],
                },
            }
        ),
        "PreSupportFenceExecutorFunction": retained(
            {
                "Type": "AWS::Lambda::Function",
                "Properties": {
                    "FunctionName": ("keep-glm52-h1g-pre-support-fence-executor"),
                    "Runtime": "python3.13",
                    "Handler": "support_fence_handler.main",
                    "Role": {"Fn::GetAtt": ["PreSupportFenceExecutorRole", "Arn"]},
                    "Code": code,
                    "Timeout": 12,
                    "MemorySize": 256,
                    "ReservedConcurrentExecutions": 1,
                    "Environment": {
                        "Variables": {
                            **common_environment,
                            "GLM52_AUTHORITY_CLASS": "RETAINED_PRE_SUPPORT",
                        }
                    },
                },
            }
        ),
        "PreSupportFenceExecutorVersion": retained(
            {
                "Type": "AWS::Lambda::Version",
                "Properties": {
                    "FunctionName": {"Ref": "PreSupportFenceExecutorFunction"},
                    "CodeSha256": inputs.lambda_code_sha256,
                },
            }
        ),
        "FenceSourceSettlementMaterializerRole": retained(
            {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "RoleName": source_materializer_role_name,
                    "AssumeRolePolicyDocument": lambda_trust(),
                    "Policies": [
                        {
                            "PolicyName": "MaterializeOnlySourceSettlement",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": materializer_statements,
                            },
                        }
                    ],
                },
            }
        ),
        "FenceSourceSettlementMaterializerFunction": retained(
            {
                "Type": "AWS::Lambda::Function",
                "Properties": {
                    "FunctionName": ("keep-glm52-h1g-fence-source-settlement"),
                    "Runtime": "python3.13",
                    "Handler": "support_fence_handler.source_settlement_main",
                    "Role": {
                        "Fn::GetAtt": [
                            "FenceSourceSettlementMaterializerRole",
                            "Arn",
                        ]
                    },
                    "Code": code,
                    "Timeout": 60,
                    "MemorySize": 256,
                    "ReservedConcurrentExecutions": 1,
                    "Environment": {
                        "Variables": {
                            **common_environment,
                            "GLM52_AUTHORITY_CLASS": ("SOURCE_SETTLEMENT_MATERIALIZER"),
                            "GLM52_SOURCE_PUBLISHER_ROLE_ARN": (
                                source_publisher_role_arn
                            ),
                        }
                    },
                },
            }
        ),
        "FenceSourceSettlementMaterializerVersion": retained(
            {
                "Type": "AWS::Lambda::Version",
                "Properties": {
                    "FunctionName": {
                        "Ref": "FenceSourceSettlementMaterializerFunction"
                    },
                    "CodeSha256": inputs.lambda_code_sha256,
                },
            }
        ),
        "FenceSourceSettledArtifactPublisherRole": publisher_role(
            role_name=source_publisher_role_name,
            trusted_role_arn=materializer_role_arn,
            object_arns=source_settled_objects,
        ),
    }
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Metadata": {
            "FenceRuntimeRecordType": "glm52_h1g_retained_fence_runtime_v2",
            "SupportDeletionSurvivors": sorted(resources),
        },
        "Outputs": {
            "PreSupportFenceExecutorVersionArn": {
                "Value": {"Ref": "PreSupportFenceExecutorVersion"}
            },
            "FenceSourceSettlementMaterializerVersionArn": {
                "Value": {"Ref": "FenceSourceSettlementMaterializerVersion"}
            },
        },
        "Resources": resources,
    }


def build_retained_fence_runtime_fragment(
    inputs: RetainedFenceRuntimeInputs,
) -> Mapping[str, object]:
    if type(inputs) is not RetainedFenceRuntimeInputs:
        raise TypeError("retained fence runtime requires exact manifest-bound inputs")
    return _build_h1g_retained_fence_fragment(inputs)


def build_pre_support_retained_runtime_fragment(
    inputs: object,
) -> Mapping[str, object]:
    """Build Task10/Task12 retained runtime before support creation."""

    if type(inputs) not in {
        PreSupportRuntimeInputs,
        SupportBuildInputs,
        SupportInputs,
    }:
        raise TypeError(
            "pre-support retained runtime requires exact authenticated inputs"
        )
    task12 = _build_task12_retained_fragment(inputs)
    task10 = _build_task10_retained_fragment(inputs)
    fence = (
        _build_h1g_retained_fence_fragment(inputs)
        if type(inputs) is SupportBuildInputs
        else {"Outputs": {}, "Metadata": {}, "Resources": {}}
    )
    fragments = (task12, task10, fence)
    resources_seen: set[str] = set()
    parameters_seen: set[str] = set()
    for retained_fragment in fragments:
        resource_overlap = resources_seen & set(retained_fragment.get("Resources", {}))
        parameter_overlap = parameters_seen & set(
            retained_fragment.get("Parameters", {})
        )
        if resource_overlap or parameter_overlap:
            raise ValueError("retained runtime resource or parameter collision")
        resources_seen.update(retained_fragment.get("Resources", {}))
        parameters_seen.update(retained_fragment.get("Parameters", {}))
    fragment = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Outputs": {
            "Task12TerminalV2VersionArn": {
                "Value": {"Ref": "TerminalV2Version"},
                "Export": {"Name": "KeepGlm52Task12TerminalV2VersionArn"},
            },
            "Task12WorkerDrainVersionArn": {
                "Value": {"Ref": "WorkerDrainVersion"},
                "Export": {"Name": "KeepGlm52Task12WorkerDrainVersionArn"},
            },
            "Task9LiabilityWatcherVersionArn": {
                "Value": {"Ref": "Task9LiabilityWatcherFunctionVersion"},
                "Export": {"Name": "KeepGlm52Task9LiabilityWatcherVersionArn"},
            },
            **deepcopy(fence.get("Outputs", {})),
        },
        "Metadata": {
            "RecordType": "glm52_h1g_retained_support_augmentation_v1",
            "RetainedStackId": inputs.retained_stack_id,
            "SupportImportsRetainedOnly": [
                name
                for name in inputs.retained_export_names
                if name
                not in {
                    "KeepGlm52Task12TerminalV2VersionArn",
                    "KeepGlm52Task12WorkerDrainVersionArn",
                    "KeepGlm52Task9LiabilityWatcherVersionArn",
                }
            ],
            "RetainedImportsSupport": [],
            **deepcopy(task12["Metadata"]),
            **deepcopy(task10["Metadata"]),
            **deepcopy(fence["Metadata"]),
        },
        "Resources": {
            **deepcopy(task12["Resources"]),
            **deepcopy(task10["Resources"]),
            **deepcopy(fence["Resources"]),
        },
    }
    parameters = {
        **deepcopy(task12.get("Parameters", {})),
        **deepcopy(task10.get("Parameters", {})),
        **deepcopy(fence.get("Parameters", {})),
    }
    if parameters:
        fragment["Parameters"] = parameters
    return fragment


def build_postcreate_retained_support_fragment(
    inputs: SupportInputs,
) -> Mapping[str, object]:
    """Build only physical-ID-dependent deletion and lifecycle resources."""

    if type(inputs) is not SupportInputs:
        raise TypeError("postcreate retained support requires SupportInputs")

    from .fence_executor import (
        build_support_deletion_workflow_definition,
        support_deletion_workflow_history_event_ceiling,
    )

    grace = bootstrap_grace_contract(inputs)
    nat_schedule_name = _nat_accumulator_schedule_name(inputs)
    nat_schedule_end_date = _nat_accumulator_schedule_end_date(inputs)
    nat_schedule_arn = _nat_accumulator_schedule_arn(inputs)
    workflow_definition = build_support_deletion_workflow_definition()
    workflow_definition_sha256 = hashlib.sha256(
        canonical_json_bytes(workflow_definition)
    ).hexdigest()
    inventory = inputs.support_deletion_inventory
    stack_rows = inventory.get("stack_resources") if type(inventory) is dict else None
    physical_by_logical = (
        {
            row["logical_id"]: row["physical_id"]
            for row in stack_rows
            if (
                type(row) is dict
                and type(row.get("logical_id")) is str
                and type(row.get("physical_id")) is str
            )
        }
        if type(stack_rows) is list
        else {}
    )
    if physical_by_logical:
        host_instance_id = physical_by_logical.get("CombinedHost")
        host_security_group_id = physical_by_logical.get("CombinedHostSecurityGroup")
    else:
        action_resources = (
            inventory.get("action_resources") if type(inventory) is dict else None
        )
        termination_resources = (
            action_resources.get("ec2:TerminateInstances")
            if type(action_resources) is dict
            else None
        )
        egress_resources = (
            action_resources.get("ec2:RevokeSecurityGroupEgress")
            if type(action_resources) is dict
            else None
        )
        host_instance_id = (
            termination_resources[0].rsplit("/", 1)[-1]
            if type(termination_resources) is list
            and len(termination_resources) == 1
            and type(termination_resources[0]) is str
            else None
        )
        host_security_group_id = (
            sorted(egress_resources)[0].rsplit("/", 1)[-1]
            if type(egress_resources) is list
            and egress_resources
            and all(type(value) is str for value in egress_resources)
            else None
        )
    if (
        type(host_instance_id) is not str
        or re.fullmatch(r"i-[0-9a-f]{17}", host_instance_id) is None
        or type(host_security_group_id) is not str
        or re.fullmatch(r"sg-[0-9a-f]{17}", host_security_group_id) is None
    ):
        raise ValueError("retained fail-closed host coordinates are absent")

    lifecycle_statements = [
        {
            "Sid": "ReadExactSupportHostAndEgress",
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeInstances",
                "ec2:DescribeSecurityGroups",
            ],
            "Resource": "*",
            "Condition": {"StringEquals": {"aws:RequestedRegion": REGION}},
        },
        {
            "Sid": "StopExactTaggedSupportHost",
            "Effect": "Allow",
            "Action": "ec2:StopInstances",
            "Resource": f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:instance/*",
            "Condition": {
                "StringEquals": {
                    "ec2:ResourceTag/ActivationId": inputs.activation_id,
                    "ec2:ResourceTag/RunId": RUN_ID,
                    "ec2:ResourceTag/Purpose": "combined-host",
                }
            },
        },
        {
            "Sid": "DisableExactTaggedSupportEgress",
            "Effect": "Allow",
            "Action": "ec2:RevokeSecurityGroupEgress",
            "Resource": (f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:security-group/*"),
            "Condition": {
                "StringEquals": {
                    "ec2:ResourceTag/ActivationId": inputs.activation_id,
                    "ec2:ResourceTag/RunId": RUN_ID,
                    "ec2:ResourceTag/Purpose": "combined-host-egress",
                }
            },
        },
        {
            "Sid": "ReadExactSupportStackOnly",
            "Effect": "Allow",
            "Action": "cloudformation:DescribeStacks",
            "Resource": inputs.support_stack_id,
        },
        {
            "Sid": "WriteAndReadExactDeadlineControl",
            "Effect": "Allow",
            "Action": ["dynamodb:PutItem", "dynamodb:GetItem"],
            "Resource": inputs.ledger_table_arn,
            "Condition": {
                "ForAllValues:StringEquals": {"dynamodb:LeadingKeys": [f"RUN#{RUN_ID}"]}
            },
        },
        {
            "Sid": "PageExactOperatorTopic",
            "Effect": "Allow",
            "Action": "sns:Publish",
            "Resource": SUPPORT_OPERATOR_TOPIC_ARN,
        },
        {
            "Sid": "ReadActivationCumulativeNatCounters",
            "Effect": "Allow",
            "Action": "cloudwatch:GetMetricData",
            "Resource": "*",
        },
        {
            "Sid": "PublishExactActivationCumulativeNatMetric",
            "Effect": "Allow",
            "Action": "cloudwatch:PutMetricData",
            "Resource": "*",
            "Condition": {"StringEquals": {"cloudwatch:namespace": "GLM52/H1g"}},
        },
        {
            "Sid": "RetireExactNatObservationSchedule",
            "Effect": "Allow",
            "Action": [
                "scheduler:GetSchedule",
                "scheduler:UpdateSchedule",
                "scheduler:DeleteSchedule",
            ],
            "Resource": nat_schedule_arn,
        },
        {
            "Sid": "StartExactRetainedLifecycleVersion",
            "Effect": "Allow",
            "Action": "states:StartExecution",
            "Resource": {"Ref": "H1gRetainedLifecycleStateMachineVersion"},
        },
    ]
    workflow_statements = [
        {
            "Sid": "MutateAndReadExactSupportStackOnly",
            "Effect": "Allow",
            "Action": [
                "cloudformation:DescribeStacks",
                "cloudformation:UpdateTerminationProtection",
                "cloudformation:DeleteStack",
            ],
            "Resource": inputs.support_stack_id,
        },
        {
            "Sid": "PassExactSupportDeletionRole",
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": SUPPORT_DELETION_ROLE_ARN,
            "Condition": {
                "StringEquals": {"iam:PassedToService": "cloudformation.amazonaws.com"}
            },
        },
    ]

    def retained(
        resource_type: str, properties: Mapping[str, object]
    ) -> dict[str, object]:
        return {
            "Type": resource_type,
            "DeletionPolicy": "Retain",
            "UpdateReplacePolicy": "Retain",
            "Properties": dict(properties),
        }

    resources: dict[str, object] = {
        "H1gRetainedSupportOperatorTopic": retained(
            "AWS::SNS::Topic",
            {
                "TopicName": "keep-glm52-h1g-operator-alerts",
                "Tags": list(_tags(inputs, "retained-support-alerts")),
            },
        ),
        "H1gSupportDeletionServiceRole": retained(
            "AWS::IAM::Role",
            {
                **_service_role(
                    service="cloudformation.amazonaws.com",
                    policies=(
                        {
                            "PolicyName": "DeleteExactSupportInventoryOnly",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": list(
                                    _support_deletion_policy_statements(inputs)
                                ),
                            },
                        },
                    ),
                ),
                "RoleName": "keep-glm52-h1g-support-deletion",
            },
        ),
        "H1gRetainedLifecycleStateMachineRole": retained(
            "AWS::IAM::Role",
            {
                **_service_role(
                    service="states.amazonaws.com",
                    policies=(
                        {
                            "PolicyName": ("ExecuteExactRetainedSupportLifecycleOnly"),
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": workflow_statements,
                            },
                        },
                    ),
                ),
                "RoleName": ("keep-glm52-h1g-retained-lifecycle-workflow"),
            },
        ),
        "H1gRetainedLifecycleStateMachine": retained(
            "AWS::StepFunctions::StateMachine",
            {
                "Definition": workflow_definition,
                "RoleArn": {
                    "Fn::GetAtt": [
                        "H1gRetainedLifecycleStateMachineRole",
                        "Arn",
                    ]
                },
                "StateMachineName": ("keep-glm52-h1g-retained-lifecycle"),
                "StateMachineType": "STANDARD",
                "Tags": list(_tags(inputs, "retained-support-lifecycle")),
            },
        ),
        "H1gRetainedLifecycleStateMachineVersion": retained(
            "AWS::StepFunctions::StateMachineVersion",
            {
                "Description": (f"definition={workflow_definition_sha256}"),
                "StateMachineArn": {"Ref": "H1gRetainedLifecycleStateMachine"},
            },
        ),
        "H1gRetainedSupportLifecycleRole": retained(
            "AWS::IAM::Role",
            {
                **_service_role(
                    service="lambda.amazonaws.com",
                    policies=(
                        {
                            "PolicyName": "EnforceSupportLifecycleOnly",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": lifecycle_statements,
                            },
                        },
                    ),
                ),
                "RoleName": "keep-glm52-h1g-retained-support-lifecycle",
            },
        ),
        "H1gRetainedSupportLifecycleFunction": retained(
            "AWS::Lambda::Function",
            {
                "Code": {
                    "S3Bucket": inputs.lambda_code_bucket,
                    "S3Key": inputs.lambda_code_key,
                    "S3ObjectVersion": inputs.lambda_code_version_id,
                },
                "Environment": {
                    "Variables": {
                        "GLM52_ACTIVATION_ID": inputs.activation_id,
                        "GLM52_RUN_ID": RUN_ID,
                        "GLM52_SUPPORT_STACK_ID": inputs.support_stack_id,
                        "GLM52_SUPPORT_DELETION_ROLE_ARN": (SUPPORT_DELETION_ROLE_ARN),
                        "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
                        "GLM52_OPERATOR_TOPIC_ARN": (SUPPORT_OPERATOR_TOPIC_ARN),
                        "GLM52_RETAINED_LIFECYCLE_VERSION_ARN": {
                            "Ref": ("H1gRetainedLifecycleStateMachineVersion")
                        },
                        "GLM52_WORK_STOP_HOURS": "68",
                        "GLM52_DELETE_REQUEST_HOURS": "71",
                        "GLM52_ABSENCE_EXPECTED_HOURS": "72",
                        "GLM52_POST_72H_RECONCILIATIONS": "1",
                        "GLM52_DIRECT_CHILD_DELETE_ACTIONS": "FORBIDDEN",
                        "GLM52_BOOTSTRAP_GRACE_STARTED_AT": (grace["grace_started_at"]),
                        "GLM52_BOOTSTRAP_GRACE_DEADLINE": (grace["grace_deadline"]),
                        "GLM52_BOOTSTRAP_GRACE_SECONDS": str(
                            grace["grace_duration_seconds"]
                        ),
                        "GLM52_BOOTSTRAP_GRACE_IDENTITY_SHA256": (
                            grace["canonical_body_sha256"]
                        ),
                        "GLM52_MISSING_METRIC_AFTER_GRACE": "DRAIN",
                        "GLM52_WORK_AUTHORIZATION": "METRICS_OK_ONLY",
                        "GLM52_NAT_GATEWAY_ID": inputs.nat_gateway_id,
                        "GLM52_SUPPORT_HOST_INSTANCE_ID": (host_instance_id),
                        "GLM52_SUPPORT_HOST_SECURITY_GROUP_ID": (
                            host_security_group_id
                        ),
                        "GLM52_ACTIVATION_STARTED_AT": (inputs.activation_started_at),
                        "GLM52_NAT_COUNTER_METRICS": (
                            "BytesInFromSource,BytesOutToDestination,"
                            "BytesInFromDestination,BytesOutToSource"
                        ),
                        "GLM52_NAT_ACCUMULATOR_MODE": (
                            "ACTIVATION_CUMULATIVE_FOUR_COUNTERS_60S"
                        ),
                        "GLM52_NAT_ACCUMULATOR_MAX_WINDOWS": str(
                            _NAT_ACCUMULATOR_MAX_WINDOWS
                        ),
                        "GLM52_NAT_SCHEDULE_NAME": nat_schedule_name,
                        "GLM52_NAT_SCHEDULE_GROUP": "default",
                        "GLM52_NAT_SCHEDULE_END_DATE": (nat_schedule_end_date),
                    }
                },
                "Handler": "retained_support_lifecycle_handler.main",
                "MemorySize": 256,
                "ReservedConcurrentExecutions": 1,
                "Role": {
                    "Fn::GetAtt": [
                        "H1gRetainedSupportLifecycleRole",
                        "Arn",
                    ]
                },
                "Runtime": "python3.12",
                "Timeout": 20,
            },
        ),
        "H1gRetainedSupportLifecycleVersion": retained(
            "AWS::Lambda::Version",
            {
                "Description": f"code={inputs.lambda_code_sha256}",
                "FunctionName": {"Ref": "H1gRetainedSupportLifecycleFunction"},
            },
        ),
        "H1gRetainedSupportScheduleRole": retained(
            "AWS::IAM::Role",
            {
                **_service_role(
                    service="scheduler.amazonaws.com",
                    policies=(
                        {
                            "PolicyName": (
                                "InvokeExactRetainedSupportLifecycleVersion"
                            ),
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [
                                    {
                                        "Effect": "Allow",
                                        "Action": "lambda:InvokeFunction",
                                        "Resource": {
                                            "Ref": (
                                                "H1gRetainedSupportLifecycleVersion"
                                            )
                                        },
                                    }
                                ],
                            },
                        },
                    ),
                ),
                "RoleName": ("keep-glm52-h1g-retained-support-schedule-invoke"),
            },
        ),
    }
    start = datetime.fromisoformat(inputs.activation_started_at.replace("Z", "+00:00"))
    for logical_id, hours, payload in (
        ("H1gRetainedWorkStopSchedule", 68, "WORK_STOP"),
        ("H1gRetainedDeleteRequestSchedule", 71, "DELETE_REQUEST"),
        ("H1gRetainedAbsenceDeadlineSchedule", 72, "ABSENCE_EXPECTED"),
    ):
        at = (start + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
        resources[logical_id] = retained(
            "AWS::Scheduler::Schedule",
            {
                "FlexibleTimeWindow": {"Mode": "OFF"},
                "ScheduleExpression": f"at({at})",
                "ScheduleExpressionTimezone": "UTC",
                "State": "ENABLED",
                "Target": {
                    "Arn": {"Ref": "H1gRetainedSupportLifecycleVersion"},
                    "Input": (
                        '{"activation_id":"'
                        f'{inputs.activation_id}","deadline":"{payload}"'
                        "}"
                    ),
                    "RetryPolicy": {
                        "MaximumEventAgeInSeconds": 60,
                        "MaximumRetryAttempts": 0,
                    },
                    "RoleArn": {
                        "Fn::GetAtt": [
                            "H1gRetainedSupportScheduleRole",
                            "Arn",
                        ]
                    },
                },
            },
        )
    resources["H1gRetainedNatAccumulatorSchedule"] = retained(
        "AWS::Scheduler::Schedule",
        {
            "ActionAfterCompletion": "DELETE",
            "EndDate": nat_schedule_end_date,
            "FlexibleTimeWindow": {"Mode": "OFF"},
            "GroupName": "default",
            "Name": nat_schedule_name,
            "ScheduleExpression": "rate(1 minute)",
            "State": "ENABLED",
            "Target": {
                "Arn": {"Ref": "H1gRetainedSupportLifecycleVersion"},
                "Input": (
                    '{"action":"ACCUMULATE_NAT_60S","activation_id":"'
                    f'{inputs.activation_id}","activation_started_at":"'
                    f'{inputs.activation_started_at}","nat_gateway_id":"'
                    f'{inputs.nat_gateway_id}"'
                    "}"
                ),
                "RetryPolicy": {
                    "MaximumEventAgeInSeconds": 60,
                    "MaximumRetryAttempts": 0,
                },
                "RoleArn": {
                    "Fn::GetAtt": [
                        "H1gRetainedSupportScheduleRole",
                        "Arn",
                    ]
                },
            },
        },
    )
    resources["H1gRetainedNatObservationMissingAlarm"] = retained(
        "AWS::CloudWatch::Alarm",
        {
            "AlarmName": (
                f"keep-glm52-h1g-{inputs.activation_id}-"
                "nat-observation-missing-after-grace"
            ),
            "ComparisonOperator": "GreaterThanThreshold",
            "DatapointsToAlarm": 16,
            "Dimensions": [
                {
                    "Name": "ActivationId",
                    "Value": inputs.activation_id,
                },
                {
                    "Name": "NatGatewayId",
                    "Value": inputs.nat_gateway_id,
                },
            ],
            "EvaluationPeriods": 16,
            "MetricName": "ActivationCumulativeNatProcessedBytes",
            "Namespace": "GLM52/H1g",
            "Period": 60,
            "Statistic": "Maximum",
            "Threshold": 10**30,
            "TreatMissingData": "breaching",
        },
    )
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Metadata": {
            "SupportStackId": inputs.support_stack_id,
            "DirectChildDeletionByLifecycle": False,
            "ActiveFinalizationOwnerMax": 1,
            "FinalizationExecutionsMax": 12,
            "NatGatewayId": inputs.nat_gateway_id,
            "NatObservationMode": ("ACTIVATION_CUMULATIVE_FOUR_COUNTERS_60S"),
            "NatObservationMaximumWindows": _NAT_ACCUMULATOR_MAX_WINDOWS,
            "NatAccumulatorScheduleArn": nat_schedule_arn,
            "NatAccumulatorScheduleEndDate": nat_schedule_end_date,
            "RetainedLifecycleWorkflowType": "STANDARD",
            "RetainedLifecycleDefinitionSha256": (workflow_definition_sha256),
            "RetainedLifecycleHistoryEventCeiling": (
                support_deletion_workflow_history_event_ceiling()
            ),
            "BootstrapGraceIdentitySha256": (grace["canonical_body_sha256"]),
            **_support_deletion_inventory_metadata(inputs.support_deletion_inventory),
        },
        "Resources": resources,
    }


def _build_retained_augmentation(inputs: SupportInputs) -> Mapping[str, object]:
    """Backward-compatible composition of pre- and post-support fragments."""

    if type(inputs) is not SupportInputs:
        raise TypeError("retained augmentation requires exact support inputs")
    base = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": ("GLM-5.2 retained support lifecycle augmentation fragment"),
        "Parameters": {},
        "Outputs": {},
        "Metadata": {},
        "Resources": {},
    }
    return compose_complete_retained_template(
        base,
        build_pre_support_retained_runtime_fragment(inputs),
        build_postcreate_retained_support_fragment(inputs),
    )


def _template_resources(
    template: Mapping[str, object], *, label: str
) -> Mapping[str, Mapping[str, object]]:
    if type(template) is not dict or type(template.get("Resources")) is not dict:
        raise ValueError(f"{label} lacks one exact resource graph")
    resources = template["Resources"]
    if any(type(item) is not dict for item in resources.values()):
        raise ValueError(f"{label} contains a malformed resource")
    return resources


def _statement_actions(statement: Mapping[str, object]) -> tuple[str, ...]:
    action = statement.get("Action")
    if type(action) is str:
        return (action,)
    if type(action) is list and action and all(type(item) is str for item in action):
        return tuple(action)
    raise ValueError("IAM statement actions are not exact")


def validate_support_security_graph(
    template: Mapping[str, object], inputs: SupportInputs
) -> bool:
    """Independently validate endpoint, reader, secret-target, and host edges."""

    resources = _template_resources(template, label="support template")
    counts = support_resource_cardinality(template)
    if (
        counts.get("AWS::EC2::VPCEndpoint") != 3
        or counts.get("AWS::SecretsManager::Secret") != 8
        or counts.get("AWS::EC2::Instance") != 1
        or any(
            resource.get("Type") == "AWS::SecretsManager::RotationSchedule"
            for resource in resources.values()
        )
        or any(
            resource.get("Properties", {}).get("ServiceName")
            == f"com.amazonaws.{REGION}.kms"
            for resource in resources.values()
        )
    ):
        raise ValueError("support endpoint/secret/host cardinality is unsafe")
    endpoint = resources.get("SecretsManagerEndpoint")
    endpoint_properties = (
        endpoint.get("Properties", {}) if type(endpoint) is dict else {}
    )
    if (
        endpoint.get("Type") != "AWS::EC2::VPCEndpoint"
        if type(endpoint) is dict
        else True
    ):
        raise ValueError("Secrets Manager endpoint is absent")
    if (
        endpoint_properties.get("ServiceName")
        != f"com.amazonaws.{REGION}.secretsmanager"
        or endpoint_properties.get("VpcEndpointType") != "Interface"
        or endpoint_properties.get("PrivateDnsEnabled") is not True
        or endpoint_properties.get("SubnetIds")
        != [
            {"Ref": "PrimaryIsolatedSubnet"},
            {"Ref": "AlternateIsolatedSubnet"},
        ]
        or endpoint_properties.get("SecurityGroupIds")
        != [{"Ref": "SecretsEndpointSecurityGroup"}]
    ):
        raise ValueError("Secrets Manager endpoint topology is not exact")
    statements = endpoint_properties.get("PolicyDocument", {}).get("Statement")
    if type(statements) is not list:
        raise ValueError("Secrets Manager endpoint policy is absent")
    allows = [item for item in statements if item.get("Effect") == "Allow"]
    denies = [item for item in statements if item.get("Effect") == "Deny"]
    expected_allows = []
    for prefix, (role, _client_sg, secret_ids) in VPC_SECRET_READERS.items():
        for logical_id in secret_ids:
            expected_allows.extend(
                [
                    {
                        "Sid": f"{prefix}{logical_id}GetExactStage",
                        "Effect": "Allow",
                        "Principal": {"AWS": {"Fn::GetAtt": [role, "Arn"]}},
                        "Action": "secretsmanager:GetSecretValue",
                        "Resource": {"Ref": logical_id},
                        "Condition": {
                            "StringEquals": {
                                "secretsmanager:VersionStage": _secret_stage(
                                    inputs, logical_id
                                )
                            }
                        },
                    },
                    {
                        "Sid": f"{prefix}{logical_id}ListExactSecret",
                        "Effect": "Allow",
                        "Principal": {"AWS": {"Fn::GetAtt": [role, "Arn"]}},
                        "Action": "secretsmanager:ListSecretVersionIds",
                        "Resource": {"Ref": logical_id},
                    },
                ]
            )
    for logical_id in REHEARSAL_PROBE_SECRET_IDS:
        expected_allows.extend(
            [
                {
                    "Sid": f"RehearsalProbe{logical_id}GetExactStage",
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": {
                            "Fn::GetAtt": [
                                "RehearsalProbeRole",
                                "Arn",
                            ]
                        }
                    },
                    "Action": "secretsmanager:GetSecretValue",
                    "Resource": {"Ref": logical_id},
                    "Condition": {
                        "StringEquals": {
                            "secretsmanager:VersionStage": _secret_stage(
                                inputs, logical_id
                            )
                        }
                    },
                },
                {
                    "Sid": f"RehearsalProbe{logical_id}ListExactSecret",
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": {
                            "Fn::GetAtt": [
                                "RehearsalProbeRole",
                                "Arn",
                            ]
                        }
                    },
                    "Action": "secretsmanager:ListSecretVersionIds",
                    "Resource": {"Ref": logical_id},
                },
            ]
        )
    if allows != expected_allows or denies != [
        {
            "Sid": "DenyCrossRunSecretAuthority",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "secretsmanager:*",
            "NotResource": [{"Ref": logical_id} for logical_id in SECRET_LOGICAL_IDS],
        }
    ]:
        raise ValueError("endpoint reader/stage/cross-run policy is not exact")
    for prefix, (_role, client_sg, _secret_ids) in VPC_SECRET_READERS.items():
        if resources.get(f"{prefix}SecretsEndpointEgress") != {
            "Type": "AWS::EC2::SecurityGroupEgress",
            "Properties": {
                "DestinationSecurityGroupId": {"Ref": "SecretsEndpointSecurityGroup"},
                "FromPort": 443,
                "GroupId": {"Ref": client_sg},
                "IpProtocol": "tcp",
                "ToPort": 443,
            },
        } or resources.get(f"{prefix}SecretsEndpointIngress") != {
            "Type": "AWS::EC2::SecurityGroupIngress",
            "Properties": {
                "FromPort": 443,
                "GroupId": {"Ref": "SecretsEndpointSecurityGroup"},
                "IpProtocol": "tcp",
                "SourceSecurityGroupId": {"Ref": client_sg},
                "ToPort": 443,
            },
        }:
            raise ValueError("isolated reader lacks its exact endpoint 443 path")
    for path in CLIENT_PATHS.values():
        client_sg = path["client_security_group"]
        port = path["port"](inputs)
        if resources.get(path["egress_logical_id"]) != {
            "Type": "AWS::EC2::SecurityGroupEgress",
            "Properties": {
                "DestinationSecurityGroupId": {"Ref": "CombinedHostSecurityGroup"},
                "FromPort": port,
                "GroupId": {"Ref": client_sg},
                "IpProtocol": "tcp",
                "ToPort": port,
            },
        } or resources.get(path["ingress_logical_id"]) != {
            "Type": "AWS::EC2::SecurityGroupIngress",
            "Properties": {
                "FromPort": port,
                "GroupId": {"Ref": "CombinedHostSecurityGroup"},
                "IpProtocol": "tcp",
                "SourceSecurityGroupId": {"Ref": client_sg},
                "ToPort": port,
            },
        }:
            raise ValueError("relay client lacks its exact host listener path")
    for custom_resource, logical_ids in (
        ("SkyBootstrapCustomResource", SECRET_LOGICAL_IDS[:2]),
        ("TlsBundleCustomResource", SECRET_LOGICAL_IDS[2:]),
    ):
        resource = resources.get(custom_resource, {})
        if resource.get("Properties", {}).get("SecretTargets") != _secret_targets(
            inputs, logical_ids
        ):
            raise ValueError("custom-resource secret targets are incomplete")
    handler_policies = (
        resources.get("TlsHandlerRole", {}).get("Properties", {}).get("Policies")
    )
    if type(handler_policies) is not list or len(handler_policies) != 1:
        raise ValueError("TLS handler policy inventory is not exact")
    handler_statements = handler_policies[0]["PolicyDocument"]["Statement"]
    observed_puts = {
        (
            statement.get("Resource", {}).get("Ref"),
            statement.get("Condition", {})
            .get("StringEquals", {})
            .get("secretsmanager:VersionStage"),
        )
        for statement in handler_statements
        if statement.get("Action") == "secretsmanager:PutSecretValue"
    }
    expected_puts = {
        (logical_id, _secret_stage(inputs, logical_id))
        for logical_id in SECRET_LOGICAL_IDS
    }
    if observed_puts != expected_puts:
        raise ValueError("handler cannot write exactly all eight staged secrets")
    host = resources.get("CombinedHost", {}).get("Properties", {})
    if (
        host.get("InstanceType") != "c6a.xlarge"
        or host.get("MetadataOptions", {}).get("HttpTokens") != "required"
        or host.get("NetworkInterfaces", [{}])[0].get("AssociatePublicIpAddress")
        is not False
        or host.get("UserData") != _host_user_data(inputs)
    ):
        raise ValueError("combined host shape or version-bound first boot changed")
    return True


_FORBIDDEN_LIFECYCLE_ACTION_PREFIXES = (
    "ec2:Delete",
    "ec2:Terminate",
    "lambda:Delete",
    "secretsmanager:Delete",
    "kms:ScheduleKeyDeletion",
    "cloudformation:UpdateStack",
)


def _validate_lifecycle_role_statements(
    statements: object,
    inputs: object,
    *,
    retained_owner: bool,
) -> None:
    if type(statements) is not list or not statements:
        raise ValueError("support lifecycle role has no executable policy")
    actions = {
        action for statement in statements for action in _statement_actions(statement)
    }
    common = {
        "ec2:StopInstances",
        "ec2:RevokeSecurityGroupEgress",
        "cloudformation:DescribeStacks",
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "sns:Publish",
    }
    retained_only = {
        "cloudformation:UpdateTerminationProtection",
        "cloudformation:DeleteStack",
        "iam:PassRole",
    }
    continuation_only = {
        "dynamodb:UpdateItem",
        "dynamodb:Query",
        "s3:ListBucketVersions",
        "s3:GetObjectVersion",
        "s3:GetObjectVersionAttributes",
        "kms:Decrypt",
        "lambda:InvokeFunction",
        "events:PutEvents",
        "ec2:DescribeInstances",
        "ec2:TerminateInstances",
        "ssm:DescribeInstanceInformation",
        "states:GetExecutionHistory",
    }
    required = common | retained_only if retained_owner else common | continuation_only
    if actions != required or any(
        action.startswith(_FORBIDDEN_LIFECYCLE_ACTION_PREFIXES)
        for statement in statements
        if statement.get("Effect") == "Allow"
        for action in _statement_actions(statement)
    ):
        raise ValueError("lifecycle role cannot enforce the closed outcome")
    stack_read = [
        statement
        for statement in statements
        if "cloudformation:DescribeStacks" in _statement_actions(statement)
    ]
    if (
        len(stack_read) != 1
        or stack_read[0].get("Resource")
        != (inputs.support_stack_id if retained_owner else {"Ref": "AWS::StackId"})
        or any(
            type(statement.get("Resource")) is str
            and statement.get("Resource")
            in {inputs.retained_stack_id, "keep-glm52-h1g-fence"}
            for statement in statements
        )
    ):
        raise ValueError("lifecycle role stack target is foreign")
    if not retained_owner:
        termination_denies = [
            statement
            for statement in statements
            if "ec2:TerminateInstances" in _statement_actions(statement)
        ]
        exact_invoke = [
            statement
            for statement in statements
            if statement.get("Sid") == "InvokeExactContinuationCallees"
        ]
        if termination_denies != [
            {
                "Sid": "DenyUnguardedInstanceTermination",
                "Effect": "Deny",
                "Action": "ec2:TerminateInstances",
                "Resource": "*",
            }
        ] or exact_invoke != [
            {
                "Sid": "InvokeExactContinuationCallees",
                "Effect": "Allow",
                "Action": "lambda:InvokeFunction",
                "Resource": [
                    {"Ref": "NumericBindingVersion"},
                    {"Ref": "RetainedCancellationVersion"},
                    {"Fn::ImportValue": ("KeepGlm52Task12WorkerDrainVersionArn")},
                    {"Fn::ImportValue": ("KeepGlm52Task12TerminalV2VersionArn")},
                    {"Fn::ImportValue": ("KeepGlm52Task9LiabilityWatcherVersionArn")},
                ],
            }
        ]:
            raise ValueError("support continuation drain authority is not exact")
        exact_history = [
            statement
            for statement in statements
            if statement.get("Sid") == "ReadExactRetainedLifecycleExecutionHistory"
        ]
        if exact_history != [
            {
                "Sid": "ReadExactRetainedLifecycleExecutionHistory",
                "Effect": "Allow",
                "Action": "states:GetExecutionHistory",
                "Resource": (
                    f"arn:aws:states:{REGION}:{ACCOUNT_ID}:"
                    "execution:keep-glm52-h1g-retainedlifecycle:*"
                ),
            }
        ]:
            raise ValueError("support continuation history authority is not exact")
        return
    update = [
        statement
        for statement in statements
        if "cloudformation:UpdateTerminationProtection" in _statement_actions(statement)
    ]
    delete = [
        statement
        for statement in statements
        if "cloudformation:DeleteStack" in _statement_actions(statement)
    ]
    pass_role = [
        statement
        for statement in statements
        if "iam:PassRole" in _statement_actions(statement)
    ]
    if (
        len(update) != 1
        or update[0].get("Resource") != inputs.support_stack_id
        or len(delete) != 1
        or delete[0].get("Resource") != inputs.support_stack_id
        or len(pass_role) != 1
        or pass_role[0].get("Resource") != SUPPORT_DELETION_ROLE_ARN
        or pass_role[0].get("Condition")
        != {"StringEquals": {"iam:PassedToService": "cloudformation.amazonaws.com"}}
    ):
        raise ValueError("retained lifecycle deletion authority is foreign")


def validate_support_lifecycle_resources(
    template: Mapping[str, object], inputs: object
) -> bool:
    """Independently validate concrete deadlines, metrics, and effect authority."""

    resources = _template_resources(template, label="support template")
    policies = (
        resources.get("SupportDeadlineRole", {}).get("Properties", {}).get("Policies")
    )
    if type(policies) is not list or len(policies) != 1:
        raise ValueError("support deadline role policy is not exact")
    _validate_lifecycle_role_statements(
        policies[0]["PolicyDocument"]["Statement"],
        inputs,
        retained_owner=False,
    )
    variables = (
        resources.get("SupportDeadlineFunction", {})
        .get("Properties", {})
        .get("Environment", {})
        .get("Variables")
    )
    expected_variables = {
        "GLM52_ACTIVATION_ID": inputs.activation_id,
        "GLM52_RUN_ID": RUN_ID,
        "GLM52_SUPPORT_STACK_ID": {"Ref": "AWS::StackId"},
        "GLM52_SUPPORT_HOST_INSTANCE_ID": {"Ref": "CombinedHost"},
        "GLM52_COMBINED_HOST_INSTANCE_ID": {"Ref": "CombinedHost"},
        "GLM52_SUPPORT_HOST_SECURITY_GROUP_ID": {"Ref": "CombinedHostSecurityGroup"},
        "GLM52_LEDGER_TABLE_NAME": inputs.ledger_table_name,
        "GLM52_ACCOUNT_ID": ACCOUNT_ID,
        "GLM52_CAMPAIGN_BUCKET": inputs.model_bucket_name,
        "GLM52_EXPECTED_BUCKET_OWNER": ACCOUNT_ID,
        "GLM52_NUMERIC_BINDING_VERSION_ARN": {"Ref": "NumericBindingVersion"},
        "GLM52_RETAINED_CANCELLATION_VERSION_ARN": {
            "Ref": "RetainedCancellationVersion"
        },
        "GLM52_WORKER_DRAIN_VERSION_ARN": {
            "Fn::ImportValue": "KeepGlm52Task12WorkerDrainVersionArn"
        },
        "GLM52_TASK9_LIABILITY_WATCHER_VERSION_ARN": {
            "Fn::ImportValue": ("KeepGlm52Task9LiabilityWatcherVersionArn")
        },
        "GLM52_TERMINAL_V2_VERSION_ARN": {
            "Fn::ImportValue": "KeepGlm52Task12TerminalV2VersionArn"
        },
        "GLM52_SUPPORT_QUIESCENCE_SECONDS": "60",
        "GLM52_OPERATOR_TOPIC_ARN": SUPPORT_OPERATOR_TOPIC_ARN,
        "GLM52_WORK_STOP_HOURS": "68",
        "GLM52_DELETE_REQUEST_HOURS": "71",
        "GLM52_ABSENCE_EXPECTED_HOURS": "72",
        "GLM52_HOST_EGRESS_WARNING_BYTES": str(5 * 1024**3),
        "GLM52_HOST_EGRESS_DRAIN_BYTES": str(6 * 1024**3),
        "GLM52_NAT_AUTHORITY_BYTES": str(10 * 1024**3),
        "GLM52_NAT_GATEWAY_ID": {"Ref": "NatGateway"},
        "GLM52_DIRECT_CHILD_DELETE_ACTIONS": "FORBIDDEN",
        "GLM52_POST_72H_RECONCILIATIONS": "1",
        "GLM52_DELETE_STACK_OWNER": "RETAINED_LIFECYCLE_ONLY",
    }
    if variables != expected_variables:
        raise ValueError("deadline runtime contract is not exact")
    start = datetime.fromisoformat(inputs.activation_started_at.replace("Z", "+00:00"))
    for logical_id, hours, payload in (
        ("WorkStopSchedule", 68, "WORK_STOP"),
        ("DeleteRequestSchedule", 71, "DELETE_REQUEST"),
        ("AbsenceDeadlineSchedule", 72, "ABSENCE_EXPECTED"),
    ):
        properties = resources.get(logical_id, {}).get("Properties", {})
        at = (start + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
        if (
            properties.get("ScheduleExpression") != f"at({at})"
            or properties.get("Target", {}).get("Arn")
            != {"Ref": "SupportDeadlineVersion"}
            or properties.get("Target", {}).get("Input")
            != (f'{{"activation_id":"{inputs.activation_id}","deadline":"{payload}"}}')
        ):
            raise ValueError("support absolute deadline schedule is not exact")
    for logical_id, threshold in (
        ("HostEgressWarningAlarm", 5 * 1024**3),
        ("HostEgressDrainAlarm", 6 * 1024**3),
        ("NatEgressWarningAlarm", 5 * 1024**3),
        ("NatEgressDrainAlarm", 6 * 1024**3),
        ("NatEgressAuthorityAlarm", 10 * 1024**3),
    ):
        properties = resources.get(logical_id, {}).get("Properties", {})
        if (
            properties.get("Threshold") != threshold
            or properties.get("TreatMissingData") != "notBreaching"
        ):
            raise ValueError("support egress metric threshold is not exact")
        expected_metric = (
            "HostNonEndpointBytes"
            if logical_id.startswith("Host")
            else "ActivationCumulativeNatProcessedBytes"
        )
        if (
            properties.get("Namespace") != "GLM52/H1g"
            or properties.get("MetricName") != expected_metric
            or properties.get("Dimensions")
            != (
                [
                    {
                        "Name": "ActivationId",
                        "Value": inputs.activation_id,
                    }
                ]
                if logical_id.startswith("Host")
                else [
                    {
                        "Name": "ActivationId",
                        "Value": inputs.activation_id,
                    },
                    {
                        "Name": "NatGatewayId",
                        "Value": {"Ref": "NatGateway"},
                    },
                ]
            )
            or properties.get("Period") != 60
            or properties.get("Statistic") != "Maximum"
            or "Metrics" in properties
        ):
            raise ValueError("support egress metric is not cumulative and exact")
    rule = resources.get("SupportEgressAlarmRule", {}).get("Properties", {})
    if rule.get("EventPattern", {}).get("detail", {}).get("state", {}).get("value") != [
        "ALARM"
    ] or rule.get("Targets") != [
        {
            "Arn": {"Ref": "SupportDeadlineVersion"},
            "Id": "ExactSupportDeadlineVersion",
            "RetryPolicy": {
                "MaximumEventAgeInSeconds": 60,
                "MaximumRetryAttempts": 0,
            },
        }
    ]:
        raise ValueError("egress alarm event path is not exact")
    return True


def validate_support_deletion_role_policy(
    augmentation: Mapping[str, object], inputs: SupportInputs
) -> bool:
    """Validate exact support-stack deletion authority from authenticated input."""

    resources = _template_resources(augmentation, label="retained support augmentation")
    properties = resources.get("H1gSupportDeletionServiceRole", {}).get("Properties")
    if (
        type(properties) is not dict
        or properties.get("RoleName") != "keep-glm52-h1g-support-deletion"
        or properties.get("AssumeRolePolicyDocument")
        != {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "cloudformation.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
    ):
        raise ValueError("support deletion service role trust is not exact")
    policies = properties.get("Policies")
    if (
        type(policies) is not list
        or len(policies) != 1
        or policies[0].get("PolicyName") != "DeleteExactSupportInventoryOnly"
        or policies[0].get("PolicyDocument", {}).get("Version") != "2012-10-17"
        or type(policies[0].get("PolicyDocument", {}).get("Statement")) is not list
    ):
        raise ValueError("support deletion service role policy is not exact")
    statements = policies[0]["PolicyDocument"]["Statement"]
    by_action: dict[str, Mapping[str, object]] = {}
    for statement in statements:
        if (
            type(statement) is not dict
            or statement.get("Effect") != "Allow"
            or type(statement.get("Action")) is not str
            or statement["Action"] in by_action
            or statement.get("Sid") != _deletion_statement_sid(statement["Action"])
            or set(statement)
            not in (
                {"Sid", "Effect", "Action", "Resource"},
                {"Sid", "Effect", "Action", "Resource", "Condition"},
            )
        ):
            raise ValueError("support deletion policy statement is not exact")
        by_action[statement["Action"]] = statement
    if set(by_action) != _SUPPORT_DELETION_ACTIONS:
        raise ValueError("support deletion action inventory is incomplete")
    expected_resources = inputs.support_deletion_inventory["action_resources"]
    for action, statement in by_action.items():
        resource = statement["Resource"]
        actual = [resource] if type(resource) is str else resource
        if (
            type(actual) is not list
            or actual != expected_resources[action]
            or not actual
            or any(
                type(item) is not str
                or (
                    item == "*"
                    and action not in _UNSCOPED_SUPPORT_DELETION_READ_ACTIONS
                )
                for item in actual
            )
        ):
            raise ValueError("support deletion resource allowlist is foreign")
        expected_condition: Mapping[str, object] | None = None
        if action == "ec2:CreateSnapshot":
            expected_condition = {
                "StringEquals": {
                    "aws:RequestTag/ActivationId": inputs.activation_id,
                    "aws:RequestTag/RunId": RUN_ID,
                    "aws:RequestTag/Purpose": (
                        "forensic-data-volume-deletion-snapshot"
                    ),
                },
                "ForAllValues:StringEquals": {
                    "aws:TagKeys": ["ActivationId", "Purpose", "RunId"]
                },
            }
        elif action == "kms:CreateGrant":
            expected_condition = {
                "Bool": {"kms:GrantIsForAWSResource": "true"},
                "StringEquals": {
                    "kms:EncryptionContext:ActivationId": (inputs.activation_id),
                    "kms:EncryptionContext:RunId": RUN_ID,
                },
            }
        elif action in _UNSCOPED_SUPPORT_DELETION_READ_ACTIONS:
            expected_condition = {"StringEquals": {"aws:RequestedRegion": REGION}}
        if statement.get("Condition") != expected_condition:
            raise ValueError("support deletion mutation condition is foreign")
    inventory = inputs.support_deletion_inventory
    metadata = augmentation.get("Metadata")
    expected_metadata = _support_deletion_inventory_metadata(inventory)
    if type(metadata) is not dict or any(
        metadata.get(key) != value for key, value in expected_metadata.items()
    ):
        raise ValueError("support deletion inventory metadata is foreign")
    return True


def validate_retained_augmentation(
    augmentation: Mapping[str, object], inputs: SupportInputs
) -> bool:
    """Validate the one frozen retained runtime and Standard Workflow graph."""

    from .fence_executor import (
        build_support_deletion_workflow_definition,
        support_deletion_workflow_history_event_ceiling,
    )

    if type(inputs) is not SupportInputs:
        raise TypeError("retained augmentation requires exact support inputs")
    grace = bootstrap_grace_contract(inputs)
    workflow_definition = build_support_deletion_workflow_definition()
    workflow_sha256 = hashlib.sha256(
        canonical_json_bytes(workflow_definition)
    ).hexdigest()
    nat_schedule_name = _nat_accumulator_schedule_name(inputs)
    nat_schedule_end_date = _nat_accumulator_schedule_end_date(inputs)
    resources = _template_resources(augmentation, label="retained support augmentation")
    task12 = _build_task12_retained_fragment(inputs)

    task12_metadata = task12["Metadata"]
    actual_task12 = {
        "Metadata": {
            key: augmentation.get("Metadata", {}).get(key) for key in task12_metadata
        },
        "Resources": {
            logical_id: resources.get(logical_id) for logical_id in task12["Resources"]
        },
    }
    if "Parameters" in task12:
        actual_task12["Parameters"] = augmentation.get("Parameters")
    if actual_task12 != task12:
        raise ValueError(
            "Task 12 retained runtime is not the authenticated exact graph"
        )
    task10 = _build_task10_retained_fragment(inputs)
    from .task10_support_plane import (
        Task10SupportInputs,
        validate_task10_support_plane_fragment,
    )

    task10_metadata = task10["Metadata"]
    actual_task10 = {
        "Parameters": {},
        "Metadata": {
            key: augmentation.get("Metadata", {}).get(key) for key in task10_metadata
        },
        "Resources": {
            logical_id: resources.get(logical_id) for logical_id in task10["Resources"]
        },
    }
    validate_task10_support_plane_fragment(
        actual_task10,
        inputs=Task10SupportInputs(
            lambda_code_bucket=inputs.lambda_code_bucket,
            lambda_code_key=inputs.lambda_code_key,
            lambda_code_version=inputs.lambda_code_version_id,
            lambda_code_sha256=inputs.lambda_code_sha256,
            ledger_table_arn=inputs.ledger_table_arn,
            campaign_bucket_arn=inputs.model_bucket_arn,
        ),
    )
    expected_metadata = {
        "RecordType": "glm52_h1g_retained_support_augmentation_v1",
        "SupportStackId": inputs.support_stack_id,
        "RetainedStackId": inputs.retained_stack_id,
        "SupportImportsRetainedOnly": [
            name
            for name in inputs.retained_export_names
            if name
            not in {
                "KeepGlm52Task12TerminalV2VersionArn",
                "KeepGlm52Task12WorkerDrainVersionArn",
                "KeepGlm52Task9LiabilityWatcherVersionArn",
            }
        ],
        "RetainedImportsSupport": [],
        "DirectChildDeletionByLifecycle": False,
        "ActiveFinalizationOwnerMax": 1,
        "FinalizationExecutionsMax": 12,
        "NatGatewayId": inputs.nat_gateway_id,
        "NatObservationMode": ("ACTIVATION_CUMULATIVE_FOUR_COUNTERS_60S"),
        "NatObservationMaximumWindows": _NAT_ACCUMULATOR_MAX_WINDOWS,
        "NatAccumulatorScheduleArn": _nat_accumulator_schedule_arn(inputs),
        "NatAccumulatorScheduleEndDate": nat_schedule_end_date,
        "RetainedLifecycleWorkflowType": "STANDARD",
        "RetainedLifecycleDefinitionSha256": workflow_sha256,
        "RetainedLifecycleHistoryEventCeiling": (
            support_deletion_workflow_history_event_ceiling()
        ),
        "BootstrapGraceIdentitySha256": grace["canonical_body_sha256"],
        **task12["Metadata"],
        **task10["Metadata"],
        **_support_deletion_inventory_metadata(inputs.support_deletion_inventory),
    }
    if (
        augmentation.get("AWSTemplateFormatVersion") != "2010-09-09"
        or augmentation.get("Metadata") != expected_metadata
    ):
        raise ValueError("retained support augmentation metadata is not exact")
    required = (
        {
            "H1gRetainedSupportOperatorTopic",
            "H1gSupportDeletionServiceRole",
            "H1gRetainedLifecycleStateMachineRole",
            "H1gRetainedLifecycleStateMachine",
            "H1gRetainedLifecycleStateMachineVersion",
            "H1gRetainedSupportLifecycleRole",
            "H1gRetainedSupportLifecycleFunction",
            "H1gRetainedSupportLifecycleVersion",
            "H1gRetainedSupportScheduleRole",
            "H1gRetainedWorkStopSchedule",
            "H1gRetainedDeleteRequestSchedule",
            "H1gRetainedAbsenceDeadlineSchedule",
            "H1gRetainedNatAccumulatorSchedule",
            "H1gRetainedNatObservationMissingAlarm",
        }
        | set(task12["Resources"])
        | set(task10["Resources"])
    )
    if set(resources) != required:
        raise ValueError("retained support augmentation resources are not exact")
    if {
        "H1gTask7SupportCounterTable",
        "H1gRetainedNatAccumulatorRole",
        "H1gRetainedNatAccumulatorFunction",
        "H1gRetainedNatAccumulatorVersion",
    } & set(resources):
        raise ValueError("Task 7 cannot add a retained NAT sidecar family")
    for logical_id, resource in resources.items():
        if (
            resource.get("DeletionPolicy") != "Retain"
            or resource.get("UpdateReplacePolicy") != "Retain"
        ):
            raise ValueError(f"{logical_id} is not retained")

    machine = resources["H1gRetainedLifecycleStateMachine"]["Properties"]
    if machine != {
        "Definition": workflow_definition,
        "RoleArn": {
            "Fn::GetAtt": [
                "H1gRetainedLifecycleStateMachineRole",
                "Arn",
            ]
        },
        "StateMachineName": "keep-glm52-h1g-retained-lifecycle",
        "StateMachineType": "STANDARD",
        "Tags": list(_tags(inputs, "retained-support-lifecycle")),
    }:
        raise ValueError("retained Standard Workflow definition is not exact")
    version = resources["H1gRetainedLifecycleStateMachineVersion"]["Properties"]
    if version != {
        "Description": f"definition={workflow_sha256}",
        "StateMachineArn": {"Ref": "H1gRetainedLifecycleStateMachine"},
    }:
        raise ValueError("retained lifecycle version is not exact")
    mutation_resources = {
        "DisableSupportTerminationProtection": (
            "arn:aws:states:::aws-sdk:cloudformation:updateTerminationProtection"
        ),
        "DeleteSupportStack": ("arn:aws:states:::aws-sdk:cloudformation:deleteStack"),
    }
    states = machine["Definition"]["States"]
    for state_name, task_resource in mutation_resources.items():
        state = states.get(state_name)
        if (
            type(state) is not dict
            or state.get("Type") != "Task"
            or state.get("Resource") != task_resource
            or "Retry" in state
            or any(
                catcher.get("Next") == state_name
                for catcher in state.get("Catch", [])
                if type(catcher) is dict
            )
        ):
            raise ValueError("retained mutation task is resubmittable")
    if (
        states["SupportStackAbsenceConfirmed"] != {"Type": "Succeed"}
        or states["DescribeSupportStackForAbsence"]["Resource"]
        != "arn:aws:states:::aws-sdk:cloudformation:describeStacks"
    ):
        raise ValueError("retained deletion lacks terminal absence readback")

    lifecycle_policies = resources["H1gRetainedSupportLifecycleRole"]["Properties"].get(
        "Policies"
    )
    if type(lifecycle_policies) is not list or len(lifecycle_policies) != 1:
        raise ValueError("retained lifecycle runtime role is not exact")
    lifecycle_statements = lifecycle_policies[0]["PolicyDocument"]["Statement"]
    lifecycle_actions = {
        action
        for statement in lifecycle_statements
        for action in _statement_actions(statement)
    }
    if lifecycle_actions != {
        "ec2:DescribeInstances",
        "ec2:DescribeSecurityGroups",
        "ec2:StopInstances",
        "ec2:RevokeSecurityGroupEgress",
        "cloudformation:DescribeStacks",
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "sns:Publish",
        "cloudwatch:GetMetricData",
        "cloudwatch:PutMetricData",
        "scheduler:GetSchedule",
        "scheduler:UpdateSchedule",
        "scheduler:DeleteSchedule",
        "states:StartExecution",
    }:
        raise ValueError("retained lifecycle runtime authority drifted")
    forbidden_runtime = {
        "cloudformation:UpdateTerminationProtection",
        "cloudformation:DeleteStack",
        "iam:PassRole",
        "dynamodb:UpdateItem",
    }
    if lifecycle_actions & forbidden_runtime:
        raise ValueError("process-local retained runtime owns a mutation")

    workflow_policies = resources["H1gRetainedLifecycleStateMachineRole"][
        "Properties"
    ].get("Policies")
    workflow_statements = (
        workflow_policies[0]["PolicyDocument"]["Statement"]
        if type(workflow_policies) is list and len(workflow_policies) == 1
        else None
    )
    if type(workflow_statements) is not list:
        raise ValueError("retained workflow role is not exact")
    workflow_actions = {
        action
        for statement in workflow_statements
        for action in _statement_actions(statement)
    }
    if workflow_actions != {
        "cloudformation:DescribeStacks",
        "cloudformation:UpdateTerminationProtection",
        "cloudformation:DeleteStack",
        "iam:PassRole",
    }:
        raise ValueError("retained workflow mutation authority drifted")
    for statement in workflow_statements:
        actions = set(_statement_actions(statement))
        if (
            actions
            & {
                "cloudformation:DescribeStacks",
                "cloudformation:UpdateTerminationProtection",
                "cloudformation:DeleteStack",
            }
            and statement.get("Resource") != inputs.support_stack_id
        ):
            raise ValueError("retained workflow stack target is foreign")
        if "iam:PassRole" in actions and (
            statement.get("Resource") != SUPPORT_DELETION_ROLE_ARN
            or statement.get("Condition")
            != {"StringEquals": {"iam:PassedToService": "cloudformation.amazonaws.com"}}
        ):
            raise ValueError("retained workflow role pass is foreign")

    variables = resources["H1gRetainedSupportLifecycleFunction"]["Properties"][
        "Environment"
    ]["Variables"]
    if (
        variables.get("GLM52_NAT_ACCUMULATOR_MODE")
        != "ACTIVATION_CUMULATIVE_FOUR_COUNTERS_60S"
        or variables.get("GLM52_NAT_GATEWAY_ID") != inputs.nat_gateway_id
        or not re.fullmatch(
            r"i-[0-9a-f]{17}",
            str(variables.get("GLM52_SUPPORT_HOST_INSTANCE_ID")),
        )
        or not re.fullmatch(
            r"sg-[0-9a-f]{17}",
            str(variables.get("GLM52_SUPPORT_HOST_SECURITY_GROUP_ID")),
        )
        or variables.get("GLM52_ACTIVATION_STARTED_AT") != inputs.activation_started_at
        or variables.get("GLM52_MISSING_METRIC_AFTER_GRACE") != "DRAIN"
        or variables.get("GLM52_WORK_AUTHORIZATION") != "METRICS_OK_ONLY"
        or "GLM52_COUNTER_TABLE_NAME" in variables
        or resources["H1gRetainedSupportLifecycleFunction"]["Properties"].get("Handler")
        != "retained_support_lifecycle_handler.main"
    ):
        raise ValueError("single retained lifecycle runtime is not exact")
    schedule = resources["H1gRetainedNatAccumulatorSchedule"]["Properties"]
    if (
        schedule.get("ScheduleExpression") != "rate(1 minute)"
        or schedule.get("Name") != nat_schedule_name
        or schedule.get("EndDate") != nat_schedule_end_date
        or schedule.get("Target", {}).get("Arn")
        != {"Ref": "H1gRetainedSupportLifecycleVersion"}
        or schedule.get("Target", {}).get("Input")
        != (
            '{"action":"ACCUMULATE_NAT_60S","activation_id":"'
            f'{inputs.activation_id}","activation_started_at":"'
            f'{inputs.activation_started_at}","nat_gateway_id":"'
            f'{inputs.nat_gateway_id}"'
            "}"
        )
        or schedule.get("Target", {}).get("RetryPolicy")
        != {
            "MaximumEventAgeInSeconds": 60,
            "MaximumRetryAttempts": 0,
        }
    ):
        raise ValueError("retained NAT observation schedule is not exact")
    missing_alarm = resources.get("H1gRetainedNatObservationMissingAlarm", {}).get(
        "Properties", {}
    )
    if (
        missing_alarm.get("AlarmName")
        != (
            f"keep-glm52-h1g-{inputs.activation_id}-nat-observation-missing-after-grace"
        )
        or missing_alarm.get("ComparisonOperator") != "GreaterThanThreshold"
        or missing_alarm.get("DatapointsToAlarm") != 16
        or missing_alarm.get("EvaluationPeriods") != 16
        or missing_alarm.get("MetricName") != "ActivationCumulativeNatProcessedBytes"
        or missing_alarm.get("Namespace") != "GLM52/H1g"
        or missing_alarm.get("Period") != 60
        or missing_alarm.get("Statistic") != "Maximum"
        or missing_alarm.get("Threshold") != 10**30
        or missing_alarm.get("TreatMissingData") != "breaching"
        or missing_alarm.get("Dimensions")
        != [
            {
                "Name": "ActivationId",
                "Value": inputs.activation_id,
            },
            {
                "Name": "NatGatewayId",
                "Value": inputs.nat_gateway_id,
            },
        ]
    ):
        raise ValueError("retained NAT missing-observation alarm is not exact")
    validate_support_deletion_role_policy(augmentation, inputs)
    return True


def _validate_task12_support_first_version_readback(
    inventory: Mapping[str, object],
) -> None:
    rows = inventory.get("stack_resources")
    if type(rows) is not list:
        raise ValueError("support first-version readback is absent")
    physical_by_logical = {
        row["logical_id"]: row["physical_id"]
        for row in rows
        if (
            type(row) is dict
            and type(row.get("logical_id")) is str
            and type(row.get("physical_id")) is str
        )
    }
    expected = _task12_future_support_bindings()
    for logical_id in (
        "NumericBindingVersion",
        "RetainedCancellationVersion",
    ):
        if physical_by_logical.get(logical_id) != expected[logical_id]:
            raise ValueError(
                f"{logical_id} first-version readback mismatched "
                "the pre-support binding"
            )


def _build_support_postcreate_plane(
    *,
    inputs: SupportBuildInputs,
    template: Mapping[str, object],
    support_stack_id: str,
    services: SupportMaterializationServices,
) -> SupportPostcreateBundle:
    """Read AWS truth and build retained authority in one non-splittable call."""

    materialization = _materialize_support_postcreate(
        template=template,
        expected_support_stack_id=support_stack_id,
        services=services,
    )
    if type(materialization) is not dict or set(materialization) != {
        "support_stack_id",
        "nat_gateway_id",
        "support_template_body_sha256",
        "stack_resources",
        "derived_resources",
        "snapshot_contract",
        "deletion_inventory",
        "stack_outputs",
        "combined_host_identity",
    }:
        raise TypeError("postcreate materialization is not exact service truth")
    materialized_stack_id = materialization["support_stack_id"]
    materialized_nat_gateway_id = materialization["nat_gateway_id"]
    template_identity = materialization["support_template_body_sha256"]
    inventory = materialization["deletion_inventory"]
    if (
        type(template_identity) is not str
        or _SHA256.fullmatch(template_identity) is None
        or type(inventory) is not dict
    ):
        raise TypeError("postcreate service truth has no exact identity")
    _support_deletion_inventory_metadata(inventory)
    _validate_task12_support_first_version_readback(inventory)
    stack_rows = inventory.get("stack_resources")
    nat_rows = (
        [
            row
            for row in stack_rows
            if type(row) is dict and row.get("logical_id") == "NatGateway"
        ]
        if type(stack_rows) is list
        else []
    )
    if (
        _stack_id(materialized_stack_id, SUPPORT_STACK_NAME) != materialized_stack_id
        or len(nat_rows) != 1
        or nat_rows[0].get("physical_id") != materialized_nat_gateway_id
        or inventory.get("support_stack_id") != materialized_stack_id
        or inventory.get("support_template_body_sha256") != template_identity
    ):
        raise ValueError("materialized support coordinates are incoherent")
    values: dict[str, object] = {}
    for field in _INPUT_FIELDS:
        if field == "schema_version":
            values[field] = 1
        elif field == "record_type":
            values[field] = "glm52_h1g_support_inputs_v1"
        elif field == "nat_gateway_id":
            values[field] = materialized_nat_gateway_id
        elif field == "support_stack_id":
            values[field] = materialized_stack_id
        elif field == "support_deletion_inventory":
            values[field] = inventory
        elif field == "fence_manifest_coordinate":
            values[field] = inputs.bootstrap_manifest_coordinate
        else:
            values[field] = getattr(inputs, field)
    materialized_inputs = SupportInputs(**values)
    postcreate_retained = build_postcreate_retained_support_fragment(
        materialized_inputs
    )
    retained = _build_retained_augmentation(materialized_inputs)
    validate_retained_augmentation(retained, materialized_inputs)
    outputs = materialization["stack_outputs"]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_postcreate_manifest_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": inputs.activation_id,
        "support_stack_id": materialized_stack_id,
        "nat_gateway_id": materialized_nat_gateway_id,
        "support_template_body_sha256": template_identity,
        "support_postcreate_inventory_sha256": (inventory["canonical_body_sha256"]),
        "direct_grant_evidence_coordinate": outputs["DirectGrantEvidenceCoordinate"],
        "direct_grant_evidence_sha256": outputs["DirectGrantEvidenceSha256"],
        "retained_augmentation_body_sha256": hashlib.sha256(
            canonical_json_bytes(retained)
        ).hexdigest(),
        "postcreate_retained_fragment_body_sha256": hashlib.sha256(
            canonical_json_bytes(postcreate_retained)
        ).hexdigest(),
        "task12_deployment_authority": (_task12_deployment_authority_manifest()),
        "caller_physical_ids_accepted": False,
        "aws_truth_claimed": True,
    }
    manifest["canonical_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    from .task9_contract import (
        build_task9_contract,
        build_task9_deployed_identity,
    )

    static_contract = build_task9_contract(Path(__file__).resolve().parents[2])
    combined = materialization["combined_host_identity"]
    physical_by_logical = {
        row["logical_id"]: row["physical_id"] for row in inventory["stack_resources"]
    }
    service_identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "COMBINED_HOST_SKY_SERVICE_V1",
                "instance_id": combined["instance_id"],
                "private_ip": combined["private_ip"],
                "ami_id": combined["ami_id"],
                "boot_identity_sha256": inputs.host_boot_identity_sha256,
                "describe_instances_response_sha256": combined[
                    "describe_instances_response_sha256"
                ],
            }
        )
    ).hexdigest()
    consolidation_identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "EFFECTIVE_CONTROLLER_CONSOLIDATION_V1",
                "instance_id": combined["instance_id"],
                "service_identity_sha256": service_identity,
                "separate_controller_count": 0,
            }
        )
    ).hexdigest()
    output_prefixes = {
        "ATTESTATION": "Attestation",
        "LAUNCH_ADMISSION": "LaunchAdmission",
        "NUMERIC_BINDING": "NumericBinding",
        "RETAINED_CANCELLATION": "RetainedCancellation",
    }
    client_logical_ids = {
        "ATTESTATION": "AttestationClientTlsSecret",
        "LAUNCH_ADMISSION": "LaunchAdmissionClientTlsSecret",
        "NUMERIC_BINDING": "NumericBindingClientTlsSecret",
        "RETAINED_CANCELLATION": ("RetainedCancellationClientTlsSecret"),
    }
    deployed_relays = []
    for purpose, policy in static_contract["relay_policy"].items():
        output_prefix = output_prefixes[purpose]
        logical_id = client_logical_ids[purpose]
        deployed_relays.append(
            {
                "purpose": purpose,
                "client_secret_arn": physical_by_logical[logical_id],
                "client_secret_version_id": outputs[logical_id + "VersionId"],
                "client_secret_version_stage": secret_stage(
                    inputs.activation_id,
                    logical_id,
                ),
                "client_certificate_der_sha256": outputs[
                    output_prefix + "ClientDerSha256"
                ],
                "server_secret_arn": physical_by_logical["CombinedHostTlsSecret"],
                "server_secret_version_id": outputs["CombinedHostTlsSecretVersionId"],
                "server_secret_version_stage": secret_stage(
                    inputs.activation_id,
                    "CombinedHostTlsSecret",
                ),
                "server_certificate_der_sha256": outputs[
                    output_prefix + "ServerDerSha256"
                ],
                "port": policy["port"],
                "principal_arn": policy["principal_arn"],
                "allowed_paths": policy["allowed_paths"],
            }
        )
    deployed_identity = build_task9_deployed_identity(
        static_contract=static_contract,
        activation_id=inputs.activation_id,
        support_binding={
            "support_stack_id": materialized_stack_id,
            "support_template_body_sha256": template_identity,
            "postcreate_manifest_sha256": manifest["canonical_body_sha256"],
            "describe_stacks_response_sha256": inventory[
                "describe_stacks_response_sha256"
            ],
        },
        combined_host={
            "instance_id": combined["instance_id"],
            "private_ip": combined["private_ip"],
            "ami_id": combined["ami_id"],
            "instance_profile_arn": combined["instance_profile_arn"],
            "boot_identity_sha256": inputs.host_boot_identity_sha256,
            "service_identity_sha256": service_identity,
            "consolidation_signal_identity_sha256": (consolidation_identity),
        },
        tls={
            "issuance_id": outputs["IssuanceId"],
            "bundle_sha256": outputs["BundleSha256"],
            "ca_certificate_der_sha256": outputs["TlsCaDerSha256"],
            "not_valid_before": outputs["TlsNotValidBefore"],
            "not_valid_after": outputs["TlsNotValidAfter"],
            "relays": deployed_relays,
        },
    )
    return SupportPostcreateBundle(
        inputs=materialized_inputs,
        postcreate_retained_fragment=postcreate_retained,
        retained_augmentation=retained,
        manifest=manifest,
        task9_deployed_identity=deployed_identity,
    )


def coordinate_support_postcreate(
    *,
    inputs: SupportBuildInputs,
    template: Mapping[str, object],
    support_stack_id: str,
    services: SupportMaterializationServices,
) -> SupportPostcreateBundle:
    """Read, validate, and consume AWS truth inside one production boundary."""

    if type(inputs) is not SupportBuildInputs:
        raise TypeError("postcreate coordinator requires exact build inputs")
    if _stack_id(support_stack_id, SUPPORT_STACK_NAME) != support_stack_id:
        raise ValueError("postcreate coordinator stack ID is not exact")
    expected_template = _build_support_template(inputs)
    validate_support_template(expected_template, inputs)
    if type(template) is not dict or canonical_json_bytes(
        template
    ) != canonical_json_bytes(expected_template):
        raise ValueError("postcreate template is not the canonical precreate template")
    return _build_support_postcreate_plane(
        inputs=inputs,
        template=expected_template,
        support_stack_id=support_stack_id,
        services=services,
    )


def coordinate_support_task12_postpublication(
    *,
    inputs: object,
    services: object,
) -> Mapping[str, object]:
    """Write Task 12 authority only after retained CFN version readback."""

    from .task12_postpublication import (
        Task12PostpublicationInputs,
        Task12PostpublicationServices,
        materialize_task12_postpublication,
    )

    if (
        type(inputs) is not Task12PostpublicationInputs
        or type(services) is not Task12PostpublicationServices
    ):
        raise TypeError(
            "Task 12 postpublication coordinator requires exact typed boundaries"
        )
    return materialize_task12_postpublication(
        inputs=inputs,
        services=services,
    )


def _task12_deployment_authority_manifest() -> Mapping[str, object]:
    """Describe the exact post-publication custody required by Task 12."""

    return {
        "record_type": "glm52_task12_lambda_deployment_v1",
        "materialization_phase": ("AFTER_RETAINED_STACK_VERSION_PUBLICATION"),
        "lookup_consistency": "STRONGLY_CONSISTENT",
        "lookup_key_source": "context.invoked_function_arn",
        "partition_key": f"RUN#{RUN_ID}",
        "sort_key_prefix": "TASK12_LAMBDA_DEPLOYMENT#",
        "sort_key_fields": [
            "context.invoked_function_arn",
            "event.caller_state_machine_arn",
            "event.activation_id",
            "event.generation_text",
        ],
        "function_version_count": 8,
        "caller_edge_count": 10,
        "record_count": 10,
        "operation_input_record_count": 32,
        "operation_input_source": (
            "AUTHENTICATED_VERSIONED_S3_STATIC_DESCRIPTOR_AUTHORITY"
        ),
        "operation_input_record_kind": "IMMUTABLE_OPERATION_DESCRIPTOR",
        "future_runtime_payloads_materialized": False,
        "runtime_input_source": (
            "AUTHENTICATED_EXACT_READ_OF_CANONICAL_RETAINED_RECORD"
        ),
        "producer_consumer_matrix": "CLOSED_ALL_32_NO_ORPHAN_CONSUMERS",
        "caller_supplied_prehashed_operation_rows_forbidden": True,
        "versioned_authority_manifest_count": 10,
        "versioned_authority_manifest_record_type": (
            "glm52_task12_lambda_invocation_authority_v1"
        ),
        "authority_operation_inventory": "EXACT_CLOSED_32_ASL_STATE_NAMES",
        "operation_input_sort_key_fields": [
            "activation_id",
            "handler_kind",
            "operation_kind",
            "generation_text",
        ],
        "production_materializer": ("coordinate_support_task12_postpublication"),
        "conditional_dynamodb_writes_required": True,
        "ambiguous_write_exact_readback_required": True,
        "caller_state_machine_version_required": True,
        "caller_version_proof": (
            "DescribeExecution.stateMachineVersionArn equals deployment record"
        ),
        "function_version_required": True,
        "versioned_s3_authority_required": True,
        "strongly_consistent_dynamodb_input_required": True,
        "inline_self_hashed_config_forbidden": True,
        "static_worker_or_snapshot_ids_forbidden": True,
    }


def build_support_precreate_plane(
    *, inputs: SupportBuildInputs, price_card: SupportPriceCard
) -> SupportPrecreateBundle:
    """Build the logical support template without post-create physical truth."""

    if type(inputs) is not SupportBuildInputs:
        raise TypeError("precreate build requires exact SupportBuildInputs")
    template = _build_support_template(inputs)
    validate_support_template(template, inputs)
    retained_runtime = build_pre_support_retained_runtime_fragment(inputs)
    spend = build_support_spend_descriptor(
        inputs=inputs,
        price_card=price_card,
    )
    validate_support_spend_descriptor(spend, inputs)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_precreate_manifest_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": inputs.activation_id,
        "retained_stack_id": inputs.retained_stack_id,
        "support_build_inputs_identity_sha256": (support_build_inputs_identity(inputs)),
        "support_template_body_sha256": hashlib.sha256(
            canonical_json_bytes(template)
        ).hexdigest(),
        "support_spend_body_sha256": hashlib.sha256(
            canonical_json_bytes(spend)
        ).hexdigest(),
        "pre_support_retained_runtime_body_sha256": hashlib.sha256(
            canonical_json_bytes(retained_runtime)
        ).hexdigest(),
        "postcreate_materialization_required": True,
        "postcreate_required_authorities": [
            "ListStackResources",
            "DescribeStackResource",
            "DescribeInstances",
            "DescribeVpcEndpoints",
        ],
        "snapshot_binding": "CAPTURE_AT_DELETION",
        "aws_truth_claimed": False,
    }
    manifest["canonical_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return SupportPrecreateBundle(
        inputs=inputs,
        price_card=price_card,
        support_template=template,
        retained_runtime_fragment=retained_runtime,
        spend_descriptor=spend,
        manifest=manifest,
    )


def build_support_plane(
    *, inputs: SupportInputs, price_card: SupportPriceCard
) -> SupportPlaneBundle:
    """Build all deterministic Task 7 artifacts without AWS access."""

    template = _build_support_template(inputs)
    validate_support_template(template, inputs)
    spend = build_support_spend_descriptor(inputs=inputs, price_card=price_card)
    validate_support_spend_descriptor(spend, inputs)
    retained = _build_retained_augmentation(inputs)
    validate_retained_augmentation(retained, inputs)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_manifest_v1",
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "activation_id": inputs.activation_id,
        "support_stack_id": inputs.support_stack_id,
        "retained_stack_id": inputs.retained_stack_id,
        "support_inputs_identity_sha256": support_inputs_identity(inputs),
        "support_template_body_sha256": hashlib.sha256(
            canonical_json_bytes(template)
        ).hexdigest(),
        "retained_augmentation_body_sha256": hashlib.sha256(
            canonical_json_bytes(retained)
        ).hexdigest(),
        "task12_deployment_authority": (_task12_deployment_authority_manifest()),
        "support_spend_body_sha256": hashlib.sha256(
            canonical_json_bytes(spend)
        ).hexdigest(),
        "resource_ownership": {
            logical_id: SUPPORT_STACK_NAME for logical_id in template["Resources"]
        },
        "retained_imports_support": [],
        "support_imports_retained_only": list(inputs.retained_export_names),
        "kms_grant_provenance_contract": {
            "state": "REQUIRES_AUTHENTICATED_POSTCREATE_BINDING",
            "active_inventory_equality": (
                "baseline_plus_h1g_created_plus_service_created"
            ),
            "final_restoration_equality": "restored_equals_frozen_baseline",
            "direct_required_fields": [
                "request_identity_sha256",
                "response_identity_sha256",
                *_GRANT_PROJECTION_FIELDS,
            ],
            "service_required_fields": [
                "baseline_identity_sha256",
                "diff_identity_sha256",
                "list_grants_identity_sha256",
                "cloudtrail_event_identity_sha256",
                "cloudtrail_request_identity_sha256",
                "cloudtrail_response_identity_sha256",
                *_GRANT_PROJECTION_FIELDS,
                "originating_resource",
                "originating_resource_identity",
                "originating_service",
                "settling_window_seconds",
            ],
            "reject": [
                "unknown",
                "missing",
                "duplicate",
                "overbroad",
                "unattributed",
                "unrevocable",
                "snapshot-recovery",
            ],
        },
        "aws_truth_claimed": False,
    }
    manifest["canonical_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return SupportPlaneBundle(
        inputs=inputs,
        price_card=price_card,
        support_template=template,
        retained_augmentation=retained,
        spend_descriptor=spend,
        manifest=manifest,
    )


def build_support_contract_artifacts() -> Mapping[str, Mapping[str, object]]:
    """Generate non-live checked-in contracts for exact-input bound builds."""

    input_contract: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_build_input_contract_v1",
        "fixed_coordinates": {
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "run_id": RUN_ID,
            "fence_stack_name": FENCE_STACK_NAME,
            "retained_stack_name": RETAINED_STACK_NAME,
            "support_stack_name": SUPPORT_STACK_NAME,
        },
        "exact_required_fields": list(_BUILD_INPUT_FIELDS),
        "environment_specific_authenticated_fields": [
            "bootstrap_manifest_coordinate",
            "fence_template_inventory",
            "task11_writer_bindings",
            "fence_stack_id",
            "fence_service_role_arn",
            "retained_stack_id",
            "retained_vpc_id",
            "retained_vpc_cidr",
            "existing_subnet_cidrs",
            "existing_secondary_cidrs",
            "primary_az",
            "alternate_az",
            "primary_public_subnet_id",
            "retained_public_s3_endpoint_id",
            "retained_kms_key_arn",
            "retained_kms_key_id",
            "ledger_table_name",
            "ledger_table_arn",
            "model_bucket_name",
            "model_bucket_arn",
            "host_ami_id",
            "host_private_ip",
            "host_user_data",
            "host_user_data_sha256",
            "host_boot_identity_sha256",
            "root_volume_gib",
            "root_volume_type",
            "root_volume_iops",
            "root_volume_throughput_mibps",
            "cryptography_layer_arn",
            "cryptography_layer_sha256",
            "lambda_code_bucket",
            "lambda_code_key",
            "lambda_code_version_id",
            "lambda_code_sha256",
            "attestation_port",
            "launch_admission_port",
            "numeric_binding_port",
            "retained_cancellation_port",
            "activation_started_at",
            "runtime_credential_cutoff_at",
            "price_card_identity_sha256",
            "retained_export_names",
        ],
        "environment_specific_defaults": {},
        "aws_readback_required_before_bound_build": False,
        "postcreate_materialization": {
            "required": True,
            "caller_physical_ids_accepted": False,
            "authorities": [
                "ListStackResources",
                "DescribeStackResource",
                "DescribeInstances",
                "DescribeVpcEndpoints",
            ],
            "outputs": [
                "support_stack_id",
                "nat_gateway_id",
                "support_deletion_inventory",
                "retained_support_augmentation",
            ],
            "snapshot_binding": "CAPTURE_AT_DELETION",
        },
    }
    template_contract: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_template_contract_v1",
        "sole_owner": SUPPORT_STACK_NAME,
        "retained_import_direction": "support-imports-retained-only",
        "support_cidrs": list(SUPPORT_CIDRS),
        "resource_cardinality": {
            "host_egress_subnets": 1,
            "isolated_lambda_subnets": 2,
            "route_tables": 3,
            "route_table_associations": 3,
            "nat_gateways": 1,
            "elastic_ips": 1,
            "s3_gateway_endpoints": 1,
            "dynamodb_gateway_endpoints": 1,
            "secrets_manager_interface_endpoints": 1,
            "secrets_manager_interface_endpoint_enis": 2,
            "combined_hosts": 1,
            "additional_data_volumes": 1,
            "secrets": 8,
            "support_standard_workflows": 1,
            "rehearsal_buckets": 1,
            "rehearsal_bucket_policies": 1,
        },
        "secret_logical_ids": list(SECRET_LOGICAL_IDS),
        "runtime_function_families": list(RUNTIME_FUNCTIONS),
        "forbidden_support_resource_types": [
            "AWS::DynamoDB::Table",
            "AWS::KMS::Alias",
            "AWS::KMS::Key",
        ],
        "host": {
            "instance_type": "c6a.xlarge",
            "public_ip": False,
            "ssh_ingress": False,
            "imds_http_tokens": "required",
            "data_volume_gib": 50,
            "data_volume_type": "gp3",
            "data_volume_deletion_policy": "Snapshot",
            "data_volume_update_replace_policy": "Snapshot",
        },
        "custom_resource_callback_max_bytes": 4096,
        "custom_resource_creates": 2,
        "custom_resource_identity_preserving_updates": 2,
        "custom_resource_deletes": 2,
    }
    spend_contract: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_spend_envelope_v1",
        "estimated_support_ceiling_usd": "25.00",
        "support_price_terms": list(_SUPPORT_PRICE_TERMS),
        "separately_itemized_retained_price_terms": list(_RETAINED_PRICE_TERMS),
        "envelope": dict(_SUPPORT_ENVELOPE),
        "usage_ceilings": dict(SUPPORT_USAGE_CEILINGS),
        "gpu_residual_reserve_included": False,
        "worker_root_volume_tail_included": False,
    }
    task11_workflow_contract: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_task11_workflow_contract_v1",
        "lambda_configured_timeout_seconds": (LAMBDA_CONFIGURED_TIMEOUT_SECONDS),
        "closure_deadline_seconds": CLOSURE_DEADLINE_SECONDS,
        "step_functions_task_timeout_seconds": (STEP_FUNCTION_TASK_TIMEOUT_SECONDS),
        "minimum_remaining_before_token_seconds": (MIN_REMAINING_BEFORE_TOKEN_SECONDS),
        "closure_phase_ceilings_seconds": dict(CLOSURE_PHASE_CEILINGS),
        "suffix_phase_ceilings_seconds": dict(SUFFIX_PHASE_CEILINGS),
        "closure_steps": list(CLOSURE_STEPS),
        "authority_audit_kinds": list(AUTHORITY_AUDIT_KINDS),
        "handoff_audit_kind": HANDOFF_AUDIT_KIND,
        "workflow_definition": build_task11_workflow_definition(),
        "production_enablement_default": "CLOSURE_BUDGET_UNPROVEN",
    }
    task12_runtime_contract: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_task12_runtime_contract_v1",
        "retained_wrapper_handlers": [
            "retained_execution_observer_handler.main",
            "retained_terminal_v2_handler.main",
            "retained_finalizer_handler.main",
            "retained_h1g_drained_handler.main",
            "retained_worker_drain_handler.main",
            "retained_operator_disposition_handler.main",
            "retained_orphan_audit_handler.main",
            "retained_snapshot_cleanup_handler.main",
        ],
        "retained_function_versions": 8,
        "retained_standard_workflow_versions": 2,
        "retained_fragment_resources": 60,
        "merged_retained_resources": 83,
        "required_cloudformation_parameters": [
            "Task12ActivationOrdinal",
            "Task12Generation",
            "Task12GenerationText",
            "Task12DispatchIdentitySha256",
        ],
        "support_deletion_subworkflow_preserved": (
            "H1gRetainedLifecycleStateMachineVersion"
        ),
        "support_owned_resources": [],
        "support_stack_may_delete_retained_resources": False,
        "deployment_authority_records": 10,
        "deployment_authority": _task12_deployment_authority_manifest(),
        "postpublication_materializer": {
            "handler": (
                "materialize_h1g_support_plane.py --publish-task12-postpublication"
            ),
            "operation_input_records": 32,
            "versioned_authority_manifests": 10,
            "deployment_records": 10,
            "operation_source_authority": (
                "VERSIONED_AUTHENTICATED_STATIC_S3_COORDINATE"
            ),
            "operation_record_kind": "IMMUTABLE_OPERATION_DESCRIPTOR",
            "runtime_observations_pre_materialized": False,
            "descriptor_source_graph": (
                "CANONICAL_RETAINED_DDB_AND_VERSIONED_S3_AUTHORITIES"
            ),
            "caller_supplied_operation_rows_allowed": False,
            "conditional_write": (
                "attribute_not_exists(PK) AND attribute_not_exists(SK)"
            ),
            "ambiguous_write_reconciliation": (
                "AUTHENTICATED_STRONGLY_CONSISTENT_EXACT_READBACK"
            ),
            "iam_actions": [
                "cloudformation:DescribeStacks",
                "cloudformation:ListStackResources",
                "dynamodb:GetItem",
                "dynamodb:PutItem",
                "s3:GetObject",
                "s3:GetObjectVersion",
                "s3:ListBucketVersions",
                "s3:PutObject",
            ],
        },
        "retained_cost_terms": [
            "lambda",
            "step_functions",
            "logs",
            "eventbridge",
            "scheduler",
        ],
        "retained_cost_model": {
            "liability_watcher": [
                "lambda",
                "step_functions",
                "logs",
                "eventbridge",
            ],
            "snapshot_cleanup": ["lambda", "step_functions", "logs", "scheduler"],
            "invented_price_values": False,
        },
    }
    task10_production_contract: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_task10_production_contract_v1",
        "handler": "task10_sole_sender_handler.main",
        "runtime_adapter": "task10_sole_sender_runtime.py",
        "authority_record_type": ("glm52_task10_retained_sole_sender_authority_v1"),
        "terminal_record_type": ("glm52_task10_capacity_reconciliation_v1"),
        "retained_resources": 9,
        "merged_retained_resources": 83,
        "workflow_type": "STANDARD",
        "workflow_invocation_count": 1,
        "availability_zones": ["us-west-2" + letter for letter in "abcdef"],
        "maximum_ec2_calls": 6,
        "ec2_calls_per_az": 1,
        "stop_on_first_worker_success": True,
        "launch_shape": {
            "market": "on-demand",
            "instance_type": "p5.48xlarge",
            "imds_v2_required": True,
            "root_volume": {
                "delete_on_termination": True,
                "encrypted": True,
                "size_gib": 300,
                "type": "gp3",
                "iops": 3000,
                "throughput_mibps": 125,
            },
            "data_disk_count": 0,
            "exact_tag_count": 15,
            "spot_allowed": False,
            "capacity_block_allowed": False,
        },
        "direct_and_ambiguous_describe_readback_required": True,
        "termination_only_liability_preserved": True,
        "terminal_writer": {
            "bucket": ("keep-glm52-models-246813579024-us-west-2"),
            "key_pattern": (
                "campaigns/glm52-sky-20260724/submissions/production/"
                "generations/<generation_text>/workflow/"
                "LAUNCH_OUTCOME.json"
            ),
            "exact_version_required": True,
            "lost_response_readback_required": True,
        },
    }
    artifacts: dict[str, Mapping[str, object]] = {
        "support-input-contract-v1.json": input_contract,
        "support-template-contract-v1.json": template_contract,
        "support-spend-envelope-v1.json": spend_contract,
        "support-task11-workflow-v1.json": task11_workflow_contract,
        "support-task12-runtime-v1.json": task12_runtime_contract,
        "support-task10-production-v1.json": task10_production_contract,
    }
    manifest_body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_h1g_support_contract_manifest_v1",
        "artifacts": [
            {
                "file_name": file_name,
                "bytes": len(canonical_json_bytes(projection)) + 1,
                "sha256": hashlib.sha256(
                    canonical_json_bytes(projection) + b"\n"
                ).hexdigest(),
                "canonical_body_sha256": hashlib.sha256(
                    canonical_json_bytes(projection)
                ).hexdigest(),
            }
            for file_name, projection in artifacts.items()
        ],
        "contains_live_aws_identifiers": False,
        "requires_authenticated_bound_build": True,
    }
    manifest_body["canonical_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest_body)
    ).hexdigest()
    artifacts["support-contract-manifest-v1.json"] = manifest_body
    return artifacts
