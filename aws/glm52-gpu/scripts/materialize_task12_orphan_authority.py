#!/usr/bin/env python3
"""Produce PRECREATE and versioned POSTCREATE Task 12 orphan authority."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.support_plane import (  # noqa: E402
    ACCOUNT_ID,
    REGION,
    support_inputs_from_mapping,
)
from glm52_enforcement.task12_orphan_audit import RetainedResource  # noqa: E402
from glm52_enforcement.task12_orphan_authority import (  # noqa: E402
    build_activation_orphan_authority,
    capture_precreate_baseline,
    publish_activation_authority,
    read_activation_authority_coordinate,
    read_direct_grant_evidence,
    write_o_excl_authority,
)


def _canonical(path: Path) -> Mapping[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(str(path) + " is not canonical ASCII JSON") from exc
    if (
        type(value) is not dict
        or raw
        not in {
            canonical_json_bytes(value),
            canonical_json_bytes(value) + b"\n",
        }
    ):
        raise ValueError(str(path) + " bytes are not canonical JSON")
    return value


def _clients(profile: str) -> Mapping[str, object]:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("boto3 and botocore are required") from exc
    config = Config(
        region_name=REGION,
        connect_timeout=5,
        read_timeout=30,
        retries={"mode": "standard", "total_max_attempts": 1},
    )
    session = boto3.Session(profile_name=profile, region_name=REGION)
    if session.region_name != REGION:
        raise ValueError("AWS session region drifted")
    services = (
        "sts",
        "kms",
        "dynamodb",
        "s3",
        "cloudtrail",
        "cloudformation",
    )
    return {
        name: session.client(name, config=config)
        for name in services
    }


def _metadata(value: object, label: str) -> str:
    metadata = value.get("ResponseMetadata") if type(value) is dict else None
    if (
        type(metadata) is not dict
        or metadata.get("HTTPStatusCode") != 200
        or metadata.get("RetryAttempts") != 0
        or type(metadata.get("RequestId")) is not str
        or not metadata["RequestId"]
    ):
        raise ValueError(label + " is not authenticated zero-retry success")
    return metadata["RequestId"]


def _guard_account(clients: Mapping[str, object]) -> None:
    response = clients["sts"].get_caller_identity()
    _metadata(response, "GetCallerIdentity")
    if response.get("Account") != ACCOUNT_ID:
        raise ValueError("AWS caller account is foreign")


def _change_set_ready_and_unexecuted(
    *,
    cloudformation: object,
    stack_name: str,
    change_set_name: str,
) -> None:
    response = cloudformation.describe_change_set(
        StackName=stack_name,
        ChangeSetName=change_set_name,
    )
    _metadata(response, "DescribeChangeSet")
    if (
        response.get("Status") != "CREATE_COMPLETE"
        or response.get("ExecutionStatus") != "AVAILABLE"
        or response.get("ChangeSetName") != change_set_name
        or response.get("StackId") != stack_name
    ):
        raise ValueError(
            "PRECREATE requires the reviewed support change set before execution"
        )


def _cloudtrail_events(
    *,
    client: object,
    start: datetime,
    end: datetime,
) -> tuple[Mapping[str, object], ...]:
    token = None
    seen = set()
    events = []
    while True:
        request: dict[str, object] = {
            "LookupAttributes": [
                {
                    "AttributeKey": "EventName",
                    "AttributeValue": "CreateGrant",
                }
            ],
            "StartTime": start,
            "EndTime": end,
            "MaxResults": 50,
        }
        if token is not None:
            request["NextToken"] = token
        response = client.lookup_events(**request)
        _metadata(response, "LookupEvents")
        page = response.get("Events")
        if type(page) is not list or any(type(item) is not dict for item in page):
            raise ValueError("CloudTrail CreateGrant page is malformed")
        events.extend(page)
        next_token = response.get("NextToken")
        if next_token is None:
            break
        if type(next_token) is not str or not next_token or next_token in seen:
            raise ValueError("CloudTrail CreateGrant pagination is incomplete")
        seen.add(next_token)
        token = next_token
    return tuple(events)


def _utc(value: str) -> datetime:
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    return parsed.replace(tzinfo=timezone.utc)


def _precreate(args: argparse.Namespace, clients: Mapping[str, object]) -> None:
    _change_set_ready_and_unexecuted(
        cloudformation=clients["cloudformation"],
        stack_name=args.support_stack_name,
        change_set_name=args.change_set_name,
    )
    expected = (
        RetainedResource("LEDGER", args.ledger_table, "DDB_RETAINED"),
        RetainedResource("KMS_KEY", args.kms_key_arn, "KMS_RETAINED"),
        RetainedResource(
            "EVIDENCE_BUCKET",
            args.evidence_bucket,
            "S3_RETAINED",
        ),
        RetainedResource(
            "PRODUCTION_FENCE_STACK",
            args.fence_stack_id,
            "CFN_RETAINED",
        ),
        RetainedResource(
            "LIFECYCLE_RESOURCE",
            args.lifecycle_resource_id,
            args.lifecycle_cost_class,
        ),
        RetainedResource(
            "SOURCE_PUBLISHER_IDENTITY",
            args.source_publisher_arn,
            "IAM_RETAINED",
        ),
    )
    observed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    value = capture_precreate_baseline(
        kms=clients["kms"],
        activation_id=args.activation_id,
        retained_kms_key_arn=args.kms_key_arn,
        expected_retained=expected,
        observed_at=observed_at,
    )
    result = write_o_excl_authority(path=args.output, value=value)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


def _postcreate(args: argparse.Namespace, clients: Mapping[str, object]) -> None:
    baseline = _canonical(args.baseline)
    support_inputs = support_inputs_from_mapping(_canonical(args.support_inputs))
    postcreate_manifest = _canonical(args.postcreate_manifest)
    if support_inputs.activation_id != args.activation_id:
        raise ValueError("POSTCREATE activation input drifted")
    retained = baseline.get("expected_retained")
    evidence_buckets = (
        [
            item.get("resource_id")
            for item in retained
            if type(item) is dict
            and item.get("resource_type") == "EVIDENCE_BUCKET"
        ]
        if type(retained) is list
        else []
    )
    if evidence_buckets != [support_inputs.model_bucket_name]:
        raise ValueError(
            "POSTCREATE evidence bucket does not match the deployed model bucket"
        )
    existing = read_activation_authority_coordinate(
        dynamodb=clients["dynamodb"],
        ledger_table_name=support_inputs.ledger_table_name,
        activation_id=args.activation_id,
    )
    if existing is not None:
        expected_key = (
            f"campaigns/{support_inputs.run_id}/task12/orphans/"
            f"{args.activation_id}/"
            f"{existing['authority_body_sha256']}.json"
        )
        if (
            existing.get("bucket") != args.authority_bucket
            or existing.get("key") != expected_key
        ):
            raise ValueError(
                "POSTCREATE durable authority coordinate is foreign"
            )
        print(json.dumps(existing, sort_keys=True, separators=(",", ":")))
        return
    manifest_body = dict(postcreate_manifest)
    manifest_identity = manifest_body.pop("canonical_body_sha256", None)
    expected_coordinate_prefix = (
        f"ACTIVATION#{args.activation_id}#CUSTOM_RESOURCE_GRANT#"
    )
    coordinate = postcreate_manifest.get(
        "direct_grant_evidence_coordinate"
    )
    if (
        postcreate_manifest.get("record_type")
        != "glm52_h1g_support_postcreate_manifest_v1"
        or postcreate_manifest.get("activation_id") != args.activation_id
        or manifest_identity
        != hashlib.sha256(
            canonical_json_bytes(manifest_body)
        ).hexdigest()
        or type(coordinate) is not str
        or not coordinate.startswith(expected_coordinate_prefix)
    ):
        raise ValueError("POSTCREATE manifest/direct-grant coordinate drifted")
    request_id = coordinate.removeprefix(expected_coordinate_prefix)
    direct = read_direct_grant_evidence(
        dynamodb=clients["dynamodb"],
        ledger_table_name=support_inputs.ledger_table_name,
        activation_id=args.activation_id,
        custom_resource_request_id=request_id,
    )
    if direct["canonical_body_sha256"] != postcreate_manifest.get(
        "direct_grant_evidence_sha256"
    ):
        raise ValueError("POSTCREATE direct-grant ledger digest drifted")
    events = _cloudtrail_events(
        client=clients["cloudtrail"],
        start=_utc(args.cloudtrail_start),
        end=_utc(args.cloudtrail_end),
    )
    observed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    authority = build_activation_orphan_authority(
        baseline=baseline,
        support_inputs=support_inputs,
        direct_grant_evidence=direct,
        active_kms=clients["kms"],
        cloudtrail_events=events,
        observed_at=observed_at,
        settling_window_seconds=args.settling_window_seconds,
    )
    expected_key = (
        f"campaigns/{support_inputs.run_id}/task12/orphans/"
        f"{args.activation_id}/{authority['canonical_body_sha256']}.json"
    )
    if args.authority_bucket != support_inputs.model_bucket_name:
        raise ValueError("POSTCREATE versioned authority coordinate is not exact")
    coordinate = publish_activation_authority(
        s3=clients["s3"],
        dynamodb=clients["dynamodb"],
        ledger_table_name=support_inputs.ledger_table_name,
        bucket=args.authority_bucket,
        key=expected_key,
        authority=authority,
        observed_at=observed_at,
    )
    print(json.dumps(coordinate, sort_keys=True, separators=(",", ":")))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    pre = subparsers.add_parser("PRECREATE")
    for name in (
        "activation-id",
        "kms-key-arn",
        "ledger-table",
        "evidence-bucket",
        "fence-stack-id",
        "lifecycle-resource-id",
        "lifecycle-cost-class",
        "source-publisher-arn",
        "support-stack-name",
        "change-set-name",
    ):
        pre.add_argument("--" + name, required=True)
    pre.add_argument("--output", required=True, type=Path)

    post = subparsers.add_parser("POSTCREATE")
    post.add_argument("--activation-id", required=True)
    post.add_argument("--baseline", required=True, type=Path)
    post.add_argument("--support-inputs", required=True, type=Path)
    post.add_argument("--postcreate-manifest", required=True, type=Path)
    post.add_argument("--cloudtrail-start", required=True)
    post.add_argument("--cloudtrail-end", required=True)
    post.add_argument("--settling-window-seconds", required=True, type=int)
    post.add_argument("--authority-bucket", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.phase == "PRECREATE" and not args.output.is_absolute():
        raise ValueError("PRECREATE output must be an absolute path")
    clients = _clients(args.profile)
    _guard_account(clients)
    if args.phase == "PRECREATE":
        _precreate(args, clients)
    else:
        _postcreate(args, clients)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
