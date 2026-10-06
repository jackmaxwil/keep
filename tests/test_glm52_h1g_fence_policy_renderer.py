"""H.1g deterministic S3 fence-policy renderer contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import replace
from types import MappingProxyType

from fnmatch import fnmatchcase
import pytest

from glm52_enforcement.canonical import canonical_json_bytes


def _api():
    try:
        from glm52_enforcement import fence_policy_renderer
    except ImportError:
        pytest.fail("H.1g fence-policy renderer implementation is missing", pytrace=False)
    return fence_policy_renderer


def _identity(index: int = 1):
    api = _api()
    return api.PrincipalIdentity(
        binding_id=f"writer-{index}",
        arn=f"arn:aws:iam::246813579024:role/keep-glm52-writer-{index}",
        role_id=f"AROATESTWRITER{index:03d}",
    )


def _cohort(
    *, index: int = 1, resource: str | None = None, member_count: int = 1
):
    api = _api()
    return api.WriterCohort(
        cohort_id=f"reservation-{index}",
        members=tuple(
            _identity(index + offset) for offset in range(member_count)
        ),
        guard_resources=(
            resource
            or "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2/"
            "campaigns/glm52-sky-20260724/*",
        ),
        cross_member_denial_evidence_sha256="1" * 64,
    )


def _source_families():
    api = _api()
    return tuple(
        api.ReservedFamily(family_id=family_id, resources=(resource,))
        for family_id, resource in api.SOURCE_FAMILY_SPECS
    )


def _stage_evidence(head):
    api = _api()
    predecessor = "a" * 64
    if head is api.PolicyHead.BRIDGE_SEED:
        return {}
    values = {
        "predecessor_policy_sha256": predecessor,
        "publisher_deny_policy_sha256": "9" * 64,
    }
    if head in {
        api.PolicyHead.SOURCE_FAMILIES_FROZEN,
        api.PolicyHead.BATCH,
        api.PolicyHead.CLOSED_SOURCE,
        api.PolicyHead.TERMINAL,
    }:
        values["freeze_denial_evidence_sha256"] = "b" * 64
    if head in {api.PolicyHead.BATCH, api.PolicyHead.TERMINAL}:
        values.update(
            {
                "source_publication_sealed_sha256": "c" * 64,
                "all_version_inventory_sha256": "d" * 64,
                "source_settlement_sha256": "e" * 64,
                "publisher_deny_policy_sha256": "8" * 64,
            }
        )
    if head in {api.PolicyHead.CLOSED_SOURCE, api.PolicyHead.TERMINAL}:
        values["terminal_prerequisite_sha256"] = "f" * 64
    return values


def _render_input(**overrides):
    api = _api()
    values = {
        "build_mode": api.BuildMode.LEGACY_DISABLED,
        "policy_head": api.PolicyHead.BRIDGE_SEED,
        "bucket_name": "keep-glm52-models-246813579024-us-west-2",
        "account_id": "246813579024",
        "kms_key_arn": (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "11111111-2222-3333-4444-555555555555"
        ),
        "writer_cohorts": (_cohort(member_count=2),),
        "reserved_families": _source_families(),
    }
    values.update(
        _stage_evidence(overrides.get("policy_head", values["policy_head"]))
    )
    values.update(overrides)
    return api.FencePolicyInput(**values)


def _slot_input(head):
    api = _api()
    if head is api.PolicyHead.BRIDGE_SEED:
        return _render_input()
    validation_reader = api.PrincipalIdentity(
        binding_id="source-validation-reader",
        arn=(
            "arn:aws:iam::246813579024:"
            "role/keep-glm52-source-validation-reader"
        ),
        role_id="AROATESTSOURCEVALIDATION",
    )
    inventory_reader = api.PrincipalIdentity(
        binding_id="source-inventory-reader",
        arn=(
            "arn:aws:iam::246813579024:"
            "role/keep-glm52-source-inventory-reader"
        ),
        role_id="AROATESTSOURCEINVENTORY",
    )
    terminal_reader = api.PrincipalIdentity(
        binding_id="terminal-audit-reader",
        arn=(
            "arn:aws:iam::246813579024:"
            "role/keep-glm52-terminal-audit-reader"
        ),
        role_id="AROATESTTERMINALAUDIT",
    )
    selected_keys = (
        "campaigns/glm52-sky-20260724/spend-snapshots/"
        + "a" * 64
        + "/GPU_SPEND_SNAPSHOT.json",
        "campaigns/glm52-sky-20260724/submissions/production/intents/"
        + "b" * 64
        + "/SKYPILOT_SUBMISSION_INTENT.json",
        "campaigns/glm52-sky-20260724/production/controller-baselines/"
        + "c" * 64
        + "/CONTROLLER_BASELINE.json",
        "campaigns/glm52-sky-20260724/monitor/must-start/production/"
        + "d" * 64
        + "/control-plane-ready/"
        + "e" * 64
        + "/CONTROL_PLANE_READY.json",
        "campaigns/glm52-sky-20260724/submissions/production/acquisitions/"
        + "f" * 64
        + "/SUBMISSION_ACQUIRED.json",
    )
    nonselected_keys = (
        "campaigns/glm52-sky-20260724/spend-snapshots/"
        + "1" * 64
        + "/GPU_SPEND_SNAPSHOT.json",
    )
    selected_resources = tuple(
        api.FIXED_BUCKET_ARN + "/" + key for key in selected_keys
    )
    terminal = head in {api.PolicyHead.CLOSED_SOURCE, api.PolicyHead.TERMINAL}
    if head is api.PolicyHead.RESERVATION:
        writer_cohort = _cohort(index=20, member_count=7)
    elif terminal:
        writer_cohort = None
    else:
        writer_cohort = _cohort(index=20, member_count=2)
    baseline_retired_resources = (
        "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2/"
        "campaigns/glm52-sky-20260724/retired/genesis.json",
    )
    source_publishers = tuple(_identity(index) for index in range(22, 27))
    source_retired_resources = (
        selected_resources
        if head in {api.PolicyHead.BATCH, api.PolicyHead.TERMINAL}
        else ()
    )
    return _render_input(
        policy_head=head,
        permanent_enrolled_resources=(
            "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2/"
            "campaigns/glm52-sky-20260724/authorities/fence/FENCE_GENESIS.json",
        ),
        retired_publishers=(
            (_identity(9),) + source_publishers
            if head in {api.PolicyHead.BATCH, api.PolicyHead.TERMINAL}
            else (_identity(9),)
        ),
        retired_publisher_resources=(
            baseline_retired_resources + source_retired_resources
        ),
        reserved_families=(
            () if head is api.PolicyHead.RESERVATION else _source_families()
        ),
        writer_cohorts=(() if writer_cohort is None else (writer_cohort,)),
        writer_owned_resources=api.CORE_WRITER_RESOURCES,
        source_validation_readers=(validation_reader,),
        source_inventory_readers=(inventory_reader,),
        terminal_audit_readers=(terminal_reader,),
        selected_source_keys=(
            selected_keys
            if head in {api.PolicyHead.BATCH, api.PolicyHead.TERMINAL}
            else ()
        ),
        nonselected_source_keys=(
            nonselected_keys
            if head in {api.PolicyHead.BATCH, api.PolicyHead.TERMINAL}
            else ()
        ),
    )


def _bound_policy_inputs():
    api = _api()
    heads = (
        api.PolicyHead.BRIDGE_SEED,
        api.PolicyHead.PREPARE,
        api.PolicyHead.RESERVATION,
        api.PolicyHead.SOURCE_FAMILIES_FROZEN,
        api.PolicyHead.BATCH,
        api.PolicyHead.CLOSED_SOURCE,
        api.PolicyHead.TERMINAL,
    )
    provisional = tuple(_slot_input(head) for head in heads)
    hashes = {
        value.policy_head: api.render_fence_policy(value).policy_sha256
        for value in provisional
    }
    predecessors = {
        api.PolicyHead.BRIDGE_SEED: None,
        api.PolicyHead.PREPARE: api.PolicyHead.BRIDGE_SEED,
        api.PolicyHead.RESERVATION: api.PolicyHead.PREPARE,
        api.PolicyHead.SOURCE_FAMILIES_FROZEN: api.PolicyHead.RESERVATION,
        api.PolicyHead.BATCH: api.PolicyHead.SOURCE_FAMILIES_FROZEN,
        api.PolicyHead.CLOSED_SOURCE: api.PolicyHead.SOURCE_FAMILIES_FROZEN,
        api.PolicyHead.TERMINAL: api.PolicyHead.BATCH,
    }
    bound = []
    for value in provisional:
        predecessor = predecessors[value.policy_head]
        predecessor_sha256 = (
            None if predecessor is None else hashes[predecessor]
        )
        bound.append(
            replace(
                value,
                predecessor_policy_sha256=predecessor_sha256,
            )
        )
    return tuple(bound), predecessors


def _policy_set_sha256(policies, input_identities):
    api = _api()
    digest_input = [
        {
            "policy_head": head.value,
            "policy_sha256": policies[head].policy_sha256,
            "render_input_identity_sha256": input_identities[head],
            "rendered_policy_bytes": policies[head].rendered_policy_bytes,
        }
        for head in (
            api.PolicyHead.BRIDGE_SEED,
            api.PolicyHead.PREPARE,
            api.PolicyHead.RESERVATION,
            api.PolicyHead.SOURCE_FAMILIES_FROZEN,
            api.PolicyHead.BATCH,
            api.PolicyHead.CLOSED_SOURCE,
            api.PolicyHead.TERMINAL,
        )
    ]
    return hashlib.sha256(canonical_json_bytes(digest_input)).hexdigest()


def _condition_matches(condition, context):
    for operator, clauses in condition.items():
        for key, expected in clauses.items():
            expected_values = expected if isinstance(expected, list) else [expected]
            present = key in context
            actual = context.get(key)
            if operator == "Null":
                matched = (not present) is (expected == "true")
            elif operator in {"Bool", "BoolIfExists"}:
                matched = (
                    True
                    if operator == "BoolIfExists" and not present
                    else present and str(actual).lower() == str(expected).lower()
                )
            elif operator in {"StringNotEquals", "StringNotEqualsIfExists"}:
                matched = not present or actual not in expected_values
            elif operator == "ArnNotEqualsIfExists":
                matched = not present or actual not in expected_values
            elif operator == "StringNotLikeIfExists":
                matched = not present or not any(
                    fnmatchcase(str(actual), pattern)
                    for pattern in expected_values
                )
            elif operator == "ArnEquals":
                matched = present and actual in expected_values
            elif operator == "StringLike":
                matched = present and any(
                    fnmatchcase(str(actual), pattern)
                    for pattern in expected_values
                )
            else:
                raise AssertionError(f"unsupported test condition operator: {operator}")
            if not matched:
                return False
    return True


def _matching_deny_sids(policy, *, action, resource, context):
    matches = []
    for statement in policy["Statement"]:
        if not any(fnmatchcase(action, pattern) for pattern in statement["Action"]):
            continue
        if not any(fnmatchcase(resource, pattern) for pattern in statement["Resource"]):
            continue
        if "Condition" in statement and not _condition_matches(
            statement["Condition"], context
        ):
            continue
        matches.append(statement["Sid"])
    return matches



def test_red_frozen_action_arrays_are_real_s3_iam_actions() -> None:
    """Break caught: SDK operations or new object mutations escape the deny set."""
    api = _api()

    assert api.CREATE == ("s3:PutObject",)
    assert api.NON_CREATE_MUTATION == (
        "s3:AbortMultipartUpload",
        "s3:DeleteObject",
        "s3:DeleteObjectVersion",
        "s3:PutObjectAcl",
        "s3:PutObjectVersionAcl",
        "s3:PutObjectTagging",
        "s3:PutObjectVersionTagging",
        "s3:DeleteObjectTagging",
        "s3:DeleteObjectVersionTagging",
        "s3:PutObjectAnnotation",
        "s3:PutObjectVersionAnnotation",
        "s3:DeleteObjectAnnotation",
        "s3:DeleteObjectVersionAnnotation",
        "s3:UpdateObjectEncryption",
        "s3:PutObjectLegalHold",
        "s3:PutObjectRetention",
        "s3:BypassGovernanceRetention",
        "s3:RestoreObject",
        "s3:ObjectOwnerOverrideToBucketOwner",
        "s3:InitiateReplication",
        "s3:ReplicateObject",
        "s3:ReplicateObjectAnnotation",
        "s3:ReplicateDelete",
        "s3:ReplicateTags",
    )
    assert api.FULL_MUTATION == api.CREATE + api.NON_CREATE_MUTATION
    assert api.SOURCE_OBJECT_READ == (
        "s3:GetObject",
        "s3:GetObjectAcl",
        "s3:GetObjectAnnotation",
        "s3:GetObjectAttributes",
        "s3:GetObjectLegalHold",
        "s3:GetObjectRetention",
        "s3:GetObjectTagging",
        "s3:GetObjectTorrent",
        "s3:GetObjectVersion",
        "s3:GetObjectVersionAcl",
        "s3:GetObjectVersionAnnotation",
        "s3:GetObjectVersionAnnotationForReplication",
        "s3:GetObjectVersionAttributes",
        "s3:GetObjectVersionForReplication",
        "s3:GetObjectVersionTagging",
        "s3:GetObjectVersionTorrent",
        "s3:ListObjectAnnotations",
        "s3:ListObjectVersionAnnotations",
        "s3:SelectObjectContent",
    )
    assert not {
        "s3:CopyObject",
        "s3:CreateMultipartUpload",
        "s3:CompleteMultipartUpload",
    }.intersection(api.FULL_MUTATION)


@pytest.mark.parametrize(
    "head_name",
    [
        "BRIDGE_SEED",
        "PREPARE",
        "RESERVATION",
        "SOURCE_FAMILIES_FROZEN",
        "BATCH",
        "CLOSED_SOURCE",
        "TERMINAL",
    ],
)
def test_red_every_closed_policy_head_renders_with_required_semantic_families(
    head_name,
) -> None:
    """Break caught: one legal head silently omits or retains a policy family."""
    api = _api()
    head = api.PolicyHead(head_name)
    rendered = api.render_fence_policy(_slot_input(head))
    sids = tuple(row.sid for row in rendered.statement_ledger)

    assert rendered.rendered_policy_bytes <= 17_920
    assert all(
        rendered.component_bytes[component] <= maximum
        for component, maximum in _slot_input(head).policy_limits.component_max_bytes.items()
    )
    source_closures = [
        row
        for row in rendered.policy["Statement"]
        if row["Sid"] == "DenyAllReservedFamilyMutation_" + head.value
    ]
    assert len(source_closures) == (0 if head is api.PolicyHead.RESERVATION else 1)
    if source_closures:
        assert source_closures[0]["Resource"] == [
            resource for _, resource in api.SOURCE_FAMILY_SPECS
        ]
    terminal = head in {api.PolicyHead.CLOSED_SOURCE, api.PolicyHead.TERMINAL}
    assert (any(sid.startswith("DenyNonCreateMutation_") for sid in sids)) is (
        not terminal
    )
    assert ("DenyTerminalClosedMutation" in sids) is terminal


def test_red_policy_head_shape_validation_rejects_semantic_omission_or_reopening() -> (
    None
):
    api = _api()
    with pytest.raises(ValueError, match="exact five source-family closures"):
        api.render_fence_policy(
            replace(_slot_input(api.PolicyHead.BRIDGE_SEED), reserved_families=())
        )
    with pytest.raises(ValueError, match="must not contain"):
        api.render_fence_policy(
            replace(
                _slot_input(api.PolicyHead.RESERVATION),
                reserved_families=_source_families(),
            )
        )
    with pytest.raises(ValueError, match="permanent enrolled-source"):
        api.render_fence_policy(
            replace(
                _slot_input(api.PolicyHead.PREPARE),
                permanent_enrolled_resources=(),
            )
        )
    with pytest.raises(ValueError, match="forbids an active writer cohort"):
        api.render_fence_policy(
            replace(
                _slot_input(api.PolicyHead.TERMINAL),
                writer_cohorts=(_cohort(),),
            )
        )


def test_red_rendered_hash_state_cannot_be_mutated_through_public_projections() -> None:
    """Break caught: a caller mutates policy/limits after their hashes are fixed."""
    api = _api()
    render_input = _render_input()
    rendered = api.render_fence_policy(render_input)
    statement_count = len(rendered.policy["Statement"])

    projected = rendered.policy
    projected["Statement"].clear()
    assert len(rendered.policy["Statement"]) == statement_count
    assert hashlib.sha256(rendered.policy_bytes).hexdigest() == rendered.policy_sha256
    with pytest.raises(TypeError):
        rendered.component_bytes["global_guards"] = 0
    with pytest.raises(TypeError):
        render_input.policy_limits.component_max_bytes["global_guards"] = 0
    with pytest.raises(TypeError, match="immutable tuple"):
        api.render_fence_policy(
            replace(render_input, writer_cohorts=[_cohort()])
        )


def test_red_golden_reservation_policy_is_byte_exact_and_fully_ledgered() -> None:
    """Break caught: statement/key/array order or byte accounting drifts."""
    api = _api()
    rendered = api.render_fence_policy(_render_input())
    bucket = "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2"
    guard = bucket + "/campaigns/glm52-sky-20260724/*"
    arns = [
        "arn:aws:iam::246813579024:role/keep-glm52-writer-1",
        "arn:aws:iam::246813579024:role/keep-glm52-writer-2",
    ]
    role_ids = ["AROATESTWRITER001:*", "AROATESTWRITER002:*"]
    kms = "arn:aws:kms:us-west-2:246813579024:key/11111111-2222-3333-4444-555555555555"
    non_create = list(api.NON_CREATE_MUTATION)

    expected = {
        "Statement": [
            {
                "Action": ["s3:*"],
                "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                "Effect": "Deny",
                "Principal": "*",
                "Resource": [bucket, bucket + "/*"],
                "Sid": "DenyInsecureTransport",
            },
            {
                "Action": ["s3:*"],
                "Condition": {
                    "BoolIfExists": {
                        "aws:PrincipalIsAWSService": "false"
                    },
                    "Null": {"aws:PrincipalAccount": "false"},
                    "StringNotEquals": {"aws:PrincipalAccount": "246813579024"},
                },
                "Effect": "Deny",
                "Principal": "*",
                "Resource": [bucket, bucket + "/*"],
                "Sid": "DenyWrongPrincipalAccount",
            },
            {
                "Action": ["s3:*"],
                "Condition": {
                    "BoolIfExists": {"aws:PrincipalIsAWSService": "false"},
                    "Null": {"aws:PrincipalAccount": "true"},
                },
                "Effect": "Deny",
                "Principal": "*",
                "Resource": [bucket, bucket + "/*"],
                "Sid": "DenyMissingPrincipalAccount",
            },
            {
                "Action": list(api.FULL_MUTATION),
                "Effect": "Deny",
                "Principal": "*",
                "Resource": [
                    bucket
                    + "/campaigns/glm52-sky-20260724/spend-snapshots/*/"
                    "GPU_SPEND_SNAPSHOT.json",
                    bucket
                    + "/campaigns/glm52-sky-20260724/submissions/production/"
                    "intents/*/SKYPILOT_SUBMISSION_INTENT.json",
                    bucket
                    + "/campaigns/glm52-sky-20260724/production/"
                    "controller-baselines/*/CONTROLLER_BASELINE.json",
                    bucket
                    + "/campaigns/glm52-sky-20260724/monitor/must-start/"
                    "production/*/control-plane-ready/*/"
                    "CONTROL_PLANE_READY.json",
                    bucket
                    + "/campaigns/glm52-sky-20260724/submissions/production/"
                    "acquisitions/*/SUBMISSION_ACQUIRED.json",
                ],
                "Sid": "DenyAllReservedFamilyMutation_BRIDGE_SEED",
            },
            {
                "Action": ["s3:PutObject"],
                "Condition": {
                    "ArnNotEqualsIfExists": {"aws:PrincipalArn": arns}
                },
                "Effect": "Deny",
                "Principal": "*",
                "Resource": [guard],
                "Sid": "DenyCreateWrongWriterArn_reservation-1",
            },
            {
                "Action": ["s3:PutObject"],
                "Condition": {
                    "StringNotLikeIfExists": {"aws:userid": role_ids}
                },
                "Effect": "Deny",
                "Principal": "*",
                "Resource": [guard],
                "Sid": "DenyCreateWrongWriterId_reservation-1",
            },
            {
                "Action": ["s3:PutObject"],
                "Condition": {"Null": {"s3:if-none-match": "true"}},
                "Effect": "Deny",
                "Principal": "*",
                "Resource": [guard],
                "Sid": "DenyMissingIfNoneMatch_reservation-1",
            },
            {
                "Action": ["s3:PutObject"],
                "Condition": {"StringNotEquals": {"s3:if-none-match": "*"}},
                "Effect": "Deny",
                "Principal": "*",
                "Resource": [guard],
                "Sid": "DenyWrongIfNoneMatch_reservation-1",
            },
            {
                "Action": ["s3:PutObject"],
                "Condition": {"Null": {"s3:x-amz-copy-source": "false"}},
                "Effect": "Deny",
                "Principal": "*",
                "Resource": [guard],
                "Sid": "DenyCopySourceHeader_reservation-1",
            },
            {
                "Action": ["s3:PutObject"],
                "Condition": {
                    "StringNotEqualsIfExists": {
                        "s3:x-amz-server-side-encryption": "aws:kms"
                    }
                },
                "Effect": "Deny",
                "Principal": "*",
                "Resource": [guard],
                "Sid": "DenyWrongSseAlgorithm_reservation-1",
            },
            {
                "Action": ["s3:PutObject"],
                "Condition": {
                    "StringNotEqualsIfExists": {
                        "s3:x-amz-server-side-encryption-aws-kms-key-id": kms
                    }
                },
                "Effect": "Deny",
                "Principal": "*",
                "Resource": [guard],
                "Sid": "DenyWrongSseKmsKey_reservation-1",
            },
            {
                "Action": non_create,
                "Effect": "Deny",
                "Principal": "*",
                "Resource": [guard],
                "Sid": "DenyNonCreateMutation_reservation-1",
            },
        ],
        "Version": "2012-10-17",
    }
    expected_bytes = canonical_json_bytes(expected)

    assert rendered.policy == expected
    assert rendered.policy_bytes == expected_bytes
    assert rendered.policy_sha256 == (
        "616d4fe7ced2f74bed57e6e964e8c82585c31e31104e330f1d2c698dabd20c0b"
    )
    assert rendered.rendered_policy_bytes == len(expected_bytes)
    assert tuple(row.sid for row in rendered.statement_ledger) == tuple(
        statement["Sid"] for statement in expected["Statement"]
    )
    assert rendered.statement_ledger[0].leading_comma_bytes == 0
    assert all(row.leading_comma_bytes == 1 for row in rendered.statement_ledger[1:])
    for row, statement in zip(rendered.statement_ledger, expected["Statement"]):
        statement_bytes = canonical_json_bytes(statement)
        assert row.statement_bytes == len(statement_bytes)
        assert row.statement_sha256 == hashlib.sha256(statement_bytes).hexdigest()
        assert rendered.policy_bytes[row.start_offset : row.end_offset] == statement_bytes
    assert sum(rendered.component_bytes.values()) == len(expected_bytes)
    assert json.loads(rendered.policy_bytes) == expected


def test_red_checked_in_golden_artifact_reconciles_to_renderer() -> None:
    """Break caught: the review artifact and executable renderer diverge."""
    api = _api()
    artifact_path = (
        Path(__file__).parents[1]
        / ".superpowers/sdd/goal-objective/"
        "task-3ph1g-h1g-fence-policy-renderer-golden-v1.json"
    )
    raw = artifact_path.read_bytes()
    assert b"\n" not in raw
    artifact = json.loads(raw)
    identity = artifact.pop("canonical_identity_sha256")
    assert identity == (
        "f0f1d4d8e03e399d3090813f7b196702ee00dbc8161628c3101d33d326dd29bc"
    )
    assert hashlib.sha256(canonical_json_bytes(artifact)).hexdigest() == identity
    artifact["canonical_identity_sha256"] = identity

    rendered = api.render_fence_policy(_render_input())
    assert artifact["policy"] == rendered.policy
    assert artifact["policy_sha256"] == rendered.policy_sha256
    assert artifact["rendered_policy_bytes"] == rendered.rendered_policy_bytes
    assert artifact["statement_ledger"] == [
        {
            "component": row.component,
            "end_offset": row.end_offset,
            "leading_comma_bytes": row.leading_comma_bytes,
            "sequence": row.sequence,
            "sid": row.sid,
            "start_offset": row.start_offset,
            "statement_bytes": row.statement_bytes,
            "statement_sha256": row.statement_sha256,
        }
        for row in rendered.statement_ledger
    ]


def test_red_writer_guards_fail_closed_for_every_create_mutant() -> None:
    """Break caught: one malformed create or non-create action escapes all denies."""
    api = _api()
    rendered = api.render_fence_policy(_render_input())
    by_sid = {statement["Sid"]: statement for statement in rendered.policy["Statement"]}

    assert by_sid["DenyCreateWrongWriterArn_reservation-1"]["Condition"] == {
        "ArnNotEqualsIfExists": {
            "aws:PrincipalArn": [
                "arn:aws:iam::246813579024:role/keep-glm52-writer-1",
                "arn:aws:iam::246813579024:role/keep-glm52-writer-2",
            ]
        }
    }
    assert by_sid["DenyCreateWrongWriterId_reservation-1"]["Condition"] == {
        "StringNotLikeIfExists": {
            "aws:userid": [
                "AROATESTWRITER001:*",
                "AROATESTWRITER002:*",
            ]
        }
    }
    assert by_sid["DenyMissingIfNoneMatch_reservation-1"]["Condition"] == {
        "Null": {"s3:if-none-match": "true"}
    }
    assert by_sid["DenyWrongIfNoneMatch_reservation-1"]["Condition"] == {
        "StringNotEquals": {"s3:if-none-match": "*"}
    }
    assert by_sid["DenyCopySourceHeader_reservation-1"]["Condition"] == {
        "Null": {"s3:x-amz-copy-source": "false"}
    }
    assert set(by_sid["DenyNonCreateMutation_reservation-1"]["Action"]) == set(
        api.NON_CREATE_MUTATION
    )


def test_red_policy_semantics_kill_every_writer_and_global_guard_mutant() -> None:
    """Break caught: exact JSON looks right but a Deny condition is ineffective."""
    api = _api()
    policy = api.render_fence_policy(_render_input()).policy
    guard_resource = (
        "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2/"
        "campaigns/glm52-sky-20260724/authorities/fence/FENCE_GENESIS.json"
    )
    context = {
        "aws:SecureTransport": "true",
        "aws:PrincipalAccount": "246813579024",
        "aws:PrincipalIsAWSService": "false",
        "aws:PrincipalArn": (
            "arn:aws:iam::246813579024:role/keep-glm52-writer-1"
        ),
        "aws:userid": "AROATESTWRITER001:session",
        "s3:if-none-match": "*",
        "s3:x-amz-server-side-encryption": "aws:kms",
        "s3:x-amz-server-side-encryption-aws-kms-key-id": (
            "arn:aws:kms:us-west-2:246813579024:key/"
            "11111111-2222-3333-4444-555555555555"
        ),
    }

    assert _matching_deny_sids(
        policy,
        action="s3:PutObject",
        resource=guard_resource,
        context=context,
    ) == []

    mutations = [
        (
            {"aws:PrincipalArn": "arn:aws:iam::246813579024:role/wrong"},
            "DenyCreateWrongWriterArn_reservation-1",
        ),
        (
            {"aws:userid": "AROATESTWRITER999:session"},
            "DenyCreateWrongWriterId_reservation-1",
        ),
        (
            {"s3:if-none-match": None},
            "DenyMissingIfNoneMatch_reservation-1",
        ),
        (
            {"s3:if-none-match": "\"etag\""},
            "DenyWrongIfNoneMatch_reservation-1",
        ),
        (
            {"s3:x-amz-copy-source": "/foreign/key"},
            "DenyCopySourceHeader_reservation-1",
        ),
        (
            {"s3:x-amz-server-side-encryption": None},
            "DenyWrongSseAlgorithm_reservation-1",
        ),
        (
            {"s3:x-amz-server-side-encryption": "AES256"},
            "DenyWrongSseAlgorithm_reservation-1",
        ),
        (
            {
                "s3:x-amz-server-side-encryption-aws-kms-key-id": (
                    "arn:aws:kms:us-west-2:246813579024:key/wrong"
                )
            },
            "DenyWrongSseKmsKey_reservation-1",
        ),
    ]
    for updates, expected_sid in mutations:
        mutated = dict(context)
        for key, value in updates.items():
            if value is None:
                mutated.pop(key, None)
            else:
                mutated[key] = value
        assert expected_sid in _matching_deny_sids(
            policy,
            action="s3:PutObject",
            resource=guard_resource,
            context=mutated,
        )

    for action in api.NON_CREATE_MUTATION:
        assert "DenyNonCreateMutation_reservation-1" in _matching_deny_sids(
            policy,
            action=action,
            resource=guard_resource,
            context=context,
        )

    source_resource = (
        "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2/"
        "campaigns/glm52-sky-20260724/spend-snapshots/"
        + "a" * 64
        + "/GPU_SPEND_SNAPSHOT.json"
    )
    assert (
        "DenyAllReservedFamilyMutation_BRIDGE_SEED"
        in _matching_deny_sids(
            policy,
            action="s3:PutObject",
            resource=source_resource,
            context=context,
        )
    )

    bucket = "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2"
    assert _matching_deny_sids(
        policy,
        action="s3:GetBucketLocation",
        resource=bucket,
        context={
            "aws:SecureTransport": "true",
            "aws:PrincipalIsAWSService": "true",
        },
    ) == []
    assert "DenyWrongPrincipalAccount" in _matching_deny_sids(
        policy,
        action="s3:GetBucketLocation",
        resource=bucket,
        context={
            "aws:SecureTransport": "true",
            "aws:PrincipalAccount": "111122223333",
        },
    )


@pytest.mark.parametrize(
    "change,match",
    [
        (
            {"writer_cohorts": ()},
            "requires exactly one active writer cohort",
        ),
        (
            {"writer_cohorts": (_cohort(index=1), _cohort(index=2))},
            "requires exactly one active writer cohort",
        ),
        (
            {"account_id": "98417495574"},
            "account_id",
        ),
        (
            {"bucket_name": "kéep-glm52-models-246813579024-us-west-2"},
            "ASCII",
        ),
        (
            {"policy_head": "RESERVATION"},
            "PolicyHead",
        ),
    ],
)
def test_red_strict_input_validation_rejects_missing_broad_or_noncanonical_input(
    change, match
) -> None:
    api = _api()
    with pytest.raises((TypeError, ValueError), match=match):
        api.render_fence_policy(_render_input(**change))


def test_red_duplicate_or_recreated_principal_identity_is_fatal() -> None:
    """Break caught: one ARN is accepted with two RoleIds or duplicate bindings."""
    api = _api()
    duplicate = api.PrincipalIdentity(
        binding_id="writer-recreated",
        arn=_identity().arn,
        role_id="AROARECREATED999",
    )
    cohort = api.WriterCohort(
        cohort_id="reservation-1",
        members=(_identity(), duplicate),
        guard_resources=(_cohort().guard_resources[0],),
        cross_member_denial_evidence_sha256="2" * 64,
    )
    with pytest.raises(ValueError, match="another RoleId"):
        api.render_fence_policy(_render_input(writer_cohorts=(cohort,)))


def test_red_legacy_mode_and_statement_order_are_authenticated_inputs() -> None:
    api = _api()
    legacy = (
        {
            "Sid": "LegacyDenyOne",
            "Effect": "Deny",
            "Principal": "*",
            "Action": ["s3:PutObject"],
            "Resource": [
                "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2/legacy/*"
            ],
        },
        {
            "Sid": "LegacyDenyTwo",
            "Effect": "Deny",
            "Principal": "*",
            "Action": ["s3:DeleteObject"],
            "Resource": [
                "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2/legacy/*"
            ],
        },
    )
    enabled = api.render_fence_policy(
        _render_input(
            build_mode=api.BuildMode.LEGACY_ENABLED,
            legacy_statements=legacy,
        )
    )
    assert [row.sid for row in enabled.statement_ledger][3:5] == [
        "LegacyDenyOne",
        "LegacyDenyTwo",
    ]

    with pytest.raises(ValueError, match="LEGACY_ENABLED requires"):
        api.render_fence_policy(
            _render_input(build_mode=api.BuildMode.LEGACY_ENABLED)
        )
    with pytest.raises(ValueError, match="LEGACY_DISABLED forbids"):
        api.render_fence_policy(_render_input(legacy_statements=legacy))


def test_red_policy_size_and_component_budgets_fail_closed() -> None:
    api = _api()
    relaxed = api.PolicyLimits(
        max_policy_bytes=20_480,
        unallocated_headroom_bytes=0,
    )
    with pytest.raises(ValueError, match="frozen H.1g budget"):
        api.render_fence_policy(_render_input(policy_limits=relaxed))

    oversized_legacy = (
        {
            "Sid": "LegacyOversizedDeny",
            "Effect": "Deny",
            "Principal": "*",
            "Action": ["s3:PutObject"],
            "Resource": [
                "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2/"
                f"legacy/{index:02d}/"
                + "x" * 80
                for index in range(32)
            ],
        },
    )
    with pytest.raises(
        api.PolicyComponentBudgetExhausted,
        match="legacy_fragment",
    ):
        api.render_fence_policy(
            _render_input(
                build_mode=api.BuildMode.LEGACY_ENABLED,
                legacy_statements=oversized_legacy,
            )
        )


def test_red_reader_and_terminal_constructors_preserve_exact_identity_order() -> None:
    api = _api()
    reader_two = api.PrincipalIdentity(
        binding_id="reader-2",
        arn="arn:aws:iam::246813579024:role/keep-glm52-reader-2",
        role_id="AROATESTREADER002",
    )
    reader_one = api.PrincipalIdentity(
        binding_id="reader-1",
        arn="arn:aws:iam::246813579024:role/keep-glm52-reader-1",
        role_id="AROATESTREADER001",
    )
    terminal_input = _slot_input(api.PolicyHead.TERMINAL)
    rendered = api.render_fence_policy(
        replace(
            terminal_input,
            terminal_audit_readers=(reader_two, reader_one),
        )
    )
    statements = {row["Sid"]: row for row in rendered.policy["Statement"]}
    assert statements[
        "DenySourceReadWrongReaderArn_nonselected-source"
    ]["Condition"] == {
        "ArnNotEqualsIfExists": {
            "aws:PrincipalArn": [reader_two.arn, reader_one.arn]
        }
    }
    assert statements[
        "DenySourceListWrongReaderId_inventory-audit"
    ]["Condition"] == {
        "StringNotLikeIfExists": {
            "aws:userid": [
                terminal_input.source_inventory_readers[0].role_id + ":*",
                reader_two.role_id + ":*",
                reader_one.role_id + ":*",
            ]
        }
    }
    assert statements["DenyTerminalClosedMutation"]["Action"] == list(
        api.FULL_MUTATION
    )


def test_red_complete_policy_set_is_pairwise_distinct_and_cross_bound() -> None:
    """Break caught: a slot aliases another head or names the wrong predecessor."""
    api = _api()
    bound, predecessors = _bound_policy_inputs()
    heads = tuple(value.policy_head for value in bound)
    rendered_set = api.render_fence_policy_set(bound)
    policy_hashes = tuple(
        rendered_set.policies[head].policy_sha256 for head in heads
    )
    assert len(set(policy_hashes)) == 7
    assert len(rendered_set.policy_set_sha256) == 64
    assert rendered_set.topology_predecessors == predecessors
    assert tuple(rendered_set.render_input_identity_sha256_by_head) == heads
    assert rendered_set.render_input_identity_sha256_by_head == {
        value.policy_head: value.render_input_identity_sha256
        for value in bound
    }
    with pytest.raises(TypeError):
        rendered_set.render_input_identity_sha256_by_head[
            api.PolicyHead.BRIDGE_SEED
        ] = "0" * 64
    with pytest.raises(TypeError):
        rendered_set.policies[api.PolicyHead.BRIDGE_SEED] = None

    bad_batch = replace(
        bound[4],
        predecessor_policy_sha256="0" * 64,
        publisher_deny_policy_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="predecessor policy hash"):
        api.render_fence_policy_set(
            bound[:4] + (bad_batch,) + bound[5:]
        )


@pytest.mark.parametrize("mutation", ["missing", "foreign", "wrong-order"])
def test_red_rendered_policy_set_constructor_requires_exact_policy_heads(
    mutation,
) -> None:
    api = _api()
    bound, _ = _bound_policy_inputs()
    rendered_set = api.render_fence_policy_set(bound)
    policies = dict(rendered_set.policies)
    bridge = api.PolicyHead.BRIDGE_SEED

    if mutation == "missing":
        policies.pop(bridge)
    elif mutation == "foreign":
        policies["BRIDGE_SEED"] = policies.pop(bridge)
    else:
        policies = {
            head: policies[head]
            for head in reversed(tuple(rendered_set.policies))
        }

    with pytest.raises(ValueError, match="exact seven-head order"):
        api.RenderedPolicySet(
            policies=policies,
            render_input_identity_sha256_by_head=(
                rendered_set.render_input_identity_sha256_by_head
            ),
            topology_predecessors=rendered_set.topology_predecessors,
            policy_set_sha256=rendered_set.policy_set_sha256,
        )


def test_red_rendered_policy_set_constructor_requires_rendered_policy_values() -> None:
    api = _api()
    bound, _ = _bound_policy_inputs()
    rendered_set = api.render_fence_policy_set(bound)
    policies = dict(rendered_set.policies)
    policies[api.PolicyHead.BRIDGE_SEED] = object()

    with pytest.raises(TypeError, match="must be a RenderedPolicy"):
        api.RenderedPolicySet(
            policies=policies,
            render_input_identity_sha256_by_head=(
                rendered_set.render_input_identity_sha256_by_head
            ),
            topology_predecessors=rendered_set.topology_predecessors,
            policy_set_sha256=rendered_set.policy_set_sha256,
        )


def test_red_rendered_policy_set_constructor_rejects_policy_head_misalignment() -> None:
    api = _api()
    bound, _ = _bound_policy_inputs()
    rendered_set = api.render_fence_policy_set(bound)
    policies = dict(rendered_set.policies)
    input_identities = dict(
        rendered_set.render_input_identity_sha256_by_head
    )
    bridge = api.PolicyHead.BRIDGE_SEED
    prepare = api.PolicyHead.PREPARE
    policies[bridge], policies[prepare] = policies[prepare], policies[bridge]

    with pytest.raises(ValueError, match="does not align with policy head"):
        api.RenderedPolicySet(
            policies=policies,
            render_input_identity_sha256_by_head=input_identities,
            topology_predecessors=rendered_set.topology_predecessors,
            policy_set_sha256=_policy_set_sha256(policies, input_identities),
        )


@pytest.mark.parametrize("mutation", ["missing", "foreign", "wrong-order"])
def test_red_rendered_policy_set_constructor_requires_exact_input_identity_heads(
    mutation,
) -> None:
    api = _api()
    bound, _ = _bound_policy_inputs()
    rendered_set = api.render_fence_policy_set(bound)
    identities = dict(rendered_set.render_input_identity_sha256_by_head)
    bridge = api.PolicyHead.BRIDGE_SEED

    if mutation == "missing":
        identities.pop(bridge)
    elif mutation == "foreign":
        identities["BRIDGE_SEED"] = identities.pop(bridge)
    else:
        identities = {
            head: identities[head]
            for head in reversed(tuple(rendered_set.policies))
        }

    with pytest.raises(ValueError, match="exact seven-head order"):
        api.RenderedPolicySet(
            policies=rendered_set.policies,
            render_input_identity_sha256_by_head=identities,
            topology_predecessors=rendered_set.topology_predecessors,
            policy_set_sha256=rendered_set.policy_set_sha256,
        )


def test_red_rendered_policy_set_constructor_requires_canonical_input_identities() -> None:
    api = _api()
    bound, _ = _bound_policy_inputs()
    rendered_set = api.render_fence_policy_set(bound)
    identities = dict(rendered_set.render_input_identity_sha256_by_head)
    identities[api.PolicyHead.BRIDGE_SEED] = "A" * 64

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        api.RenderedPolicySet(
            policies=rendered_set.policies,
            render_input_identity_sha256_by_head=identities,
            topology_predecessors=rendered_set.topology_predecessors,
            policy_set_sha256=rendered_set.policy_set_sha256,
        )


def test_red_rendered_policy_set_constructor_requires_exact_topology() -> None:
    api = _api()
    bound, _ = _bound_policy_inputs()
    rendered_set = api.render_fence_policy_set(bound)
    topology = dict(rendered_set.topology_predecessors)
    topology[api.PolicyHead.TERMINAL] = api.PolicyHead.CLOSED_SOURCE

    with pytest.raises(ValueError, match="exact topology predecessor map"):
        api.RenderedPolicySet(
            policies=rendered_set.policies,
            render_input_identity_sha256_by_head=(
                rendered_set.render_input_identity_sha256_by_head
            ),
            topology_predecessors=topology,
            policy_set_sha256=rendered_set.policy_set_sha256,
        )


@pytest.mark.parametrize("policy_set_sha256", ["A" * 64, "0" * 63, "0" * 65])
def test_red_rendered_policy_set_constructor_requires_canonical_sha_shape(
    policy_set_sha256,
) -> None:
    api = _api()
    bound, _ = _bound_policy_inputs()
    rendered_set = api.render_fence_policy_set(bound)

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(rendered_set, policy_set_sha256=policy_set_sha256)


def test_red_rendered_policy_set_constructor_reconciles_policy_set_sha256() -> None:
    api = _api()
    bound, _ = _bound_policy_inputs()
    rendered_set = api.render_fence_policy_set(bound)

    with pytest.raises(ValueError, match="does not reconcile"):
        replace(rendered_set, policy_set_sha256="0" * 64)


def test_red_rendered_policy_set_constructor_detaches_caller_mappings() -> None:
    api = _api()
    bound, _ = _bound_policy_inputs()
    canonical = api.render_fence_policy_set(bound)
    policies = dict(canonical.policies)
    identities = dict(canonical.render_input_identity_sha256_by_head)
    topology = dict(canonical.topology_predecessors)
    rendered_set = api.RenderedPolicySet(
        policies=policies,
        render_input_identity_sha256_by_head=identities,
        topology_predecessors=topology,
        policy_set_sha256=canonical.policy_set_sha256,
    )

    bridge = api.PolicyHead.BRIDGE_SEED
    prepare = api.PolicyHead.PREPARE
    policies[bridge] = policies[prepare]
    identities[bridge] = identities[prepare]
    topology[prepare] = None

    assert rendered_set.policies[bridge] is canonical.policies[bridge]
    assert (
        rendered_set.render_input_identity_sha256_by_head[bridge]
        == canonical.render_input_identity_sha256_by_head[bridge]
    )
    assert rendered_set.topology_predecessors[prepare] is bridge
    with pytest.raises(TypeError):
        rendered_set.policies[bridge] = policies[prepare]
    with pytest.raises(TypeError):
        rendered_set.render_input_identity_sha256_by_head[bridge] = identities[
            prepare
        ]
    with pytest.raises(TypeError):
        rendered_set.topology_predecessors[prepare] = None


def test_red_stage_evidence_and_bridge_writer_shape_are_exact() -> None:
    api = _api()
    with pytest.raises(TypeError, match="source_settlement_sha256"):
        api.render_fence_policy(
            replace(
                _slot_input(api.PolicyHead.BATCH),
                source_settlement_sha256=None,
            )
        )
    with pytest.raises(ValueError, match="forbids stage evidence"):
        api.render_fence_policy(
            replace(
                _slot_input(api.PolicyHead.PREPARE),
                freeze_denial_evidence_sha256="1" * 64,
            )
        )
    with pytest.raises(ValueError, match="exact two-member"):
        api.render_fence_policy(
            replace(_render_input(), writer_cohorts=(_cohort(),))
        )


def test_red_policy_ledger_summary_reconciles_all_three_headrooms() -> None:
    api = _api()
    rendered = api.render_fence_policy(_render_input())
    summary = rendered.ledger_summary
    assert summary.policy_prefix_bytes == len(b'{"Statement":[')
    assert summary.policy_suffix_bytes == len(b'],"Version":"2012-10-17"}')
    assert summary.statement_count == len(rendered.statement_ledger)
    assert sum(summary.component_statement_counts.values()) == len(
        rendered.statement_ledger
    )
    assert summary.working_limit_bytes == 17_920
    assert summary.working_headroom_bytes == 17_920 - len(
        rendered.policy_bytes
    )
    assert summary.design_headroom_bytes == 18_432 - len(
        rendered.policy_bytes
    )
    assert summary.s3_headroom_bytes == 20_480 - len(rendered.policy_bytes)
    with pytest.raises(TypeError):
        summary.component_statement_counts["global_guards"] = 0
    with pytest.raises(TypeError):
        rendered.component_bytes["global_guards"] = 0


def test_red_rendered_policy_constructor_rejects_wrong_ledger_entry_type() -> None:
    api = _api()
    rendered = api.render_fence_policy(_render_input())
    malformed_ledger = (object(),) + rendered.statement_ledger[1:]

    with pytest.raises(TypeError):
        replace(rendered, statement_ledger=malformed_ledger)


@pytest.mark.parametrize("field_name", ["component_bytes", "component_statement_counts"])
@pytest.mark.parametrize("mutation", ["missing", "foreign"])
def test_red_direct_constructors_require_exact_component_mapping(
    field_name, mutation
) -> None:
    api = _api()
    rendered = api.render_fence_policy(_render_input())
    target = (
        rendered
        if field_name == "component_bytes"
        else rendered.ledger_summary
    )
    components = dict(getattr(target, field_name))
    if mutation == "missing":
        components.pop("terminal_closure")
    else:
        components["unaccounted_component"] = 0

    with pytest.raises(ValueError):
        replace(target, **{field_name: components})


@pytest.mark.parametrize(
    ("bad_count", "error_type"),
    [
        (True, TypeError),
        (-1, ValueError),
    ],
)
def test_red_policy_ledger_summary_constructor_rejects_malformed_counts(
    bad_count, error_type
) -> None:
    api = _api()
    summary = api.render_fence_policy(_render_input()).ledger_summary
    counts = dict(summary.component_statement_counts)
    counts["global_guards"] = bad_count

    with pytest.raises(error_type):
        replace(summary, component_statement_counts=counts)


def test_red_policy_ledger_summary_constructor_rejects_count_total_drift() -> None:
    api = _api()
    summary = api.render_fence_policy(_render_input()).ledger_summary
    counts = dict(summary.component_statement_counts)
    counts["global_guards"] += 1

    with pytest.raises(ValueError):
        replace(summary, component_statement_counts=counts)


def test_red_rendered_policy_constructor_rejects_component_byte_drift() -> None:
    api = _api()
    rendered = api.render_fence_policy(_render_input())
    component_bytes = dict(rendered.component_bytes)
    component_bytes["global_guards"] -= 1
    component_bytes["legacy_fragment"] += 1
    assert sum(component_bytes.values()) == rendered.rendered_policy_bytes

    with pytest.raises(ValueError):
        replace(rendered, component_bytes=component_bytes)


def test_red_rendered_policy_constructor_rejects_component_count_drift() -> None:
    api = _api()
    rendered = api.render_fence_policy(_render_input())
    counts = dict(rendered.ledger_summary.component_statement_counts)
    counts["global_guards"] -= 1
    counts["legacy_fragment"] += 1
    assert sum(counts.values()) == len(rendered.statement_ledger)
    malformed_summary = replace(
        rendered.ledger_summary,
        component_statement_counts=counts,
    )

    with pytest.raises(ValueError):
        replace(rendered, ledger_summary=malformed_summary)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_entry",
        "sequence",
        "sid",
        "component",
        "leading_comma_bytes",
        "statement_bytes",
        "start_offset",
        "end_offset",
        "statement_sha256",
    ],
)
def test_red_rendered_policy_constructor_reconciles_each_statement_ledger_entry(
    mutation,
) -> None:
    api = _api()
    rendered = api.render_fence_policy(_render_input())
    ledger = list(rendered.statement_ledger)
    if mutation == "missing_entry":
        ledger.pop()
    else:
        entry = ledger[0]
        replacements = {
            "sequence": {"sequence": entry.sequence + 1},
            "sid": {"sid": entry.sid + "-drift"},
            "component": {"component": "legacy_fragment"},
            "leading_comma_bytes": {"leading_comma_bytes": 1},
            "statement_bytes": {"statement_bytes": entry.statement_bytes + 1},
            "start_offset": {"start_offset": entry.start_offset + 1},
            "end_offset": {"end_offset": entry.end_offset - 1},
            "statement_sha256": {"statement_sha256": "0" * 64},
        }
        ledger[0] = replace(entry, **replacements[mutation])

    with pytest.raises(ValueError):
        replace(rendered, statement_ledger=tuple(ledger))


@pytest.mark.parametrize(
    "arn",
    [
        "arn:aws:iam::246813579024:role/",
        "arn:aws:iam::246813579024:role/invalid*role",
        "arn:aws:iam::111122223333:role/keep-glm52-writer-1",
        "arn:aws:iam::246813579024:user/not-a-role",
        "arn:aws:iam::246813579024:role/path//role",
    ],
)
def test_red_principal_arn_parser_rejects_nonexact_roles(arn) -> None:
    api = _api()
    invalid = api.PrincipalIdentity(
        binding_id="invalid-role",
        arn=arn,
        role_id="AROATESTINVALID01",
    )
    cohort = replace(
        _cohort(member_count=2),
        members=(invalid, _identity(2)),
    )
    with pytest.raises(ValueError, match="exact fixed-account IAM role ARN"):
        api.render_fence_policy(
            replace(_render_input(), writer_cohorts=(cohort,))
        )


def test_red_role_id_and_binding_id_are_bijective() -> None:
    api = _api()
    first = _identity(1)
    role_id_reuse = api.PrincipalIdentity(
        binding_id="different-binding",
        arn="arn:aws:iam::246813579024:role/different-role",
        role_id=first.role_id,
    )
    cohort = replace(
        _cohort(member_count=2),
        members=(first, role_id_reuse),
    )
    with pytest.raises(ValueError, match="names another principal ARN"):
        api.render_fence_policy(
            replace(_render_input(), writer_cohorts=(cohort,))
        )


@pytest.mark.parametrize(
    "kms_key_arn",
    [
        "arn:aws:kms:us-west-2:246813579024:key/",
        "arn:aws:kms:us-west-2:246813579024:key/not-a-uuid",
        "arn:aws:kms:us-east-1:246813579024:key/"
        "11111111-2222-3333-4444-555555555555",
        "arn:aws:kms:us-west-2:111122223333:key/"
        "11111111-2222-3333-4444-555555555555",
        "arn:aws:kms:us-west-2:246813579024:alias/not-a-key",
    ],
)
def test_red_kms_key_arn_parser_rejects_nonexact_keys(kms_key_arn) -> None:
    api = _api()
    with pytest.raises(ValueError, match="exact us-west-2"):
        api.render_fence_policy(
            replace(_render_input(), kms_key_arn=kms_key_arn)
        )


def test_red_legacy_snapshot_rejects_stateful_mappings_and_empty_conditions() -> None:
    api = _api()

    class StatefulStatement(dict):
        pass

    base = {
        "Sid": "LegacyStableDeny",
        "Effect": "Deny",
        "Principal": "*",
        "Action": ["s3:PutObject"],
        "Resource": [
            "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2/legacy/*"
        ],
    }
    with pytest.raises(TypeError, match="plain object"):
        api.render_fence_policy(
            _render_input(
                build_mode=api.BuildMode.LEGACY_ENABLED,
                legacy_statements=(StatefulStatement(base),),
            )
        )
    with pytest.raises(ValueError, match="non-empty object"):
        api.render_fence_policy(
            _render_input(
                build_mode=api.BuildMode.LEGACY_ENABLED,
                legacy_statements=({**base, "Condition": {}},),
            )
        )

    mutable_action = ["s3:PutObject"]
    mutable_statement = {**base, "Action": mutable_action}
    rendered = api.render_fence_policy(
        _render_input(
            build_mode=api.BuildMode.LEGACY_ENABLED,
            legacy_statements=(mutable_statement,),
        )
    )
    policy_sha256 = rendered.policy_sha256
    mutable_action.append("s3:DeleteObject")
    mutable_statement["Sid"] = "ChangedAfterRender"
    assert rendered.policy_sha256 == policy_sha256
    assert hashlib.sha256(rendered.policy_bytes).hexdigest() == policy_sha256
    assert "ChangedAfterRender" not in rendered.policy_bytes.decode("ascii")


def test_red_legacy_snapshot_rejects_nested_stateful_mappings() -> None:
    api = _api()

    class StatefulCondition(dict):
        pass

    statement = {
        "Sid": "LegacyNestedStatefulMapping",
        "Effect": "Deny",
        "Principal": "*",
        "Action": ["s3:PutObject"],
        "Resource": [
            "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2/legacy/*"
        ],
        "Condition": StatefulCondition(
            {"StringEquals": {"s3:ExistingObjectTag/fence": "true"}}
        ),
    }
    for root in (statement, MappingProxyType(statement)):
        with pytest.raises(TypeError, match="non-exact JSON value"):
            _render_input(
                build_mode=api.BuildMode.LEGACY_ENABLED,
                legacy_statements=(root,),
            )


def test_red_legacy_mapping_proxy_is_deep_snapshotted_before_identity() -> None:
    api = _api()
    mutable_action = ["s3:PutObject"]
    backing_statement = {
        "Sid": "LegacyProxySnapshot",
        "Effect": "Deny",
        "Principal": "*",
        "Action": mutable_action,
        "Resource": [
            "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2/legacy/*"
        ],
    }
    render_input = _render_input(
        build_mode=api.BuildMode.LEGACY_ENABLED,
        legacy_statements=(MappingProxyType(backing_statement),),
    )
    original_identity = render_input.render_input_identity_sha256

    mutable_action.append("s3:DeleteObject")
    backing_statement["Sid"] = "MutatedProxyBacking"
    rendered = api.render_fence_policy(render_input)

    assert render_input.render_input_identity_sha256 == original_identity
    assert "MutatedProxyBacking" not in rendered.policy_bytes.decode("ascii")
    assert "s3:DeleteObject" not in rendered.policy["Statement"][3]["Action"]


def test_red_source_reader_actions_cover_current_annotation_reads() -> None:
    api = _api()
    assert {
        "s3:GetObjectAnnotation",
        "s3:GetObjectVersionAnnotation",
        "s3:GetObjectVersionAnnotationForReplication",
        "s3:ListObjectAnnotations",
        "s3:ListObjectVersionAnnotations",
    }.issubset(api.SOURCE_OBJECT_READ)


def test_red_global_principal_guards_exempt_aws_service_principals() -> None:
    api = _api()
    policy = api.render_fence_policy(_render_input()).policy
    bucket = "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2"
    for context in (
        {
            "aws:SecureTransport": "true",
            "aws:PrincipalAccount": "111122223333",
            "aws:PrincipalIsAWSService": "true",
        },
        {
            "aws:SecureTransport": "true",
            "aws:PrincipalIsAWSService": "true",
        },
    ):
        assert not _matching_deny_sids(
            policy,
            action="s3:GetBucketLocation",
            resource=bucket,
            context=context,
        )


def test_red_checked_in_seven_policy_golden_set_reconciles() -> None:
    api = _api()
    artifact_path = (
        Path(__file__).parents[1]
        / ".superpowers/sdd/goal-objective/"
        "task-3ph1g-h1g-fence-policy-renderer-golden-set-v1.json"
    )
    raw = artifact_path.read_bytes()
    assert b"\n" not in raw
    artifact = json.loads(raw)
    identity = artifact.pop("canonical_identity_sha256")
    assert identity == (
        "d9d0e173bc4dd71cb10584841a43aa9f95a44c8a0b789320540575708d0d0250"
    )
    assert hashlib.sha256(canonical_json_bytes(artifact)).hexdigest() == identity

    bound, _ = _bound_policy_inputs()
    rendered_set = api.render_fence_policy_set(bound)
    assert artifact["policy_set_sha256"] == rendered_set.policy_set_sha256
    assert [row["policy_head"] for row in artifact["heads"]] == [
        value.policy_head.value for value in bound
    ]
    for value, row in zip(bound, artifact["heads"]):
        rendered = rendered_set.policies[value.policy_head]
        summary = rendered.ledger_summary
        assert row["render_input_identity_sha256"] == (
            value.render_input_identity_sha256
        )
        assert row["policy_sha256"] == rendered.policy_sha256
        assert row["rendered_policy_bytes"] == rendered.rendered_policy_bytes
        assert row["policy"] == rendered.policy
        assert row["component_bytes"] == dict(rendered.component_bytes)
        assert row["ledger_summary"] == {
            "policy_prefix_bytes": summary.policy_prefix_bytes,
            "policy_suffix_bytes": summary.policy_suffix_bytes,
            "statement_count": summary.statement_count,
            "max_policy_bytes": summary.max_policy_bytes,
            "working_limit_bytes": summary.working_limit_bytes,
            "reserved_design_headroom_bytes": (
                summary.reserved_design_headroom_bytes
            ),
            "working_headroom_bytes": summary.working_headroom_bytes,
            "design_headroom_bytes": summary.design_headroom_bytes,
            "s3_headroom_bytes": summary.s3_headroom_bytes,
            "component_statement_counts": dict(
                summary.component_statement_counts
            ),
        }
        assert row["statement_ledger"] == [
            {
                "sequence": item.sequence,
                "sid": item.sid,
                "component": item.component,
                "leading_comma_bytes": item.leading_comma_bytes,
                "statement_bytes": item.statement_bytes,
                "start_offset": item.start_offset,
                "end_offset": item.end_offset,
                "statement_sha256": item.statement_sha256,
            }
            for item in rendered.statement_ledger
        ]
    largest = max(
        artifact["heads"], key=lambda row: row["rendered_policy_bytes"]
    )
    assert artifact["max_policy_head"] == largest["policy_head"]
    assert artifact["max_rendered_policy_bytes"] == largest[
        "rendered_policy_bytes"
    ]


def test_red_public_types_freeze_constructor_inputs_and_projection_identity() -> None:
    api = _api()
    with pytest.raises(TypeError, match="immutable tuple"):
        api.WriterCohort(
            cohort_id="invalid",
            members=[_identity()],
            guard_resources=(_cohort().guard_resources[0],),
            cross_member_denial_evidence_sha256="1" * 64,
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        api.ObjectReaderScope(
            scope_id="invalid",
            readers=(_identity(),),
            resources=[
                "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2/a"
            ],
        )

    action = ["s3:PutObject"]
    statement = {
        "Sid": "LegacyConstructorSnapshot",
        "Effect": "Deny",
        "Principal": "*",
        "Action": action,
        "Resource": [
            "arn:aws:s3:::keep-glm52-models-246813579024-us-west-2/legacy/*"
        ],
    }
    render_input = _render_input(
        build_mode=api.BuildMode.LEGACY_ENABLED,
        legacy_statements=(statement,),
    )
    original_identity = render_input.render_input_identity_sha256
    action.append("s3:DeleteObject")
    statement["Sid"] = "MutatedBeforeRender"
    rendered = api.render_fence_policy(render_input)
    assert "MutatedBeforeRender" not in rendered.policy_bytes.decode("ascii")
    assert render_input.render_input_identity_sha256 == original_identity
    changed = replace(
        render_input,
        terminal_prerequisite_sha256=None,
    )
    assert changed.render_input_identity_sha256 == original_identity
    changed_resource = replace(
        render_input,
        writer_cohorts=(
            replace(
                render_input.writer_cohorts[0],
                cross_member_denial_evidence_sha256="2" * 64,
            ),
        ),
    )
    assert (
        changed_resource.render_input_identity_sha256
        != render_input.render_input_identity_sha256
    )


def test_red_whole_policy_limit_fails_independently_of_component_caps() -> None:
    api = _api()
    limits = api.PolicyLimits()
    builder = api._StatementBuilder()
    for index, component in enumerate(api._COMPONENT_ORDER):
        payload = "x" * (limits.component_max_bytes[component] - 300)
        builder.add(
            component=component,
            sid=f"IndependentWholeLimit{index}",
            actions=("s3:PutObject",),
            resources=("arn:aws:s3:::synthetic/" + payload,),
        )

    component_bytes = {
        component: 0 for component in api._COMPONENT_ORDER
    }
    component_bytes["global_guards"] = len(b'{"Statement":[') + len(
        b'],"Version":"2012-10-17"}'
    )
    for sequence, pending in enumerate(builder.pending):
        component_bytes[pending.component] += (
            (0 if sequence == 0 else 1)
            + len(api._policy_json_bytes(pending.value))
        )
    assert sum(component_bytes.values()) > 17_920
    assert all(
        component_bytes[component] <= limits.component_max_bytes[component]
        for component in api._COMPONENT_ORDER
    )
    with pytest.raises(
        api.PolicyComponentBudgetExhausted,
        match="working limit 17920",
    ):
        api._assemble(builder, limits)


def test_red_policy_set_rejects_unapproved_semantic_deny_removal() -> None:
    """Break caught: a legal-looking edge silently drops a deny family."""
    api = _api()
    bound, _ = _bound_policy_inputs()
    replacement_reader = api.PrincipalIdentity(
        binding_id="replacement-list-reader",
        arn=(
            "arn:aws:iam::246813579024:"
            "role/keep-glm52-replacement-list-reader"
        ),
        role_id="AROATESTREPLACEMENTLISTREADER",
    )
    changed_reservation = replace(
        bound[2],
        source_inventory_readers=(replacement_reader,),
    )
    changed_reservation_hash = api.render_fence_policy(
        changed_reservation
    ).policy_sha256
    changed_frozen = replace(
        bound[3],
        predecessor_policy_sha256=changed_reservation_hash,
    )

    with pytest.raises(
        ValueError,
        match="POLICY_SEMANTIC_ROLLBACK.*RESERVATION.*unrelated reader",
    ):
        api.render_fence_policy_set(
            bound[:2]
            + (changed_reservation, changed_frozen)
            + bound[4:]
        )


@pytest.mark.parametrize("component", ("writer", "object_reader"))
def test_red_batch_rejects_unrelated_allowlisted_identity_replacement(
    component,
) -> None:
    """Break caught: allowlisted component replacement injects a new role."""
    api = _api()
    bound, _ = _bound_policy_inputs()
    unrelated = api.PrincipalIdentity(
        binding_id="unrelated-" + component,
        arn=(
            "arn:aws:iam::246813579024:role/keep-glm52-unrelated-"
            + component
        ),
        role_id="AROATESTUNRELATED" + component.replace("_", "").upper(),
    )
    if component == "writer":
        changed_batch = replace(
            bound[4],
            writer_cohorts=(
                replace(
                    bound[4].writer_cohorts[0],
                    members=(unrelated,),
                ),
            ),
        )
    else:
        changed_batch = replace(
            bound[4],
            source_validation_readers=(unrelated,),
        )
    changed_terminal = replace(
        bound[6],
        predecessor_policy_sha256=api.render_fence_policy(
            changed_batch
        ).policy_sha256,
    )

    with pytest.raises(
        ValueError,
        match="POLICY_SEMANTIC_ROLLBACK.*unrelated.*identity",
    ):
        api.render_fence_policy_set(
            bound[:4]
            + (changed_batch, bound[5], changed_terminal)
        )


def test_red_policy_set_rejects_unapproved_terminal_resource_replacement() -> None:
    """Break caught: TERMINAL swaps one protected writer-owned resource."""
    api = _api()
    bound, _ = _bound_policy_inputs()
    changed_terminal = replace(
        bound[6],
        writer_owned_resources=bound[6].writer_owned_resources
        + (
            api.FIXED_BUCKET_ARN
            + "/campaigns/glm52-sky-20260724/closed/replaced.json",
        ),
    )

    with pytest.raises(
        ValueError,
        match="POLICY_SEMANTIC_ROLLBACK: TERMINAL writer-owned terminal resource projection drifted",
    ):
        api.render_fence_policy_set(bound[:6] + (changed_terminal,))


def _production_template(policy):
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "H1gProductionFenceBucketPolicy": {
                "DeletionPolicy": "Retain",
                "Properties": {
                    "Bucket": "keep-glm52-models-246813579024-us-west-2",
                    "PolicyDocument": policy,
                },
                "Type": "AWS::S3::BucketPolicy",
                "UpdateReplacePolicy": "Retain",
            }
        },
    }


def _production_policy_changes():
    return [
        {
            "Type": "Resource",
            "ResourceChange": {
                "Action": "Modify",
                "LogicalResourceId": "H1gProductionFenceBucketPolicy",
                "PhysicalResourceId": (
                    "keep-glm52-models-246813579024-us-west-2"
                ),
                "ResourceType": "AWS::S3::BucketPolicy",
                "Replacement": "False",
                "Scope": ["Properties"],
                "Details": [],
            },
        }
    ]


def test_red_production_policy_check_recomputes_exact_bytes_and_plan_shape() -> None:
    api = _api()
    rendered = api.render_fence_policy(_slot_input(api.PolicyHead.BRIDGE_SEED))
    checked = api.check_production_policy_plan(
        rendered=rendered,
        template_body=_production_template(rendered.policy),
        changes=_production_policy_changes(),
    )

    assert checked.policy_sha256 == rendered.policy_sha256
    assert checked.rendered_policy_bytes == len(rendered.policy_bytes)
    assert checked.s3_headroom_bytes == 20_480 - len(rendered.policy_bytes)
    assert checked.logical_resource_id == "H1gProductionFenceBucketPolicy"
    assert checked.replacement == "False"


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda template, changes: template.update({"Outputs": {}}),
            "template shape is not exact",
        ),
        (
            lambda template, changes: template["Resources"][
                "H1gProductionFenceBucketPolicy"
            ]["Properties"]["PolicyDocument"].update({"Version": "2008-10-17"}),
            "PolicyDocument bytes do not match rendered policy",
        ),
        (
            lambda template, changes: changes.append(changes[0]),
            "one nonreplacement policy modify",
        ),
        (
            lambda template, changes: changes[0]["ResourceChange"].update(
                {"Replacement": "True"}
            ),
            "one nonreplacement policy modify",
        ),
        (
            lambda template, changes: changes[0]["ResourceChange"].update(
                {"Details": [{"Target": {"Name": "Bucket"}}]}
            ),
            "one nonreplacement policy modify",
        ),
    ],
)
def test_red_production_policy_check_rejects_template_and_plan_mutants(
    mutator,
    message,
) -> None:
    api = _api()
    rendered = api.render_fence_policy(_slot_input(api.PolicyHead.BRIDGE_SEED))
    template = _production_template(rendered.policy)
    changes = _production_policy_changes()
    mutator(template, changes)

    with pytest.raises(ValueError, match=message):
        api.check_production_policy_plan(
            rendered=rendered,
            template_body=template,
            changes=changes,
        )


def test_red_production_shaped_budget_fixture_pins_compacted_boundary() -> None:
    api = _api()
    bound, _ = _bound_policy_inputs()
    rendered_set = api.render_fence_policy_set(bound)
    batch = rendered_set.policies[api.PolicyHead.BATCH]
    seed = rendered_set.policies[api.PolicyHead.BRIDGE_SEED]

    assert seed.rendered_policy_bytes == 5_395
    assert seed.component_bytes["source_family_closures"] == 1_477
    assert batch.rendered_policy_bytes == 17_128
    assert batch.component_bytes["permanent_lineage"] == 5_307
    assert batch.component_bytes["source_family_closures"] == 1_471
    assert batch.component_bytes["object_reader_guards"] == 5_706
    assert batch.ledger_summary.working_headroom_bytes == 792
    assert batch.ledger_summary.s3_headroom_bytes == 3_352

    second_nonselected = (
        "campaigns/glm52-sky-20260724/spend-snapshots/"
        + "2" * 64
        + "/GPU_SPEND_SNAPSHOT.json"
    )
    third_nonselected = (
        "campaigns/glm52-sky-20260724/spend-snapshots/"
        + "3" * 64
        + "/GPU_SPEND_SNAPSHOT.json"
    )
    two_nonselected = replace(
        bound[4],
        nonselected_source_keys=(
            bound[4].nonselected_source_keys + (second_nonselected,)
        ),
    )
    assert api.render_fence_policy(two_nonselected).rendered_policy_bytes == 17_508
    with pytest.raises(
        api.PolicyComponentBudgetExhausted,
        match="object_reader_guards",
    ):
        api.render_fence_policy(
            replace(
                two_nonselected,
                nonselected_source_keys=(
                    two_nonselected.nonselected_source_keys
                    + (third_nonselected,)
                ),
            )
        )
