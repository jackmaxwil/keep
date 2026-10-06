from __future__ import annotations

import base64
import hashlib
import io
from datetime import UTC, datetime

import pytest

from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.cloudformation_stacks import (
    MigrationRoleEvidenceV2,
    StackIdentity,
    StackKind,
    build_bridge_seed_ownership_plan_v2,
    build_stack_migration_transfer_checkpoint_v2,
)
from glm52_enforcement.fence_artifacts import (
    BOOTSTRAP_PUBLICATION_ORDER,
    FenceSlot,
    parse_bridge_seed_artifact,
    build_fence_template_bytes,
    parse_fence_manifest,
)
from glm52_enforcement.fence_policy_renderer import (
    BuildMode,
    CORE_WRITER_RESOURCES,
    FencePolicyInput,
    PolicyHead,
    PrincipalIdentity,
    ReservedFamily,
    RUN_GUARD_RESOURCE,
    SOURCE_FAMILY_SPECS,
    WriterCohort,
    render_fence_policy,
)
from glm52_enforcement.task13_fixed_artifacts import (
    ACCOUNT_ID,
    CAMPAIGN_BUCKET,
    REGION,
    RUN_ID,
    FixedKeyPublicationDisposition,
    Task13FixedArtifactServices,
    publish_fixed_key_bytes_with_outcome,
)
from glm52_enforcement.task13_production_operations import ProductionServices


def SHA(marker: str) -> str:
    return marker * 64


KMS_KEY_ARN = (
    "arn:aws:kms:us-west-2:246813579024:key/"
    "01234567-89ab-cdef-0123-456789abcdef"
)
ACTIVATION_ID = "glm52-v2-amber-quartz"


def _metadata(marker: str) -> dict[str, object]:
    return {
        "HTTPStatusCode": 200,
        "RequestId": marker,
        "RetryAttempts": 0,
    }


class _Sts:
    def get_caller_identity(self) -> dict[str, object]:
        return {
            "Account": ACCOUNT_ID,
            "Arn": "arn:aws:iam::246813579024:role/ExactMigrationApiCaller",
            "UserId": "AROAAPICALLER12345678:publication",
            "ResponseMetadata": _metadata("sts"),
        }


class _S3:
    def __init__(self, *, ambiguous: bool = False) -> None:
        self.ambiguous = ambiguous
        self.rows: dict[str, list[dict[str, object]]] = {}
        self.put_keys: list[str] = []
        self.failure: str | None = None
        self._version = 0

    def get_bucket_versioning(self, **request: object) -> dict[str, object]:
        assert request == {
            "Bucket": CAMPAIGN_BUCKET,
            "ExpectedBucketOwner": ACCOUNT_ID,
        }
        return {"Status": "Enabled", "ResponseMetadata": _metadata("versioning")}

    def list_object_versions(self, **request: object) -> dict[str, object]:
        key = request["Prefix"]
        versions = [
            {
                "Key": key,
                "VersionId": row["version_id"],
                "Size": len(row["raw"]),
            }
            for row in self.rows.get(key, [])
        ]
        delete_markers: list[dict[str, object]] = []
        if self.failure == "sibling":
            versions.append(
                {"Key": str(key) + ".sibling", "VersionId": "sibling", "Size": 1}
            )
        if self.failure == "delete":
            delete_markers.append({"Key": key, "VersionId": "deleted"})
        return {
            "Versions": versions,
            "DeleteMarkers": delete_markers,
            "IsTruncated": False,
            "ResponseMetadata": _metadata("list"),
        }

    def put_object(self, **request: object) -> dict[str, object]:
        key = str(request["Key"])
        self.put_keys.append(key)
        self._version += 1
        version_id = f"version-{self._version}"
        self.rows.setdefault(key, []).append(
            {
                "version_id": version_id,
                "raw": request["Body"],
                "content_type": request["ContentType"],
                "checksum": request["ChecksumSHA256"],
                "metadata": request["Metadata"],
                "kms": request["SSEKMSKeyId"],
            }
        )
        if self.ambiguous:
            raise TimeoutError("response lost after durable write")
        return {
            "VersionId": version_id,
            "ChecksumSHA256": request["ChecksumSHA256"],
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": request["SSEKMSKeyId"],
            "ResponseMetadata": _metadata("put"),
        }

    def get_object(self, **request: object) -> dict[str, object]:
        rows = self.rows[str(request["Key"])]
        row = next(item for item in rows if item["version_id"] == request["VersionId"])
        checksum = row["checksum"]
        metadata = row["metadata"]
        kms = row["kms"]
        if self.failure == "checksum":
            checksum = base64.b64encode(b"wrong-checksum").decode("ascii")
        elif self.failure == "metadata":
            metadata = {"foreign": "metadata"}
        elif self.failure == "kms":
            kms = KMS_KEY_ARN[:-1] + "0"
        return {
            "Body": io.BytesIO(row["raw"]),
            "ContentLength": len(row["raw"]),
            "ContentType": row["content_type"],
            "VersionId": row["version_id"],
            "ChecksumSHA256": checksum,
            "Metadata": metadata,
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": kms,
            "ObjectLockMode": None,
            "ObjectLockRetainUntilDate": None,
            "ObjectLockLegalHoldStatus": None,
            "ResponseMetadata": _metadata("get"),
        }

    def get_object_tagging(self, **request: object) -> dict[str, object]:
        return {"TagSet": [], "ResponseMetadata": _metadata("tags")}

class _ReadS3:
    def __init__(self, backing: _S3) -> None:
        self._backing = backing

    def __getattr__(self, name: str) -> object:
        if name == "put_object":
            raise AssertionError("read client attempted PutObject")
        return getattr(self._backing, name)


class _PublisherS3:
    def __init__(self, backing: _S3) -> None:
        self._backing = backing

    def put_object(self, **request: object) -> dict[str, object]:
        return self._backing.put_object(**request)


def _fixed_services(s3: _S3) -> Task13FixedArtifactServices:
    return Task13FixedArtifactServices(sts=_Sts(), s3=s3, total_max_attempts=1)


def _services(s3: _S3) -> ProductionServices:
    unused = object()
    return ProductionServices(
        sts=_Sts(),
        cloudformation=unused,
        iam=unused,
        s3=_ReadS3(s3),
        organizations=unused,
        ec2=unused,
        ssm=unused,
        kms=unused,
        dynamodb=unused,
        lambda_client=unused,
        states=unused,
        cloudtrail=unused,
        total_max_attempts=1,
        fixed_artifact_publisher_s3=_PublisherS3(s3),
        credential_expiration=datetime(2026, 8, 1, tzinfo=UTC),
    )


def _principal(name: str, suffix: str) -> PrincipalIdentity:
    return PrincipalIdentity(
        binding_id=name,
        arn=f"arn:aws:iam::{ACCOUNT_ID}:role/{name}",
        role_id="AROA" + suffix.ljust(16, "0")[:16],
    )


def _cohort(name: str, members: tuple[PrincipalIdentity, ...]) -> WriterCohort:
    return WriterCohort(
        cohort_id=name,
        members=members,
        guard_resources=(RUN_GUARD_RESOURCE,),
        cross_member_denial_evidence_sha256=SHA("7"),
    )


def _renderer_inputs() -> tuple[FencePolicyInput, ...]:
    seed_writers = (_principal("artifact-writer-one", "ARTIFACTONE"), _principal("artifact-writer-two", "ARTIFACTTWO"))
    families = tuple(
        ReservedFamily(family_id=family_id, resources=(resource,))
        for family_id, resource in SOURCE_FAMILY_SPECS
    )
    seed = FencePolicyInput(
        build_mode=BuildMode.LEGACY_DISABLED,
        policy_head=PolicyHead.BRIDGE_SEED,
        bucket_name=CAMPAIGN_BUCKET,
        account_id=ACCOUNT_ID,
        kms_key_arn=KMS_KEY_ARN,
        writer_cohorts=(_cohort("bridge-seed", seed_writers),),
        reserved_families=families,
    )
    rendered_seed = render_fence_policy(seed)
    reader_validation = _principal("source-validation-reader", "VALIDATION")
    reader_inventory = _principal("source-inventory-reader", "INVENTORY")
    reader_terminal = _principal("terminal-audit-reader", "TERMINALAUDIT")
    retired = _principal("retired-publisher", "RETIRED")
    permanent = (
        f"arn:aws:s3:::{CAMPAIGN_BUCKET}/campaigns/{RUN_ID}/authorities/fence/FENCE_GENESIS.json",
    )
    retired_resources = (
        f"arn:aws:s3:::{CAMPAIGN_BUCKET}/campaigns/{RUN_ID}/retired/genesis.json",
    )
    common = {
        "build_mode": BuildMode.LEGACY_DISABLED,
        "bucket_name": CAMPAIGN_BUCKET,
        "account_id": ACCOUNT_ID,
        "kms_key_arn": KMS_KEY_ARN,
        "permanent_enrolled_resources": permanent,
        "retired_publishers": (retired,),
        "retired_publisher_resources": retired_resources,
        "writer_owned_resources": CORE_WRITER_RESOURCES,
        "source_validation_readers": (reader_validation,),
        "source_inventory_readers": (reader_inventory,),
        "terminal_audit_readers": (reader_terminal,),
    }
    publisher_hash = SHA("8")
    prepare = FencePolicyInput(
        **common,
        policy_head=PolicyHead.PREPARE,
        predecessor_policy_sha256=rendered_seed.policy_sha256,
        publisher_deny_policy_sha256=publisher_hash,
        reserved_families=families,
        writer_cohorts=(_cohort("prepare", (_principal("prepare-writer", "PREPARE"),)),),
    )
    rendered_prepare = render_fence_policy(prepare)
    reservation = FencePolicyInput(
        **common,
        policy_head=PolicyHead.RESERVATION,
        predecessor_policy_sha256=rendered_prepare.policy_sha256,
        publisher_deny_policy_sha256=publisher_hash,
        writer_cohorts=(_cohort("reservation", (_principal("reservation-writer", "RESERVATION"),)),),
    )
    rendered_reservation = render_fence_policy(reservation)
    frozen = FencePolicyInput(
        **common,
        policy_head=PolicyHead.SOURCE_FAMILIES_FROZEN,
        predecessor_policy_sha256=rendered_reservation.policy_sha256,
        freeze_denial_evidence_sha256=SHA("9"),
        publisher_deny_policy_sha256=publisher_hash,
        reserved_families=families,
        writer_cohorts=(_cohort("frozen", (_principal("frozen-writer", "FROZEN"),)),),
    )
    rendered_frozen = render_fence_policy(frozen)
    closed = FencePolicyInput(
        **common,
        policy_head=PolicyHead.CLOSED_SOURCE,
        predecessor_policy_sha256=rendered_frozen.policy_sha256,
        freeze_denial_evidence_sha256=SHA("9"),
        publisher_deny_policy_sha256=publisher_hash,
        terminal_prerequisite_sha256=SHA("a"),
        reserved_families=families,
        writer_cohorts=(),
    )
    return seed, prepare, reservation, frozen, closed


def _policy_input_projection(value: FencePolicyInput) -> dict[str, object]:
    def principal(item: PrincipalIdentity) -> dict[str, str]:
        return {
            "binding_id": item.binding_id,
            "arn": item.arn,
            "role_id": item.role_id,
        }

    return {
        "build_mode": value.build_mode.value,
        "policy_head": value.policy_head.value,
        "bucket_name": value.bucket_name,
        "account_id": value.account_id,
        "kms_key_arn": value.kms_key_arn,
        "predecessor_policy_sha256": value.predecessor_policy_sha256,
        "freeze_denial_evidence_sha256": value.freeze_denial_evidence_sha256,
        "source_publication_sealed_sha256": value.source_publication_sealed_sha256,
        "all_version_inventory_sha256": value.all_version_inventory_sha256,
        "source_settlement_sha256": value.source_settlement_sha256,
        "publisher_deny_policy_sha256": value.publisher_deny_policy_sha256,
        "terminal_prerequisite_sha256": value.terminal_prerequisite_sha256,
        "legacy_statements": [dict(item) for item in value.legacy_statements],
        "permanent_enrolled_resources": list(value.permanent_enrolled_resources),
        "retired_publishers": [principal(item) for item in value.retired_publishers],
        "retired_publisher_resources": list(value.retired_publisher_resources),
        "reserved_families": [
            {"family_id": item.family_id, "resources": list(item.resources)}
            for item in value.reserved_families
        ],
        "writer_cohorts": [
            {
                "cohort_id": item.cohort_id,
                "members": [principal(member) for member in item.members],
                "guard_resources": list(item.guard_resources),
                "cross_member_denial_evidence_sha256": item.cross_member_denial_evidence_sha256,
            }
            for item in value.writer_cohorts
        ],
        "writer_owned_resources": list(value.writer_owned_resources),
        "source_validation_readers": [principal(item) for item in value.source_validation_readers],
        "source_inventory_readers": [principal(item) for item in value.source_inventory_readers],
        "terminal_audit_readers": [principal(item) for item in value.terminal_audit_readers],
        "selected_source_keys": list(value.selected_source_keys),
        "nonselected_source_keys": list(value.nonselected_source_keys),
        "policy_limits": {
            "max_policy_bytes": value.policy_limits.max_policy_bytes,
            "unallocated_headroom_bytes": value.policy_limits.unallocated_headroom_bytes,
            "component_max_bytes": dict(value.policy_limits.component_max_bytes),
        },
    }


def _tags() -> tuple[tuple[str, str], ...]:
    return (
        ("Authority", "H1g"),
        ("Campaign", "GLM-5.2"),
        ("Environment", "production"),
        ("ManagedBy", "CloudFormation"),
        ("Project", "KEEP"),
        ("RunId", RUN_ID),
    )


def _stack(kind: StackKind, suffix: str) -> StackIdentity:
    names = {
        StackKind.RETAINED: "keep-glm52-gpu",
        StackKind.FENCE: "keep-glm52-h1g-fence",
        StackKind.SUPPORT: "keep-glm52-h1g-support",
    }
    name = names[kind]
    return StackIdentity(
        kind=kind,
        name=name,
        stack_id=f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:stack/{name}/{suffix}",
        account_id=ACCOUNT_ID,
        region=REGION,
        termination_protection=True,
        tags=_tags(),
    )


def _role(name: str, role_id: str, marker: str) -> MigrationRoleEvidenceV2:
    return MigrationRoleEvidenceV2(
        role_arn=f"arn:aws:iam::{ACCOUNT_ID}:role/{name}",
        role_id=role_id,
        trust_policy_sha256=SHA(marker),
        permission_policy_sha256=SHA(marker),
    )


def _checkpoint(inputs: tuple[FencePolicyInput, ...]):
    seed_rendered = render_fence_policy(inputs[0])
    seed_raw = seed_rendered.policy_bytes
    seed_digest = hashlib.sha256(seed_raw).hexdigest()
    seed = parse_bridge_seed_artifact(
        {
            "artifact_kind": "BRIDGE_SEED_POLICY",
            "record_type": "glm52_h1g_bridge_seed_policy_v1",
            "bucket": CAMPAIGN_BUCKET,
            "expected_live_preseed_policy_sha256": SHA("b"),
            "key": BOOTSTRAP_PUBLICATION_ORDER[0],
            "version_id": "migration-seed-version",
            "file_sha256": seed_digest,
            "body_sha256": seed_digest,
            "policy_sha256": seed_digest,
            "rendered_policy_bytes": len(seed_raw),
            "statement_ledger": [
                {
                    "sequence": row.sequence,
                    "sid": row.sid,
                    "component": row.component,
                    "leading_comma_bytes": row.leading_comma_bytes,
                    "statement_bytes": row.statement_bytes,
                    "start_offset": row.start_offset,
                    "end_offset": row.end_offset,
                    "statement_sha256": row.statement_sha256,
                }
                for row in seed_rendered.statement_ledger
            ],
            "publisher_deny_policy_sha256": None,
        },
        raw_bytes=seed_raw,
    )
    retained_template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "ModelBucket": {"Type": "AWS::S3::Bucket", "Properties": {"BucketName": CAMPAIGN_BUCKET}},
            "RetainedRole": {"Type": "AWS::IAM::Role", "Properties": {}},
        },
    }
    plan = build_bridge_seed_ownership_plan_v2(
        retained_template=retained_template,
        bucket_name=CAMPAIGN_BUCKET,
        bridge_seed=seed,
        current_policy_logical_id=None,
    )
    api = _role("ExactMigrationApiCaller", "AROAAPICALLER12345678", "1")
    migration = _role("keep-glm52-h1g-cloudformation-deployment", "AROAMIGRATIONSVC123456", "2")
    fence = _role("keep-glm52-h1g-fence-service", "AROAFENCESERVICE1234567", "3")
    transfer_hash = SHA("4")
    return build_stack_migration_transfer_checkpoint_v2(
        bridge_seed_plan=plan,
        retained=_stack(StackKind.RETAINED, "11111111-1111-4111-8111-111111111111"),
        fence=_stack(StackKind.FENCE, "22222222-2222-4222-8222-222222222222"),
        support=_stack(StackKind.SUPPORT, "33333333-3333-4333-8333-333333333333"),
        bucket_name=CAMPAIGN_BUCKET,
        fence_import_template_sha256=SHA("5"),
        fence_transfer_template_sha256=transfer_hash,
        active_original_template_sha256=transfer_hash,
        active_processed_template_sha256=transfer_hash,
        direct_policy_sha256=(plan.seed_policy_sha256, plan.seed_policy_sha256),
        api_caller=api,
        migration_service_role=migration,
        fence_service_role=fence,
        associated_stack_role_arn=migration.role_arn,
        associated_stack_role_id=migration.role_id,
        import_change_set_id=(
            f"arn:aws:cloudformation:{REGION}:{ACCOUNT_ID}:changeSet/"
            "h1g-import-production-fence/cccccccc-1111-4222-8333-444444444444"
        ),
        support_prestate_template_sha256=SHA("6"),
    )


def _request(inputs: tuple[FencePolicyInput, ...], checkpoint) -> dict[str, object]:
    return {
        "schema_version": 2,
        "record_type": "glm52_fence_bootstrap_publication_request_v2",
        "activation_id": ACTIVATION_ID,
        "generation": 1,
        "materializer_function_version_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-fence-bootstrap-materializer:1"
        ),
        "checkpoint_identity_sha256": checkpoint.canonical_identity_sha256,
        "renderer_inputs": [_policy_input_projection(item) for item in inputs],
        "manifest_authority": {
            "mutation_authority_inventory": [
                {"authority_class": "RETAINED_PRE_SUPPORT"},
                {"authority_class": "SUPPORT_RUNTIME"},
            ],
            "member_account_authority": {
                "account": ACCOUNT_ID,
                "arn": checkpoint.api_caller.role_arn,
                "region": REGION,
                "observed_at": "2026-07-31T00:00:00Z",
            },
            "bucket_control_plane": {"versioning": "Enabled"},
            "kms_key_identity": {"key_arn": KMS_KEY_ARN, "enabled": True, "multi_region": False},
            "executor_inventory": [
                {"authority_class": "RETAINED_PRE_SUPPORT"},
                {"authority_class": "SUPPORT_RUNTIME"},
            ],
            "principal_inventory": [{"binding_id": "ExactMigrationApiCaller"}],
            "writer_inventory": [{"binding_id": "artifact-writer-one"}],
            "writer_policy_cohorts": [{"slot": "PREPARE_GENESIS_LIVE_STATE"}],
            "source_inventory": [{"source": index, "version_id": f"genesis-{index}"} for index in range(7)],
        },
    }


def _preseed(s3: _S3, inputs: tuple[FencePolicyInput, ...]) -> None:
    raw = render_fence_policy(inputs[0]).policy_bytes
    checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    s3.rows[BOOTSTRAP_PUBLICATION_ORDER[0]] = [{
        "version_id": "migration-seed-version",
        "raw": raw,
        "content_type": "application/json",
        "checksum": checksum,
        "metadata": {},
        "kms": KMS_KEY_ARN,
    }]


def test_raw_schema_is_exact_and_forbids_derived_artifacts() -> None:
    from glm52_enforcement.fence_bootstrap_publication import (
        BOOTSTRAP_PUBLICATION_REQUEST_FIELDS,
        FENCE_POLICY_INPUT_FIELDS,
        MANIFEST_AUTHORITY_FIELDS,
        parse_bootstrap_publication_request_v2,
    )

    inputs = _renderer_inputs()
    checkpoint = _checkpoint(inputs)
    request = _request(inputs, checkpoint)
    parsed = parse_bootstrap_publication_request_v2(request, checkpoint=checkpoint)
    assert set(request) == BOOTSTRAP_PUBLICATION_REQUEST_FIELDS
    assert set(request["renderer_inputs"][0]) == FENCE_POLICY_INPUT_FIELDS
    assert set(request["manifest_authority"]) == MANIFEST_AUTHORITY_FIELDS
    assert parsed.activation_id == ACTIVATION_ID
    for field in ("rendered_bytes", "entries", "manifest", "canonical_identity_sha256"):
        mutant = dict(request)
        mutant[field] = b"caller-authored" if field == "rendered_bytes" else {}
        with pytest.raises(ValueError, match="schema|field"):
            parse_bootstrap_publication_request_v2(mutant, checkpoint=checkpoint)
    missing = dict(request)
    missing.pop("renderer_inputs")
    with pytest.raises(ValueError, match="schema|field"):
        parse_bootstrap_publication_request_v2(missing, checkpoint=checkpoint)
    for mutate in ("unknown", "missing"):
        nested = _request(inputs, checkpoint)
        renderer = dict(nested["renderer_inputs"][0])
        if mutate == "unknown":
            renderer["policy_bytes"] = "{}"
        else:
            renderer.pop("policy_head")
        nested["renderer_inputs"][0] = renderer
        with pytest.raises(ValueError, match="schema|field"):
            parse_bootstrap_publication_request_v2(nested, checkpoint=checkpoint)


def test_publication_derives_renderer_bytes_and_manifest_last() -> None:
    from glm52_enforcement.fence_bootstrap_publication import publish_bootstrap_fence_artifacts_v2

    inputs = _renderer_inputs()
    checkpoint = _checkpoint(inputs)
    s3 = _S3()
    _preseed(s3, inputs)
    result = publish_bootstrap_fence_artifacts_v2(
        request=_request(inputs, checkpoint), checkpoint=checkpoint, services=_services(s3)
    )
    assert tuple(item.key for item in result.coordinates) == BOOTSTRAP_PUBLICATION_ORDER
    assert s3.put_keys == list(BOOTSTRAP_PUBLICATION_ORDER[1:])
    assert s3.put_keys[-1] == BOOTSTRAP_PUBLICATION_ORDER[-1]
    expected_by_slot = {
        FenceSlot.PREPARE_GENESIS_LIVE_STATE: inputs[1],
        FenceSlot.RESERVATION_ONLY: inputs[2],
        FenceSlot.SOURCE_FAMILIES_FROZEN: inputs[3],
        FenceSlot.CLOSED_SOURCE: inputs[4],
    }
    for coordinate in result.coordinates[1:-1]:
        slot = next(slot for slot, key in zip(
            (FenceSlot.PREPARE_GENESIS_LIVE_STATE, FenceSlot.RESERVATION_ONLY, FenceSlot.SOURCE_FAMILIES_FROZEN, FenceSlot.CLOSED_SOURCE),
            BOOTSTRAP_PUBLICATION_ORDER[1:-1],
        ) if key == coordinate.key)
        assert s3.rows[coordinate.key][0]["raw"] == build_fence_template_bytes(render_fence_policy(expected_by_slot[slot]))
    manifest_raw = s3.rows[result.manifest_coordinate.key][0]["raw"]
    manifest = parse_fence_manifest(__import__("json").loads(manifest_raw[:-1]))
    assert manifest.canonical_identity_sha256 == result.manifest_coordinate.canonical_identity_sha256
    assert result.prepare_entry_identity_sha256 == manifest.entry(FenceSlot.PREPARE_GENESIS_LIVE_STATE).entry_identity_sha256
    assert result.checkpoint_identity_sha256 == checkpoint.canonical_identity_sha256
    assert result.seed_adopted is True


@pytest.mark.parametrize("ambiguous", [False, True])
def test_nominal_and_ambiguous_publication_converge(ambiguous: bool) -> None:
    from glm52_enforcement.fence_bootstrap_publication import (
        publish_bootstrap_fence_artifacts_v2,
        read_bootstrap_fence_publication_v2,
    )

    inputs = _renderer_inputs()
    checkpoint = _checkpoint(inputs)
    request = _request(inputs, checkpoint)
    s3 = _S3(ambiguous=ambiguous)
    _preseed(s3, inputs)
    written = publish_bootstrap_fence_artifacts_v2(
        request=request, checkpoint=checkpoint, services=_services(s3)
    )
    adopted = read_bootstrap_fence_publication_v2(
        request=request, checkpoint=checkpoint, services=_services(s3)
    )
    assert adopted.to_dict() == written.to_dict()
    assert adopted.seed_adopted is True


def test_fixed_key_outcome_distinguishes_written_from_adopted() -> None:
    s3 = _S3()
    raw = canonical_json_bytes({"record_type": "outcome-proof"})
    key = BOOTSTRAP_PUBLICATION_ORDER[0]
    first = publish_fixed_key_bytes_with_outcome(
        services=_fixed_services(s3), bucket=CAMPAIGN_BUCKET, key=key, raw=raw,
        record_type="outcome-proof", sse_kms_key_id=KMS_KEY_ARN,
    )
    second = publish_fixed_key_bytes_with_outcome(
        services=_fixed_services(s3), bucket=CAMPAIGN_BUCKET, key=key, raw=raw,
        record_type="outcome-proof", sse_kms_key_id=KMS_KEY_ARN,
    )
    assert first.disposition is FixedKeyPublicationDisposition.WRITTEN
    assert second.disposition is FixedKeyPublicationDisposition.ADOPTED
    assert first.version_id == second.version_id

def test_fixed_key_publication_separates_read_and_write_clients() -> None:
    backing = _S3()

    class ReadOnlyS3:
        get_bucket_versioning = backing.get_bucket_versioning
        get_object = backing.get_object
        get_object_tagging = backing.get_object_tagging
        list_object_versions = backing.list_object_versions

        def put_object(self, **_request: object) -> dict[str, object]:
            raise AssertionError("read client attempted PutObject")

    class WriteOnlyS3:
        put_object = backing.put_object

        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"write client attempted {name}")

    raw = canonical_json_bytes({"record_type": "split-client-proof"})
    key = BOOTSTRAP_PUBLICATION_ORDER[0]
    services = Task13FixedArtifactServices(
        sts=_Sts(),
        s3=ReadOnlyS3(),
        publisher_s3=WriteOnlyS3(),
        total_max_attempts=1,
    )

    first = publish_fixed_key_bytes_with_outcome(
        services=services,
        bucket=CAMPAIGN_BUCKET,
        key=key,
        raw=raw,
        record_type="split-client-proof",
        sse_kms_key_id=KMS_KEY_ARN,
    )
    second = publish_fixed_key_bytes_with_outcome(
        services=services,
        bucket=CAMPAIGN_BUCKET,
        key=key,
        raw=raw,
        record_type="split-client-proof",
        sse_kms_key_id=KMS_KEY_ARN,
    )

    assert first.disposition is FixedKeyPublicationDisposition.WRITTEN
    assert second.disposition is FixedKeyPublicationDisposition.ADOPTED
    assert backing.put_keys == [key]


def test_bridge_seed_publication_is_separate_and_assumed_role_only() -> None:
    from glm52_enforcement.fence_bootstrap_publication import (
        publish_bridge_seed_policy_v2,
        read_bridge_seed_publication_v2,
    )

    backing = _S3()

    class ReadOnlyS3:
        get_bucket_versioning = backing.get_bucket_versioning
        get_object = backing.get_object
        get_object_tagging = backing.get_object_tagging
        list_object_versions = backing.list_object_versions

    class PublisherS3:
        put_object = backing.put_object

    seed_input = _renderer_inputs()[0]
    request = {
        "schema_version": 2,
        "record_type": "glm52_h1g_bridge_seed_publication_request_v2",
        "activation_id": ACTIVATION_ID,
        "generation": 1,
        "materializer_function_version_arn": (
            "arn:aws:lambda:us-west-2:246813579024:function:"
            "keep-glm52-h1g-fence-bootstrap-materializer:1"
        ),
        "expected_live_preseed_policy_sha256": SHA("b"),
        "renderer_input": _policy_input_projection(seed_input),
    }
    services = Task13FixedArtifactServices(
        sts=_Sts(),
        s3=ReadOnlyS3(),
        publisher_s3=PublisherS3(),
        total_max_attempts=1,
    )

    first = publish_bridge_seed_policy_v2(
        request=request,
        services=services,
    )
    second = publish_bridge_seed_policy_v2(
        request=request,
        services=services,
    )
    adopted = read_bridge_seed_publication_v2(
        request=request,
        services=Task13FixedArtifactServices(
            sts=_Sts(),
            s3=ReadOnlyS3(),
            total_max_attempts=1,
        ),
    )

    assert first.artifact.to_dict() == second.artifact.to_dict()
    assert first.artifact.to_dict()["key"] == BOOTSTRAP_PUBLICATION_ORDER[0]
    assert first.artifact.policy_sha256 == render_fence_policy(
        seed_input
    ).policy_sha256
    assert first.disposition is FixedKeyPublicationDisposition.WRITTEN
    assert second.disposition is FixedKeyPublicationDisposition.ADOPTED
    assert adopted.artifact.to_dict() == first.artifact.to_dict()
    assert adopted.disposition is FixedKeyPublicationDisposition.ADOPTED
    assert backing.put_keys == [BOOTSTRAP_PUBLICATION_ORDER[0]]

    with pytest.raises(ValueError, match="publisher"):
        publish_bridge_seed_policy_v2(
            request=request,
            services=Task13FixedArtifactServices(
                sts=_Sts(),
                s3=backing,
                publisher_s3=backing,
                total_max_attempts=1,
            ),
        )

def test_checkpoint_identity_drift_is_rejected_before_s3() -> None:
    from glm52_enforcement.fence_bootstrap_publication import publish_bootstrap_fence_artifacts_v2

    inputs = _renderer_inputs()
    checkpoint = _checkpoint(inputs)
    request = _request(inputs, checkpoint)
    request["checkpoint_identity_sha256"] = SHA("f")
    s3 = _S3()
    with pytest.raises(ValueError, match="checkpoint"):
        publish_bootstrap_fence_artifacts_v2(
            request=request, checkpoint=checkpoint, services=_services(s3)
        )
    assert s3.put_keys == []


@pytest.mark.parametrize("failure", ["sibling", "delete", "history", "kms", "metadata", "checksum"])
def test_seed_adoption_rejects_foreign_history_and_readback_drift(failure: str) -> None:
    from glm52_enforcement.fence_bootstrap_publication import publish_bootstrap_fence_artifacts_v2

    inputs = _renderer_inputs()
    checkpoint = _checkpoint(inputs)
    s3 = _S3()
    _preseed(s3, inputs)
    if failure == "history":
        s3.rows[BOOTSTRAP_PUBLICATION_ORDER[0]].append(dict(s3.rows[BOOTSTRAP_PUBLICATION_ORDER[0]][0], version_id="foreign-history"))
    else:
        s3.failure = failure
    with pytest.raises(ValueError, match="foreign|delete|singular|drift"):
        publish_bootstrap_fence_artifacts_v2(
            request=_request(inputs, checkpoint), checkpoint=checkpoint, services=_services(s3)
        )
    assert s3.put_keys == []


def test_missing_bridge_seed_is_never_created_or_mislabeled_adopted() -> None:
    from glm52_enforcement.fence_bootstrap_publication import publish_bootstrap_fence_artifacts_v2

    inputs = _renderer_inputs()
    checkpoint = _checkpoint(inputs)
    s3 = _S3()
    with pytest.raises(ValueError, match="seed|adopt|durable"):
        publish_bootstrap_fence_artifacts_v2(
            request=_request(inputs, checkpoint), checkpoint=checkpoint, services=_services(s3)
        )
    assert s3.put_keys == []
