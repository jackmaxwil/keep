from __future__ import annotations

import base64
import hashlib
import io
import json

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
BUCKET = "keep-glm52-models-246813579024-us-west-2"
STACK_ID = (
    "arn:aws:cloudformation:us-west-2:246813579024:stack/"
    "keep-glm52-h1g-fence/01234567-89ab-cdef-0123-456789abcdef"
)
MIGRATION_ROLE = (
    "arn:aws:iam::246813579024:role/"
    "keep-glm52-h1g-cloudformation-deployment"
)
FENCE_ROLE = "arn:aws:iam::246813579024:role/keep-glm52-h1g-fence-service"


def SHA(character: str) -> str:
    return character * 64


EXPECTED_KEYS = (
    "campaigns/glm52-sky-20260724/authorities/fence/seeds/"
    "glm52-v2-amber-quartz/00000001/BRIDGE_SEED_POLICY.json",
    "campaigns/glm52-sky-20260724/authorities/fence/templates/"
    "glm52-v2-amber-quartz/00000001/PREPARE_GENESIS_LIVE_STATE.json",
    "campaigns/glm52-sky-20260724/authorities/fence/templates/"
    "glm52-v2-amber-quartz/00000001/BATCH_FIVE_SOURCE_ACTIVATION.json",
    "campaigns/glm52-sky-20260724/authorities/fence/templates/"
    "glm52-v2-amber-quartz/00000001/RESERVATION_ONLY.json",
    "campaigns/glm52-sky-20260724/authorities/fence/templates/"
    "glm52-v2-amber-quartz/00000001/SOURCE_FAMILIES_FROZEN.json",
    "campaigns/glm52-sky-20260724/authorities/fence/templates/"
    "glm52-v2-amber-quartz/00000001/CLOSED_SOURCE.json",
    "campaigns/glm52-sky-20260724/authorities/fence/templates/"
    "glm52-v2-amber-quartz/00000001/TERMINAL.json",
    "campaigns/glm52-sky-20260724/authorities/fence/manifests/"
    "glm52-v2-amber-quartz/00000001/FENCE_BOOTSTRAP_MANIFEST.json",
    "campaigns/glm52-sky-20260724/authorities/fence/manifests/"
    "glm52-v2-amber-quartz/00000001/FENCE_SOURCE_SETTLED_MANIFEST.json",
)

MANIFEST_FIELDS = {
    "schema_version",
    "record_type",
    "manifest_stage",
    "account_id",
    "region",
    "run_id",
    "mutation_authority_inventory",
    "mutation_authority_inventory_sha256",
    "activation_id",
    "member_account_authority",
    "generation",
    "legacy_mode",
    "bucket_name",
    "bucket_arn",
    "bucket_control_plane",
    "kms_key_identity",
    "stack_name",
    "stack_id",
    "executor_inventory",
    "executor_inventory_sha256",
    "support_runtime_identity_coordinate",
    "logical_id",
    "migration_service_role",
    "fence_service_role",
    "policy_limits",
    "render_input_identity_sha256",
    "legacy_fragment_sha256",
    "bridge_seed",
    "transition_graph",
    "principal_inventory",
    "writer_inventory",
    "writer_inventory_sha256",
    "writer_policy_cohorts",
    "writer_policy_cohorts_sha256",
    "source_inventory",
    "source_inventory_sha256",
    "bootstrap_manifest_coordinate",
    "source_settlement",
    "entries",
    "canonical_identity_sha256",
}

ENTRY_FIELDS = {
    "slot",
    "transition_class",
    "allowed_predecessor_heads",
    "terminal_head",
    "template_key",
    "template_url",
    "version_id",
    "template_sha256",
    "template_body_sha256",
    "policy_sha256",
    "expected_prestate_policy_sha256",
    "expected_prestate_stack_role_arn",
    "expected_poststate_stack_role_arn",
    "publisher_deny_policy_sha256",
    "rendered_policy_bytes",
    "entry_identity_sha256",
    "statement_ledger",
    "change_set_name",
    "request_skeleton",
    "request_skeleton_sha256",
    "batch_projection_contract",
    "successor_contract_sha256",
    "allowed_create_authority_classes",
    "allowed_execute_authority_classes",
}

REQUEST_FIELDS = {
    "manifest_coordinate",
    "manifest_stage",
    "selected_entry_identity_sha256",
    "slot",
    "allowed_create_authority_classes",
    "allowed_execute_authority_classes",
    "support_runtime_identity_coordinate",
    "predecessor_head",
    "observed_h1f_active_head_coordinate",
    "h1f_successor_coordinate",
    "batch_projection",
    "source_settled_manifest_coordinate",
    "request_skeleton_sha256",
    "canonical_identity_sha256",
}


def _rendered(marker: str):
    from glm52_enforcement.fence_policy_renderer import (
        PolicyHead,
        render_fence_policy,
    )
    from test_glm52_h1g_fence_policy_renderer import _slot_input

    head = {
        "BRIDGE_SEED": PolicyHead.BRIDGE_SEED,
        "PREPARE": PolicyHead.PREPARE,
        "PREPARE_GENESIS_LIVE_STATE": PolicyHead.PREPARE,
        "RESERVATION_ONLY": PolicyHead.RESERVATION,
        "SOURCE_FAMILIES_FROZEN": PolicyHead.SOURCE_FAMILIES_FROZEN,
        "BATCH_FIVE_SOURCE_ACTIVATION": PolicyHead.BATCH,
        "CLOSED_SOURCE": PolicyHead.CLOSED_SOURCE,
        "TERMINAL": PolicyHead.TERMINAL,
    }[marker]
    return render_fence_policy(_slot_input(head))


def _coordinate(key: str, identity: str = SHA("a")):
    from glm52_enforcement.fence_artifacts import ArtifactCoordinate

    return ArtifactCoordinate(
        bucket=BUCKET,
        key=key,
        version_id="opaque/+?= version",
        file_sha256=SHA("b"),
        canonical_identity_sha256=identity,
    )


def _runtime_coordinate():
    from glm52_enforcement.fence_artifacts import SupportRuntimeIdentityCoordinate

    return SupportRuntimeIdentityCoordinate(
        bucket=BUCKET,
        key=(
            "campaigns/glm52-sky-20260724/authorities/fence/runtime/"
            "glm52-v2-amber-quartz/00000001/SUPPORT_RUNTIME_IDENTITY.json"
        ),
        version_id="runtime-version/+?",
        file_sha256=SHA("c"),
        canonical_identity_sha256=SHA("d"),
    )


def _source_seal():
    from glm52_enforcement.fence_artifacts import build_source_settlement_seal
    inventory = tuple(
        {"family": index, "version_id": f"source-{index}"}
        for index in range(5)
    )
    inventory_sha256 = canonical_sha256(inventory)
    first_observation_body = {
        "observed_at": "2026-07-31T00:00:09Z",
        "inventory_sha256": inventory_sha256,
        "page_request_ids": ("first-inventory-request",),
    }
    second_observation_body = {
        "observed_at": "2026-07-31T00:00:10Z",
        "inventory_sha256": inventory_sha256,
        "page_request_ids": ("second-inventory-request",),
    }

    return build_source_settlement_seal(
        {
            "generation": 1,
            "reservation_coordinate": {"identity": SHA("1")},
            "freeze_entry_coordinate": {"identity": SHA("2")},
            "freeze_execution_evidence": {"identity": SHA("3")},
            "freeze_denial_probe_rows": ({"round": 1}, {"round": 2}),
            "source_action_terminal_rows": tuple(
                {"source": index, "status": "SUCCEEDED"} for index in range(5)
            ),
            "source_lambda_execution_terminal_rows": tuple(
                {"source": index, "request_id": f"request-{index}"}
                for index in range(5)
            ),
            "workflow_post_source_state_identity": SHA("4"),
            "publisher_reachability_proof": {"closed": True},
            "selected_source_rows": tuple(
                {"family": index, "version_id": f"source-{index}"}
                for index in range(5)
            ),
            "nonselected_provisional_rows": (),
            "complete_family_version_inventory": inventory,
            "first_inventory_observation": {
                **first_observation_body,
                "canonical_identity_sha256": canonical_sha256(
                    first_observation_body
                ),
            },
            "second_inventory_observation": {
                **second_observation_body,
                "canonical_identity_sha256": canonical_sha256(
                    second_observation_body
                ),
            },
            "inventory_sha256": inventory_sha256,
            "sealed_at": "2026-07-31T00:00:11Z",
        }
    )


def _manifest_common_fields():
    return {
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "run_id": RUN_ID,
        "mutation_authority_inventory": (
            {"authority_class": "RETAINED_PRE_SUPPORT"},
            {"authority_class": "SUPPORT_RUNTIME"},
        ),
        "activation_id": "glm52-v2-amber-quartz",
        "member_account_authority": {
            "account": ACCOUNT_ID,
            "arn": "arn:aws:iam::246813579024:role/operator",
            "user_id": "AROATEST:operator",
            "region": REGION,
            "organization_id": "o-example1234",
            "management_account_id": ACCOUNT_ID,
            "feature_set": "ALL",
            "observed_at": "2026-07-31T00:00:00Z",
            "cloudtrail_cursor": "cursor-1",
            "owner_decision_file_sha256": SHA("6"),
        },
        "generation": 1,
        "legacy_mode": "ENABLED",
        "bucket_name": BUCKET,
        "bucket_arn": "arn:aws:s3:::" + BUCKET,
        "bucket_control_plane": {"versioning": "Enabled"},
        "kms_key_identity": {
            "key_arn": (
                "arn:aws:kms:us-west-2:246813579024:key/"
                "01234567-89ab-cdef-0123-456789abcdef"
            ),
            "enabled": True,
            "multi_region": False,
        },
        "stack_name": "keep-glm52-h1g-fence",
        "stack_id": STACK_ID,
        "executor_inventory": (
            {"authority_class": "RETAINED_PRE_SUPPORT"},
            {"authority_class": "SUPPORT_RUNTIME"},
        ),
        "logical_id": "H1gProductionFenceBucketPolicy",
        "migration_service_role": {
            "arn": MIGRATION_ROLE,
            "role_id": "AROAMIGRATION123456789",
            "canonical_identity_sha256": SHA("7"),
        },
        "fence_service_role": {
            "arn": FENCE_ROLE,
            "role_id": "AROAFENCESERVICE12345",
            "canonical_identity_sha256": SHA("8"),
        },
        "policy_limits": {
            "working_limit_bytes": 17_920,
            "design_limit_bytes": 18_432,
            "s3_limit_bytes": 20_480,
        },
        "render_input_identity_sha256": SHA("9"),
        "legacy_fragment_sha256": SHA("a"),
        "principal_inventory": ({"binding_id": "operator"},),
        "writer_inventory": ({"binding_id": "artifact-publisher"},),
        "writer_policy_cohorts": ({"slot": "PREPARE_GENESIS_LIVE_STATE"},),
        "source_inventory": tuple(
            {"source": index, "version_id": f"genesis-{index}"}
            for index in range(7)
        ),
    }


def _build_entries(
    stage,
    bootstrap_coordinate=None,
    bridge_seed_hash=SHA("0"),
    version_ids=None,
):
    from glm52_enforcement.fence_artifacts import (
        FenceSlot,
        ManifestStage,
        build_fence_entry,
    )

    policy = {
        slot: _rendered(slot.value)
        for slot in FenceSlot
    }
    order = (
        (
            FenceSlot.PREPARE_GENESIS_LIVE_STATE,
            FenceSlot.RESERVATION_ONLY,
            FenceSlot.CLOSED_SOURCE,
            FenceSlot.SOURCE_FAMILIES_FROZEN,
        )
        if stage is ManifestStage.BOOTSTRAP
        else (
            FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION,
            FenceSlot.TERMINAL,
        )
    )
    predecessor_hashes = {
        FenceSlot.PREPARE_GENESIS_LIVE_STATE: bridge_seed_hash,
        FenceSlot.RESERVATION_ONLY: policy[
            FenceSlot.PREPARE_GENESIS_LIVE_STATE
        ].policy_sha256,
        FenceSlot.SOURCE_FAMILIES_FROZEN: policy[
            FenceSlot.RESERVATION_ONLY
        ].policy_sha256,
        FenceSlot.CLOSED_SOURCE: policy[
            FenceSlot.SOURCE_FAMILIES_FROZEN
        ].policy_sha256,
        FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION: policy[
            FenceSlot.SOURCE_FAMILIES_FROZEN
        ].policy_sha256,
        FenceSlot.TERMINAL: policy[
            FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION
        ].policy_sha256,
    }
    return tuple(
        build_fence_entry(
            slot=slot,
            rendered=policy[slot],
            version_id=(
                version_ids[slot]
                if version_ids is not None
                else "version /+?=" + slot.value
            ),
            render_input_identity_sha256=SHA("9"),
            stack_id=STACK_ID,
            migration_service_role_arn=MIGRATION_ROLE,
            fence_service_role_arn=FENCE_ROLE,
            bridge_seed_policy_sha256=bridge_seed_hash,
            expected_prestate_policy_sha256=predecessor_hashes[slot],
            publisher_deny_policy_sha256=(
                SHA("f")
                if slot
                in (
                    FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION,
                    FenceSlot.TERMINAL,
                )
                else SHA("e")
            ),
            batch_projection_contract=(
                {
                    "generation": 1,
                    "selected_source_rows": tuple(
                        {"family": index, "version_id": f"source-{index}"}
                        for index in range(5)
                    ),
                    "nonselected_inventory_sha256": SHA("c"),
                    "publisher_identities": tuple(
                        {
                            "arn": (
                                "arn:aws:iam::246813579024:role/"
                                f"publisher-{index}"
                            ),
                            "role_id": f"AROAPUBLISHER{index:05d}",
                        }
                        for index in range(5)
                    ),
                }
                if slot is FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION
                else None
            ),
            successor_contract_sha256=(
                None
                if slot is FenceSlot.PREPARE_GENESIS_LIVE_STATE
                else SHA("d")
            ),
            bootstrap_manifest_coordinate=(
                bootstrap_coordinate
                if slot is FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION
                else None
            ),
        )
        for slot in order
    )


def _build_bootstrap_manifest():
    from glm52_enforcement.fence_artifacts import (
        ManifestStage,
        build_bridge_seed_artifact,
        build_fence_manifest,
    )

    seed = build_bridge_seed_artifact(
        rendered=_rendered("BRIDGE_SEED"),
        expected_live_preseed_policy_sha256=SHA("f"),
        version_id="seed-version/+?",
    )
    return build_fence_manifest(
        stage=ManifestStage.BOOTSTRAP,
        fields=_manifest_common_fields(),
        bridge_seed=seed,
        entries=_build_entries(
            ManifestStage.BOOTSTRAP, bridge_seed_hash=seed.policy_sha256
        ),
    )


def _build_source_manifest(bootstrap):
    from glm52_enforcement.fence_artifacts import (
        ManifestStage,
        build_fence_manifest,
    )

    bootstrap_coordinate = _coordinate(
        EXPECTED_KEYS[7], bootstrap.canonical_identity_sha256
    )
    return build_fence_manifest(
        stage=ManifestStage.SOURCE_SETTLED,
        fields=_manifest_common_fields(),
        bridge_seed=bootstrap.bridge_seed,
        entries=_build_entries(
            ManifestStage.SOURCE_SETTLED,
            bootstrap_coordinate,
            bootstrap.bridge_seed.policy_sha256,
        ),
        bootstrap_manifest_coordinate=bootstrap_coordinate,
        support_runtime_identity_coordinate=_runtime_coordinate(),
        source_settlement=_source_seal(),
    )


def test_exact_nine_keys_templates_and_versioned_url_encoding() -> None:
    from glm52_enforcement.fence_artifacts import (
        FENCE_ARTIFACT_KEYS,
        FenceSlot,
        build_fence_entry,
        build_fence_template_bytes,
    )

    assert FENCE_ARTIFACT_KEYS == EXPECTED_KEYS
    entry = build_fence_entry(
        slot=FenceSlot.PREPARE_GENESIS_LIVE_STATE,
        rendered=_rendered("PREPARE"),
        version_id="opaque /+?=%☃",
        render_input_identity_sha256=SHA("9"),
        stack_id=STACK_ID,
        migration_service_role_arn=MIGRATION_ROLE,
        fence_service_role_arn=FENCE_ROLE,
        bridge_seed_policy_sha256=SHA("0"),
        expected_prestate_policy_sha256=SHA("0"),
        publisher_deny_policy_sha256=SHA("e"),
        batch_projection_contract=None,
        successor_contract_sha256=None,
    )
    value = entry.to_dict()
    assert set(value) == ENTRY_FIELDS
    assert value["template_url"] == (
        "https://keep-glm52-models-246813579024-us-west-2."
        "s3.us-west-2.amazonaws.com/"
        + EXPECTED_KEYS[1]
        + "?versionId=opaque%20%2F%2B%3F%3D%25%E2%98%83"
    )
    template = entry.template_bytes
    assert template == build_fence_template_bytes(_rendered("PREPARE"))
    assert template.endswith(b"\n") and not template.endswith(b"\n\n")
    body = json.loads(template)
    assert set(body) == {"AWSTemplateFormatVersion", "Resources"}
    resource = body["Resources"]["H1gProductionFenceBucketPolicy"]
    assert set(resource) == {
        "DeletionPolicy",
        "Properties",
        "Type",
        "UpdateReplacePolicy",
    }
    assert resource["Properties"]["Bucket"] == BUCKET


def test_manifest_stage_discriminants_order_self_hashes_and_v1_rejection() -> None:
    from glm52_enforcement.fence_artifacts import (
        BOOTSTRAP_ENTRY_ORDER,
        SOURCE_SETTLED_ENTRY_ORDER,
        parse_fence_manifest,
    )

    bootstrap = _build_bootstrap_manifest()
    source = _build_source_manifest(bootstrap)
    bootstrap_value = bootstrap.to_dict()
    source_value = source.to_dict()

    assert set(bootstrap_value) == MANIFEST_FIELDS
    assert [row["slot"] for row in bootstrap_value["entries"]] == list(
        BOOTSTRAP_ENTRY_ORDER
    )
    assert [row["slot"] for row in source_value["entries"]] == list(
        SOURCE_SETTLED_ENTRY_ORDER
    )
    for value in (bootstrap_value, source_value):
        unsigned = dict(value)
        identity = unsigned.pop("canonical_identity_sha256")
        assert identity == canonical_sha256(unsigned)
        assert parse_fence_manifest(value).to_dict() == value

    v1 = dict(bootstrap_value)
    v1["record_type"] = "glm52_fence_bootstrap_manifest_v1"
    with pytest.raises(ValueError, match="v2|record_type"):
        parse_fence_manifest(v1)
    reordered = dict(bootstrap_value)
    reordered["entries"] = list(reversed(reordered["entries"]))
    with pytest.raises(ValueError, match="order|entries"):
        parse_fence_manifest(reordered)


def test_manifest_rejects_stage_nullability_graph_arrays_and_derived_hashes() -> None:
    from glm52_enforcement.fence_artifacts import parse_fence_manifest

    bootstrap = _build_bootstrap_manifest().to_dict()
    source = _build_source_manifest(_build_bootstrap_manifest()).to_dict()
    for field in (
        "bootstrap_manifest_coordinate",
        "support_runtime_identity_coordinate",
        "source_settlement",
    ):
        malformed = json.loads(json.dumps(bootstrap))
        malformed[field] = source[field]
        with pytest.raises(ValueError, match=field + "|BOOTSTRAP"):
            parse_fence_manifest(malformed)

    for field in (
        "bootstrap_manifest_coordinate",
        "support_runtime_identity_coordinate",
        "source_settlement",
    ):
        malformed = json.loads(json.dumps(source))
        malformed[field] = None
        with pytest.raises(ValueError, match=field + "|SOURCE_SETTLED"):
            parse_fence_manifest(malformed)

    mutations = []
    wrong_predecessor = json.loads(json.dumps(bootstrap))
    wrong_predecessor["entries"][1]["allowed_predecessor_heads"][0][
        "policy_sha256"
    ] = SHA("1")
    mutations.append(wrong_predecessor)
    duplicate_authority = json.loads(json.dumps(bootstrap))
    duplicate_authority["entries"][0]["allowed_create_authority_classes"] = [
        "RETAINED_PRE_SUPPORT",
        "RETAINED_PRE_SUPPORT",
    ]
    mutations.append(duplicate_authority)
    foreign_slot = json.loads(json.dumps(bootstrap))
    foreign_slot["entries"][0]["slot"] = "FOREIGN"
    mutations.append(foreign_slot)
    entry_hash = json.loads(json.dumps(bootstrap))
    entry_hash["entries"][0]["entry_identity_sha256"] = SHA("2")
    mutations.append(entry_hash)
    request_hash = json.loads(json.dumps(bootstrap))
    request_hash["entries"][0]["request_skeleton_sha256"] = SHA("3")
    mutations.append(request_hash)
    inventory_hash = json.loads(json.dumps(bootstrap))
    inventory_hash["writer_inventory_sha256"] = SHA("4")
    mutations.append(inventory_hash)
    for malformed in mutations:
        with pytest.raises(ValueError):
            parse_fence_manifest(malformed)

def test_manifest_rejects_stack_id_size_and_source_seal_mutants() -> None:
    from glm52_enforcement.fence_artifacts import parse_fence_manifest

    def rehash_entry_and_manifest(value, index=0):
        entry = value["entries"][index]
        entry["request_skeleton_sha256"] = canonical_sha256(
            entry["request_skeleton"]
        )
        unsigned_entry = dict(entry)
        unsigned_entry.pop("entry_identity_sha256")
        entry["entry_identity_sha256"] = canonical_sha256(unsigned_entry)
        unsigned_manifest = dict(value)
        unsigned_manifest.pop("canonical_identity_sha256")
        value["canonical_identity_sha256"] = canonical_sha256(
            unsigned_manifest
        )

    malformed_stack = _build_bootstrap_manifest().to_dict()
    malformed_stack["entries"][0]["request_skeleton"]["StackName"] = (
        "keep-glm52-h1g-fence"
    )
    rehash_entry_and_manifest(malformed_stack)
    with pytest.raises(ValueError, match="request_skeleton"):
        parse_fence_manifest(malformed_stack)

    foreign_stack = _build_bootstrap_manifest().to_dict()
    foreign_stack["entries"][0]["request_skeleton"]["StackName"] = (
        STACK_ID.replace(
            "01234567-89ab-cdef-0123-456789abcdef",
            "11234567-89ab-cdef-0123-456789abcdef",
        )
    )
    rehash_entry_and_manifest(foreign_stack)
    with pytest.raises(ValueError, match="entry derivation|StackName"):
        parse_fence_manifest(foreign_stack)

    oversize = _build_bootstrap_manifest().to_dict()
    oversize["entries"][0]["rendered_policy_bytes"] = 17_921
    rehash_entry_and_manifest(oversize)
    with pytest.raises(ValueError, match="design limit"):
        parse_fence_manifest(oversize)

    for mutation in ("inventory_hash", "observation"):
        source = _build_source_manifest(_build_bootstrap_manifest()).to_dict()
        seal = source["source_settlement"]
        if mutation == "inventory_hash":
            seal["inventory_sha256"] = SHA("e")
        else:
            seal["second_inventory_observation"] = {"foreign": True}
        unsigned_seal = dict(seal)
        unsigned_seal.pop("canonical_identity_sha256")
        seal["canonical_identity_sha256"] = canonical_sha256(unsigned_seal)
        unsigned_manifest = dict(source)
        unsigned_manifest.pop("canonical_identity_sha256")
        source["canonical_identity_sha256"] = canonical_sha256(
            unsigned_manifest
        )
        with pytest.raises(ValueError, match="inventory"):
            parse_fence_manifest(source)


def test_publication_rejects_policy_document_and_intrinsic_mutants() -> None:
    from glm52_enforcement.fence_artifacts import (
        FenceSlot,
        ManifestStage,
        publish_fence_stage,
    )

    entries = _build_entries(
        ManifestStage.SOURCE_SETTLED,
        bootstrap_coordinate=_coordinate(EXPECTED_KEYS[7]),
        bridge_seed_hash=SHA("0"),
    )
    by_slot = {entry.slot: entry.template_bytes for entry in entries}
    batch_template = json.loads(
        by_slot[FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION]
    )
    policies = []
    wrong_version = json.loads(json.dumps(batch_template))
    wrong_version["Resources"]["H1gProductionFenceBucketPolicy"][
        "Properties"
    ]["PolicyDocument"]["Version"] = "2008-10-17"
    policies.append(wrong_version)
    ref_intrinsic = json.loads(json.dumps(batch_template))
    ref_intrinsic["Resources"]["H1gProductionFenceBucketPolicy"][
        "Properties"
    ]["PolicyDocument"]["Ref"] = "Foreign"
    policies.append(ref_intrinsic)
    fn_intrinsic = json.loads(json.dumps(batch_template))
    fn_intrinsic["Resources"]["H1gProductionFenceBucketPolicy"][
        "Properties"
    ]["PolicyDocument"]["Statement"][0]["Fn::Sub"] = "foreign"
    policies.append(fn_intrinsic)
    dynamic_reference = json.loads(json.dumps(batch_template))
    dynamic_reference["Resources"]["H1gProductionFenceBucketPolicy"][
        "Properties"
    ]["PolicyDocument"]["Statement"][0]["Resource"] = (
        "{{resolve:ssm:foreign}}"
    )
    policies.append(dynamic_reference)

    for policy in policies:
        template_bytes = dict(by_slot)
        template_bytes[FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION] = (
            canonical_json_bytes(policy) + b"\n"
        )
        s3 = _FenceS3()
        with pytest.raises(ValueError, match="PolicyDocument|intrinsic|dynamic"):
            publish_fence_stage(
                stage=ManifestStage.SOURCE_SETTLED,
                template_bytes=template_bytes,
                manifest_builder=lambda _coordinates, _seed: b"{}\n",
                kms_key_arn=(
                    "arn:aws:kms:us-west-2:246813579024:key/"
                    "01234567-89ab-cdef-0123-456789abcdef"
                ),
                services=_services(s3),
            )
        assert not [call for call in s3.calls if call[0] == "put"]




def test_transition_request_exact_fields_hashes_and_stage_specific_nullability() -> None:
    from glm52_enforcement.fence_artifacts import (
        FenceSlot,
        ManifestStage,
        SourceSettledManifestCoordinate,
        build_fence_transition_request,
        parse_fence_transition_request,
    )

    bootstrap = _build_bootstrap_manifest()
    source = _build_source_manifest(bootstrap)
    bootstrap_coordinate = _coordinate(
        EXPECTED_KEYS[7], bootstrap.canonical_identity_sha256
    )
    source_coordinate = SourceSettledManifestCoordinate(
        bucket=BUCKET,
        key=EXPECTED_KEYS[8],
        version_id="source-manifest/+?",
        file_sha256=SHA("a"),
        canonical_identity_sha256=source.canonical_identity_sha256,
        source_settlement_identity_sha256=(
            source.source_settlement.canonical_identity_sha256
        ),
    )
    prepare = bootstrap.entry(FenceSlot.PREPARE_GENESIS_LIVE_STATE)
    prepare_request = build_fence_transition_request(
        manifest_coordinate=bootstrap_coordinate,
        manifest_stage=ManifestStage.BOOTSTRAP,
        selected_entry=prepare,
        support_runtime_identity_coordinate=None,
        observed_h1f_active_head_coordinate={"identity": SHA("1")},
        h1f_successor_coordinate=None,
        source_settled_manifest_coordinate=None,
    )
    prepare_value = prepare_request.to_dict()
    assert set(prepare_value) == REQUEST_FIELDS
    unsigned = dict(prepare_value)
    identity = unsigned.pop("canonical_identity_sha256")
    assert identity == canonical_sha256(unsigned)
    assert parse_fence_transition_request(prepare_value).to_dict() == prepare_value

    batch = source.entry(FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION)
    batch_request = build_fence_transition_request(
        manifest_coordinate=source_coordinate.artifact_coordinate,
        manifest_stage=ManifestStage.SOURCE_SETTLED,
        selected_entry=batch,
        support_runtime_identity_coordinate=_runtime_coordinate(),
        observed_h1f_active_head_coordinate={"identity": SHA("2")},
        h1f_successor_coordinate={"identity": SHA("3")},
        source_settled_manifest_coordinate=source_coordinate,
    )
    batch_value = batch_request.to_dict()
    assert batch_value["batch_projection"] == batch.to_dict()[
        "batch_projection_contract"
    ]
    assert parse_fence_transition_request(batch_value).to_dict() == batch_value

    malformed = dict(prepare_value)
    malformed["record_type"] = "glm52_fence_transition_request_v1"
    with pytest.raises(ValueError, match="field|v1|exact"):
        parse_fence_transition_request(malformed)
    for field, wrong in (
        ("support_runtime_identity_coordinate", _runtime_coordinate().to_dict()),
        ("h1f_successor_coordinate", {"identity": SHA("4")}),
        ("batch_projection", {}),
        ("source_settled_manifest_coordinate", source_coordinate.to_dict()),
    ):
        malformed = dict(prepare_value)
        malformed[field] = wrong
        with pytest.raises(ValueError, match=field + "|PREPARE"):
            parse_fence_transition_request(malformed)


def test_request_and_entry_frozen_projections_do_not_alias_mutable_inputs() -> None:
    from glm52_enforcement.fence_artifacts import parse_fence_manifest

    value = _build_bootstrap_manifest().to_dict()
    parsed = parse_fence_manifest(value)
    value["writer_inventory"][0]["binding_id"] = "mutated"
    assert parsed.to_dict()["writer_inventory"][0]["binding_id"] == (
        "artifact-publisher"
    )
    with pytest.raises(TypeError):
        parsed.entries[0].request_skeleton["StackName"] = "mutated"


def test_authenticated_wrapper_direct_construction_rejects_malformed_values() -> None:
    from glm52_enforcement.fence_artifacts import (
        BridgeSeedArtifact,
        FenceArtifactEntry,
        FenceManifest,
        FenceTransitionRequest,
        ManifestStage,
        SourceSettlementSeal,
        build_fence_transition_request,
    )

    source_seal = _source_seal()
    malformed_source = source_seal.to_dict()
    malformed_source["generation"] = 2
    with pytest.raises(ValueError, match="generation"):
        SourceSettlementSeal(_value=malformed_source)

    manifest = _build_bootstrap_manifest()
    malformed_seed = manifest.bridge_seed.to_dict()
    malformed_seed["artifact_kind"] = "FOREIGN"
    with pytest.raises(ValueError, match="discriminant"):
        BridgeSeedArtifact(
            _value=malformed_seed,
            raw_bytes=manifest.bridge_seed.raw_bytes,
        )

    entry = manifest.entries[0]
    malformed_entry = entry.to_dict()
    malformed_entry["transition_class"] = "FOREIGN"
    with pytest.raises(ValueError, match="transition_class"):
        FenceArtifactEntry(
            _value=malformed_entry,
            template_bytes=entry.template_bytes,
        )

    malformed_manifest = manifest.to_dict()
    malformed_manifest["schema_version"] = 1
    with pytest.raises(ValueError, match="record_type|v2"):
        FenceManifest(
            _value=malformed_manifest,
            entries=manifest.entries,
            bridge_seed=manifest.bridge_seed,
            source_settlement=manifest.source_settlement,
            support_runtime_identity_coordinate=(
                manifest.support_runtime_identity_coordinate
            ),
            bootstrap_manifest_coordinate=manifest.bootstrap_manifest_coordinate,
        )

    request = build_fence_transition_request(
        manifest_coordinate=_coordinate(
            EXPECTED_KEYS[7], manifest.canonical_identity_sha256
        ),
        manifest_stage=ManifestStage.BOOTSTRAP,
        selected_entry=entry,
        support_runtime_identity_coordinate=None,
        observed_h1f_active_head_coordinate={"identity": SHA("1")},
        h1f_successor_coordinate=None,
        source_settled_manifest_coordinate=None,
    )
    malformed_request = request.to_dict()
    malformed_request["manifest_stage"] = "FOREIGN"
    with pytest.raises(ValueError, match="manifest_stage"):
        FenceTransitionRequest(_value=malformed_request)


def test_direct_manifest_construction_rejects_entries_value_inconsistency() -> None:
    from glm52_enforcement.fence_artifacts import FenceManifest

    manifest = _build_bootstrap_manifest()
    with pytest.raises(ValueError, match="entries.*_value|entries.*match"):
        FenceManifest(
            _value=manifest.to_dict(),
            entries=tuple(reversed(manifest.entries)),
            bridge_seed=manifest.bridge_seed,
            source_settlement=manifest.source_settlement,
            support_runtime_identity_coordinate=(
                manifest.support_runtime_identity_coordinate
            ),
            bootstrap_manifest_coordinate=manifest.bootstrap_manifest_coordinate,
        )


def test_direct_authenticated_wrappers_deep_detach_nested_caller_state() -> None:
    from glm52_enforcement.fence_artifacts import (
        BridgeSeedArtifact,
        FenceArtifactEntry,
        FenceManifest,
        FenceTransitionRequest,
        ManifestStage,
        SourceSettlementSeal,
        build_fence_transition_request,
    )

    source_value = _source_seal().to_dict()
    source = SourceSettlementSeal(_value=source_value)
    source_value["selected_source_rows"][0]["version_id"] = "mutated"
    assert source.to_dict()["selected_source_rows"][0]["version_id"] == "source-0"

    manifest = _build_bootstrap_manifest()
    seed_value = manifest.bridge_seed.to_dict()
    seed = BridgeSeedArtifact(
        _value=seed_value,
        raw_bytes=manifest.bridge_seed.raw_bytes,
    )
    seed_value["statement_ledger"][0]["sid"] = "mutated"
    assert seed.to_dict()["statement_ledger"][0]["sid"] != "mutated"

    entry_value = manifest.entries[0].to_dict()
    entry = FenceArtifactEntry(
        _value=entry_value,
        template_bytes=manifest.entries[0].template_bytes,
    )
    entry_value["request_skeleton"]["StackName"] = "mutated"
    assert entry.request_skeleton["StackName"] == STACK_ID

    manifest_value = manifest.to_dict()
    direct_manifest = FenceManifest(
        _value=manifest_value,
        entries=manifest.entries,
        bridge_seed=manifest.bridge_seed,
        source_settlement=manifest.source_settlement,
        support_runtime_identity_coordinate=(
            manifest.support_runtime_identity_coordinate
        ),
        bootstrap_manifest_coordinate=manifest.bootstrap_manifest_coordinate,
    )
    manifest_value["writer_inventory"][0]["binding_id"] = "mutated"
    manifest_value["entries"][0]["request_skeleton"]["StackName"] = "mutated"
    assert direct_manifest.to_dict()["writer_inventory"][0]["binding_id"] == (
        "artifact-publisher"
    )
    assert direct_manifest.entries[0].request_skeleton["StackName"] == STACK_ID

    request = build_fence_transition_request(
        manifest_coordinate=_coordinate(
            EXPECTED_KEYS[7], manifest.canonical_identity_sha256
        ),
        manifest_stage=ManifestStage.BOOTSTRAP,
        selected_entry=manifest.entries[0],
        support_runtime_identity_coordinate=None,
        observed_h1f_active_head_coordinate={"identity": SHA("1")},
        h1f_successor_coordinate=None,
        source_settled_manifest_coordinate=None,
    )
    request_value = request.to_dict()
    direct_request = FenceTransitionRequest(_value=request_value)
    request_value["observed_h1f_active_head_coordinate"]["identity"] = "mutated"
    assert direct_request.to_dict()["observed_h1f_active_head_coordinate"][
        "identity"
    ] == SHA("1")


class _LostPut(RuntimeError):
    pass


class _FenceS3:
    def __init__(self, *, mode: str = "normal") -> None:
        self.mode = mode
        self.calls = []
        self.rows = {}
        self.version = 0

    @staticmethod
    def _metadata(label: str):
        return {
            "HTTPStatusCode": 200,
            "RequestId": label,
            "RetryAttempts": 0,
        }

    def get_bucket_versioning(self, **request):
        return {"Status": "Enabled", "ResponseMetadata": self._metadata("v")}

    def list_object_versions(self, **request):
        self.calls.append(("list", request["Prefix"]))
        key = request["Prefix"]
        rows = list(self.rows.get(key, ()))
        versions = [
            {"Key": key, "VersionId": row["version_id"], "Size": len(row["raw"])}
            for row in rows
        ]
        delete_markers = []
        if self.mode == "foreign":
            versions.append({"Key": key + ".foreign", "VersionId": "foreign", "Size": 1})
        if self.mode == "delete":
            delete_markers.append({"Key": key, "VersionId": "delete"})
        return {
            "Versions": versions,
            "DeleteMarkers": delete_markers,
            "IsTruncated": False,
            "ResponseMetadata": self._metadata("l"),
        }

    def put_object(self, **request):
        self.calls.append(("put", request["Key"]))
        assert set(request) == {
            "Bucket",
            "Key",
            "Body",
            "ContentType",
            "ChecksumAlgorithm",
            "ChecksumSHA256",
            "IfNoneMatch",
            "ExpectedBucketOwner",
            "Metadata",
            "ServerSideEncryption",
            "SSEKMSKeyId",
        }
        assert request["Metadata"] == {}
        assert request["ServerSideEncryption"] == "aws:kms"
        self.version += 1
        version_id = f"version-{self.version:04d}"
        self.rows.setdefault(request["Key"], []).append(
            {
                "version_id": version_id,
                "raw": request["Body"],
                "content_type": request["ContentType"],
                "checksum": request["ChecksumSHA256"],
                "kms": request["SSEKMSKeyId"],
            }
        )
        if self.mode == "lost":
            self.mode = "normal"
            raise _LostPut("lost response")
        return {
            "VersionId": version_id,
            "ChecksumSHA256": request["ChecksumSHA256"],
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": request["SSEKMSKeyId"],
            "ResponseMetadata": self._metadata("p"),
        }

    def get_object(self, **request):
        self.calls.append(("get", request["Key"]))
        row = next(
            row
            for row in self.rows[request["Key"]]
            if row["version_id"] == request["VersionId"]
        )
        return {
            "Body": io.BytesIO(row["raw"]),
            "ContentLength": len(row["raw"]),
            "ContentType": row["content_type"],
            "VersionId": row["version_id"],
            "ChecksumSHA256": row["checksum"],
            "Metadata": {},
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": row["kms"],
            "ObjectLockMode": None,
            "ObjectLockRetainUntilDate": None,
            "ObjectLockLegalHoldStatus": None,
            "ResponseMetadata": self._metadata("g"),
        }

    def get_object_tagging(self, **request):
        self.calls.append(("tags", request["Key"]))
        return {"TagSet": [], "ResponseMetadata": self._metadata("t")}


class _Sts:
    def get_caller_identity(self):
        return {
            "Account": ACCOUNT_ID,
            "Arn": "arn:aws:iam::246813579024:role/operator",
            "UserId": "AROATEST:operator",
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": "sts",
                "RetryAttempts": 0,
            },
        }


def _services(s3):
    from glm52_enforcement.task13_fixed_artifacts import Task13FixedArtifactServices

    return Task13FixedArtifactServices(
        sts=_Sts(), s3=s3, total_max_attempts=1
    )


def test_publication_uses_singular_primitive_and_manifest_last_order() -> None:
    from glm52_enforcement.fence_artifacts import (
        BOOTSTRAP_PUBLICATION_ORDER,
        ManifestStage,
        build_bridge_seed_artifact,
        build_fence_manifest,
        publish_fence_stage,
    )

    seed_rendered = _rendered("BRIDGE_SEED")
    entries = _build_entries(
        ManifestStage.BOOTSTRAP,
        bridge_seed_hash=seed_rendered.policy_sha256,
    )
    seed_bytes = seed_rendered.policy_bytes
    template_bytes = {
        entry.slot: entry.template_bytes for entry in entries
    }

    def manifest_builder(slot_coordinates, seed_coordinate):
        assert seed_coordinate is not None
        seed = build_bridge_seed_artifact(
            rendered=seed_rendered,
            expected_live_preseed_policy_sha256=SHA("f"),
            version_id=seed_coordinate.version_id,
        )
        bound_entries = _build_entries(
            ManifestStage.BOOTSTRAP,
            bridge_seed_hash=seed.policy_sha256,
            version_ids={
                slot: coordinate.version_id
                for slot, coordinate in slot_coordinates.items()
            },
        )
        manifest = build_fence_manifest(
            stage=ManifestStage.BOOTSTRAP,
            fields=_manifest_common_fields(),
            bridge_seed=seed,
            entries=bound_entries,
        )
        return canonical_json_bytes(manifest.to_dict()) + b"\n"

    kms = (
        "arn:aws:kms:us-west-2:246813579024:key/"
        "01234567-89ab-cdef-0123-456789abcdef"
    )
    s3 = _FenceS3(mode="lost")
    seed_checksum = base64.b64encode(hashlib.sha256(seed_bytes).digest()).decode(
        "ascii"
    )
    s3.rows[EXPECTED_KEYS[0]] = [
        {
            "version_id": "seed-version",
            "raw": seed_bytes,
            "content_type": "application/json",
            "checksum": seed_checksum,
            "kms": kms,
        }
    ]
    coordinates = publish_fence_stage(
        stage=ManifestStage.BOOTSTRAP,
        template_bytes=template_bytes,
        manifest_builder=manifest_builder,
        kms_key_arn=kms,
        services=_services(s3),
        bridge_seed_bytes=seed_bytes,
    )
    put_keys = [key for operation, key in s3.calls if operation == "put"]
    assert [coordinate.key for coordinate in coordinates] == list(
        BOOTSTRAP_PUBLICATION_ORDER
    )
    assert put_keys == list(BOOTSTRAP_PUBLICATION_ORDER[1:])
    assert coordinates[-1].key == EXPECTED_KEYS[7]
    assert len(s3.rows[EXPECTED_KEYS[0]]) == 1

    duplicate = _FenceS3()
    raw = seed_bytes
    checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    duplicate.rows[EXPECTED_KEYS[0]] = [
        {
            "version_id": "one",
            "raw": raw,
            "content_type": "application/json",
            "checksum": checksum,
            "kms": kms,
        },
        {
            "version_id": "two",
            "raw": raw,
            "content_type": "application/json",
            "checksum": checksum,
            "kms": kms,
        },
    ]
    with pytest.raises(ValueError, match="singular|multiple"):
        publish_fence_stage(
            stage=ManifestStage.BOOTSTRAP,
            template_bytes=template_bytes,
            manifest_builder=manifest_builder,
            kms_key_arn=kms,
            services=_services(duplicate),
            bridge_seed_bytes=seed_bytes,
        )
    for mode in ("foreign", "delete"):
        with pytest.raises(ValueError, match="foreign|delete"):
            publish_fence_stage(
                stage=ManifestStage.BOOTSTRAP,
                template_bytes=template_bytes,
                manifest_builder=manifest_builder,
                kms_key_arn=kms,
                services=_services(_FenceS3(mode=mode)),
                bridge_seed_bytes=seed_bytes,
            )


def test_publish_fence_stage_does_not_create_a_missing_bootstrap_seed() -> None:
    from glm52_enforcement.fence_artifacts import (
        ManifestStage,
        build_bridge_seed_artifact,
        build_fence_manifest,
        publish_fence_stage,
    )

    seed_rendered = _rendered("BRIDGE_SEED")
    entries = _build_entries(
        ManifestStage.BOOTSTRAP,
        bridge_seed_hash=seed_rendered.policy_sha256,
    )
    template_bytes = {
        entry.slot: entry.template_bytes for entry in entries
    }

    def manifest_builder(slot_coordinates, seed_coordinate):
        assert seed_coordinate is not None
        seed = build_bridge_seed_artifact(
            rendered=seed_rendered,
            expected_live_preseed_policy_sha256=SHA("f"),
            version_id=seed_coordinate.version_id,
        )
        bound_entries = _build_entries(
            ManifestStage.BOOTSTRAP,
            bridge_seed_hash=seed.policy_sha256,
            version_ids={
                slot: coordinate.version_id
                for slot, coordinate in slot_coordinates.items()
            },
        )
        manifest = build_fence_manifest(
            stage=ManifestStage.BOOTSTRAP,
            fields=_manifest_common_fields(),
            bridge_seed=seed,
            entries=bound_entries,
        )
        return canonical_json_bytes(manifest.to_dict()) + b"\n"

    s3 = _FenceS3()
    with pytest.raises(ValueError):
        publish_fence_stage(
            stage=ManifestStage.BOOTSTRAP,
            template_bytes=template_bytes,
            manifest_builder=manifest_builder,
            kms_key_arn=(
                "arn:aws:kms:us-west-2:246813579024:key/"
                "01234567-89ab-cdef-0123-456789abcdef"
            ),
            services=_services(s3),
            bridge_seed_bytes=seed_rendered.policy_bytes,
        )

    assert EXPECTED_KEYS[0] not in s3.rows
    assert ("put", EXPECTED_KEYS[0]) not in s3.calls


def test_stage_publication_builds_manifest_after_service_version_ids() -> None:
    from glm52_enforcement.fence_artifacts import (
        FenceSlot,
        ManifestStage,
        build_fence_manifest,
        publish_fence_stage,
    )

    bootstrap = _build_bootstrap_manifest()
    bootstrap_coordinate = _coordinate(
        EXPECTED_KEYS[7], bootstrap.canonical_identity_sha256
    )
    provisional_entries = _build_entries(
        ManifestStage.SOURCE_SETTLED,
        bootstrap_coordinate,
        bootstrap.bridge_seed.policy_sha256,
    )
    manifest_observations = []

    def manifest_after_versions(slot_coordinates, seed_coordinate):
        assert seed_coordinate is None
        manifest_observations.append(tuple(slot_coordinates.items()))
        entries = _build_entries(
            ManifestStage.SOURCE_SETTLED,
            bootstrap_coordinate,
            bootstrap.bridge_seed.policy_sha256,
            version_ids={
                slot: coordinate.version_id
                for slot, coordinate in slot_coordinates.items()
            },
        )
        manifest = build_fence_manifest(
            stage=ManifestStage.SOURCE_SETTLED,
            fields=_manifest_common_fields(),
            bridge_seed=bootstrap.bridge_seed,
            entries=entries,
            bootstrap_manifest_coordinate=bootstrap_coordinate,
            support_runtime_identity_coordinate=_runtime_coordinate(),
            source_settlement=_source_seal(),
        )
        return canonical_json_bytes(manifest.to_dict()) + b"\n"

    template_bytes = {
        entry.slot: entry.template_bytes for entry in provisional_entries
    }
    s3 = _FenceS3()
    coordinates = publish_fence_stage(
        stage=ManifestStage.SOURCE_SETTLED,
        template_bytes=template_bytes,
        manifest_builder=manifest_after_versions,
        kms_key_arn=(
            "arn:aws:kms:us-west-2:246813579024:key/"
            "01234567-89ab-cdef-0123-456789abcdef"
        ),
        services=_services(s3),
    )

    assert tuple(slot for slot, _coordinate_value in manifest_observations[0]) == (
        FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION,
        FenceSlot.TERMINAL,
    )
    assert [key for operation, key in s3.calls if operation == "put"] == [
        EXPECTED_KEYS[2],
        EXPECTED_KEYS[6],
        EXPECTED_KEYS[8],
    ]
    published_manifest = json.loads(s3.rows[EXPECTED_KEYS[8]][0]["raw"])
    assert {
        entry["version_id"] for entry in published_manifest["entries"]
    } == {coordinate.version_id for coordinate in coordinates[:2]}
