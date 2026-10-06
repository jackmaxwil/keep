from __future__ import annotations

from dataclasses import replace

import pytest


SHA = "a" * 64
SHA_B = "b" * 64


def _resource(resource_type: str, resource_id: str, cost_class: str):
    from glm52_enforcement.task12_orphan_audit import RetainedResource

    return RetainedResource(
        resource_type=resource_type,
        resource_id=resource_id,
        cost_class=cost_class,
    )


def _grant(grant_id: str, *, operations=("Decrypt",)):
    from glm52_enforcement.task12_orphan_audit import KmsGrantIdentity

    return KmsGrantIdentity(
        grant_id=grant_id,
        grant_name="grant-" + grant_id,
        grantee_principal="arn:aws:iam::246813579024:role/grantee",
        retiring_principal="arn:aws:iam::246813579024:role/retirer",
        operations=tuple(operations),
        encryption_context_identity_sha256=SHA,
    )


def _expected():
    return (
        _resource("LEDGER", "glm52-ledger", "DDB_RETAINED"),
        _resource("KMS_KEY", "key-1", "KMS_RETAINED"),
        _resource("EVIDENCE_BUCKET", "evidence-bucket", "S3_RETAINED"),
        _resource(
            "PRODUCTION_FENCE_STACK",
            "keep-glm52-production-fence",
            "CFN_RETAINED",
        ),
        _resource("LIFECYCLE_RESOURCE", "cleanup-schedule", "EVENTBRIDGE_RETAINED"),
        _resource(
            "SOURCE_PUBLISHER_IDENTITY",
            "source-publisher-role",
            "IAM_RETAINED",
        ),
    )


def _inventory(*, resources=None, complete: bool = True, covered=None):
    from glm52_enforcement.task12_orphan_audit import (
        AUDITED_RESOURCE_TYPES,
        ResourceInventoryScan,
    )

    return ResourceInventoryScan(
        covered_resource_types=(
            tuple(AUDITED_RESOURCE_TYPES) if covered is None else tuple(covered)
        ),
        resources=tuple(_expected() if resources is None else resources),
        unreadable_resource_ids=(),
        pagination_complete=complete,
        observed_at="2026-07-29T12:01:00Z",
        evidence_identity_sha256=SHA,
    )


def _direct(grant=None):
    from glm52_enforcement.task12_orphan_audit import DirectGrantAttribution

    return DirectGrantAttribution(
        grant=_grant("activation-direct") if grant is None else grant,
        provenance="DIRECT_RESPONSE",
        create_request_identity_sha256=SHA,
        create_response_identity_sha256=SHA_B,
        reconciliation_list_grants_identity_sha256=None,
        revoke_or_retire_identity_sha256=SHA,
    )


def test_direct_attribution_provenance_union_is_truthful() -> None:
    from glm52_enforcement.task12_orphan_audit import (
        DirectGrantAttribution,
        Task12OrphanAuditError,
        audit_orphans,
    )

    baseline = (_grant("baseline"),)
    reconciled = DirectGrantAttribution(
        grant=_grant("activation-direct"),
        provenance="LIST_GRANTS_RECONCILIATION",
        create_request_identity_sha256=SHA,
        create_response_identity_sha256=None,
        reconciliation_list_grants_identity_sha256=SHA_B,
        revoke_or_retire_identity_sha256=SHA,
    )
    proof = audit_orphans(
        expected_retained=_expected(),
        inventory=_inventory(),
        retained_grant_baseline=baseline,
        pre_cleanup_grants=_grant_scan(
            baseline + (reconciled.grant,),
            observed_at="2026-07-29T12:01:00Z",
        ),
        final_grants=_grant_scan(
            baseline,
            observed_at="2026-07-29T12:05:00Z",
        ),
        direct_grants=(reconciled,),
        service_grants=(),
        settling_deadline="2026-07-29T12:04:00Z",
    )
    assert proof.final_grants == baseline

    for foreign in (
        replace(
            reconciled,
            create_response_identity_sha256=SHA,
        ),
        replace(
            reconciled,
            reconciliation_list_grants_identity_sha256=None,
        ),
        replace(
            reconciled,
            provenance="UNKNOWN",
        ),
        replace(
            _direct(),
            reconciliation_list_grants_identity_sha256=SHA,
        ),
    ):
        with pytest.raises(Task12OrphanAuditError, match="direct grant"):
            audit_orphans(
                expected_retained=_expected(),
                inventory=_inventory(),
                retained_grant_baseline=baseline,
                pre_cleanup_grants=_grant_scan(
                    baseline + (foreign.grant,),
                    observed_at="2026-07-29T12:01:00Z",
                ),
                final_grants=_grant_scan(
                    baseline,
                    observed_at="2026-07-29T12:05:00Z",
                ),
                direct_grants=(foreign,),
                service_grants=(),
                settling_deadline="2026-07-29T12:04:00Z",
            )


def _service(grant=None):
    from glm52_enforcement.task12_orphan_audit import ServiceGrantAttribution

    return ServiceGrantAttribution(
        grant=_grant("activation-service") if grant is None else grant,
        baseline_diff_identity_sha256=SHA,
        list_grants_identity_sha256=SHA_B,
        cloudtrail_request_identity_sha256=SHA,
        cloudtrail_response_identity_sha256=SHA_B,
        originating_resource_identity="arn:aws:ec2:us-west-2:246813579024:volume/vol-1",
        originating_service="ec2.amazonaws.com",
        encryption_context_correlation_identity_sha256=SHA,
        revoke_or_retire_identity_sha256=SHA_B,
    )


def _grant_scan(grants, *, observed_at: str, complete: bool = True):
    from glm52_enforcement.task12_orphan_audit import KmsGrantScan

    return KmsGrantScan(
        grants=tuple(grants),
        pagination_complete=complete,
        observed_at=observed_at,
        list_grants_identity_sha256=SHA,
    )


def _audit():
    from glm52_enforcement.task12_orphan_audit import audit_orphans

    baseline = (_grant("baseline"),)
    direct = _direct()
    service = _service()
    return audit_orphans(
        expected_retained=_expected(),
        inventory=_inventory(),
        retained_grant_baseline=baseline,
        pre_cleanup_grants=_grant_scan(
            baseline + (direct.grant, service.grant),
            observed_at="2026-07-29T12:01:00Z",
        ),
        final_grants=_grant_scan(
            baseline,
            observed_at="2026-07-29T12:05:00Z",
        ),
        direct_grants=(direct,),
        service_grants=(service,),
        settling_deadline="2026-07-29T12:04:00Z",
    )


def test_orphan_audit_accepts_exact_retained_inventory_and_grant_baseline() -> None:
    proof = _audit()
    assert proof.retained_resources == _expected()
    assert proof.final_grants == (_grant("baseline"),)
    assert proof.orphan_resource_ids == ()


def test_orphan_audit_rejects_unknown_billable_and_unreadable_resources() -> None:
    from glm52_enforcement.task12_orphan_audit import (
        ResourceInventoryScan,
        Task12OrphanAuditError,
        audit_orphans,
    )

    extra = _resource("EC2_INSTANCE", "i-0123456789abcdef0", "EC2_BILLABLE")
    bad_inventory = replace(
        _inventory(resources=_expected() + (extra,)),
        unreadable_resource_ids=("eni-unknown",),
    )
    with pytest.raises(
        Task12OrphanAuditError,
        match="unknown, unreadable, missing, or billable",
    ):
        audit_orphans(
            expected_retained=_expected(),
            inventory=bad_inventory,
            retained_grant_baseline=(),
            pre_cleanup_grants=_grant_scan(
                (), observed_at="2026-07-29T12:01:00Z"
            ),
            final_grants=_grant_scan(
                (), observed_at="2026-07-29T12:05:00Z"
            ),
            direct_grants=(),
            service_grants=(),
            settling_deadline="2026-07-29T12:04:00Z",
        )
    assert isinstance(bad_inventory, ResourceInventoryScan)


def test_orphan_audit_requires_every_resource_family_and_complete_pagination() -> None:
    from glm52_enforcement.task12_orphan_audit import (
        AUDITED_RESOURCE_TYPES,
        Task12OrphanAuditError,
        audit_orphans,
    )

    with pytest.raises(Task12OrphanAuditError, match="resource-family"):
        audit_orphans(
            expected_retained=_expected(),
            inventory=_inventory(covered=AUDITED_RESOURCE_TYPES[:-1]),
            retained_grant_baseline=(),
            pre_cleanup_grants=_grant_scan(
                (), observed_at="2026-07-29T12:01:00Z"
            ),
            final_grants=_grant_scan(
                (), observed_at="2026-07-29T12:05:00Z"
            ),
            direct_grants=(),
            service_grants=(),
            settling_deadline="2026-07-29T12:04:00Z",
        )
    with pytest.raises(Task12OrphanAuditError, match="complete"):
        audit_orphans(
            expected_retained=_expected(),
            inventory=_inventory(complete=False),
            retained_grant_baseline=(),
            pre_cleanup_grants=_grant_scan(
                (), observed_at="2026-07-29T12:01:00Z"
            ),
            final_grants=_grant_scan(
                (), observed_at="2026-07-29T12:05:00Z"
            ),
            direct_grants=(),
            service_grants=(),
            settling_deadline="2026-07-29T12:04:00Z",
        )


def test_kms_unknown_duplicate_or_overbroad_grant_blocks_drained() -> None:
    from glm52_enforcement.task12_orphan_audit import (
        Task12OrphanAuditError,
        audit_orphans,
    )

    baseline = (_grant("baseline"),)
    direct = _direct()
    unknown = _grant("unknown", operations=("CreateGrant", "Decrypt"))
    with pytest.raises(Task12OrphanAuditError, match="unknown or overbroad"):
        audit_orphans(
            expected_retained=_expected(),
            inventory=_inventory(),
            retained_grant_baseline=baseline,
            pre_cleanup_grants=_grant_scan(
                baseline + (direct.grant, unknown),
                observed_at="2026-07-29T12:01:00Z",
            ),
            final_grants=_grant_scan(
                baseline, observed_at="2026-07-29T12:05:00Z"
            ),
            direct_grants=(direct,),
            service_grants=(),
            settling_deadline="2026-07-29T12:04:00Z",
        )


def test_service_grant_requires_cloudtrail_resource_context_and_settling() -> None:
    from glm52_enforcement.task12_orphan_audit import (
        Task12OrphanAuditError,
        audit_orphans,
    )

    baseline = (_grant("baseline"),)
    service = replace(_service(), originating_service="")
    with pytest.raises(Task12OrphanAuditError, match="service attribution"):
        audit_orphans(
            expected_retained=_expected(),
            inventory=_inventory(),
            retained_grant_baseline=baseline,
            pre_cleanup_grants=_grant_scan(
                baseline + (service.grant,),
                observed_at="2026-07-29T12:01:00Z",
            ),
            final_grants=_grant_scan(
                baseline,
                observed_at="2026-07-29T12:03:59Z",
            ),
            direct_grants=(),
            service_grants=(service,),
            settling_deadline="2026-07-29T12:04:00Z",
        )

    with pytest.raises(Task12OrphanAuditError, match="settling"):
        audit_orphans(
            expected_retained=_expected(),
            inventory=_inventory(),
            retained_grant_baseline=baseline,
            pre_cleanup_grants=_grant_scan(
                baseline + (_service().grant,),
                observed_at="2026-07-29T12:01:00Z",
            ),
            final_grants=_grant_scan(
                baseline,
                observed_at="2026-07-29T12:03:59Z",
            ),
            direct_grants=(),
            service_grants=(_service(),),
            settling_deadline="2026-07-29T12:04:00Z",
        )


def test_final_grants_must_equal_frozen_baseline_exactly() -> None:
    from glm52_enforcement.task12_orphan_audit import (
        Task12OrphanAuditError,
        audit_orphans,
    )

    baseline = (_grant("baseline"),)
    direct = _direct()
    with pytest.raises(Task12OrphanAuditError, match="baseline"):
        audit_orphans(
            expected_retained=_expected(),
            inventory=_inventory(),
            retained_grant_baseline=baseline,
            pre_cleanup_grants=_grant_scan(
                baseline + (direct.grant,),
                observed_at="2026-07-29T12:01:00Z",
            ),
            final_grants=_grant_scan(
                baseline + (direct.grant,),
                observed_at="2026-07-29T12:05:00Z",
            ),
            direct_grants=(direct,),
            service_grants=(),
            settling_deadline="2026-07-29T12:04:00Z",
        )


def test_h1g_drained_marker_is_last_for_settled_or_transferred_liability() -> None:
    from glm52_enforcement.task12_orphan_audit import (
        MarkerLastPrerequisites,
        Task12OrphanAuditError,
        build_h1g_drained_prerequisites,
    )

    base = MarkerLastPrerequisites(
        terminal_v2_identity_sha256=SHA,
        finalization_identity_sha256=SHA_B,
        snapshot_cleanup_control_identity_sha256=SHA,
        controller_quiesced_identity_sha256=SHA_B,
        spend_ledger_head_identity_sha256=SHA,
        orphan_audit=_audit(),
        support_stack_absent=True,
        snapshot_cleanup_armed=True,
        all_workers_terminal=True,
        all_allocations_closed=True,
        liability_state="SETTLED",
        liability_identity_sha256=SHA_B,
    )
    result = build_h1g_drained_prerequisites(base)
    assert result.marker_write_order == "LAST_CONDITIONAL_CREATE"
    transferred = build_h1g_drained_prerequisites(
        replace(base, liability_state="RETAINED_TERMINATION_ONLY")
    )
    assert transferred.liability_state == "RETAINED_TERMINATION_ONLY"

    with pytest.raises(Task12OrphanAuditError, match="snapshot"):
        build_h1g_drained_prerequisites(
            replace(base, snapshot_cleanup_armed=False)
        )
    with pytest.raises(Task12OrphanAuditError, match="orphan audit"):
        build_h1g_drained_prerequisites(
            replace(
                base,
                orphan_audit=replace(
                    base.orphan_audit,
                    canonical_identity_sha256=SHA_B,
                ),
            )
        )
    with pytest.raises(Task12OrphanAuditError, match="liability"):
        build_h1g_drained_prerequisites(
            replace(
                base,
                liability_state="UNRESOLVED",
                liability_identity_sha256=None,
            )
        )
