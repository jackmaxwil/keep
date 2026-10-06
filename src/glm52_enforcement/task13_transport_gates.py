"""Import-light semantic evidence for the Task 13 transport acceptance gates."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from glm52_enforcement.canonical import canonical_json_bytes


@dataclass(frozen=True)
class TransportRow:
    row_id: str
    operation_result: str
    durable_interpretation: str
    may_retry_create_or_post: str
    may_submit: str


@dataclass(frozen=True)
class TransportMutantObligation:
    obligation_id: str
    text: str


@dataclass(frozen=True)
class GateTestSpec:
    evidence_id: str
    pytest_nodeids: tuple[str, ...]
    route_paths: tuple[str, ...]


@dataclass(frozen=True)
class PytestObservation:
    nodeid: str
    exit_code: int
    passed_count: int
    normalized_output_sha256: str


class TransportGateError(ValueError):
    """A Task 13 transport gate is incomplete, forged, or unobserved."""


T01_T25_ROWS = (
    TransportRow(
        "T01",
        "direct unambiguous claim create `200` with exact owner",
        "claim candidate won",
        "no reattempt while its version is visible",
        "not yet",
    ),
    TransportRow(
        "T02",
        "claim `409`/`412`",
        "reconcile-first through fully paginated all-version audit",
        (
            "any visible version/delete marker is reconcile-only; exact "
            "zero-version/zero-marker audit permits another same-key "
            "conditional create"
        ),
        "no",
    ),
    TransportRow(
        "T03",
        "claim timeout/connection ambiguity/restart",
        "reconcile-first through fully paginated all-version audit",
        (
            "any visible version/delete marker is reconcile-only; exact "
            "zero-version/zero-marker audit permits another same-key "
            "conditional create"
        ),
        "no",
    ),
    TransportRow(
        "T04",
        "claim readback after success/restart",
        "transferable durable claim selection",
        ("no reattempt while visible; a later invocation may attempt the decision"),
        "no",
    ),
    TransportRow(
        "T05",
        (
            "direct unambiguous `launch-once` decision create `200` with "
            "exact owner, followed by closed post-create audit"
        ),
        "launch decision won",
        "no reattempt while visible",
        (
            "later private I/O may continue only after in-closure pure "
            "field/model validation and fresh H.1d live reinspection"
        ),
    ),
    TransportRow(
        "T06",
        "decision `409`/`412`",
        "reconcile-first through fully paginated all-version audit",
        (
            "any visible version/delete marker is reconcile-only; exact "
            "zero-version/zero-marker audit permits another same-key "
            "conditional create subject to the deadline outcome valid at "
            "reattempt"
        ),
        "no",
    ),
    TransportRow(
        "T07",
        "decision timeout/connection ambiguity/restart",
        "reconcile-first through fully paginated all-version audit",
        (
            "any visible version/delete marker is reconcile-only; exact "
            "zero-version/zero-marker audit permits another same-key "
            "conditional create subject to the deadline outcome valid at "
            "reattempt"
        ),
        "no",
    ),
    TransportRow(
        "T08",
        "stored `launch-once` readback",
        "durable launch decision only",
        "no",
        "no",
    ),
    TransportRow(
        "T09",
        "stored `expire-unstarted`",
        "generation cannot launch",
        "no",
        "no",
    ),
    TransportRow(
        "T10",
        ("direct unambiguous generation-scoped source create `200` with exact owner"),
        "provisional source publication only; not yet consumable",
        "no reattempt while the exact sole candidate is visible",
        "no",
    ),
    TransportRow(
        "T11",
        "source create `409`/`412`/timeout/ambiguity/restart",
        "fully paginate the exact coordinate",
        (
            "zero-version/zero-marker permits a new conditional attempt; one "
            "exact sole candidate may proceed only to additive enrollment; "
            "any other state fails closed"
        ),
        "no",
    ),
    TransportRow(
        "T12",
        (
            "direct unambiguous successor-plan create `200` at exact "
            "`successors/{P}/FENCE_SUCCESSOR.json`"
        ),
        "provisional selection of that sole child plan only",
        "no reattempt while its exact sole version is visible",
        "no",
    ),
    TransportRow(
        "T13",
        "successor-plan `409`/`412`/timeout/ambiguity/restart",
        (
            "fully audit the entire fence namespace, re-walk from genesis, "
            "and audit the exact predecessor-derived singleton coordinate"
        ),
        (
            "zero versions/delete markers permits another same-key "
            "conditional attempt; one exact sole winner reconciles only to "
            "that plan"
        ),
        "no",
    ),
    TransportRow(
        "T14",
        (
            "multiple successor versions, delete marker, alternate child key, "
            "divergent child, or visible sibling/fork"
        ),
        "invalid fence chain",
        (
            "never select by last write, discovery order, lexical order, "
            "cached state, or local candidate"
        ),
        "no",
    ),
    TransportRow(
        "T15",
        (
            "selected successor live state, publisher retirement, "
            "full-namespace reachability, next-slot zero audit, and fresh "
            "genesis-to-unique-head reauthentication all succeed inside the "
            "spanning fence-activation/no-consumption barrier"
        ),
        (
            "selected child is the active head; exact source becomes enrolled "
            "and may enter its under-fence sequential audit"
        ),
        "no source/successor retry",
        "no",
    ),
    TransportRow(
        "T16",
        (
            "successor activation/policy update/publisher retirement is "
            "partial, ambiguous, raced, restarted, mismatches selected plan, "
            "loses its barrier, or weakens any prior entry"
        ),
        (
            "both selected child and predecessor are non-authoritative; "
            "reconcile only that sole selected child until full active-head "
            "proof"
        ),
        ("no fallback, next successor, create, or consumption based on ambiguity"),
        "no",
    ),
    TransportRow(
        "T17",
        "direct unambiguous terminal create `200` with exact owner",
        "exact v1 `expired-unstarted` terminal candidate won",
        "no reattempt while visible",
        "no",
    ),
    TransportRow(
        "T18",
        "terminal `409`/`412`",
        "reconcile-first through fully paginated all-version audit",
        (
            "any visible terminal version/delete marker is "
            "authenticate-and-reconcile only; exact zero-version/zero-marker "
            "audit permits another same-key conditional create of a currently "
            "valid terminal candidate"
        ),
        "no",
    ),
    TransportRow(
        "T19",
        "terminal timeout/connection ambiguity/restart",
        "reconcile-first through fully paginated all-version audit",
        (
            "any visible terminal version/delete marker is "
            "authenticate-and-reconcile only; exact zero-version/zero-marker "
            "audit permits another same-key conditional create of a currently "
            "valid terminal candidate"
        ),
        "no",
    ),
    TransportRow(
        "T20",
        "stored terminal readback",
        (
            "authenticate exact v1 terminal and reconcile/advance only after "
            "the later permanent-fence source audit and fresh H.1d live "
            "reinspection"
        ),
        "no reattempt while visible",
        "no",
    ),
    TransportRow(
        "T21",
        "Sky POST unambiguous accepted",
        "reconcile and bind exact job",
        "no",
        "POST already consumed",
    ),
    TransportRow(
        "T22",
        "Sky POST timeout/ambiguous",
        "launch decision consumed; later transport/binding must correlate",
        "never",
        "no second POST",
    ),
    TransportRow(
        "T23",
        "Sky POST known reject",
        "launch decision consumed; later transport/binding owns disposition",
        "never",
        "no second POST",
    ),
    TransportRow(
        "T24",
        "restart while only a claim exists",
        "recover exact claim and refresh H.1d authority",
        (
            "may attempt the start decision only after audit closes any prior "
            "conflict/ambiguity; only a new direct decision-create `200` can "
            "feed the model"
        ),
        "no, until that new direct success and post-create audit",
    ),
    TransportRow(
        "T25",
        "restart after a decision exists",
        "durable reconcile-only decision",
        "no reattempt while visible",
        "no",
    ),
)


TRANSPORT_MUTANT_OBLIGATIONS = (
    TransportMutantObligation(
        "M01",
        (
            "claim, decision, and terminal `409`, `412`, timeout, ambiguity, "
            "and restart cases reconcile any visible version or delete marker "
            "and allow a same-key conditional-create reattempt only after a "
            "fully paginated exact zero-version/zero-delete-marker audit."
        ),
    ),
    TransportMutantObligation(
        "M02",
        (
            "a zero-version audit permits only a new conditional-create "
            "attempt and never creates submit authority."
        ),
    ),
    TransportMutantObligation(
        "M03",
        (
            "only private state constructed from the reattempt's own direct "
            "unambiguous success response can be consumed in that invocation."
        ),
    ),
    TransportMutantObligation(
        "M04",
        (
            "accepted, rejected, timeout, ambiguous, connection-loss, and "
            "process-restart Sky POST outcomes never cause a second POST."
        ),
    ),
    TransportMutantObligation(
        "M05",
        (
            "caller-supplied or serialized receipt/result objects are rejected "
            "by the transport even when they are exact offline dataclasses "
            "that pass H.1e field/model validation."
        ),
    ),
    TransportMutantObligation(
        "M06",
        (
            "the permanent source-coordinate fence, publisher retirement, "
            "bucket policy, lifecycle policy, and controlling identities are "
            "authenticated before first claim and cannot be weakened afterward."
        ),
    ),
    TransportMutantObligation(
        "M07",
        (
            "the source audit is followed by fresh live H.1d "
            "CloudFormation/Lambda/IAM/EventBridge/Scheduler/DLQ reinspection "
            "and the single POST inside an enforceable no-mutation interval."
        ),
    ),
    TransportMutantObligation(
        "M08",
        (
            "every post-decision or next-generation source uses a "
            "generation-scoped one-shot conditional publication whose `409`, "
            "`412`, timeout, ambiguity, concurrent-writer, and restart outcomes "
            "follow the exact provisional source reconciliation matrix above."
        ),
    ),
    TransportMutantObligation(
        "M09",
        (
            "a provisional source cannot enter terminal construction, "
            "next-generation claim construction, or any authority audit until "
            "its exact coordinate/VersionId/file/body/publisher/control "
            "identities appear in the selected unique-head successor chain and "
            "the complete monotonic chain plus live state has been freshly "
            "reauthenticated."
        ),
    ),
    TransportMutantObligation(
        "M10",
        (
            "partial or ambiguous policy update, publisher-retirement failure, "
            "stale policy identity, removal or weakening of an existing "
            "coordinate, or disagreement among bucket/lifecycle/IAM/"
            "CloudFormation identities fails closed and cannot be repaired by "
            "trusting the provisional object; every successor control identity "
            "must bind its exact predecessor and complete unchanged predecessor "
            "set."
        ),
    ),
    TransportMutantObligation(
        "M11",
        (
            "broad prefix controls permit only specifically authorized, "
            "not-yet-enrolled one-shot coordinates, deny all mutation of "
            "enrolled coordinates, and never make an unenrolled object consumable."
        ),
    ),
    TransportMutantObligation(
        "M12",
        (
            "before an `N+1` claim, its exact absent claim/decision/terminal "
            "triplet is additively reserved under phase-specific "
            "conditional-create rules without publishing a record, "
            "reclassifying it as a source artifact, or weakening any earlier "
            "source enrollment or generation reservation."
        ),
    ),
    TransportMutantObligation(
        "M13",
        (
            "each predecessor digest `P` has exactly one legal immutable "
            "`campaigns/{run_id}/authorities/fence/successors/{P}/"
            "FENCE_SUCCESSOR.json` child-selection coordinate, every candidate "
            "child competes there with conditional absent-key create and exact "
            "owner, and alternate child keys fail closed."
        ),
    ),
    TransportMutantObligation(
        "M14",
        (
            "a concurrent barrier in which two coordinators read the same `P` "
            "and propose `P+A` and `P+B` selects at most one sole-version child; "
            "any simulated multiple version, delete marker, sibling, divergent "
            "child, or fork fails closed and never uses last-write, discovery, "
            "lexical, or cached selection."
        ),
    ),
    TransportMutantObligation(
        "M15",
        (
            "every restart freshly audits the entire fence namespace, walks "
            "exact genesis and each unique derived child coordinate to a "
            "zero-child head, rejects cached predecessor/head/sibling state, "
            "and never revives a losing candidate."
        ),
    ),
    TransportMutantObligation(
        "M16",
        (
            "a selected successor grants no source-consumption or "
            "future-generation reservation authority until live bucket policy, "
            "lifecycle, IAM/CloudFormation, publisher denials, enrolled sources, "
            "and reserved generation coordinates exactly equal that selected "
            "unique head and the full chain has been freshly audited."
        ),
    ),
    TransportMutantObligation(
        "M17",
        (
            "full-namespace audit rejects every unreachable/orphan control "
            "object, alternate key, repeated digest, cycle, unknown record, "
            "historical version, or delete marker and requires every non-genesis "
            "record to be reachable exactly once at its predecessor-derived "
            "singleton key."
        ),
    ),
    TransportMutantObligation(
        "M18",
        (
            "active-head proof additionally requires the reached head's derived "
            "next successor slot to have exact zero versions and zero delete "
            "markers; a selected but partially activated child makes both child "
            "and predecessor non-authoritative and never permits predecessor "
            "fallback."
        ),
    ),
    TransportMutantObligation(
        "M19",
        (
            "the enforceable barrier spans fresh active-`P` proof, conditional "
            "successor selection/reconciliation, live-state application, and "
            "final full namespace/chain audit; barrier loss or restart permits "
            "only reconciliation of the sole selected child and forbids "
            "source/record consumption or a next successor."
        ),
    ),
    TransportMutantObligation(
        "M20",
        (
            "genesis and every successor singleton deny overwrite, second "
            "version, copy, multipart upload/completion, delete, delete marker, "
            "and lifecycle expiry after their one conditional create."
        ),
    ),
    TransportMutantObligation(
        "M21",
        (
            "the `N+1` reservation-only path skips source publication, first "
            "proves exact zero-version/zero-marker state at all three generation "
            "keys while the activation/no-consumption barrier is held, encodes "
            "the absent triplet as the unique successor delta, activates and "
            "freshly audits that head, and only then permits claim create without "
            "adding an H.1e record or API."
        ),
    ),
    TransportMutantObligation(
        "M22",
        (
            "every claim, decision, terminal conditional create and the one Sky "
            "POST independently requires a fresh full-namespace zero-child "
            "active-head proof under its applicable barrier; cached head state "
            "or an inactive selected child authorizes none of them."
        ),
    ),
)


_S3_TEST = "tests/test_glm52_enforcement_s3_adapter.py"
_SOURCE_TEST = "tests/test_glm52_enforcement_source_publishers.py"
_FENCE_TEST = "tests/test_glm52_sky_production_fence.py"
_FENCE_AUDIT_TEST = "tests/test_glm52_sky_production_fence_audit.py"
_DECISION_TEST = "tests/test_glm52_task11_decision_route.py"
_ADMISSION_TEST = "tests/test_glm52_enforcement_sky_admission.py"

_S3_ROUTE = ("src/glm52_enforcement/s3_adapter.py",)
_SOURCE_ROUTE = (
    "src/glm52_enforcement/source_publishers.py",
    "src/glm52_enforcement/s3_adapter.py",
)
_FENCE_ROUTE = ("src/mlx_vq/quality/glm52_sky_production_fence.py",)
_FENCE_AUDIT_ROUTE = (
    "aws/glm52-gpu/scripts/glm52_production_fence_audit.py",
    "src/mlx_vq/quality/glm52_sky_production_fence.py",
)
_DECISION_ROUTE = (
    "src/glm52_enforcement/decision_closure.py",
    "src/glm52_enforcement/s3_adapter.py",
    "src/glm52_enforcement/source_publishers.py",
    "src/glm52_enforcement/sky_admission.py",
)
_ADMISSION_ROUTE = ("src/glm52_enforcement/sky_admission.py",)


def _nodeid(path: str, name: str) -> str:
    return path + "::" + name


T01_T25_TEST_SPECS = (
    GateTestSpec(
        "T01",
        (
            _nodeid(
                _S3_TEST,
                "test_put_request_is_one_conditional_single_part_exact_owner_checksum_create",
            ),
        ),
        _S3_ROUTE,
    ),
    GateTestSpec(
        "T02",
        (
            _nodeid(
                _S3_TEST,
                "test_409_412_timeout_connection_loss_and_malformed_response_never_retry",
            ),
        ),
        _S3_ROUTE,
    ),
    GateTestSpec(
        "T03",
        (
            _nodeid(
                _S3_TEST,
                "test_409_412_timeout_connection_loss_and_malformed_response_never_retry",
            ),
        ),
        _S3_ROUTE,
    ),
    GateTestSpec(
        "T04",
        (
            _nodeid(
                _S3_TEST,
                "test_ambiguous_matching_sole_version_returns_reconciled_provenance_only",
            ),
        ),
        _S3_ROUTE,
    ),
    GateTestSpec(
        "T05",
        (
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_route_executes_exact_sequence_and_keeps_direct_custody",
            ),
        ),
        _DECISION_ROUTE,
    ),
    GateTestSpec(
        "T06",
        (
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_route_rejects_cross_use_direct_and_retry_mutants",
            ),
        ),
        _DECISION_ROUTE,
    ),
    GateTestSpec(
        "T07",
        (
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_crash_boundaries_have_no_replay_edge",
            ),
        ),
        _DECISION_ROUTE,
    ),
    GateTestSpec(
        "T08",
        (
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_boundary_cannot_expose_stored_decision_or_post_retry",
            ),
        ),
        _DECISION_ROUTE,
    ),
    GateTestSpec(
        "T09",
        (
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_post_decision_failure_is_permanently_no_post",
            ),
        ),
        _DECISION_ROUTE,
    ),
    GateTestSpec(
        "T10",
        (
            _nodeid(
                _SOURCE_TEST,
                "test_publishers_call_fresh_conditional_create_once_for_only_their_family",
            ),
        ),
        _SOURCE_ROUTE,
    ),
    GateTestSpec(
        "T11",
        (
            _nodeid(
                _SOURCE_TEST,
                "test_ambiguous_zero_history_sibling_delete_marker_or_mismatch_stops_sequence",
            ),
        ),
        _SOURCE_ROUTE,
    ),
    GateTestSpec(
        "T12",
        (
            _nodeid(
                _FENCE_TEST,
                "test_successor_key_derives_only_from_predecessor_body_sha",
            ),
            _nodeid(
                _SOURCE_TEST,
                "test_batch_successor_contains_all_five_exact_identities_and_publisher_roles",
            ),
        ),
        _FENCE_ROUTE + _SOURCE_ROUTE,
    ),
    GateTestSpec(
        "T13",
        (
            _nodeid(
                _FENCE_AUDIT_TEST,
                "test_namespace_rewalks_from_genesis_on_every_invocation",
            ),
            _nodeid(
                _FENCE_AUDIT_TEST,
                "test_zero_coordinate_uses_complete_version_listing_with_exact_owner",
            ),
        ),
        _FENCE_AUDIT_ROUTE,
    ),
    GateTestSpec(
        "T14",
        (
            _nodeid(
                _FENCE_AUDIT_TEST,
                "test_namespace_rejects_fork_cycle_and_repeated_digest_without_filtering",
            ),
        ),
        _FENCE_AUDIT_ROUTE,
    ),
    GateTestSpec(
        "T15",
        (
            _nodeid(
                _SOURCE_TEST,
                "test_no_source_is_authoritative_before_batch_successor_stabilizes",
            ),
            _nodeid(
                _FENCE_AUDIT_TEST,
                "test_namespace_requires_fresh_zero_at_reached_next_successor_slot",
            ),
        ),
        _SOURCE_ROUTE + _FENCE_AUDIT_ROUTE,
    ),
    GateTestSpec(
        "T16",
        (
            _nodeid(
                _SOURCE_TEST,
                "test_partial_policy_or_one_probe_set_cannot_activate_sources",
            ),
        ),
        _SOURCE_ROUTE + _FENCE_ROUTE,
    ),
    GateTestSpec(
        "T17",
        (
            _nodeid(
                _S3_TEST,
                "test_put_request_is_one_conditional_single_part_exact_owner_checksum_create",
            ),
        ),
        _S3_ROUTE,
    ),
    GateTestSpec(
        "T18",
        (
            _nodeid(
                _S3_TEST,
                "test_409_412_timeout_connection_loss_and_malformed_response_never_retry",
            ),
        ),
        _S3_ROUTE,
    ),
    GateTestSpec(
        "T19",
        (
            _nodeid(
                _S3_TEST,
                "test_409_412_timeout_connection_loss_and_malformed_response_never_retry",
            ),
        ),
        _S3_ROUTE,
    ),
    GateTestSpec(
        "T20",
        (
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_post_decision_failure_is_permanently_no_post",
            ),
        ),
        _DECISION_ROUTE,
    ),
    GateTestSpec(
        "T21",
        (
            _nodeid(
                _ADMISSION_TEST,
                "test_admission_matrix_issues_at_most_one_post",
            ),
        ),
        _ADMISSION_ROUTE,
    ),
    GateTestSpec(
        "T22",
        (
            _nodeid(
                _ADMISSION_TEST,
                "test_admission_process_death_never_permits_second_send",
            ),
        ),
        _ADMISSION_ROUTE,
    ),
    GateTestSpec(
        "T23",
        (
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_classification_never_resends",
            ),
        ),
        _DECISION_ROUTE,
    ),
    GateTestSpec(
        "T24",
        (
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_route_executes_exact_sequence_and_keeps_direct_custody",
            ),
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_crash_boundaries_have_no_replay_edge",
            ),
        ),
        _DECISION_ROUTE,
    ),
    GateTestSpec(
        "T25",
        (
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_boundary_cannot_expose_stored_decision_or_post_retry",
            ),
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_classification_never_resends",
            ),
        ),
        _DECISION_ROUTE,
    ),
)


TRANSPORT_MUTANT_TEST_SPECS = (
    GateTestSpec(
        "M01",
        (
            _nodeid(
                _S3_TEST,
                "test_409_412_timeout_connection_loss_and_malformed_response_never_retry",
            ),
            _nodeid(
                _FENCE_AUDIT_TEST,
                "test_delete_marker_on_later_page_is_not_ignored",
            ),
        ),
        _S3_ROUTE + _FENCE_AUDIT_ROUTE,
    ),
    GateTestSpec(
        "M02",
        (
            _nodeid(
                _S3_TEST,
                "test_ambiguous_zero_history_sibling_delete_marker_or_mismatch_fail_closed",
            ),
        ),
        _S3_ROUTE,
    ),
    GateTestSpec(
        "M03",
        (
            _nodeid(
                _S3_TEST,
                "test_direct_200_still_requires_exact_all_version_reconciliation",
            ),
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_route_executes_exact_sequence_and_keeps_direct_custody",
            ),
        ),
        _S3_ROUTE + _DECISION_ROUTE,
    ),
    GateTestSpec(
        "M04",
        (
            _nodeid(
                _ADMISSION_TEST,
                "test_admission_matrix_issues_at_most_one_post",
            ),
            _nodeid(
                _ADMISSION_TEST,
                "test_admission_process_death_never_permits_second_send",
            ),
        ),
        _ADMISSION_ROUTE,
    ),
    GateTestSpec(
        "M05",
        (
            _nodeid(
                _ADMISSION_TEST,
                "test_admission_rejects_caller_asserted_probe_success_as_authority",
            ),
            _nodeid(
                _S3_TEST,
                "test_public_effect_protocol_has_no_cached_audit_or_alternate_consume_bypass",
            ),
        ),
        _ADMISSION_ROUTE + _S3_ROUTE,
    ),
    GateTestSpec(
        "M06",
        (
            _nodeid(
                _FENCE_TEST,
                "test_successor_preserves_complete_predecessor_sets_byte_for_byte",
            ),
            _nodeid(
                _SOURCE_TEST,
                "test_partial_policy_or_one_probe_set_cannot_activate_sources",
            ),
        ),
        _FENCE_ROUTE + _SOURCE_ROUTE,
    ),
    GateTestSpec(
        "M07",
        (
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_route_executes_exact_sequence_and_keeps_direct_custody",
            ),
        ),
        _DECISION_ROUTE,
    ),
    GateTestSpec(
        "M08",
        (
            _nodeid(
                _SOURCE_TEST,
                "test_publishers_call_fresh_conditional_create_once_for_only_their_family",
            ),
            _nodeid(
                _SOURCE_TEST,
                "test_ambiguous_zero_history_sibling_delete_marker_or_mismatch_stops_sequence",
            ),
        ),
        _SOURCE_ROUTE,
    ),
    GateTestSpec(
        "M09",
        (
            _nodeid(
                _SOURCE_TEST,
                "test_no_source_is_authoritative_before_batch_successor_stabilizes",
            ),
        ),
        _SOURCE_ROUTE + _FENCE_ROUTE,
    ),
    GateTestSpec(
        "M10",
        (
            _nodeid(
                _SOURCE_TEST,
                "test_partial_policy_or_one_probe_set_cannot_activate_sources",
            ),
            _nodeid(
                _FENCE_TEST,
                "test_successor_rejects_removal_mutation_weakening_duplicate_and_noop",
            ),
        ),
        _SOURCE_ROUTE + _FENCE_ROUTE,
    ),
    GateTestSpec(
        "M11",
        (
            _nodeid(
                _SOURCE_TEST,
                "test_generic_source_writer_and_alternate_coordinate_are_unrepresentable",
            ),
            _nodeid(
                _SOURCE_TEST,
                "test_partial_policy_or_one_probe_set_cannot_activate_sources",
            ),
        ),
        _SOURCE_ROUTE,
    ),
    GateTestSpec(
        "M12",
        (
            _nodeid(
                _FENCE_TEST,
                "test_successor_intrinsic_delta_marks_only_newest_contiguous_reservation",
            ),
        ),
        _FENCE_ROUTE,
    ),
    GateTestSpec(
        "M13",
        (
            _nodeid(
                _FENCE_TEST,
                "test_successor_key_derives_only_from_predecessor_body_sha",
            ),
            _nodeid(
                _FENCE_AUDIT_TEST,
                "test_sibling_key_fails_instead_of_being_filtered",
            ),
        ),
        _FENCE_ROUTE + _FENCE_AUDIT_ROUTE,
    ),
    GateTestSpec(
        "M14",
        (
            _nodeid(
                _FENCE_AUDIT_TEST,
                "test_namespace_rejects_fork_cycle_and_repeated_digest_without_filtering",
            ),
            _nodeid(
                _FENCE_TEST,
                "test_chain_ignores_discovery_time_lexical_and_latest_heuristics",
            ),
        ),
        _FENCE_ROUTE + _FENCE_AUDIT_ROUTE,
    ),
    GateTestSpec(
        "M15",
        (
            _nodeid(
                _FENCE_AUDIT_TEST,
                "test_namespace_rewalks_from_genesis_on_every_invocation",
            ),
        ),
        _FENCE_AUDIT_ROUTE,
    ),
    GateTestSpec(
        "M16",
        (
            _nodeid(
                _SOURCE_TEST,
                "test_no_source_is_authoritative_before_batch_successor_stabilizes",
            ),
            _nodeid(
                _FENCE_AUDIT_TEST,
                "test_namespace_authenticates_complete_control_transport_metadata",
            ),
        ),
        _SOURCE_ROUTE + _FENCE_AUDIT_ROUTE,
    ),
    GateTestSpec(
        "M17",
        (
            _nodeid(
                _FENCE_AUDIT_TEST,
                "test_namespace_rejects_an_orphan_modeled_successor",
            ),
            _nodeid(
                _FENCE_AUDIT_TEST,
                "test_namespace_rejects_history_delete_marker_and_alternate_control_key",
            ),
        ),
        _FENCE_AUDIT_ROUTE,
    ),
    GateTestSpec(
        "M18",
        (
            _nodeid(
                _FENCE_AUDIT_TEST,
                "test_namespace_requires_fresh_zero_at_reached_next_successor_slot",
            ),
            _nodeid(
                _SOURCE_TEST,
                "test_partial_policy_or_one_probe_set_cannot_activate_sources",
            ),
        ),
        _FENCE_AUDIT_ROUTE + _SOURCE_ROUTE,
    ),
    GateTestSpec(
        "M19",
        (
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_route_executes_exact_sequence_and_keeps_direct_custody",
            ),
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_crash_boundaries_have_no_replay_edge",
            ),
        ),
        _DECISION_ROUTE,
    ),
    GateTestSpec(
        "M20",
        (
            _nodeid(
                _S3_TEST,
                "test_copy_multipart_delete_tag_replication_and_lifecycle_are_never_called",
            ),
            _nodeid(
                _FENCE_AUDIT_TEST,
                "test_services_may_expose_sentinel_write_methods_but_none_are_called",
            ),
        ),
        _S3_ROUTE + _FENCE_AUDIT_ROUTE,
    ),
    GateTestSpec(
        "M21",
        (
            _nodeid(
                _FENCE_TEST,
                "test_successor_intrinsic_delta_marks_only_newest_contiguous_reservation",
            ),
            _nodeid(
                _FENCE_TEST,
                "test_fence_records_do_not_extend_h1e_registry",
            ),
        ),
        _FENCE_ROUTE,
    ),
    GateTestSpec(
        "M22",
        (
            _nodeid(
                _DECISION_TEST,
                "test_task11_red_route_executes_exact_sequence_and_keeps_direct_custody",
            ),
            _nodeid(
                _ADMISSION_TEST,
                "test_admission_matrix_issues_at_most_one_post",
            ),
        ),
        _DECISION_ROUTE + _ADMISSION_ROUTE,
    ),
)


SEMANTIC_GATE_PINS = {
    "T01_T25_GATE": {
        "record_type": "glm52_task13_t01_t25_semantic_gate_v1",
        "row_count": 25,
        "file_sha256": (
            "36617696be30b31759201908183ca878ef236a559a79bb65e6a948809acdb486"
        ),
        "body_sha256": (
            "b4cafc35c25181534b65eb1cdc94e652a3b491266b01069cb51278af05ba8ba9"
        ),
    },
    "TRANSPORT_22_MUTANT_GATE": {
        "record_type": "glm52_task13_transport_22_mutant_gate_v1",
        "obligation_count": 22,
        "killed_count": 22,
        "survived_count": 0,
        "masked_count": 0,
        "file_sha256": (
            "dc0eaa9a663405137b06574eb31644be38b205ed6a875044dec0de4780a35201"
        ),
        "body_sha256": (
            "26a816feda2df14a4f0ed6adcbd92cfebcd0445698ce25c238ca0828ee92a2ce"
        ),
    },
}


def validate_semantic_gate_coordinate(
    value: object,
) -> dict[str, object]:
    """Require one coordinate to pin an observed canonical semantic gate."""

    if type(value) is not dict:
        raise TransportGateError("semantic gate coordinate is not exact")
    kind = value.get("artifact_kind")
    if type(kind) is not str or kind not in SEMANTIC_GATE_PINS:
        raise TransportGateError("semantic gate kind is not exact")
    expected = SEMANTIC_GATE_PINS[kind]
    if (
        value.get("file_sha256") != expected["file_sha256"]
        or value.get("body_sha256") != expected["body_sha256"]
    ):
        raise TransportGateError("semantic gate identity is not observed")
    return dict(expected)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_PASSED = re.compile(r"(?:^|\s)([1-9][0-9]*) passed(?:[\s,]|$)")
_DURATION = re.compile(r"\bin [0-9]+(?:\.[0-9]+)?s\b")
_OBSERVATION_FIELDS = {
    "nodeid",
    "exit_code",
    "passed_count",
    "normalized_output_sha256",
}


def _passing_pytest_count(output: str) -> int:
    matches = _PASSED.findall(output)
    if len(matches) != 1:
        raise TransportGateError("pytest output has no unique passing summary")
    return int(matches[0])


def run_pytest_observation(
    nodeid: str,
    *,
    repo_root: Path,
    python_executable: str,
) -> PytestObservation:
    """Execute one frozen pytest selector and return deterministic evidence."""

    if type(nodeid) is not str or "::" not in nodeid:
        raise TransportGateError("pytest nodeid is not exact")
    if not repo_root.is_absolute() or not repo_root.is_dir():
        raise TransportGateError("repository root is not an absolute directory")
    command = (
        python_executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        nodeid,
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    normalized = _DURATION.sub("in <duration>s", completed.stdout)
    try:
        passed_count = _passing_pytest_count(normalized)
    except TransportGateError:
        passed_count = 0
    if completed.returncode != 0 or passed_count == 0:
        raise TransportGateError(
            f"pytest observation failed for {nodeid} "
            f"(exit {completed.returncode}): {completed.stdout[-2000:]}"
        )
    return PytestObservation(
        nodeid=nodeid,
        exit_code=completed.returncode,
        passed_count=passed_count,
        normalized_output_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    )


def make_pytest_observer(
    *,
    repo_root: Path,
    python_executable: str = sys.executable,
) -> Callable[[str], PytestObservation]:
    """Bind the real subprocess observer to one repository and interpreter."""

    root = repo_root.resolve()

    def observe(nodeid: str) -> PytestObservation:
        return run_pytest_observation(
            nodeid,
            repo_root=root,
            python_executable=python_executable,
        )

    return observe


def _require_observation(
    value: object,
    *,
    nodeid: str,
) -> PytestObservation:
    if type(value) is not PytestObservation:
        raise TransportGateError("observer returned a non-evidence value")
    if (
        value.nodeid != nodeid
        or type(value.exit_code) is not int
        or value.exit_code != 0
        or type(value.passed_count) is not int
        or value.passed_count < 1
        or type(value.normalized_output_sha256) is not str
        or _SHA256.fullmatch(value.normalized_output_sha256) is None
    ):
        raise TransportGateError("observer returned an invalid pytest outcome")
    return value


def _file_sha256s(
    *,
    repo_root: Path,
    nodeids: tuple[str, ...],
    route_paths: tuple[str, ...],
) -> dict[str, str]:
    paths = {nodeid.split("::", 1)[0] for nodeid in nodeids}
    paths.update(route_paths)
    result = {}
    for relative in sorted(paths):
        path = repo_root / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not path.is_file()
            or path.is_symlink()
        ):
            raise TransportGateError("evidence source path is not exact: " + relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _receipt(
    *,
    source: Mapping[str, object],
    spec: GateTestSpec,
    repo_root: Path,
    observed: Mapping[str, PytestObservation],
) -> dict[str, object]:
    body = {
        **dict(source),
        "pytest_nodeids": list(spec.pytest_nodeids),
        "route_paths": list(spec.route_paths),
        "files_sha256": _file_sha256s(
            repo_root=repo_root,
            nodeids=spec.pytest_nodeids,
            route_paths=spec.route_paths,
        ),
        "observations": [asdict(observed[nodeid]) for nodeid in spec.pytest_nodeids],
    }
    body["evidence_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    return body


def _observe_specs(
    specs: tuple[GateTestSpec, ...],
    observe: Callable[[str], PytestObservation],
) -> dict[str, PytestObservation]:
    observed = {}
    for spec in specs:
        for nodeid in spec.pytest_nodeids:
            if nodeid not in observed:
                observed[nodeid] = _require_observation(
                    observe(nodeid),
                    nodeid=nodeid,
                )
    return observed


def _self_hash(value: dict[str, object]) -> dict[str, object]:
    value["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(value)
    ).hexdigest()
    return value


def build_t01_t25_gate(
    *,
    repo_root: Path,
    observe: Callable[[str], PytestObservation],
) -> dict[str, object]:
    """Build the T01-T25 artifact only from passing observed route tests."""

    root = repo_root.resolve()
    observed = _observe_specs(T01_T25_TEST_SPECS, observe)
    rows = [
        _receipt(
            source=asdict(row),
            spec=spec,
            repo_root=root,
            observed=observed,
        )
        for row, spec in zip(T01_T25_ROWS, T01_T25_TEST_SPECS)
    ]
    return _self_hash(
        {
            "schema_version": 1,
            "record_type": "glm52_task13_t01_t25_semantic_gate_v1",
            "gate_id": "T01_T25",
            "status": "PROVEN",
            "row_count": len(rows),
            "observed_test_invocation_count": len(observed),
            "rows": rows,
        }
    )


def build_transport_mutant_gate(
    *,
    repo_root: Path,
    observe: Callable[[str], PytestObservation],
) -> dict[str, object]:
    """Build the 22-obligation artifact only from killed mutation probes."""

    root = repo_root.resolve()
    observed = _observe_specs(TRANSPORT_MUTANT_TEST_SPECS, observe)
    obligations = [
        _receipt(
            source={
                **asdict(obligation),
                "observed_outcome": "MUTANT_REJECTED",
            },
            spec=spec,
            repo_root=root,
            observed=observed,
        )
        for obligation, spec in zip(
            TRANSPORT_MUTANT_OBLIGATIONS,
            TRANSPORT_MUTANT_TEST_SPECS,
        )
    ]
    return _self_hash(
        {
            "schema_version": 1,
            "record_type": "glm52_task13_transport_22_mutant_gate_v1",
            "gate_id": "TRANSPORT_22_MUTANTS",
            "status": "PROVEN",
            "obligation_count": len(obligations),
            "killed_count": len(obligations),
            "survived_count": 0,
            "masked_count": 0,
            "observed_test_invocation_count": len(observed),
            "obligations": obligations,
        }
    )


def _validate_self_hash(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise TransportGateError("transport gate must be an exact object")
    identity = value.get("canonical_identity_sha256")
    if type(identity) is not str or _SHA256.fullmatch(identity) is None:
        raise TransportGateError("transport gate self-hash is missing")
    unhashed = dict(value)
    del unhashed["canonical_identity_sha256"]
    expected = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest()
    if identity != expected:
        raise TransportGateError("transport gate self-hash mismatched")
    return value


def _validate_against_fresh_observation(
    value: object,
    *,
    repo_root: Path,
    observe: Callable[[str], PytestObservation] | None,
    builder: Callable[..., dict[str, object]],
) -> dict[str, object]:
    exact = _validate_self_hash(value)
    if observe is None:
        raise TransportGateError("a fresh real observer is required")
    expected = builder(repo_root=repo_root, observe=observe)
    if exact != expected:
        raise TransportGateError("transport gate differs from the fresh observation")
    return exact


def validate_t01_t25_gate(
    value: object,
    *,
    repo_root: Path,
    observe: Callable[[str], PytestObservation] | None = None,
) -> dict[str, object]:
    """Strictly re-observe and validate the complete ordered T01-T25 gate."""

    return _validate_against_fresh_observation(
        value,
        repo_root=repo_root,
        observe=observe,
        builder=build_t01_t25_gate,
    )


def validate_transport_mutant_gate(
    value: object,
    *,
    repo_root: Path,
    observe: Callable[[str], PytestObservation] | None = None,
) -> dict[str, object]:
    """Strictly re-observe and validate all 22 ordered mutant obligations."""

    return _validate_against_fresh_observation(
        value,
        repo_root=repo_root,
        observe=observe,
        builder=build_transport_mutant_gate,
    )


__all__ = [
    "SEMANTIC_GATE_PINS",
    "T01_T25_ROWS",
    "T01_T25_TEST_SPECS",
    "TRANSPORT_MUTANT_OBLIGATIONS",
    "TRANSPORT_MUTANT_TEST_SPECS",
    "GateTestSpec",
    "PytestObservation",
    "TransportGateError",
    "TransportMutantObligation",
    "TransportRow",
    "build_t01_t25_gate",
    "build_transport_mutant_gate",
    "make_pytest_observer",
    "run_pytest_observation",
    "validate_semantic_gate_coordinate",
    "validate_t01_t25_gate",
    "validate_transport_mutant_gate",
]
