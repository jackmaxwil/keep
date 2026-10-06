from __future__ import annotations

import base64
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import inspect
import json

import pytest


OWNER = "246813579024"
REGION = "us-west-2"
RUN_ID = "glm52-sky-20260724"
ACTIVATION_ID = "activation-0001"
BUCKET = "keep-glm52-production"
NOW = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
GENESIS_KEY = f"campaigns/{RUN_ID}/authorities/fence/FENCE_GENESIS.json"
CANDIDATE_KEY = (
    f"campaigns/{RUN_ID}/submissions/production/generations/"
    "00000001/terminal/PRODUCTION_TERMINAL_V2.json"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _self_hashed(body: dict[str, object], field: str) -> bytes:
    return _canonical(
        {**body, field: hashlib.sha256(_canonical(body)).hexdigest()}
    ) + b"\n"


def _control_metadata(raw: bytes) -> tuple[tuple[str, str], ...]:
    record = json.loads(raw)
    return tuple(
        sorted(
            (
                ("glm52-body-sha256", record["fence_body_sha256"]),
                (
                    "glm52-campaign-identity-sha256",
                    record["campaign_identity_sha256"],
                ),
                ("glm52-file-sha256", hashlib.sha256(raw).hexdigest()),
                ("glm52-record-type", record["record_type"]),
                ("glm52-run-id", RUN_ID),
            )
        )
    )


def _live_state(*, bucket_policy_sha256: str) -> dict[str, object]:
    absent = hashlib.sha256(b'{"state":"absent"}\n').hexdigest()
    return {
        "account_id": OWNER,
        "region": REGION,
        "bucket": BUCKET,
        "bucket_versioning_status": "Enabled",
        "bucket_mfa_delete_status": "Disabled",
        "bucket_policy_sha256": bucket_policy_sha256,
        "lifecycle_configuration_state": "absent",
        "lifecycle_configuration_sha256": absent,
        "replication_configuration_state": "absent",
        "replication_configuration_sha256": absent,
        "cloudformation_stack_id": "glm52-production-fence",
        "cloudformation_template_sha256": "1" * 64,
        "cloudformation_parameters_sha256": "2" * 64,
        "executor_identity_sha256": "3" * 64,
        "publisher_deny_policy_sha256": "4" * 64,
        "h1d_control_plane_ready_body_sha256": "5" * 64,
    }


def _genesis_raw() -> bytes:
    return _self_hashed(
        {
            "account_id": OWNER,
            "bucket": BUCKET,
            "campaign_identity_sha256": "c" * 64,
            "created_at": "2026-07-28T11:59:00Z",
            "enrolled_sources": [],
            "genesis_key": GENESIS_KEY,
            "live_state": _live_state(bucket_policy_sha256="d" * 64),
            "record_type": "glm52_sky_production_fence_genesis_v1",
            "region": REGION,
            "reserved_generations": [],
            "run_id": RUN_ID,
            "schema_version": 1,
        },
        "fence_body_sha256",
    )


def _successor_raw(genesis_raw: bytes) -> tuple[str, bytes]:
    genesis = json.loads(genesis_raw)
    key = (
        f"campaigns/{RUN_ID}/authorities/fence/successors/"
        f"{genesis['fence_body_sha256']}/FENCE_SUCCESSOR.json"
    )
    raw = _self_hashed(
        {
            "account_id": OWNER,
            "added_generation_reservations": [{"generation": 1}],
            "added_sources": [],
            "bucket": BUCKET,
            "campaign_identity_sha256": "c" * 64,
            "enrolled_sources": [],
            "live_state": _live_state(bucket_policy_sha256="e" * 64),
            "predecessor_control_body_sha256": genesis[
                "fence_body_sha256"
            ],
            "predecessor_control_file_sha256": hashlib.sha256(
                genesis_raw
            ).hexdigest(),
            "predecessor_control_key": GENESIS_KEY,
            "predecessor_control_version_id": "genesis-version",
            "record_type": "glm52_sky_production_fence_successor_v1",
            "region": REGION,
            "reserved_generations": [{"generation": 1}],
            "run_id": RUN_ID,
            "schema_version": 1,
            "selected_at": "2026-07-28T11:59:30Z",
            "successor_key": key,
        },
        "fence_body_sha256",
    )
    return key, raw


class Body:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.close_count = 0

    def read(self) -> bytes:
        return self.raw

    def close(self) -> None:
        self.close_count += 1


class S3:
    def __init__(self) -> None:
        genesis = _genesis_raw()
        successor_key, successor = _successor_raw(genesis)
        self.objects = {
            (GENESIS_KEY, "genesis-version"): genesis,
            (successor_key, "successor-version"): successor,
        }
        self.metadata = {
            (GENESIS_KEY, "genesis-version"): _control_metadata(genesis),
            (successor_key, "successor-version"): _control_metadata(successor),
        }
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.bodies: list[Body] = []
        self.page_size = 1
        self.cycle_markers = False

    def _record(self, name: str, values: dict[str, object]) -> None:
        self.calls.append((name, dict(values)))

    def list_object_versions(self, **kwargs: object) -> dict[str, object]:
        values = dict(kwargs)
        self._record("list_object_versions", values)
        assert values["ExpectedBucketOwner"] == OWNER
        prefix = str(values["Prefix"])
        rows = [
            (key, version, raw)
            for (key, version), raw in self.objects.items()
            if key.startswith(prefix)
        ]
        offset = int(str(values.get("VersionIdMarker", "page-0")).split("-")[-1])
        page = rows[offset : offset + self.page_size]
        next_offset = offset + len(page)
        truncated = next_offset < len(rows)
        response: dict[str, object] = {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": f"list-{len(self.calls)}",
            },
            "Name": BUCKET,
            "Prefix": prefix,
            "KeyMarker": values.get("KeyMarker", ""),
            "VersionIdMarker": values.get("VersionIdMarker", ""),
            "IsTruncated": truncated,
            "Versions": [
                {
                    "Key": key,
                    "VersionId": version,
                    "IsLatest": True,
                    "ETag": f'"{hashlib.md5(raw).hexdigest()}"',
                    "Size": len(raw),
                    "LastModified": NOW,
                }
                for key, version, raw in page
            ],
            "DeleteMarkers": [],
            "CommonPrefixes": [],
        }
        if truncated:
            response["NextKeyMarker"] = f"{prefix}page-{next_offset}"
            response["NextVersionIdMarker"] = f"page-{next_offset}"
        if self.cycle_markers and "KeyMarker" in values:
            response["IsTruncated"] = True
            response["NextKeyMarker"] = values["KeyMarker"]
            response["NextVersionIdMarker"] = values["VersionIdMarker"]
        return response

    def _transport(self, key: str, version: str) -> dict[str, object]:
        raw = self.objects[(key, version)]
        return {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "RequestId": f"read-{len(self.calls)}",
            },
            "VersionId": version,
            "ETag": f'"{hashlib.md5(raw).hexdigest()}"',
            "ContentLength": len(raw),
            "LastModified": NOW,
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(raw).digest()
            ).decode("ascii"),
            "ChecksumType": "FULL_OBJECT",
            "ContentType": "application/json",
            "Metadata": dict(self.metadata[(key, version)]),
            "MissingMeta": 0,
        }

    def get_object(self, **kwargs: object) -> dict[str, object]:
        values = dict(kwargs)
        self._record("get_object", values)
        key, version = str(values["Key"]), str(values["VersionId"])
        body = Body(self.objects[(key, version)])
        self.bodies.append(body)
        return {**self._transport(key, version), "Body": body}

    def head_object(self, **kwargs: object) -> dict[str, object]:
        values = dict(kwargs)
        self._record("head_object", values)
        return self._transport(str(values["Key"]), str(values["VersionId"]))


def _candidate():
    from glm52_enforcement.s3_records import build_immutable_json_candidate

    body = {
        "account_id": OWNER,
        "activation_id": ACTIVATION_ID,
        "generation": 1,
        "generation_text": "00000001",
        "record_type": "glm52_production_terminal_v2",
        "region": REGION,
        "run_id": RUN_ID,
        "schema_version": 2,
    }
    return build_immutable_json_candidate(
        record_kind="terminal-v2",
        bucket=BUCKET,
        key=CANDIDATE_KEY,
        raw=_self_hashed(body, "canonical_body_sha256"),
        activation_id=ACTIVATION_ID,
        generation=1,
    )


def _authority_snapshot():
    from glm52_enforcement.h1f_adapter import H1fAuthoritySnapshot

    genesis = _genesis_raw()
    successor_key, successor = _successor_raw(genesis)
    successor_record = json.loads(successor)
    return H1fAuthoritySnapshot(
        authority_domain="ACTIVATION",
        bucket=BUCKET,
        run_id=RUN_ID,
        activation_id=ACTIVATION_ID,
        generation=1,
        epoch=1,
        execution_arn="arn:aws:states:us-west-2:246813579024:execution:x:y",
        barrier_nonce_sha256="b" * 64,
        closing_revision=7,
        active_head_key=successor_key,
        active_head_version_id="successor-version",
        active_head_file_sha256=hashlib.sha256(successor).hexdigest(),
        active_head_body_sha256=successor_record["fence_body_sha256"],
    )


def test_namespace_walk_pages_versions_and_delete_markers_to_zero_child() -> None:
    from glm52_enforcement.h1f_adapter import (
        FreshH1fAuditService,
        H1fAuditRequest,
    )

    s3 = S3()
    result = FreshH1fAuditService(
        authority_reader=_authority_snapshot
    ).fresh_audit(
        s3=s3,
        request=H1fAuditRequest(
            operation_kind="S3_CREATE",
            action_key=(
                "ACTIVATION#activation-0001#ACTION#00000001#"
                "S3_CREATE#00000001"
            ),
            candidate=_candidate(),
        ),
    )
    assert result.closing_revision == 7
    assert result.expected_authorized_revision == 8
    assert result.active_head_key == _authority_snapshot().active_head_key
    assert [name for name, _ in s3.calls].count("list_object_versions") == 4
    assert all(body.close_count == 1 for body in s3.bodies)


def test_every_irreversible_operation_constructs_a_new_h1f_walk() -> None:
    from glm52_enforcement.h1f_adapter import (
        FreshH1fAuditService,
        H1fAuditError,
        H1fAuditRequest,
    )

    s3 = S3()
    reads = 0

    def current():
        nonlocal reads
        reads += 1
        return _authority_snapshot()

    service = FreshH1fAuditService(authority_reader=current)
    request = H1fAuditRequest(
        operation_kind="S3_CREATE",
        action_key="ACTION#one",
        candidate=_candidate(),
    )
    service.fresh_audit(s3=s3, request=request)
    list_count = [name for name, _ in s3.calls].count("list_object_versions")
    del s3.objects[(GENESIS_KEY, "genesis-version")]
    with pytest.raises(H1fAuditError):
        service.fresh_audit(s3=s3, request=request)
    assert reads == 2
    assert [name for name, _ in s3.calls].count(
        "list_object_versions"
    ) > list_count


def test_supplied_cached_or_cross_domain_audit_is_unrepresentable() -> None:
    from glm52_enforcement.h1f_adapter import (
        FreshH1fAuditService,
        H1fAuditRequest,
    )

    assert tuple(inspect.signature(FreshH1fAuditService.fresh_audit).parameters) == (
        "self",
        "s3",
        "request",
    )
    assert tuple(H1fAuditRequest.__dataclass_fields__) == (
        "operation_kind",
        "action_key",
        "candidate",
    )
    service = FreshH1fAuditService(authority_reader=_authority_snapshot)
    with pytest.raises(TypeError):
        service.fresh_audit(  # type: ignore[call-arg]
            s3=S3(),
            request=H1fAuditRequest(
                operation_kind="S3_CREATE",
                action_key="ACTION#one",
                candidate=_candidate(),
            ),
            modeled_head=object(),
        )


@pytest.mark.parametrize(
    "case",
    [
        "marker-cycle",
        "fork",
        "orphan",
        "history",
        "stale-head",
        "current-mismatch",
    ],
)
def test_namespace_walk_rejects_marker_cycle_fork_orphan_history_and_stale_head(
    case: str,
) -> None:
    from glm52_enforcement.h1f_adapter import (
        FreshH1fAuditService,
        H1fAuditError,
        H1fAuditRequest,
    )

    s3 = S3()
    snapshot = _authority_snapshot()
    genesis = s3.objects[(GENESIS_KEY, "genesis-version")]
    successor_key, successor = _successor_raw(genesis)
    if case == "marker-cycle":
        s3.cycle_markers = True
    elif case in {"fork", "history"}:
        s3.objects[(successor_key, "other-version")] = successor
        s3.metadata[(successor_key, "other-version")] = _control_metadata(
            successor
        )
    elif case == "orphan":
        del s3.objects[(GENESIS_KEY, "genesis-version")]
    elif case == "stale-head":
        snapshot = replace(snapshot, active_head_body_sha256="f" * 64)
    else:
        record = json.loads(successor)
        record["enrolled_sources"] = [
            {"key": "foreign", "source_kind": "gpu-spend-snapshot"}
        ]
        record.pop("fence_body_sha256")
        successor = _self_hashed(record, "fence_body_sha256")
        s3.objects[(successor_key, "successor-version")] = successor
        s3.metadata[(successor_key, "successor-version")] = (
            _control_metadata(successor)
        )
        snapshot = replace(
            snapshot,
            active_head_file_sha256=hashlib.sha256(successor).hexdigest(),
            active_head_body_sha256=json.loads(successor)[
                "fence_body_sha256"
            ],
        )
    service = FreshH1fAuditService(authority_reader=lambda: snapshot)
    with pytest.raises(H1fAuditError):
        service.fresh_audit(
            s3=s3,
            request=H1fAuditRequest(
                operation_kind="S3_CREATE",
                action_key="ACTION#one",
                candidate=_candidate(),
            ),
        )


@pytest.mark.parametrize(
    "case",
    [
        "foreign-account",
        "foreign-bucket",
        "suspended-versioning",
        "lifecycle-present",
        "replication-present",
        "extra-field",
        "missing-field",
        "publisher-change-without-source",
        "cloudformation-change-without-readiness",
    ],
)
def test_rehashed_live_state_semantic_mutants_fail_before_coordinate_audit(
    case: str,
) -> None:
    from glm52_enforcement.h1f_adapter import (
        FreshH1fAuditService,
        H1fAuditError,
        H1fAuditRequest,
    )

    s3 = S3()
    snapshot = _authority_snapshot()
    successor_key = snapshot.active_head_key
    record = json.loads(
        s3.objects[(successor_key, "successor-version")]
    )
    live = record["live_state"]
    assert isinstance(live, dict)
    if case == "foreign-account":
        live["account_id"] = "000000000000"
    elif case == "foreign-bucket":
        live["bucket"] = "foreign-bucket"
    elif case == "suspended-versioning":
        live["bucket_versioning_status"] = "Suspended"
    elif case == "lifecycle-present":
        live["lifecycle_configuration_state"] = "present"
        live["lifecycle_configuration_sha256"] = hashlib.sha256(
            b'{"state":"present"}\n'
        ).hexdigest()
    elif case == "replication-present":
        live["replication_configuration_state"] = "present"
        live["replication_configuration_sha256"] = hashlib.sha256(
            b'{"state":"present"}\n'
        ).hexdigest()
    elif case == "extra-field":
        live["foreign"] = "member"
    elif case == "missing-field":
        del live["executor_identity_sha256"]
    elif case == "publisher-change-without-source":
        live["publisher_deny_policy_sha256"] = "6" * 64
    elif case == "cloudformation-change-without-readiness":
        live["cloudformation_template_sha256"] = "7" * 64
    else:
        raise AssertionError("unhandled case")
    record.pop("fence_body_sha256")
    raw = _self_hashed(record, "fence_body_sha256")
    s3.objects[(successor_key, "successor-version")] = raw
    s3.metadata[(successor_key, "successor-version")] = _control_metadata(raw)
    snapshot = replace(
        snapshot,
        active_head_file_sha256=hashlib.sha256(raw).hexdigest(),
        active_head_body_sha256=json.loads(raw)["fence_body_sha256"],
    )

    with pytest.raises(H1fAuditError):
        FreshH1fAuditService(authority_reader=lambda: snapshot).fresh_audit(
            s3=s3,
            request=H1fAuditRequest(
                operation_kind="S3_CREATE",
                action_key="ACTION#one",
                candidate=_candidate(),
            ),
        )
    assert not [
        call
        for call in s3.calls
        if call[0] == "list_object_versions"
        and call[1]["Prefix"] == CANDIDATE_KEY
    ]


def test_live_state_policy_identity_cannot_roll_back_to_any_ancestor() -> None:
    from glm52_enforcement import s3_keys
    from glm52_enforcement.h1f_adapter import (
        FreshH1fAuditService,
        H1fAuditError,
        H1fAuditRequest,
    )

    def successor(
        *,
        predecessor_key: str,
        predecessor_version: str,
        predecessor_raw: bytes,
        generation: int,
        policy_sha256: str,
    ) -> tuple[str, bytes]:
        predecessor = json.loads(predecessor_raw)
        key = s3_keys.fence_successor_s3_key(
            run_id=RUN_ID,
            predecessor_body_sha256=predecessor["fence_body_sha256"],
        )
        raw = _self_hashed(
            {
                "account_id": OWNER,
                "added_generation_reservations": [
                    {"generation": generation}
                ],
                "added_sources": [],
                "bucket": BUCKET,
                "campaign_identity_sha256": "c" * 64,
                "enrolled_sources": [],
                "live_state": _live_state(
                    bucket_policy_sha256=policy_sha256
                ),
                "predecessor_control_body_sha256": predecessor[
                    "fence_body_sha256"
                ],
                "predecessor_control_file_sha256": hashlib.sha256(
                    predecessor_raw
                ).hexdigest(),
                "predecessor_control_key": predecessor_key,
                "predecessor_control_version_id": predecessor_version,
                "record_type": (
                    "glm52_sky_production_fence_successor_v1"
                ),
                "region": REGION,
                "reserved_generations": [
                    {"generation": value}
                    for value in range(1, generation + 1)
                ],
                "run_id": RUN_ID,
                "schema_version": 1,
                "selected_at": "2026-07-28T11:59:45Z",
                "successor_key": key,
            },
            "fence_body_sha256",
        )
        return key, raw

    s3 = S3()
    first_key = _authority_snapshot().active_head_key
    first_raw = s3.objects[(first_key, "successor-version")]
    second_key, second_raw = successor(
        predecessor_key=first_key,
        predecessor_version="successor-version",
        predecessor_raw=first_raw,
        generation=2,
        policy_sha256="8" * 64,
    )
    third_key, third_raw = successor(
        predecessor_key=second_key,
        predecessor_version="second-version",
        predecessor_raw=second_raw,
        generation=3,
        policy_sha256="e" * 64,
    )
    s3.objects[(second_key, "second-version")] = second_raw
    s3.metadata[(second_key, "second-version")] = _control_metadata(second_raw)
    s3.objects[(third_key, "third-version")] = third_raw
    s3.metadata[(third_key, "third-version")] = _control_metadata(third_raw)
    snapshot = replace(
        _authority_snapshot(),
        active_head_key=third_key,
        active_head_version_id="third-version",
        active_head_file_sha256=hashlib.sha256(third_raw).hexdigest(),
        active_head_body_sha256=json.loads(third_raw)["fence_body_sha256"],
    )

    with pytest.raises(H1fAuditError):
        FreshH1fAuditService(authority_reader=lambda: snapshot).fresh_audit(
            s3=s3,
            request=H1fAuditRequest(
                operation_kind="S3_CREATE",
                action_key="ACTION#one",
                candidate=_candidate(),
            ),
        )


def test_exact_coordinate_requires_zero_or_one_sole_current_nondelete_version() -> None:
    from glm52_enforcement.h1f_adapter import (
        H1fAuditError,
        reconcile_exact_candidate,
    )

    s3 = S3()
    candidate = _candidate()
    assert reconcile_exact_candidate(
        s3=s3, candidate=candidate
    ).state == "zero"
    s3.objects[(candidate.key, "candidate-version")] = candidate.raw
    s3.metadata[(candidate.key, "candidate-version")] = candidate.metadata
    result = reconcile_exact_candidate(s3=s3, candidate=candidate)
    assert result.state == "sole-version"
    assert result.object_identity is not None
    assert result.object_identity.version_id == "candidate-version"
    s3.objects[(candidate.key, "candidate-history")] = candidate.raw
    s3.metadata[(candidate.key, "candidate-history")] = candidate.metadata
    with pytest.raises(H1fAuditError):
        reconcile_exact_candidate(s3=s3, candidate=candidate)


def test_exact_get_then_head_requires_complete_identity_agreement_and_closes_body() -> None:
    from glm52_enforcement.h1f_adapter import (
        H1fAuditError,
        reconcile_exact_candidate,
    )

    class DriftS3(S3):
        def head_object(self, **kwargs: object) -> dict[str, object]:
            response = super().head_object(**kwargs)
            response["ContentType"] = "text/plain"
            return response

    s3 = DriftS3()
    candidate = _candidate()
    s3.objects[(candidate.key, "candidate-version")] = candidate.raw
    s3.metadata[(candidate.key, "candidate-version")] = candidate.metadata
    with pytest.raises(H1fAuditError):
        reconcile_exact_candidate(s3=s3, candidate=candidate)
    assert s3.bodies[-1].close_count == 1
    calls = [name for name, _ in s3.calls]
    assert calls[-2:] == ["get_object", "head_object"]


def test_get_object_hostile_mapping_detach_failure_closes_exposed_body_once() -> None:
    from collections.abc import Iterator, Mapping

    from glm52_enforcement.h1f_adapter import (
        H1fAuditError,
        reconcile_exact_candidate,
    )

    class HostileMapping(Mapping[str, object]):
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def __getitem__(self, key: str) -> object:
            raise RuntimeError(f"hostile detach at {key}")

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError("hostile detach")

        def __len__(self) -> int:
            return len(self.payload)

        def get(self, key: str, default: object = None) -> object:
            return self.payload.get(key, default)

    class HostileS3(S3):
        def get_object(self, **kwargs: object) -> object:
            return HostileMapping(super().get_object(**kwargs))

    s3 = HostileS3()
    candidate = _candidate()
    s3.objects[(candidate.key, "candidate-version")] = candidate.raw
    s3.metadata[(candidate.key, "candidate-version")] = candidate.metadata
    with pytest.raises(H1fAuditError):
        reconcile_exact_candidate(s3=s3, candidate=candidate)
    assert s3.bodies[-1].close_count == 1
    assert not [name for name, _ in s3.calls if name == "head_object"]
