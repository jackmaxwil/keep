from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from glm52_enforcement import support_fence_handler as handler
from glm52_enforcement.fence_artifacts import (
    ArtifactCoordinate,
    ExecutorAuthorityClass,
    FenceSlot,
)
from glm52_enforcement.fence_executor import FenceExecutor


BUCKET = "keep-glm52-models-246813579024-us-west-2"
KEY = (
    "campaigns/glm52-sky-20260724/authorities/fence/requests/"
    "act-20260728-0001/00000001/CLOSED_SOURCE.json"
)


def _metadata(request_id: str) -> dict[str, object]:
    return {
        "ResponseMetadata": {
            "HTTPStatusCode": 200,
            "RequestId": request_id,
            "RetryAttempts": 0,
        }
    }


def _coordinate(raw: bytes, *, key: str = KEY) -> ArtifactCoordinate:
    return ArtifactCoordinate(
        bucket=BUCKET,
        key=key,
        version_id="request-version-1",
        file_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_identity_sha256="a" * 64,
    )


class ExactS3:
    def __init__(
        self,
        raw: bytes,
        *,
        versions: list[dict[str, object]] | None = None,
        delete_markers: list[dict[str, object]] | None = None,
        is_truncated: bool = False,
    ) -> None:
        self.raw = raw
        self.versions = versions
        self.delete_markers = delete_markers or []
        self.is_truncated = is_truncated
        self.calls: list[tuple[str, dict[str, object]]] = []

    def list_object_versions(self, **request: object) -> dict[str, object]:
        self.calls.append(("list_object_versions", dict(request)))
        key = request["Prefix"]
        versions = self.versions
        if versions is None:
            versions = [
                {
                    "Key": key,
                    "VersionId": "request-version-1",
                    "IsLatest": True,
                }
            ]
        return {
            **_metadata("history-request"),
            "Versions": versions,
            "DeleteMarkers": self.delete_markers,
            "IsTruncated": self.is_truncated,
        }

    def get_object(self, **request: object) -> dict[str, object]:
        self.calls.append(("get_object", dict(request)))
        return {
            **_metadata("get-request"),
            "VersionId": request["VersionId"],
            "Body": self.raw,
        }


def _executor() -> FenceExecutor:
    return FenceExecutor(client=object(), record_store=object())


def _services(
    *,
    authority_class: ExecutorAuthorityClass = ExecutorAuthorityClass.SUPPORT_RUNTIME,
    live=None,
    s3: object | None = None,
) -> handler.FenceHandlerServices:
    return handler.FenceHandlerServices(
        s3=s3 or object(),
        executor=_executor(),
        authority_class=authority_class,
        live_support_runtime_identity=live,
    )


def test_fixed_key_loader_accepts_one_exact_version_and_no_delete_marker() -> None:
    raw = b'{"record_type":"fixture"}\n'
    coordinate = _coordinate(raw)
    s3 = ExactS3(raw)

    assert handler._load_exact_json(
        s3=s3, coordinate=coordinate, label="fixture"
    ) == {"record_type": "fixture"}
    assert s3.calls == [
        (
            "list_object_versions",
            {
                "Bucket": BUCKET,
                "Prefix": KEY,
                "ExpectedBucketOwner": "246813579024",
            },
        ),
        (
            "get_object",
            {
                "Bucket": BUCKET,
                "Key": KEY,
                "VersionId": "request-version-1",
                "ExpectedBucketOwner": "246813579024",
                "ChecksumMode": "ENABLED",
            },
        ),
    ]


@pytest.mark.parametrize(
    ("versions", "delete_markers"),
    [
        ([], []),
        (
            [
                {
                    "Key": KEY,
                    "VersionId": "request-version-1",
                    "IsLatest": True,
                },
                {
                    "Key": KEY,
                    "VersionId": "older-version",
                    "IsLatest": False,
                },
            ],
            [],
        ),
        (
            [
                {
                    "Key": KEY,
                    "VersionId": "request-version-1",
                    "IsLatest": True,
                }
            ],
            [{"Key": KEY, "VersionId": "deleted"}],
        ),
        (
            [
                {
                    "Key": KEY,
                    "VersionId": "foreign-version",
                    "IsLatest": True,
                }
            ],
            [],
        ),
    ],
)
def test_fixed_key_loader_rejects_absent_ambiguous_deleted_or_drifted_history(
    versions: list[dict[str, object]],
    delete_markers: list[dict[str, object]],
) -> None:
    raw = b'{"record_type":"fixture"}\n'
    with pytest.raises(ValueError, match="FIXED_KEY_HISTORY_DRIFT"):
        handler._load_exact_json(
            s3=ExactS3(
                raw,
                versions=versions,
                delete_markers=delete_markers,
            ),
            coordinate=_coordinate(raw),
            label="fixture",
        )


def test_handler_event_rejects_embedded_records_static_arn_and_v1_shape() -> None:
    coordinate = _coordinate(b"{}\n").to_dict()
    for event in (
        {"request_coordinate": coordinate, "manifest": {}},
        {"request_coordinate": coordinate, "change_set_arn": "arn:static"},
        {"request_coordinate": coordinate, "prepared": {}},
        {
            "schema_version": 1,
            "record_type": "glm52_fence_transition_request_v1",
            "request_coordinate": coordinate,
        },
    ):
        with pytest.raises(ValueError, match="only request_coordinate"):
            handler.main(event, object(), services=_services())


def test_handler_rejects_v1_record_loaded_from_an_exact_coordinate() -> None:
    body = {
        "schema_version": 1,
        "record_type": "glm52_fence_transition_request_v1",
    }
    raw = json.dumps(body, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    coordinate = _coordinate(raw)
    with pytest.raises(ValueError, match="field|v2|schema"):
        handler.main(
            {"request_coordinate": coordinate.to_dict()},
            object(),
            services=_services(s3=ExactS3(raw)),
        )


class FakeRequest:
    def __init__(self, slot: FenceSlot) -> None:
        self.slot = slot
        self.canonical_identity_sha256 = "1" * 64

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_coordinate": _coordinate(b"{}\n").to_dict(),
            "manifest_stage": "BOOTSTRAP",
            "selected_entry_identity_sha256": "2" * 64,
            "support_runtime_identity_coordinate": None,
        }


class FakeManifest:
    manifest_stage = SimpleNamespace(value="BOOTSTRAP")
    canonical_identity_sha256 = "a" * 64

    def to_dict(self) -> dict[str, object]:
        return {
            "executor_inventory": [
                {
                    "authority_class": "SUPPORT_RUNTIME",
                    "expected_contract": {"fixture": "expected"},
                }
            ]
        }


class FakePrepared:
    def __init__(
        self,
        arn: str,
        *,
        request_identity_sha256: str = "1" * 64,
        manifest_identity_sha256: str = "a" * 64,
        entry_identity_sha256: str = "2" * 64,
        slot: FenceSlot = FenceSlot.CLOSED_SOURCE,
    ) -> None:
        self.change_set_arn = arn
        self.canonical_identity_sha256 = "3" * 64
        self.request_identity_sha256 = request_identity_sha256
        self.manifest_identity_sha256 = manifest_identity_sha256
        self.entry_identity_sha256 = entry_identity_sha256
        self.slot = slot

    def to_dict(self) -> dict[str, object]:
        return {
            "record_type": "glm52_prepared_fence_change_set_v2",
            "change_set_arn": self.change_set_arn,
            "request_identity_sha256": self.request_identity_sha256,
            "manifest_identity_sha256": self.manifest_identity_sha256,
            "entry_identity_sha256": self.entry_identity_sha256,
            "slot": self.slot.value,
            "canonical_identity_sha256": self.canonical_identity_sha256,
        }


def _patch_loaded_request(
    monkeypatch,
    request: FakeRequest,
    *,
    allowed_create_authority_classes: tuple[str, ...] = (),
) -> None:
    manifest = FakeManifest()
    entry = SimpleNamespace(
        slot=request.slot,
        entry_identity_sha256="2" * 64,
        allowed_create_authority_classes=allowed_create_authority_classes,
    )
    monkeypatch.setattr(handler, "_load_request", lambda **_: request)
    monkeypatch.setattr(
        handler, "load_pinned_fence_manifest", lambda **_: manifest
    )
    monkeypatch.setattr(handler, "_select_entry", lambda *_: entry)
    monkeypatch.setattr(
        handler, "load_pinned_fence_entry_template", lambda **_: entry
    )
    monkeypatch.setattr(
        handler, "_authenticate_runtime_identity", lambda **_: None
    )


def test_handler_rejects_create_authority_class_mismatch_before_create(
    monkeypatch,
) -> None:
    request = FakeRequest(FenceSlot.CLOSED_SOURCE)
    _patch_loaded_request(monkeypatch, request)
    called = {"prepare": False}
    monkeypatch.setattr(
        FenceExecutor,
        "load_create_authority",
        lambda *_: SimpleNamespace(
            authority_class=ExecutorAuthorityClass.RETAINED_PRE_SUPPORT
        ),
    )

    def prepare(*args, **kwargs):
        called["prepare"] = True
        raise AssertionError("must not create")

    monkeypatch.setattr(FenceExecutor, "prepare_change_set", prepare)
    with pytest.raises(ValueError, match="create authority class"):
        handler.handle_request_coordinate(
            coordinate=_coordinate(b"{}\n"), services=_services()
        )
    assert called == {"prepare": False}


def test_handler_reloads_prepared_custody_and_passes_only_its_dynamic_arn(
    monkeypatch,
) -> None:
    request = FakeRequest(FenceSlot.CLOSED_SOURCE)
    _patch_loaded_request(monkeypatch, request)
    monkeypatch.setattr(handler, "PreparedFenceChangeSet", FakePrepared)
    created = FakePrepared(
        "arn:aws:cloudformation:us-west-2:246813579024:"
        "changeSet/dynamic/00000000-0000-4000-8000-000000000001"
    )
    loaded = FakePrepared(created.change_set_arn)
    authority = SimpleNamespace(
        authority_class=ExecutorAuthorityClass.SUPPORT_RUNTIME
    )
    calls: list[str] = []
    monkeypatch.setattr(
        FenceExecutor,
        "load_create_authority",
        lambda *_: calls.append("create-authority") or authority,
    )
    monkeypatch.setattr(
        FenceExecutor,
        "load_prestate_snapshot",
        lambda *_, phase: calls.append("prestate-" + phase) or object(),
    )
    monkeypatch.setattr(
        FenceExecutor,
        "prepare_change_set",
        lambda *_, **__: calls.append("prepare") or created,
    )
    monkeypatch.setattr(
        FenceExecutor,
        "load_prepared_change_set",
        lambda *_: calls.append("prepared-readback") or loaded,
    )
    monkeypatch.setattr(
        FenceExecutor,
        "load_execute_authority",
        lambda *_, **__: calls.append("execute-authority") or authority,
    )
    result = SimpleNamespace(record_type="glm52_fence_execution_result_v2")

    def execute(*args, **kwargs):
        calls.append("execute")
        assert kwargs["prepared"] is loaded
        assert kwargs["prepared"].change_set_arn == created.change_set_arn
        assert "change_set_arn" not in request.to_dict()
        return result

    monkeypatch.setattr(FenceExecutor, "execute_prepared_change_set", execute)
    assert (
        handler.handle_request_coordinate(
            coordinate=_coordinate(b"{}\n"), services=_services()
        )
        is result
    )
    assert calls == [
        "create-authority",
        "prestate-CREATE",
        "prepare",
        "prepared-readback",
        "execute-authority",
        "prestate-EXECUTE",
        "execute",
    ]


def test_retained_closed_source_with_create_authority_uses_normal_prepare_path(
    monkeypatch,
) -> None:
    request = FakeRequest(FenceSlot.CLOSED_SOURCE)
    _patch_loaded_request(
        monkeypatch,
        request,
        allowed_create_authority_classes=(
            ExecutorAuthorityClass.SUPPORT_RUNTIME.value,
            ExecutorAuthorityClass.RETAINED_PRE_SUPPORT.value,
        ),
    )
    monkeypatch.setattr(handler, "PreparedFenceChangeSet", FakePrepared)
    created = FakePrepared(
        "arn:aws:cloudformation:us-west-2:246813579024:"
        "changeSet/retained-created/00000000-0000-4000-8000-000000000001"
    )
    loaded = FakePrepared(created.change_set_arn)
    retained = SimpleNamespace(
        authority_class=ExecutorAuthorityClass.RETAINED_PRE_SUPPORT
    )
    calls: list[str] = []
    monkeypatch.setattr(
        FenceExecutor,
        "load_create_authority",
        lambda *_: calls.append("create-authority") or retained,
    )
    monkeypatch.setattr(
        FenceExecutor,
        "load_prestate_snapshot",
        lambda *_, phase: calls.append("prestate-" + phase) or object(),
    )

    def prepare(*args, **kwargs):
        calls.append("prepare")
        assert (
            ExecutorAuthorityClass.RETAINED_PRE_SUPPORT.value
            in kwargs["entry"].allowed_create_authority_classes
        )
        assert kwargs["authority"] is retained
        return created

    monkeypatch.setattr(FenceExecutor, "prepare_change_set", prepare)

    def load_prepared(*args, **kwargs):
        assert "prepare" in calls, "CLOSED_SOURCE must create before prepared readback"
        calls.append("prepared-readback")
        return loaded

    monkeypatch.setattr(
        FenceExecutor, "load_prepared_change_set", load_prepared
    )
    monkeypatch.setattr(
        FenceExecutor,
        "load_execute_authority",
        lambda *_, **__: calls.append("execute-authority") or retained,
    )
    result = object()

    def execute(*args, **kwargs):
        calls.append("execute")
        assert kwargs["prepared"] is loaded
        return result

    monkeypatch.setattr(FenceExecutor, "execute_prepared_change_set", execute)
    assert (
        handler.handle_request_coordinate(
            coordinate=_coordinate(b"{}\n"),
            services=_services(
                authority_class=ExecutorAuthorityClass.RETAINED_PRE_SUPPORT
            ),
        )
        is result
    )
    assert calls == [
        "create-authority",
        "prestate-CREATE",
        "prepare",
        "prepared-readback",
        "execute-authority",
        "prestate-EXECUTE",
        "execute",
    ]


def test_retained_source_families_frozen_failover_is_execute_only_against_committed_prepared_custody(
    monkeypatch,
) -> None:
    request = FakeRequest(FenceSlot.SOURCE_FAMILIES_FROZEN)
    _patch_loaded_request(monkeypatch, request)
    monkeypatch.setattr(handler, "PreparedFenceChangeSet", FakePrepared)
    prepared = FakePrepared(
        "arn:aws:cloudformation:us-west-2:246813579024:"
        "changeSet/prearmed/00000000-0000-4000-8000-000000000001",
        slot=FenceSlot.SOURCE_FAMILIES_FROZEN,
    )
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        raise AssertionError("retained freeze failover must not create")

    monkeypatch.setattr(FenceExecutor, "load_create_authority", forbidden)
    monkeypatch.setattr(FenceExecutor, "prepare_change_set", forbidden)
    monkeypatch.setattr(
        FenceExecutor,
        "load_prepared_change_set",
        lambda *_: calls.append("prepared-readback") or prepared,
    )
    retained = SimpleNamespace(
        authority_class=ExecutorAuthorityClass.RETAINED_PRE_SUPPORT
    )
    monkeypatch.setattr(
        FenceExecutor,
        "load_execute_authority",
        lambda *_, **__: calls.append("execute-authority") or retained,
    )
    monkeypatch.setattr(
        FenceExecutor,
        "load_prestate_snapshot",
        lambda *_, phase: calls.append("prestate-" + phase) or object(),
    )
    freeze_delta_audit = object()
    monkeypatch.setattr(
        FenceExecutor,
        "load_freeze_execute_delta_audit",
        lambda *_, **__: (
            calls.append("freeze-delta-audit") or freeze_delta_audit
        ),
    )
    result = object()

    def execute(*args, **kwargs):
        calls.append("execute")
        assert kwargs["prepared"] is prepared
        assert kwargs["freeze_delta_audit"] is freeze_delta_audit
        return result

    monkeypatch.setattr(FenceExecutor, "execute_prepared_change_set", execute)
    assert (
        handler.handle_request_coordinate(
            coordinate=_coordinate(b"{}\n"),
            services=_services(
                authority_class=ExecutorAuthorityClass.RETAINED_PRE_SUPPORT
            ),
        )
        is result
    )
    assert calls == [
        "prepared-readback",
        "execute-authority",
        "prestate-EXECUTE",
        "freeze-delta-audit",
        "execute",
    ]


@pytest.mark.parametrize(
    "prepared",
    [
        object(),
        FakePrepared(
            "arn:aws:cloudformation:us-west-2:246813579024:"
            "changeSet/drifted/00000000-0000-4000-8000-000000000001",
            request_identity_sha256="9" * 64,
            slot=FenceSlot.SOURCE_FAMILIES_FROZEN,
        ),
    ],
)
def test_retained_source_families_frozen_failover_rejects_absent_or_differing_prepared_custody(
    monkeypatch,
    prepared: object,
) -> None:
    request = FakeRequest(FenceSlot.SOURCE_FAMILIES_FROZEN)
    _patch_loaded_request(monkeypatch, request)
    monkeypatch.setattr(handler, "PreparedFenceChangeSet", FakePrepared)

    def forbidden(*args, **kwargs):
        raise AssertionError("retained freeze failover must not create or execute")

    monkeypatch.setattr(FenceExecutor, "load_create_authority", forbidden)
    monkeypatch.setattr(FenceExecutor, "prepare_change_set", forbidden)
    monkeypatch.setattr(
        FenceExecutor,
        "load_prepared_change_set",
        lambda *_: prepared,
    )
    monkeypatch.setattr(FenceExecutor, "load_execute_authority", forbidden)
    monkeypatch.setattr(FenceExecutor, "execute_prepared_change_set", forbidden)
    with pytest.raises(ValueError, match="prepared change-set custody"):
        handler.handle_request_coordinate(
            coordinate=_coordinate(b"{}\n"),
            services=_services(
                authority_class=ExecutorAuthorityClass.RETAINED_PRE_SUPPORT
            ),
        )


def _runtime_coordinate() -> ArtifactCoordinate:
    return ArtifactCoordinate(
        bucket=BUCKET,
        key=(
            "campaigns/glm52-sky-20260724/authorities/fence/runtime/"
            "act-20260728-0001/00000001/SUPPORT_RUNTIME_IDENTITY.json"
        ),
        version_id="runtime-version-1",
        file_sha256="4" * 64,
        canonical_identity_sha256="5" * 64,
    )


class RuntimeRequest(FakeRequest):
    def to_dict(self) -> dict[str, object]:
        value = super().to_dict()
        value["support_runtime_identity_coordinate"] = (
            _runtime_coordinate().to_dict()
        )
        return value


def test_retained_failover_survives_deleted_support_only_for_freeze_or_closed(
    monkeypatch,
) -> None:
    identity = SimpleNamespace(canonical_identity_sha256="5" * 64)
    monkeypatch.setattr(handler, "_load_exact_json", lambda **_: {})
    monkeypatch.setattr(
        handler, "parse_support_runtime_identity", lambda _: identity
    )
    services = _services(
        authority_class=ExecutorAuthorityClass.RETAINED_PRE_SUPPORT,
        live=None,
    )
    for slot in (FenceSlot.SOURCE_FAMILIES_FROZEN, FenceSlot.CLOSED_SOURCE):
        handler._authenticate_runtime_identity(
            services=services,
            request=RuntimeRequest(slot),
            manifest=FakeManifest(),
        )
    for slot in (
        FenceSlot.RESERVATION_ONLY,
        FenceSlot.BATCH_FIVE_SOURCE_ACTIVATION,
        FenceSlot.TERMINAL,
    ):
        with pytest.raises(ValueError, match="only freeze or CLOSED_SOURCE"):
            handler._authenticate_runtime_identity(
                services=services,
                request=RuntimeRequest(slot),
                manifest=FakeManifest(),
            )


def test_support_runtime_requires_fresh_live_reader_and_exact_identity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(handler, "_load_exact_json", lambda **_: {})
    monkeypatch.setattr(
        handler,
        "parse_support_runtime_identity",
        lambda _: SimpleNamespace(canonical_identity_sha256="5" * 64),
    )
    with pytest.raises(ValueError, match="fresh live equality reader"):
        handler._authenticate_runtime_identity(
            services=_services(live=None),
            request=RuntimeRequest(FenceSlot.CLOSED_SOURCE),
            manifest=FakeManifest(),
        )
    monkeypatch.setattr(
        handler,
        "parse_support_runtime_identity",
        lambda _: SimpleNamespace(canonical_identity_sha256="6" * 64),
    )
    with pytest.raises(ValueError, match="coordinate drifted"):
        handler._authenticate_runtime_identity(
            services=_services(live=lambda: {}),
            request=RuntimeRequest(FenceSlot.CLOSED_SOURCE),
            manifest=FakeManifest(),
        )
