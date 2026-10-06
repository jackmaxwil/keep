from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import tempfile
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from mlx_vq.recovery_campaign import (
    EvidenceRecord,
    LedgerError,
    LedgerEvent,
    VerificationResult,
    append_event,
    append_evidence,
    evidence_from_event,
    load_ledger,
)


NOW = "2026-07-11T12:34:56.123456Z"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _event_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "sequence": 1,
        "timestamp": NOW,
        "event_kind": "transition_started",
        "experiment": "full75-e8",
        "payload": {"argv": ["python", "worker.py"], "attempt": 1},
        "previous_event_sha256": None,
    }
    body.update(overrides)
    return body


def _encoded_event(**overrides: object) -> bytes:
    body = _event_body(**overrides)
    event = {**body, "event_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    return _canonical(event) + b"\n"


def _verification() -> VerificationResult:
    return VerificationResult(
        name="campaign-unit-tests",
        argv=("python", "-m", "pytest", "tests/test_recovery_campaign_ledger.py"),
        exit_code=0,
        stdout_sha256=SHA_C,
        stderr_sha256=SHA_D,
    )


def _evidence(**overrides: object) -> EvidenceRecord:
    values: dict[str, object] = {
        "artifact_path": "artifacts/recovery/full75/conversion-manifest.json",
        "content_sha256": SHA_A,
        "manifest_identity_sha256": SHA_B,
        "candidate_identity_sha256": SHA_C,
        "parent_identity_sha256": None,
        "baseline_identity_sha256": SHA_D,
        "evidence_class": "release",
        "release_eligible": True,
        "recovery_levers": (
            "selection_diagonal_hessian_importance_weighted_reround_v1",
        ),
        "argv": ("python", "benchmarks/check_recovery.py", "--json"),
        "verification_results": (_verification(),),
    }
    values.update(overrides)
    return EvidenceRecord(**values)  # type: ignore[arg-type]


def test_append_event_writes_exact_canonical_hash_chain(tmp_path: Path) -> None:
    path = tmp_path / "campaign.jsonl"

    first = append_event(
        path,
        event_kind="transition_started",
        experiment="full75-e8",
        payload={"z": 2, "a": [True, None, "utf8-é"]},
        timestamp=NOW,
    )
    second = append_event(
        path,
        event_kind="transition_completed",
        experiment="full75-e8",
        payload={"ok": True},
        timestamp="2026-07-11T12:35:00Z",
    )

    assert isinstance(first, LedgerEvent)
    assert (first.sequence, second.sequence) == (1, 2)
    assert first.previous_event_sha256 is None
    assert second.previous_event_sha256 == first.event_sha256
    raw_lines = path.read_bytes().splitlines()
    assert len(raw_lines) == 2
    for raw_line, event in zip(raw_lines, (first, second), strict=True):
        parsed = json.loads(raw_line)
        assert tuple(sorted(parsed)) == tuple(
            sorted(
                (
                    "schema_version",
                    "sequence",
                    "timestamp",
                    "event_kind",
                    "experiment",
                    "payload",
                    "previous_event_sha256",
                    "event_sha256",
                )
            )
        )
        body = {key: value for key, value in parsed.items() if key != "event_sha256"}
        assert parsed["event_sha256"] == hashlib.sha256(_canonical(body)).hexdigest()
        assert parsed["event_sha256"] == event.event_sha256
        assert raw_line == _canonical(parsed)
    assert path.read_bytes().endswith(b"\n")
    assert load_ledger(path) == (first, second)


def test_retry_authorized_event_has_strict_schema_and_preserves_hash_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.jsonl"
    terminal = append_event(
        path,
        event_kind="transition_finished",
        experiment="full75-e8",
        payload={"exit_code": -15},
        timestamp=NOW,
    )
    authorization = append_event(
        path,
        event_kind="retry_authorized",
        experiment="full75-e8",
        payload={
            "campaign": "glm52-recovery-v1-20260711",
            "experiment": "full75-e8",
            "transition": "full75-rematerialize",
            "terminal_event_sha256": terminal.event_sha256,
            "exit_code": -15,
            "reason": "operator confirmed safe resume",
        },
        timestamp="2026-07-11T12:35:00Z",
    )

    loaded = load_ledger(path)
    assert loaded == (terminal, authorization)
    assert authorization.previous_event_sha256 == terminal.event_sha256

    with pytest.raises(LedgerError, match="fields|unknown"):
        append_event(
            path,
            event_kind="retry_authorized",
            experiment="full75-e8",
            payload={**dict(authorization.payload), "unknown": True},
        )


def test_ledger_events_are_frozen(tmp_path: Path) -> None:
    event = append_event(
        tmp_path / "campaign.jsonl",
        event_kind="observed",
        experiment="full75-e8",
        payload={},
        timestamp=NOW,
    )

    with pytest.raises(FrozenInstanceError):
        event.sequence = 2  # type: ignore[misc]


def test_ledger_event_payload_is_deeply_frozen_and_detached(tmp_path: Path) -> None:
    payload = {"nested": {"values": [1, {"ok": True}]}}
    event = append_event(
        tmp_path / "campaign.jsonl",
        event_kind="observed",
        experiment="full75-e8",
        payload=payload,
        timestamp=NOW,
    )
    payload["nested"]["values"].append(2)  # type: ignore[index,union-attr]

    assert event.payload == {"nested": {"values": [1, {"ok": True}]}}
    with pytest.raises(TypeError):
        event.payload["added"] = True
    with pytest.raises(TypeError):
        event.payload["nested"]["values"].append(2)  # type: ignore[index,union-attr]
    with pytest.raises(TypeError):
        event.payload["nested"]["values"][1]["ok"] = False  # type: ignore[index,union-attr]
    assert event.payload == {"nested": {"values": [1, {"ok": True}]}}


def test_ledger_payload_uses_genuine_immutable_mapping_and_sequence(
    tmp_path: Path,
) -> None:
    event = append_event(
        tmp_path / "campaign.jsonl",
        event_kind="observed",
        experiment="full75-e8",
        payload={"nested": {"values": [1, {"ok": True}]}},
        timestamp=NOW,
    )
    nested = event.payload["nested"]
    values = nested["values"]  # type: ignore[index]

    assert isinstance(event.payload, Mapping)
    assert isinstance(values, Sequence)
    assert not isinstance(event.payload, dict)
    assert not isinstance(values, list)
    assert event.payload == {"nested": {"values": [1, {"ok": True}]}}
    assert values == [1, {"ok": True}]
    with pytest.raises(TypeError):
        dict.__setitem__(event.payload, "forged", True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        list.append(values, "forged")  # type: ignore[arg-type]
    with pytest.raises((AttributeError, TypeError)):
        event.payload._items = ()  # type: ignore[attr-defined]
    with pytest.raises((AttributeError, TypeError)):
        values._values = ()  # type: ignore[attr-defined]


@pytest.mark.parametrize("schema_version", (1.0, True, "1"))
def test_load_requires_exact_integer_schema_version(
    tmp_path: Path, schema_version: object
) -> None:
    path = tmp_path / "campaign.jsonl"
    path.write_bytes(_encoded_event(schema_version=schema_version))

    with pytest.raises(LedgerError, match="schema_version"):
        load_ledger(path)


def test_returned_event_hash_reproduces_from_exact_typed_body(tmp_path: Path) -> None:
    path = tmp_path / "campaign.jsonl"
    event = append_event(
        path,
        event_kind="observed",
        experiment="full75-e8",
        payload={"nested": [1, True, None]},
        timestamp=NOW,
    )
    parsed = json.loads(path.read_bytes())
    body = {key: value for key, value in parsed.items() if key != "event_sha256"}

    assert type(event.schema_version) is int
    assert event.schema_version == 1
    assert event.event_sha256 == hashlib.sha256(_canonical(body)).hexdigest()


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        (_encoded_event(sequence=2), "sequence"),
        (_encoded_event(previous_event_sha256=SHA_A), "previous"),
        (_encoded_event(event_kind=""), "event_kind"),
        (_encoded_event(experiment=""), "experiment"),
        (_encoded_event(timestamp="2026-07-11T12:34:56+00:00"), "timestamp"),
        (
            _canonical({**_event_body(), "event_sha256": "A" * 64}) + b"\n",
            "event_sha256",
        ),
    ),
)
def test_load_rejects_invalid_event_shape_or_continuity(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    path = tmp_path / "campaign.jsonl"
    path.write_bytes(raw)

    with pytest.raises(LedgerError, match=message):
        load_ledger(path)


def test_load_rejects_mutated_event_body(tmp_path: Path) -> None:
    path = tmp_path / "campaign.jsonl"
    path.write_bytes(_encoded_event().replace(b'"attempt":1', b'"attempt":2'))

    with pytest.raises(LedgerError, match="sha256"):
        load_ledger(path)


def test_load_rejects_partial_append_without_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "campaign.jsonl"
    path.write_bytes(_encoded_event().rstrip(b"\n"))

    with pytest.raises(LedgerError, match="final newline|partial|truncat"):
        load_ledger(path)


def test_load_rejects_duplicate_unknown_and_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "campaign.jsonl"
    valid = _encoded_event().rstrip(b"\n")
    duplicate = valid.replace(b'"sequence":1', b'"sequence":1,"sequence":1')
    path.write_bytes(duplicate + b"\n")
    with pytest.raises(LedgerError, match="duplicate"):
        load_ledger(path)

    parsed = json.loads(valid)
    parsed["unknown"] = True
    path.write_bytes(_canonical(parsed) + b"\n")
    with pytest.raises(LedgerError, match="unknown|fields"):
        load_ledger(path)

    parsed.pop("unknown")
    parsed.pop("payload")
    path.write_bytes(_canonical(parsed) + b"\n")
    with pytest.raises(LedgerError, match="missing|fields"):
        load_ledger(path)


@pytest.mark.parametrize(
    "raw",
    (
        b"\xff\n",
        _encoded_event().replace(b'"attempt":1', b'"attempt":NaN'),
        _encoded_event() + b"\n",
    ),
)
def test_load_rejects_invalid_utf8_nonfinite_and_blank_lines(
    tmp_path: Path, raw: bytes
) -> None:
    path = tmp_path / "campaign.jsonl"
    path.write_bytes(raw)

    with pytest.raises(LedgerError):
        load_ledger(path)


def test_load_rejects_symlink_and_nonregular_file(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_bytes(_encoded_event())
    symlink = tmp_path / "symlink.jsonl"
    symlink.symlink_to(target.name)
    fifo = tmp_path / "ledger.fifo"
    os.mkfifo(fifo)

    with pytest.raises(LedgerError, match="symlink|regular"):
        load_ledger(symlink)
    with pytest.raises(LedgerError, match="regular"):
        load_ledger(fifo)


def test_external_head_and_count_anchors_detect_clean_tail_deletion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.jsonl"
    first = append_event(
        path,
        event_kind="one",
        experiment="full75-e8",
        payload={},
        timestamp=NOW,
    )
    second = append_event(
        path,
        event_kind="two",
        experiment="full75-e8",
        payload={},
        timestamp="2026-07-11T12:35:00Z",
    )
    first_line = path.read_bytes().splitlines(keepends=True)[0]
    path.write_bytes(first_line)

    assert load_ledger(path) == (first,)
    with pytest.raises(LedgerError, match="head"):
        load_ledger(path, expected_head_sha256=second.event_sha256)
    with pytest.raises(LedgerError, match="count"):
        load_ledger(path, expected_event_count=2)


def test_append_requires_existing_real_parent(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "campaign.jsonl"
    with pytest.raises(LedgerError, match="parent"):
        append_event(
            missing,
            event_kind="observed",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
        )
    assert not missing.parent.exists()

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(LedgerError, match="parent|directory|symlink"):
        append_event(
            linked / "campaign.jsonl",
            event_kind="observed",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
        )


def test_append_rejects_an_intermediate_parent_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    nested = real / "nested"
    nested.mkdir(parents=True)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(LedgerError, match="parent|directory|symlink"):
        append_event(
            linked / "nested" / "campaign.jsonl",
            event_kind="observed",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
        )
    assert not (nested / "campaign.jsonl").exists()


def test_append_rejects_leaf_replacement_while_waiting_for_lock(
    tmp_path: Path, monkeypatch
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"
    first = append_event(
        path,
        event_kind="one",
        experiment="full75-e8",
        payload={},
        timestamp=NOW,
    )
    original_raw = path.read_bytes()
    locked_fd = os.open(path, os.O_RDWR)
    real_flock = ledger_module.fcntl.flock
    real_flock(locked_fd, fcntl.LOCK_EX)
    waiting = threading.Event()

    def tracking_flock(fd: int, operation: int) -> object:
        if fd != locked_fd and operation & fcntl.LOCK_EX:
            waiting.set()
        return real_flock(fd, operation)

    monkeypatch.setattr(ledger_module.fcntl, "flock", tracking_flock)
    parked = tmp_path / "parked.jsonl"
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                append_event,
                path,
                event_kind="two",
                experiment="full75-e8",
                payload={},
                timestamp="2026-07-11T12:35:00Z",
                expected_head_sha256=first.event_sha256,
                expected_event_count=1,
            )
            assert waiting.wait(timeout=5), "append did not reach the held lock"
            path.replace(parked)
            path.write_bytes(_encoded_event(event_kind="replacement"))
            real_flock(locked_fd, fcntl.LOCK_UN)
            with pytest.raises(LedgerError, match="replaced|identity|mutat|ABA"):
                future.result(timeout=5)
    finally:
        real_flock(locked_fd, fcntl.LOCK_UN)
        os.close(locked_fd)

    assert parked.read_bytes() == original_raw


def test_append_rejects_materially_changed_leaf_after_same_inode_restore(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"
    first = append_event(
        path,
        event_kind="one",
        experiment="full75-e8",
        payload={},
        timestamp=NOW,
    )
    original_raw = path.read_bytes()
    locked_fd = os.open(path, os.O_RDWR)
    real_flock = ledger_module.fcntl.flock
    real_flock(locked_fd, fcntl.LOCK_EX)
    waiting = threading.Event()

    def tracking_flock(fd: int, operation: int) -> object:
        if fd != locked_fd and operation & fcntl.LOCK_EX:
            waiting.set()
        return real_flock(fd, operation)

    monkeypatch.setattr(ledger_module.fcntl, "flock", tracking_flock)
    parked = tmp_path / "parked.jsonl"
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                append_event,
                path,
                event_kind="two",
                experiment="full75-e8",
                payload={},
                timestamp="2026-07-11T12:35:00Z",
                expected_head_sha256=first.event_sha256,
                expected_event_count=1,
            )
            assert waiting.wait(timeout=5), "append did not reach the held lock"
            path.replace(parked)
            parked.write_bytes(_encoded_event(event_kind="material-change"))
            parked.replace(path)
            real_flock(locked_fd, fcntl.LOCK_UN)
            with pytest.raises(LedgerError, match="head|sha256|sequence|event"):
                future.result(timeout=5)
    finally:
        real_flock(locked_fd, fcntl.LOCK_UN)
        os.close(locked_fd)


@pytest.mark.parametrize("operation", ("load", "append"))
def test_full_ancestor_chain_rejects_replacement_while_waiting_for_lock(
    tmp_path: Path,
    monkeypatch,
    operation: str,
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    ancestor = tmp_path / "ancestor"
    parent = ancestor / "nested"
    parent.mkdir(parents=True)
    path = parent / "campaign.jsonl"
    append_event(
        path,
        event_kind="one",
        experiment="full75-e8",
        payload={},
        timestamp=NOW,
    )
    original_raw = path.read_bytes()
    locked_fd = os.open(path, os.O_RDWR)
    real_flock = ledger_module.fcntl.flock
    real_flock(locked_fd, fcntl.LOCK_EX)
    waiting = threading.Event()

    def tracking_flock(fd: int, lock_operation: int) -> object:
        if fd != locked_fd and lock_operation & (fcntl.LOCK_SH | fcntl.LOCK_EX):
            waiting.set()
        return real_flock(fd, lock_operation)

    monkeypatch.setattr(ledger_module.fcntl, "flock", tracking_flock)
    parked = tmp_path / "parked-ancestor"
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            if operation == "load":
                future = executor.submit(load_ledger, path)
            else:
                future = executor.submit(
                    append_event,
                    path,
                    event_kind="two",
                    experiment="full75-e8",
                    payload={},
                    timestamp="2026-07-11T12:35:00Z",
                )
            assert waiting.wait(timeout=5), "operation did not reach the held lock"
            ancestor.replace(parked)
            parent.mkdir(parents=True)
            path.write_bytes(original_raw)
            real_flock(locked_fd, fcntl.LOCK_UN)
            with pytest.raises(LedgerError, match="ancestor|identity|mutat|ABA"):
                future.result(timeout=5)
    finally:
        real_flock(locked_fd, fcntl.LOCK_UN)
        os.close(locked_fd)

    authoritative = parked / "nested" / path.name if parked.exists() else path
    assert authoritative.read_bytes() == original_raw


def test_ancestor_restore_rejects_replaced_descendant_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    ancestor = tmp_path / "ancestor"
    parent = ancestor / "nested"
    parent.mkdir(parents=True)
    path = parent / "campaign.jsonl"
    append_event(
        path,
        event_kind="one",
        experiment="full75-e8",
        payload={},
        timestamp=NOW,
    )
    original_raw = path.read_bytes()
    locked_fd = os.open(path, os.O_RDWR)
    real_flock = ledger_module.fcntl.flock
    real_flock(locked_fd, fcntl.LOCK_EX)
    waiting = threading.Event()

    def tracking_flock(fd: int, operation: int) -> object:
        if fd != locked_fd and operation & fcntl.LOCK_SH:
            waiting.set()
        return real_flock(fd, operation)

    monkeypatch.setattr(ledger_module.fcntl, "flock", tracking_flock)
    parked = tmp_path / "parked-ancestor"
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(load_ledger, path)
            assert waiting.wait(timeout=5), "load did not reach the held lock"
            ancestor.replace(parked)
            old_nested = parked / "old-nested"
            (parked / "nested").replace(old_nested)
            (parked / "nested").mkdir()
            (parked / "nested" / path.name).write_bytes(original_raw)
            parked.replace(ancestor)
            real_flock(locked_fd, fcntl.LOCK_UN)
            with pytest.raises(LedgerError, match="ancestor|identity|replaced"):
                future.result(timeout=5)
    finally:
        real_flock(locked_fd, fcntl.LOCK_UN)
        os.close(locked_fd)


@pytest.mark.parametrize("operation", ("load", "append"))
def test_ledger_allows_unrelated_sibling_churn_while_waiting_for_lock(
    tmp_path: Path,
    monkeypatch,
    operation: str,
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    parent = tmp_path / "ancestor" / "nested"
    parent.mkdir(parents=True)
    path = parent / "campaign.jsonl"
    append_event(
        path,
        event_kind="one",
        experiment="full75-e8",
        payload={},
        timestamp=NOW,
    )
    locked_fd = os.open(path, os.O_RDWR)
    real_flock = ledger_module.fcntl.flock
    real_flock(locked_fd, fcntl.LOCK_EX)
    waiting = threading.Event()

    def tracking_flock(fd: int, lock_operation: int) -> object:
        if fd != locked_fd and lock_operation & (fcntl.LOCK_SH | fcntl.LOCK_EX):
            waiting.set()
        return real_flock(fd, lock_operation)

    monkeypatch.setattr(ledger_module.fcntl, "flock", tracking_flock)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            if operation == "load":
                future = executor.submit(load_ledger, path)
            else:
                future = executor.submit(
                    append_event,
                    path,
                    event_kind="two",
                    experiment="full75-e8",
                    payload={},
                    timestamp="2026-07-11T12:35:00Z",
                )
            assert waiting.wait(timeout=5), "operation did not reach the held lock"
            (tmp_path / "unrelated" / "child").mkdir(parents=True)
            real_flock(locked_fd, fcntl.LOCK_UN)
            result = future.result(timeout=5)
    finally:
        real_flock(locked_fd, fcntl.LOCK_UN)
        os.close(locked_fd)

    if operation == "load":
        assert len(result) == 1
    else:
        assert result.sequence == 2


def test_load_missing_ledger_visible_race_is_bounded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"
    real_open = ledger_module.os.open
    real_stat = ledger_module.os.stat
    ledger_open_attempts = 0

    def racing_open(name, flags, *args, **kwargs):
        nonlocal ledger_open_attempts
        if name == path.name and kwargs.get("dir_fd") is not None:
            ledger_open_attempts += 1
            raise FileNotFoundError(name)
        return real_open(name, flags, *args, **kwargs)

    def racing_stat(name, *args, **kwargs):
        if name == path.name and kwargs.get("dir_fd") is not None:
            return real_stat(tmp_path)
        return real_stat(name, *args, **kwargs)

    monkeypatch.setattr(ledger_module.os, "open", racing_open)
    monkeypatch.setattr(ledger_module.os, "stat", racing_stat)

    with pytest.raises(LedgerError, match="kept appearing"):
        load_ledger(path)

    assert ledger_open_attempts == 8


def test_append_refuses_corrupt_existing_ledger_without_changing_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.jsonl"
    path.write_bytes(_encoded_event().replace(b'"attempt":1', b'"attempt":2'))
    before = path.read_bytes()

    with pytest.raises(LedgerError, match="sha256"):
        append_event(
            path,
            event_kind="next",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
        )

    assert path.read_bytes() == before


def test_append_rolls_back_a_short_write(tmp_path: Path, monkeypatch) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"
    original = append_event(
        path,
        event_kind="one",
        experiment="full75-e8",
        payload={},
        timestamp=NOW,
    )
    before = path.read_bytes()
    real_write = ledger_module.os.write

    def short_write(fd: int, raw: bytes) -> int:
        return real_write(fd, raw[: max(1, len(raw) // 2)])

    monkeypatch.setattr(ledger_module.os, "write", short_write)
    with pytest.raises(LedgerError, match="short|append"):
        append_event(
            path,
            event_kind="two",
            experiment="full75-e8",
            payload={},
            timestamp="2026-07-11T12:35:00Z",
        )

    assert path.read_bytes() == before
    assert load_ledger(path) == (original,)


def test_created_ledger_fsyncs_file_then_validates_then_fsyncs_parent(
    tmp_path: Path, monkeypatch
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"
    calls: list[str] = []
    real_write = ledger_module.os.write
    real_fsync = ledger_module.os.fsync
    real_verify = ledger_module._verify_private_identity
    real_publish = ledger_module._publish_private_noreplace
    real_verify_published = ledger_module._verify_published_identity

    def tracking_write(fd: int, raw: bytes) -> int:
        calls.append("write")
        return real_write(fd, raw)

    def tracking_fsync(fd: int) -> None:
        kind = "dir_fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_fsync"
        calls.append(kind)
        real_fsync(fd)

    def tracking_verify(opened) -> None:
        calls.append("verify")
        real_verify(opened)

    def tracking_publish(opened) -> None:
        calls.append("publish")
        real_publish(opened)

    def tracking_verify_published(opened) -> None:
        calls.append("published_verify")
        real_verify_published(opened)

    monkeypatch.setattr(ledger_module.os, "write", tracking_write)
    monkeypatch.setattr(ledger_module.os, "fsync", tracking_fsync)
    monkeypatch.setattr(
        ledger_module, "_verify_private_identity", tracking_verify
    )
    monkeypatch.setattr(
        ledger_module, "_publish_private_noreplace", tracking_publish
    )
    monkeypatch.setattr(
        ledger_module, "_verify_published_identity", tracking_verify_published
    )

    append_event(
        path,
        event_kind="created",
        experiment="full75-e8",
        payload={},
        timestamp=NOW,
    )

    write_index = calls.index("write")
    assert calls[write_index:] == [
        "write",
        "file_fsync",
        "verify",
        "publish",
        "verify",
        "verify",
        "published_verify",
        "dir_fsync",
        "published_verify",
    ]


def test_created_ledger_directory_fsync_failure_leaves_valid_uncertain_commit(
    tmp_path: Path, monkeypatch
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"
    real_fsync = ledger_module.os.fsync
    failed_once = False

    def fail_first_directory_fsync(fd: int) -> None:
        nonlocal failed_once
        if stat.S_ISDIR(os.fstat(fd).st_mode) and not failed_once:
            failed_once = True
            raise OSError("injected directory fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(ledger_module.os, "fsync", fail_first_directory_fsync)
    with pytest.raises(LedgerError, match="uncertain committed state"):
        append_event(
            path,
            event_kind="created",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
        )

    assert failed_once is True
    assert len(load_ledger(path)) == 1


def test_private_temp_name_can_never_equal_public_final_name(
    tmp_path: Path, monkeypatch
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    token = "a" * 48
    path = tmp_path / f".recovery-ledger-{token}.tmp"
    monkeypatch.setattr(ledger_module.secrets, "token_hex", lambda _size: token)

    append_event(
        path,
        event_kind="created",
        experiment="full75-e8",
        payload={},
        timestamp=NOW,
    )

    assert len(load_ledger(path)) == 1
    assert tuple(item.name for item in tmp_path.iterdir()) == (path.name,)


def test_concurrent_creators_and_writers_form_one_chain_repeatedly() -> None:
    with tempfile.TemporaryDirectory(
        prefix=".recovery-ledger-test-", dir=Path.cwd()
    ) as directory:
        for repetition in range(3):
            path = Path(directory) / f"campaign-{repetition}.jsonl"

            def append(index: int) -> LedgerEvent:
                return append_event(
                    path,
                    event_kind="worker_completed",
                    experiment=f"worker-{index}",
                    payload={"index": index},
                    timestamp="2026-07-11T12:35:00Z",
                )

            with ThreadPoolExecutor(max_workers=16) as executor:
                returned = tuple(executor.map(append, range(32)))

            loaded = load_ledger(path)
            assert len(loaded) == len(returned) == 32
            assert tuple(event.sequence for event in loaded) == tuple(range(1, 33))
            assert {event.event_sha256 for event in loaded} == {
                event.event_sha256 for event in returned
            }
            assert all(
                current.previous_event_sha256 == previous.event_sha256
                for previous, current in zip(loaded, loaded[1:])
            )


@pytest.mark.parametrize(
    "payload",
    (
        {1: "not a string key"},
        {"bad": float("nan")},
        {"bad": float("inf")},
        {"not_json": (1, 2)},
    ),
)
def test_append_rejects_payloads_outside_finite_json(
    tmp_path: Path, payload: dict[object, object]
) -> None:
    with pytest.raises((LedgerError, TypeError, ValueError)):
        append_event(
            tmp_path / "campaign.jsonl",
            event_kind="observed",
            experiment="full75-e8",
            payload=payload,  # type: ignore[arg-type]
            timestamp=NOW,
        )


def test_evidence_registration_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    evidence = _evidence()

    first = append_evidence(
        first_path,
        experiment="full75-e8",
        evidence=evidence,
        timestamp=NOW,
    )
    second = append_evidence(
        second_path,
        experiment="full75-e8",
        evidence=evidence,
        timestamp=NOW,
    )

    assert first.event_kind == "evidence_registered"
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert evidence_from_event(load_ledger(first_path)[0]) == evidence
    assert first.payload == {
        "artifact_path": evidence.artifact_path,
        "content_sha256": evidence.content_sha256,
        "manifest_identity_sha256": evidence.manifest_identity_sha256,
        "candidate_identity_sha256": evidence.candidate_identity_sha256,
        "parent_identity_sha256": evidence.parent_identity_sha256,
        "baseline_identity_sha256": evidence.baseline_identity_sha256,
        "evidence_class": evidence.evidence_class,
        "release_eligible": evidence.release_eligible,
        "recovery_levers": list(evidence.recovery_levers),
        "argv": list(evidence.argv),
        "verification_results": [
            {
                "name": _verification().name,
                "argv": list(_verification().argv),
                "exit_code": _verification().exit_code,
                "stdout_sha256": _verification().stdout_sha256,
                "stderr_sha256": _verification().stderr_sha256,
            }
        ],
    }


def test_evidence_record_is_frozen() -> None:
    evidence = _evidence()
    with pytest.raises(FrozenInstanceError):
        evidence.release_eligible = False  # type: ignore[misc]


def test_evidence_identity_relationships_are_optional() -> None:
    evidence = EvidenceRecord(
        artifact_path="artifacts/recovery/audit.json",
        content_sha256=SHA_A,
        evidence_class="diagnostic_only",
        release_eligible=False,
        recovery_levers=("selection_reround",),
        argv=("python", "audit.py"),
        verification_results=(_verification(),),
    )

    assert evidence.manifest_identity_sha256 is None
    assert evidence.candidate_identity_sha256 is None
    assert evidence.parent_identity_sha256 is None
    assert evidence.baseline_identity_sha256 is None


def test_evidence_requires_exact_verification_result_records() -> None:
    with pytest.raises((TypeError, ValueError), match="verification_results"):
        _evidence(verification_results=())


def test_diagnostic_evidence_cannot_be_laundered_as_release_eligible() -> None:
    with pytest.raises(ValueError, match="diagnostic_only|release_eligible"):
        _evidence(evidence_class="diagnostic_only", release_eligible=True)


@pytest.mark.parametrize(
    "field",
    (
        "candidate_identity_sha256",
        "manifest_identity_sha256",
        "baseline_identity_sha256",
    ),
)
def test_release_eligible_evidence_requires_authoritative_identities(
    field: str,
) -> None:
    with pytest.raises(ValueError, match="release_eligible|identity"):
        _evidence(**{field: None})


def test_release_eligible_evidence_requires_all_verifications_to_pass() -> None:
    failed = replace(_verification(), exit_code=1)

    with pytest.raises(ValueError, match="release_eligible|verification|exit"):
        _evidence(verification_results=(failed,))


@pytest.mark.parametrize("relationship", ("parent", "baseline"))
@pytest.mark.parametrize("evidence_class", ("release", "diagnostic_only"))
def test_candidate_identity_must_differ_from_parent_and_baseline(
    relationship: str, evidence_class: str
) -> None:
    overrides: dict[str, object] = {
        "evidence_class": evidence_class,
        "release_eligible": evidence_class == "release",
        f"{relationship}_identity_sha256": SHA_C,
    }

    with pytest.raises(ValueError, match="candidate|identity|distinct"):
        _evidence(**overrides)


@pytest.mark.parametrize(
    "artifact_path",
    (
        "/absolute/artifact.json",
        "../escape.json",
        "artifacts/../escape.json",
        "artifacts\\windows.json",
        "",
    ),
)
def test_evidence_rejects_unsafe_artifact_paths(artifact_path: str) -> None:
    with pytest.raises(ValueError, match="artifact_path|relative|safe"):
        _evidence(artifact_path=artifact_path)


@pytest.mark.parametrize(
    "argv",
    (
        "python worker.py",
        ("sh", "-c", "python worker.py"),
        ("/bin/bash", "-lc", "python worker.py"),
    ),
)
def test_evidence_rejects_shell_commands(argv: object) -> None:
    with pytest.raises((TypeError, ValueError), match="argv|shell|tuple"):
        _evidence(argv=argv)


@pytest.mark.parametrize(
    "argv",
    (
        ("/usr/bin/env", "bash", "-c", "true"),
        ("env", "FOO=bar", "/bin/sh", "-c", "true"),
        ("/usr/bin/env", "-S", "bash -c true"),
        ("/usr/bin/env", "env", "MODE=test", "sh", "-c", "true"),
    ),
)
def test_evidence_rejects_env_mediated_shells_and_split_string(
    argv: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="env|shell|argv|-S"):
        _evidence(argv=argv)


def test_verification_rejects_env_mediated_shell() -> None:
    with pytest.raises(ValueError, match="env|shell|argv"):
        replace(
            _verification(),
            argv=("/usr/bin/env", "-i", "bash", "-c", "true"),
        )


@pytest.mark.parametrize(
    "wrapper",
    (
        "nice",
        "nohup",
        "sudo",
        "doas",
        "xargs",
        "command",
        "time",
        "timeout",
        "setsid",
        "stdbuf",
        "chrt",
        "ionice",
    ),
)
def test_evidence_rejects_unsupported_launch_wrappers(wrapper: str) -> None:
    with pytest.raises(ValueError, match="wrapper|argv|unsupported"):
        _evidence(argv=(wrapper, "python", "audit.py"))


@pytest.mark.parametrize(
    "argv",
    (
        ("nice", "sh", "-c", "true"),
        ("/usr/bin/env", "nice", "bash", "-c", "true"),
    ),
)
def test_evidence_rejects_direct_and_env_nested_nice_shells(
    argv: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="wrapper|argv|unsupported"):
        _evidence(argv=argv)


@pytest.mark.parametrize(
    "argv",
    (
        ("nohup", "sh", "-c", "true"),
        ("env", "MODE=test", "nohup", "bash", "-c", "true"),
        ("nice", "bash", "-c", "true"),
        ("env", "nice", "sh", "-c", "true"),
        ("sudo", "python", "audit.py"),
        ("env", "timeout", "30", "python", "audit.py"),
    ),
)
def test_verification_rejects_direct_nested_and_representative_wrappers(
    argv: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="wrapper|argv|unsupported"):
        replace(_verification(), argv=argv)


def test_env_wrapper_allows_narrow_direct_command_form() -> None:
    evidence = _evidence(
        argv=(
            "/usr/bin/env",
            "-i",
            "--unset",
            "PYTHONPATH",
            "MODE=audit",
            "python",
            "audit.py",
        )
    )

    assert evidence.argv[-2:] == ("python", "audit.py")


@pytest.mark.parametrize(
    "executable",
    (
        "python",
        "python3",
        "python3.11",
        "/opt/tools/python3.14",
        "pytest",
        "keep",
        "uv",
        "git",
        "rg",
        "node",
    ),
)
def test_registry_accepts_only_supported_direct_executable_variants(
    executable: str,
) -> None:
    evidence = _evidence(argv=(executable, "--version"))

    assert evidence.argv == (executable, "--version")


@pytest.mark.parametrize(
    "executable",
    (
        "arch",
        "caffeinate",
        "custom-runner",
        "PYTHON",
        "python2",
        "python3.x",
        "python3.11m",
    ),
)
def test_registry_rejects_every_unrecognized_direct_executable(
    executable: str,
) -> None:
    with pytest.raises(ValueError, match="allow|supported|executable|argv"):
        _evidence(argv=(executable, "python", "audit.py"))


def test_registry_applies_allowlist_after_recursive_env_unwrap() -> None:
    allowed = _evidence(
        argv=("env", "MODE=outer", "env", "-i", "MODE=inner", "uv", "run")
    )

    assert allowed.argv[-2:] == ("uv", "run")
    with pytest.raises(ValueError, match="allow|supported|executable|argv"):
        _evidence(argv=("env", "MODE=outer", "env", "arch", "python", "x.py"))


@pytest.mark.parametrize(
    "argv",
    (
        ("caffeinate", "python", "audit.py"),
        ("env", "MODE=audit", "arch", "python", "audit.py"),
    ),
)
def test_verification_registry_rejects_unknown_direct_and_nested_executables(
    argv: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="allow|supported|executable|argv"):
        replace(_verification(), argv=argv)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("artifact_path", "artifacts/recovery/bad\x00name.json"),
        ("artifact_path", "artifacts/recovery/bad\x7fname.json"),
        ("argv", ("python", "bad\nargument")),
        ("recovery_levers", ("selection\tlever",)),
    ),
)
def test_evidence_rejects_ascii_control_characters(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match="control|ASCII|safe|argv|lever"):
        _evidence(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    (
        {"name": "bad\nverification"},
        {"argv": ("python", "bad\x01argument")},
    ),
)
def test_verification_rejects_ascii_control_characters(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="control|ASCII|argv|name"):
        replace(_verification(), **overrides)


@pytest.mark.parametrize(
    ("field", "value"),
    (("event_kind", "bad\nevent"), ("experiment", "bad\x7fexperiment")),
)
def test_append_rejects_control_characters_in_event_names(
    tmp_path: Path, field: str, value: str
) -> None:
    arguments = {
        "event_kind": "observed",
        "experiment": "full75-e8",
        "payload": {},
        "timestamp": NOW,
        field: value,
    }

    with pytest.raises(LedgerError, match="control|ASCII|event|experiment"):
        append_event(tmp_path / "campaign.jsonl", **arguments)  # type: ignore[arg-type]
    assert not (tmp_path / "campaign.jsonl").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("content_sha256", "a" * 63),
        ("manifest_identity_sha256", "A" * 64),
        ("candidate_identity_sha256", "not-a-hash"),
        ("parent_identity_sha256", ""),
        ("baseline_identity_sha256", 7),
    ),
)
def test_evidence_rejects_malformed_identities(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="sha256|identity|hash"):
        _evidence(**{field: value})


def test_evidence_registry_records_identity_without_reading_artifact(
    tmp_path: Path,
) -> None:
    evidence = _evidence(artifact_path="artifacts/does-not-exist.json")

    event = append_evidence(
        tmp_path / "campaign.jsonl",
        experiment="full75-e8",
        evidence=evidence,
        timestamp=NOW,
    )

    assert event.payload["content_sha256"] == SHA_A
    assert not (tmp_path / evidence.artifact_path).exists()


def test_evidence_from_event_rejects_other_event_kinds(tmp_path: Path) -> None:
    event = append_event(
        tmp_path / "campaign.jsonl",
        event_kind="observed",
        experiment="full75-e8",
        payload={},
        timestamp=NOW,
    )

    with pytest.raises(LedgerError, match="evidence_registered"):
        evidence_from_event(event)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda event: replace(event, schema_version=1.0),
        lambda event: replace(event, schema_version=True),
        lambda event: replace(event, sequence=0),
        lambda event: replace(event, sequence=True),
        lambda event: replace(event, timestamp="2026-07-11T12:34:56+00:00"),
        lambda event: replace(event, experiment=""),
        lambda event: replace(event, previous_event_sha256=SHA_A),
        lambda event: replace(event, event_sha256="0" * 64),
        lambda event: replace(
            event,
            payload={**dict(event.payload), "release_eligible": False},
        ),
    ),
)
def test_evidence_from_event_rejects_forged_standalone_event(
    tmp_path: Path, mutate
) -> None:
    event = append_evidence(
        tmp_path / "campaign.jsonl",
        experiment="full75-e8",
        evidence=_evidence(),
        timestamp=NOW,
    )

    with pytest.raises(
        LedgerError,
        match="event|schema|sequence|timestamp|experiment|previous|sha256|payload",
    ):
        evidence_from_event(mutate(event))


def test_anchored_append_rejects_clean_tail_deletion_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.jsonl"
    first = append_event(
        path,
        event_kind="one",
        experiment="full75-e8",
        payload={},
        timestamp=NOW,
    )
    second = append_event(
        path,
        event_kind="two",
        experiment="full75-e8",
        payload={},
        timestamp="2026-07-11T12:35:00Z",
    )
    first_line = path.read_bytes().splitlines(keepends=True)[0]
    path.write_bytes(first_line)
    before = path.read_bytes()

    with pytest.raises(LedgerError, match="head|count"):
        append_event(
            path,
            event_kind="three",
            experiment="full75-e8",
            payload={},
            timestamp="2026-07-11T12:36:00Z",
            expected_head_sha256=second.event_sha256,
            expected_event_count=2,
        )

    assert path.read_bytes() == before
    assert load_ledger(path) == (first,)


@pytest.mark.parametrize(
    "anchor",
    (
        {"expected_head_sha256": SHA_A},
        {"expected_event_count": 1},
    ),
)
def test_missing_ledger_anchor_mismatch_never_publishes_final_leaf(
    tmp_path: Path, anchor: dict[str, object]
) -> None:
    path = tmp_path / "campaign.jsonl"

    with pytest.raises(LedgerError, match="head|count"):
        append_event(
            path,
            event_kind="one",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
            **anchor,  # type: ignore[arg-type]
        )

    assert not path.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_private_ledger_flock_failure_never_publishes_final_leaf(
    tmp_path: Path, monkeypatch
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"

    def fail_flock(_fd: int, _operation: int) -> None:
        raise OSError("injected flock failure")

    monkeypatch.setattr(ledger_module.fcntl, "flock", fail_flock)
    with pytest.raises(LedgerError, match="append|flock|lock"):
        append_event(
            path,
            event_kind="one",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
        )

    assert not path.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_private_cleanup_never_unlinks_adversarial_public_replacement(
    tmp_path: Path, monkeypatch
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"

    def create_public_then_fail(_opened) -> None:
        path.write_bytes(b"replacement")
        raise LedgerError("injected pre-publication failure")

    monkeypatch.setattr(
        ledger_module, "_before_private_publication", create_public_then_fail
    )
    with pytest.raises(LedgerError, match="pre-publication"):
        append_event(
            path,
            event_kind="one",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
        )

    assert path.read_bytes() == b"replacement"
    assert [item.name for item in tmp_path.iterdir()] == [path.name]


@pytest.mark.parametrize("force_link_fallback", (False, True))
def test_public_staging_replacement_cannot_substitute_published_source(
    tmp_path: Path, monkeypatch, force_link_fallback: bool
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"
    parked_name = "parked-private-stage"
    replacement_raw = b"attacker-controlled-incomplete-bytes"

    def replace_public_staging_entry(opened) -> None:
        assert opened.staging_name is not None
        os.rename(
            opened.staging_name,
            parked_name,
            src_dir_fd=opened.final_parent_fd,
            dst_dir_fd=opened.final_parent_fd,
        )
        os.mkdir(opened.staging_name, mode=0o700, dir_fd=opened.final_parent_fd)
        replacement_fd = os.open(
            opened.staging_name,
            os.O_RDONLY | os.O_DIRECTORY,
            dir_fd=opened.final_parent_fd,
        )
        try:
            fake_fd = os.open(
                opened.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=replacement_fd,
            )
            try:
                os.write(fake_fd, replacement_raw)
            finally:
                os.close(fake_fd)
        finally:
            os.close(replacement_fd)

    monkeypatch.setattr(
        ledger_module,
        "_before_private_publication",
        replace_public_staging_entry,
    )
    if force_link_fallback:
        monkeypatch.setattr(
            ledger_module,
            "_renameatx_np_noreplace",
            lambda _opened: False,
        )
        monkeypatch.setattr(
            ledger_module,
            "_renameat2_noreplace",
            lambda _opened: False,
        )
    with pytest.raises(LedgerError, match="uncertain committed state"):
        append_event(
            path,
            event_kind="one",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
        )

    loaded = load_ledger(path)
    assert len(loaded) == 1
    assert path.read_bytes() != replacement_raw


def _substitute_private_source(opened, replacement_raw: bytes) -> None:
    parked_name = f"parked-{opened.name}"
    os.rename(
        opened.name,
        parked_name,
        src_dir_fd=opened.parent_fd,
        dst_dir_fd=opened.parent_fd,
    )
    replacement_fd = os.open(
        opened.name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=opened.parent_fd,
    )
    try:
        os.write(replacement_fd, replacement_raw)
    finally:
        os.close(replacement_fd)


@pytest.mark.parametrize("force_link_fallback", (False, True))
def test_publication_outer_reauthentication_rejects_source_substitution(
    tmp_path: Path, monkeypatch, force_link_fallback: bool
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"
    replacement_raw = b"attacker-source-before-publication"
    cleanup_called = False

    def substitute(opened) -> None:
        _substitute_private_source(opened, replacement_raw)

    def reject_cleanup(_opened) -> None:
        nonlocal cleanup_called
        cleanup_called = True
        raise AssertionError("source-authentication failure must bypass cleanup")

    monkeypatch.setattr(ledger_module, "_before_private_publication", substitute)
    monkeypatch.setattr(ledger_module, "_cleanup_private_ledger", reject_cleanup)
    if force_link_fallback:
        monkeypatch.setattr(
            ledger_module,
            "_renameatx_np_noreplace",
            lambda _opened: False,
        )
        monkeypatch.setattr(
            ledger_module,
            "_renameat2_noreplace",
            lambda _opened: False,
        )

    with pytest.raises(ledger_module.LedgerPublicationSourceError) as raised:
        append_event(
            path,
            event_kind="one",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
        )

    message = str(raised.value).lower()
    assert "source" in message and "publication" in message
    assert "complete" not in message and "uncertain" not in message
    assert cleanup_called is False
    assert not path.exists()


@pytest.mark.parametrize("force_link_fallback", (False, True))
def test_syscall_window_source_substitution_is_classified_corrupt(
    tmp_path: Path, monkeypatch, force_link_fallback: bool
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"
    replacement_raw = b"attacker-source-inside-syscall-window"
    source_path: Path | None = None
    remove_staging_called = False
    real_remove_staging = ledger_module._remove_staging_directory

    def substitute(opened) -> None:
        nonlocal source_path
        assert opened.staging_name is not None
        source_path = tmp_path / opened.staging_name / opened.name
        _substitute_private_source(opened, replacement_raw)

    def track_remove_staging(opened) -> None:
        nonlocal remove_staging_called
        remove_staging_called = True
        real_remove_staging(opened)

    monkeypatch.setattr(
        ledger_module,
        "_remove_staging_directory",
        track_remove_staging,
    )

    if force_link_fallback:
        monkeypatch.setattr(
            ledger_module,
            "_renameatx_np_noreplace",
            lambda _opened: False,
        )
        monkeypatch.setattr(
            ledger_module,
            "_renameat2_noreplace",
            lambda _opened: False,
        )
        monkeypatch.setattr(
            ledger_module,
            "_after_link_source_reauthentication",
            substitute,
            raising=False,
        )
    else:
        monkeypatch.setattr(
            ledger_module,
            "_after_native_source_reauthentication",
            substitute,
            raising=False,
        )

    with pytest.raises(ledger_module.LedgerCorruptPublicationError) as raised:
        append_event(
            path,
            event_kind="one",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
        )

    message = str(raised.value).lower()
    assert "corrupt" in message or "unverified" in message
    assert "complete" not in message and "uncertain" not in message
    assert path.exists()
    assert path.read_bytes() == replacement_raw
    assert remove_staging_called is False
    if force_link_fallback:
        assert source_path is not None
        assert source_path.read_bytes() == replacement_raw
    with pytest.raises(LedgerError):
        load_ledger(path)


@pytest.mark.parametrize("force_link_fallback", (False, True))
def test_syscall_window_same_inode_content_mutation_is_classified_corrupt(
    tmp_path: Path, monkeypatch, force_link_fallback: bool
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"
    replacement_raw = b"same-inode-attacker-content"
    source_path: Path | None = None
    remove_staging_called = False
    real_remove_staging = ledger_module._remove_staging_directory

    def mutate_same_inode(opened) -> None:
        nonlocal source_path
        assert opened.staging_name is not None
        source_path = tmp_path / opened.staging_name / opened.name
        replacement_fd = os.open(
            opened.name,
            os.O_WRONLY | os.O_TRUNC,
            dir_fd=opened.parent_fd,
        )
        try:
            os.write(replacement_fd, replacement_raw)
        finally:
            os.close(replacement_fd)

    def track_remove_staging(opened) -> None:
        nonlocal remove_staging_called
        remove_staging_called = True
        real_remove_staging(opened)

    monkeypatch.setattr(
        ledger_module,
        "_remove_staging_directory",
        track_remove_staging,
    )
    if force_link_fallback:
        monkeypatch.setattr(
            ledger_module,
            "_renameatx_np_noreplace",
            lambda _opened: False,
        )
        monkeypatch.setattr(
            ledger_module,
            "_renameat2_noreplace",
            lambda _opened: False,
        )
        monkeypatch.setattr(
            ledger_module,
            "_after_link_source_reauthentication",
            mutate_same_inode,
        )
    else:
        monkeypatch.setattr(
            ledger_module,
            "_after_native_source_reauthentication",
            mutate_same_inode,
        )

    with pytest.raises(ledger_module.LedgerCorruptPublicationError) as raised:
        append_event(
            path,
            event_kind="one",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
        )

    message = str(raised.value).lower()
    assert "content" in message or "bytes" in message
    assert "complete" not in message and "uncertain" not in message
    assert path.read_bytes() == replacement_raw
    assert remove_staging_called is False
    if force_link_fallback:
        assert source_path is not None
        assert source_path.read_bytes() == replacement_raw
    with pytest.raises(LedgerError):
        load_ledger(path)


def test_final_post_durability_verification_rejects_same_inode_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"
    replacement_raw = b"post-publication-same-inode-content"
    real_fsync = ledger_module.os.fsync
    mutated = False

    def mutate_before_directory_fsync(fd: int) -> None:
        nonlocal mutated
        if stat.S_ISDIR(os.fstat(fd).st_mode) and not mutated:
            replacement_fd = os.open(path, os.O_WRONLY | os.O_TRUNC)
            try:
                os.write(replacement_fd, replacement_raw)
            finally:
                os.close(replacement_fd)
            mutated = True
        real_fsync(fd)

    monkeypatch.setattr(ledger_module.os, "fsync", mutate_before_directory_fsync)
    with pytest.raises(ledger_module.LedgerCorruptPublicationError) as raised:
        append_event(
            path,
            event_kind="one",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
        )

    assert mutated is True
    assert "complete" not in str(raised.value).lower()
    assert path.read_bytes() == replacement_raw
    with pytest.raises(LedgerError):
        load_ledger(path)


def test_publication_race_retries_fresh_without_overwriting_winner(
    tmp_path: Path, monkeypatch
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"
    winner_raw = _encoded_event(experiment="race-winner")
    invoked = False

    def publish_winner_once(_opened) -> None:
        nonlocal invoked
        if not invoked:
            invoked = True
            path.write_bytes(winner_raw)

    monkeypatch.setattr(
        ledger_module, "_before_private_publication", publish_winner_once
    )
    monkeypatch.setattr(ledger_module, "_renameatx_np_noreplace", lambda _opened: False)
    monkeypatch.setattr(ledger_module, "_renameat2_noreplace", lambda _opened: False)
    appended = append_event(
        path,
        event_kind="race-loser-retried",
        experiment="race-loser",
        payload={},
        timestamp="2026-07-11T12:35:00Z",
    )

    assert path.read_bytes().startswith(winner_raw)
    loaded = load_ledger(path)
    assert len(loaded) == 2
    assert loaded[0].experiment == "race-winner"
    assert loaded[1] == appended
    assert tuple(item.name for item in tmp_path.iterdir()) == (path.name,)


def test_exclusive_link_fallback_cleanup_failure_leaves_valid_uncertain_commit(
    tmp_path: Path, monkeypatch
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"
    monkeypatch.setattr(ledger_module, "_renameatx_np_noreplace", lambda _opened: False)
    monkeypatch.setattr(ledger_module, "_renameat2_noreplace", lambda _opened: False)

    def fail_private_unlink(*_args, **_kwargs) -> None:
        raise OSError("injected private-link cleanup failure")

    monkeypatch.setattr(ledger_module.os, "unlink", fail_private_unlink)
    with pytest.raises(LedgerError, match="uncertain committed state"):
        append_event(
            path,
            event_kind="one",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
        )

    assert len(load_ledger(path)) == 1
    assert path.exists()
    assert len(tuple(tmp_path.iterdir())) == 2


def test_private_cleanup_fsync_failure_reports_unproven_rollback_without_public_leaf(
    tmp_path: Path, monkeypatch
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"
    real_fsync = ledger_module.os.fsync

    def fail_directory_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("injected cleanup directory fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(ledger_module.os, "fsync", fail_directory_fsync)
    with pytest.raises(LedgerError, match="rollback|cleanup|durab"):
        append_event(
            path,
            event_kind="one",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
            expected_event_count=1,
        )

    assert not path.exists()


def test_private_cleanup_unlink_failure_never_creates_public_leaf(
    tmp_path: Path, monkeypatch
) -> None:
    import mlx_vq.recovery_campaign.ledger as ledger_module

    path = tmp_path / "campaign.jsonl"

    def fail_unlink(*_args, **_kwargs) -> None:
        raise OSError("injected cleanup unlink failure")

    monkeypatch.setattr(ledger_module.os, "unlink", fail_unlink)
    with pytest.raises(LedgerError, match="rollback|cleanup|durab"):
        append_event(
            path,
            event_kind="one",
            experiment="full75-e8",
            payload={},
            timestamp=NOW,
            expected_event_count=1,
        )

    assert not path.exists()
    private_names = [item.name for item in tmp_path.iterdir()]
    assert len(private_names) == 1
    assert private_names[0] != path.name


def test_append_evidence_accepts_anchors_under_the_append_transaction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.jsonl"
    first = append_event(
        path,
        event_kind="one",
        experiment="full75-e8",
        payload={},
        timestamp=NOW,
    )

    event = append_evidence(
        path,
        experiment="full75-e8",
        evidence=_evidence(),
        timestamp="2026-07-11T12:35:00Z",
        expected_head_sha256=first.event_sha256,
        expected_event_count=1,
    )

    assert event.sequence == 2


def test_replace_still_validates_evidence_invariants() -> None:
    with pytest.raises(ValueError, match="diagnostic_only|release_eligible"):
        replace(_evidence(), evidence_class="diagnostic_only")
