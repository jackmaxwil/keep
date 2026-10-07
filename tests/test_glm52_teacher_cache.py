from __future__ import annotations

import fcntl
import json
import os
import stat
import struct
import importlib.util
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/mlx_vq/quality/glm52_teacher_cache.py"
)
SPEC = importlib.util.spec_from_file_location("glm52_teacher_cache_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
cache_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cache_module
SPEC.loader.exec_module(cache_module)

GLM52ProducerPhase = cache_module.GLM52ProducerPhase
GLM52TeacherCacheContract = cache_module.GLM52TeacherCacheContract
GLM52TeacherCachePrompt = cache_module.GLM52TeacherCachePrompt
audit_glm52_teacher_cache = cache_module.audit_glm52_teacher_cache
build_glm52_teacher_cache_manifest = cache_module.build_glm52_teacher_cache_manifest
compute_cache_content_sha256 = cache_module.compute_cache_content_sha256
compute_manifest_body_sha256 = cache_module.compute_manifest_body_sha256
publish_glm52_teacher_cache_manifest = cache_module.publish_glm52_teacher_cache_manifest
write_glm52_teacher_cache_shard = cache_module.write_glm52_teacher_cache_shard


def _prompts() -> tuple[GLM52TeacherCachePrompt, ...]:
    return (
        GLM52TeacherCachePrompt(
            prompt_id="tiny_report_route_000",
            split="report",
            domain="route",
            tuning_eligible=False,
            encoded_token_ids=(1, 2, 3),
        ),
        GLM52TeacherCachePrompt(
            prompt_id="tiny_selection_math_000",
            split="selection",
            domain="math",
            tuning_eligible=True,
            encoded_token_ids=(4, 5, 6, 7),
        ),
        GLM52TeacherCachePrompt(
            prompt_id="tiny_holdout_instruction_000",
            split="holdout",
            domain="instruction",
            tuning_eligible=False,
            encoded_token_ids=(8, 9),
        ),
    )


@pytest.fixture
def contract() -> GLM52TeacherCacheContract:
    return GLM52TeacherCacheContract.for_testing(
        prompts=_prompts(),
        vocab_size=32,
    )


def _phase(
    *, pageouts_delta: int | None = 0, swapouts_delta: int | None = 0
) -> GLM52ProducerPhase:
    return GLM52ProducerPhase.create(
        phase_id="phase-000",
        ordinal=0,
        start_boundary="embedded-input",
        end_boundary="all-logits-written",
        contribution_range="layers-00000-00077-and-lm-head",
        pageouts_delta=pageouts_delta,
        swapouts_delta=swapouts_delta,
        system_wired_default=True,
        input_identity_sha256="1" * 64,
        output_identity_sha256="2" * 64,
    )


def _logits(prompt: GLM52TeacherCachePrompt, vocab_size: int) -> np.ndarray:
    rows = prompt.token_count - 1
    values = np.arange(rows * vocab_size, dtype=np.float32).reshape(rows, vocab_size)
    return values + np.float32(prompt.encoded_token_ids[0] / 10)


def _build_cache(
    root: Path,
    contract: GLM52TeacherCacheContract,
    *,
    phase: GLM52ProducerPhase | None = None,
) -> tuple[object, dict[str, dict[str, object]]]:
    phase = phase or _phase()
    ledger_path = _ledger_path(root)
    shards = []
    ledgers: dict[str, dict[str, object]] = {}
    for prompt in contract.prompts:
        result = write_glm52_teacher_cache_shard(
            root,
            ledger_path=ledger_path,
            prompt=prompt,
            logits=_logits(prompt, contract.vocab_size),
            vocab_size=contract.vocab_size,
            producer_phase_ids=(phase.phase_id,),
            bound_identity_sha256=contract.bound_identity_sha256,
        )
        shards.append(result.shard)
        ledgers[prompt.prompt_id] = result.ledger_record
    manifest = build_glm52_teacher_cache_manifest(
        contract=contract,
        shards=shards,
        producer_phases=(phase,),
        created_at="2026-07-10T00:00:00Z",
    )
    publish_glm52_teacher_cache_manifest(
        root,
        manifest,
        ledger_path=ledger_path,
        bound_identity_sha256=contract.bound_identity_sha256,
    )
    return manifest, ledgers


def _load_manifest(root: Path) -> dict[str, object]:
    return json.loads(
        (root / "glm52-teacher-cache-fp32-manifest.json").read_text()
    )


def _resign_manifest(payload: dict[str, object]) -> None:
    payload["cache_content_sha256"] = compute_cache_content_sha256(payload)
    payload["manifest_body_sha256"] = compute_manifest_body_sha256(payload)


def _write_manifest_payload(root: Path, payload: dict[str, object]) -> None:
    (root / "glm52-teacher-cache-fp32-manifest.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _first_shard(root: Path) -> Path:
    return root / "teacher_logits" / "tiny_report_route_000.safetensors"


def _ledger_path(root: Path) -> Path:
    return root.parent / f"{root.name}-teacher-cache-ledger.jsonl"


def _rewrite_safetensors_header(path: Path, header_text: str, payload: bytes) -> None:
    header = header_text.encode("utf-8")
    padding = (-len(header)) % 8
    header += b" " * padding
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def _raw_payload(path: Path) -> bytes:
    raw = path.read_bytes()
    header_size = struct.unpack("<Q", raw[:8])[0]
    return raw[8 + header_size :]


def test_happy_path_write_and_strict_audit_round_trip(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    manifest, _ = _build_cache(tmp_path, contract)

    report = audit_glm52_teacher_cache(tmp_path, contract=contract)

    assert report.valid is True
    assert report.release_eligible is True
    assert report.cache_content_sha256 == manifest.cache_content_sha256
    assert report.prompt_count == 3
    assert report.predictor_position_count == 6
    assert report.fp32_value_count == 192
    assert report.raw_tensor_bytes == 768


def test_frozen_production_contract_has_exact_66_prompt_identities(
    contract: GLM52TeacherCacheContract,
) -> None:
    prompt_pack = (
        Path(__file__).resolve().parents[1]
        / "artifacts/quality/glm52-family-eval-prompts-20260709-v2.json"
    )

    production = GLM52TeacherCacheContract.from_frozen_prompt_pack(
        prompt_pack,
        producer=contract.producer,
        source_evidence=contract.source_evidence,
        non_vq_package=contract.non_vq_package,
    )

    assert len(production.prompts) == 66
    assert production.source_token_count == 810
    assert production.predictor_position_count == 744
    assert production.fp32_value_count == 115_230_720
    assert production.raw_tensor_bytes == 460_922_880
    assert production.totals()["splits"] == {
        "holdout": {"position_count": 251, "raw_tensor_bytes": 155_499_520},
        "report": {"position_count": 238, "raw_tensor_bytes": 147_445_760},
        "selection": {"position_count": 255, "raw_tensor_bytes": 157_977_600},
    }


def test_valid_existing_shard_is_reused_after_restart_from_durable_ledger(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    prompt = contract.prompts[0]
    first = write_glm52_teacher_cache_shard(
        tmp_path,
        ledger_path=_ledger_path(tmp_path),
        prompt=prompt,
        logits=_logits(prompt, contract.vocab_size),
        vocab_size=contract.vocab_size,
        producer_phase_ids=("phase-000",),
        bound_identity_sha256=contract.bound_identity_sha256,
    )
    inode = _first_shard(tmp_path).stat().st_ino

    reused = write_glm52_teacher_cache_shard(
        tmp_path,
        ledger_path=_ledger_path(tmp_path),
        prompt=prompt,
        logits=_logits(prompt, contract.vocab_size),
        vocab_size=contract.vocab_size,
        producer_phase_ids=("phase-000",),
        bound_identity_sha256=contract.bound_identity_sha256,
    )

    assert reused.reused is True
    assert _first_shard(tmp_path).stat().st_ino == inode
    assert reused.shard == first.shard


def test_two_synchronized_publishers_racing_same_shard_have_exactly_one_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "same-name.safetensors"
    barrier = threading.Barrier(2)
    real_link = os.link

    def synchronized_link(*args: object, **kwargs: object) -> None:
        barrier.wait(timeout=5)
        real_link(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cache_module.os, "link", synchronized_link)

    def publish(payload: bytes) -> tuple[bytes, BaseException | None]:
        try:
            cache_module._durable_publish_bytes(final_path, payload)
        except BaseException as error:
            return payload, error
        return payload, None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (b"publisher-a", b"publisher-b")))

    winners = [outcome for outcome in outcomes if outcome[1] is None]
    losers = [outcome for outcome in outcomes if outcome[1] is not None]
    assert len(winners) == 1
    assert len(losers) == 1
    assert isinstance(losers[0][1], ValueError)
    assert "refusing to overwrite concurrently published final file" in str(losers[0][1])
    assert final_path.read_bytes() == winners[0][0]


def test_different_shard_publishers_serialize_one_valid_ledger_chain(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    cache_root = tmp_path / "cache"
    ledger_path = _ledger_path(cache_root)
    lock_path = Path(f"{ledger_path}.lock")
    lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
    started = threading.Barrier(3)

    def publish(prompt: GLM52TeacherCachePrompt) -> object:
        started.wait(timeout=5)
        return write_glm52_teacher_cache_shard(
            cache_root,
            ledger_path=ledger_path,
            prompt=prompt,
            logits=_logits(prompt, contract.vocab_size),
            vocab_size=contract.vocab_size,
            producer_phase_ids=("phase-000",),
            bound_identity_sha256=contract.bound_identity_sha256,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(publish, prompt) for prompt in contract.prompts[:2]]
            started.wait(timeout=5)
            threading.Event().wait(0.05)
            assert all(not future.done() for future in futures)
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            results = [future.result(timeout=5) for future in futures]
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)

    assert len(results) == 2
    records = cache_module._read_shard_ledger(
        ledger_path,
        bound_identity_sha256=contract.bound_identity_sha256,
    )
    assert len(records) == 2
    assert {record["shard"]["prompt_id"] for record in records} == {
        prompt.prompt_id for prompt in contract.prompts[:2]
    }


def test_shard_publication_orders_fsync_before_durable_ledger_append(
    tmp_path: Path,
    contract: GLM52TeacherCacheContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    ledger_path = _ledger_path(tmp_path)
    logits_dir = tmp_path / "teacher_logits"
    real_fsync = os.fsync
    real_link = os.link
    real_unlink = os.unlink
    real_write = os.write

    def recording_fsync(descriptor: int) -> None:
        descriptor_stat = os.fstat(descriptor)
        if stat.S_ISDIR(descriptor_stat.st_mode):
            directory_event = (
                "shard_directory_fsync"
                if logits_dir.exists()
                and descriptor_stat.st_ino == logits_dir.stat().st_ino
                else "ledger_directory_fsync"
            )
            events.append(directory_event)
        elif "ledger_append" in events:
            events.append("ledger_file_fsync")
        else:
            events.append("temp_file_fsync")
        real_fsync(descriptor)

    def recording_link(*args: object, **kwargs: object) -> None:
        events.append("link_final")
        real_link(*args, **kwargs)  # type: ignore[arg-type]

    def recording_unlink(*args: object, **kwargs: object) -> None:
        events.append("unlink_temp")
        real_unlink(*args, **kwargs)  # type: ignore[arg-type]

    def recording_write(descriptor: int, payload: bytes) -> int:
        events.append("ledger_append")
        return real_write(descriptor, payload)

    monkeypatch.setattr(cache_module.os, "fsync", recording_fsync)
    monkeypatch.setattr(cache_module.os, "link", recording_link)
    monkeypatch.setattr(cache_module.os, "unlink", recording_unlink)
    monkeypatch.setattr(cache_module.os, "write", recording_write)

    prompt = contract.prompts[0]
    write_glm52_teacher_cache_shard(
        tmp_path,
        ledger_path=ledger_path,
        prompt=prompt,
        logits=_logits(prompt, contract.vocab_size),
        vocab_size=contract.vocab_size,
        producer_phase_ids=("phase-000",),
        bound_identity_sha256=contract.bound_identity_sha256,
    )

    assert events == [
        "temp_file_fsync",
        "link_final",
        "unlink_temp",
        "shard_directory_fsync",
        "ledger_append",
        "ledger_file_fsync",
        "ledger_directory_fsync",
    ]


def test_tampered_durable_ledger_line_fails_loudly(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    prompt = contract.prompts[0]
    ledger_path = _ledger_path(tmp_path)
    write_glm52_teacher_cache_shard(
        tmp_path,
        ledger_path=ledger_path,
        prompt=prompt,
        logits=_logits(prompt, contract.vocab_size),
        vocab_size=contract.vocab_size,
        producer_phase_ids=("phase-000",),
        bound_identity_sha256=contract.bound_identity_sha256,
    )
    record = json.loads(ledger_path.read_text())
    record["record_sha256"] = "f" * 64
    ledger_path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(ValueError, match="ledger.*SHA-256 mismatch"):
        write_glm52_teacher_cache_shard(
            tmp_path,
            ledger_path=ledger_path,
            prompt=prompt,
            logits=_logits(prompt, contract.vocab_size),
            vocab_size=contract.vocab_size,
            producer_phase_ids=("phase-000",),
            bound_identity_sha256=contract.bound_identity_sha256,
        )


def test_existing_shard_without_ledger_fails_and_is_never_overwritten(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    prompt = contract.prompts[0]
    write_glm52_teacher_cache_shard(
        tmp_path,
        ledger_path=_ledger_path(tmp_path),
        prompt=prompt,
        logits=_logits(prompt, contract.vocab_size),
        vocab_size=contract.vocab_size,
        producer_phase_ids=("phase-000",),
        bound_identity_sha256=contract.bound_identity_sha256,
    )
    before = _first_shard(tmp_path).read_bytes()
    ledger_path = _ledger_path(tmp_path)
    ledger_path.unlink()

    with pytest.raises(ValueError, match="existing shard.*ledger"):
        write_glm52_teacher_cache_shard(
            tmp_path,
            prompt=prompt,
            logits=_logits(prompt, contract.vocab_size),
            vocab_size=contract.vocab_size,
            producer_phase_ids=("phase-000",),
            bound_identity_sha256=contract.bound_identity_sha256,
            ledger_path=ledger_path,
        )

    assert _first_shard(tmp_path).read_bytes() == before


def test_invalid_existing_shard_fails_and_is_never_overwritten(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    prompt = contract.prompts[0]
    write_glm52_teacher_cache_shard(
        tmp_path,
        ledger_path=_ledger_path(tmp_path),
        prompt=prompt,
        logits=_logits(prompt, contract.vocab_size),
        vocab_size=contract.vocab_size,
        producer_phase_ids=("phase-000",),
        bound_identity_sha256=contract.bound_identity_sha256,
    )
    path = _first_shard(tmp_path)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="existing shard.*invalid"):
        write_glm52_teacher_cache_shard(
            tmp_path,
            ledger_path=_ledger_path(tmp_path),
            prompt=prompt,
            logits=_logits(prompt, contract.vocab_size),
            vocab_size=contract.vocab_size,
            producer_phase_ids=("phase-000",),
            bound_identity_sha256=contract.bound_identity_sha256,
        )

    assert path.read_bytes() == bytes(raw)


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("/tmp/escape.safetensors", "absolute"),
        ("teacher_logits\\escape.safetensors", "backslash"),
        ("teacher_logits/../escape.safetensors", "dot component"),
        ("./teacher_logits/tiny_report_route_000.safetensors", "dot component"),
    ],
)
def test_auditor_rejects_untrusted_manifest_paths(
    tmp_path: Path,
    contract: GLM52TeacherCacheContract,
    relative_path: str,
    message: str,
) -> None:
    _build_cache(tmp_path, contract)
    payload = _load_manifest(tmp_path)
    payload["shards"][0]["relative_path"] = relative_path  # type: ignore[index]
    _resign_manifest(payload)
    _write_manifest_payload(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        audit_glm52_teacher_cache(tmp_path, contract=contract)


@pytest.mark.parametrize("kind", ["extra", "missing", "fifo"])
def test_auditor_rejects_extra_missing_and_special_files(
    tmp_path: Path,
    contract: GLM52TeacherCacheContract,
    kind: str,
) -> None:
    _build_cache(tmp_path, contract)
    if kind == "extra":
        (tmp_path / "extra.txt").write_text("unexpected")
    elif kind == "missing":
        _first_shard(tmp_path).unlink()
    else:
        os.mkfifo(tmp_path / "teacher_logits" / "special.safetensors")

    with pytest.raises(ValueError, match="extra|missing|special"):
        audit_glm52_teacher_cache(tmp_path, contract=contract)


def test_auditor_rejects_symlinked_root_file_and_directory(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    real = tmp_path / "real"
    _build_cache(real, contract)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="root.*symlink"):
        audit_glm52_teacher_cache(root_link, contract=contract)

    shard = _first_shard(real)
    moved = tmp_path / "moved.safetensors"
    shard.rename(moved)
    shard.symlink_to(moved)
    with pytest.raises(ValueError, match="symlink"):
        audit_glm52_teacher_cache(real, contract=contract)

    shard.unlink()
    moved.rename(shard)
    logits = real / "teacher_logits"
    moved_dir = tmp_path / "moved-logits"
    logits.rename(moved_dir)
    logits.symlink_to(moved_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="directory.*symlink|symlink.*directory"):
        audit_glm52_teacher_cache(real, contract=contract)


def test_auditor_rejects_duplicate_manifest_key_and_nonfinite_json(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    _build_cache(tmp_path, contract)
    path = tmp_path / "glm52-teacher-cache-fp32-manifest.json"
    raw = path.read_text()
    path.write_text('{"schema_version":1,' + raw[1:])
    with pytest.raises(ValueError, match="duplicate JSON key"):
        audit_glm52_teacher_cache(tmp_path, contract=contract)

    _build_cache(tmp_path / "nan", contract)
    nan_path = tmp_path / "nan" / "glm52-teacher-cache-fp32-manifest.json"
    raw = nan_path.read_text()
    nan_path.write_text(raw.replace('"created_at":"2026', '"created_at":NaN,"ignored":"2026', 1))
    with pytest.raises(ValueError, match="non-finite JSON"):
        audit_glm52_teacher_cache(tmp_path / "nan", contract=contract)


def test_auditor_rejects_duplicate_and_nonfinite_safetensors_header(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    _build_cache(tmp_path, contract)
    path = _first_shard(tmp_path)
    payload = _raw_payload(path)
    descriptor = '{"dtype":"F32","shape":[2,32],"data_offsets":[0,256]}'
    _rewrite_safetensors_header(
        path,
        '{"logits":' + descriptor + ',"logits":' + descriptor + "}",
        payload,
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        audit_glm52_teacher_cache(tmp_path, contract=contract)

    other = tmp_path / "nan-header"
    _build_cache(other, contract)
    path = _first_shard(other)
    _rewrite_safetensors_header(
        path,
        '{"logits":{"dtype":"F32","shape":[2,32],"data_offsets":[0,256],"x":NaN}}',
        _raw_payload(path),
    )
    with pytest.raises(ValueError, match="non-finite JSON"):
        audit_glm52_teacher_cache(other, contract=contract)


@pytest.mark.parametrize(
    ("header", "message"),
    [
        (
            '{"logits":{"dtype":"F32","shape":[2,32],"data_offsets":[0,256]},'
            '"other":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
            "tensor inventory",
        ),
        (
            '{"__metadata__":{"producer":"untrusted"},'
            '"logits":{"dtype":"F32","shape":[2,32],"data_offsets":[0,256]}}',
            "tensor inventory",
        ),
        (
            '{"logits":{"dtype":"F32","shape":[1,64],"data_offsets":[0,256]}}',
            "shape",
        ),
        (
            '{"logits":{"dtype":"F32","shape":[2,32],"data_offsets":[4,260]}}',
            "offsets",
        ),
    ],
)
def test_auditor_rejects_tensor_inventory_metadata_shape_and_offsets(
    tmp_path: Path,
    contract: GLM52TeacherCacheContract,
    header: str,
    message: str,
) -> None:
    _build_cache(tmp_path, contract)
    path = _first_shard(tmp_path)
    _rewrite_safetensors_header(path, header, _raw_payload(path))

    with pytest.raises(ValueError, match=message):
        audit_glm52_teacher_cache(tmp_path, contract=contract)


@pytest.mark.parametrize("mutation", ["same-size-byte", "wrong-dtype", "trailing"])
def test_auditor_rejects_tampered_dtype_and_physical_extent(
    tmp_path: Path,
    contract: GLM52TeacherCacheContract,
    mutation: str,
) -> None:
    _build_cache(tmp_path, contract)
    path = _first_shard(tmp_path)
    if mutation == "same-size-byte":
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        match = "SHA-256"
    elif mutation == "wrong-dtype":
        raw = path.read_bytes().replace(b'"F32"', b'"F16"', 1)
        path.write_bytes(raw)
        match = "dtype"
    else:
        path.write_bytes(path.read_bytes() + b"trailing")
        match = "physical extent|trailing"

    with pytest.raises(ValueError, match=match):
        audit_glm52_teacher_cache(tmp_path, contract=contract)


def test_auditor_rejects_constant_causal_row_before_trusting_hashes(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    _build_cache(tmp_path, contract)
    path = _first_shard(tmp_path)
    raw = bytearray(path.read_bytes())
    header_size = struct.unpack("<Q", raw[:8])[0]
    start = 8 + header_size
    raw[start : start + contract.vocab_size * 4] = np.zeros(
        contract.vocab_size, dtype="<f4"
    ).tobytes()
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="constant causal row"):
        audit_glm52_teacher_cache(tmp_path, contract=contract)


def test_auditor_rejects_nonfinite_tensor_value(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    _build_cache(tmp_path, contract)
    path = _first_shard(tmp_path)
    raw = bytearray(path.read_bytes())
    header_size = struct.unpack("<Q", raw[:8])[0]
    start = 8 + header_size
    raw[start : start + 4] = np.array([np.nan], dtype="<f4").tobytes()
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="non-finite value"):
        audit_glm52_teacher_cache(tmp_path, contract=contract)


def test_auditor_rejects_wrong_manifest_hash_and_unknown_schema_field(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    _build_cache(tmp_path, contract)
    payload = _load_manifest(tmp_path)
    payload["shards"][0]["file_sha256"] = "f" * 64  # type: ignore[index]
    _resign_manifest(payload)
    _write_manifest_payload(tmp_path, payload)
    with pytest.raises(ValueError, match="file SHA-256"):
        audit_glm52_teacher_cache(tmp_path, contract=contract)

    other = tmp_path / "schema"
    _build_cache(other, contract)
    payload = _load_manifest(other)
    payload["precision"]["surprise"] = True  # type: ignore[index]
    _resign_manifest(payload)
    _write_manifest_payload(other, payload)
    with pytest.raises(ValueError, match="precision fields"):
        audit_glm52_teacher_cache(other, contract=contract)


def test_auditor_rejects_prompt_order_and_token_identity_mismatch(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    _build_cache(tmp_path, contract)
    payload = _load_manifest(tmp_path)
    payload["shards"][0]["token_ids_sha256"] = "e" * 64  # type: ignore[index]
    _resign_manifest(payload)
    _write_manifest_payload(tmp_path, payload)
    with pytest.raises(ValueError, match="token_ids_sha256"):
        audit_glm52_teacher_cache(tmp_path, contract=contract)

    other = tmp_path / "order"
    _build_cache(other, contract)
    payload = _load_manifest(other)
    payload["shards"][0], payload["shards"][1] = (  # type: ignore[index]
        payload["shards"][1],
        payload["shards"][0],
    )
    _resign_manifest(payload)
    _write_manifest_payload(other, payload)
    with pytest.raises(ValueError, match="prompt_id"):
        audit_glm52_teacher_cache(other, contract=contract)


def test_auditor_independently_recomputes_both_stable_identities(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    _build_cache(tmp_path, contract)
    payload = _load_manifest(tmp_path)
    payload["cache_content_sha256"] = "f" * 64
    payload["manifest_body_sha256"] = compute_manifest_body_sha256(payload)
    _write_manifest_payload(tmp_path, payload)
    with pytest.raises(ValueError, match="cache content SHA-256"):
        audit_glm52_teacher_cache(tmp_path, contract=contract)

    other = tmp_path / "body"
    _build_cache(other, contract)
    payload = _load_manifest(other)
    payload["manifest_body_sha256"] = "f" * 64
    _write_manifest_payload(other, payload)
    with pytest.raises(ValueError, match="manifest body SHA-256"):
        audit_glm52_teacher_cache(other, contract=contract)


def test_auditor_reconciles_rows_splits_and_global_totals(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    _build_cache(tmp_path, contract)
    payload = _load_manifest(tmp_path)
    payload["totals"]["splits"]["report"]["position_count"] += 1  # type: ignore[index]
    _resign_manifest(payload)
    _write_manifest_payload(tmp_path, payload)

    with pytest.raises(ValueError, match="split.*position_count|totals"):
        audit_glm52_teacher_cache(tmp_path, contract=contract)


def test_unknown_memory_counters_remain_unknown_and_block_release(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    manifest, _ = _build_cache(
        tmp_path,
        contract,
        phase=_phase(pageouts_delta=None, swapouts_delta=None),
    )

    report = audit_glm52_teacher_cache(tmp_path, contract=contract)

    assert manifest.all_producer_memory_counters_known is False
    assert manifest.all_producer_memory_clean is False
    assert manifest.release_eligible is False
    assert report.valid is True
    assert report.release_eligible is False


def test_known_nonzero_memory_counter_preserves_payload_but_blocks_release(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    manifest, _ = _build_cache(
        tmp_path,
        contract,
        phase=_phase(pageouts_delta=1, swapouts_delta=0),
    )

    report = audit_glm52_teacher_cache(tmp_path, contract=contract)

    assert manifest.payload_integrity_pass is True
    assert manifest.all_producer_memory_counters_known is True
    assert manifest.all_producer_memory_clean is False
    assert manifest.release_eligible is False
    assert report.valid is True
    assert report.release_eligible is False


@pytest.mark.parametrize("counter", [False, True])
def test_boolean_memory_counter_is_rejected_not_coerced_to_zero(
    counter: bool,
) -> None:
    with pytest.raises(ValueError, match="pageouts_delta.*integer or null"):
        _phase(pageouts_delta=counter)  # type: ignore[arg-type]


def test_mutation_during_audit_is_detected_by_file_identity_check(
    tmp_path: Path,
    contract: GLM52TeacherCacheContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_cache(tmp_path, contract)
    original = cache_module._assert_file_identity_unchanged
    mutated = False

    def mutate_then_check(path: Path, before: object, after: object) -> None:
        nonlocal mutated
        if path == _first_shard(tmp_path) and not mutated:
            mutated = True
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        original(path, before, after)

    monkeypatch.setattr(
        cache_module, "_assert_file_identity_unchanged", mutate_then_check
    )

    with pytest.raises(ValueError, match="changed during audit"):
        audit_glm52_teacher_cache(tmp_path, contract=contract)


def test_manifest_is_reverified_after_traversal(
    tmp_path: Path,
    contract: GLM52TeacherCacheContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_cache(tmp_path, contract)
    original = cache_module._audit_shard
    mutated = False

    def mutate_manifest_once(*args: object, **kwargs: object) -> object:
        nonlocal mutated
        result = original(*args, **kwargs)
        if not mutated:
            mutated = True
            path = tmp_path / "glm52-teacher-cache-fp32-manifest.json"
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        return result

    monkeypatch.setattr(cache_module, "_audit_shard", mutate_manifest_once)

    with pytest.raises(ValueError, match="manifest changed during audit"):
        audit_glm52_teacher_cache(tmp_path, contract=contract)


def test_orphan_temp_file_is_never_adopted_or_a_completion_marker(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    logits_dir = tmp_path / "teacher_logits"
    logits_dir.mkdir(parents=True)
    orphan = logits_dir / ".tiny_report_route_000.safetensors.orphan.tmp"
    orphan.write_bytes(b"not a shard")
    prompt = contract.prompts[0]

    result = write_glm52_teacher_cache_shard(
        tmp_path,
        ledger_path=_ledger_path(tmp_path),
        prompt=prompt,
        logits=_logits(prompt, contract.vocab_size),
        vocab_size=contract.vocab_size,
        producer_phase_ids=("phase-000",),
        bound_identity_sha256=contract.bound_identity_sha256,
    )

    assert result.reused is False
    assert _first_shard(tmp_path).is_file()
    assert orphan.read_bytes() == b"not a shard"
    with pytest.raises(ValueError, match="missing|extra"):
        audit_glm52_teacher_cache(tmp_path, contract=contract)


def test_manifest_is_published_last_and_refuses_incomplete_tree(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    prompt = contract.prompts[0]
    shard = write_glm52_teacher_cache_shard(
        tmp_path,
        ledger_path=_ledger_path(tmp_path),
        prompt=prompt,
        logits=_logits(prompt, contract.vocab_size),
        vocab_size=contract.vocab_size,
        producer_phase_ids=("phase-000",),
        bound_identity_sha256=contract.bound_identity_sha256,
    ).shard
    with pytest.raises(ValueError, match="ordered shard inventory"):
        build_glm52_teacher_cache_manifest(
            contract=contract,
            shards=(shard,),
            producer_phases=(_phase(),),
            created_at="2026-07-10T00:00:00Z",
        )
    assert not (tmp_path / "glm52-teacher-cache-fp32-manifest.json").exists()


def test_manifest_publication_refuses_shard_without_durable_ledger_record(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    ledger_path = _ledger_path(tmp_path)
    phase = _phase()
    shards = []
    for prompt in contract.prompts:
        result = write_glm52_teacher_cache_shard(
            tmp_path,
            ledger_path=ledger_path,
            prompt=prompt,
            logits=_logits(prompt, contract.vocab_size),
            vocab_size=contract.vocab_size,
            producer_phase_ids=(phase.phase_id,),
            bound_identity_sha256=contract.bound_identity_sha256,
        )
        shards.append(result.shard)
    manifest = build_glm52_teacher_cache_manifest(
        contract=contract,
        shards=shards,
        producer_phases=(phase,),
        created_at="2026-07-10T00:00:00Z",
    )
    ledger_lines = ledger_path.read_bytes().splitlines(keepends=True)
    ledger_path.write_bytes(b"".join(ledger_lines[:-1]))

    with pytest.raises(ValueError, match="one-to-one"):
        publish_glm52_teacher_cache_manifest(
            tmp_path,
            manifest,
            ledger_path=ledger_path,
            bound_identity_sha256=contract.bound_identity_sha256,
        )
    assert not (tmp_path / "glm52-teacher-cache-fp32-manifest.json").exists()


def test_manifest_publication_rejects_ledger_bound_to_wrong_cache_identity(
    tmp_path: Path, contract: GLM52TeacherCacheContract
) -> None:
    ledger_path = _ledger_path(tmp_path)
    phase = _phase()
    shards = []
    wrong_identity = "f" * 64
    for prompt in contract.prompts:
        result = write_glm52_teacher_cache_shard(
            tmp_path,
            ledger_path=ledger_path,
            prompt=prompt,
            logits=_logits(prompt, contract.vocab_size),
            vocab_size=contract.vocab_size,
            producer_phase_ids=(phase.phase_id,),
            bound_identity_sha256=wrong_identity,
        )
        shards.append(result.shard)
    manifest = build_glm52_teacher_cache_manifest(
        contract=contract,
        shards=shards,
        producer_phases=(phase,),
        created_at="2026-07-10T00:00:00Z",
    )

    with pytest.raises(ValueError, match="bound identity"):
        publish_glm52_teacher_cache_manifest(
            tmp_path,
            manifest,
            ledger_path=ledger_path,
            bound_identity_sha256=contract.bound_identity_sha256,
        )
    assert not (tmp_path / "glm52-teacher-cache-fp32-manifest.json").exists()


def test_manifest_dataclass_is_immutable(contract: GLM52TeacherCacheContract) -> None:
    phase = _phase()
    prompt = contract.prompts[0]
    shard = cache_module.GLM52TeacherCacheShard.from_logits_bytes(
        prompt=prompt,
        vocab_size=contract.vocab_size,
        relative_path=f"teacher_logits/{prompt.prompt_id}.safetensors",
        file_bytes=b"file",
        raw_tensor_bytes=b"raw",
        producer_phase_ids=(phase.phase_id,),
    )
    assert replace(shard, dtype="F16").dtype == "F16"
    with pytest.raises(AttributeError):
        shard.dtype = "F16"  # type: ignore[misc]
