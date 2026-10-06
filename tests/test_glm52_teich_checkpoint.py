"""Durability contracts for resumable Teich teacher production."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from mlx_vq.quality.glm52_teich_checkpoint import (
    CheckpointValidationError,
    DeadlinePolicy,
    LocalLMHeadSliceStore,
    LocalTeacherCheckpointStore,
    TeacherCheckpointConfig,
    TeacherRunIdentity,
    should_checkpoint_after_layer,
)


def _sha(byte: str) -> str:
    return byte * 64


def _identity(*, code_sha256: str | None = None) -> TeacherRunIdentity:
    return TeacherRunIdentity(
        run_id="teich-v2-20260717",
        model_sha256=_sha("1"),
        non_vq_package_sha256=_sha("2"),
        prompt_pack_sha256=_sha("3"),
        code_sha256=code_sha256 or _sha("4"),
        generation_config_sha256=_sha("5"),
    )


def _checkpoint_payload(next_layer: int) -> dict[str, object]:
    return {
        "next_layer": next_layer,
        "hidden_bf16_bits": np.full((2, 5, 3), next_layer, dtype=np.uint16),
        "prev_topk_indices": np.full((2, 5, 2), next_layer, dtype=np.int32),
        "ordered_prompt_ids": ("prompt-a", "prompt-b"),
        "ordered_token_hashes": (_sha("a"), _sha("b")),
        "batch_shape": (2, 5),
        "valid_mask_sha256": _sha("c"),
        "generation_parameters": {"top_k": 2048, "lm_head_slice_size": 64},
    }


def test_checkpoint_config_defaults_and_deadline_validation(tmp_path) -> None:
    deadline = datetime.now(timezone.utc) + timedelta(hours=48)
    config = TeacherCheckpointConfig(
        local_checkpoint_dir=tmp_path / "checkpoints",
        s3_checkpoint_prefix="s3://bucket/runs/teich/checkpoints/",
        stop_file=tmp_path / "STOP",
        capacity_block_deadline=deadline,
        resume=True,
        identity=_identity(),
    )

    assert config.checkpoint_every_layers == 4
    assert config.capacity_block_deadline == deadline

    with pytest.raises(ValueError, match="timezone-aware"):
        TeacherCheckpointConfig(
            local_checkpoint_dir=tmp_path,
            capacity_block_deadline=datetime(2026, 7, 19, 0, 0),
            identity=_identity(),
        )


def test_checkpoint_config_accepts_exactly_one_generic_execution_deadline(
    tmp_path,
) -> None:
    deadline = datetime(2026, 7, 25, 8, 0, tzinfo=timezone.utc)
    config = TeacherCheckpointConfig(
        local_checkpoint_dir=tmp_path,
        execution_deadline=deadline,
        identity=_identity(),
    )
    assert config.effective_deadline == deadline
    with pytest.raises(ValueError, match="at most one"):
        TeacherCheckpointConfig(
            local_checkpoint_dir=tmp_path,
            capacity_block_deadline=deadline,
            execution_deadline=deadline,
            identity=_identity(),
        )


def test_forward_checkpoint_round_trip_restores_dsa_state_and_chain(tmp_path) -> None:
    store = LocalTeacherCheckpointStore(tmp_path, identity=_identity())
    first = store.publish_forward(**_checkpoint_payload(4))
    second = store.publish_forward(**_checkpoint_payload(8))

    loaded = store.load_latest()
    assert loaded.next_layer == 8
    assert loaded.previous_checkpoint_sha256 == first.checkpoint_sha256
    assert loaded.previous_ledger_record_sha256 == first.ledger_record_sha256
    assert second.previous_checkpoint_sha256 == first.checkpoint_sha256
    np.testing.assert_array_equal(
        loaded.hidden_bf16_bits,
        _checkpoint_payload(8)["hidden_bf16_bits"],
    )
    np.testing.assert_array_equal(
        loaded.prev_topk_indices,
        _checkpoint_payload(8)["prev_topk_indices"],
    )


def test_failed_latest_publication_leaves_previous_checkpoint_authoritative(
    tmp_path, monkeypatch
) -> None:
    store = LocalTeacherCheckpointStore(tmp_path, identity=_identity())
    first = store.publish_forward(**_checkpoint_payload(4))

    def fail_latest(_payload: dict[str, object]) -> None:
        raise OSError("simulated marker publication failure")

    monkeypatch.setattr(store, "_publish_latest", fail_latest)
    with pytest.raises(OSError, match="marker publication"):
        store.publish_forward(**_checkpoint_payload(8))

    recovered = LocalTeacherCheckpointStore(tmp_path, identity=_identity()).load_latest()
    assert recovered.next_layer == 4
    assert recovered.checkpoint_sha256 == first.checkpoint_sha256


def test_resume_rejects_corrupt_object_and_foreign_identity(tmp_path) -> None:
    store = LocalTeacherCheckpointStore(tmp_path, identity=_identity())
    published = store.publish_forward(**_checkpoint_payload(4))
    object_path = tmp_path / "objects" / published.checkpoint_filename
    object_path.write_bytes(object_path.read_bytes() + b"corruption")

    with pytest.raises(CheckpointValidationError, match="checkpoint SHA-256"):
        store.load_latest()

    clean = tmp_path / "clean"
    LocalTeacherCheckpointStore(clean, identity=_identity()).publish_forward(
        **_checkpoint_payload(4)
    )
    foreign = LocalTeacherCheckpointStore(
        clean,
        identity=_identity(code_sha256=_sha("6")),
    )
    with pytest.raises(CheckpointValidationError, match="run identity"):
        foreign.load_latest()


def test_resume_rejects_noncontiguous_or_tampered_chain(tmp_path) -> None:
    store = LocalTeacherCheckpointStore(tmp_path, identity=_identity())
    store.publish_forward(**_checkpoint_payload(4))
    latest = store.publish_forward(**_checkpoint_payload(8))
    record_path = tmp_path / "ledger" / latest.ledger_record_filename
    record = json.loads(record_path.read_text())
    record["previous_checkpoint_sha256"] = _sha("f")
    record_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))

    with pytest.raises(CheckpointValidationError, match="ledger record SHA-256"):
        store.load_latest()


def test_layer_and_deadline_stop_policy(tmp_path) -> None:
    stop_file = tmp_path / "STOP"
    now = datetime(2026, 7, 19, 9, 0, tzinfo=timezone.utc)
    policy = DeadlinePolicy(
        capacity_block_end=now + timedelta(minutes=59),
        stop_file=stop_file,
    )

    assert should_checkpoint_after_layer(4, every_layers=4)
    assert not should_checkpoint_after_layer(5, every_layers=4)
    assert policy.stop_assigning(now=now)
    assert not policy.force_terminate(now=now)
    stop_file.touch()
    assert policy.stop_requested(now=now)


def test_deadline_policy_accepts_generic_execution_deadline() -> None:
    end = datetime(2026, 7, 25, 8, 0, tzinfo=timezone.utc)
    policy = DeadlinePolicy(execution_deadline=end)
    assert policy.stop_assigning(now=end - timedelta(minutes=60))
    assert policy.force_terminate(now=end - timedelta(minutes=50))
    assert policy.final_sync_due(now=end - timedelta(minutes=35))


def _logit_slice(start: int, end: int, *, k: int = 3) -> dict[str, np.ndarray]:
    rows = end - start
    base = np.arange(start, end, dtype=np.int32)[:, None]
    return {
        "topk_logit_ids": np.broadcast_to(base, (rows, k)).copy(),
        "topk_logit_values": np.full((rows, k), float(start), dtype=np.float16),
        "logsumexp": np.arange(start, end, dtype=np.float32),
        "tail_mass": np.full((rows,), 0.125, dtype=np.float16),
    }


def test_lm_head_slices_resume_without_duplicate_or_omitted_positions(tmp_path) -> None:
    positions = np.array([2, 5, 9, 10, 15], dtype=np.int32)
    targets = np.array([7, 8, 9, 10, 11], dtype=np.int32)
    store = LocalLMHeadSliceStore(
        tmp_path,
        identity=_identity(),
        prompt_id="prompt-a",
        token_ids_sha256=_sha("a"),
        positions=positions,
        target_token_ids=targets,
        top_k=3,
    )
    store.publish_slice(start=0, end=2, tensors=_logit_slice(0, 2))
    resumed = LocalLMHeadSliceStore(
        tmp_path,
        identity=_identity(),
        prompt_id="prompt-a",
        token_ids_sha256=_sha("a"),
        positions=positions,
        target_token_ids=targets,
        top_k=3,
    )
    assert resumed.missing_ranges(slice_size=2) == ((2, 4), (4, 5))
    resumed.publish_slice(start=2, end=4, tensors=_logit_slice(2, 4))
    resumed.publish_slice(start=4, end=5, tensors=_logit_slice(4, 5))

    assembled = resumed.assemble()
    np.testing.assert_array_equal(assembled["positions"], positions)
    np.testing.assert_array_equal(assembled["target_token_ids"], targets)
    np.testing.assert_array_equal(assembled["logsumexp"], np.arange(5, dtype=np.float32))
    assert assembled["topk_logit_ids"].shape == (5, 3)


def test_lm_head_interrupted_marker_publication_leaves_slice_missing(
    tmp_path, monkeypatch
) -> None:
    store = LocalLMHeadSliceStore(
        tmp_path,
        identity=_identity(),
        prompt_id="prompt-a",
        token_ids_sha256=_sha("a"),
        positions=np.arange(4, dtype=np.int32),
        target_token_ids=np.arange(4, dtype=np.int32),
        top_k=3,
    )

    def fail_marker(_path, _payload) -> None:
        raise OSError("simulated slice marker failure")

    monkeypatch.setattr(store, "_publish_slice_marker", fail_marker)
    with pytest.raises(OSError, match="slice marker"):
        store.publish_slice(start=0, end=2, tensors=_logit_slice(0, 2))

    recovered = LocalLMHeadSliceStore(
        tmp_path,
        identity=_identity(),
        prompt_id="prompt-a",
        token_ids_sha256=_sha("a"),
        positions=np.arange(4, dtype=np.int32),
        target_token_ids=np.arange(4, dtype=np.int32),
        top_k=3,
    )
    assert recovered.missing_ranges(slice_size=2) == ((0, 2), (2, 4))
