"""Schema-v3 sparse teacher-cache finalization and audit tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mlx_vq.quality.glm52_teich_training_cache import (
    TeacherCacheV3Error,
    audit_glm52_teich_teacher_cache,
    finalize_glm52_teich_teacher_cache,
)
from mlx_vq.quality.glm52_adapter_training import (
    PreparedGLM52TeichTeacherRow,
    PreparedGLM52TeichDiskRow,
    prepare_glm52_teich_teacher_rows,
    validate_tuning_row,
)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _row(index: int, provider: str, positions: int) -> dict[str, object]:
    sparse_positions = list(range(0, positions * 2, 2))
    tokens = list(
        range(1000 + index * 20, 1000 + index * 20 + sparse_positions[-1] + 2)
    )
    return {
        "prompt_id": f"teich-{provider}-{index}",
        "split": "train",
        "tuning_eligible": True,
        "domain": "coding_agent",
        "encoded_token_ids": tokens,
        "positions": sparse_positions,
        "target_token_ids": [tokens[position + 1] for position in sparse_positions],
        "token_count": len(tokens),
        "token_ids_sha256": _canonical_sha(tokens),
        "provenance": {
            "provider": provider,
            "source_session_id": f"{provider}-session-{index}",
        },
    }


def _write_fixture(root: Path, *, row_count: int = 12) -> tuple[Path, Path, Path, int]:
    rows = [_row(i, ("cursor", "claude", "codex")[i % 3], 3 + (i % 5)) for i in range(row_count)]
    pack = {"record_type": "glm52_coding_agent_corpus", "prompt_rows": rows}
    pack_path = root / "pack.json"
    pack_path.write_text(json.dumps(pack))
    frozen = {
        "prompt_rows": [
            {
                "prompt_id": "frozen-only",
                "encoded_token_ids": [1, 2, 3],
                "token_ids_sha256": _canonical_sha([1, 2, 3]),
            }
        ]
    }
    frozen_path = root / "frozen.json"
    frozen_path.write_text(json.dumps(frozen))
    captures = root / "captures"
    captures.mkdir()
    hidden_size = 6
    for row in rows:
        p = len(row["positions"])
        router_ids = np.broadcast_to(
            np.arange(8, dtype=np.int32), (8, p, 8)
        ).copy()
        router_weights = np.full((8, p, 8), 0.125, dtype=np.float16)
        np.savez_compressed(
            captures / f"{row['prompt_id']}.npz",
            positions=np.asarray(row["positions"], dtype=np.int32),
            target_token_ids=np.asarray(row["target_token_ids"], dtype=np.int32),
            topk_logit_ids=np.zeros((p, 4), dtype=np.int32),
            topk_logit_values=np.zeros((p, 4), dtype=np.float16),
            logsumexp=np.zeros((p,), dtype=np.float32),
            tail_mass=np.full((p,), 0.25, dtype=np.float16),
            layer_77_hidden_probe=np.zeros((p, hidden_size), dtype=np.float16),
            router_top8_expert_ids=router_ids,
            router_top8_normalized_weights=router_weights,
        )
    return pack_path, frozen_path, captures, sum(len(row["positions"]) for row in rows)


def test_finalize_and_fresh_audit_sparse_schema_v3(tmp_path) -> None:
    pack, frozen, captures, total = _write_fixture(tmp_path)
    cache = tmp_path / "cache"

    ready = finalize_glm52_teich_teacher_cache(
        capture_dir=captures,
        prompt_pack_path=pack,
        frozen_prompt_pack_path=frozen,
        cache_dir=cache,
        expected_session_count=12,
        expected_supervised_positions=total,
        top_k=4,
        hidden_size=6,
    )
    audit = audit_glm52_teich_teacher_cache(
        cache_dir=cache,
        prompt_pack_path=pack,
        frozen_prompt_pack_path=frozen,
        expected_session_count=12,
        expected_supervised_positions=total,
        top_k=4,
        hidden_size=6,
    )

    assert ready["manifest_sha256"] == audit.manifest_sha256
    assert audit.session_count == 12
    assert audit.supervised_positions == total
    assert (cache / "TEACHER_CACHE_READY.json").exists()
    manifest = json.loads((cache / "glm52-teacher-signal-cache-v3-manifest.json").read_text())
    assert manifest["schema_version"] == 3
    assert {row["split"] for row in manifest["shards"]} == {
        "train",
        "validation",
        "holdout",
    }
    assert all(row["tuning_eligible"] == (row["split"] == "train") for row in manifest["shards"])
    assert all(row["sparse_positions"] is True for row in manifest["shards"])
    assert set(manifest["shards"][0]["tensors"]) == {
        "positions",
        "target_token_ids",
        "topk_logit_ids",
        "topk_logit_values",
        "logsumexp",
        "tail_mass",
        "layer_77_hidden_probe",
        "router_top8_expert_ids",
        "router_top8_normalized_weights",
    }


def test_audit_rejects_shard_mutation(tmp_path) -> None:
    pack, frozen, captures, total = _write_fixture(tmp_path)
    cache = tmp_path / "cache"
    finalize_glm52_teich_teacher_cache(
        capture_dir=captures,
        prompt_pack_path=pack,
        frozen_prompt_pack_path=frozen,
        cache_dir=cache,
        expected_session_count=12,
        expected_supervised_positions=total,
        top_k=4,
        hidden_size=6,
    )
    shard = next((cache / "teacher_signal").glob("*.safetensors"))
    shard.write_bytes(shard.read_bytes() + b"corruption")

    with pytest.raises(TeacherCacheV3Error, match="file SHA-256"):
        audit_glm52_teich_teacher_cache(
            cache_dir=cache,
            prompt_pack_path=pack,
            frozen_prompt_pack_path=frozen,
            expected_session_count=12,
            expected_supervised_positions=total,
            top_k=4,
            hidden_size=6,
        )


def test_frozen_overlap_or_incomplete_capture_never_publishes_ready(tmp_path) -> None:
    pack, frozen, captures, total = _write_fixture(tmp_path)
    frozen_payload = json.loads(frozen.read_text())
    first = json.loads(pack.read_text())["prompt_rows"][0]
    frozen_payload["prompt_rows"].append(
        {
            "prompt_id": "different-name",
            "encoded_token_ids": first["encoded_token_ids"],
            "token_ids_sha256": first["token_ids_sha256"],
        }
    )
    frozen.write_text(json.dumps(frozen_payload))
    cache = tmp_path / "cache"

    with pytest.raises(TeacherCacheV3Error, match="frozen-66 overlap"):
        finalize_glm52_teich_teacher_cache(
            capture_dir=captures,
            prompt_pack_path=pack,
            frozen_prompt_pack_path=frozen,
            cache_dir=cache,
            expected_session_count=12,
            expected_supervised_positions=total,
            top_k=4,
            hidden_size=6,
        )
    assert not (cache / "TEACHER_CACHE_READY.json").exists()

    frozen_payload["prompt_rows"].pop()
    frozen.write_text(json.dumps(frozen_payload))
    next(captures.glob("*.npz")).unlink()
    with pytest.raises(TeacherCacheV3Error, match="capture is missing"):
        finalize_glm52_teich_teacher_cache(
            capture_dir=captures,
            prompt_pack_path=pack,
            frozen_prompt_pack_path=frozen,
            cache_dir=cache,
            expected_session_count=12,
            expected_supervised_positions=total,
            top_k=4,
            hidden_size=6,
        )
    assert not (cache / "glm52-teacher-signal-cache-v3-manifest.json").exists()
    assert not (cache / "TEACHER_CACHE_READY.json").exists()


def test_sparse_teich_loader_accepts_all_splits_but_only_train_can_optimize(tmp_path) -> None:
    pack, frozen, captures, total = _write_fixture(tmp_path)
    cache = tmp_path / "cache"
    finalize_glm52_teich_teacher_cache(
        capture_dir=captures,
        prompt_pack_path=pack,
        frozen_prompt_pack_path=frozen,
        cache_dir=cache,
        expected_session_count=12,
        expected_supervised_positions=total,
        top_k=4,
        hidden_size=6,
    )
    audit = audit_glm52_teich_teacher_cache(
        cache_dir=cache,
        prompt_pack_path=pack,
        frozen_prompt_pack_path=frozen,
        expected_session_count=12,
        expected_supervised_positions=total,
        top_k=4,
        hidden_size=6,
    )

    loaded = {
        split: prepare_glm52_teich_teacher_rows(
            teacher_cache_dir=cache,
            prompt_pack_path=pack,
            frozen_prompt_pack_path=frozen,
            split=split,
            expected_manifest_sha256=audit.manifest_sha256,
            allow_non_release_teacher_cache=True,
        )
        for split in ("train", "validation", "holdout")
    }
    assert all(loaded.values())
    train_row = loaded["train"][0]
    assert isinstance(train_row, PreparedGLM52TeichTeacherRow)
    assert train_row.positions != tuple(range(len(train_row.input_token_ids) - 1))
    assert str(train_row.topk_logit_values.dtype) == "mlx.core.float16"
    assert set(train_row.teacher_router_topk_ids) == set(range(70, 78))
    validate_tuning_row(train_row)
    with pytest.raises(ValueError, match="train-only"):
        validate_tuning_row(loaded["holdout"][0])

    lazy_rows = prepare_glm52_teich_teacher_rows(
        teacher_cache_dir=cache,
        prompt_pack_path=pack,
        frozen_prompt_pack_path=frozen,
        split="train",
        expected_manifest_sha256=audit.manifest_sha256,
        allow_non_release_teacher_cache=True,
        lazy=True,
    )
    assert isinstance(lazy_rows[0], PreparedGLM52TeichDiskRow)
    window = lazy_rows[0].load_window(0, min(2, len(lazy_rows[0].positions)))
    assert isinstance(window, PreparedGLM52TeichTeacherRow)
    assert len(window.positions) <= 2
    evaluation_window = lazy_rows[0].load_distillation_window(
        0, min(2, len(lazy_rows[0].positions))
    )
    assert evaluation_window.teacher_probe_hidden_states is None
    assert evaluation_window.teacher_router_topk_ids is None
