from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "benchmarks/verify_dsv4_task5_teacher_cache.py"


def _load_verifier():
    assert VERIFIER.is_file(), "Task 5 replay verifier is not committed"
    spec = importlib.util.spec_from_file_location("task5_verifier", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_session_file(path: Path, *, duplicate_middle_id: bool = False) -> dict:
    token_ids = [1, 2, 3, 4, 5]
    positions = np.asarray([0, 1, 2], dtype=np.int32)
    targets = np.asarray([2, 3, 4], dtype=np.int32)
    topk_ids = np.tile(np.asarray([0, 1], dtype=np.int32), (3, 2, 1))
    if duplicate_middle_id:
        topk_ids[1, 0] = 0
    arrays = {
        "positions": positions,
        "target_token_ids": targets,
        "mtp_draft_width": np.asarray(2, dtype=np.int32),
        "mtp_target_token_ids": np.asarray([[2, 3], [3, 4], [4, 5]], dtype=np.int32),
        "mtp_target_valid": np.ones((3, 2), dtype=np.bool_),
        "mtp_topk_logit_ids": topk_ids,
        "mtp_topk_logit_values": np.tile(
            np.asarray([3.0, 2.0], dtype=np.float16), (3, 2, 1)
        ),
        "mtp_logsumexp": np.full((3, 2), 4.0, dtype=np.float32),
        "mtp_tail_mass": np.full((3, 2), 0.1, dtype=np.float16),
        "mtp_final_hidden": np.arange(18, dtype=np.float16).reshape(3, 2, 3),
        "record_type": np.asarray("dsv4_teacher_mtp_targets_v1"),
        "schema_version": np.asarray(1, dtype=np.int32),
        "prompt_id": np.asarray("tiny"),
        "campaign_split": np.asarray("mtp-train"),
        "token_ids_sha256": np.asarray(
            hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode()
            ).hexdigest()
        ),
        "token_count": np.asarray(len(token_ids), dtype=np.int64),
        "prefill_chunk_tokens": np.asarray(4, dtype=np.int32),
        "generation_config_sha256": np.asarray("a" * 64),
        "tokens_per_s": np.asarray(2.5, dtype=np.float64),
        "session_seconds": np.asarray(2.0, dtype=np.float64),
    }
    np.savez(path, **arrays)
    return {
        "prompt_id": "tiny",
        "campaign_split": "mtp-train",
        "token_ids": token_ids,
        "positions": [0, 1, 2],
        "target_token_ids": [2, 3, 4],
        "token_ids_sha256": str(arrays["token_ids_sha256"]),
    }


def test_replay_verifier_accepts_an_exact_session(tmp_path):
    verifier = _load_verifier()
    path = tmp_path / "tiny.npz"
    session = _tiny_session_file(path)

    result = verifier.verify_session_file(
        path,
        session=session,
        generation_sha256="a" * 64,
        chunk=4,
        top_k=2,
        width=2,
        hidden_size=3,
        vocab_size=8,
    )

    assert result["prompt_id"] == "tiny"
    assert result["bytes"] == path.stat().st_size
    assert len(result["sha256"]) == 64


def test_replay_verifier_checks_every_position_for_duplicate_ids(tmp_path):
    verifier = _load_verifier()
    path = tmp_path / "tiny.npz"
    session = _tiny_session_file(path, duplicate_middle_id=True)

    with pytest.raises(ValueError, match="duplicate top-k id.*row 1"):
        verifier.verify_session_file(
            path,
            session=session,
            generation_sha256="a" * 64,
            chunk=4,
            top_k=2,
            width=2,
            hidden_size=3,
            vocab_size=8,
        )


def test_legacy_digest_transport_mismatch_is_reported_not_trusted():
    verifier = _load_verifier()
    payload = {
        "record_type": "dsv4_task5_terminal_audit_v1",
        "schema_version": 1,
        "result": "PASS",
        "contract_checks": {name: True for name in verifier.SEMANTIC_CLAIMS},
        "session_files": [{"value": 1}],
        "session_files_contract_sha256": verifier.LEGACY_SESSION_CONTRACT_SHA256,
        "audit_contract_sha256": verifier.LEGACY_AUDIT_CONTRACT_SHA256,
    }

    result = verifier.summarize_legacy_evidence(
        payload,
        file_sha256=verifier.LEGACY_EVIDENCE_SHA256,
    )

    assert result["authority"] == "historical-record-only"
    assert result["session_contract"]["claimed_sha256"] == (
        verifier.LEGACY_SESSION_CONTRACT_SHA256
    )
    assert result["session_contract"]["stored_preimage_sha256"] != (
        verifier.LEGACY_SESSION_CONTRACT_SHA256
    )
    assert result["audit_contract"]["stored_preimage_sha256"] != (
        verifier.LEGACY_AUDIT_CONTRACT_SHA256
    )
    assert result["semantic_claim_names"] == sorted(verifier.SEMANTIC_CLAIMS)
