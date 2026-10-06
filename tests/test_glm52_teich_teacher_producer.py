"""Tests for the teich top-K teacher producer (pure parts)."""

from __future__ import annotations

import json

import numpy as np
import pytest

import mlx_vq.quality.glm52_teich_teacher_producer as teich_producer

from mlx_vq.quality.glm52_teacher_cache import canonical_sha256
from mlx_vq.quality.glm52_teich_teacher_producer import (
    _atomic_savez,
    load_teich_prompt_rows,
    pack_sessions_into_chunks,
    topk_capture_numpy,
    validate_full_v2_capture,
)


class TestPostLayerCacheClear:
    def test_explicit_skip_flag_suppresses_post_layer_clear(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("GLM52_TEICH_SKIP_POST_LAYER_CLEAR_CACHE", "1")
        events: list[str] = []

        teich_producer._clear_post_layer_cache(
            lambda: events.append("clear_cache")
        )

        assert events == []

    def test_default_path_preserves_post_layer_clear(self, monkeypatch) -> None:
        monkeypatch.delenv("GLM52_TEICH_SKIP_POST_LAYER_CLEAR_CACHE", raising=False)
        events: list[str] = []

        teich_producer._clear_post_layer_cache(
            lambda: events.append("clear_cache")
        )

        assert events == ["clear_cache"]


def _make_row(prompt_id: str, tokens: list[int], positions: list[int]) -> dict:
    return {
        "prompt_id": prompt_id,
        "split": "train",
        "domain": "coding_agent",
        "tuning_eligible": True,
        "encoded_token_ids": tokens,
        "positions": positions,
        "target_token_ids": [tokens[p + 1] for p in positions],
        "token_count": len(tokens),
        "token_ids_sha256": canonical_sha256(tokens),
    }


def _write_pack(tmp_path, rows):
    pack = {
        "record_type": "glm52_coding_agent_corpus",
        "prompt_row_count": len(rows),
        "prompt_rows": rows,
    }
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(pack))
    return path


class TestLoadTeichPromptRows:
    def test_loads_valid_pack(self, tmp_path):
        rows = [
            _make_row("teich_a", [5, 9, 2, 7, 1], [1, 3]),
            _make_row("teich_b", [4, 4, 8], [0, 1]),
        ]
        sessions = load_teich_prompt_rows(_write_pack(tmp_path, rows))
        assert len(sessions) == 2
        assert sessions[0].prompt_id == "teich_a"
        assert sessions[0].token_count == 5
        np.testing.assert_array_equal(sessions[0].positions, [1, 3])
        np.testing.assert_array_equal(sessions[0].target_token_ids, [2, 1])

    def test_rejects_misaligned_targets(self, tmp_path):
        row = _make_row("teich_a", [5, 9, 2, 7, 1], [1, 3])
        row["target_token_ids"] = [2, 9]  # wrong second target
        with pytest.raises(ValueError, match="encoded\\[pos\\+1\\]"):
            load_teich_prompt_rows(_write_pack(tmp_path, [row]))

    def test_rejects_position_at_last_token(self, tmp_path):
        row = _make_row("teich_a", [5, 9, 2], [0])
        row["positions"] = [2]  # no next token to predict
        row["target_token_ids"] = [0]
        with pytest.raises(ValueError, match="predictor range"):
            load_teich_prompt_rows(_write_pack(tmp_path, [row]))

    def test_rejects_bad_token_sha(self, tmp_path):
        row = _make_row("teich_a", [5, 9, 2, 7, 1], [1])
        row["token_ids_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="token_ids_sha256"):
            load_teich_prompt_rows(_write_pack(tmp_path, [row]))

    def test_rejects_duplicate_ids(self, tmp_path):
        rows = [
            _make_row("teich_a", [5, 9, 2], [0]),
            _make_row("teich_a", [4, 4, 8], [0]),
        ]
        with pytest.raises(ValueError, match="unique"):
            load_teich_prompt_rows(_write_pack(tmp_path, rows))

    def test_rejects_non_increasing_positions(self, tmp_path):
        row = _make_row("teich_a", [5, 9, 2, 7, 1], [3, 1])
        row["target_token_ids"] = [1, 2]
        with pytest.raises(ValueError, match="strictly increasing"):
            load_teich_prompt_rows(_write_pack(tmp_path, [row]))


class TestTopkCaptureNumpy:
    def test_matches_bruteforce(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(scale=6.0, size=(11, 997)).astype(np.float32)
        ids, values, logsumexp, tail = topk_capture_numpy(logits, k=32)
        assert ids.shape == (11, 32) and values.shape == (11, 32)
        # descending values, ids consistent with logits
        assert np.all(np.diff(values, axis=1) <= 1e-6)
        np.testing.assert_allclose(
            np.take_along_axis(logits, ids, axis=1), values, rtol=0, atol=1e-6
        )
        # exact top-32 selection
        expected_top = np.sort(logits, axis=1)[:, -32:][:, ::-1]
        np.testing.assert_allclose(values, expected_top, atol=1e-6)
        # logsumexp + tail mass against float64 softmax
        full = logits.astype(np.float64)
        ref_lse = np.log(np.exp(full - full.max(1, keepdims=True)).sum(1)) + full.max(1)
        np.testing.assert_allclose(logsumexp, ref_lse, rtol=1e-6)
        probs = np.exp(full - ref_lse[:, None])
        ref_tail = 1.0 - np.take_along_axis(probs, ids, axis=1).sum(1)
        np.testing.assert_allclose(tail, np.clip(ref_tail, 0, None), atol=1e-6)
        assert np.all(tail >= 0.0) and np.all(tail < 1.0)

    def test_k_equals_vocab_gives_zero_tail(self):
        rng = np.random.default_rng(3)
        logits = rng.normal(size=(4, 50)).astype(np.float32)
        _, _, _, tail = topk_capture_numpy(logits, k=50)
        np.testing.assert_allclose(tail, np.zeros(4), atol=1e-6)

    def test_rejects_bad_k(self):
        logits = np.zeros((2, 10), dtype=np.float32)
        with pytest.raises(ValueError):
            topk_capture_numpy(logits, k=0)
        with pytest.raises(ValueError):
            topk_capture_numpy(logits, k=11)


class TestTopkCaptureMlxParity:
    def test_mlx_matches_numpy_reference(self):
        mx = pytest.importorskip("mlx.core")
        from mlx_vq.quality.glm52_teich_teacher_producer import _topk_capture_mlx

        rng = np.random.default_rng(11)
        logits = rng.normal(scale=8.0, size=(7, 1531)).astype(np.float32)
        ref_ids, ref_values, ref_lse, ref_tail = topk_capture_numpy(logits, k=64)
        ids, values, lse, tail = _topk_capture_mlx(mx.array(logits), k=64)
        # ids may differ on exact logit ties; compare via gathered values
        np.testing.assert_allclose(values, ref_values, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(
            np.take_along_axis(logits, ids.astype(np.int64), axis=1),
            values,
            rtol=0,
            atol=1e-5,
        )
        np.testing.assert_allclose(lse, ref_lse, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(tail, ref_tail, rtol=1e-4, atol=1e-5)


class TestPackSessionsIntoChunks:
    @staticmethod
    def _fake_sessions(lengths):
        class _S:
            def __init__(self, i, n):
                self.prompt_id = f"s{i}"
                self.token_count = n

        return [_S(i, n) for i, n in enumerate(lengths)]

    def test_budget_respected_and_complete(self):
        sessions = self._fake_sessions([100, 900, 500, 400, 250, 60])
        chunks = pack_sessions_into_chunks(sessions, max_batch_tokens=1000)
        seen = sorted(i for chunk in chunks for i in chunk)
        assert seen == list(range(6))
        for chunk in chunks:
            max_len = max(sessions[i].token_count for i in chunk)
            assert max_len * len(chunk) <= 1000

    def test_oversize_session_raises(self):
        sessions = self._fake_sessions([2000])
        with pytest.raises(ValueError, match="exceeds"):
            pack_sessions_into_chunks(sessions, max_batch_tokens=1000)

    def test_singleton_chunks_when_budget_tight(self):
        sessions = self._fake_sessions([800, 700, 600])
        chunks = pack_sessions_into_chunks(sessions, max_batch_tokens=900)
        assert all(len(c) == 1 for c in chunks)
        assert len(chunks) == 3


class TestAtomicSavez:
    def test_writes_loadable_file_at_exact_target_path(self, tmp_path):
        target = tmp_path / "teich_cursor_cursor-391780b9.npz"
        arrays = {
            "positions": np.arange(5, dtype=np.int32),
            "topk_logit_values": np.zeros((5, 8), dtype=np.float32),
        }
        _atomic_savez(target, **arrays)
        assert target.exists()
        # no stray temp file left behind (the historical bug: np.savez_compressed
        # appends ".npz" to a tmp path that doesn't already end with it, so the
        # real file landed at "<target>.tmp.npz" while os.replace looked for a
        # different path and raised FileNotFoundError)
        assert sorted(p.name for p in tmp_path.iterdir()) == [target.name]
        loaded = np.load(target)
        np.testing.assert_array_equal(loaded["positions"], arrays["positions"])
        np.testing.assert_array_equal(
            loaded["topk_logit_values"], arrays["topk_logit_values"]
        )

    def test_overwrites_existing_target(self, tmp_path):
        target = tmp_path / "session.npz"
        _atomic_savez(target, positions=np.array([1], dtype=np.int32))
        _atomic_savez(target, positions=np.array([1, 2, 3], dtype=np.int32))
        loaded = np.load(target)
        np.testing.assert_array_equal(loaded["positions"], np.array([1, 2, 3]))


class TestFullV2CaptureContract:
    @staticmethod
    def _capture() -> dict[str, np.ndarray]:
        p, k, hidden = 5, 4, 6
        return {
            "positions": np.array([1, 3, 5, 7, 9], dtype=np.int32),
            "target_token_ids": np.arange(p, dtype=np.int32),
            "topk_logit_ids": np.zeros((p, k), dtype=np.int32),
            "topk_logit_values": np.zeros((p, k), dtype=np.float16),
            "logsumexp": np.zeros((p,), dtype=np.float32),
            "tail_mass": np.full((p,), 0.25, dtype=np.float16),
            "layer_77_hidden_probe": np.zeros((p, hidden), dtype=np.float16),
            "router_top8_expert_ids": np.zeros((8, p, 8), dtype=np.int32),
            "router_top8_normalized_weights": np.full(
                (8, p, 8), 0.125, dtype=np.float16
            ),
        }

    def test_accepts_exact_dtype_and_shape_contract(self):
        validate_full_v2_capture(self._capture(), top_k=4, hidden_size=6)

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            (lambda value: value.update(topk_logit_values=value["topk_logit_values"].astype(np.float32)), "topk_logit_values"),
            (lambda value: value.update(router_top8_expert_ids=value["router_top8_expert_ids"][:7]), "router_top8_expert_ids"),
            (lambda value: value.update(layer_77_hidden_probe=value["layer_77_hidden_probe"][:, :5]), "layer_77_hidden_probe"),
        ],
    )
    def test_rejects_dtype_or_shape_drift(self, mutation, message):
        capture = self._capture()
        mutation(capture)
        with pytest.raises(ValueError, match=message):
            validate_full_v2_capture(capture, top_k=4, hidden_size=6)
