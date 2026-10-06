from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/diag_glm52_candidate_capture_consistency.py"
)


def _load_module():
    module_name = "diag_glm52_candidate_capture_consistency"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeMX:
    int32 = np.int32
    float32 = np.float32

    @staticmethod
    def array(value, dtype=None):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def eval(*_values):
        return None


class _TinyModel:
    def __init__(self, logits: np.ndarray) -> None:
        self.args = SimpleNamespace(vocab_size=logits.shape[-1])
        self._logits = logits
        self.calls: list[tuple[np.ndarray, dict[str, object]]] = []

    def __call__(self, tokens, **kwargs):
        self.calls.append((np.array(tokens), kwargs))
        return self._logits


def test_diagnose_capture_compares_plain_unpadded_forward_to_injected_shards() -> None:
    diag = _load_module()
    plain = np.asarray(
        [[[2.0, 1.0, 0.0, -1.0], [0.0, 3.0, 1.0, 2.0]]],
        dtype=np.float32,
    )
    candidate = plain[0].copy()
    teacher = np.asarray(
        [[0.0, 0.5, 4.0, 1.0], [0.0, 3.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    model = _TinyModel(plain)
    prompt = SimpleNamespace(
        prompt_id="tiny_prompt",
        split="report",
        domain="math",
        encoded_token_ids=(3, 1, 2),
        token_count=3,
    )

    evidence = diag.diagnose_capture_consistency(
        model=model,
        prompt=prompt,
        prompt_index=7,
        candidate_logits=candidate,
        teacher_logits=teacher,
        mx=_FakeMX,
    )

    assert len(model.calls) == 1
    tokens, kwargs = model.calls[0]
    assert tokens.tolist() == [[3, 1]]
    assert kwargs == {}
    assert evidence["capture_consistency_pass"] is True
    assert evidence["plain_single_prompt"]["shape"] == [2, 4]
    assert evidence["candidate_cache"]["shape"] == [2, 4]
    assert evidence["plain_single_prompt"]["sha256"] == hashlib.sha256(
        candidate.tobytes(order="C")
    ).hexdigest()
    assert evidence["candidate_cache"]["sha256"] == evidence[
        "plain_single_prompt"
    ]["sha256"]
    assert evidence["positions"] == [
        {
            "position": 0,
            "max_abs_diff": 0.0,
            "exact_equal": True,
            "plain_top1_token_id": 0,
            "candidate_cache_top1_token_id": 0,
            "top1_match": True,
        },
        {
            "position": 1,
            "max_abs_diff": 0.0,
            "exact_equal": True,
            "plain_top1_token_id": 1,
            "candidate_cache_top1_token_id": 1,
            "top1_match": True,
        },
    ]
    assert evidence["true_next_token_ids"] == [1, 2]
    assert evidence["teacher_cache"]["top1_token_ids"] == [2, 1]
    assert evidence["plain_single_prompt"]["mean_true_token_logprob"] == pytest.approx(
        (-1.4401897 - 2.4401897) / 2
    )
    assert evidence["candidate_cache"]["mean_true_token_logprob"] == pytest.approx(
        evidence["plain_single_prompt"]["mean_true_token_logprob"]
    )
    teacher_lse = np.log(np.exp(teacher.astype(np.float64)).sum(axis=-1))
    expected_teacher = np.mean(
        teacher[np.arange(2), np.asarray([1, 2])] - teacher_lse
    )
    assert evidence["teacher_cache"]["mean_true_token_logprob"] == pytest.approx(
        expected_teacher
    )


def test_diagnose_capture_reports_non_exact_and_top1_mismatch() -> None:
    diag = _load_module()
    plain = np.asarray([[[3.0, 2.0], [1.0, 4.0]]], dtype=np.float32)
    candidate = np.asarray([[3.0, 2.5], [5.0, 4.0]], dtype=np.float32)
    teacher = plain[0].copy()
    model = _TinyModel(plain)
    prompt = SimpleNamespace(
        prompt_id="tiny_prompt",
        split="selection",
        domain="route",
        encoded_token_ids=(0, 1, 0),
        token_count=3,
    )

    evidence = diag.diagnose_capture_consistency(
        model=model,
        prompt=prompt,
        prompt_index=0,
        candidate_logits=candidate,
        teacher_logits=teacher,
        mx=_FakeMX,
    )

    assert evidence["capture_consistency_pass"] is False
    assert evidence["exact_position_count"] == 0
    assert evidence["top1_match_position_count"] == 1
    assert evidence["positions"][0]["max_abs_diff"] == 0.5
    assert evidence["positions"][1]["top1_match"] is False


@pytest.mark.parametrize(
    ("candidate", "teacher", "match"),
    [
        (
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            "candidate cache logits shape",
        ),
        (
            np.zeros((2, 3), dtype=np.float16),
            np.zeros((2, 3), dtype=np.float32),
            "candidate cache logits dtype",
        ),
        (
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 4), dtype=np.float32),
            "teacher cache logits shape",
        ),
    ],
)
def test_diagnose_capture_fails_loud_on_shard_shape_or_dtype(
    candidate: np.ndarray,
    teacher: np.ndarray,
    match: str,
) -> None:
    diag = _load_module()
    plain = np.zeros((1, 2, 3), dtype=np.float32)
    model = _TinyModel(plain)
    prompt = SimpleNamespace(
        prompt_id="tiny_prompt",
        split="holdout",
        domain="instruction",
        encoded_token_ids=(0, 1, 2),
        token_count=3,
    )

    with pytest.raises(ValueError, match=match):
        diag.diagnose_capture_consistency(
            model=model,
            prompt=prompt,
            prompt_index=0,
            candidate_logits=candidate,
            teacher_logits=teacher,
            mx=_FakeMX,
        )
