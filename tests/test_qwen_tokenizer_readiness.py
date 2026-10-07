from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from benchmarks.probe_qwen_tokenizer_readiness import evaluate_qwen_tokenizer_readiness


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "hello": 1, "world": 2}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        eos_token="<unk>",
        pad_token="<unk>",
    )


def test_qwen_tokenizer_readiness_passes_when_prompt_encodes_tokens() -> None:
    payload = evaluate_qwen_tokenizer_readiness(
        _tiny_tokenizer(),
        prompt="hello world",
        min_token_count=2,
        expected_vocab_size=3,
    )

    assert payload["record_type"] == "qwen_tokenizer_readiness_probe"
    assert payload["tokenizer_load_pass"] is True
    assert payload["tokenizer_payload_ready"] is True
    assert payload["token_count"] == 2
    assert payload["vocab_size"] == 3
    assert payload["encoded_token_ids"] == [1, 2]
    assert payload["missing_requirements"] == []


def test_qwen_tokenizer_readiness_accepts_model_vocab_larger_than_tokenizer() -> None:
    payload = evaluate_qwen_tokenizer_readiness(
        _tiny_tokenizer(),
        prompt="hello world",
        min_token_count=2,
        expected_vocab_size=8,
    )

    assert payload["tokenizer_payload_ready"] is True
    assert payload["vocab_size"] == 3
    assert payload["expected_vocab_size"] == 8
    assert payload["max_token_id"] == 2
    assert payload["missing_requirements"] == []


def test_qwen_tokenizer_readiness_rejects_empty_encode() -> None:
    payload = evaluate_qwen_tokenizer_readiness(
        _tiny_tokenizer(),
        prompt="",
        min_token_count=1,
        expected_vocab_size=3,
    )

    assert payload["tokenizer_load_pass"] is True
    assert payload["tokenizer_payload_ready"] is False
    assert payload["token_count"] == 0
    assert payload["missing_requirements"] == ["prompt_encodes_tokens"]
