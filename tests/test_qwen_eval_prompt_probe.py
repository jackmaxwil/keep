from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from benchmarks.prepare_qwen_family_eval_prompts import (
    DEFAULT_QWEN_EVAL_PROMPTS,
    build_qwen_family_eval_prompt_probe,
)


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"<unk>": 0}
    for prompt in [
        prompt
        for split_prompts in DEFAULT_QWEN_EVAL_PROMPTS.values()
        for prompt in split_prompts
    ]:
        for token in prompt.split():
            vocab.setdefault(token, len(vocab))
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        eos_token="<unk>",
        pad_token="<unk>",
    )


def test_qwen_eval_prompt_probe_builds_three_split_prompt_pack() -> None:
    payload = build_qwen_family_eval_prompt_probe(
        _tiny_tokenizer(),
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        min_prompts_per_split=22,
        min_tokens_per_prompt=2,
        expected_vocab_size=1000,
    )

    assert payload["record_type"] == "qwen_family_eval_prompt_probe"
    assert payload["prompt_pack_ready"] is True
    assert payload["split_counts"] == {"holdout": 22, "report": 22, "selection": 22}
    assert payload["encoded_prompt_count"] == 66
    assert payload["missing_requirements"] == []
    first_row = payload["prompt_rows"][0]
    assert first_row["record_type"] == "qwen_family_eval_prompt"
    assert first_row["split"] == "report"
    assert first_row["prompt_id"] == "qwen_report_000"
    assert first_row["token_count"] >= 2


def test_qwen_eval_prompt_probe_reports_underfilled_split() -> None:
    payload = build_qwen_family_eval_prompt_probe(
        _tiny_tokenizer(),
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        prompts_by_split={"report": ["hello world"], "selection": [], "holdout": []},
        min_prompts_per_split=2,
        min_tokens_per_prompt=1,
    )

    assert payload["prompt_pack_ready"] is False
    assert payload["split_counts"] == {"holdout": 0, "report": 1, "selection": 0}
    assert payload["missing_requirements"] == [
        "qwen_report_eval_prompts",
        "qwen_selection_eval_prompts",
        "qwen_holdout_eval_prompts",
    ]
