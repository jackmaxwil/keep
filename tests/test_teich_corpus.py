from __future__ import annotations

import copy

import pytest

from keep.quality.teich_corpus import (
    assign_session_disjoint_splits,
    build_keep_best_window_row,
    build_keep_prompt_row,
    build_keep_window_rows,
    deduplicate_prompt_rows,
)


class StubTokenizer:
    def __init__(self, tokens: list[tuple[str, int]]):
        self._tokens = tokens

    def __call__(self, text=None, **kwargs):
        assert kwargs["add_special_tokens"] is False
        assert kwargs["return_offsets_mapping"] is True
        assert text == "".join(token for token, _token_id in self._tokens)
        offsets = []
        cursor = 0
        for token, _token_id in self._tokens:
            offsets.append((cursor, cursor + len(token)))
            cursor += len(token)
        return {
            "input_ids": [token_id for _token, token_id in self._tokens],
            "offset_mapping": offsets,
        }


def _span(text: str, fragment: str, role: str, kind: str = "content") -> dict[str, object]:
    start = text.index(fragment)
    return {
        "start": start,
        "end": start + len(fragment),
        "source_start": 0,
        "source_end": len(fragment),
        "kind": kind,
        "role": role,
    }


def test_build_row_maps_only_assistant_targets_to_causal_positions():
    tokens = [("SYS ", 10), ("USER ", 11), ("think ", 12), ("call ", 13), ("done", 14)]
    text = "".join(token for token, _token_id in tokens)
    spans = [
        _span(text, "SYS ", "system"),
        _span(text, "USER ", "user"),
        _span(text, "think ", "assistant", "reasoning"),
        _span(text, "call ", "assistant", "tool_call"),
        _span(text, "done", "assistant", "final_answer"),
    ]

    row = build_keep_prompt_row(
        input_ids=[10, 11, 12, 13, 14],
        text=text,
        teich_supervised_spans=spans,
        tokenizer=StubTokenizer(tokens),
        prompt_id="row-1",
        provider="claude",
        source_session_id="session-1",
        teich_audit_refs={"converted_row": 7},
    )

    assert row["positions"] == [1, 2, 3]
    assert row["target_token_ids"] == [12, 13, 14]
    assert row["encoded_token_ids"] == [10, 11, 12, 13, 14]
    assert row["provenance"]["provider"] == "claude"


def test_build_row_excludes_token_that_straddles_assistant_span_boundary():
    tokens = [("prefix", 1), ("XY", 2), ("suffix", 3)]
    text = "prefixXYsuffix"
    assistant_span = {
        "start": text.index("Y"),
        "end": text.index("Y") + 1,
        "source_start": 0,
        "source_end": 1,
        "kind": "final_answer",
        "role": "assistant",
    }

    with pytest.raises(ValueError, match="no supervised positions"):
        build_keep_prompt_row(
            input_ids=[1, 2, 3],
            text=text,
            teich_supervised_spans=[assistant_span],
            tokenizer=StubTokenizer(tokens),
            prompt_id="row-boundary",
            provider="codex",
            source_session_id="session-boundary",
            teich_audit_refs={},
        )


def test_build_row_fails_closed_on_empty_assistant_supervision():
    tokens = [("user", 1), (" text", 2)]
    text = "user text"

    with pytest.raises(ValueError, match="no supervised positions"):
        build_keep_prompt_row(
            input_ids=[1, 2],
            text=text,
            teich_supervised_spans=[_span(text, "user text", "user")],
            tokenizer=StubTokenizer(tokens),
            prompt_id="row-empty",
            provider="cursor",
            source_session_id="session-empty",
            teich_audit_refs={},
        )


def test_build_row_fails_closed_when_retokenized_ids_do_not_match():
    tokens = [("user ", 1), ("answer", 9)]
    text = "user answer"

    with pytest.raises(ValueError, match="re-tokenized IDs do not match"):
        build_keep_prompt_row(
            input_ids=[1, 2],
            text=text,
            teich_supervised_spans=[_span(text, "answer", "assistant")],
            tokenizer=StubTokenizer(tokens),
            prompt_id="row-mismatch",
            provider="claude",
            source_session_id="session-mismatch",
            teich_audit_refs={},
        )


def _row(prompt_id: str, provider: str, session_id: str, token_hash: str) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "split": "train",
        "tuning_eligible": True,
        "encoded_token_ids": [1, 2, 3],
        "positions": [1],
        "target_token_ids": [3],
        "token_count": 3,
        "token_ids_sha256": token_hash,
        "domain": "coding_agent",
        "provenance": {"provider": provider, "source_session_id": session_id},
    }


def test_dedup_and_session_disjoint_split_invariants():
    rows = [
        _row("a", "claude", "c1", "hash-a"),
        _row("a-duplicate", "claude", "c1", "hash-a"),
        _row("b", "claude", "c2", "hash-b"),
        _row("c", "codex", "x1", "hash-c"),
        _row("d", "codex", "x2", "hash-d"),
        _row("e", "cursor", "u1", "hash-e"),
        _row("f", "cursor", "u2", "hash-f"),
    ]
    original = copy.deepcopy(rows)

    deduped, duplicate_count = deduplicate_prompt_rows(rows)
    split_rows = assign_session_disjoint_splits(deduped, holdout_fraction=0.5, seed=17)

    assert duplicate_count == 1
    assert len(deduped) == 6
    assert rows == original
    assert {row["split"] for row in split_rows} == {"train", "holdout"}
    assert all(row["tuning_eligible"] is (row["split"] == "train") for row in split_rows)
    session_splits: dict[tuple[str, str], set[str]] = {}
    for row in split_rows:
        provenance = row["provenance"]
        key = (provenance["provider"], provenance["source_session_id"])
        session_splits.setdefault(key, set()).add(row["split"])
    assert all(len(splits) == 1 for splits in session_splits.values())


def test_session_split_balances_supervised_token_weight_not_hash_prefix():
    rows = []
    for session_id, supervised_tokens in (("huge", 90), ("small1", 5), ("small2", 5)):
        row = _row(session_id, "claude", session_id, f"hash-{session_id}")
        row["positions"] = list(range(supervised_tokens))
        row["target_token_ids"] = [3] * supervised_tokens
        rows.append(row)

    split_rows = assign_session_disjoint_splits(rows, holdout_fraction=0.1, seed=2)

    assert sum(len(row["positions"]) for row in split_rows if row["split"] == "holdout") == 10
    assert next(row for row in split_rows if row["provenance"]["source_session_id"] == "huge")[
        "split"
    ] == "train"


def test_window_rows_use_local_causal_positions_and_partial_tail():
    tokens = [(character, ord(character)) for character in "abcdefghijklmn"]
    text = "".join(character for character, _token_id in tokens)

    rows = build_keep_window_rows(
        input_ids=[token_id for _character, token_id in tokens],
        text=text,
        teich_supervised_spans=[_span(text, "cdefghijklmn", "assistant")],
        tokenizer=StubTokenizer(tokens),
        prompt_id_prefix="teich_codex_session",
        provider="codex",
        source_session_id="session",
        teich_audit_refs={"prepared_row": 0},
        window_size=8,
        stride=6,
        minimum_supervised_tokens=2,
        max_windows=40,
        max_session_tokens=12,
    )

    assert [row["prompt_id"] for row in rows] == [
        "teich_codex_session_w00",
        "teich_codex_session_w01",
    ]
    assert [row["encoded_token_ids"] for row in rows] == [
        [ord(character) for character in "abcdefgh"],
        [ord(character) for character in "ghijkl"],
    ]
    assert rows[0]["positions"] == [1, 2, 3, 4, 5, 6]
    assert rows[0]["target_token_ids"] == [ord(character) for character in "cdefgh"]
    assert rows[1]["positions"] == [0, 1, 2, 3, 4]
    assert rows[1]["target_token_ids"] == [ord(character) for character in "hijkl"]
    assert rows[1]["provenance"]["window_index"] == 1
    assert rows[1]["provenance"]["session_token_len"] == 12


def test_window_rows_evenly_cap_qualifying_windows():
    tokens = [("x", 120)] * 30
    text = "x" * 30

    rows = build_keep_window_rows(
        input_ids=[120] * len(text),
        text=text,
        teich_supervised_spans=[_span(text, text, "assistant")],
        tokenizer=StubTokenizer(tokens),
        prompt_id_prefix="teich_claude_session",
        provider="claude",
        source_session_id="session",
        teich_audit_refs={},
        window_size=6,
        stride=3,
        minimum_supervised_tokens=2,
        max_windows=4,
        max_session_tokens=30,
    )

    assert [row["provenance"]["window_index"] for row in rows] == [0, 3, 6, 8]


def test_best_window_finds_exact_non_stride_start_with_earliest_tie_break():
    tokens = [(character, ord(character)) for character in "abcdefghijklmno"]
    text = "".join(character for character, _token_id in tokens)

    row = build_keep_best_window_row(
        input_ids=[token_id for _character, token_id in tokens],
        text=text,
        teich_supervised_spans=[_span(text, "fghijk", "assistant")],
        tokenizer=StubTokenizer(tokens),
        prompt_id="teich_cursor_session_best64k",
        provider="cursor",
        source_session_id="session",
        teich_audit_refs={},
        window_size=6,
        minimum_supervised_tokens=2,
    )

    assert row["encoded_token_ids"] == [ord(character) for character in "efghij"]
    assert row["positions"] == [0, 1, 2, 3, 4]
    assert row["target_token_ids"] == [ord(character) for character in "fghij"]
    assert row["provenance"]["window_start"] == 4
    assert row["provenance"]["window_rule"] == "maximum supervised targets; earliest start tie-break"
