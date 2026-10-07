from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(_canonical_bytes(list(token_ids))).hexdigest()


def _flat_list(value: Any, *, label: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = list(value)
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"{label} must contain one tokenized sequence")
        value = value[0]
    return value


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    kwargs = {
        "add_special_tokens": False,
        "return_attention_mask": False,
        "return_offsets_mapping": True,
    }
    try:
        encoded = tokenizer(text=text, **kwargs)
    except TypeError:
        encoded = tokenizer(text, **kwargs)
    if not isinstance(encoded, Mapping):
        raise ValueError("tokenizer output must be a mapping")
    if "input_ids" not in encoded or "offset_mapping" not in encoded:
        raise ValueError("tokenizer must return input_ids and offset_mapping")
    token_ids = _flat_list(encoded["input_ids"], label="input_ids")
    offsets = _flat_list(encoded["offset_mapping"], label="offset_mapping")
    normalized_offsets: list[tuple[int, int]] = []
    for index, offset in enumerate(offsets):
        if (
            not isinstance(offset, (list, tuple))
            or len(offset) != 2
            or any(type(value) is not int for value in offset)
        ):
            raise ValueError(f"offset_mapping[{index}] must be an integer pair")
        normalized_offsets.append((offset[0], offset[1]))
    return token_ids, normalized_offsets


def _assistant_char_spans(
    text: str, teich_supervised_spans: Sequence[Mapping[str, Any]]
) -> list[tuple[int, int]]:
    assistant_spans: list[tuple[int, int]] = []
    for index, span in enumerate(teich_supervised_spans):
        if not isinstance(span, Mapping):
            raise ValueError(f"teich_supervised_spans[{index}] must be a mapping")
        role = span.get("role")
        start = span.get("start")
        end = span.get("end")
        if role == "assistant":
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end <= start
                or end > len(text)
            ):
                raise ValueError(f"assistant span {index} has invalid character bounds")
            assistant_spans.append((start, end))
    return sorted(set(assistant_spans))


def build_keep_prompt_row(
    *,
    input_ids: Sequence[int],
    text: str,
    teich_supervised_spans: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    prompt_id: str,
    provider: str,
    source_session_id: str,
    teich_audit_refs: Mapping[str, Any],
    split: str = "train",
    domain: str = "coding_agent",
) -> dict[str, Any]:
    """Map one prepared teich example to a causal KEEP prompt-pack row.

    Assistant-character spans identify target tokens. KEEP positions index the
    logits that predict those tokens, so each target token at index ``i`` maps
    to position ``i - 1`` and target ``input_ids[i]``.
    """

    if not isinstance(text, str):
        raise ValueError("text must be a string")
    token_ids = list(input_ids)
    if len(token_ids) < 2 or any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise ValueError("input_ids must contain at least two non-negative integers")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError("prompt_id must be a non-empty string")
    if not isinstance(provider, str) or not provider:
        raise ValueError("provider must be a non-empty string")
    if not isinstance(source_session_id, str) or not source_session_id:
        raise ValueError("source_session_id must be a non-empty string")
    if split not in {"train", "holdout"}:
        raise ValueError("split must be 'train' or 'holdout'")

    retokenized_ids, offsets = _tokenize_with_offsets(tokenizer, text)
    if retokenized_ids != token_ids:
        raise ValueError("re-tokenized IDs do not match input_ids")
    if len(offsets) != len(token_ids):
        raise ValueError("offset count does not match input_ids")

    assistant_spans = _assistant_char_spans(text, teich_supervised_spans)
    target_indices = sorted(
        {
            token_index
            for token_index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start
            and any(
                span_start <= token_start and token_end <= span_end
                for span_start, span_end in assistant_spans
            )
        }
    )
    positions = [target_index - 1 for target_index in target_indices if target_index > 0]
    if not positions:
        raise ValueError("row has no supervised positions")
    if any(position < 0 or position + 1 >= len(token_ids) for position in positions):
        raise ValueError("supervised position is out of range")
    target_token_ids = [token_ids[position + 1] for position in positions]
    if len(positions) != len(target_token_ids):
        raise ValueError("positions and target_token_ids must have equal lengths")

    return {
        "prompt_id": prompt_id,
        "split": split,
        "tuning_eligible": split == "train",
        "encoded_token_ids": token_ids,
        "positions": positions,
        "target_token_ids": target_token_ids,
        "token_count": len(token_ids),
        "token_ids_sha256": _token_ids_sha256(token_ids),
        "domain": domain,
        "provenance": {
            "provider": provider,
            "source_session_id": source_session_id,
            "teich_audit_refs": dict(teich_audit_refs),
        },
    }


def build_keep_window_rows(
    *,
    input_ids: Sequence[int],
    text: str,
    teich_supervised_spans: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    prompt_id_prefix: str,
    provider: str,
    source_session_id: str,
    teich_audit_refs: Mapping[str, Any],
    window_size: int,
    stride: int,
    minimum_supervised_tokens: int,
    max_windows: int,
    max_session_tokens: int,
    split: str = "train",
    domain: str = "coding_agent",
) -> list[dict[str, Any]]:
    """Map one prepared session to bounded causal windows.

    Windows retain only targets whose causal predictor is also inside the
    window. If more than ``max_windows`` qualify, windows are sampled evenly
    from the complete qualifying sequence, including both endpoints.
    """

    positive_limits = {
        "window_size": window_size,
        "stride": stride,
        "minimum_supervised_tokens": minimum_supervised_tokens,
        "max_windows": max_windows,
        "max_session_tokens": max_session_tokens,
    }
    for label, value in positive_limits.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if not isinstance(prompt_id_prefix, str) or not prompt_id_prefix:
        raise ValueError("prompt_id_prefix must be a non-empty string")
    if split not in {"train", "holdout"}:
        raise ValueError("split must be 'train' or 'holdout'")

    token_ids = list(input_ids)
    retokenized_ids, offsets = _tokenize_with_offsets(tokenizer, text)
    if retokenized_ids != token_ids:
        raise ValueError("re-tokenized IDs do not match input_ids")
    session_token_len = min(len(token_ids), max_session_tokens)
    token_ids = token_ids[:session_token_len]
    offsets = offsets[:session_token_len]
    if len(token_ids) < 2:
        return []

    assistant_spans = _assistant_char_spans(text, teich_supervised_spans)
    target_indices = {
        token_index
        for token_index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start
        and any(
            span_start <= token_start and token_end <= span_end
            for span_start, span_end in assistant_spans
        )
    }

    qualifying: list[dict[str, Any]] = []
    window_index = 0
    window_start = 0
    while window_start < session_token_len:
        window_end = min(window_start + window_size, session_token_len)
        global_targets = sorted(
            index
            for index in target_indices
            if window_start < index < window_end
        )
        if len(global_targets) >= minimum_supervised_tokens:
            positions = [index - window_start - 1 for index in global_targets]
            window_ids = token_ids[window_start:window_end]
            audit_refs = dict(teich_audit_refs)
            audit_refs.update(
                {
                    "window_index": window_index,
                    "window_start": window_start,
                    "window_end": window_end,
                    "session_token_len": session_token_len,
                }
            )
            qualifying.append(
                {
                    "prompt_id": f"{prompt_id_prefix}_w{window_index:02d}",
                    "split": split,
                    "tuning_eligible": split == "train",
                    "encoded_token_ids": window_ids,
                    "positions": positions,
                    "target_token_ids": [window_ids[position + 1] for position in positions],
                    "token_count": len(window_ids),
                    "token_ids_sha256": _token_ids_sha256(window_ids),
                    "domain": domain,
                    "provenance": {
                        "provider": provider,
                        "source_session_id": source_session_id,
                        "teich_audit_refs": audit_refs,
                        "window_index": window_index,
                        "window_start": window_start,
                        "window_end": window_end,
                        "session_token_len": session_token_len,
                    },
                }
            )
        if window_end == session_token_len:
            break
        window_index += 1
        window_start += stride

    if len(qualifying) <= max_windows:
        return qualifying
    last = len(qualifying) - 1
    denominator = max_windows - 1
    if denominator == 0:
        selected_indices = [0]
    else:
        selected_indices = [
            (rank * last + denominator - 1) // denominator
            for rank in range(max_windows)
        ]
    return [qualifying[index] for index in selected_indices]


def build_keep_best_window_row(
    *,
    input_ids: Sequence[int],
    text: str,
    teich_supervised_spans: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    prompt_id: str,
    provider: str,
    source_session_id: str,
    teich_audit_refs: Mapping[str, Any],
    window_size: int,
    minimum_supervised_tokens: int,
    split: str = "train",
    domain: str = "coding_agent",
) -> dict[str, Any]:
    """Return the exact highest-supervision contiguous causal window."""

    if type(window_size) is not int or window_size <= 1:
        raise ValueError("window_size must be an integer greater than one")
    if type(minimum_supervised_tokens) is not int or minimum_supervised_tokens <= 0:
        raise ValueError("minimum_supervised_tokens must be a positive integer")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError("prompt_id must be a non-empty string")
    if not isinstance(provider, str) or not provider:
        raise ValueError("provider must be a non-empty string")
    if not isinstance(source_session_id, str) or not source_session_id:
        raise ValueError("source_session_id must be a non-empty string")
    if split not in {"train", "holdout"}:
        raise ValueError("split must be 'train' or 'holdout'")

    token_ids = list(input_ids)
    if len(token_ids) < 2 or any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise ValueError("input_ids must contain at least two non-negative integers")
    retokenized_ids, offsets = _tokenize_with_offsets(tokenizer, text)
    if retokenized_ids != token_ids:
        raise ValueError("re-tokenized IDs do not match input_ids")
    if len(offsets) != len(token_ids):
        raise ValueError("offset count does not match input_ids")

    assistant_spans = _assistant_char_spans(text, teich_supervised_spans)
    target_indices = sorted(
        token_index
        for token_index, (token_start, token_end) in enumerate(offsets)
        if token_index > 0
        and token_end > token_start
        and any(
            span_start <= token_start and token_end <= span_end
            for span_start, span_end in assistant_spans
        )
    )
    actual_window_size = min(window_size, len(token_ids))
    max_start = len(token_ids) - actual_window_size
    events: dict[int, int] = defaultdict(int)
    for target_index in target_indices:
        first_start = max(0, target_index - actual_window_size + 1)
        last_start = min(target_index - 1, max_start)
        if first_start <= last_start:
            events[first_start] += 1
            events[last_start + 1] -= 1

    count = 0
    best_count = -1
    best_start = 0
    for start in sorted(events):
        count += events[start]
        if start <= max_start and count > best_count:
            best_count = count
            best_start = start
    if best_count < minimum_supervised_tokens:
        raise ValueError("best window has fewer than minimum_supervised_tokens")

    window_end = best_start + actual_window_size
    selected_targets = [
        target_index
        for target_index in target_indices
        if best_start < target_index < window_end
    ]
    positions = [target_index - best_start - 1 for target_index in selected_targets]
    window_ids = token_ids[best_start:window_end]
    return {
        "prompt_id": prompt_id,
        "split": split,
        "tuning_eligible": split == "train",
        "encoded_token_ids": window_ids,
        "positions": positions,
        "target_token_ids": [window_ids[position + 1] for position in positions],
        "token_count": len(window_ids),
        "token_ids_sha256": _token_ids_sha256(window_ids),
        "domain": domain,
        "provenance": {
            "provider": provider,
            "source_session_id": source_session_id,
            "teich_audit_refs": dict(teich_audit_refs),
            "window_start": best_start,
            "window_end": window_end,
            "session_token_len": len(token_ids),
            "window_rule": "maximum supervised targets; earliest start tie-break",
        },
    }


def deduplicate_prompt_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Keep the first row for each token hash without mutating inputs."""

    seen: set[str] = set()
    deduplicated: list[dict[str, Any]] = []
    duplicate_count = 0
    for index, row in enumerate(rows):
        token_hash = row.get("token_ids_sha256")
        if not isinstance(token_hash, str) or not token_hash:
            raise ValueError(f"row {index} has no token_ids_sha256")
        if token_hash in seen:
            duplicate_count += 1
            continue
        seen.add(token_hash)
        deduplicated.append(dict(row))
    return deduplicated, duplicate_count


def assign_session_disjoint_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    holdout_fraction: float = 0.1,
    seed: int = 20260712,
) -> list[dict[str, Any]]:
    """Assign deterministic provider-stratified splits at session granularity."""

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between zero and one")
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    weights: dict[tuple[str, str], int] = defaultdict(int)
    for index, row in enumerate(rows):
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"row {index} has no provenance mapping")
        provider = provenance.get("provider")
        session_id = provenance.get("source_session_id")
        if not isinstance(provider, str) or not isinstance(session_id, str):
            raise ValueError(f"row {index} has invalid provider/session provenance")
        positions = row.get("positions")
        if not isinstance(positions, list) or not positions:
            raise ValueError(f"row {index} has no supervised positions")
        grouped[provider][session_id].append(index)
        weights[(provider, session_id)] += len(positions)

    holdout_sessions: set[tuple[str, str]] = set()
    for provider, sessions in grouped.items():
        if len(sessions) < 2:
            continue
        hash_ordered = sorted(
            sessions,
            key=lambda session_id: hashlib.sha256(
                f"{seed}:{provider}:{session_id}".encode("utf-8")
            ).digest(),
        )
        hash_rank = {session_id: rank for rank, session_id in enumerate(hash_ordered)}
        ordered = sorted(
            sessions,
            key=lambda session_id: (-weights[(provider, session_id)], hash_rank[session_id]),
        )
        total_weight = sum(weights[(provider, session_id)] for session_id in ordered)
        target_weight = total_weight * holdout_fraction
        chosen: list[str] = []
        chosen_weight = 0
        for session_id in ordered:
            if len(chosen) >= len(ordered) - 1:
                break
            session_weight = weights[(provider, session_id)]
            if chosen_weight + session_weight <= target_weight:
                chosen.append(session_id)
                chosen_weight += session_weight
        if not chosen:
            chosen = [min(ordered, key=lambda session_id: weights[(provider, session_id)])]
            chosen_weight = weights[(provider, chosen[0])]
        remaining = [session_id for session_id in ordered if session_id not in chosen]
        if len(chosen) < len(ordered) - 1 and remaining:
            candidate = min(
                remaining,
                key=lambda session_id: (
                    abs(chosen_weight + weights[(provider, session_id)] - target_weight),
                    hash_rank[session_id],
                ),
            )
            if abs(chosen_weight + weights[(provider, candidate)] - target_weight) < abs(
                chosen_weight - target_weight
            ):
                chosen.append(candidate)
        holdout_sessions.update((provider, session_id) for session_id in chosen)

    assigned: list[dict[str, Any]] = []
    for row in rows:
        provenance = row["provenance"]
        key = (provenance["provider"], provenance["source_session_id"])
        split = "holdout" if key in holdout_sessions else "train"
        updated = dict(row)
        updated["split"] = split
        updated["tuning_eligible"] = split == "train"
        assigned.append(updated)
    return assigned


__all__ = [
    "assign_session_disjoint_splits",
    "build_keep_best_window_row",
    "build_keep_prompt_row",
    "build_keep_window_rows",
    "deduplicate_prompt_rows",
]
