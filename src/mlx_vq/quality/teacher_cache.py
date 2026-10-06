from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_vq.io.source_safetensors import (
    SafetensorsTensorHeader,
    read_safetensors_tensor_header,
    read_safetensors_tensor_mlx,
)


ALLOWED_TEACHER_KINDS = {"bf16_source", "q8", "other_high_bit"}


def read_teacher_cache_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"teacher cache row {line_number} is invalid JSON: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"teacher cache row {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"teacher cache metadata {path} contained no rows")
    return rows


def _header_record(header: SafetensorsTensorHeader, *, tensor_name: str, path: Path) -> dict[str, Any]:
    return {
        "tensor": tensor_name,
        "path": str(path),
        "dtype": header.dtype,
        "shape": [int(dim) for dim in header.shape],
        "element_count": int(header.element_count),
    }


def _inline_record(values: Any) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "source": "inline",
        "dtype": str(array.dtype),
        "shape": [int(dim) for dim in array.shape],
        "element_count": int(array.size),
    }


def _check_int_list(row: dict[str, Any], key: str, errors: list[str]) -> list[int]:
    value = row.get(key)
    if not isinstance(value, list) or not value:
        errors.append(f"{key} must be a non-empty list")
        return []
    result: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            errors.append(f"{key}[{index}] must be an integer")
            continue
        result.append(int(item))
    return result


def _check_non_negative_ints(key: str, values: list[int], errors: list[str]) -> None:
    for index, value in enumerate(values):
        if value < 0:
            errors.append(f"{key}[{index}] must be non-negative, got {value}")


def _check_unique_ints(key: str, values: list[int], errors: list[str]) -> None:
    seen: set[int] = set()
    for index, value in enumerate(values):
        if value in seen:
            errors.append(f"{key}[{index}] duplicates earlier value {value}")
            continue
        seen.add(value)


def _check_shape(
    *,
    name: str,
    shape: list[int],
    expected_rank: int | None,
    expected_first_dim: int | None,
    errors: list[str],
) -> None:
    if expected_rank is not None and len(shape) != expected_rank:
        errors.append(f"{name} must have rank {expected_rank}, got shape {shape}")
    if expected_first_dim is not None and shape and shape[0] != expected_first_dim:
        errors.append(f"{name} has {shape[0]} positions, expected {expected_first_dim}")


_FLOAT_RECORD_DTYPES = {"BF16", "F16", "F32", "float16", "float32", "float64"}
_TOKEN_ID_RECORD_DTYPES = {"I32", "I64", "int32", "int64"}


def _check_record_dtype(
    record: dict[str, Any],
    *,
    name: str,
    expected: str,
    allowed: set[str],
    errors: list[str],
) -> None:
    dtype = str(record.get("dtype"))
    if dtype not in allowed:
        errors.append(f"{name} dtype {dtype} must be {expected}")


def _resolve_cache_shard_path(cache_root: Path, shard: str) -> Path:
    shard_path = Path(shard)
    if shard_path.is_absolute():
        raise ValueError(f"logit_shard must be relative to cache_root, got {shard!r}")
    root = cache_root.resolve(strict=False)
    candidate = (root / shard_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"logit_shard escapes cache_root: {shard!r}") from error
    return candidate


def _check_memory_counter(
    row: dict[str, Any],
    key: str,
    errors: list[str],
    warnings: list[str],
) -> int | None:
    if key not in row:
        errors.append(f"{key} must be recorded")
        return None
    value = row.get(key)
    if value is None:
        warnings.append(f"{key} is null; memory cleanliness is unknown")
        return None
    if not isinstance(value, int):
        errors.append(f"{key} must be an integer or null, got {type(value).__name__}")
        return None
    if value < 0:
        errors.append(f"{key} must be non-negative or null, got {value}")
        return int(value)
    if value != 0:
        warnings.append(f"{key} is nonzero: {value}")
    return int(value)


def _check_runtime_seconds(row: dict[str, Any], key: str, errors: list[str]) -> float | None:
    if key not in row:
        errors.append(f"{key} must be recorded")
        return None
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{key} must be a non-negative finite number")
        return None
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0:
        errors.append(f"{key} must be a non-negative finite number")
        return numeric
    return numeric


def _check_runtime_byte_counter(
    row: dict[str, Any],
    key: str,
    errors: list[str],
) -> int | None:
    if key not in row:
        errors.append(f"{key} must be recorded")
        return None
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{key} must be a non-negative integer")
        return None
    if value < 0:
        errors.append(f"{key} must be a non-negative integer")
        return int(value)
    return int(value)


def _check_vm_total_counter(
    row: dict[str, Any],
    key: str,
    errors: list[str],
    warnings: list[str],
) -> int | None:
    if key not in row:
        errors.append(f"{key} must be recorded")
        return None
    value = row.get(key)
    if value is None:
        warnings.append(f"{key} is null; VM totals are unknown")
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{key} must be a non-negative integer or null")
        return None
    if value < 0:
        errors.append(f"{key} must be a non-negative integer or null")
        return int(value)
    return int(value)


def _check_topk_width(
    *,
    topk_ids: dict[str, Any] | None,
    topk_logprobs: dict[str, Any] | None,
    logits: dict[str, Any] | None,
    min_top_k: int,
    errors: list[str],
) -> None:
    if min_top_k <= 0:
        errors.append(f"min_top_k must be positive, got {min_top_k}")
        return
    if topk_ids is None or topk_logprobs is None:
        return
    shape = topk_ids.get("shape", [])
    if len(shape) != 2:
        return
    expected_min = min_top_k
    if logits is not None:
        logit_shape = logits.get("shape", [])
        if len(logit_shape) == 2 and logit_shape[1] > 0:
            expected_min = min(expected_min, int(logit_shape[1]))
    if int(shape[1]) < expected_min:
        errors.append(f"topk width {shape[1]} must be at least {expected_min}")


def _read_validation_values(
    row: dict[str, Any],
    *,
    cache_root: Path,
    name: str,
    inline_key: str,
    tensor_key: str,
    default_tensor: str,
    dtype: np.dtype | type,
    errors: list[str],
) -> np.ndarray | None:
    try:
        return _inline_or_tensor(
            row,
            cache_root=cache_root,
            inline_key=inline_key,
            shard_key="logit_shard",
            tensor_key=tensor_key,
            default_tensor=default_tensor,
            dtype=dtype,
        )
    except (KeyError, ValueError) as error:
        errors.append(f"{name} values could not be read: {error}")
        return None


def _check_finite_values(name: str, values: np.ndarray | None, errors: list[str]) -> None:
    if values is not None and not bool(np.all(np.isfinite(values))):
        errors.append(f"{name} values must be finite")


def _check_logprob_values(name: str, values: np.ndarray | None, errors: list[str]) -> None:
    if values is not None and bool(np.any(values > 0.0)):
        errors.append(f"{name} values must be log-probabilities <= 0")


def _check_topk_probability_mass(
    name: str,
    values: np.ndarray | None,
    errors: list[str],
) -> None:
    if values is None or values.ndim != 2:
        return
    masses = np.sum(np.exp(values), axis=-1)
    for row_index, mass in enumerate(masses):
        if float(mass) > 1.00001:
            errors.append(
                f"{name}[{row_index}] probability mass must be <= 1.0, "
                f"got {float(mass):.9g}"
            )


def _check_topk_logprob_order(
    name: str,
    values: np.ndarray | None,
    errors: list[str],
) -> None:
    if values is None or values.ndim != 2:
        return
    tolerance = 1e-6
    for row_index, row in enumerate(values):
        for column_index in range(1, row.shape[0]):
            previous = float(row[column_index - 1])
            current = float(row[column_index])
            if not (np.isfinite(previous) and np.isfinite(current)):
                continue
            if current > previous + tolerance:
                errors.append(
                    f"{name}[{row_index}] must be sorted from highest to lowest "
                    f"log-probability; {name}[{row_index},{column_index}]={current} "
                    f"is greater than {name}[{row_index},{column_index - 1}]={previous}"
                )
                break


def _check_token_id_values(
    name: str,
    values: np.ndarray | None,
    *,
    vocab_size: int | None,
    errors: list[str],
) -> None:
    if values is None:
        return
    if bool(np.any(values < 0)):
        errors.append(f"{name} values must be non-negative")
    if vocab_size is not None and bool(np.any(values >= vocab_size)):
        errors.append(f"{name} values must be less than logits vocab size {vocab_size}")


def _check_unique_token_id_rows(
    name: str,
    values: np.ndarray | None,
    errors: list[str],
) -> None:
    if values is None or values.ndim != 2:
        return
    for row_index, row in enumerate(values):
        seen: set[int] = set()
        for value in row:
            token_id = int(value)
            if token_id in seen:
                errors.append(f"{name}[{row_index}] contains duplicate token id {token_id}")
                break
            seen.add(token_id)


def _check_target_logprobs_match_topk(
    *,
    topk_ids: np.ndarray | None,
    topk_logprobs: np.ndarray | None,
    target_logprobs: np.ndarray | None,
    target_token_ids: list[int],
    errors: list[str],
) -> None:
    if topk_ids is None or topk_logprobs is None or target_logprobs is None:
        return
    if topk_ids.ndim != 2 or topk_logprobs.ndim != 2 or target_logprobs.ndim != 1:
        return
    if topk_ids.shape != topk_logprobs.shape:
        return
    if (
        topk_ids.shape[0] != target_logprobs.shape[0]
        or topk_ids.shape[0] != len(target_token_ids)
    ):
        return
    for row_index, target_token_id in enumerate(target_token_ids):
        matches = np.flatnonzero(topk_ids[row_index] == int(target_token_id))
        if matches.size == 0:
            actual = float(target_logprobs[row_index])
            cutoff = float(topk_logprobs[row_index, -1])
            if (
                np.isfinite(actual)
                and np.isfinite(cutoff)
                and actual > cutoff + 1e-6
            ):
                errors.append(
                    f"target_logprobs[{row_index}]={actual} for target token id "
                    f"{int(target_token_id)} is absent from topk_ids[{row_index}] "
                    f"but must be <= topk_logprobs[{row_index},-1]={cutoff}"
                )
            continue
        topk_index = int(matches[0])
        expected = float(topk_logprobs[row_index, topk_index])
        actual = float(target_logprobs[row_index])
        if not bool(np.isclose(actual, expected, rtol=1e-5, atol=1e-6)):
            errors.append(
                f"target_logprobs[{row_index}]={actual} must match "
                f"topk_logprobs[{row_index},{topk_index}]={expected} "
                f"for target token id {int(target_token_id)}"
            )


def _check_teacher_top1_consistency(
    *,
    logits: np.ndarray | None,
    topk_ids: np.ndarray | None,
    topk_logprobs: np.ndarray | None,
    teacher_top1_ids: np.ndarray | None,
    errors: list[str],
) -> None:
    if teacher_top1_ids is None or teacher_top1_ids.ndim != 1:
        return
    if (
        logits is not None
        and logits.ndim == 2
        and logits.shape[0] == teacher_top1_ids.shape[0]
    ):
        expected_top1 = np.argmax(logits, axis=-1).astype(np.int64)
        for row_index, expected in enumerate(expected_top1):
            actual = int(teacher_top1_ids[row_index])
            if actual != int(expected):
                errors.append(
                    f"teacher_top1_ids[{row_index}]={actual} must match "
                    f"argmax(logits[{row_index}])={int(expected)}"
                )
    if (
        topk_ids is None
        or topk_ids.ndim != 2
        or topk_ids.shape[0] != teacher_top1_ids.shape[0]
    ):
        return
    for row_index, actual_value in enumerate(teacher_top1_ids):
        actual = int(actual_value)
        matches = np.flatnonzero(topk_ids[row_index] == actual)
        if matches.size == 0:
            errors.append(
                f"teacher_top1_ids[{row_index}]={actual} must appear in "
                f"topk_ids[{row_index}]"
            )
            continue
        first_ranked = int(topk_ids[row_index, 0])
        if actual == first_ranked:
            continue
        tied_with_first = False
        if (
            topk_logprobs is not None
            and topk_logprobs.ndim == 2
            and topk_logprobs.shape == topk_ids.shape
        ):
            first_logprob = float(topk_logprobs[row_index, 0])
            actual_logprob = float(topk_logprobs[row_index, int(matches[0])])
            tied_with_first = bool(
                np.isfinite(first_logprob)
                and np.isfinite(actual_logprob)
                and np.isclose(actual_logprob, first_logprob, rtol=1e-5, atol=1e-6)
            )
        if tied_with_first:
            continue
        errors.append(
            f"teacher_top1_ids[{row_index}]={actual} must match first-ranked "
            f"topk_ids[{row_index},0]={first_ranked}"
        )


def _check_tensor_header(
    row: dict[str, Any],
    *,
    cache_root: Path,
    shard: str | None,
    tensor_key: str,
    default_tensor: str,
    record_key: str,
    required: bool,
    expected_rank: int | None,
    expected_first_dim: int | None,
    errors: list[str],
) -> dict[str, Any] | None:
    tensor_name = row.get(tensor_key, default_tensor)
    if tensor_name is None:
        if required:
            errors.append(f"{record_key} tensor name is required")
        return None
    if not isinstance(tensor_name, str) or not tensor_name:
        errors.append(f"{record_key} tensor name must be a non-empty string")
        return None
    if shard is None:
        if required:
            errors.append(f"{record_key} requires logit_shard")
        return None
    try:
        path = _resolve_cache_shard_path(cache_root, shard)
    except ValueError as error:
        errors.append(str(error))
        return None
    if not path.exists():
        if required:
            errors.append(f"{record_key} shard is missing: {path}")
        return None
    try:
        header = read_safetensors_tensor_header(path, tensor_name)
    except KeyError:
        if required:
            errors.append(f"{record_key} tensor {tensor_name!r} is missing from {path}")
        return None
    except ValueError as error:
        errors.append(f"{record_key} shard is invalid: {error}")
        return None

    record = _header_record(header, tensor_name=tensor_name, path=path)
    _check_shape(
        name=record_key,
        shape=record["shape"],
        expected_rank=expected_rank,
        expected_first_dim=expected_first_dim,
        errors=errors,
    )
    return record


def _check_inline_or_tensor_header(
    row: dict[str, Any],
    *,
    cache_root: Path,
    shard: str | None,
    inline_key: str,
    tensor_key: str,
    default_tensor: str,
    record_key: str,
    required: bool,
    expected_rank: int | None,
    expected_first_dim: int | None,
    errors: list[str],
) -> dict[str, Any] | None:
    if inline_key in row and row[inline_key] is not None:
        record = _inline_record(row[inline_key])
        _check_shape(
            name=record_key,
            shape=record["shape"],
            expected_rank=expected_rank,
            expected_first_dim=expected_first_dim,
            errors=errors,
        )
        return record
    return _check_tensor_header(
        row,
        cache_root=cache_root,
        shard=shard,
        tensor_key=tensor_key,
        default_tensor=default_tensor,
        record_key=record_key,
        required=required,
        expected_rank=expected_rank,
        expected_first_dim=expected_first_dim,
        errors=errors,
    )


def validate_teacher_cache_row(
    row: dict[str, Any],
    *,
    cache_root: str | Path,
    row_index: int | None = None,
    min_top_k: int = 128,
    check_values: bool = False,
) -> dict[str, Any]:
    """Validate one off-box teacher-cache metadata row without loading the VQ model."""

    root = Path(cache_root)
    errors: list[str] = []
    warnings: list[str] = []
    elapsed_seconds = _check_runtime_seconds(row, "elapsed_seconds", errors)
    mlx_active_bytes = _check_runtime_byte_counter(row, "mlx_active_bytes", errors)
    mlx_peak_bytes = _check_runtime_byte_counter(row, "mlx_peak_bytes", errors)
    mlx_cache_bytes = _check_runtime_byte_counter(row, "mlx_cache_bytes", errors)
    rss_bytes = _check_runtime_byte_counter(row, "rss_bytes", errors)
    pageouts_total = _check_vm_total_counter(row, "pageouts_total", errors, warnings)
    swapouts_total = _check_vm_total_counter(row, "swapouts_total", errors, warnings)
    pageouts_delta = _check_memory_counter(row, "pageouts_delta", errors, warnings)
    swapouts_delta = _check_memory_counter(row, "swapouts_delta", errors, warnings)
    if row.get("schema_version") != 1:
        errors.append(f"schema_version must be 1, got {row.get('schema_version')!r}")
    for key in ("model_id", "revision", "teacher_kind", "prompt_id"):
        if not isinstance(row.get(key), str) or not row.get(key):
            errors.append(f"{key} must be a non-empty string")
    teacher_kind = row.get("teacher_kind")
    if isinstance(teacher_kind, str) and teacher_kind not in ALLOWED_TEACHER_KINDS:
        errors.append(
            f"teacher_kind {teacher_kind!r} is not an allowed high-bit authority "
            f"({', '.join(sorted(ALLOWED_TEACHER_KINDS))})"
        )
    full_logits_flag = row.get("full_logits_available")
    if not isinstance(full_logits_flag, bool):
        errors.append(
            "full_logits_available must be a boolean, "
            f"got {type(full_logits_flag).__name__}"
        )
    full_logits_requested = full_logits_flag is True

    input_token_ids = _check_int_list(row, "input_token_ids", errors)
    target_token_ids = _check_int_list(row, "target_token_ids", errors)
    positions = _check_int_list(row, "positions", errors)
    _check_non_negative_ints("input_token_ids", input_token_ids, errors)
    _check_non_negative_ints("target_token_ids", target_token_ids, errors)
    _check_non_negative_ints("positions", positions, errors)
    _check_unique_ints("positions", positions, errors)
    if positions and target_token_ids and len(positions) != len(target_token_ids):
        errors.append(
            f"positions length {len(positions)} must match target_token_ids length {len(target_token_ids)}"
        )
    if input_token_ids and positions and max(positions) >= len(input_token_ids) - 1:
        errors.append(
            f"max position {max(positions)} must be less than input_token_ids length - 1"
        )
    if input_token_ids and positions and target_token_ids and len(positions) == len(target_token_ids):
        for index, (position, target_token_id) in enumerate(zip(positions, target_token_ids)):
            if position < 0:
                continue
            if position + 1 >= len(input_token_ids):
                continue
            expected_target = input_token_ids[position + 1]
            if target_token_id != expected_target:
                errors.append(
                    "target_token_ids"
                    f"[{index}]={target_token_id} must match input_token_ids"
                    f"[{position + 1}]={expected_target} for position {position}"
                )

    position_count = len(target_token_ids)
    shard = row.get("logit_shard")
    if not isinstance(shard, str) or not shard:
        errors.append("logit_shard must be a non-empty string")
        shard = None

    tensors: dict[str, dict[str, Any]] = {}
    logits = _check_tensor_header(
        row,
        cache_root=root,
        shard=shard,
        tensor_key="logit_tensor",
        default_tensor="logits",
        record_key="logits",
        required=full_logits_requested,
        expected_rank=2,
        expected_first_dim=position_count or None,
        errors=errors,
    )
    if logits is not None:
        tensors["logits"] = logits
        _check_record_dtype(
            logits,
            name="logits",
            expected="floating point",
            allowed=_FLOAT_RECORD_DTYPES,
            errors=errors,
        )
        logit_shape = logits.get("shape", [])
        if len(logit_shape) == 2 and int(logit_shape[1]) > 0:
            vocab_size = int(logit_shape[1])
            for index, target_token_id in enumerate(target_token_ids):
                if target_token_id >= vocab_size:
                    errors.append(
                        f"target_token_ids[{index}]={target_token_id} must be less than "
                        f"logits vocab size {vocab_size}"
                    )

    topk_ids = _check_inline_or_tensor_header(
        row,
        cache_root=root,
        shard=shard,
        inline_key="topk_ids",
        tensor_key="topk_tensor",
        default_tensor="topk_ids",
        record_key="topk_ids",
        required=True,
        expected_rank=2,
        expected_first_dim=position_count or None,
        errors=errors,
    )
    if topk_ids is not None:
        tensors["topk_ids"] = topk_ids
        _check_record_dtype(
            topk_ids,
            name="topk_ids",
            expected="integer token ids",
            allowed=_TOKEN_ID_RECORD_DTYPES,
            errors=errors,
        )

    topk_logprobs = _check_inline_or_tensor_header(
        row,
        cache_root=root,
        shard=shard,
        inline_key="topk_logprobs",
        tensor_key="topk_logprob_tensor",
        default_tensor="topk_logprobs",
        record_key="topk_logprobs",
        required=True,
        expected_rank=2,
        expected_first_dim=position_count or None,
        errors=errors,
    )
    if topk_logprobs is not None:
        tensors["topk_logprobs"] = topk_logprobs
        _check_record_dtype(
            topk_logprobs,
            name="topk_logprobs",
            expected="floating point log-probabilities",
            allowed=_FLOAT_RECORD_DTYPES,
            errors=errors,
        )

    target_logprobs = _check_inline_or_tensor_header(
        row,
        cache_root=root,
        shard=shard,
        inline_key="target_logprobs",
        tensor_key="target_logprob_tensor",
        default_tensor="target_logprobs",
        record_key="target_logprobs",
        required=True,
        expected_rank=1,
        expected_first_dim=position_count or None,
        errors=errors,
    )
    if target_logprobs is not None:
        tensors["target_logprobs"] = target_logprobs
        _check_record_dtype(
            target_logprobs,
            name="target_logprobs",
            expected="floating point log-probabilities",
            allowed=_FLOAT_RECORD_DTYPES,
            errors=errors,
        )

    teacher_top1 = _check_inline_or_tensor_header(
        row,
        cache_root=root,
        shard=shard,
        inline_key="teacher_top1_ids",
        tensor_key="teacher_top1_tensor",
        default_tensor="teacher_top1_ids",
        record_key="teacher_top1_ids",
        required=True,
        expected_rank=1,
        expected_first_dim=position_count or None,
        errors=errors,
    )
    if teacher_top1 is not None:
        tensors["teacher_top1_ids"] = teacher_top1
        _check_record_dtype(
            teacher_top1,
            name="teacher_top1_ids",
            expected="integer token ids",
            allowed=_TOKEN_ID_RECORD_DTYPES,
            errors=errors,
        )

    if topk_ids is not None and topk_logprobs is not None:
        if topk_ids["shape"] != topk_logprobs["shape"]:
            errors.append(
                f"topk_ids shape {topk_ids['shape']} must match topk_logprobs shape {topk_logprobs['shape']}"
            )
        _check_topk_width(
            topk_ids=topk_ids,
            topk_logprobs=topk_logprobs,
            logits=logits,
            min_top_k=min_top_k,
            errors=errors,
        )

    vocab_size_for_values: int | None = None
    if logits is not None:
        logit_shape = logits.get("shape", [])
        if len(logit_shape) == 2 and int(logit_shape[1]) > 0:
            vocab_size_for_values = int(logit_shape[1])

    if check_values:
        logits_values: np.ndarray | None = None
        topk_id_values: np.ndarray | None = None
        topk_logprob_values: np.ndarray | None = None
        target_logprob_values: np.ndarray | None = None
        teacher_top1_values: np.ndarray | None = None
        if logits is not None:
            logits_values = _read_validation_values(
                row,
                cache_root=root,
                name="logits",
                inline_key="logits",
                tensor_key="logit_tensor",
                default_tensor="logits",
                dtype=np.float64,
                errors=errors,
            )
            _check_finite_values("logits", logits_values, errors)
        if topk_ids is not None:
            topk_id_values = _read_validation_values(
                row,
                cache_root=root,
                name="topk_ids",
                inline_key="topk_ids",
                tensor_key="topk_tensor",
                default_tensor="topk_ids",
                dtype=np.int64,
                errors=errors,
            )
            _check_token_id_values(
                "topk_ids",
                topk_id_values,
                vocab_size=vocab_size_for_values,
                errors=errors,
            )
            _check_unique_token_id_rows("topk_ids", topk_id_values, errors)
        if topk_logprobs is not None:
            topk_logprob_values = _read_validation_values(
                row,
                cache_root=root,
                name="topk_logprobs",
                inline_key="topk_logprobs",
                tensor_key="topk_logprob_tensor",
                default_tensor="topk_logprobs",
                dtype=np.float64,
                errors=errors,
            )
            _check_finite_values("topk_logprobs", topk_logprob_values, errors)
            _check_logprob_values("topk_logprobs", topk_logprob_values, errors)
            _check_topk_probability_mass("topk_logprobs", topk_logprob_values, errors)
            _check_topk_logprob_order("topk_logprobs", topk_logprob_values, errors)
        if target_logprobs is not None:
            target_logprob_values = _read_validation_values(
                row,
                cache_root=root,
                name="target_logprobs",
                inline_key="target_logprobs",
                tensor_key="target_logprob_tensor",
                default_tensor="target_logprobs",
                dtype=np.float64,
                errors=errors,
            )
            _check_finite_values("target_logprobs", target_logprob_values, errors)
            _check_logprob_values("target_logprobs", target_logprob_values, errors)
        _check_target_logprobs_match_topk(
            topk_ids=topk_id_values,
            topk_logprobs=topk_logprob_values,
            target_logprobs=target_logprob_values,
            target_token_ids=target_token_ids,
            errors=errors,
        )
        if teacher_top1 is not None:
            teacher_top1_values = _read_validation_values(
                row,
                cache_root=root,
                name="teacher_top1_ids",
                inline_key="teacher_top1_ids",
                tensor_key="teacher_top1_tensor",
                default_tensor="teacher_top1_ids",
                dtype=np.int64,
                errors=errors,
            )
            _check_token_id_values(
                "teacher_top1_ids",
                teacher_top1_values,
                vocab_size=vocab_size_for_values,
                errors=errors,
            )
        _check_teacher_top1_consistency(
            logits=logits_values,
            topk_ids=topk_id_values,
            topk_logprobs=topk_logprob_values,
            teacher_top1_ids=teacher_top1_values,
            errors=errors,
        )

    full_logits_available = logits is not None
    if full_logits_requested and not full_logits_available:
        warnings.append("full_logits_available was true but logits were not usable")
    if full_logits_flag is False and full_logits_available:
        errors.append("full_logits_available was false but logits were usable")
    kld_mode = "exact_full_logits" if full_logits_available else "teacher_topk_lower_bound"
    if topk_ids is None or topk_logprobs is None:
        kld_mode = "unavailable" if not full_logits_available else kld_mode

    return {
        "schema_version": 1,
        "record_type": "teacher_cache_validation_row",
        "row_index": row_index,
        "prompt_id": row.get("prompt_id"),
        "model_id": row.get("model_id"),
        "revision": row.get("revision"),
        "teacher_kind": row.get("teacher_kind"),
        "cache_root": str(root),
        "shard": shard,
        "position_count": int(position_count),
        "input_token_count": int(len(input_token_ids)),
        "target_token_count": int(len(target_token_ids)),
        "elapsed_seconds": elapsed_seconds,
        "mlx_active_bytes": mlx_active_bytes,
        "mlx_peak_bytes": mlx_peak_bytes,
        "mlx_cache_bytes": mlx_cache_bytes,
        "rss_bytes": rss_bytes,
        "pageouts_total": pageouts_total,
        "swapouts_total": swapouts_total,
        "pageouts_delta": pageouts_delta,
        "swapouts_delta": swapouts_delta,
        "memory_counters_known": pageouts_delta is not None and swapouts_delta is not None,
        "memory_clean": pageouts_delta == 0 and swapouts_delta == 0,
        "min_top_k": int(min_top_k),
        "value_checks_enabled": bool(check_values),
        "full_logits_available": bool(full_logits_available),
        "kld_mode": kld_mode,
        "tensors": tensors,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def validate_teacher_cache_metadata(
    path: str | Path,
    *,
    cache_root: str | Path | None = None,
    min_top_k: int = 128,
    check_values: bool = False,
    row_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Validate a teacher-cache metadata JSONL and referenced tensor headers."""

    metadata_path = Path(path)
    root = Path(cache_root) if cache_root is not None else metadata_path.parent
    rows = read_teacher_cache_rows(metadata_path)
    indexed_rows = (
        [(index, rows[index]) for index in row_indices]
        if row_indices is not None
        else list(enumerate(rows))
    )
    reports = [
        validate_teacher_cache_row(
            row,
            cache_root=root,
            row_index=index,
            min_top_k=min_top_k,
            check_values=check_values,
        )
        for index, row in indexed_rows
    ]
    ok_reports = [report for report in reports if report["ok"]]
    kld_modes: dict[str, int] = {}
    for report in reports:
        mode = str(report["kld_mode"])
        kld_modes[mode] = kld_modes.get(mode, 0) + 1
    summary = {
        "schema_version": 1,
        "record_type": "teacher_cache_validation_summary",
        "teacher_jsonl": str(metadata_path),
        "cache_root": str(root),
        "row_count": len(reports),
        "source_row_count": len(rows),
        "selected_row_indices": list(row_indices) if row_indices is not None else None,
        "ok_count": len(ok_reports),
        "error_count": len(reports) - len(ok_reports),
        "ok": len(ok_reports) == len(reports),
        "min_top_k": int(min_top_k),
        "check_values": bool(check_values),
        "memory_counter_row_count": sum(
            1 for report in reports if report["memory_counters_known"]
        ),
        "memory_clean_row_count": sum(1 for report in reports if report["memory_clean"]),
        "all_memory_clean": all(report["memory_clean"] for report in reports),
        "kld_modes": kld_modes,
        "full_logits_row_count": sum(1 for report in reports if report["full_logits_available"]),
        "topk_only_row_count": sum(
            1 for report in reports
            if report["ok"] and report["kld_mode"] == "teacher_topk_lower_bound"
        ),
        "rows": reports,
    }
    json.dumps(summary, sort_keys=True)
    return summary


def _to_numpy_float(array: mx.array) -> np.ndarray:
    return np.array(array.astype(mx.float32), copy=False).astype(np.float64, copy=False)


def _load_tensor(
    row: dict[str, Any],
    *,
    cache_root: Path,
    shard_key: str,
    tensor_key: str,
    default_tensor: str | None = None,
) -> np.ndarray | None:
    shard = row.get(shard_key)
    explicit_tensor = tensor_key in row
    tensor = row.get(tensor_key, default_tensor)
    if shard is None or tensor is None:
        return None
    path = _resolve_cache_shard_path(cache_root, str(shard))
    try:
        return _to_numpy_float(read_safetensors_tensor_mlx(path, str(tensor)))
    except KeyError:
        if not explicit_tensor and default_tensor is not None:
            return None
        raise


def _inline_or_tensor(
    row: dict[str, Any],
    *,
    cache_root: Path,
    inline_key: str,
    shard_key: str,
    tensor_key: str,
    default_tensor: str | None = None,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray | None:
    if inline_key in row and row[inline_key] is not None:
        return np.asarray(row[inline_key], dtype=dtype)
    loaded = _load_tensor(
        row,
        cache_root=cache_root,
        shard_key=shard_key,
        tensor_key=tensor_key,
        default_tensor=default_tensor,
    )
    if loaded is None:
        return None
    return loaded.astype(dtype, copy=False)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _topk_indices(values: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    count = min(top_k, values.shape[-1])
    partition = np.argpartition(-values, count - 1, axis=-1)[..., :count]
    partition_values = np.take_along_axis(values, partition, axis=-1)
    order = np.argsort(-partition_values, axis=-1)
    return np.take_along_axis(partition, order, axis=-1)


def _take_targets(log_probs: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    if log_probs.shape[0] != target_ids.shape[0]:
        raise ValueError(
            f"log-prob positions {log_probs.shape[0]} do not match targets {target_ids.shape[0]}"
        )
    return log_probs[np.arange(target_ids.shape[0]), target_ids]


def _safe_exp(value: float | None) -> float | None:
    if value is None:
        return None
    return float(np.exp(value))


def _percentile_999(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values, 0.999))


def _shape_positions(values: np.ndarray | None, positions: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    if values.shape[0] != positions:
        raise ValueError(f"{name} has {values.shape[0]} positions, expected {positions}")
    return values


def build_teacher_cache_payload(
    *,
    logits: mx.array | np.ndarray,
    input_token_ids: list[int],
    prompt_id: str,
    model_id: str,
    revision: str,
    teacher_kind: str,
    shard_path: str,
    top_k: int = 128,
    max_positions: int | None = 128,
    save_full_logits: bool = True,
) -> tuple[dict[str, Any], dict[str, mx.array]]:
    """Build one teacher-cache metadata row and safetensors tensor payload."""

    if len(input_token_ids) < 2:
        raise ValueError("teacher cache prompts need at least two input tokens")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_positions is not None and max_positions <= 0:
        raise ValueError("max_positions must be positive when provided")
    values = (
        _to_numpy_float(logits)
        if isinstance(logits, mx.array)
        else np.asarray(logits, dtype=np.float64)
    )
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise ValueError(f"logits batch must be 1, got {values.shape}")
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"logits must have shape [tokens, vocab], got {values.shape}")

    position_count = min(len(input_token_ids) - 1, values.shape[0])
    if max_positions is not None:
        position_count = min(position_count, max_positions)
    if position_count <= 0:
        raise ValueError("no teacher positions selected")

    selected_logits = values[:position_count]
    target_ids = np.asarray(input_token_ids[1 : position_count + 1], dtype=np.int64)
    log_probs = _log_softmax(selected_logits)
    target_logprobs = _take_targets(log_probs, target_ids)
    topk_ids = _topk_indices(log_probs, top_k).astype(np.int32)
    topk_logprobs = np.take_along_axis(log_probs, topk_ids, axis=-1).astype(np.float32)
    teacher_top1_ids = np.argmax(selected_logits, axis=-1).astype(np.int32)

    tensors: dict[str, mx.array] = {
        "topk_ids": mx.array(topk_ids),
        "topk_logprobs": mx.array(topk_logprobs),
        "target_logprobs": mx.array(target_logprobs.astype(np.float32)),
        "teacher_top1_ids": mx.array(teacher_top1_ids),
    }
    if save_full_logits:
        tensors["logits"] = mx.array(selected_logits.astype(np.float16))

    row = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision,
        "teacher_kind": teacher_kind,
        "prompt_id": prompt_id,
        "input_token_ids": [int(token_id) for token_id in input_token_ids],
        "target_token_ids": [int(token_id) for token_id in target_ids.tolist()],
        "positions": list(range(position_count)),
        "logit_shard": shard_path,
        "logit_tensor": "logits",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "teacher_top1_tensor": "teacher_top1_ids",
        "full_logits_available": bool(save_full_logits),
    }
    json.dumps(row, sort_keys=True)
    return row, tensors


def build_teacher_cache_topk_payload(
    *,
    topk_ids: mx.array | np.ndarray,
    topk_logprobs: mx.array | np.ndarray,
    target_logprobs: mx.array | np.ndarray,
    teacher_top1_ids: mx.array | np.ndarray,
    input_token_ids: list[int],
    prompt_id: str,
    model_id: str,
    revision: str,
    teacher_kind: str,
    shard_path: str,
    max_positions: int | None = 128,
) -> tuple[dict[str, Any], dict[str, mx.array]]:
    """Build a top-k-only teacher-cache payload without resident full logits."""

    if len(input_token_ids) < 2:
        raise ValueError("teacher cache prompts need at least two input tokens")
    if max_positions is not None and max_positions <= 0:
        raise ValueError("max_positions must be positive when provided")
    topk_id_values = (
        np.asarray(topk_ids, dtype=np.int32)
        if not isinstance(topk_ids, mx.array)
        else np.asarray(topk_ids, dtype=np.int32)
    )
    topk_logprob_values = (
        _to_numpy_float(topk_logprobs).astype(np.float32)
        if isinstance(topk_logprobs, mx.array)
        else np.asarray(topk_logprobs, dtype=np.float32)
    )
    target_logprob_values = (
        _to_numpy_float(target_logprobs).astype(np.float32)
        if isinstance(target_logprobs, mx.array)
        else np.asarray(target_logprobs, dtype=np.float32)
    )
    teacher_top1_values = (
        np.asarray(teacher_top1_ids, dtype=np.int32)
        if not isinstance(teacher_top1_ids, mx.array)
        else np.asarray(teacher_top1_ids, dtype=np.int32)
    )
    if topk_id_values.ndim != 2:
        raise ValueError(f"topk_ids must have shape [positions, top_k], got {topk_id_values.shape}")
    if topk_logprob_values.shape != topk_id_values.shape:
        raise ValueError(
            "topk_logprobs must have the same shape as topk_ids, "
            f"got {topk_logprob_values.shape} vs {topk_id_values.shape}"
        )
    position_count = int(topk_id_values.shape[0])
    if position_count <= 0:
        raise ValueError("no teacher positions selected")
    if target_logprob_values.shape != (position_count,):
        raise ValueError(
            f"target_logprobs must have shape [{position_count}], got {target_logprob_values.shape}"
        )
    if teacher_top1_values.shape != (position_count,):
        raise ValueError(
            f"teacher_top1_ids must have shape [{position_count}], got {teacher_top1_values.shape}"
        )
    if not bool(np.all(np.isfinite(topk_logprob_values))):
        raise ValueError("topk_logprobs values must be finite")
    if not bool(np.all(np.isfinite(target_logprob_values))):
        raise ValueError("target_logprobs values must be finite")
    if position_count > len(input_token_ids) - 1:
        raise ValueError(
            f"payload has {position_count} positions but only {len(input_token_ids) - 1} targets"
        )
    if max_positions is not None and position_count > max_positions:
        raise ValueError(f"payload has {position_count} positions above max_positions={max_positions}")

    target_ids = np.asarray(input_token_ids[1 : position_count + 1], dtype=np.int64)
    tensors: dict[str, mx.array] = {
        "topk_ids": mx.array(topk_id_values.astype(np.int32, copy=False)),
        "topk_logprobs": mx.array(topk_logprob_values.astype(np.float32, copy=False)),
        "target_logprobs": mx.array(target_logprob_values.astype(np.float32, copy=False)),
        "teacher_top1_ids": mx.array(teacher_top1_values.astype(np.int32, copy=False)),
    }

    row = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision,
        "teacher_kind": teacher_kind,
        "prompt_id": prompt_id,
        "input_token_ids": [int(token_id) for token_id in input_token_ids],
        "target_token_ids": [int(token_id) for token_id in target_ids.tolist()],
        "positions": list(range(position_count)),
        "logit_shard": shard_path,
        "logit_tensor": "logits",
        "topk_tensor": "topk_ids",
        "topk_logprob_tensor": "topk_logprobs",
        "target_logprob_tensor": "target_logprobs",
        "teacher_top1_tensor": "teacher_top1_ids",
        "full_logits_available": False,
    }
    json.dumps(row, sort_keys=True)
    return row, tensors


def evaluate_teacher_cache_row(
    row: dict[str, Any],
    *,
    vq_logits: mx.array | np.ndarray,
    cache_root: str | Path,
    extra: dict[str, Any] | None = None,
    watch_token_ids: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Compare selected VQ logits against one off-box teacher-cache metadata row."""

    root = Path(cache_root)
    target_ids = np.asarray(row.get("target_token_ids"), dtype=np.int64)
    if target_ids.ndim != 1 or target_ids.size == 0:
        raise ValueError("teacher cache row must contain non-empty target_token_ids")
    positions = np.asarray(row.get("positions", list(range(target_ids.size))), dtype=np.int64)
    if positions.shape[0] != target_ids.shape[0]:
        raise ValueError("positions and target_token_ids must have the same length")

    vq_values = (
        _to_numpy_float(vq_logits)
        if isinstance(vq_logits, mx.array)
        else np.asarray(vq_logits, dtype=np.float64)
    )
    if vq_values.ndim != 2:
        raise ValueError(f"vq_logits must have shape [positions, vocab], got {vq_values.shape}")
    if vq_values.shape[0] != target_ids.shape[0]:
        raise ValueError(
            f"vq_logits positions {vq_values.shape[0]} do not match targets {target_ids.shape[0]}"
        )

    vq_log_probs = _log_softmax(vq_values)
    vq_target_logprobs = _take_targets(vq_log_probs, target_ids)
    vq_nll = float(-np.mean(vq_target_logprobs))

    teacher_logits = (
        None
        if row.get("full_logits_available") is False
        else _load_tensor(
            row,
            cache_root=root,
            shard_key="logit_shard",
            tensor_key="logit_tensor",
            default_tensor="logits",
        )
    )
    teacher_logits = _shape_positions(teacher_logits, target_ids.size, "teacher logits")
    teacher_log_probs: np.ndarray | None = None
    if teacher_logits is not None:
        teacher_log_probs = _log_softmax(teacher_logits)

    teacher_target_logprobs = _inline_or_tensor(
        row,
        cache_root=root,
        inline_key="target_logprobs",
        shard_key="logit_shard",
        tensor_key="target_logprob_tensor",
        default_tensor="target_logprobs",
    )
    teacher_target_logprobs = _shape_positions(
        teacher_target_logprobs,
        target_ids.size,
        "teacher target logprobs",
    )
    if teacher_target_logprobs is None:
        if teacher_log_probs is None:
            raise ValueError(
                "teacher cache row needs target_logprobs or full logits to compute teacher NLL"
            )
        teacher_target_logprobs = _take_targets(teacher_log_probs, target_ids)
    teacher_nll = float(-np.mean(teacher_target_logprobs))

    teacher_top1_ids = _inline_or_tensor(
        row,
        cache_root=root,
        inline_key="teacher_top1_ids",
        shard_key="logit_shard",
        tensor_key="teacher_top1_tensor",
        default_tensor="teacher_top1_ids",
        dtype=np.int64,
    )
    teacher_top1_ids = _shape_positions(teacher_top1_ids, target_ids.size, "teacher top1 ids")
    if teacher_top1_ids is None:
        if teacher_logits is None:
            teacher_top1_ids = np.full((target_ids.size,), -1, dtype=np.int64)
        else:
            teacher_top1_ids = np.argmax(teacher_logits, axis=-1).astype(np.int64)
    vq_top1_ids = np.argmax(vq_values, axis=-1).astype(np.int64)
    top1_known = teacher_top1_ids >= 0
    top1_agreement = (
        float(np.mean(vq_top1_ids[top1_known] == teacher_top1_ids[top1_known]))
        if bool(np.any(top1_known))
        else None
    )
    vq_top1_logprobs = np.take_along_axis(
        vq_log_probs,
        vq_top1_ids[:, None],
        axis=-1,
    )[:, 0]
    vq_teacher_top1_logprobs: list[float | None] = []
    vq_teacher_top1_margins: list[float | None] = []
    for position_index, teacher_token_id in enumerate(teacher_top1_ids.tolist()):
        if int(teacher_token_id) < 0:
            vq_teacher_top1_logprobs.append(None)
            vq_teacher_top1_margins.append(None)
            continue
        teacher_logprob = float(vq_log_probs[position_index, int(teacher_token_id)])
        vq_teacher_top1_logprobs.append(teacher_logprob)
        vq_teacher_top1_margins.append(teacher_logprob - float(vq_top1_logprobs[position_index]))
    watch_token_logprobs: dict[str, list[float]] = {}
    watch_token_margins: dict[str, list[float]] = {}
    watch_token_required_bias: dict[str, list[float]] = {}
    normalized_watch_token_ids: list[int] = []
    if watch_token_ids is not None:
        vocab_size = int(vq_values.shape[1])
        normalized_watch_token_ids = [int(token_id) for token_id in watch_token_ids]
        for token_id in normalized_watch_token_ids:
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(
                    f"watch token id {token_id} is outside VQ logits vocab size {vocab_size}"
                )
            token_logprobs = vq_log_probs[:, token_id]
            margins = token_logprobs - vq_top1_logprobs
            required_bias = vq_top1_logprobs - token_logprobs
            key = str(token_id)
            watch_token_logprobs[key] = [float(value) for value in token_logprobs.tolist()]
            watch_token_margins[key] = [float(value) for value in margins.tolist()]
            watch_token_required_bias[key] = [float(value) for value in required_bias.tolist()]

    kld_values: np.ndarray | None = None
    kld_mode = "unavailable"
    teacher_topk_probability_mass: float | None = None
    if teacher_log_probs is not None:
        teacher_probs = np.exp(teacher_log_probs)
        kld_values = np.sum(teacher_probs * (teacher_log_probs - vq_log_probs), axis=-1)
        kld_mode = "exact_full_logits"
    else:
        topk_ids = _inline_or_tensor(
            row,
            cache_root=root,
            inline_key="topk_ids",
            shard_key="logit_shard",
            tensor_key="topk_tensor",
            default_tensor="topk_ids",
            dtype=np.int64,
        )
        topk_logprobs = _inline_or_tensor(
            row,
            cache_root=root,
            inline_key="topk_logprobs",
            shard_key="logit_shard",
            tensor_key="topk_logprob_tensor",
            default_tensor="topk_logprobs",
        )
        if topk_ids is not None and topk_logprobs is not None:
            topk_ids = _shape_positions(topk_ids, target_ids.size, "topk ids")
            topk_logprobs = _shape_positions(topk_logprobs, target_ids.size, "topk logprobs")
            assert topk_ids is not None
            assert topk_logprobs is not None
            if topk_ids.shape != topk_logprobs.shape:
                raise ValueError("topk_ids and topk_logprobs must have the same shape")
            selected_vq_logprobs = np.take_along_axis(vq_log_probs, topk_ids, axis=-1)
            topk_probs = np.exp(topk_logprobs)
            teacher_topk_probability_mass = float(np.mean(np.sum(topk_probs, axis=-1)))
            kld_values = np.sum(topk_probs * (topk_logprobs - selected_vq_logprobs), axis=-1)
            kld_mode = "teacher_topk_lower_bound"

    kld_mean = float(np.mean(kld_values)) if kld_values is not None else None
    kld_p999 = _percentile_999(kld_values) if kld_values is not None else None
    nll_delta = vq_nll - teacher_nll
    record: dict[str, Any] = {
        "schema_version": 1,
        "model_id": row.get("model_id"),
        "revision": row.get("revision"),
        "teacher_kind": row.get("teacher_kind"),
        "prompt_id": row.get("prompt_id"),
        "positions": [int(pos) for pos in positions.tolist()],
        "target_token_ids": [int(token_id) for token_id in target_ids.tolist()],
        "target_token_count": int(target_ids.size),
        "vq_nll": vq_nll,
        "teacher_nll": teacher_nll,
        "nll_delta": nll_delta,
        "vq_perplexity": _safe_exp(vq_nll),
        "teacher_perplexity": _safe_exp(teacher_nll),
        "ppl_ratio": _safe_exp(nll_delta),
        "vq_target_logprobs": [float(value) for value in vq_target_logprobs.tolist()],
        "teacher_target_logprobs": [
            float(value) for value in teacher_target_logprobs.tolist()
        ],
        "vq_top1_ids": [int(value) for value in vq_top1_ids.tolist()],
        "teacher_top1_ids": [int(value) for value in teacher_top1_ids.tolist()],
        "vq_teacher_top1_logprobs": vq_teacher_top1_logprobs,
        "vq_top1_logprobs": [float(value) for value in vq_top1_logprobs.tolist()],
        "vq_teacher_top1_margin_vs_vq_top1": vq_teacher_top1_margins,
        "top1_agreement": top1_agreement,
        "kld_mode": kld_mode,
        "mean_kld": kld_mean,
        "p999_kld": kld_p999,
        "token_klds": (
            [float(value) for value in kld_values.tolist()] if kld_values is not None else None
        ),
        "teacher_topk_probability_mass": teacher_topk_probability_mass,
        "full_logits_available": bool(teacher_logits is not None),
    }
    if watch_token_ids is not None:
        record.update(
            {
                "vq_watch_token_ids": normalized_watch_token_ids,
                "vq_watch_token_logprobs": watch_token_logprobs,
                "vq_watch_token_margin_vs_vq_top1": watch_token_margins,
                "vq_watch_token_required_bias_to_vq_top1": watch_token_required_bias,
            }
        )
    if extra:
        record.update(extra)
    json.dumps(record, sort_keys=True)
    return record


def summarize_teacher_cache_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize empty teacher-cache result set")
    clean_records = [
        record for record in records
        if record.get("pageouts_delta") == 0 and record.get("swapouts_delta") == 0
    ]
    nll_deltas = np.asarray([float(record["nll_delta"]) for record in records], dtype=np.float64)
    ppl_ratios = np.asarray([float(record["ppl_ratio"]) for record in records], dtype=np.float64)
    klds = np.asarray(
        [float(record["mean_kld"]) for record in records if record.get("mean_kld") is not None],
        dtype=np.float64,
    )
    token_klds = np.asarray(
        [
            float(value)
            for record in records
            for value in (record.get("token_klds") or [])
        ],
        dtype=np.float64,
    )
    top1 = np.asarray(
        [
            float(record["top1_agreement"])
            for record in records
            if record.get("top1_agreement") is not None
        ],
        dtype=np.float64,
    )
    row_distributions = {
        "ppl_ratio": _row_distribution(ppl_ratios),
        "mean_kld": _row_distribution(klds),
        "top1_agreement": _row_distribution(top1),
        "nll_delta": _row_distribution(nll_deltas),
    }
    return {
        "schema_version": 1,
        "record_count": len(records),
        "clean_record_count": len(clean_records),
        "mean_nll_delta": float(np.mean(nll_deltas)),
        "max_nll_delta": float(np.max(nll_deltas)),
        "mean_ppl_ratio": float(np.mean(ppl_ratios)),
        "max_ppl_ratio": float(np.max(ppl_ratios)),
        "mean_kld": float(np.mean(klds)) if klds.size else None,
        "p999_kld": _percentile_999(token_klds) if token_klds.size else None,
        "mean_top1_agreement": float(np.mean(top1)) if top1.size else None,
        "row_distributions": row_distributions,
        "all_memory_clean": len(clean_records) == len(records),
    }


def _row_distribution(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {
            "median": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }
