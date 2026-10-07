from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            row["_source_jsonl"] = str(path)
            rows.append(row)
    return rows


def _list_item(row: dict[str, Any], key: str, index: int) -> Any:
    values = row.get(key)
    if not isinstance(values, list):
        raise ValueError(f"row is missing list field {key!r}")
    if index < 0 or index >= len(values):
        raise ValueError(f"focus_position {index} out of range for {key!r} length {len(values)}")
    return values[index]


def _memory_clean(row: dict[str, Any]) -> bool:
    return int(row.get("pageouts_delta") or 0) == 0 and int(row.get("swapouts_delta") or 0) == 0


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _is_eval_row(row: dict[str, Any]) -> bool:
    return (
        isinstance(row.get("row_index"), int)
        and isinstance(row.get("positions"), list)
        and isinstance(row.get("teacher_top1_ids"), list)
        and isinstance(row.get("vq_top1_ids"), list)
        and isinstance(row.get("token_klds"), list)
        and "top1_agreement" in row
        and "mean_kld" in row
    )


def _read_eval_rows_by_index(paths: list[Path], *, label: str) -> dict[int, dict[str, Any]]:
    rows_by_index: dict[int, dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            if not _is_eval_row(row):
                continue
            row_index = int(row["row_index"])
            if row_index in rows_by_index:
                raise ValueError(f"duplicate {label} eval row_index {row_index}")
            rows_by_index[row_index] = row
    if not rows_by_index:
        raise ValueError(f"no {label} eval rows found")
    return rows_by_index


def _require_equal_length(row: dict[str, Any], keys: tuple[str, ...]) -> int:
    lengths = []
    for key in keys:
        values = row.get(key)
        if not isinstance(values, list):
            raise ValueError(f"row {row.get('row_index')} is missing list field {key!r}")
        lengths.append(len(values))
    if len(set(lengths)) != 1:
        raise ValueError(f"row {row.get('row_index')} has mismatched list lengths for {keys}")
    return lengths[0]


def _mean_delta(
    paired_rows: list[tuple[dict[str, Any], dict[str, Any]]],
    key: str,
) -> float | None:
    deltas: list[float] = []
    for baseline, candidate in paired_rows:
        if key in baseline and key in candidate:
            deltas.append(float(candidate[key]) - float(baseline[key]))
    return _mean(deltas)


def _summarize_watch_token_pairs(
    paired_rows: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    watch_token_id: int,
    candidate_bias_position_indices: set[int] | None = None,
    top_examples: int,
) -> dict[str, Any]:
    baseline_correct = 0
    candidate_correct = 0
    position_count = 0
    watch_teacher_positions = 0
    non_watch_teacher_positions = 0
    watch_teacher_top1_gain_count = 0
    watch_teacher_top1_regression_count = 0
    non_watch_teacher_top1_gain_count = 0
    non_watch_teacher_top1_regression_count = 0
    false_watch_top1_takeover_count = 0
    collateral_positions: list[dict[str, Any]] = []

    for baseline, candidate in paired_rows:
        baseline_count = _require_equal_length(
            baseline,
            ("positions", "teacher_top1_ids", "vq_top1_ids", "token_klds"),
        )
        candidate_count = _require_equal_length(
            candidate,
            ("positions", "teacher_top1_ids", "vq_top1_ids", "token_klds"),
        )
        if baseline_count != candidate_count:
            raise ValueError(f"row {baseline.get('row_index')} baseline/candidate token counts differ")
        for position_index in range(baseline_count):
            if (
                candidate_bias_position_indices is not None
                and position_index not in candidate_bias_position_indices
            ):
                continue
            baseline_position = int(baseline["positions"][position_index])
            candidate_position = int(candidate["positions"][position_index])
            if baseline_position != candidate_position:
                raise ValueError(
                    f"row {baseline.get('row_index')} position mismatch at index {position_index}"
                )
            teacher_top1_id = int(baseline["teacher_top1_ids"][position_index])
            candidate_teacher_top1_id = int(candidate["teacher_top1_ids"][position_index])
            if teacher_top1_id != candidate_teacher_top1_id:
                raise ValueError(
                    f"row {baseline.get('row_index')} teacher top1 mismatch at position {baseline_position}"
                )
            baseline_vq_top1_id = int(baseline["vq_top1_ids"][position_index])
            candidate_vq_top1_id = int(candidate["vq_top1_ids"][position_index])
            baseline_is_correct = baseline_vq_top1_id == teacher_top1_id
            candidate_is_correct = candidate_vq_top1_id == teacher_top1_id
            position_count += 1
            baseline_correct += int(baseline_is_correct)
            candidate_correct += int(candidate_is_correct)
            if teacher_top1_id == watch_token_id:
                watch_teacher_positions += 1
                if not baseline_is_correct and candidate_is_correct:
                    watch_teacher_top1_gain_count += 1
                elif baseline_is_correct and not candidate_is_correct:
                    watch_teacher_top1_regression_count += 1
            else:
                non_watch_teacher_positions += 1
                if not baseline_is_correct and candidate_is_correct:
                    non_watch_teacher_top1_gain_count += 1
                elif baseline_is_correct and not candidate_is_correct:
                    non_watch_teacher_top1_regression_count += 1
                    collateral_positions.append(
                        {
                            "source_jsonl": candidate.get("_source_jsonl"),
                            "prompt_id": candidate.get("prompt_id"),
                            "row_index": candidate.get("row_index"),
                            "position_index": position_index,
                            "position": baseline_position,
                            "teacher_top1_id": teacher_top1_id,
                            "baseline_vq_top1_id": baseline_vq_top1_id,
                            "candidate_vq_top1_id": candidate_vq_top1_id,
                            "baseline_token_kld": float(baseline["token_klds"][position_index]),
                            "candidate_token_kld": float(candidate["token_klds"][position_index]),
                            "token_kld_delta": float(candidate["token_klds"][position_index])
                            - float(baseline["token_klds"][position_index]),
                        }
                    )
                if candidate_vq_top1_id == watch_token_id and baseline_vq_top1_id != watch_token_id:
                    false_watch_top1_takeover_count += 1

    row_count = len(paired_rows)
    row_metric_scope = "all_positions" if candidate_bias_position_indices is None else "unscoped_candidate_jsonl"
    baseline_mean_top1 = None
    candidate_mean_top1 = None
    mean_top1_delta = None
    mean_kld_delta = None
    mean_ppl_ratio_delta = None
    mean_p999_kld_delta = None
    if candidate_bias_position_indices is None:
        baseline_mean_top1 = None if row_count == 0 else _mean([float(row[0]["top1_agreement"]) for row in paired_rows])
        candidate_mean_top1 = None if row_count == 0 else _mean([float(row[1]["top1_agreement"]) for row in paired_rows])
        mean_top1_delta = _mean_delta(paired_rows, "top1_agreement")
        mean_kld_delta = _mean_delta(paired_rows, "mean_kld")
        mean_ppl_ratio_delta = _mean_delta(paired_rows, "ppl_ratio")
        mean_p999_kld_delta = _mean_delta(paired_rows, "p999_kld")
    return {
        "row_count": row_count,
        "position_count": position_count,
        "row_metric_scope": row_metric_scope,
        "baseline_mean_top1": baseline_mean_top1,
        "candidate_mean_top1": candidate_mean_top1,
        "mean_top1_delta": mean_top1_delta,
        "position_top1_delta": None
        if position_count == 0
        else (candidate_correct - baseline_correct) / position_count,
        "mean_kld_delta": mean_kld_delta,
        "mean_ppl_ratio_delta": mean_ppl_ratio_delta,
        "mean_p999_kld_delta": mean_p999_kld_delta,
        "watch_teacher_position_count": watch_teacher_positions,
        "non_watch_teacher_position_count": non_watch_teacher_positions,
        "watch_teacher_top1_gain_count": watch_teacher_top1_gain_count,
        "watch_teacher_top1_regression_count": watch_teacher_top1_regression_count,
        "non_watch_teacher_top1_gain_count": non_watch_teacher_top1_gain_count,
        "non_watch_teacher_top1_regression_count": non_watch_teacher_top1_regression_count,
        "false_watch_top1_takeover_count": false_watch_top1_takeover_count,
        "collateral_positions": sorted(
            collateral_positions,
            key=lambda item: float(item["token_kld_delta"]),
            reverse=True,
        )[:top_examples],
    }


def analyze_watch_token_full_split(
    *,
    baseline_jsonls: list[Path],
    candidate_jsonls: list[Path],
    watch_token_id: int,
    label: str | None = None,
    candidate_bias_position_indices: list[int] | None = None,
    top_examples: int = 20,
) -> dict[str, Any]:
    baseline_rows = _read_eval_rows_by_index(baseline_jsonls, label="baseline")
    candidate_rows = _read_eval_rows_by_index(candidate_jsonls, label="candidate")
    paired_indices = sorted(set(baseline_rows) & set(candidate_rows))
    if not paired_indices:
        raise ValueError("no paired row_index values between baseline and candidate")
    paired_rows = [(baseline_rows[index], candidate_rows[index]) for index in paired_indices]
    clean_paired_rows = [
        pair
        for pair in paired_rows
        if _memory_clean(pair[0]) and _memory_clean(pair[1])
    ]
    candidate_bias_position_set = (
        None
        if candidate_bias_position_indices is None
        else {int(index) for index in candidate_bias_position_indices}
    )
    all_rows = _summarize_watch_token_pairs(
        paired_rows,
        watch_token_id=watch_token_id,
        candidate_bias_position_indices=candidate_bias_position_set,
        top_examples=top_examples,
    )
    clean_rows = _summarize_watch_token_pairs(
        clean_paired_rows,
        watch_token_id=watch_token_id,
        candidate_bias_position_indices=candidate_bias_position_set,
        top_examples=top_examples,
    )

    has_clean_collateral = clean_rows["non_watch_teacher_top1_regression_count"] > 0
    has_any_collateral = all_rows["non_watch_teacher_top1_regression_count"] > 0
    clean_top1_delta = clean_rows["position_top1_delta"]
    clean_kld_delta = clean_rows["mean_kld_delta"]
    all_memory_clean = len(clean_paired_rows) == len(paired_rows)
    if has_clean_collateral or (clean_top1_delta is not None and clean_top1_delta < 0):
        decision = "watch_token_global_bias_collateral_risk"
        next_action = "scope_bias_before_selection_holdout"
    elif not all_memory_clean:
        decision = "dirty_full_split_needs_memory_root_cause"
        next_action = "root_cause_pageouts_before_trusting_global_bias"
    elif has_any_collateral or (clean_kld_delta is not None and clean_kld_delta > 0.0):
        decision = "watch_token_global_bias_net_negative"
        next_action = "scope_bias_before_selection_holdout"
    else:
        decision = "watch_token_global_bias_full_split_viable_probe"
        next_action = "run_selection_holdout_guard"

    return {
        "record_type": "glm45_air_watch_token_full_split_collateral",
        "schema_version": 1,
        "label": label,
        "watch_token_id": int(watch_token_id),
        "candidate_bias_position_indices": None
        if candidate_bias_position_indices is None
        else sorted(candidate_bias_position_set or set()),
        "baseline_jsonls": [str(path) for path in baseline_jsonls],
        "candidate_jsonls": [str(path) for path in candidate_jsonls],
        "baseline_row_count": len(baseline_rows),
        "candidate_row_count": len(candidate_rows),
        "paired_row_count": len(paired_rows),
        "missing_baseline_row_indices": sorted(set(candidate_rows) - set(baseline_rows)),
        "missing_candidate_row_indices": sorted(set(baseline_rows) - set(candidate_rows)),
        "memory_clean_paired_row_count": len(clean_paired_rows),
        "memory_dirty_paired_row_indices": [
            int(candidate["row_index"])
            for baseline, candidate in paired_rows
            if not (_memory_clean(baseline) and _memory_clean(candidate))
        ],
        "all_memory_clean": all_memory_clean,
        "all_rows": all_rows,
        "memory_clean_rows": clean_rows,
        "decision": decision,
        "next_action": next_action,
    }


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def _safe_mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _log_z_delta_for_single_token_bias(*, token_logprob: float, bias: float) -> float:
    return math.log1p(math.exp(token_logprob) * math.expm1(bias))


def simulate_watch_token_bias_delta(
    *,
    jsonls: list[Path],
    watch_token_id: int,
    extra_bias: float,
    bias_position_indices: list[int] | None = None,
    label: str | None = None,
    p999_target: float = 3.0,
    top_examples: int = 20,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in jsonls:
        rows.extend(row for row in _read_jsonl(path) if _is_eval_row(row))
    if not rows:
        raise ValueError("no eval rows found for watch-token bias simulation")
    bias_position_set = None if bias_position_indices is None else {int(index) for index in bias_position_indices}

    current_token_klds: list[float] = []
    best_case_token_klds: list[float] = []
    worst_case_token_klds: list[float] = []
    top1_position_count = 0
    current_top1_correct = 0
    simulated_top1_correct = 0
    biased_position_count = 0
    teacher_watch_position_count = 0
    watch_top1_gain_count = 0
    watch_top1_regression_count = 0
    non_watch_teacher_top1_gain_count = 0
    non_watch_teacher_top1_regression_count = 0
    false_watch_top1_takeover_count = 0
    lower_bound_improvements: list[dict[str, Any]] = []
    residual_best_case_tail: list[dict[str, Any]] = []
    collateral_positions: list[dict[str, Any]] = []

    for row in rows:
        token_count = _require_equal_length(
            row,
            (
                "positions",
                "teacher_top1_ids",
                "vq_top1_ids",
                "token_klds",
                "vq_top1_logprobs",
            ),
        )
        watch_logprobs = _watch_token_values(row, "vq_watch_token_logprobs", watch_token_id)
        watch_margins = _watch_token_values(row, "vq_watch_token_margin_vs_vq_top1", watch_token_id)
        if len(watch_logprobs) != token_count or len(watch_margins) != token_count:
            raise ValueError(
                f"row {row.get('row_index')} watch-token fields do not match token count {token_count}"
            )
        for position_index in range(token_count):
            current_kld = float(row["token_klds"][position_index])
            current_token_klds.append(current_kld)
            top1_position_count += 1
            teacher_top1_id = int(row["teacher_top1_ids"][position_index])
            vq_top1_id = int(row["vq_top1_ids"][position_index])
            current_is_correct = vq_top1_id == teacher_top1_id
            current_top1_correct += int(current_is_correct)

            applies = bias_position_set is None or position_index in bias_position_set
            if not applies:
                best_case_token_klds.append(current_kld)
                worst_case_token_klds.append(current_kld)
                simulated_top1_correct += int(current_is_correct)
                continue

            biased_position_count += 1
            watch_margin = float(watch_margins[position_index])
            watch_becomes_top1 = vq_top1_id == watch_token_id or watch_margin + extra_bias >= 0.0
            simulated_top1_id = watch_token_id if watch_becomes_top1 else vq_top1_id
            simulated_is_correct = simulated_top1_id == teacher_top1_id
            simulated_top1_correct += int(simulated_is_correct)

            watch_logprob = float(watch_logprobs[position_index])
            log_z_delta = _log_z_delta_for_single_token_bias(
                token_logprob=watch_logprob,
                bias=extra_bias,
            )
            # The eval row does not store teacher probability for arbitrary watched
            # tokens. A single-token logit lift changes KL by logZ - p_teacher(token)*bias,
            # so p_teacher in [0, 1] gives a safe bound.
            best_case_kld = current_kld + log_z_delta - extra_bias
            worst_case_kld = current_kld + log_z_delta
            best_case_token_klds.append(best_case_kld)
            worst_case_token_klds.append(worst_case_kld)

            if teacher_top1_id == watch_token_id:
                teacher_watch_position_count += 1
                if not current_is_correct and simulated_is_correct:
                    watch_top1_gain_count += 1
                elif current_is_correct and not simulated_is_correct:
                    watch_top1_regression_count += 1
            else:
                if not current_is_correct and simulated_is_correct:
                    non_watch_teacher_top1_gain_count += 1
                elif current_is_correct and not simulated_is_correct:
                    non_watch_teacher_top1_regression_count += 1
                    collateral_positions.append(
                        {
                            "source_jsonl": row.get("_source_jsonl"),
                            "prompt_id": row.get("prompt_id"),
                            "row_index": row.get("row_index"),
                            "position_index": position_index,
                            "position": int(row["positions"][position_index]),
                            "teacher_top1_id": teacher_top1_id,
                            "current_vq_top1_id": vq_top1_id,
                            "simulated_vq_top1_id": simulated_top1_id,
                            "watch_margin_after_bias": watch_margin + extra_bias,
                        }
                    )
                if vq_top1_id != watch_token_id and simulated_top1_id == watch_token_id:
                    false_watch_top1_takeover_count += 1

            lower_bound_improvements.append(
                {
                    "source_jsonl": row.get("_source_jsonl"),
                    "prompt_id": row.get("prompt_id"),
                    "row_index": row.get("row_index"),
                    "position_index": position_index,
                    "position": int(row["positions"][position_index]),
                    "teacher_top1_id": teacher_top1_id,
                    "current_vq_top1_id": vq_top1_id,
                    "simulated_vq_top1_id": simulated_top1_id,
                    "current_token_kld": current_kld,
                    "best_case_token_kld": best_case_kld,
                    "worst_case_token_kld": worst_case_kld,
                    "best_case_kld_delta": best_case_kld - current_kld,
                    "worst_case_kld_delta": worst_case_kld - current_kld,
                    "watch_margin_before_bias": watch_margin,
                    "watch_margin_after_bias": watch_margin + extra_bias,
                    "watch_logprob_before_bias": watch_logprob,
                    "log_z_delta": log_z_delta,
                }
            )
            residual_best_case_tail.append(lower_bound_improvements[-1])

    current_p999 = _quantile(current_token_klds, 0.999)
    best_case_p999 = _quantile(best_case_token_klds, 0.999)
    worst_case_p999 = _quantile(worst_case_token_klds, 0.999)
    current_top1 = current_top1_correct / top1_position_count
    simulated_top1 = simulated_top1_correct / top1_position_count
    best_case_mean_kld = _safe_mean(best_case_token_klds)
    worst_case_mean_kld = _safe_mean(worst_case_token_klds)
    if false_watch_top1_takeover_count or non_watch_teacher_top1_regression_count:
        decision = "extra_bias_collateral_risk"
        next_action = "do_not_materialize_without_tighter_scope"
    elif best_case_p999 is not None and best_case_p999 > p999_target:
        decision = "extra_bias_cannot_clear_p999_even_best_case"
        next_action = "use_non_bias_tail_cleanup_or_training_signal"
    elif simulated_top1 <= current_top1:
        decision = "extra_bias_no_top1_gain"
        next_action = "skip_extra_bias"
    else:
        decision = "extra_bias_promising_needs_real_eval"
        next_action = "materialize_and_run_focused_guard"

    return {
        "record_type": "glm45_air_watch_token_bias_delta_simulation",
        "schema_version": 1,
        "label": label,
        "jsonls": [str(path) for path in jsonls],
        "watch_token_id": int(watch_token_id),
        "extra_bias": float(extra_bias),
        "bias_position_indices": None if bias_position_indices is None else sorted(bias_position_set or set()),
        "kl_assumption": "bounds_only_teacher_watch_probability_unavailable",
        "row_count": len(rows),
        "position_count": top1_position_count,
        "biased_position_count": biased_position_count,
        "teacher_watch_position_count": teacher_watch_position_count,
        "current_top1": current_top1,
        "simulated_top1": simulated_top1,
        "top1_delta": simulated_top1 - current_top1,
        "watch_top1_gain_count": watch_top1_gain_count,
        "watch_top1_regression_count": watch_top1_regression_count,
        "non_watch_teacher_top1_gain_count": non_watch_teacher_top1_gain_count,
        "non_watch_teacher_top1_regression_count": non_watch_teacher_top1_regression_count,
        "false_watch_top1_takeover_count": false_watch_top1_takeover_count,
        "current_mean_kld": _safe_mean(current_token_klds),
        "best_case_mean_kld": best_case_mean_kld,
        "worst_case_mean_kld": worst_case_mean_kld,
        "best_case_mean_kld_delta": None
        if best_case_mean_kld is None
        else best_case_mean_kld - float(_safe_mean(current_token_klds) or 0.0),
        "worst_case_mean_kld_delta": None
        if worst_case_mean_kld is None
        else worst_case_mean_kld - float(_safe_mean(current_token_klds) or 0.0),
        "current_p999_kld": current_p999,
        "best_case_p999_kld": best_case_p999,
        "worst_case_p999_kld": worst_case_p999,
        "best_case_p999_delta": None if best_case_p999 is None or current_p999 is None else best_case_p999 - current_p999,
        "worst_case_p999_delta": None if worst_case_p999 is None or current_p999 is None else worst_case_p999 - current_p999,
        "decision": decision,
        "next_action": next_action,
        "top_lower_bound_improvements": sorted(
            lower_bound_improvements,
            key=lambda item: float(item["best_case_kld_delta"]),
        )[:top_examples],
        "residual_best_case_tail": sorted(
            residual_best_case_tail,
            key=lambda item: float(item["best_case_token_kld"]),
            reverse=True,
        )[:top_examples],
        "collateral_positions": sorted(
            collateral_positions,
            key=lambda item: float(item["watch_margin_after_bias"]),
            reverse=True,
        )[:top_examples],
    }


def _watch_token_values(row: dict[str, Any], field: str, token_id: int) -> list[Any]:
    values_by_token = row.get(field)
    if not isinstance(values_by_token, dict):
        raise ValueError(f"row is missing dict field {field!r}")
    values = values_by_token.get(str(token_id))
    if not isinstance(values, list):
        raise ValueError(f"row is missing watch token {token_id} in {field!r}")
    return values


def _infer_split(row: dict[str, Any]) -> str:
    prompt_id = str(row.get("prompt_id") or "")
    prefix = prompt_id.split("_", 1)[0]
    if prefix == "select":
        return "selection"
    if prefix in {"report", "selection", "holdout"}:
        return prefix
    source = str(row.get("_source_jsonl") or "")
    if "holdout" in source:
        return "holdout"
    if "select" in source or "selection" in source:
        return "selection"
    if "report" in source:
        return "report"
    return "unknown"


def _infer_domain(row: dict[str, Any]) -> str:
    prompt_id = str(row.get("prompt_id") or "")
    parts = prompt_id.split("_")
    if len(parts) >= 2 and parts[0] in {"report", "selection", "select", "holdout"}:
        return parts[1]
    if len(parts) >= 1 and parts[0]:
        return parts[0]
    return "unknown"


def select_tail_cleanup_targets(
    *,
    jsonls: list[Path],
    teacher_token_id: int,
    focus_position: int = 0,
    max_train_rows: int = 10,
    max_validation_rows: int = 12,
    top_tail_records: int = 50,
    label: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in jsonls:
        rows.extend(row for row in _read_jsonl(path) if _is_eval_row(row))
    if not rows:
        raise ValueError("no eval rows found for tail cleanup target selection")
    if max_train_rows <= 0:
        raise ValueError("max_train_rows must be positive")
    if max_validation_rows <= 0:
        raise ValueError("max_validation_rows must be positive")
    if top_tail_records <= 0:
        raise ValueError("top_tail_records must be positive")

    tail_records: list[dict[str, Any]] = []
    weak_domains: dict[str, dict[str, Any]] = {}
    domain_tokens: dict[str, list[tuple[bool, float]]] = {}

    for row in rows:
        token_count = _require_equal_length(
            row,
            ("positions", "teacher_top1_ids", "vq_top1_ids", "token_klds", "target_token_ids"),
        )
        split = _infer_split(row)
        domain = _infer_domain(row)
        domain_key = f"{split}:{domain}"
        for position_index in range(token_count):
            teacher_top1_id = int(row["teacher_top1_ids"][position_index])
            vq_top1_id = int(row["vq_top1_ids"][position_index])
            token_kld = float(row["token_klds"][position_index])
            domain_tokens.setdefault(domain_key, []).append((teacher_top1_id == vq_top1_id, token_kld))
            if position_index != focus_position or teacher_top1_id != int(teacher_token_id):
                continue
            tail_records.append(
                {
                    "source_jsonl": row.get("_source_jsonl"),
                    "split": split,
                    "domain": domain,
                    "prompt_id": row.get("prompt_id"),
                    "row_index": int(row["row_index"]),
                    "position_index": position_index,
                    "position": int(row["positions"][position_index]),
                    "teacher_top1_id": teacher_top1_id,
                    "vq_top1_id": vq_top1_id,
                    "top1_correct": vq_top1_id == teacher_top1_id,
                    "target_token_id": int(row["target_token_ids"][position_index]),
                    "token_kld": token_kld,
                    "vq_teacher_top1_margin_vs_vq_top1": float(
                        _list_item(row, "vq_teacher_top1_margin_vs_vq_top1", position_index)
                    ),
                    "vq_target_logprob": float(_list_item(row, "vq_target_logprobs", position_index)),
                    "memory_clean": _memory_clean(row),
                }
            )

    for domain_key, token_values in domain_tokens.items():
        if not token_values:
            continue
        correct_count = sum(1 for correct, _ in token_values if correct)
        klds = [kld for _, kld in token_values]
        weak_domains[domain_key] = {
            "token_count": len(token_values),
            "position_top1": correct_count / len(token_values),
            "mean_token_kld": _safe_mean(klds),
            "max_token_kld": max(klds),
        }

    tail_records = sorted(
        tail_records,
        key=lambda item: (
            float(item["token_kld"]),
            not bool(item["top1_correct"]),
            item["split"] == "report",
        ),
        reverse=True,
    )
    top_tail = tail_records[:top_tail_records]
    top_tail_domain_counter = Counter(f"{record['split']}:{record['domain']}" for record in top_tail)

    def _eligible_tail_domain(record: dict[str, Any]) -> bool:
        domain_key = f"{record['split']}:{record['domain']}"
        domain_summary = weak_domains.get(domain_key)
        return domain_key in top_tail_domain_counter or (
            domain_summary is not None and float(domain_summary["position_top1"]) < 0.80
        )

    def _domain_balanced(records: list[dict[str, Any]], *, max_rows: int) -> list[dict[str, Any]]:
        by_domain: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            by_domain.setdefault(str(record["domain"]), []).append(record)
        for domain_records in by_domain.values():
            domain_records.sort(
                key=lambda item: (
                    bool(item["memory_clean"]),
                    float(item["token_kld"]),
                    not bool(item["top1_correct"]),
                ),
                reverse=True,
            )
        domain_order = sorted(
            by_domain,
            key=lambda domain: (
                max(float(item["token_kld"]) for item in by_domain[domain]),
                any(not bool(item["top1_correct"]) for item in by_domain[domain]),
            ),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, int]] = set()
        while len(selected) < max_rows:
            changed = False
            for domain in domain_order:
                while by_domain[domain]:
                    candidate = by_domain[domain].pop(0)
                    key = (str(candidate["split"]), int(candidate["row_index"]))
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    selected.append(candidate)
                    changed = True
                    break
                if len(selected) >= max_rows:
                    break
            if not changed:
                break
        return selected

    report_train_records = _domain_balanced(
        [
            record
            for record in tail_records
            if str(record["split"]) == "report" and _eligible_tail_domain(record)
        ],
        max_rows=max_train_rows,
    )
    report_train_rows = [int(record["row_index"]) for record in report_train_records]
    validation_records = _domain_balanced(
        [
            record
            for record in tail_records
            if str(record["split"]) in {"selection", "holdout"} and _eligible_tail_domain(record)
        ],
        max_rows=max_validation_rows,
    )

    if report_train_rows:
        decision = "tail_cleanup_report_train_packet_ready"
        next_action = "run_tiny_report_cache_tail_training_then_focused_selection_holdout_guard"
    else:
        decision = "tail_cleanup_no_report_training_rows"
        next_action = "inspect_split_mapping_or_use_selection_cache_only"

    return {
        "record_type": "glm45_air_tail_cleanup_target_selection",
        "schema_version": 1,
        "label": label,
        "jsonls": [str(path) for path in jsonls],
        "teacher_token_id": int(teacher_token_id),
        "focus_position": int(focus_position),
        "row_count": len(rows),
        "tail_candidate_count": len(tail_records),
        "top_tail_records": top_tail,
        "top_tail_domain_counts": dict(sorted(top_tail_domain_counter.items())),
        "top_tail_vq_counts": {
            str(key): value for key, value in sorted(Counter(record["vq_top1_id"] for record in top_tail).items())
        },
        "weak_domains": {
            key: value
            for key, value in sorted(weak_domains.items())
            if float(value["position_top1"]) < 0.80 or float(value["max_token_kld"]) >= 3.0
        },
        "recommended_training": {
            "train_cache": "validation",
            "train_row_indices": report_train_rows,
            "train_rows": report_train_records,
            "max_positions": 1,
            "aux_loss_position_indices": [focus_position],
            "seed_artifact_dir": rows[0].get("artifact_dir"),
            "notes": (
                "validation train-cache is the report cache; selection and holdout rows are validation-only. "
                "Use low LR/NLL-heavy selected-layer training, not another final-logit bias lift."
            ),
        },
        "validation_focus_rows": validation_records,
        "decision": decision,
        "next_action": next_action,
    }


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    target_rows: set[int] | None,
    prompt_contains: str | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if target_rows is not None and int(row.get("row_index", -1)) not in target_rows:
            continue
        if prompt_contains is not None and prompt_contains not in str(row.get("prompt_id", "")):
            continue
        selected.append(row)
    return selected


def analyze_top1_gap(
    *,
    jsonls: list[Path],
    focus_position: int,
    target_rows: list[int] | None = None,
    prompt_contains: str | None = None,
    label: str | None = None,
    watch_token_id: int | None = None,
) -> dict[str, Any]:
    target_row_set = None if target_rows is None else {int(row) for row in target_rows}
    rows: list[dict[str, Any]] = []
    for path in jsonls:
        rows.extend(_select_rows(_read_jsonl(path), target_rows=target_row_set, prompt_contains=prompt_contains))
    if not rows:
        raise ValueError("no rows matched top1 gap analysis filters")

    teacher_ids: Counter[int] = Counter()
    wrong_ids: Counter[int] = Counter()
    margins: list[float] = []
    token_klds: list[float] = []
    target_logprobs: list[float] = []
    transitions: Counter[str] = Counter()
    per_rows: list[dict[str, Any]] = []
    top1_flip_count = 0
    watch_focus_required_biases: list[float] = []
    watch_focus_margins: list[float] = []
    watch_same_row_collateral_positions: list[dict[str, Any]] = []

    for row in rows:
        teacher_top1_id = int(_list_item(row, "teacher_top1_ids", focus_position))
        vq_top1_id = int(_list_item(row, "vq_top1_ids", focus_position))
        teacher_ids[teacher_top1_id] += 1
        if vq_top1_id == teacher_top1_id:
            top1_flip_count += 1
        else:
            wrong_ids[vq_top1_id] += 1
        margin = float(_list_item(row, "vq_teacher_top1_margin_vs_vq_top1", focus_position))
        token_kld = float(_list_item(row, "token_klds", focus_position))
        target_logprob = float(_list_item(row, "vq_target_logprobs", focus_position))
        margins.append(margin)
        token_klds.append(token_kld)
        target_logprobs.append(target_logprob)
        transition = f"{teacher_top1_id}->{vq_top1_id}"
        transitions[transition] += 1
        per_rows.append(
            {
                "source_jsonl": row["_source_jsonl"],
                "prompt_id": row.get("prompt_id"),
                "row_index": row.get("row_index"),
                "position": int(_list_item(row, "positions", focus_position)),
                "teacher_top1_id": teacher_top1_id,
                "vq_top1_id": vq_top1_id,
                "teacher_margin": margin,
                "token_kld": token_kld,
                "target_logprob": target_logprob,
                "memory_clean": _memory_clean(row),
                "top1_flipped_to_teacher": vq_top1_id == teacher_top1_id,
            }
        )
        if watch_token_id is not None:
            required_biases = _watch_token_values(
                row,
                "vq_watch_token_required_bias_to_vq_top1",
                watch_token_id,
            )
            margins_vs_top1 = _watch_token_values(
                row,
                "vq_watch_token_margin_vs_vq_top1",
                watch_token_id,
            )
            focus_required_bias = float(_list_item(
                {"values": required_biases},
                "values",
                focus_position,
            ))
            focus_margin = float(_list_item(
                {"values": margins_vs_top1},
                "values",
                focus_position,
            ))
            watch_focus_required_biases.append(focus_required_bias)
            watch_focus_margins.append(focus_margin)

    stable_wrong_top1_id: int | None = None
    stable_wrong_top1_count = 0
    if wrong_ids:
        stable_wrong_top1_id, stable_wrong_top1_count = wrong_ids.most_common(1)[0]
    teacher_top1_id = teacher_ids.most_common(1)[0][0]
    all_clean = all(row["memory_clean"] for row in per_rows)
    row_count = len(per_rows)
    stable_wrong_share = 0.0 if row_count == 0 else stable_wrong_top1_count / row_count

    if top1_flip_count > 0:
        decision = "focus_top1_flip_observed"
        next_action = "run_cross_split_guard"
    elif stable_wrong_top1_id is not None and stable_wrong_share >= 0.5:
        decision = "stable_wrong_attractor_needs_new_mechanism"
        next_action = "target_teacher_vs_stable_wrong_gap_directly"
    else:
        decision = "mixed_wrong_tokens_mine_per_row_competitors"
        next_action = "mine_per_row_wrong_topk_or_change_layer"

    result: dict[str, Any] = {
        "record_type": "glm45_air_top1_gap_analysis",
        "schema_version": 1,
        "label": label,
        "jsonls": [str(path) for path in jsonls],
        "focus_position": focus_position,
        "target_rows": None if target_rows is None else [int(row) for row in target_rows],
        "prompt_contains": prompt_contains,
        "row_count": row_count,
        "memory_clean": all_clean,
        "teacher_top1_id": teacher_top1_id,
        "top1_flip_count": top1_flip_count,
        "stable_wrong_top1_id": stable_wrong_top1_id,
        "stable_wrong_top1_count": stable_wrong_top1_count,
        "stable_wrong_share": stable_wrong_share,
        "mean_teacher_margin": _mean(margins),
        "max_teacher_margin": max(margins),
        "mean_token_kld": _mean(token_klds),
        "mean_target_logprob": _mean(target_logprobs),
        "transition_counts": dict(sorted(transitions.items())),
        "wrong_top1_counts": {str(key): value for key, value in sorted(wrong_ids.items())},
        "decision": decision,
        "next_action": next_action,
        "per_rows": per_rows,
    }
    if watch_token_id is not None:
        focus_required_max = max(watch_focus_required_biases)
        for row in rows:
            required_biases = _watch_token_values(
                row,
                "vq_watch_token_required_bias_to_vq_top1",
                watch_token_id,
            )
            positions = row.get("positions")
            teacher_top1_ids = row.get("teacher_top1_ids")
            vq_top1_ids = row.get("vq_top1_ids")
            if not (
                isinstance(positions, list)
                and isinstance(teacher_top1_ids, list)
                and isinstance(vq_top1_ids, list)
            ):
                raise ValueError("row is missing positions, teacher_top1_ids, or vq_top1_ids")
            for position_index, required_bias in enumerate(required_biases):
                if position_index == focus_position:
                    continue
                if position_index >= len(positions):
                    raise ValueError("watch token required-bias length exceeds positions length")
                if int(teacher_top1_ids[position_index]) == watch_token_id:
                    continue
                if int(vq_top1_ids[position_index]) == watch_token_id:
                    continue
                required_bias_value = float(required_bias)
                if required_bias_value <= focus_required_max + 1e-12:
                    watch_same_row_collateral_positions.append(
                        {
                            "source_jsonl": row["_source_jsonl"],
                            "prompt_id": row.get("prompt_id"),
                            "row_index": row.get("row_index"),
                            "position_index": position_index,
                            "position": int(positions[position_index]),
                            "teacher_top1_id": int(teacher_top1_ids[position_index]),
                            "vq_top1_id": int(vq_top1_ids[position_index]),
                            "required_bias_to_vq_top1": required_bias_value,
                        }
                    )
        if watch_same_row_collateral_positions:
            watch_decision = "watch_token_global_bias_collateral_risk"
            watch_next_action = "prefer_position_or_mechanism_specific_bias"
        else:
            watch_decision = "watch_token_global_bias_viable_probe"
            watch_next_action = "materialize_logit_bias_candidate"
        result.update(
            {
                "watch_token_id": watch_token_id,
                "watch_token_focus_required_bias_mean": _mean(watch_focus_required_biases),
                "watch_token_focus_required_bias_max": focus_required_max,
                "watch_token_focus_margin_mean": _mean(watch_focus_margins),
                "watch_token_same_row_collateral_count": len(watch_same_row_collateral_positions),
                "watch_token_same_row_collateral_positions": watch_same_row_collateral_positions,
                "watch_token_decision": watch_decision,
                "watch_token_next_action": watch_next_action,
            }
        )
    return result


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="append", type=Path)
    parser.add_argument("--focus-position", type=int, default=0)
    parser.add_argument("--target-rows")
    parser.add_argument("--prompt-contains")
    parser.add_argument("--label")
    parser.add_argument("--watch-token-id", type=int)
    parser.add_argument("--baseline-jsonl", action="append", type=Path)
    parser.add_argument("--candidate-jsonl", action="append", type=Path)
    parser.add_argument("--candidate-bias-position-indices")
    parser.add_argument("--full-split-watch-collateral", action="store_true")
    parser.add_argument("--simulate-watch-token-bias-delta", action="store_true")
    parser.add_argument("--select-tail-cleanup-targets", action="store_true")
    parser.add_argument("--teacher-token-id", type=int)
    parser.add_argument("--extra-bias", type=float)
    parser.add_argument("--bias-position-indices")
    parser.add_argument("--p999-target", type=float, default=3.0)
    parser.add_argument("--max-train-rows", type=int, default=10)
    parser.add_argument("--max-validation-rows", type=int, default=12)
    parser.add_argument("--top-tail-records", type=int, default=50)
    parser.add_argument("--top-examples", type=int, default=20)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if args.select_tail_cleanup_targets:
        teacher_token_id = args.teacher_token_id if args.teacher_token_id is not None else args.watch_token_id
        if teacher_token_id is None:
            raise ValueError("--select-tail-cleanup-targets requires --teacher-token-id or --watch-token-id")
        if not args.jsonl:
            raise ValueError("--select-tail-cleanup-targets requires --jsonl")
        report = select_tail_cleanup_targets(
            jsonls=args.jsonl,
            teacher_token_id=teacher_token_id,
            focus_position=args.focus_position,
            max_train_rows=args.max_train_rows,
            max_validation_rows=args.max_validation_rows,
            top_tail_records=args.top_tail_records,
            label=args.label,
        )
    elif args.simulate_watch_token_bias_delta:
        if args.watch_token_id is None:
            raise ValueError("--simulate-watch-token-bias-delta requires --watch-token-id")
        if args.extra_bias is None:
            raise ValueError("--simulate-watch-token-bias-delta requires --extra-bias")
        if not args.jsonl:
            raise ValueError("--simulate-watch-token-bias-delta requires --jsonl")
        report = simulate_watch_token_bias_delta(
            jsonls=args.jsonl,
            watch_token_id=args.watch_token_id,
            extra_bias=args.extra_bias,
            bias_position_indices=_parse_int_list(args.bias_position_indices),
            label=args.label,
            p999_target=args.p999_target,
            top_examples=args.top_examples,
        )
    elif args.full_split_watch_collateral:
        if args.watch_token_id is None:
            raise ValueError("--full-split-watch-collateral requires --watch-token-id")
        if not args.baseline_jsonl or not args.candidate_jsonl:
            raise ValueError("--full-split-watch-collateral requires --baseline-jsonl and --candidate-jsonl")
        report = analyze_watch_token_full_split(
            baseline_jsonls=args.baseline_jsonl,
            candidate_jsonls=args.candidate_jsonl,
            watch_token_id=args.watch_token_id,
            label=args.label,
            candidate_bias_position_indices=_parse_int_list(args.candidate_bias_position_indices),
            top_examples=args.top_examples,
        )
    else:
        if not args.jsonl:
            raise ValueError("legacy top1 gap analysis requires --jsonl")
        report = analyze_top1_gap(
            jsonls=args.jsonl,
            focus_position=args.focus_position,
            target_rows=_parse_int_list(args.target_rows),
            prompt_contains=args.prompt_contains,
            label=args.label,
            watch_token_id=args.watch_token_id,
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
