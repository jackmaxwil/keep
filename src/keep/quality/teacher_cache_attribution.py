from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from keep.quality.teacher_cache import (
    read_teacher_cache_rows,
    summarize_teacher_cache_records,
)


DEFAULT_QUALITY_THRESHOLDS: dict[str, float] = {
    "mean_ppl_ratio_max": 1.15,
    "mean_top1_agreement_min": 0.85,
    "mean_kld_max": 0.25,
    "p999_kld_max": 2.0,
    "per_prompt_ppl_ratio_max": 1.30,
    "per_prompt_top1_agreement_min": 0.70,
}

DEFAULT_HARD_PROMPTS = frozenset({
    "short_math",
    "code_completion",
    "instruction_following",
})
DEFAULT_ROBUST_CONTROLS = frozenset({"capital_france", "long_recall_1k"})


@dataclass(frozen=True)
class TeacherCacheRun:
    label: str
    jsonl: str | None
    records: list[dict[str, Any]]


def load_teacher_cache_run(label: str, jsonl: str | Path) -> TeacherCacheRun:
    return TeacherCacheRun(
        label=label,
        jsonl=str(jsonl),
        records=read_teacher_cache_rows(jsonl),
    )


def build_teacher_cache_attribution_report(
    baseline: TeacherCacheRun,
    *,
    candidates: Iterable[TeacherCacheRun] = (),
    top_tokens: int = 20,
    top_prompts: int = 10,
    thresholds: dict[str, float] | None = None,
    hard_prompts: Iterable[str] = DEFAULT_HARD_PROMPTS,
    robust_controls: Iterable[str] = DEFAULT_ROBUST_CONTROLS,
) -> dict[str, Any]:
    if top_tokens <= 0:
        raise ValueError("top_tokens must be positive")
    if top_prompts <= 0:
        raise ValueError("top_prompts must be positive")

    active_thresholds = dict(DEFAULT_QUALITY_THRESHOLDS)
    if thresholds is not None:
        active_thresholds.update(thresholds)
    hard_prompt_set = set(hard_prompts)
    robust_control_set = set(robust_controls)

    baseline_rows = _prompt_rankings(baseline.records)
    baseline_tokens = _token_records(baseline.records)
    baseline_summary = summarize_teacher_cache_records(baseline.records)
    total_kld = sum(token["token_kld"] for token in baseline_tokens)
    ranked_tokens = sorted(
        baseline_tokens,
        key=lambda token: (
            -token["token_kld"],
            str(token["prompt_id"]),
            int(token["token_index"]),
        ),
    )
    for rank, token in enumerate(ranked_tokens, start=1):
        token["rank"] = rank
        token["kld_share"] = (
            token["token_kld"] / total_kld
            if total_kld > 0.0
            else None
        )

    ranked_prompt_rows = sorted(
        baseline_rows,
        key=lambda row: (
            -row["ppl_ratio"],
            -_number_or(row["mean_kld"], default=-float("inf")),
            -_number_or(row["top1_flip_rate"], default=-float("inf")),
            str(row["prompt_id"]),
        ),
    )
    for rank, row in enumerate(ranked_prompt_rows, start=1):
        row["rank_by_ppl"] = rank

    flip_prompt_rows = sorted(
        baseline_rows,
        key=lambda row: (
            -_number_or(row["top1_flip_rate"], default=-float("inf")),
            -row["top1_flip_count"],
            -_number_or(row["mean_kld"], default=-float("inf")),
            str(row["prompt_id"]),
        ),
    )

    report = {
        "schema_version": 1,
        "baseline": {
            "label": baseline.label,
            "jsonl": baseline.jsonl,
        },
        "thresholds": active_thresholds,
        "hard_prompts": sorted(hard_prompt_set),
        "robust_controls": sorted(robust_control_set),
        "summary": _summary_with_flip_counts(
            baseline_summary,
            baseline_rows,
            baseline_tokens,
        ),
        "quality_floor_violations": _quality_floor_violations(
            baseline_summary,
            baseline_rows,
            active_thresholds,
            robust_control_set,
        ),
        "prompt_rankings": ranked_prompt_rows[:top_prompts],
        "top1_flip_prompts": flip_prompt_rows[:top_prompts],
        "top_kld_tokens": ranked_tokens[:top_tokens],
        "next_probe_targets": _next_probe_targets(
            ranked_tokens,
            hard_prompt_set=hard_prompt_set,
            top_tokens=top_tokens,
        ),
        "candidate_comparisons": [
            _candidate_comparison(
                baseline=baseline,
                candidate=candidate,
                thresholds=active_thresholds,
                robust_controls=robust_control_set,
                top_tokens=top_tokens,
                top_prompts=top_prompts,
            )
            for candidate in candidates
        ],
    }
    return report


def _prompt_rankings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for fallback_row_index, record in enumerate(records):
        token_klds = _float_list(record.get("token_klds"))
        teacher_top1_ids = _int_list(record.get("teacher_top1_ids"))
        vq_top1_ids = _int_list(record.get("vq_top1_ids"))
        target_token_count = int(
            record.get("target_token_count")
            or len(record.get("target_token_ids") or [])
            or len(token_klds)
        )
        top1_flip_count = _top1_flip_count(teacher_top1_ids, vq_top1_ids)
        comparable_top1 = min(len(teacher_top1_ids), len(vq_top1_ids))
        rows.append({
            "row_index": int(record.get("row_index", fallback_row_index)),
            "prompt_id": str(record.get("prompt_id", f"row_{fallback_row_index}")),
            "target_token_count": target_token_count,
            "ppl_ratio": float(record["ppl_ratio"]),
            "nll_delta": float(record["nll_delta"]),
            "mean_kld": (
                float(record["mean_kld"])
                if record.get("mean_kld") is not None
                else None
            ),
            "p999_kld": _p999_from_record(record),
            "max_token_kld": max(token_klds) if token_klds else None,
            "token_kld_sum": sum(token_klds),
            "top1_agreement": (
                float(record["top1_agreement"])
                if record.get("top1_agreement") is not None
                else None
            ),
            "top1_flip_count": top1_flip_count,
            "top1_comparable_count": comparable_top1,
            "top1_flip_rate": (
                top1_flip_count / comparable_top1
                if comparable_top1
                else None
            ),
            "all_memory_clean": (
                record.get("pageouts_delta") == 0
                and record.get("swapouts_delta") == 0
            ),
        })
    return rows


def _token_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens = []
    for fallback_row_index, record in enumerate(records):
        row_index = int(record.get("row_index", fallback_row_index))
        prompt_id = str(record.get("prompt_id", f"row_{fallback_row_index}"))
        token_klds = _float_list(record.get("token_klds"))
        positions = _int_list(record.get("positions"))
        target_token_ids = _int_list(record.get("target_token_ids"))
        teacher_top1_ids = _int_list(record.get("teacher_top1_ids"))
        vq_top1_ids = _int_list(record.get("vq_top1_ids"))
        for token_index, token_kld in enumerate(token_klds):
            teacher_top1_id = _list_get(teacher_top1_ids, token_index)
            vq_top1_id = _list_get(vq_top1_ids, token_index)
            top1_match = (
                teacher_top1_id == vq_top1_id
                if teacher_top1_id is not None and vq_top1_id is not None
                else None
            )
            tokens.append({
                "row_index": row_index,
                "prompt_id": prompt_id,
                "token_index": token_index,
                "position": _list_get(positions, token_index, default=token_index),
                "target_token_id": _list_get(target_token_ids, token_index),
                "teacher_top1_id": teacher_top1_id,
                "vq_top1_id": vq_top1_id,
                "top1_match": top1_match,
                "token_kld": token_kld,
            })
    return tokens


def _candidate_comparison(
    *,
    baseline: TeacherCacheRun,
    candidate: TeacherCacheRun,
    thresholds: dict[str, float],
    robust_controls: set[str],
    top_tokens: int,
    top_prompts: int,
) -> dict[str, Any]:
    baseline_summary = summarize_teacher_cache_records(baseline.records)
    candidate_summary = summarize_teacher_cache_records(candidate.records)
    candidate_rows = _prompt_rankings(candidate.records)
    baseline_rows_by_prompt = {
        str(row["prompt_id"]): row
        for row in _prompt_rankings(baseline.records)
    }
    prompt_deltas = []
    for row in candidate_rows:
        prompt_id = str(row["prompt_id"])
        base_row = baseline_rows_by_prompt.get(prompt_id)
        if base_row is None:
            continue
        prompt_deltas.append({
            "prompt_id": prompt_id,
            "candidate_ppl_ratio": row["ppl_ratio"],
            "delta_ppl_ratio": _delta(row["ppl_ratio"], base_row["ppl_ratio"]),
            "candidate_mean_kld": row["mean_kld"],
            "delta_mean_kld": _delta(row["mean_kld"], base_row["mean_kld"]),
            "candidate_p999_kld": row["p999_kld"],
            "delta_p999_kld": _delta(row["p999_kld"], base_row["p999_kld"]),
            "candidate_top1_agreement": row["top1_agreement"],
            "delta_top1_agreement": _delta(
                row["top1_agreement"],
                base_row["top1_agreement"],
            ),
        })
    prompt_deltas.sort(
        key=lambda row: (
            -(row["delta_ppl_ratio"] or 0.0),
            -(row["delta_mean_kld"] or 0.0),
            str(row["prompt_id"]),
        )
    )

    baseline_tokens_by_key = {
        _token_key(token): token
        for token in _token_records(baseline.records)
    }
    token_deltas = []
    for token in _token_records(candidate.records):
        base_token = baseline_tokens_by_key.get(_token_key(token))
        if base_token is None:
            continue
        token_deltas.append({
            "prompt_id": token["prompt_id"],
            "row_index": token["row_index"],
            "token_index": token["token_index"],
            "position": token["position"],
            "target_token_id": token["target_token_id"],
            "baseline_token_kld": base_token["token_kld"],
            "candidate_token_kld": token["token_kld"],
            "delta_token_kld": token["token_kld"] - base_token["token_kld"],
            "baseline_top1_match": base_token["top1_match"],
            "candidate_top1_match": token["top1_match"],
            "baseline_teacher_top1_id": base_token["teacher_top1_id"],
            "candidate_teacher_top1_id": token["teacher_top1_id"],
            "baseline_vq_top1_id": base_token["vq_top1_id"],
            "candidate_vq_top1_id": token["vq_top1_id"],
        })
    worsened_tokens = sorted(
        token_deltas,
        key=lambda row: (
            -row["delta_token_kld"],
            str(row["prompt_id"]),
            int(row["token_index"]),
        ),
    )
    improved_tokens = sorted(
        token_deltas,
        key=lambda row: (
            row["delta_token_kld"],
            str(row["prompt_id"]),
            int(row["token_index"]),
        ),
    )

    return {
        "label": candidate.label,
        "jsonl": candidate.jsonl,
        "summary": _summary_with_flip_counts(
            candidate_summary,
            candidate_rows,
            _token_records(candidate.records),
        ),
        "delta_vs_baseline": {
            key: _delta(candidate_summary.get(key), baseline_summary.get(key))
            for key in (
                "mean_ppl_ratio",
                "max_ppl_ratio",
                "mean_kld",
                "p999_kld",
                "mean_top1_agreement",
            )
        },
        "quality_floor_violations": _quality_floor_violations(
            candidate_summary,
            candidate_rows,
            thresholds,
            robust_controls,
        ),
        "prompt_deltas": prompt_deltas[:top_prompts],
        "top_worsened_tokens": worsened_tokens[:top_tokens],
        "top_improved_tokens": improved_tokens[:top_tokens],
    }


def _summary_with_flip_counts(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    tokens: list[dict[str, Any]],
) -> dict[str, Any]:
    top1_flip_count = sum(1 for token in tokens if token["top1_match"] is False)
    comparable = sum(1 for token in tokens if token["top1_match"] is not None)
    enriched = dict(summary)
    enriched.update({
        "token_count": len(tokens),
        "top1_flip_count": top1_flip_count,
        "top1_flip_rate": (
            top1_flip_count / comparable
            if comparable
            else None
        ),
        "worst_prompt_by_ppl": max(rows, key=lambda row: row["ppl_ratio"])["prompt_id"]
        if rows
        else None,
        "worst_prompt_by_top1": min(
            (row for row in rows if row["top1_agreement"] is not None),
            key=lambda row: row["top1_agreement"],
            default={},
        ).get("prompt_id"),
    })
    return enriched


def _quality_floor_violations(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    thresholds: dict[str, float],
    robust_controls: set[str],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    _append_violation(
        violations,
        "mean_ppl_ratio",
        summary.get("mean_ppl_ratio"),
        "<=",
        thresholds["mean_ppl_ratio_max"],
    )
    _append_violation(
        violations,
        "mean_top1_agreement",
        summary.get("mean_top1_agreement"),
        ">=",
        thresholds["mean_top1_agreement_min"],
    )
    _append_violation(
        violations,
        "mean_kld",
        summary.get("mean_kld"),
        "<=",
        thresholds["mean_kld_max"],
    )
    _append_violation(
        violations,
        "p999_kld",
        summary.get("p999_kld"),
        "<=",
        thresholds["p999_kld_max"],
    )
    for row in rows:
        _append_violation(
            violations,
            "per_prompt_ppl_ratio",
            row.get("ppl_ratio"),
            "<=",
            thresholds["per_prompt_ppl_ratio_max"],
            prompt_id=str(row["prompt_id"]),
        )
        _append_violation(
            violations,
            "per_prompt_top1_agreement",
            row.get("top1_agreement"),
            ">=",
            thresholds["per_prompt_top1_agreement_min"],
            prompt_id=str(row["prompt_id"]),
        )
        if (
            row["prompt_id"] in robust_controls
            and _number_or(row.get("ppl_ratio"), default=1.0) > 1.0
        ):
            violations.append({
                "metric": "robust_control_ppl_ratio",
                "prompt_id": row["prompt_id"],
                "value": row["ppl_ratio"],
                "operator": "<=",
                "threshold": 1.0,
            })
    return violations


def _append_violation(
    violations: list[dict[str, Any]],
    metric: str,
    value: Any,
    operator: str,
    threshold: float,
    *,
    prompt_id: str | None = None,
) -> None:
    if value is None:
        return
    numeric = float(value)
    failed = numeric > threshold if operator == "<=" else numeric < threshold
    if not failed:
        return
    record = {
        "metric": metric,
        "value": numeric,
        "operator": operator,
        "threshold": threshold,
    }
    if prompt_id is not None:
        record["prompt_id"] = prompt_id
    violations.append(record)


def _next_probe_targets(
    ranked_tokens: list[dict[str, Any]],
    *,
    hard_prompt_set: set[str],
    top_tokens: int,
) -> list[dict[str, Any]]:
    targets = []
    seen: set[tuple[str, int]] = set()
    for token in ranked_tokens:
        prompt_id = str(token["prompt_id"])
        should_include = prompt_id in hard_prompt_set or token["top1_match"] is False
        if not should_include:
            continue
        key = (prompt_id, int(token["token_index"]))
        if key in seen:
            continue
        seen.add(key)
        reasons = ["kld_tail"]
        if prompt_id in hard_prompt_set:
            reasons.append("hard_prompt")
        if token["top1_match"] is False:
            reasons.append("top1_flip")
        targets.append({
            "prompt_id": prompt_id,
            "token_index": token["token_index"],
            "position": token["position"],
            "token_kld": token["token_kld"],
            "teacher_top1_id": token["teacher_top1_id"],
            "vq_top1_id": token["vq_top1_id"],
            "reasons": reasons,
            "suggested_probe": (
                "prompt-derived layer/projection validation at this prompt-state "
                "position; compare routed gate/up/down source-vs-artifact error"
            ),
        })
        if len(targets) >= top_tokens:
            break
    return targets


def _p999_from_record(record: dict[str, Any]) -> float | None:
    token_klds = sorted(_float_list(record.get("token_klds")))
    if not token_klds:
        return None
    if len(token_klds) == 1:
        return token_klds[0]
    index = 0.999 * (len(token_klds) - 1)
    lower = int(index)
    upper = min(lower + 1, len(token_klds) - 1)
    fraction = index - lower
    return token_klds[lower] * (1.0 - fraction) + token_klds[upper] * fraction


def _top1_flip_count(teacher_top1_ids: list[int], vq_top1_ids: list[int]) -> int:
    return sum(
        1
        for teacher_id, vq_id in zip(teacher_top1_ids, vq_top1_ids)
        if teacher_id != vq_id
    )


def _token_key(token: dict[str, Any]) -> tuple[str, int]:
    return (str(token["prompt_id"]), int(token["token_index"]))


def _float_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    return [float(value) for value in values]


def _int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        if value is None:
            continue
        result.append(int(value))
    return result


def _list_get(values: list[Any], index: int, default: Any = None) -> Any:
    if index >= len(values):
        return default
    return values[index]


def _delta(candidate_value: Any, baseline_value: Any) -> float | None:
    if candidate_value is None or baseline_value is None:
        return None
    return float(candidate_value) - float(baseline_value)


def _number_or(value: Any, *, default: float) -> float:
    if value is None:
        return default
    return float(value)
