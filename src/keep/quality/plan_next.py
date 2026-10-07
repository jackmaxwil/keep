"""Next-cycle quality-plan row selection.

Extracted verbatim from ``benchmarks/run_glm45_air_rc_pipeline.py`` so the
flip-focused candidate scoring and domain-quota selection are importable by
the ``keep plan-next`` driver as well as the RC pipeline's
``--write-quality-plan-json`` export.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from keep.quality.rc_gates import (
    QUALITY_PLAN_CATEGORY_WEIGHTS,
    QUALITY_PLAN_DOMAIN_QUOTAS,
    QUALITY_PLAN_DOMAIN_WEIGHTS,
    _float_or_none,
)

__all__ = [
    "QUALITY_PLAN_CATEGORY_WEIGHTS",
    "QUALITY_PLAN_DOMAIN_QUOTAS",
    "QUALITY_PLAN_DOMAIN_WEIGHTS",
    "_quality_plan_candidates",
    "_select_quality_plan_rows",
]


def _quality_plan_candidates(
    report: Mapping[str, Any],
    *,
    split_names: Sequence[str],
    allowed_domains: set[str],
) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, int], dict[str, Any]] = {}
    for split_name in split_names:
        split = report["eval_splits"][split_name]
        for category, rows in split.get("quality_focus", {}).items():
            category_weight = QUALITY_PLAN_CATEGORY_WEIGHTS.get(category, 0.0)
            if category_weight <= 0:
                continue
            for rank, row in enumerate(rows, start=1):
                row_index = row.get("row_index")
                domain = row.get("domain")
                if not isinstance(row_index, int) or not isinstance(domain, str):
                    continue
                if domain not in allowed_domains:
                    continue
                key = (domain, row_index)
                entry = candidates.setdefault(
                    key,
                    {
                        "domain": domain,
                        "row_index": row_index,
                        "score": 0.0,
                        "prompt_ids": [],
                        "sources": [],
                        "min_top1_agreement": None,
                        "max_mean_kld": None,
                        "max_token_kld": None,
                    },
                )
                prompt_id = row.get("prompt_id")
                if isinstance(prompt_id, str) and prompt_id not in entry["prompt_ids"]:
                    entry["prompt_ids"].append(prompt_id)
                rank_bonus = 1.0 / float(rank)
                points = category_weight + QUALITY_PLAN_DOMAIN_WEIGHTS[domain] + rank_bonus
                entry["score"] = float(entry["score"]) + points
                entry["sources"].append(
                    {
                        "split": split_name,
                        "category": category,
                        "rank": rank,
                        "points": points,
                    }
                )
                top1 = _float_or_none(row.get("top1_agreement"))
                mean_kld = _float_or_none(row.get("mean_kld"))
                token_kld = _float_or_none(row.get("max_token_kld"))
                if top1 is not None and (
                    entry["min_top1_agreement"] is None or top1 < entry["min_top1_agreement"]
                ):
                    entry["min_top1_agreement"] = top1
                if mean_kld is not None and (
                    entry["max_mean_kld"] is None or mean_kld > entry["max_mean_kld"]
                ):
                    entry["max_mean_kld"] = mean_kld
                if token_kld is not None and (
                    entry["max_token_kld"] is None or token_kld > entry["max_token_kld"]
                ):
                    entry["max_token_kld"] = token_kld
    return sorted(
        candidates.values(),
        key=lambda row: (
            -float(row["score"]),
            row["min_top1_agreement"] if row["min_top1_agreement"] is not None else 1.0,
            -float(row["max_token_kld"] or 0.0),
            -int(row["row_index"]),
        ),
    )


def _select_quality_plan_rows(
    candidates: Sequence[Mapping[str, Any]],
    *,
    quotas: Mapping[str, int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int]] = set()
    for domain, quota in quotas.items():
        domain_rows = [row for row in candidates if row["domain"] == domain]
        for row in domain_rows[:quota]:
            key = (str(row["domain"]), int(row["row_index"]))
            selected.append(dict(row))
            selected_keys.add(key)
    target_count = sum(int(value) for value in quotas.values())
    if len(selected) < target_count:
        for row in candidates:
            key = (str(row["domain"]), int(row["row_index"]))
            if key in selected_keys:
                continue
            selected.append(dict(row))
            selected_keys.add(key)
            if len(selected) >= target_count:
                break
    return sorted(
        selected,
        key=lambda row: (
            -float(row["score"]),
            row["domain"],
            int(row["row_index"]),
        ),
    )
