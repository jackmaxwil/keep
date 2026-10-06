from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

import benchmarks.bench_qwen_family_candidate_latency as qwen_candidate_latency
import benchmarks.bench_qwen_family_control_latency as qwen_control_latency
from benchmarks.define_qwen_family_gate_policy import define_qwen_family_gate_policy
from benchmarks.check_qwen_family_benchmark_gate import check_qwen_family_benchmark_gate
from benchmarks.check_qwen_family_eval_gate import check_qwen_family_eval_gate
from benchmarks.prepare_qwen_family_benchmark_rows import (
    prepare_qwen_family_benchmark_rows,
)
from benchmarks.prepare_qwen_family_benchmark_invariants import (
    prepare_qwen_family_benchmark_invariants,
)
from benchmarks.prepare_qwen_family_eval_rows import prepare_qwen_family_eval_rows
from benchmarks.prepare_qwen_family_eval_metric_rows import (
    prepare_qwen_family_eval_metric_rows,
)
from benchmarks.prepare_qwen_family_eval_candidate_logits import (
    prepare_qwen_family_eval_candidate_logits,
)
from benchmarks.prepare_qwen_family_eval_teacher_logits import (
    prepare_qwen_family_eval_teacher_logits,
)
from benchmarks.prepare_qwen_family_eval_teacher_metadata import (
    prepare_qwen_family_eval_teacher_metadata,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _prompt_pack(prompt_count: int = 1) -> dict[str, object]:
    return {
        "record_type": "qwen_family_eval_prompt_probe",
        "prompt_pack_ready": True,
        "missing_requirements": [],
        "split_counts": {
            "report": prompt_count,
            "selection": prompt_count,
            "holdout": prompt_count,
        },
        "prompt_rows": [
            {
                "record_type": "qwen_family_eval_prompt",
                "split": split,
                "prompt_id": f"qwen_{split}_{idx:03d}",
                "prompt": f"{split} prompt {idx}",
                "token_count": 2,
                "encoded_token_ids": [1, 2],
            }
            for split in ("report", "selection", "holdout")
            for idx in range(prompt_count)
        ],
    }


def _clean_benchmark_rows(
    *, candidate_latency_ms: float = 3.0, control_latency_ms: float = 2.0
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in ("prefill_1k", "decode_128"):
        rows.extend(
            [
                {
                    "record_type": "qwen_family_benchmark_row",
                    "scenario": scenario,
                    "role": "candidate",
                    "run_index": 0,
                    "latency_ms": candidate_latency_ms,
                    "pageouts_delta": 0,
                    "swapouts_delta": 0,
                    "effective_bpw": 1.25,
                    "dtype_parity": True,
                    "dense_routed_experts": False,
                    "unbound_vq_experts": False,
                    "non_expert_dtype_status": "verified",
                },
                {
                    "record_type": "qwen_family_benchmark_row",
                    "scenario": scenario,
                    "role": "control",
                    "run_index": 0,
                    "latency_ms": control_latency_ms,
                    "pageouts_delta": 0,
                    "swapouts_delta": 0,
                },
            ]
        )
    return rows


class _FakeQwenSwitch:
    input_dims = 4
    num_experts = 16

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        return mx.zeros((*indices.shape, self.input_dims), dtype=x.dtype)


def test_qwen_eval_row_prep_joins_metrics_with_verified_teacher_metadata(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_clean_rows_per_split=1,
    )
    prompt_pack = _prompt_pack()
    metrics = [
        {
            "record_type": "qwen_family_eval_metric_row",
            "split": split,
            "prompt_id": f"qwen_{split}_000",
            "nll": 1.0,
            "ppl": 2.0,
            "mean_kld": 0.1,
            "top1_match": 1.0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        }
        for split in ("report", "selection", "holdout")
    ]
    metadata = [
        {
            "record_type": "qwen_teacher_cache_metadata",
            "prompt_id": f"qwen_{split}_000",
            "teacher_model_id": "Qwen/Qwen3.6-35B-A3B",
            "teacher_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
            "teacher_cache_metadata_verified": True,
        }
        for split in ("report", "selection", "holdout")
    ]
    output_jsonl = tmp_path / "qwen-family-eval-rows.jsonl"

    payload = prepare_qwen_family_eval_rows(
        family_policy=policy,
        eval_prompt_pack=prompt_pack,
        eval_metric_rows=metrics,
        teacher_cache_metadata_rows=metadata,
        output_jsonl=output_jsonl,
    )

    assert payload["eval_row_count"] == 3
    assert payload["missing_requirements"] == []
    gate = check_qwen_family_eval_gate(
        family_policy=policy,
        eval_jsonl_paths=(output_jsonl,),
        eval_prompt_pack=prompt_pack,
    )
    assert gate["family_eval_gate_pass"] is True
    assert gate["split_counts"] == {"holdout": 1, "report": 1, "selection": 1}


def test_qwen_eval_row_prep_refuses_rows_without_teacher_metadata(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_clean_rows_per_split=1,
    )
    prompt_pack = _prompt_pack()
    metrics = [
        {
            "record_type": "qwen_family_eval_metric_row",
            "split": split,
            "prompt_id": f"qwen_{split}_000",
            "nll": 1.0,
            "ppl": 2.0,
            "mean_kld": 0.1,
            "top1_match": 1.0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        }
        for split in ("report", "selection", "holdout")
    ]
    output_jsonl = tmp_path / "qwen-family-eval-rows.jsonl"

    payload = prepare_qwen_family_eval_rows(
        family_policy=policy,
        eval_prompt_pack=prompt_pack,
        eval_metric_rows=metrics,
        teacher_cache_metadata_rows=[],
        output_jsonl=output_jsonl,
    )

    assert payload["eval_row_count"] == 0
    assert output_jsonl.read_text() == ""
    assert "qwen_eval_teacher_cache_metadata" in payload["missing_requirements"]
    assert payload["missing_teacher_metadata_counts"] == {
        "holdout": 1,
        "report": 1,
        "selection": 1,
    }


class _TinyTokenizer:
    vocab_size = 8

    def __len__(self) -> int:
        return 8

    def encode(self, prompt: str, *, add_special_tokens: bool = False) -> list[int]:
        return [1, 2]


def test_qwen_teacher_metadata_probe_feeds_eval_row_prep_metadata_only(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_clean_rows_per_split=1,
    )
    prompt_pack = _prompt_pack()
    metadata_jsonl = tmp_path / "qwen-teacher-metadata.jsonl"

    payload = prepare_qwen_family_eval_teacher_metadata(
        family_policy=policy,
        eval_prompt_pack=prompt_pack,
        tokenizer_dir=tmp_path,
        source_dir=tmp_path,
        output_jsonl=metadata_jsonl,
        tokenizer=_TinyTokenizer(),
    )
    row_probe = prepare_qwen_family_eval_rows(
        family_policy=policy,
        eval_prompt_pack=prompt_pack,
        eval_metric_rows=[],
        teacher_cache_metadata_rows=_read_jsonl(metadata_jsonl),
        output_jsonl=tmp_path / "qwen-family-eval-rows.jsonl",
    )

    assert payload["metadata_row_count"] == 3
    assert payload["missing_requirements"] == []
    assert row_probe["missing_teacher_metadata_counts"] == {
        "holdout": 0,
        "report": 0,
        "selection": 0,
    }
    assert row_probe["missing_metric_counts"] == {
        "holdout": 1,
        "report": 1,
        "selection": 1,
    }
    assert row_probe["missing_requirements"] == ["qwen_eval_metric_rows"]


def test_qwen_eval_metric_row_prep_computes_candidate_teacher_metrics(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_clean_rows_per_split=1,
    )
    prompt_pack = _prompt_pack()
    candidate_rows = [
        {
            "record_type": "qwen_candidate_eval_logits",
            "prompt_id": f"qwen_{split}_000",
            "target_token_id": 1,
            "logits": [0.0, 3.0, 1.0],
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        }
        for split in ("report", "selection", "holdout")
    ]
    teacher_rows = [
        {
            "record_type": "qwen_teacher_eval_logits",
            "prompt_id": f"qwen_{split}_000",
            "target_token_id": 1,
            "logits": [0.0, 4.0, 1.0],
        }
        for split in ("report", "selection", "holdout")
    ]
    output_jsonl = tmp_path / "qwen-family-eval-metric-rows.jsonl"

    payload = prepare_qwen_family_eval_metric_rows(
        family_policy=policy,
        eval_prompt_pack=prompt_pack,
        candidate_logits_rows=candidate_rows,
        teacher_logits_rows=teacher_rows,
        output_jsonl=output_jsonl,
    )
    rows = _read_jsonl(output_jsonl)

    assert payload["metric_row_count"] == 3
    assert payload["missing_requirements"] == []
    assert {row["split"] for row in rows} == {"report", "selection", "holdout"}
    assert all(row["record_type"] == "qwen_family_eval_metric_row" for row in rows)
    assert all(row["metric_scope"] == "candidate_vs_source_teacher_logits" for row in rows)
    assert all(row["top1_match"] == 1.0 for row in rows)
    assert all(row["nll"] > 0.0 and row["ppl"] > 1.0 for row in rows)
    assert all(row["mean_kld"] >= 0.0 for row in rows)


def test_qwen_eval_metric_row_prep_accepts_compact_token_aligned_logits(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_clean_rows_per_split=1,
    )
    prompt_pack = _prompt_pack()
    candidate_rows = [
        {
            "prompt_id": f"qwen_{split}_000",
            "target_token_id": 42,
            "logit_token_ids": [7, 42, 99],
            "logits": [0.0, 3.0, 1.0],
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        }
        for split in ("report", "selection", "holdout")
    ]
    teacher_rows = [
        {
            "prompt_id": f"qwen_{split}_000",
            "target_token_id": 42,
            "logit_token_ids": [7, 42, 99],
            "logits": [0.0, 4.0, 1.0],
        }
        for split in ("report", "selection", "holdout")
    ]

    payload = prepare_qwen_family_eval_metric_rows(
        family_policy=policy,
        eval_prompt_pack=prompt_pack,
        candidate_logits_rows=candidate_rows,
        teacher_logits_rows=teacher_rows,
        output_jsonl=tmp_path / "qwen-family-eval-metric-rows.jsonl",
    )

    assert payload["metric_row_count"] == 3
    assert payload["missing_requirements"] == []


def test_qwen_eval_metric_row_prep_reports_missing_logits(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_clean_rows_per_split=1,
    )

    payload = prepare_qwen_family_eval_metric_rows(
        family_policy=policy,
        eval_prompt_pack=_prompt_pack(),
        candidate_logits_rows=[],
        teacher_logits_rows=[],
        output_jsonl=tmp_path / "qwen-family-eval-metric-rows.jsonl",
    )

    assert payload["metric_row_count"] == 0
    assert payload["missing_candidate_logit_counts"] == {
        "holdout": 1,
        "report": 1,
        "selection": 1,
    }
    assert payload["missing_teacher_logit_counts"] == {
        "holdout": 1,
        "report": 1,
        "selection": 1,
    }
    assert payload["missing_requirements"] == [
        "qwen_candidate_eval_logits",
        "qwen_teacher_eval_logits",
        "qwen_eval_metric_rows",
    ]


def test_qwen_candidate_logits_probe_surfaces_missing_runtime(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_clean_rows_per_split=1,
    )
    output_jsonl = tmp_path / "qwen-candidate-logits.jsonl"

    payload = prepare_qwen_family_eval_candidate_logits(
        family_policy=policy,
        eval_prompt_pack=_prompt_pack(),
        output_jsonl=output_jsonl,
    )

    assert output_jsonl.read_text() == ""
    assert payload["candidate_logit_row_count"] == 0
    assert payload["candidate_runtime_available"] is False
    assert payload["required_prompt_counts"] == {
        "holdout": 1,
        "report": 1,
        "selection": 1,
    }
    assert payload["missing_requirements"] == [
        "qwen_candidate_tokenizer_to_logits_runtime",
        "qwen_candidate_eval_logits",
    ]


def test_qwen_candidate_logits_probe_emits_teacher_aligned_rows(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_clean_rows_per_split=1,
    )
    shard = tmp_path / "model.safetensors"
    mx.save_safetensors(
        str(shard),
        {
            "model.language_model.embed_tokens.weight": mx.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                ],
                dtype=mx.float32,
            ),
            "lm_head.weight": mx.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                ],
                dtype=mx.float32,
            ),
        },
    )
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.language_model.embed_tokens.weight": shard.name,
                    "lm_head.weight": shard.name,
                },
            }
        )
    )
    teacher_rows = [
        {
            "record_type": "qwen_teacher_eval_logits",
            "prompt_id": f"qwen_{split}_000",
            "target_token_id": 2,
            "logit_token_ids": [0, 2, 3],
            "logits": [1.0, 2.0, 2.0],
        }
        for split in ("report", "selection", "holdout")
    ]
    output_jsonl = tmp_path / "qwen-candidate-logits.jsonl"

    payload = prepare_qwen_family_eval_candidate_logits(
        family_policy=policy,
        eval_prompt_pack=_prompt_pack(),
        source_dir=tmp_path,
        index_path=index_path,
        teacher_logits_rows=teacher_rows,
        output_jsonl=output_jsonl,
    )
    rows = _read_jsonl(output_jsonl)

    assert payload["candidate_runtime_available"] is True
    assert payload["candidate_logit_row_count"] == 3
    assert payload["missing_requirements"] == []
    assert {row["split"] for row in rows} == {"report", "selection", "holdout"}
    assert all(row["record_type"] == "qwen_candidate_eval_logits" for row in rows)
    assert all(row["target_token_id"] == 2 for row in rows)
    assert all(row["logit_token_ids"] == [0, 2, 3] for row in rows)
    assert all(row["logits"] == [1.0, 2.0, 2.0] for row in rows)
    assert all(row["pageouts_delta"] == 0 and row["swapouts_delta"] == 0 for row in rows)


def test_qwen_teacher_logits_probe_emits_compact_rows(
    tmp_path: Path,
) -> None:
    import mlx.core as mx

    from mlx_vq.io.source_safetensors import read_safetensors_tensor_mlx

    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_clean_rows_per_split=1,
    )
    shard = tmp_path / "model.safetensors"
    mx.save_safetensors(
        str(shard),
        {
            "model.language_model.embed_tokens.weight": mx.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                ],
                dtype=mx.float32,
            ),
            "lm_head.weight": mx.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                ],
                dtype=mx.float32,
            ),
        },
    )
    # Keep the import live so this test exercises the same safetensors reader path.
    assert read_safetensors_tensor_mlx(shard, "lm_head.weight").shape == (4, 2)
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.language_model.embed_tokens.weight": shard.name,
                    "lm_head.weight": shard.name,
                },
            }
        )
    )
    output_jsonl = tmp_path / "qwen-teacher-logits.jsonl"

    payload = prepare_qwen_family_eval_teacher_logits(
        family_policy=policy,
        eval_prompt_pack=_prompt_pack(),
        source_dir=tmp_path,
        index_path=index_path,
        output_jsonl=output_jsonl,
        selected_token_position="last",
        top_k=2,
    )
    rows = _read_jsonl(output_jsonl)

    assert payload["teacher_logit_row_count"] == 3
    assert payload["missing_requirements"] == []
    assert {row["split"] for row in rows} == {"report", "selection", "holdout"}
    assert all(row["record_type"] == "qwen_teacher_eval_logits" for row in rows)
    assert all(row["target_token_id"] in row["logit_token_ids"] for row in rows)
    assert all(len(row["logits"]) == len(row["logit_token_ids"]) for row in rows)


def test_qwen_eval_gate_reports_missing_split_evidence(tmp_path: Path) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_clean_rows_per_split=2,
    )

    payload = check_qwen_family_eval_gate(
        family_policy=policy,
        eval_jsonl_paths=(),
    )

    assert payload["record_type"] == "qwen_family_eval_gate_check"
    assert payload["family_eval_gate_pass"] is False
    assert payload["missing_requirements"] == [
        "qwen_report_eval_jsonl",
        "qwen_selection_eval_jsonl",
        "qwen_holdout_eval_jsonl",
    ]
    assert payload["split_counts"] == {"holdout": 0, "report": 0, "selection": 0}


def test_qwen_eval_gate_surfaces_eval_row_probe_blockers() -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_clean_rows_per_split=1,
    )
    row_probe = {
        "record_type": "qwen_family_eval_row_probe",
        "missing_requirements": [
            "qwen_eval_metric_rows",
            "qwen_eval_teacher_cache_metadata",
        ],
    }

    payload = check_qwen_family_eval_gate(
        family_policy=policy,
        eval_jsonl_paths=(),
        eval_row_probe=row_probe,
    )

    assert payload["eval_row_probe_missing_requirements"] == [
        "qwen_eval_metric_rows",
        "qwen_eval_teacher_cache_metadata",
    ]
    assert "qwen_eval_metric_rows" in payload["missing_requirements"]
    assert "qwen_eval_teacher_cache_metadata" in payload["missing_requirements"]


def test_qwen_eval_gate_passes_clean_split_rows(tmp_path: Path) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_clean_rows_per_split=2,
    )
    rows = []
    for split in ("report", "selection", "holdout"):
        for idx in range(2):
            rows.append(
                {
                    "record_type": "qwen_family_eval_row",
                    "split": split,
                    "prompt_id": f"{split}_{idx}",
                    "teacher_model_id": "Qwen/Qwen3.6-35B-A3B",
                    "teacher_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
                    "teacher_cache_metadata_verified": True,
                    "nll": 1.0 + idx,
                    "ppl": 2.0 + idx,
                    "mean_kld": 0.1,
                    "top1_match": 1.0,
                    "pageouts_delta": 0,
                    "swapouts_delta": 0,
                }
            )
    eval_jsonl = _write_jsonl(tmp_path / "qwen-eval.jsonl", rows)

    payload = check_qwen_family_eval_gate(
        family_policy=policy,
        eval_jsonl_paths=(eval_jsonl,),
    )

    assert payload["family_eval_gate_pass"] is True
    assert payload["split_counts"] == {"holdout": 2, "report": 2, "selection": 2}
    assert payload["missing_requirements"] == []
    assert payload["metric_summary"]["top1_mean"] == 1.0


def test_qwen_eval_gate_requires_prompt_pack_prompt_ids(tmp_path: Path) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_clean_rows_per_split=2,
    )
    prompt_pack = {
        "record_type": "qwen_family_eval_prompt_probe",
        "prompt_pack_ready": True,
        "missing_requirements": [],
        "split_counts": {"report": 2, "selection": 2, "holdout": 2},
        "prompt_rows": [
            {
                "record_type": "qwen_family_eval_prompt",
                "split": split,
                "prompt_id": f"qwen_{split}_{idx:03d}",
                "token_count": 2,
            }
            for split in ("report", "selection", "holdout")
            for idx in range(2)
        ],
    }
    rows = []
    for split in ("report", "selection", "holdout"):
        rows.append(
            {
                "record_type": "qwen_family_eval_row",
                "split": split,
                "prompt_id": f"qwen_{split}_000",
                "teacher_model_id": "Qwen/Qwen3.6-35B-A3B",
                "teacher_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
                "teacher_cache_metadata_verified": True,
                "nll": 1.0,
                "ppl": 2.0,
                "mean_kld": 0.1,
                "top1_match": 1.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
            }
        )
    eval_jsonl = _write_jsonl(tmp_path / "qwen-eval-partial.jsonl", rows)

    payload = check_qwen_family_eval_gate(
        family_policy=policy,
        eval_jsonl_paths=(eval_jsonl,),
        eval_prompt_pack=prompt_pack,
    )

    assert payload["family_eval_gate_pass"] is False
    assert payload["prompt_pack_required"] is True
    assert payload["required_prompt_counts"] == {
        "holdout": 2,
        "report": 2,
        "selection": 2,
    }
    assert payload["missing_prompt_ids_by_split"] == {
        "holdout": ["qwen_holdout_001"],
        "report": ["qwen_report_001"],
        "selection": ["qwen_selection_001"],
    }
    assert "qwen_report_eval_prompt_rows" in payload["missing_requirements"]
    assert "qwen_selection_eval_prompt_rows" in payload["missing_requirements"]
    assert "qwen_holdout_eval_prompt_rows" in payload["missing_requirements"]


def test_qwen_eval_gate_rejects_rows_without_teacher_cache_metadata(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_clean_rows_per_split=1,
    )
    rows = []
    for split in ("report", "selection", "holdout"):
        rows.append(
            {
                "record_type": "qwen_family_eval_row",
                "split": split,
                "prompt_id": f"qwen_{split}_000",
                "teacher_model_id": "Qwen/Qwen3.6-35B-A3B",
                "nll": 1.0,
                "ppl": 2.0,
                "mean_kld": 0.1,
                "top1_match": 1.0,
                "pageouts_delta": 0,
                "swapouts_delta": 0,
            }
        )
    eval_jsonl = _write_jsonl(tmp_path / "qwen-eval-no-metadata.jsonl", rows)

    payload = check_qwen_family_eval_gate(
        family_policy=policy,
        eval_jsonl_paths=(eval_jsonl,),
    )

    assert payload["family_eval_gate_pass"] is False
    assert payload["split_counts"] == {"holdout": 0, "report": 0, "selection": 0}
    assert payload["row_error_count"] == 3
    assert "qwen_eval_teacher_cache_metadata" in payload["missing_requirements"]
    assert payload["row_errors"] == [
        {"row_index": 0, "error": "teacher_cache_metadata_not_verified"},
        {"row_index": 1, "error": "teacher_cache_metadata_not_verified"},
        {"row_index": 2, "error": "teacher_cache_metadata_not_verified"},
    ]


def test_qwen_benchmark_gate_reports_missing_scenario_evidence() -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_repetitions_per_scenario=1,
    )

    payload = check_qwen_family_benchmark_gate(
        family_policy=policy,
        benchmark_jsonl_paths=(),
    )

    assert payload["record_type"] == "qwen_family_benchmark_gate_check"
    assert payload["family_benchmark_gate_pass"] is False
    assert payload["missing_requirements"] == [
        "qwen_prefill_1k_candidate_benchmark_rows",
        "qwen_prefill_1k_control_benchmark_rows",
        "qwen_decode_128_candidate_benchmark_rows",
        "qwen_decode_128_control_benchmark_rows",
    ]


def test_qwen_benchmark_gate_surfaces_benchmark_row_probe_blockers() -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_repetitions_per_scenario=2,
    )
    row_probe = {
        "record_type": "qwen_family_benchmark_row_probe",
        "missing_requirements": [
            "qwen_benchmark_latency_rows",
            "qwen_candidate_benchmark_artifact_invariants",
        ],
    }

    payload = check_qwen_family_benchmark_gate(
        family_policy=policy,
        benchmark_jsonl_paths=(),
        benchmark_row_probe=row_probe,
    )

    assert payload["benchmark_row_probe_missing_requirements"] == [
        "qwen_benchmark_latency_rows",
        "qwen_candidate_benchmark_artifact_invariants",
    ]
    assert "qwen_benchmark_latency_rows" in payload["missing_requirements"]
    assert "qwen_candidate_benchmark_artifact_invariants" in payload[
        "missing_requirements"
    ]


def test_qwen_benchmark_gate_rejects_placeholder_baseline_even_with_clean_rows(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_repetitions_per_scenario=1,
    )
    policy["benchmark_gate"]["comparison_baseline"] = (
        "qwen_source_or_qwen_control_runtime"
    )
    policy["benchmark_gate"].pop("same_machine_reference", None)
    benchmark_jsonl = _write_jsonl(
        tmp_path / "qwen-benchmark-clean-placeholder.jsonl",
        _clean_benchmark_rows(),
    )

    payload = check_qwen_family_benchmark_gate(
        family_policy=policy,
        benchmark_jsonl_paths=(benchmark_jsonl,),
    )

    assert payload["family_benchmark_gate_pass"] is False
    assert payload["comparison_baseline"] == "qwen_source_or_qwen_control_runtime"
    assert payload["same_machine_reference"] is False
    assert "qwen_benchmark_same_machine_reference_ratio" in payload[
        "missing_requirements"
    ]


def test_qwen_benchmark_gate_rejects_candidate_reference_ratio_above_policy(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_repetitions_per_scenario=1,
        maximum_candidate_to_reference_ratio=1.5,
    )
    benchmark_jsonl = _write_jsonl(
        tmp_path / "qwen-benchmark-over-ratio.jsonl",
        _clean_benchmark_rows(candidate_latency_ms=4.0, control_latency_ms=2.0),
    )

    payload = check_qwen_family_benchmark_gate(
        family_policy=policy,
        benchmark_jsonl_paths=(benchmark_jsonl,),
    )

    assert payload["family_benchmark_gate_pass"] is False
    assert payload["maximum_candidate_to_reference_ratio"] == 1.5
    assert payload["scenario_latency_ratios"] == {
        "decode_128": 2.0,
        "prefill_1k": 2.0,
    }
    assert "qwen_benchmark_candidate_reference_ratio" in payload[
        "missing_requirements"
    ]


def test_qwen_benchmark_row_prep_joins_latency_with_candidate_invariants(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_repetitions_per_scenario=1,
    )
    latency_rows = []
    invariant_rows = []
    for scenario in ("prefill_1k", "decode_128"):
        latency_rows.extend(
            [
                {
                    "record_type": "qwen_benchmark_latency_row",
                    "scenario": scenario,
                    "role": "candidate",
                    "run_index": 0,
                    "latency_ms": 3.0,
                    "pageouts_delta": 0,
                    "swapouts_delta": 0,
                },
                {
                    "record_type": "qwen_benchmark_latency_row",
                    "scenario": scenario,
                    "role": "control",
                    "run_index": 0,
                    "latency_ms": 2.0,
                    "pageouts_delta": 0,
                    "swapouts_delta": 0,
                },
            ]
        )
        invariant_rows.append(
            {
                "record_type": "qwen_candidate_benchmark_invariants",
                "scenario": scenario,
                "run_index": 0,
                "effective_bpw": 1.25,
                "dtype_parity": True,
                "dense_routed_experts": False,
                "unbound_vq_experts": False,
                "non_expert_dtype_status": "verified",
            }
        )
    output_jsonl = tmp_path / "qwen-family-benchmark-rows.jsonl"

    payload = prepare_qwen_family_benchmark_rows(
        family_policy=policy,
        latency_rows=latency_rows,
        candidate_invariant_rows=invariant_rows,
        output_jsonl=output_jsonl,
    )

    assert payload["benchmark_row_count"] == 4
    assert payload["missing_requirements"] == []
    gate = check_qwen_family_benchmark_gate(
        family_policy=policy,
        benchmark_jsonl_paths=(output_jsonl,),
    )
    assert gate["family_benchmark_gate_pass"] is True
    assert gate["scenario_counts"] == {
        "decode_128": {"candidate": 1, "control": 1},
        "prefill_1k": {"candidate": 1, "control": 1},
    }


def test_qwen_candidate_latency_probe_emits_clean_candidate_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_repetitions_per_scenario=2,
    )
    monkeypatch.setattr(qwen_candidate_latency, "reset_mlx_peak_memory", lambda: None)
    monkeypatch.setattr(qwen_candidate_latency, "collect_vm_stat_counts", lambda: {})
    monkeypatch.setattr(
        qwen_candidate_latency,
        "collect_metric_snapshot",
        lambda previous_vm_stat_counts=None: {
            "mlx_active_bytes": 100,
            "mlx_peak_bytes": 200,
            "mlx_cache_bytes": 50,
            "rss_bytes": 300,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    )
    latency_jsonl = tmp_path / "qwen-candidate-latency.jsonl"

    payload = qwen_candidate_latency.run_qwen_candidate_latency_benchmark(
        family_policy=policy,
        artifact_dir=tmp_path / "artifact",
        output_jsonl=latency_jsonl,
        layer=0,
        repetitions=2,
        warmup_repetitions=0,
        top_k=2,
        switch_loader=lambda artifact_dir, layer: _FakeQwenSwitch(),
    )

    rows = _read_jsonl(latency_jsonl)
    assert payload["candidate_latency_row_count"] == 4
    assert payload["missing_requirements"] == []
    assert payload["scenario_counts"] == {"decode_128": 2, "prefill_1k": 2}
    assert {row["role"] for row in rows} == {"candidate"}
    assert {row["scenario"] for row in rows} == {"decode_128", "prefill_1k"}
    assert all(row["latency_ms"] > 0 for row in rows)
    assert all(row["pageouts_delta"] == 0 for row in rows)
    assert all(
        row["benchmark_scope"] == "qwen_vq_switch_projection_candidate_latency"
        for row in rows
    )


def test_qwen_control_latency_probe_emits_clean_control_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_repetitions_per_scenario=2,
    )
    monkeypatch.setattr(qwen_control_latency, "reset_mlx_peak_memory", lambda: None)
    monkeypatch.setattr(qwen_control_latency, "collect_vm_stat_counts", lambda: {})
    monkeypatch.setattr(
        qwen_control_latency,
        "collect_metric_snapshot",
        lambda previous_vm_stat_counts=None: {
            "mlx_active_bytes": 100,
            "mlx_peak_bytes": 200,
            "mlx_cache_bytes": 50,
            "rss_bytes": 300,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        },
    )
    latency_jsonl = tmp_path / "qwen-control-latency.jsonl"

    payload = qwen_control_latency.run_qwen_control_latency_benchmark(
        family_policy=policy,
        source_dir=tmp_path / "source",
        index_path=tmp_path / "source" / "model.safetensors.index.json",
        output_jsonl=latency_jsonl,
        layer=0,
        repetitions=2,
        warmup_repetitions=0,
        top_k=2,
        control_loader=lambda source_dir, index_path, layer: _FakeQwenSwitch(),
    )

    rows = _read_jsonl(latency_jsonl)
    assert payload["control_latency_row_count"] == 4
    assert payload["missing_requirements"] == []
    assert payload["scenario_counts"] == {"decode_128": 2, "prefill_1k": 2}
    assert {row["role"] for row in rows} == {"control"}
    assert {row["scenario"] for row in rows} == {"decode_128", "prefill_1k"}
    assert all(row["latency_ms"] > 0 for row in rows)
    assert all(row["pageouts_delta"] == 0 for row in rows)
    assert all(
        row["benchmark_scope"] == "qwen_source_switch_projection_control_latency"
        for row in rows
    )


def test_qwen_control_latency_probe_retries_dirty_measured_attempts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_repetitions_per_scenario=2,
    )
    metric_snapshots = iter(
        [
            {
                "mlx_active_bytes": 100,
                "mlx_peak_bytes": 200,
                "mlx_cache_bytes": 50,
                "rss_bytes": 300,
                "pageouts_delta": 24,
                "swapouts_delta": 0,
            },
            *[
                {
                    "mlx_active_bytes": 100,
                    "mlx_peak_bytes": 200,
                    "mlx_cache_bytes": 50,
                    "rss_bytes": 300,
                    "pageouts_delta": 0,
                    "swapouts_delta": 0,
                }
                for _ in range(4)
            ],
        ]
    )
    monkeypatch.setattr(qwen_control_latency, "reset_mlx_peak_memory", lambda: None)
    monkeypatch.setattr(qwen_control_latency, "collect_vm_stat_counts", lambda: {})
    monkeypatch.setattr(
        qwen_control_latency,
        "collect_metric_snapshot",
        lambda previous_vm_stat_counts=None: next(metric_snapshots),
    )
    latency_jsonl = tmp_path / "qwen-control-latency.jsonl"

    payload = qwen_control_latency.run_qwen_control_latency_benchmark(
        family_policy=policy,
        source_dir=tmp_path / "source",
        index_path=tmp_path / "source" / "model.safetensors.index.json",
        output_jsonl=latency_jsonl,
        layer=0,
        repetitions=2,
        warmup_repetitions=0,
        top_k=2,
        control_loader=lambda source_dir, index_path, layer: _FakeQwenSwitch(),
    )

    rows = _read_jsonl(latency_jsonl)
    assert payload["control_latency_row_count"] == 4
    assert payload["scenario_counts"] == {"decode_128": 2, "prefill_1k": 2}
    assert payload["discarded_attempt_counts"] == {
        "decode_128": 0,
        "prefill_1k": 1,
    }
    assert payload["missing_requirements"] == []
    assert {
        (row["scenario"], row["run_index"], row["pageouts_delta"])
        for row in rows
    } == {
        ("prefill_1k", 0, 0),
        ("prefill_1k", 1, 0),
        ("decode_128", 0, 0),
        ("decode_128", 1, 0),
    }


def test_qwen_benchmark_invariant_prep_emits_rows_from_audit_and_bind_probe(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_repetitions_per_scenario=2,
    )
    invariant_jsonl = tmp_path / "qwen-candidate-invariants.jsonl"

    payload = prepare_qwen_family_benchmark_invariants(
        family_policy=policy,
        artifact_audit={
            "audit_pass": True,
            "effective_routed_bpw": 1.03125,
            "dense_routed_experts": False,
            "unbound_vq_experts": False,
        },
        non_expert_bind_probe={
            "binding_pass": True,
            "loaded_model_parameter_count": 610,
            "loaded_dtype_names": {
                "language_model.layers.0.input_layernorm.weight": "bfloat16"
            },
            "missing_model_parameters": [],
        },
        output_jsonl=invariant_jsonl,
    )

    invariant_rows = _read_jsonl(invariant_jsonl)
    assert payload["candidate_invariant_row_count"] == 4
    assert payload["missing_requirements"] == []
    assert {row["scenario"] for row in invariant_rows} == {"decode_128", "prefill_1k"}
    assert {row["run_index"] for row in invariant_rows} == {0, 1}
    assert all(row["effective_bpw"] == 1.03125 for row in invariant_rows)
    assert all(row["non_expert_dtype_status"] == "verified" for row in invariant_rows)

    latency_rows = [
        {
            "record_type": "qwen_benchmark_latency_row",
            "scenario": scenario,
            "role": role,
            "run_index": run_index,
            "latency_ms": 3.0 if role == "candidate" else 2.0,
            "pageouts_delta": 0,
            "swapouts_delta": 0,
        }
        for scenario in ("prefill_1k", "decode_128")
        for role in ("candidate", "control")
        for run_index in range(2)
    ]
    benchmark_jsonl = tmp_path / "qwen-family-benchmark-rows.jsonl"
    row_payload = prepare_qwen_family_benchmark_rows(
        family_policy=policy,
        latency_rows=latency_rows,
        candidate_invariant_rows=invariant_rows,
        output_jsonl=benchmark_jsonl,
    )
    assert row_payload["benchmark_row_count"] == 8
    assert row_payload["missing_requirements"] == []

    gate = check_qwen_family_benchmark_gate(
        family_policy=policy,
        benchmark_jsonl_paths=(benchmark_jsonl,),
    )
    assert gate["family_benchmark_gate_pass"] is True


def test_qwen_benchmark_invariant_prep_refuses_incomplete_evidence(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_repetitions_per_scenario=1,
    )
    invariant_jsonl = tmp_path / "qwen-candidate-invariants.jsonl"

    payload = prepare_qwen_family_benchmark_invariants(
        family_policy=policy,
        artifact_audit={
            "audit_pass": True,
            "effective_routed_bpw": None,
            "dense_routed_experts": False,
            "unbound_vq_experts": False,
        },
        non_expert_bind_probe={
            "binding_pass": False,
            "loaded_model_parameter_count": 0,
            "loaded_dtype_names": {},
            "missing_model_parameters": ["language_model.layers.0.input_layernorm.weight"],
        },
        output_jsonl=invariant_jsonl,
    )

    assert _read_jsonl(invariant_jsonl) == []
    assert payload["candidate_invariant_row_count"] == 0
    assert "qwen_candidate_effective_bpw" in payload["missing_requirements"]
    assert "qwen_candidate_non_expert_dtype_verification" in payload[
        "missing_requirements"
    ]


def test_qwen_benchmark_row_prep_refuses_candidate_without_invariants(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_repetitions_per_scenario=1,
    )
    latency_rows = []
    for scenario in ("prefill_1k", "decode_128"):
        latency_rows.extend(
            [
                {
                    "record_type": "qwen_benchmark_latency_row",
                    "scenario": scenario,
                    "role": "candidate",
                    "run_index": 0,
                    "latency_ms": 3.0,
                    "pageouts_delta": 0,
                    "swapouts_delta": 0,
                },
                {
                    "record_type": "qwen_benchmark_latency_row",
                    "scenario": scenario,
                    "role": "control",
                    "run_index": 0,
                    "latency_ms": 2.0,
                    "pageouts_delta": 0,
                    "swapouts_delta": 0,
                },
            ]
        )
    output_jsonl = tmp_path / "qwen-family-benchmark-rows.jsonl"

    payload = prepare_qwen_family_benchmark_rows(
        family_policy=policy,
        latency_rows=latency_rows,
        candidate_invariant_rows=[],
        output_jsonl=output_jsonl,
    )

    assert payload["benchmark_row_count"] == 2
    assert payload["scenario_counts"] == {
        "decode_128": {"candidate": 0, "control": 1},
        "prefill_1k": {"candidate": 0, "control": 1},
    }
    assert "qwen_candidate_benchmark_artifact_invariants" in payload[
        "missing_requirements"
    ]
    gate = check_qwen_family_benchmark_gate(
        family_policy=policy,
        benchmark_jsonl_paths=(output_jsonl,),
    )
    assert gate["family_benchmark_gate_pass"] is False
    assert gate["scenario_counts"] == payload["scenario_counts"]


def test_qwen_benchmark_gate_passes_clean_candidate_and_control_rows(tmp_path: Path) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_repetitions_per_scenario=2,
    )
    rows = []
    for scenario in ("prefill_1k", "decode_128"):
        for role, latency in (("candidate", 3.0), ("control", 2.0)):
            for idx in range(2):
                rows.append(
                    {
                        "record_type": "qwen_family_benchmark_row",
                        "scenario": scenario,
                        "role": role,
                        "run_index": idx,
                        "latency_ms": latency + idx,
                        "effective_bpw": 1.25 if role == "candidate" else None,
                        "dtype_parity": True,
                        "dense_routed_experts": False,
                        "unbound_vq_experts": False,
                        "non_expert_dtype_status": "verified",
                        "pageouts_delta": 0,
                        "swapouts_delta": 0,
                    }
                )
    benchmark_jsonl = _write_jsonl(tmp_path / "qwen-benchmark.jsonl", rows)

    payload = check_qwen_family_benchmark_gate(
        family_policy=policy,
        benchmark_jsonl_paths=(benchmark_jsonl,),
    )

    assert payload["family_benchmark_gate_pass"] is True
    assert payload["scenario_counts"]["prefill_1k"] == {"candidate": 2, "control": 2}
    assert payload["scenario_counts"]["decode_128"] == {"candidate": 2, "control": 2}
    assert payload["missing_requirements"] == []
    assert payload["scenario_latency_ratios"]["prefill_1k"] == 1.4


def test_qwen_benchmark_gate_rejects_candidate_rows_missing_artifact_invariants(
    tmp_path: Path,
) -> None:
    policy = define_qwen_family_gate_policy(
        model_id="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        minimum_repetitions_per_scenario=1,
    )
    rows = []
    for scenario in ("prefill_1k", "decode_128"):
        rows.extend(
            [
                {
                    "record_type": "qwen_family_benchmark_row",
                    "scenario": scenario,
                    "role": "candidate",
                    "latency_ms": 3.0,
                    "pageouts_delta": 0,
                    "swapouts_delta": 0,
                },
                {
                    "record_type": "qwen_family_benchmark_row",
                    "scenario": scenario,
                    "role": "control",
                    "latency_ms": 2.0,
                    "pageouts_delta": 0,
                    "swapouts_delta": 0,
                },
            ]
        )
    benchmark_jsonl = _write_jsonl(tmp_path / "qwen-benchmark-missing-invariants.jsonl", rows)

    payload = check_qwen_family_benchmark_gate(
        family_policy=policy,
        benchmark_jsonl_paths=(benchmark_jsonl,),
    )

    assert payload["family_benchmark_gate_pass"] is False
    assert payload["scenario_counts"] == {
        "decode_128": {"candidate": 0, "control": 1},
        "prefill_1k": {"candidate": 0, "control": 1},
    }
    assert payload["candidate_invariant_error_count"] == 2
    assert "qwen_candidate_benchmark_artifact_invariants" in payload["missing_requirements"]
    assert "qwen_prefill_1k_candidate_benchmark_rows" in payload["missing_requirements"]
    assert "qwen_decode_128_candidate_benchmark_rows" in payload["missing_requirements"]
