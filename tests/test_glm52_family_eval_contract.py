from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from benchmarks.define_glm52_family_gate_policy import (
    define_glm52_family_gate_policy,
    main as policy_main,
)
from benchmarks.prepare_glm52_family_eval_prompts import (
    DEFAULT_GLM52_EVAL_PROMPTS,
    GLM52EvalPrompt,
    build_glm52_family_eval_prompt_pack,
    main as prompt_main,
    probe_glm52_family_eval_prompts,
)
from keep.quality.glm52_family import (
    GLM52_PINNED_TOKENIZER_FILES,
    canonical_sha256,
)


MODEL_ID = "0xSero/glm-5.2-reap-504B-v2"
REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--0xSero--glm-5.2-reap-504B-v2"
    / "snapshots"
    / REVISION
)
TOKENIZER_READINESS = Path(
    "artifacts/quality/glm52-tokenizer-readiness-20260709.json"
)
PROMPT_TEXT_SHA256 = "3c39f6bb133907cf2fd93260a3cc548c0b186a264230568678cd076f1638eb6a"
POLICY_CONTRACT_SHA256 = "57ae812222de79555774296129e7110750f492794b1fe21bac91ed5cc986f206"


class _Tokenizer:
    vocab_size = 154_820

    def __len__(self) -> int:
        return 154_856

    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        words = prompt.split()
        return [100 + index + len(word) for index, word in enumerate(words)]


def test_glm52_family_policy_freezes_source_relative_wow_gates() -> None:
    policy = define_glm52_family_gate_policy(
        model_id=MODEL_ID,
        revision=REVISION,
    )

    assert policy["record_type"] == "glm52_family_gate_policy"
    assert policy["policy_status"] == "glm52_family_policy_frozen"
    assert policy["model_id"] == MODEL_ID
    assert policy["revision"] == REVISION
    assert policy["thresholds_frozen_before_candidate_metrics"] is True
    assert policy["teacher"] == {
        "source_kind": "deterministically_dequantized_pinned_modelopt_nvfp4",
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": "modelopt_nvfp4_v1",
        "activation_quantization_emulated": False,
        "exact_w4a4_runtime_parity_claimed": False,
        "bf16_teacher_claimed": False,
    }
    assert policy["eval_gate"] == {
        "required_splits": ["report", "selection", "holdout"],
        "required_domains": ["route", "math", "instruction"],
        "prompt_text_sha256": PROMPT_TEXT_SHA256,
        "minimum_clean_rows_per_split": 22,
        "minimum_total_clean_rows": 66,
        "full_vocabulary_logits_required": True,
        "compact_topk_kld_accepted": False,
        "mean_kld_max": 0.30,
        "p999_kld_max": 3.0,
        "top1_min": 0.85,
        "domain_top1_min": 0.80,
        "mean_ppl_ratio_max": 1.05,
        "dirty_rows_accepted": False,
        "holdout_tuning_forbidden": True,
    }
    assert policy["benchmark_gate"] == {
        "required_scenarios": ["prefill_1k", "decode_128"],
        "minimum_repetitions_per_scenario": 3,
        "comparison_baseline": "same_machine_pinned_fp4_source_streaming_control",
        "maximum_candidate_to_reference_ratio": 1.15,
        "candidate_and_control_same_machine": True,
        "pageouts_and_swapouts_must_be_zero": True,
    }
    assert policy["artifact_target"] == {
        "tensor_payload_bytes": 98_433_923_808,
        "whole_main_bpw": 1.5934433,
        "target_accepted_by_user": True,
        "actual_full_artifact_required": True,
    }
    assert policy["tokenizer_identity"]["files"] == GLM52_PINNED_TOKENIZER_FILES
    assert policy["policy_contract_sha256"] == POLICY_CONTRACT_SHA256


def test_glm52_prompt_pack_is_frozen_balanced_and_holdout_safe() -> None:
    payload = build_glm52_family_eval_prompt_pack(
        _Tokenizer(),
        model_id=MODEL_ID,
        revision=REVISION,
        expected_vocab_size=154_880,
    )

    assert payload["record_type"] == "glm52_family_eval_prompt_pack"
    assert payload["prompt_pack_status"] == "glm52_family_eval_prompt_pack_frozen"
    assert payload["prompt_pack_ready"] is True
    assert payload["prompt_pack_frozen_before_candidate_metrics"] is True
    assert payload["holdout_tuning_forbidden"] is True
    assert payload["required_splits"] == ["report", "selection", "holdout"]
    assert payload["required_domains"] == ["route", "math", "instruction"]
    assert payload["split_counts"] == {
        "report": 22,
        "selection": 22,
        "holdout": 22,
    }
    assert payload["domain_counts_by_split"] == {
        split: {"route": 8, "math": 7, "instruction": 7}
        for split in ("report", "selection", "holdout")
    }
    assert payload["prompt_row_count"] == 66
    assert payload["prompt_text_sha256"] == PROMPT_TEXT_SHA256
    assert payload["tokenizer_vocab_size"] == 154_820
    assert payload["tokenizer_length"] == 154_856
    assert payload["expected_vocab_size"] == 154_880
    assert payload["missing_requirements"] == []

    rows = payload["prompt_rows"]
    assert len({row["prompt_id"] for row in rows}) == 66
    assert len({row["prompt"] for row in rows}) == 66
    assert all(row["token_count"] >= 2 for row in rows)
    assert all(row["max_token_id"] < 154_880 for row in rows)
    assert all(row["teacher_logit_scope"] == "full_vocabulary" for row in rows)
    assert all(row["candidate_logit_scope"] == "full_vocabulary" for row in rows)
    assert all(row["kld_scope"] == "full_vocabulary" for row in rows)
    assert all(row["memory_clean_required"] is True for row in rows)
    assert all(
        row["tuning_eligible"] is (row["split"] == "selection")
        for row in rows
    )


def test_default_prompt_authority_has_exact_frozen_quota() -> None:
    assert set(DEFAULT_GLM52_EVAL_PROMPTS) == {"report", "selection", "holdout"}
    for rows in DEFAULT_GLM52_EVAL_PROMPTS.values():
        assert len(rows) == 22
        assert [row.domain for row in rows].count("route") == 8
        assert [row.domain for row in rows].count("math") == 7
        assert [row.domain for row in rows].count("instruction") == 7


def test_prompt_pack_rejects_out_of_vocabulary_token() -> None:
    class _BadTokenizer(_Tokenizer):
        def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
            return [1, 154_880]

    with pytest.raises(ValueError, match="outside expected model vocabulary"):
        build_glm52_family_eval_prompt_pack(
            _BadTokenizer(),
            model_id=MODEL_ID,
            revision=REVISION,
            expected_vocab_size=154_880,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "wrong/model", "model_id"),
        ("revision", "main", "revision"),
    ],
)
def test_policy_rejects_unpinned_identity(
    field: str,
    value: str,
    message: str,
) -> None:
    kwargs = {"model_id": MODEL_ID, "revision": REVISION}
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        define_glm52_family_gate_policy(**kwargs)


def test_real_pinned_prompt_probe_uses_ready_local_tokenizer(
    tmp_path: Path,
) -> None:
    if not (SNAPSHOT / "tokenizer.json").is_file() or not TOKENIZER_READINESS.is_file():
        pytest.skip("pinned GLM52 tokenizer evidence is not resident")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            define_glm52_family_gate_policy(
                model_id=MODEL_ID,
                revision=REVISION,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    payload = probe_glm52_family_eval_prompts(
        tokenizer_dir=SNAPSHOT,
        tokenizer_readiness_json=TOKENIZER_READINESS,
        family_policy_json=policy_path,
        model_id=MODEL_ID,
        revision=REVISION,
    )

    assert payload["prompt_pack_ready"] is True
    assert payload["prompt_row_count"] == 66
    assert payload["prompt_text_sha256"] == PROMPT_TEXT_SHA256
    assert payload["family_policy_contract_sha256"] == POLICY_CONTRACT_SHA256
    assert payload["local_files_only"] is True
    assert payload["trust_remote_code"] is False
    assert all(row["max_token_id"] < 154_880 for row in payload["prompt_rows"])
    finalized = dict(payload)
    embedded_digest = finalized.pop("prompt_pack_contract_sha256")
    assert canonical_sha256(finalized) == embedded_digest
    finalized["prompt_rows"][0]["encoded_token_ids"][0] += 1
    assert canonical_sha256(finalized) != embedded_digest


def test_policy_and_prompt_clis_write_durable_success_and_failure_json(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / "policy.json"
    assert policy_main(
        [
            "--model-id",
            MODEL_ID,
            "--revision",
            REVISION,
            "--output-json",
            str(policy_path),
        ]
    ) == 0
    assert json.loads(policy_path.read_text())["policy_contract_sha256"] == (
        POLICY_CONTRACT_SHA256
    )

    failed_policy_path = tmp_path / "failed-policy.json"
    assert policy_main(
        [
            "--model-id",
            MODEL_ID,
            "--revision",
            "main",
            "--output-json",
            str(failed_policy_path),
        ]
    ) == 1
    failed_policy = json.loads(failed_policy_path.read_text())
    assert failed_policy["policy_status"] == "glm52_family_policy_failed"
    assert failed_policy["thresholds_frozen_before_candidate_metrics"] is False

    failed_prompt_path = tmp_path / "failed-prompts.json"
    assert prompt_main(
        [
            "--tokenizer-dir",
            str(tmp_path / "missing-tokenizer"),
            "--tokenizer-readiness-json",
            str(tmp_path / "missing-readiness.json"),
            "--family-policy-json",
            str(policy_path),
            "--model-id",
            MODEL_ID,
            "--revision",
            REVISION,
            "--output-json",
            str(failed_prompt_path),
        ]
    ) == 1
    failed_prompt = json.loads(failed_prompt_path.read_text())
    assert failed_prompt["prompt_pack_ready"] is False
    assert failed_prompt["holdout_tuning_forbidden"] is True


def test_prompt_probe_rejects_policy_body_mutation_with_copied_digest(
    tmp_path: Path,
) -> None:
    if not (SNAPSHOT / "tokenizer.json").is_file() or not TOKENIZER_READINESS.is_file():
        pytest.skip("pinned GLM52 tokenizer evidence is not resident")
    policy = define_glm52_family_gate_policy(
        model_id=MODEL_ID,
        revision=REVISION,
    )
    policy["eval_gate"]["compact_topk_kld_accepted"] = True
    policy_path = tmp_path / "mutated-policy.json"
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="digest does not authenticate"):
        probe_glm52_family_eval_prompts(
            tokenizer_dir=SNAPSHOT,
            tokenizer_readiness_json=TOKENIZER_READINESS,
            family_policy_json=policy_path,
            model_id=MODEL_ID,
            revision=REVISION,
        )

    policy_body = dict(policy)
    policy_body.pop("policy_contract_sha256")
    policy["policy_contract_sha256"] = canonical_sha256(policy_body)
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="body does not match"):
        probe_glm52_family_eval_prompts(
            tokenizer_dir=SNAPSHOT,
            tokenizer_readiness_json=TOKENIZER_READINESS,
            family_policy_json=policy_path,
            model_id=MODEL_ID,
            revision=REVISION,
        )

    policy["policy_contract_sha256"] = POLICY_CONTRACT_SHA256
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")

    output = tmp_path / "failed-prompts.json"
    assert prompt_main(
        [
            "--tokenizer-dir",
            str(SNAPSHOT),
            "--tokenizer-readiness-json",
            str(TOKENIZER_READINESS),
            "--family-policy-json",
            str(policy_path),
            "--model-id",
            MODEL_ID,
            "--revision",
            REVISION,
            "--output-json",
            str(output),
        ]
    ) == 1
    failure = json.loads(output.read_text())
    assert failure["prompt_pack_ready"] is False
    assert "digest does not authenticate" in failure["input_error"]["message"]


def test_prompt_probe_rejects_tokenizer_byte_drift_despite_ready_claim(
    tmp_path: Path,
) -> None:
    if not (SNAPSHOT / "tokenizer.json").is_file() or not TOKENIZER_READINESS.is_file():
        pytest.skip("pinned GLM52 tokenizer evidence is not resident")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    for filename in GLM52_PINNED_TOKENIZER_FILES:
        shutil.copy2(SNAPSHOT / filename, tokenizer_dir / filename)
    tokenizer_config = tokenizer_dir / "tokenizer_config.json"
    raw = bytearray(tokenizer_config.read_bytes())
    raw[-2] ^= 1
    tokenizer_config.write_bytes(raw)

    readiness = json.loads(TOKENIZER_READINESS.read_text())
    readiness["tokenizer_dir"] = str(tokenizer_dir)
    for filename, record in readiness["file_identities"].items():
        record["path"] = str(tokenizer_dir / filename)
    readiness_path = tmp_path / "fabricated-readiness.json"
    readiness_path.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            define_glm52_family_gate_policy(
                model_id=MODEL_ID,
                revision=REVISION,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="identity mismatch for tokenizer_config"):
        probe_glm52_family_eval_prompts(
            tokenizer_dir=tokenizer_dir,
            tokenizer_readiness_json=readiness_path,
            family_policy_json=policy_path,
            model_id=MODEL_ID,
            revision=REVISION,
        )


def test_prompt_probe_rejects_fabricated_tokenizer_readiness_claim(
    tmp_path: Path,
) -> None:
    if not (SNAPSHOT / "tokenizer.json").is_file() or not TOKENIZER_READINESS.is_file():
        pytest.skip("pinned GLM52 tokenizer evidence is not resident")
    readiness = json.loads(TOKENIZER_READINESS.read_text())
    readiness["trust_remote_code"] = True
    readiness_path = tmp_path / "fabricated-readiness.json"
    readiness_path.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            define_glm52_family_gate_policy(
                model_id=MODEL_ID,
                revision=REVISION,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="trust_remote_code must be False"):
        probe_glm52_family_eval_prompts(
            tokenizer_dir=SNAPSHOT,
            tokenizer_readiness_json=readiness_path,
            family_policy_json=policy_path,
            model_id=MODEL_ID,
            revision=REVISION,
        )


def test_prompt_builder_rejects_quota_valid_replacement_corpus() -> None:
    prompts = {split: tuple(rows) for split, rows in DEFAULT_GLM52_EVAL_PROMPTS.items()}
    report = list(prompts["report"])
    report[0] = GLM52EvalPrompt(
        domain="route",
        prompt="This quota valid replacement must not become the frozen corpus.",
    )
    prompts["report"] = tuple(report)

    with pytest.raises(ValueError, match="does not match the pinned GLM52 corpus"):
        build_glm52_family_eval_prompt_pack(
            _Tokenizer(),
            model_id=MODEL_ID,
            revision=REVISION,
            expected_vocab_size=154_880,
            prompts_by_split=prompts,
        )


def test_prompt_cli_rejects_output_aliases_without_mutating_inputs(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            define_glm52_family_gate_policy(
                model_id=MODEL_ID,
                revision=REVISION,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    original_policy = policy_path.read_bytes()
    assert prompt_main(
        [
            "--tokenizer-dir",
            str(tmp_path / "tokenizer"),
            "--tokenizer-readiness-json",
            str(tmp_path / "readiness.json"),
            "--family-policy-json",
            str(policy_path),
            "--model-id",
            MODEL_ID,
            "--revision",
            REVISION,
            "--output-json",
            str(policy_path),
        ]
    ) == 1
    assert policy_path.read_bytes() == original_policy

    if not (SNAPSHOT / "config.json").is_file() or not TOKENIZER_READINESS.is_file():
        pytest.skip("pinned GLM52 tokenizer evidence is not resident")
    config_path = SNAPSHOT / "config.json"
    original_config = config_path.read_bytes()
    assert prompt_main(
        [
            "--tokenizer-dir",
            str(SNAPSHOT),
            "--tokenizer-readiness-json",
            str(TOKENIZER_READINESS),
            "--family-policy-json",
            str(policy_path),
            "--model-id",
            MODEL_ID,
            "--revision",
            REVISION,
            "--output-json",
            str(config_path),
        ]
    ) == 1
    assert config_path.read_bytes() == original_config

    blob_path = config_path.resolve()
    original_blob = blob_path.read_bytes()
    assert prompt_main(
        [
            "--tokenizer-dir",
            str(SNAPSHOT),
            "--tokenizer-readiness-json",
            str(TOKENIZER_READINESS),
            "--family-policy-json",
            str(policy_path),
            "--model-id",
            MODEL_ID,
            "--revision",
            REVISION,
            "--output-json",
            str(blob_path),
        ]
    ) == 1
    assert blob_path.read_bytes() == original_blob

    tokenizer_alias = tmp_path / "tokenizer-alias"
    tokenizer_alias.symlink_to(SNAPSHOT, target_is_directory=True)
    config_alias = tmp_path / "config-alias.json"
    config_alias.symlink_to(config_path)
    assert prompt_main(
        [
            "--tokenizer-dir",
            str(tokenizer_alias),
            "--tokenizer-readiness-json",
            str(TOKENIZER_READINESS),
            "--family-policy-json",
            str(policy_path),
            "--model-id",
            MODEL_ID,
            "--revision",
            REVISION,
            "--output-json",
            str(config_alias),
        ]
    ) == 1
    assert config_path.read_bytes() == original_config
    assert blob_path.read_bytes() == original_blob
