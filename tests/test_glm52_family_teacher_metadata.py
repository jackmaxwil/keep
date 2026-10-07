from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.prepare_glm52_family_eval_teacher_metadata import (
    main,
    prepare_glm52_family_eval_teacher_metadata,
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
PROFILE = Path("models/glm52-reap-504b-v2.yaml")
CONFIG = SNAPSHOT / "config.json"
INDEX = SNAPSHOT / "model.safetensors.index.json"
SOURCE_AUDIT = Path("artifacts/quality/glm52-reap-source-audit-20260709.json")
SOURCE_PAYLOAD_AUDIT = Path(
    "artifacts/quality/glm52-reap-source-payload-audit-20260709.json"
)
TOKENIZER_READINESS = Path(
    "artifacts/quality/glm52-tokenizer-readiness-20260709.json"
)
FAMILY_POLICY = Path("artifacts/quality/glm52-family-policy-20260709-v2.json")
PROMPT_PACK = Path("artifacts/quality/glm52-family-eval-prompts-20260709-v2.json")


def _require_resident_authority() -> None:
    required = (
        PROFILE,
        CONFIG,
        INDEX,
        SOURCE_AUDIT,
        SOURCE_PAYLOAD_AUDIT,
        TOKENIZER_READINESS,
        FAMILY_POLICY,
        PROMPT_PACK,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("pinned GLM52 teacher metadata authority is not resident")


def _prepare(output_dir: Path) -> dict[str, object]:
    return prepare_glm52_family_eval_teacher_metadata(
        source_dir=SNAPSHOT,
        profile_path=PROFILE,
        config_path=CONFIG,
        index_path=INDEX,
        source_audit_json=SOURCE_AUDIT,
        source_payload_audit_json=SOURCE_PAYLOAD_AUDIT,
        tokenizer_readiness_json=TOKENIZER_READINESS,
        family_policy_json=FAMILY_POLICY,
        eval_prompt_pack_json=PROMPT_PACK,
        model_id=MODEL_ID,
        revision=REVISION,
        output_dir=output_dir,
    )


def test_real_teacher_source_metadata_is_authenticated_but_payload_absent(
    tmp_path: Path,
) -> None:
    _require_resident_authority()
    output_dir = tmp_path / "metadata"

    payload = _prepare(output_dir)

    assert payload["record_type"] == "glm52_family_eval_teacher_metadata_probe"
    assert payload["metadata_status"] == "glm52_teacher_source_metadata_ready"
    assert payload["metadata_ready"] is True
    assert payload["metadata_row_count"] == 66
    assert payload["metadata_counts_by_split"] == {
        "report": 22,
        "selection": 22,
        "holdout": 22,
    }
    assert payload["metadata_counts_by_domain"] == {
        "route": 24,
        "math": 21,
        "instruction": 21,
    }
    assert payload["source_index_revalidated"] is True
    assert payload["source_payload_headers_revalidated"] is True
    assert payload["source_payload_group_count"] == 225
    assert payload["source_payload_shard_count"] == 61
    assert payload["source_weight_encoding"] == "modelopt_nvfp4"
    assert payload["source_decoder"] == "modelopt_nvfp4_v1"
    assert payload["teacher_source_kind"] == (
        "deterministically_dequantized_pinned_modelopt_nvfp4"
    )
    assert payload["activation_quantization_emulated"] is False
    assert payload["exact_w4a4_runtime_parity_claimed"] is False
    assert payload["bf16_teacher_claimed"] is False
    assert payload["teacher_cache_payload_present"] is False
    assert payload["teacher_cache_ready"] is False
    assert payload["teacher_logits_generated"] is False
    assert payload["tensor_payloads_read"] is False
    assert payload["full_model_constructed"] is False
    assert payload["candidate_artifact_used"] is False
    assert payload["intended_logit_scope"] == "full_vocabulary"
    assert payload["intended_kld_mode"] == "exact_full_logits"
    assert payload["missing_requirements"] == [
        "dequantized_source_teacher_cache_payload"
    ]

    metadata_path = output_dir / "glm52_teacher_source_metadata.jsonl"
    rows = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    assert len(rows) == 66
    assert {row["prompt_id"] for row in rows} == {
        row["prompt_id"]
        for row in json.loads(PROMPT_PACK.read_text())["prompt_rows"]
    }
    assert all(row["teacher_cache_payload_present"] is False for row in rows)
    assert all(row["full_logits_available"] is False for row in rows)
    assert all(row["tuning_eligible"] is (row["split"] == "selection") for row in rows)
    assert all(row["holdout_tuning_forbidden"] is True for row in rows)


def test_teacher_metadata_rejects_prompt_mutation_even_with_rehashed_final_digest(
    tmp_path: Path,
) -> None:
    _require_resident_authority()
    prompt_pack = json.loads(PROMPT_PACK.read_text())
    prompt_pack["prompt_rows"][0]["encoded_token_ids"][0] += 1
    prompt_pack.pop("prompt_pack_contract_sha256")
    from keep.quality.glm52_family import canonical_sha256

    prompt_pack["prompt_pack_contract_sha256"] = canonical_sha256(prompt_pack)
    mutated = tmp_path / "mutated-prompts.json"
    mutated.write_text(json.dumps(prompt_pack, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="prompt content digest"):
        prepare_glm52_family_eval_teacher_metadata(
            source_dir=SNAPSHOT,
            profile_path=PROFILE,
            config_path=CONFIG,
            index_path=INDEX,
            source_audit_json=SOURCE_AUDIT,
            source_payload_audit_json=SOURCE_PAYLOAD_AUDIT,
            tokenizer_readiness_json=TOKENIZER_READINESS,
            family_policy_json=FAMILY_POLICY,
            eval_prompt_pack_json=mutated,
            model_id=MODEL_ID,
            revision=REVISION,
            output_dir=tmp_path / "metadata",
        )


def test_teacher_metadata_rejects_tampered_source_payload_evidence(
    tmp_path: Path,
) -> None:
    _require_resident_authority()
    payload_audit = json.loads(SOURCE_PAYLOAD_AUDIT.read_text())
    payload_audit["selected_groups"] -= 1
    tampered = tmp_path / "tampered-source-payload-audit.json"
    tampered.write_text(json.dumps(payload_audit, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="payload audit evidence"):
        prepare_glm52_family_eval_teacher_metadata(
            source_dir=SNAPSHOT,
            profile_path=PROFILE,
            config_path=CONFIG,
            index_path=INDEX,
            source_audit_json=SOURCE_AUDIT,
            source_payload_audit_json=tampered,
            tokenizer_readiness_json=TOKENIZER_READINESS,
            family_policy_json=FAMILY_POLICY,
            eval_prompt_pack_json=PROMPT_PACK,
            model_id=MODEL_ID,
            revision=REVISION,
            output_dir=tmp_path / "metadata",
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("tokenizer_readiness_sha256", "0" * 64),
        ("family_policy_sha256", "1" * 64),
        ("tokenizer_dir", "/tmp/not-the-pinned-tokenizer"),
        ("tokenizer_readiness_json", "/tmp/not-the-readiness-evidence.json"),
        ("family_policy_json", "/tmp/not-the-policy-evidence.json"),
    ),
)
def test_teacher_metadata_rejects_rehashed_prompt_provenance_mutation(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    _require_resident_authority()
    prompt_pack = json.loads(PROMPT_PACK.read_text())
    prompt_pack[field] = replacement
    prompt_pack.pop("prompt_pack_contract_sha256")
    from keep.quality.glm52_family import canonical_sha256

    prompt_pack["prompt_pack_contract_sha256"] = canonical_sha256(prompt_pack)
    mutated = tmp_path / f"mutated-{field}.json"
    mutated.write_text(json.dumps(prompt_pack, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=f"prompt pack provenance {field}"):
        prepare_glm52_family_eval_teacher_metadata(
            source_dir=SNAPSHOT,
            profile_path=PROFILE,
            config_path=CONFIG,
            index_path=INDEX,
            source_audit_json=SOURCE_AUDIT,
            source_payload_audit_json=SOURCE_PAYLOAD_AUDIT,
            tokenizer_readiness_json=TOKENIZER_READINESS,
            family_policy_json=FAMILY_POLICY,
            eval_prompt_pack_json=mutated,
            model_id=MODEL_ID,
            revision=REVISION,
            output_dir=tmp_path / "metadata",
        )


def test_teacher_metadata_cli_writes_success_and_rejects_output_alias(
    tmp_path: Path,
) -> None:
    _require_resident_authority()
    output_dir = tmp_path / "metadata"
    output_json = tmp_path / "evidence.json"
    args = [
        "--source-dir",
        str(SNAPSHOT),
        "--profile-path",
        str(PROFILE),
        "--config-path",
        str(CONFIG),
        "--index-path",
        str(INDEX),
        "--source-audit-json",
        str(SOURCE_AUDIT),
        "--source-payload-audit-json",
        str(SOURCE_PAYLOAD_AUDIT),
        "--tokenizer-readiness-json",
        str(TOKENIZER_READINESS),
        "--family-policy-json",
        str(FAMILY_POLICY),
        "--eval-prompt-pack-json",
        str(PROMPT_PACK),
        "--model-id",
        MODEL_ID,
        "--revision",
        REVISION,
        "--output-dir",
        str(output_dir),
        "--output-json",
        str(output_json),
    ]
    assert main(args) == 0
    assert json.loads(output_json.read_text())["metadata_ready"] is True

    protected = FAMILY_POLICY.read_bytes()
    alias_args = list(args)
    alias_args[alias_args.index("--output-json") + 1] = str(FAMILY_POLICY)
    assert main(alias_args) == 1
    assert FAMILY_POLICY.read_bytes() == protected

    collision_dir = tmp_path / "metadata-collision"
    collision_args = list(args)
    collision_args[collision_args.index("--output-dir") + 1] = str(collision_dir)
    collision_args[collision_args.index("--output-json") + 1] = str(
        collision_dir / "glm52_teacher_source_metadata.jsonl"
    )
    assert main(collision_args) == 1
    assert not collision_dir.exists()

    mutated_prompt = json.loads(PROMPT_PACK.read_text())
    mutated_prompt["family_policy_sha256"] = "2" * 64
    mutated_prompt.pop("prompt_pack_contract_sha256")
    from keep.quality.glm52_family import canonical_sha256

    mutated_prompt["prompt_pack_contract_sha256"] = canonical_sha256(
        mutated_prompt
    )
    mutated_path = tmp_path / "mutated-prompt-provenance.json"
    mutated_path.write_text(
        json.dumps(mutated_prompt, indent=2, sort_keys=True) + "\n"
    )
    provenance_dir = tmp_path / "metadata-provenance"
    provenance_json = tmp_path / "provenance-evidence.json"
    provenance_args = list(args)
    provenance_args[
        provenance_args.index("--eval-prompt-pack-json") + 1
    ] = str(mutated_path)
    provenance_args[provenance_args.index("--output-dir") + 1] = str(
        provenance_dir
    )
    provenance_args[provenance_args.index("--output-json") + 1] = str(
        provenance_json
    )
    assert main(provenance_args) == 1
    assert json.loads(provenance_json.read_text())["metadata_ready"] is False
    assert not provenance_dir.exists()
