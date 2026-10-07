from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from keep.quality import glm52_family

from benchmarks.probe_glm52_tokenizer_readiness import (
    GLM52TokenizerIdentityContract,
    PinnedFileIdentity,
    PINNED_GLM52_TOKENIZER_CONTRACT,
    main,
    probe_glm52_tokenizer_readiness,
)


_CHAT_TEMPLATE = """\
{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}{% endfor %}
{% if add_generation_prompt %}<|assistant|>{% if enable_thinking is not defined or enable_thinking %}<think>{% else %}<think></think>{% endif %}{% endif %}
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_contract(root: Path) -> GLM52TokenizerIdentityContract:
    identities = tuple(
        PinnedFileIdentity(
            filename=filename,
            size_bytes=(root / filename).stat().st_size,
            sha256=_sha256(root / filename),
        )
        for filename in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
        )
    )
    return GLM52TokenizerIdentityContract(
        model_id="fixture/glm52",
        revision="0123456789abcdef0123456789abcdef01234567",
        model_type="glm_moe_dsa",
        model_vocab_size=8,
        tokenizer_base_vocab_size=4,
        tokenizer_length=6,
        pad_token_id=3,
        eos_token_ids=(3, 4, 5),
        files=identities,
    )


def _write_fixture(
    root: Path,
    *,
    chat_template: str = _CHAT_TEMPLATE,
    eos_token: str = "<eos>",
    hello_token_id: int = 1,
) -> GLM52TokenizerIdentityContract:
    backend = Tokenizer(
        WordLevel(
            {"<unk>": 0, "hello": hello_token_id, "world": 2, "<eos>": 3},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token=eos_token,
        pad_token="<eos>",
        additional_special_tokens=["<stop1>", "<stop2>"],
        chat_template=chat_template,
    )
    tokenizer.save_pretrained(root)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "vocab_size": 8,
                "pad_token_id": 3,
                "eos_token_id": [3, 4, 5],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps({"pad_token_id": 3, "eos_token_id": [3, 4, 5]}) + "\n",
        encoding="utf-8",
    )
    return _fixture_contract(root)


def test_pinned_glm52_contract_records_exact_production_identities() -> None:
    contract = PINNED_GLM52_TOKENIZER_CONTRACT
    identities = {identity.filename: identity for identity in contract.files}

    assert contract.model_id == "0xSero/glm-5.2-reap-504B-v2"
    assert contract.revision == "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
    assert contract.model_vocab_size == 154880
    assert contract.tokenizer_base_vocab_size == 154820
    assert contract.tokenizer_length == 154856
    assert contract.pad_token_id == 154820
    assert contract.eos_token_ids == (154820, 154827, 154829)
    assert contract.mlx_lm_detokenizer_class == "BPEStreamingDetokenizer"
    assert contract.mlx_lm_tool_parser_module == "mlx_lm.tool_parsers.glm47"
    assert contract.mlx_lm_has_chat_template is True
    assert contract.mlx_lm_has_thinking is True
    assert identities["tokenizer.json"].size_bytes == 20_217_442
    assert (
        identities["tokenizer.json"].sha256
        == "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
    )
    assert identities["tokenizer_config.json"].sha256 == (
        "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"
    )
    assert identities["chat_template.jinja"].sha256 == (
        "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679"
    )
    assert identities["generation_config.json"].sha256 == (
        "ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55"
    )


def test_probe_proves_local_tokenizer_chat_and_full_eos_readiness(tmp_path: Path) -> None:
    contract = _write_fixture(tmp_path)

    payload = probe_glm52_tokenizer_readiness(
        tokenizer_dir=tmp_path,
        config_path=tmp_path / "config.json",
        model_id=contract.model_id,
        revision=contract.revision,
        prompt="hello world",
        identity_contract=contract,
    )

    assert payload["record_type"] == "glm52_tokenizer_readiness_probe"
    assert payload["probe_status"] == "glm52_tokenizer_readiness_probe_ready"
    assert payload["probe_pass"] is True
    assert payload["readiness_scope"] == "tokenizer_payload"
    assert payload["tokenizer_payload_ready"] is True
    assert payload["production_ready"] is False
    assert payload["production_identity_contract"] is False
    assert payload["source_authority"] == "test_fixture_contract"
    assert payload["local_files_only"] is True
    assert payload["trust_remote_code"] is False
    assert payload["remote_code_required"] is False
    assert payload["immutable_file_identity_pass"] is True
    assert all(record["identity_pass"] for record in payload["file_identities"].values())
    assert payload["tokenizer_load_pass"] is True
    assert payload["mlx_lm_loader_pass"] is True
    assert payload["mlx_lm_detokenizer_class"] == "NaiveStreamingDetokenizer"
    assert payload["mlx_lm_tool_parser_module"] is None
    assert payload["mlx_lm_has_chat_template"] is True
    assert payload["mlx_lm_has_thinking"] is False
    assert payload["tokenizer_base_vocab_size"] == 4
    assert payload["tokenizer_length"] == 6
    assert payload["model_vocab_size"] == 8
    assert payload["tokenizer_fits_model_vocab"] is True
    assert payload["pad_token_id"] == 3
    assert payload["model_eos_token_ids"] == [3, 4, 5]
    assert payload["mlx_lm_wrapper_eos_token_ids"] == [3, 4, 5]
    assert payload["mlx_lm_wrapper_eos_pass"] is True
    assert payload["prompt_token_count"] == 2
    assert payload["prompt_decoded_text"] == "hello world"
    assert payload["prompt_roundtrip_pass"] is True
    assert payload["thinking_enabled_pass"] is True
    assert payload["thinking_disabled_pass"] is True
    assert payload["assistant_generation_marker"] == "<|assistant|>"
    assert payload["assistant_generation_marker_pass"] is True
    assert payload["whole_model_runtime_proven"] is False
    assert payload["full_model_bind_proven"] is False
    assert payload["generation_proven"] is False
    assert payload["missing_requirements"] == []


@pytest.mark.parametrize(
    "relative_path",
    (
        "added_tokens.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "chat_templates/evil.jinja",
    ),
)
def test_probe_rejects_unpinned_tokenizer_consumed_inputs(
    tmp_path: Path,
    relative_path: str,
) -> None:
    contract = _write_fixture(tmp_path)
    unexpected = tmp_path / relative_path
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_text("{}\n", encoding="utf-8")

    payload = probe_glm52_tokenizer_readiness(
        tokenizer_dir=tmp_path,
        config_path=tmp_path / "config.json",
        model_id=contract.model_id,
        revision=contract.revision,
        prompt="hello world",
        identity_contract=contract,
    )

    assert payload["probe_pass"] is False
    assert payload["tokenizer_load_pass"] is False
    assert payload["failure_stage"] == "immutable_file_identity"
    assert payload["missing_requirements"] == [
        f"unexpected_tokenizer_input:{relative_path}"
    ]


def test_readiness_validator_rejects_unpinned_tokenizer_consumed_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _write_fixture(tmp_path)
    pinned = {
        identity.filename: {
            "size_bytes": identity.size_bytes,
            "sha256": identity.sha256,
        }
        for identity in contract.files
    }
    monkeypatch.setattr(glm52_family, "GLM52_PINNED_TOKENIZER_FILES", pinned)
    identities = {
        filename: {
            "path": str(tmp_path / filename),
            "size_bytes": identity["size_bytes"],
            "expected_size_bytes": identity["size_bytes"],
            "sha256": identity["sha256"],
            "expected_sha256": identity["sha256"],
            "identity_pass": True,
        }
        for filename, identity in pinned.items()
    }
    payload = {
        "record_type": "glm52_tokenizer_readiness_probe",
        "probe_status": "glm52_tokenizer_readiness_probe_ready",
        "probe_pass": True,
        "tokenizer_payload_ready": True,
        "production_ready": True,
        "model_id": glm52_family.PINNED_GLM52_MODEL_ID,
        "source_revision": glm52_family.PINNED_GLM52_REVISION,
        "source_authority": "pinned_huggingface_snapshot",
        "production_identity_contract": True,
        "immutable_file_identity_pass": True,
        "config_lineage_pass": True,
        "tokenizer_load_pass": True,
        "mlx_lm_loader_pass": True,
        "mlx_lm_wrapper_eos_pass": True,
        "prompt_token_ids_within_model_vocab": True,
        "prompt_roundtrip_pass": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "remote_code_required": False,
        "tokenizer_base_vocab_size": glm52_family.GLM52_TOKENIZER_BASE_VOCAB_SIZE,
        "tokenizer_length": glm52_family.GLM52_TOKENIZER_LENGTH,
        "model_vocab_size": glm52_family.GLM52_MODEL_VOCAB_SIZE,
        "tokenizer_fits_model_vocab": True,
        "pad_token_id": glm52_family.GLM52_EOS_TOKEN_IDS[0],
        "tokenizer_eos_token_id": glm52_family.GLM52_EOS_TOKEN_IDS[0],
        "model_eos_token_ids": list(glm52_family.GLM52_EOS_TOKEN_IDS),
        "mlx_lm_wrapper_eos_token_ids": list(glm52_family.GLM52_EOS_TOKEN_IDS),
        "thinking_disabled_pass": True,
        "assistant_generation_marker_pass": True,
        "missing_requirements": [],
        "tokenizer_dir": str(tmp_path),
        "file_identities": identities,
    }
    (tmp_path / "added_tokens.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected tokenizer input"):
        glm52_family.validate_glm52_tokenizer_readiness(
            payload,
            tokenizer_dir=tmp_path,
        )


def test_probe_fails_loud_on_model_or_revision_lineage_mismatch(tmp_path: Path) -> None:
    contract = _write_fixture(tmp_path)

    payload = probe_glm52_tokenizer_readiness(
        tokenizer_dir=tmp_path,
        config_path=tmp_path / "config.json",
        model_id="wrong/model",
        revision="wrong-revision",
        prompt="hello world",
        identity_contract=contract,
    )

    assert payload["probe_pass"] is False
    assert payload["production_ready"] is False
    assert payload["failure_stage"] == "source_lineage"
    assert payload["missing_requirements"] == ["pinned_model_id", "pinned_revision"]
    assert payload["tokenizer_load_pass"] is False


def test_main_writes_durable_failure_json_and_returns_nonzero_on_identity_drift(
    tmp_path: Path,
) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    contract = _write_fixture(tokenizer_dir)
    (tokenizer_dir / "tokenizer_config.json").write_text(
        (tokenizer_dir / "tokenizer_config.json").read_text() + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "failed-probe.json"

    exit_code = main(
        [
            "--tokenizer-dir",
            str(tokenizer_dir),
            "--config-path",
            str(tokenizer_dir / "config.json"),
            "--model-id",
            contract.model_id,
            "--revision",
            contract.revision,
            "--prompt",
            "hello world",
            "--output-json",
            str(output_path),
        ],
        identity_contract=contract,
    )

    assert exit_code == 1
    failure = json.loads(output_path.read_text())
    assert failure["probe_status"] == "glm52_tokenizer_readiness_probe_failed"
    assert failure["probe_pass"] is False
    assert failure["production_ready"] is False
    assert failure["failure_stage"] == "immutable_file_identity"
    assert failure["file_identities"]["tokenizer_config.json"]["identity_pass"] is False
    assert failure["missing_requirements"] == [
        "immutable_file_identity:tokenizer_config.json"
    ]


def test_probe_rejects_chat_template_without_thinking_modes_or_generation_marker(
    tmp_path: Path,
) -> None:
    contract = _write_fixture(
        tmp_path,
        chat_template=(
            "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        ),
    )

    payload = probe_glm52_tokenizer_readiness(
        tokenizer_dir=tmp_path,
        config_path=tmp_path / "config.json",
        model_id=contract.model_id,
        revision=contract.revision,
        prompt="hello world",
        identity_contract=contract,
    )

    assert payload["probe_pass"] is False
    assert payload["failure_stage"] == "tokenizer_capabilities"
    assert payload["thinking_enabled_pass"] is False
    assert payload["thinking_disabled_pass"] is False
    assert payload["assistant_generation_marker_pass"] is False
    assert payload["missing_requirements"] == [
        "chat_template_thinking_enabled",
        "chat_template_thinking_disabled",
        "assistant_generation_marker",
    ]


def test_probe_rejects_tokenizer_eos_that_does_not_match_first_model_eos(
    tmp_path: Path,
) -> None:
    contract = _write_fixture(tmp_path, eos_token="<stop1>")

    payload = probe_glm52_tokenizer_readiness(
        tokenizer_dir=tmp_path,
        config_path=tmp_path / "config.json",
        model_id=contract.model_id,
        revision=contract.revision,
        prompt="hello world",
        identity_contract=contract,
    )

    assert payload["probe_pass"] is False
    assert payload["tokenizer_eos_token_id"] == 4
    assert payload["missing_requirements"] == ["tokenizer_eos_token_id"]


def test_probe_rejects_encoded_token_id_outside_model_vocabulary(tmp_path: Path) -> None:
    contract = _write_fixture(tmp_path, hello_token_id=9)

    payload = probe_glm52_tokenizer_readiness(
        tokenizer_dir=tmp_path,
        config_path=tmp_path / "config.json",
        model_id=contract.model_id,
        revision=contract.revision,
        prompt="hello world",
        identity_contract=contract,
    )

    assert payload["prompt_encoded_token_ids"] == [9, 2]
    assert payload["probe_pass"] is False
    assert payload["missing_requirements"] == [
        "prompt_token_ids_within_model_vocab"
    ]
