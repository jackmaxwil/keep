from __future__ import annotations

import importlib.util
import inspect
import hashlib
import json
import os
from pathlib import Path
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src/keep/quality/glm52_candidate_eval.py"
SPEC = importlib.util.spec_from_file_location("glm52_candidate_eval_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
candidate_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = candidate_module
SPEC.loader.exec_module(candidate_module)

GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256 = (
    candidate_module.GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
)
build_glm52_candidate_cache_contract = (
    candidate_module.build_glm52_candidate_cache_contract
)
compare_glm52_candidate_caches = candidate_module.compare_glm52_candidate_caches
load_glm52_prompt_pack_accounting = candidate_module.load_glm52_prompt_pack_accounting
prepare_glm52_candidate_predictor_batch = (
    candidate_module.prepare_glm52_candidate_predictor_batch
)
summarize_rows = candidate_module._summarize_rows
GLM52ProducerPhase = candidate_module.GLM52ProducerPhase
GLM52TeacherCacheContract = candidate_module.GLM52TeacherCacheContract
GLM52TeacherCachePrompt = candidate_module.GLM52TeacherCachePrompt
build_glm52_teacher_cache_manifest = candidate_module.build_glm52_teacher_cache_manifest
publish_glm52_teacher_cache_manifest = candidate_module.publish_glm52_teacher_cache_manifest
write_glm52_teacher_cache_shard = candidate_module.write_glm52_teacher_cache_shard
audit_glm52_teacher_cache = candidate_module.audit_glm52_teacher_cache


POLICY_PATH = REPO_ROOT / "artifacts/quality/glm52-family-policy-20260709-v2.json"
PROMPT_PACK_PATH = (
    REPO_ROOT / "artifacts/quality/glm52-family-eval-prompts-20260709-v2.json"
)


def _prompts() -> tuple[GLM52TeacherCachePrompt, ...]:
    return (
        GLM52TeacherCachePrompt(
            prompt_id="tiny_report_route_000",
            split="report",
            domain="route",
            tuning_eligible=False,
            encoded_token_ids=(1, 0, 0),
        ),
        GLM52TeacherCachePrompt(
            prompt_id="tiny_selection_math_000",
            split="selection",
            domain="math",
            tuning_eligible=True,
            encoded_token_ids=(1, 0, 0),
        ),
        GLM52TeacherCachePrompt(
            prompt_id="tiny_holdout_instruction_000",
            split="holdout",
            domain="instruction",
            tuning_eligible=False,
            encoded_token_ids=(1, 0, 0),
        ),
    )


@pytest.fixture
def contracts() -> tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract]:
    teacher = GLM52TeacherCacheContract.for_testing(
        prompts=_prompts(),
        vocab_size=8,
    )
    candidate = build_glm52_candidate_cache_contract(teacher)
    return teacher, candidate


def _phase(
    identity: str,
    *,
    pageouts_delta: int = 0,
    swapouts_delta: int = 0,
) -> GLM52ProducerPhase:
    return GLM52ProducerPhase.create(
        phase_id="phase-000",
        ordinal=0,
        start_boundary="before-forward",
        end_boundary="after-shards",
        contribution_range="synthetic-full-vocabulary-logits",
        pageouts_delta=pageouts_delta,
        swapouts_delta=swapouts_delta,
        system_wired_default=True,
        input_identity_sha256=identity,
        output_identity_sha256="d" * 64,
    )


def _teacher_logits(prompt: GLM52TeacherCachePrompt) -> np.ndarray:
    row = np.array([3.0, 2.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
    return np.ascontiguousarray(np.tile(row, (prompt.token_count - 1, 1)))


def _publish_cache(
    root: Path,
    contract: GLM52TeacherCacheContract,
    logits_for_prompt,
    *,
    pageouts_delta: int = 0,
) -> None:
    ledger = root.parent / f"{root.name}.ledger.jsonl"
    phase = _phase(
        contract.bound_identity_sha256,
        pageouts_delta=pageouts_delta,
    )
    shards = []
    for prompt in contract.prompts:
        shards.append(
            write_glm52_teacher_cache_shard(
                root,
                ledger_path=ledger,
                prompt=prompt,
                logits=logits_for_prompt(prompt),
                vocab_size=contract.vocab_size,
                producer_phase_ids=(phase.phase_id,),
                bound_identity_sha256=contract.bound_identity_sha256,
            ).shard
        )
    manifest = build_glm52_teacher_cache_manifest(
        contract=contract,
        shards=shards,
        producer_phases=(phase,),
        created_at="2026-07-10T00:00:00Z",
    )
    publish_glm52_teacher_cache_manifest(
        root,
        manifest,
        ledger_path=ledger,
        bound_identity_sha256=contract.bound_identity_sha256,
    )


def _publish_diagnostic_candidate_manifest(
    candidate_root: Path,
    *,
    teacher_identity_sha256: str,
) -> Path:
    canonical = json.loads(
        (candidate_root / candidate_module.cache_api.MANIFEST_FILENAME).read_text()
    )
    canonical.update(
        {
            "evidence_class": "diagnostic_only",
            "release_eligible": False,
            "reason": (
                "teacher cache release_eligible=false waived for diagnostic-only "
                f"evaluation; teacher_cache_identity_sha256={teacher_identity_sha256}"
            ),
            "teacher_cache_identity_sha256": teacher_identity_sha256,
        }
    )
    path = candidate_root.parent / f"{candidate_root.name}-diagnostic-manifest.json"
    path.write_text(json.dumps(canonical, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def test_dirty_teacher_cache_requires_explicit_diagnostic_opt_in(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, candidate_contract = contracts
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "candidate"
    _publish_cache(
        teacher_root,
        teacher_contract,
        _teacher_logits,
        pageouts_delta=1,
    )
    _publish_cache(candidate_root, candidate_contract, _teacher_logits)

    with pytest.raises(
        ValueError,
        match="teacher cache passed payload audit but is not release eligible",
    ):
        compare_glm52_candidate_caches(
            teacher_root,
            candidate_root,
            policy_path=POLICY_PATH,
            teacher_contract=teacher_contract,
            candidate_contract=candidate_contract,
        )


def test_dirty_teacher_cache_produces_diagnostic_only_comparison_when_allowed(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, candidate_contract = contracts
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "candidate"
    _publish_cache(
        teacher_root,
        teacher_contract,
        _teacher_logits,
        pageouts_delta=1,
    )
    _publish_cache(
        candidate_root,
        candidate_contract,
        _teacher_logits,
        pageouts_delta=1,
    )
    teacher_audit = audit_glm52_teacher_cache(
        teacher_root,
        contract=teacher_contract,
    )
    teacher_identity = candidate_module._cache_identity_envelope(
        record_type=candidate_module.TEACHER_CACHE_IDENTITY_RECORD_TYPE,
        contract=teacher_contract,
        audit=teacher_audit,
        manifest_path=teacher_root / candidate_module.cache_api.MANIFEST_FILENAME,
    )["identity_sha256"]
    diagnostic_path = _publish_diagnostic_candidate_manifest(
        candidate_root,
        teacher_identity_sha256=teacher_identity,
    )

    report = compare_glm52_candidate_caches(
        teacher_root,
        candidate_root,
        policy_path=POLICY_PATH,
        teacher_contract=teacher_contract,
        candidate_contract=candidate_contract,
        allow_non_release_teacher_cache=True,
    )

    assert report["evidence_class"] == "diagnostic_only"
    assert report["release_eligible"] is False
    assert "teacher cache release_eligible=false" in report["reason"]
    assert "candidate cache release_eligible=false" in report["reason"]
    assert report["waived_reasons"] == [
        "teacher cache release_eligible=false",
        "candidate cache release_eligible=false",
    ]
    assert report["teacher_cache_identity"]["identity_sha256"] in report["reason"]
    assert report["family_eval_gate_pass"] is False
    assert all(report["checks"].values())
    assert report["metrics"] == {
        "mean_kld": pytest.approx(0.0, abs=1e-15),
        "p999_kld": pytest.approx(0.0, abs=1e-15),
        "top1_agreement": 1.0,
        "mean_ppl_ratio": pytest.approx(1.0, abs=1e-15),
    }
    assert report["diagnostic_candidate_manifest_path"] == str(diagnostic_path)


def test_diagnostic_candidate_requires_explicit_opt_in(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, candidate_contract = contracts
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "candidate"
    _publish_cache(teacher_root, teacher_contract, _teacher_logits)
    _publish_cache(
        candidate_root,
        candidate_contract,
        _teacher_logits,
        pageouts_delta=1,
    )

    with pytest.raises(
        ValueError,
        match="candidate cache passed payload audit but is not release eligible",
    ):
        compare_glm52_candidate_caches(
            teacher_root,
            candidate_root,
            policy_path=POLICY_PATH,
            teacher_contract=teacher_contract,
            candidate_contract=candidate_contract,
        )


def test_diagnostic_compare_requires_sibling_manifest(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, candidate_contract = contracts
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "candidate"
    _publish_cache(
        teacher_root,
        teacher_contract,
        _teacher_logits,
        pageouts_delta=1,
    )
    _publish_cache(candidate_root, candidate_contract, _teacher_logits)

    with pytest.raises(ValueError, match="diagnostic candidate manifest"):
        compare_glm52_candidate_caches(
            teacher_root,
            candidate_root,
            policy_path=POLICY_PATH,
            teacher_contract=teacher_contract,
            candidate_contract=candidate_contract,
            allow_non_release_teacher_cache=True,
        )


def test_diagnostic_produce_migrates_legacy_manifest_and_keeps_root_auditable(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_contract, candidate_contract = contracts
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "candidate-cache"
    _publish_cache(
        teacher_root,
        teacher_contract,
        _teacher_logits,
        pageouts_delta=1,
    )
    _publish_cache(candidate_root, candidate_contract, _teacher_logits)
    canonical_manifest = json.loads(
        (candidate_root / candidate_module.cache_api.MANIFEST_FILENAME).read_text()
    )
    teacher_audit = audit_glm52_teacher_cache(
        teacher_root,
        contract=teacher_contract,
    )
    teacher_identity = candidate_module._cache_identity_envelope(
        record_type=candidate_module.TEACHER_CACHE_IDENTITY_RECORD_TYPE,
        contract=teacher_contract,
        audit=teacher_audit,
        manifest_path=teacher_root / candidate_module.cache_api.MANIFEST_FILENAME,
    )["identity_sha256"]
    legacy_path = candidate_root / "glm52-candidate-diagnostic-manifest.json"
    _publish_diagnostic_candidate_manifest(
        candidate_root,
        teacher_identity_sha256=teacher_identity,
    ).replace(legacy_path)
    producer_api = candidate_module._load_teacher_producer_api()

    def fake_produce(**_kwargs):
        audit = audit_glm52_teacher_cache(candidate_root, contract=candidate_contract)
        assert audit.valid is True
        manifest = SimpleNamespace(to_dict=lambda: canonical_manifest)
        return producer_api.ProducerResult(
            completed=True,
            manifest=manifest,
            manifest_path=candidate_root / candidate_module.cache_api.MANIFEST_FILENAME,
            produced_prompt_ids=(),
            resumed_prompt_ids=tuple(prompt.prompt_id for prompt in candidate_contract.prompts),
            shards=(),
        )

    monkeypatch.setattr(
        candidate_module,
        "_load_teacher_producer_api",
        lambda: SimpleNamespace(produce_glm52_teacher_cache=fake_produce),
    )
    monkeypatch.setattr(
        candidate_module,
        "_contract_from_cache_manifest",
        lambda *_args, **_kwargs: teacher_contract,
    )

    result = candidate_module.produce_glm52_candidate_cache(
        contract=candidate_contract,
        profile_path="unused",
        config_path="unused",
        source_index_path="unused",
        tokenizer_dir="unused",
        tokenizer_readiness_json="unused",
        family_policy_json="unused",
        non_vq_artifact_dir="unused",
        non_vq_evidence_json="unused",
        routed_artifact_dir="unused",
        composite_audit_json="unused",
        materialization_runs_jsonl="unused",
        full_bind_preflight_json="unused",
        teacher_cache_root=teacher_root,
        prompt_pack_path="unused",
        cache_root=candidate_root,
        ledger_path=tmp_path / "candidate.ledger.jsonl",
        allow_non_release_teacher_cache=True,
    )

    sibling_path = tmp_path / "candidate-cache-diagnostic-manifest.json"
    assert not legacy_path.exists()
    assert result.manifest_path == sibling_path
    assert sibling_path.is_file()
    assert audit_glm52_teacher_cache(
        candidate_root,
        contract=candidate_contract,
    ).valid is True


def test_diagnostic_opt_in_does_not_waive_strict_payload_audit(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, candidate_contract = contracts
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "candidate"
    _publish_cache(
        teacher_root,
        teacher_contract,
        _teacher_logits,
        pageouts_delta=1,
    )
    _publish_cache(candidate_root, candidate_contract, _teacher_logits)
    first_shard = teacher_root / "teacher_logits" / "tiny_report_route_000.safetensors"
    first_shard.write_bytes(first_shard.read_bytes()[:-1])

    with pytest.raises(ValueError, match="teacher cache failed strict audit"):
        compare_glm52_candidate_caches(
            teacher_root,
            candidate_root,
            policy_path=POLICY_PATH,
            teacher_contract=teacher_contract,
            candidate_contract=candidate_contract,
            allow_non_release_teacher_cache=True,
        )


def test_identical_full_vocab_caches_pass_every_frozen_gate(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, candidate_contract = contracts
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "candidate"
    _publish_cache(teacher_root, teacher_contract, _teacher_logits)
    _publish_cache(candidate_root, candidate_contract, _teacher_logits)

    report = compare_glm52_candidate_caches(
        teacher_root,
        candidate_root,
        policy_path=POLICY_PATH,
        teacher_contract=teacher_contract,
        candidate_contract=candidate_contract,
    )

    assert report["metrics"]["mean_kld"] == pytest.approx(0.0, abs=1e-15)
    assert report["metrics"]["p999_kld"] == pytest.approx(0.0, abs=1e-15)
    assert report["metrics"]["top1_agreement"] == 1.0
    assert report["metrics"]["mean_ppl_ratio"] == pytest.approx(1.0, abs=1e-15)
    assert report["domain_top1_agreement"] == {
        "instruction": 1.0,
        "math": 1.0,
        "route": 1.0,
    }
    assert report["checks"] == {
        "domain_top1": True,
        "mean_kld": True,
        "mean_ppl_ratio": True,
        "p999_kld": True,
        "top1": True,
    }
    assert report["family_eval_gate_pass"] is True
    assert report["holdout_discipline"]["selection_is_only_tuning_eligible_split"] is True
    assert report["holdout_discipline"]["holdout_may_guide_recovery"] is False
    assert (
        report["teacher_cache_identity"]["record_type"]
        == "glm52_source_teacher_cache_identity_v1"
    )
    assert (
        report["candidate_cache_identity"]["record_type"]
        == "glm52_production_composite_candidate_cache_identity_v1"
    )
    assert (
        report["candidate_cache_identity"]["teacher_cache_identity_sha256"]
        == report["teacher_cache_identity"]["identity_sha256"]
    )
    assert all(
        row["teacher_cache_identity_sha256"]
        == report["teacher_cache_identity"]["identity_sha256"]
        for row in report["rows"]
    )


def test_same_candidate_root_on_both_sides_is_rejected(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    _teacher_contract, candidate_contract = contracts
    candidate_root = tmp_path / "candidate"
    _publish_cache(candidate_root, candidate_contract, _teacher_logits)

    with pytest.raises(ValueError, match="distinct cache roots"):
        compare_glm52_candidate_caches(
            candidate_root,
            candidate_root,
            policy_path=POLICY_PATH,
            teacher_contract=candidate_contract,
            candidate_contract=candidate_contract,
        )


def test_candidate_kind_manifest_on_teacher_side_is_rejected(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    _teacher_contract, candidate_contract = contracts
    alleged_teacher_root = tmp_path / "alleged-teacher"
    candidate_root = tmp_path / "candidate"
    _publish_cache(alleged_teacher_root, candidate_contract, _teacher_logits)
    _publish_cache(candidate_root, candidate_contract, _teacher_logits)

    with pytest.raises(ValueError, match="source-teacher producer"):
        compare_glm52_candidate_caches(
            alleged_teacher_root,
            candidate_root,
            policy_path=POLICY_PATH,
            teacher_contract=candidate_contract,
            candidate_contract=candidate_contract,
        )


def test_shared_teacher_candidate_shard_file_identity_is_rejected(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, candidate_contract = contracts
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "candidate"
    _publish_cache(teacher_root, teacher_contract, _teacher_logits)
    _publish_cache(candidate_root, candidate_contract, _teacher_logits)
    prompt_id = teacher_contract.prompts[0].prompt_id
    teacher_shard = teacher_root / "teacher_logits" / f"{prompt_id}.safetensors"
    candidate_shard = candidate_root / "teacher_logits" / f"{prompt_id}.safetensors"
    candidate_shard.unlink()
    os.link(teacher_shard, candidate_shard)

    with pytest.raises(ValueError, match="distinct file identities"):
        compare_glm52_candidate_caches(
            teacher_root,
            candidate_root,
            policy_path=POLICY_PATH,
            teacher_contract=teacher_contract,
            candidate_contract=candidate_contract,
        )


def test_modified_policy_body_with_copied_digest_is_rejected(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, candidate_contract = contracts
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "candidate"
    _publish_cache(teacher_root, teacher_contract, _teacher_logits)
    _publish_cache(candidate_root, candidate_contract, _teacher_logits)
    policy = json.loads(POLICY_PATH.read_text())
    policy["eval_gate"]["mean_kld_max"] = 100.0
    modified_policy = tmp_path / "modified-policy-copied-digest.json"
    modified_policy.write_text(json.dumps(policy, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="frozen policy"):
        compare_glm52_candidate_caches(
            teacher_root,
            candidate_root,
            policy_path=modified_policy,
            teacher_contract=teacher_contract,
            candidate_contract=candidate_contract,
        )


def test_modified_policy_body_with_recomputed_digest_is_rejected_by_file_pin(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, candidate_contract = contracts
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "candidate"
    _publish_cache(teacher_root, teacher_contract, _teacher_logits)
    _publish_cache(candidate_root, candidate_contract, _teacher_logits)
    policy = json.loads(POLICY_PATH.read_text())
    policy["eval_gate"]["mean_kld_max"] = 100.0
    policy_without_digest = dict(policy)
    policy_without_digest.pop("policy_contract_sha256")
    canonical = json.dumps(
        policy_without_digest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    policy["policy_contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    modified_policy = tmp_path / "modified-policy-recomputed-digest.json"
    modified_policy.write_text(json.dumps(policy, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="frozen policy"):
        compare_glm52_candidate_caches(
            teacher_root,
            candidate_root,
            policy_path=modified_policy,
            teacher_contract=teacher_contract,
            candidate_contract=candidate_contract,
        )


def test_unequal_length_rows_use_prompt_weighted_gate_means() -> None:
    rows = [
        {
            "position_count": 1,
            "mean_kld": 0.0,
            "top1_agreement": 1.0,
            "top1_match_count": 1,
            "ppl_ratio": 1.0,
            "token_klds": [0.0],
        },
        {
            "position_count": 9,
            "mean_kld": 0.5,
            "top1_agreement": 0.8,
            "top1_match_count": 7,
            "ppl_ratio": 1.04,
            "token_klds": [0.5] * 9,
        },
    ]

    metrics = summarize_rows(rows)

    assert metrics["mean_kld"] == pytest.approx(0.25)
    assert metrics["mean_kld"] <= 0.30
    assert metrics["top1_agreement"] == pytest.approx(0.90)
    assert metrics["top1_agreement"] >= 0.85
    assert metrics["mean_ppl_ratio"] == pytest.approx(1.02)
    assert metrics["p999_kld"] == pytest.approx(0.5)


def test_perturbed_candidate_fails_exactly_mean_kld_gate(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, candidate_contract = contracts
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "candidate"
    _publish_cache(teacher_root, teacher_contract, _teacher_logits)

    def perturbed(prompt: GLM52TeacherCachePrompt) -> np.ndarray:
        logits = _teacher_logits(prompt)
        logits[:, [1, 2]] = logits[:, [2, 1]]
        return np.ascontiguousarray(logits)

    _publish_cache(candidate_root, candidate_contract, perturbed)

    report = compare_glm52_candidate_caches(
        teacher_root,
        candidate_root,
        policy_path=POLICY_PATH,
        teacher_contract=teacher_contract,
        candidate_contract=candidate_contract,
    )

    assert report["metrics"]["mean_kld"] > 0.30
    assert report["metrics"]["p999_kld"] < 3.0
    assert report["metrics"]["top1_agreement"] == 1.0
    assert report["metrics"]["mean_ppl_ratio"] == pytest.approx(1.0)
    assert [name for name, passed in report["checks"].items() if not passed] == [
        "mean_kld"
    ]
    assert report["family_eval_gate_pass"] is False


def test_real_frozen_prompt_pack_split_accounting_reconciles() -> None:
    accounting = load_glm52_prompt_pack_accounting(PROMPT_PACK_PATH)

    assert accounting["prompt_count"] == 66
    assert accounting["source_token_count"] == 810
    assert accounting["predictor_position_count"] == 744
    assert accounting["split_prompt_counts"] == {
        "holdout": 22,
        "report": 22,
        "selection": 22,
    }
    assert accounting["split_position_counts"] == {
        "holdout": 251,
        "report": 238,
        "selection": 255,
    }
    assert sum(accounting["split_position_counts"].values()) == 744


def test_real_frozen_prompt_pack_builds_exact_right_padded_predictor_batch() -> None:
    payload = json.loads(PROMPT_PACK_PATH.read_text())
    prompts = tuple(
        GLM52TeacherCachePrompt.from_prompt_pack_row(row)
        for row in payload["prompt_rows"]
    )

    token_batch, valid_mask, right_padding = prepare_glm52_candidate_predictor_batch(
        prompts,
        vocab_size=154_880,
    )

    assert token_batch.shape == (66, 18)
    assert valid_mask.shape == (66, 18)
    assert int(valid_mask.sum()) == 744
    assert right_padding.tolist() == [18 - (prompt.token_count - 1) for prompt in prompts]


def test_validator_refuses_root_that_has_not_passed_strict_audit(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, candidate_contract = contracts
    teacher_root = tmp_path / "teacher"
    _publish_cache(teacher_root, teacher_contract, _teacher_logits)
    unaudited_candidate_root = tmp_path / "candidate"
    unaudited_candidate_root.mkdir()

    with pytest.raises(ValueError, match="candidate cache failed strict audit"):
        compare_glm52_candidate_caches(
            teacher_root,
            unaudited_candidate_root,
            policy_path=POLICY_PATH,
            teacher_contract=teacher_contract,
            candidate_contract=candidate_contract,
        )


def test_candidate_contract_records_kind_and_frozen_composite_identity(
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    _teacher, candidate = contracts

    assert "candidate_kind=production_composite" in candidate.producer["implementation"]
    assert (
        candidate.non_vq_package["bound_package_identity"]
        == GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
    )
    assert candidate.bound_identity_sha256 == GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256


def test_recovered_candidate_contract_has_canonical_recovery_specific_identity(
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher, _candidate = contracts
    recovery_candidate_identity = "5" * 64
    recovery_manifest_body = "6" * 64
    accepted_audit = "f" * 64

    recovered = build_glm52_candidate_cache_contract(
        teacher,
        recovery_candidate_identity_sha256=recovery_candidate_identity,
        recovery_manifest_body_sha256=recovery_manifest_body,
        accepted_composite_audit_sha256=accepted_audit,
    )
    expected = candidate_module.canonical_sha256(
        {
            "schema_version": 1,
            "identity_kind": "glm52_recovered_candidate_cache_contract_v1",
            "accepted_baseline_composite_identity_sha256": (
                GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
            ),
            "recovery_candidate_identity_sha256": recovery_candidate_identity,
            "recovery_manifest_body_sha256": recovery_manifest_body,
            "accepted_composite_audit_sha256": accepted_audit,
        }
    )

    assert recovered.bound_identity_sha256 == expected
    assert recovered.non_vq_package["bound_package_identity"] == expected
    assert recovered.bound_identity_sha256 != GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
    assert "candidate_kind=recovered_composite" in recovered.producer["implementation"]


def test_recovered_candidate_contract_round_trips_from_durable_cache_manifest(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher, _candidate = contracts
    recovered = build_glm52_candidate_cache_contract(
        teacher,
        recovery_candidate_identity_sha256="5" * 64,
        recovery_manifest_body_sha256="6" * 64,
        accepted_composite_audit_sha256="f" * 64,
    )
    root = tmp_path / "recovered-cache"
    _publish_cache(root, recovered, _teacher_logits)

    loaded = candidate_module._contract_from_cache_manifest(
        root,
        prompt_pack_path=PROMPT_PACK_PATH,
        candidate=True,
    )

    assert loaded.bound_identity_sha256 == recovered.bound_identity_sha256
    assert loaded.producer == recovered.producer
    assert loaded.non_vq_package == recovered.non_vq_package
    assert loaded.bound_identity_sha256 != GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256


def _recovery_comparison_authority() -> dict[str, str]:
    return {
        "expected_recovery_candidate_identity_sha256": "5" * 64,
        "expected_recovery_manifest_body_sha256": "6" * 64,
        "expected_accepted_composite_audit_sha256": "f" * 64,
    }


def _recovered_candidate_contract(
    teacher_contract: GLM52TeacherCacheContract,
) -> GLM52TeacherCacheContract:
    authority = _recovery_comparison_authority()
    return build_glm52_candidate_cache_contract(
        teacher_contract,
        recovery_candidate_identity_sha256=authority[
            "expected_recovery_candidate_identity_sha256"
        ],
        recovery_manifest_body_sha256=authority[
            "expected_recovery_manifest_body_sha256"
        ],
        accepted_composite_audit_sha256=authority[
            "expected_accepted_composite_audit_sha256"
        ],
    )


def test_compare_accepts_exact_externally_authorized_recovered_candidate(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, _baseline_contract = contracts
    recovered_contract = _recovered_candidate_contract(teacher_contract)
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "recovered"
    _publish_cache(teacher_root, teacher_contract, _teacher_logits)
    _publish_cache(candidate_root, recovered_contract, _teacher_logits)

    report = compare_glm52_candidate_caches(
        teacher_root,
        candidate_root,
        policy_path=POLICY_PATH,
        teacher_contract=teacher_contract,
        candidate_contract=recovered_contract,
        **_recovery_comparison_authority(),
    )

    assert report["family_eval_gate_pass"] is True
    assert report["candidate_kind"] == "recovered_composite"
    assert report["candidate_cache_identity"]["candidate_kind"] == "recovered_composite"
    assert "candidate_kind=recovered_composite" in report[
        "candidate_cache_identity"
    ]["producer"]["implementation"]
    assert (
        report["candidate_artifact_identity_sha256"]
        == _recovery_comparison_authority()[
            "expected_recovery_candidate_identity_sha256"
        ]
    )


@pytest.mark.parametrize(
    "missing_name",
    [
        "expected_recovery_candidate_identity_sha256",
        "expected_recovery_manifest_body_sha256",
        "expected_accepted_composite_audit_sha256",
    ],
)
def test_compare_rejects_incomplete_recovery_authority(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
    missing_name: str,
) -> None:
    teacher_contract, _baseline_contract = contracts
    recovered_contract = _recovered_candidate_contract(teacher_contract)
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "recovered"
    _publish_cache(teacher_root, teacher_contract, _teacher_logits)
    _publish_cache(candidate_root, recovered_contract, _teacher_logits)
    authority = _recovery_comparison_authority()
    authority.pop(missing_name)

    with pytest.raises(ValueError, match="requires all three external identities"):
        compare_glm52_candidate_caches(
            teacher_root,
            candidate_root,
            policy_path=POLICY_PATH,
            teacher_contract=teacher_contract,
            candidate_contract=recovered_contract,
            **authority,
        )


def test_compare_rejects_wrong_recovery_identity_authority(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, _baseline_contract = contracts
    recovered_contract = _recovered_candidate_contract(teacher_contract)
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "recovered"
    _publish_cache(teacher_root, teacher_contract, _teacher_logits)
    _publish_cache(candidate_root, recovered_contract, _teacher_logits)
    authority = _recovery_comparison_authority()
    authority["expected_recovery_candidate_identity_sha256"] = "7" * 64

    with pytest.raises(ValueError, match="not the exact externally authorized recovery"):
        compare_glm52_candidate_caches(
            teacher_root,
            candidate_root,
            policy_path=POLICY_PATH,
            teacher_contract=teacher_contract,
            candidate_contract=recovered_contract,
            **authority,
        )


def test_compare_rejects_baseline_candidate_relabelled_as_recovery(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, baseline_contract = contracts
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "baseline"
    _publish_cache(teacher_root, teacher_contract, _teacher_logits)
    _publish_cache(candidate_root, baseline_contract, _teacher_logits)

    with pytest.raises(ValueError, match="not the exact externally authorized recovery"):
        compare_glm52_candidate_caches(
            teacher_root,
            candidate_root,
            policy_path=POLICY_PATH,
            teacher_contract=teacher_contract,
            candidate_contract=baseline_contract,
            **_recovery_comparison_authority(),
        )


def test_compare_rejects_recovered_candidate_relabelled_as_baseline(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
) -> None:
    teacher_contract, _baseline_contract = contracts
    recovered_contract = _recovered_candidate_contract(teacher_contract)
    teacher_root = tmp_path / "teacher"
    candidate_root = tmp_path / "recovered"
    _publish_cache(teacher_root, teacher_contract, _teacher_logits)
    _publish_cache(candidate_root, recovered_contract, _teacher_logits)

    with pytest.raises(ValueError, match="derived from the source teacher"):
        compare_glm52_candidate_caches(
            teacher_root,
            candidate_root,
            policy_path=POLICY_PATH,
            teacher_contract=teacher_contract,
            candidate_contract=recovered_contract,
        )


def _candidate_runner_prompts() -> tuple[GLM52TeacherCachePrompt, ...]:
    predictor_lengths = (18, *(11 for _ in range(58)), *(13 for _ in range(4)), *(12 for _ in range(3)))
    assert len(predictor_lengths) == 66
    assert sum(predictor_lengths) == 744
    return tuple(
        GLM52TeacherCachePrompt(
            prompt_id=f"fake-{index:03d}",
            split="selection",
            domain="route",
            tuning_eligible=True,
            encoded_token_ids=tuple((position % 31) + 1 for position in range(length + 1)),
        )
        for index, length in enumerate(predictor_lengths)
    )


def _install_fake_candidate_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: list[tuple[str, int]],
):
    class FakeMX(ModuleType):
        float32 = np.float32

        def array(self, value):
            return np.asarray(value)

        def contiguous(self, value):
            return np.ascontiguousarray(value)

        def eval(self, *_values):
            return None

        def take(self, value, indices, axis=0):
            return np.take(value, indices, axis=axis)

        def zeros(self, shape, dtype):
            return np.zeros(shape, dtype=dtype)

        def put_along_axis(self, value, indices, updates, axis=None):
            result = np.array(value, copy=True)
            if axis is not None:
                raise AssertionError("fake runtime only supports flattened scatter")
            result.reshape(-1)[np.asarray(indices)] = np.asarray(updates)
            return result

        def clear_cache(self):
            return None

    class FakeMoE:
        sharding_group = None

        def __init__(self, layer_index: int):
            self.layer_index = layer_index
            self.switch_mlp = self._switch

        def gate(self, valid_hidden):
            events.append(("gate", self.layer_index))
            rows = valid_hidden.shape[0]
            expert_ids = np.tile(np.arange(8, dtype=np.int64), (rows, 1))
            scores = np.full((rows, 8), 1.0 / 8.0, dtype=np.float64)
            return expert_ids, scores

        def _switch(self, valid_hidden, _expert_ids):
            events.append(("routed", self.layer_index))
            return np.repeat(valid_hidden[:, None, :], 8, axis=1)

        def get(self, _name):
            return None

    class FakeAttention:
        def __call__(self, hidden, _mask, _cache, _prev_topk):
            return np.zeros_like(hidden), None

    class FakeLayer:
        def __init__(self, layer_index: int):
            self.layer_idx = layer_index
            self.self_attn = FakeAttention()
            self.input_layernorm = lambda value: value
            self.post_attention_layernorm = lambda value: value
            self.mlp = FakeMoE(layer_index) if layer_index >= 3 else (lambda value: np.zeros_like(value))

    fake_mx = FakeMX("mlx.core")
    fake_adapter = ModuleType("ramp.models.glm52_vq_adapter")
    fake_adapter.Glm52VQMoE = FakeMoE
    fake_composite = ModuleType("ramp.models.glm52_composite_loader")
    fake_composite.assert_glm52_production_inputs_unchanged = lambda _fingerprint: None
    fake_base = ModuleType("mlx_lm.models.base")
    fake_base.create_causal_mask = lambda length, right_padding: np.zeros((length, length), dtype=np.float32)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "ramp.models.glm52_vq_adapter", fake_adapter)
    monkeypatch.setitem(sys.modules, "ramp.models.glm52_composite_loader", fake_composite)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.base", fake_base)

    model = SimpleNamespace(
        args=SimpleNamespace(
            vocab_size=32,
            hidden_size=2,
            num_experts_per_tok=8,
        ),
        model=SimpleNamespace(
            embed_tokens=lambda tokens: np.stack((tokens, tokens + 1), axis=-1).astype(np.float32),
            layers=[FakeLayer(index) for index in range(78)],
            norm=lambda hidden: hidden,
        ),
        lm_head=lambda hidden: np.repeat(hidden[..., :1], 32, axis=-1),
    )
    return candidate_module.GLM52CandidateWorkload(
        model=model,
        validated_inputs=SimpleNamespace(input_fingerprint=object()),
        load_report=SimpleNamespace(),
    )


def test_candidate_runner_same_pass_capture_is_exact_and_byte_neutral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = _candidate_runner_prompts()

    def run(capture: bool):
        events: list[tuple[str, int]] = []
        workload = _install_fake_candidate_runtime(monkeypatch, events=events)
        logits: dict[int, bytes] = {}
        traces = []

        evidence = candidate_module.run_glm52_candidate_to_sink(
            workload,
            prompts,
            pending_prompt_indices=tuple(range(len(prompts))),
            checkpoint_dir=tmp_path,
            checkpoint_identities=None,
            phase_boundary=lambda: None,
            sink=lambda index, value: logits.__setitem__(index, value.tobytes()),
            capture_route_trace=capture,
            route_trace_collector=lambda trace: (
                events.append(("observed", trace.layer_index)), traces.append(trace)
            ),
        )
        return evidence, logits, traces, events

    uncaptured, uncaptured_logits, uncaptured_traces, _ = run(False)
    captured, captured_logits, traces, events = run(True)

    assert uncaptured.route_traces is None
    assert uncaptured_traces == []
    assert captured.route_traces == tuple(traces)
    assert captured.output_identity_sha256 == uncaptured.output_identity_sha256
    assert captured_logits == uncaptured_logits
    assert [trace.layer_index for trace in traces] == list(range(3, 78))
    assert all(trace.expert_ids.shape == (744, 8) for trace in traces)
    assert all(trace.scores.shape == (744, 8) for trace in traces)
    assert all(trace.expert_ids.dtype == np.dtype("int32") for trace in traces)
    assert all(trace.scores.dtype == np.dtype("float32") for trace in traces)
    assert all(trace.valid_assignment_count == 744 * 8 for trace in traces)
    assert all(trace.padded_assignment_count == 0 for trace in traces)
    assert events == [
        event
        for layer_index in range(3, 78)
        for event in (
            ("gate", layer_index),
            ("routed", layer_index),
            ("observed", layer_index),
        )
    ]


def _recovery_producer_kwargs(tmp_path: Path, contract) -> dict[str, object]:
    return {
        "contract": contract,
        "profile_path": tmp_path / "profile.yaml",
        "config_path": tmp_path / "config.json",
        "source_index_path": tmp_path / "index.json",
        "tokenizer_dir": tmp_path / "tokenizer",
        "tokenizer_readiness_json": tmp_path / "tokenizer-readiness.json",
        "family_policy_json": tmp_path / "policy.json",
        "non_vq_artifact_dir": tmp_path / "non-vq",
        "non_vq_evidence_json": tmp_path / "non-vq.json",
        "routed_artifact_dir": tmp_path / "accepted-baseline",
        "composite_audit_json": tmp_path / "composite.json",
        "materialization_runs_jsonl": tmp_path / "runs.jsonl",
        "full_bind_preflight_json": tmp_path / "preflight.json",
        "teacher_cache_root": tmp_path / "teacher-cache",
        "prompt_pack_path": tmp_path / "prompts.json",
        "cache_root": tmp_path / "candidate-cache",
        "ledger_path": tmp_path / "candidate.ledger.jsonl",
        "route_trace_root": tmp_path / "candidate-route-trace",
        "recovery_dir": tmp_path / "recovery",
        "expected_seed_manifest_sha256": "1" * 64,
        "expected_stats_manifest_sha256": "2" * 64,
        "expected_full_source_blob_inventory_sha256": "3" * 64,
        "expected_routed_source_blob_inventory_sha256": "4" * 64,
        "expected_recovery_lever": "selection_diagonal_hessian_importance_weighted_reround_v1",
        "expected_recovery_policy": {
            "recovery_lever": "selection_diagonal_hessian_importance_weighted_reround_v1",
            "scale_estimator": "importance_weighted_least_squares",
            "rounding_objective": "selection_diagonal_hessian_weighted_squared_error",
        },
        "expected_composite_audit_sha256": "f" * 64,
        "expected_recovery_mixed_artifact_identity_sha256": "5" * 64,
        "expected_recovery_manifest_body_sha256": "6" * 64,
    }


def test_candidate_producer_audits_recovery_before_load_and_forwards_v2_authority(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    testing_teacher, _testing_candidate = contracts
    teacher_contract = GLM52TeacherCacheContract.from_frozen_prompt_pack(
        PROMPT_PACK_PATH,
        producer=testing_teacher.producer,
        source_evidence=testing_teacher.source_evidence,
        non_vq_package=testing_teacher.non_vq_package,
    )
    candidate_contract = build_glm52_candidate_cache_contract(teacher_contract)
    events: list[str] = []
    forwarded: dict[str, object] = {}

    class RecoveryAudit:
        recovery_dir = str(tmp_path / "recovery")
        candidate_identity_sha256 = "5" * 64
        manifest_body_sha256 = "6" * 64
        accepted_composite_audit_sha256 = "f" * 64
        audit_pass = True

        def verify_current_identity(self):
            events.append("recovery-recheck")

    def audit_recovery(recovery_dir, **kwargs):
        events.append("recovery-audit")
        assert recovery_dir == tmp_path / "recovery"
        assert kwargs["profile"] is profile
        assert kwargs["expected_seed_manifest_sha256"] == "1" * 64
        assert kwargs["expected_stats_manifest_sha256"] == "2" * 64
        return RecoveryAudit()

    @dataclass(frozen=True)
    class Validated:
        profile: object
        routed_artifact_dir: Path
        artifact_identity: object
        input_fingerprint: object

    profile = SimpleNamespace(name="glm52-test")
    validated = Validated(
        profile=profile,
        routed_artifact_dir=tmp_path / "accepted-baseline",
        artifact_identity=SimpleNamespace(
            sha256=GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
        ),
        input_fingerprint=object(),
    )

    class FakeComposite:
        @staticmethod
        def validate_glm52_production_inputs(**_kwargs):
            events.append("baseline-audit")
            return validated

        @staticmethod
        def load_authenticated_glm52_composite(value):
            events.append("load")
            assert value.routed_artifact_dir == tmp_path / "recovery" / "artifact"
            return object(), SimpleNamespace(
                artifact_identity_sha256=GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
            )

    producer_api = candidate_module._load_teacher_producer_api()

    def fake_candidate_runner(*_args, capture_route_trace=False, **_kwargs):
        events.append("forward")
        assert capture_route_trace is True
        return producer_api.SourceRunEvidence(
            False,
            "7" * 64,
            route_traces=(SimpleNamespace(layer_index=3),),
        )

    def fake_trace_writer(root, layers, **kwargs):
        events.append("trace")
        forwarded.update({"root": root, "layers": layers, **kwargs})
        return {"published": True}

    diagnostics = SimpleNamespace(
        RECOVERY_CANDIDATE_TRACE_SCHEMA_VERSION=2,
        write_route_trace_artifact=fake_trace_writer,
    )

    def fake_produce(**kwargs):
        assert kwargs["route_trace_root"] == tmp_path / "candidate-route-trace"
        kwargs["non_vq_package_auditor"]()
        workload = kwargs["source_loader"]()
        evidence = kwargs["source_runner"](
            workload,
            candidate_contract.prompts,
            pending_prompt_indices=(0, 1, 2),
            checkpoint_dir=tmp_path,
            checkpoint_identities=None,
            phase_boundary=lambda: None,
            sink=lambda *_args: None,
            capture_route_trace=True,
        )
        assert evidence.route_traces is not None
        events.append("manifest")
        kwargs["route_trace_writer"](
            tmp_path / "candidate-route-trace",
            evidence.route_traces,
            side="source",
            authority={"capture_output_sha256": "8" * 64},
        )
        return SimpleNamespace(manifest=SimpleNamespace(), manifest_path=tmp_path / "candidate-cache/manifest.json")

    original_import = candidate_module.importlib.import_module
    monkeypatch.setattr(
        candidate_module.importlib,
        "import_module",
        lambda name: FakeComposite if name == "ramp.models.glm52_composite_loader" else original_import(name),
    )
    monkeypatch.setattr(
        candidate_module,
        "_load_teacher_producer_api",
        lambda: SimpleNamespace(
            SourceRunEvidence=producer_api.SourceRunEvidence,
            produce_glm52_teacher_cache=fake_produce,
        ),
    )
    monkeypatch.setattr(candidate_module, "_load_recovery_audit_api", lambda: SimpleNamespace(audit_glm52_recovery_mixed_artifact=audit_recovery), raising=False)
    monkeypatch.setattr(candidate_module, "_load_route_diagnostics_api", lambda: diagnostics, raising=False)
    monkeypatch.setattr(candidate_module, "run_glm52_candidate_to_sink", fake_candidate_runner)
    monkeypatch.setattr(candidate_module, "_load_json_object", lambda *_args, **_kwargs: {"prompt": "authenticated"})
    monkeypatch.setattr(candidate_module, "_contract_from_cache_manifest", lambda *_args, **_kwargs: teacher_contract)
    monkeypatch.setattr(
        candidate_module,
        "_strict_audit",
        lambda *_args, label, **_kwargs: SimpleNamespace(
            valid=True,
            release_eligible=True,
            cache_content_sha256=("9" if label == "teacher" else "a") * 64,
            manifest_body_sha256=("b" if label == "teacher" else "c") * 64,
        ),
    )
    monkeypatch.setattr(
        candidate_module,
        "_cache_identity_envelope",
        lambda **kwargs: {"identity_sha256": ("d" if kwargs["record_type"] == candidate_module.TEACHER_CACHE_IDENTITY_RECORD_TYPE else "e") * 64},
    )

    candidate_module.produce_glm52_candidate_cache(
        **_recovery_producer_kwargs(tmp_path, candidate_contract)
    )

    assert events == [
        "baseline-audit",
        "recovery-audit",
        "recovery-recheck",
        "load",
        "forward",
        "recovery-recheck",
        "manifest",
        "recovery-recheck",
        "trace",
        "recovery-recheck",
    ]
    assert forwarded["side"] == "candidate"
    assert forwarded["schema_version"] == 2
    assert forwarded["authority"] == {
        "teacher_cache_path": str(tmp_path / "teacher-cache" / candidate_module.cache_api.MANIFEST_FILENAME),
        "teacher_cache_content_sha256": "9" * 64,
        "accepted_baseline_composite_path": str(tmp_path / "accepted-baseline"),
        "accepted_baseline_composite_identity_sha256": GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256,
        "candidate_cache_identity_sha256": "e" * 64,
        "recovery_mixed_artifact_identity_sha256": "5" * 64,
        "recovery_manifest_body_sha256": "6" * 64,
        "accepted_composite_audit_sha256": "f" * 64,
        "prompt_pack_path": str(candidate_contract.prompt_authority["path"]),
        "prompt_pack_sha256": str(candidate_contract.prompt_authority["file_sha256"]),
        "model_id": str(candidate_contract.source["model_id"]),
        "model_revision": str(candidate_contract.source["revision"]),
        "producer_implementation_id": candidate_module.CANDIDATE_ROUTE_TRACE_PRODUCER_IMPLEMENTATION_ID,
        "capture_output_path": str(tmp_path / "candidate-route-trace"),
        "capture_output_sha256": "8" * 64,
        "evidence_class": "release",
    }


@pytest.mark.parametrize("mutation_timing", ["immediately-before-writer", "during-writer"])
def test_recovery_trace_identity_mutation_rolls_back_all_outputs_and_cleanly_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_timing: str,
) -> None:
    producer_api = candidate_module._load_teacher_producer_api()
    teacher_contract = GLM52TeacherCacheContract.for_testing(
        prompts=_candidate_runner_prompts(),
        vocab_size=32,
    )
    baseline_contract = build_glm52_candidate_cache_contract(teacher_contract)
    kwargs = _recovery_producer_kwargs(tmp_path, baseline_contract)
    attempts = 0
    writer_calls = 0

    class RecoveryAudit:
        recovery_dir = str(tmp_path / "recovery")
        candidate_identity_sha256 = "5" * 64
        manifest_body_sha256 = "6" * 64
        accepted_composite_audit_sha256 = "f" * 64
        audit_pass = True
        mutated = False

        def verify_current_identity(self) -> None:
            if self.mutated:
                raise RuntimeError("recovery identity mutated")

    recovery_audit = RecoveryAudit()

    def audit_recovery(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        recovery_audit.mutated = False
        return recovery_audit

    @dataclass(frozen=True)
    class Validated:
        profile: object
        routed_artifact_dir: Path
        artifact_identity: object

    validated = Validated(
        profile=SimpleNamespace(name="glm52-test"),
        routed_artifact_dir=tmp_path / "accepted-baseline",
        artifact_identity=SimpleNamespace(
            sha256=GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
        ),
    )
    composite = SimpleNamespace(
        validate_glm52_production_inputs=lambda **_kwargs: validated,
        load_authenticated_glm52_composite=lambda _value: (
            object(),
            SimpleNamespace(
                artifact_identity_sha256=GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256
            ),
        ),
    )

    base_ids = np.tile(np.arange(8, dtype=np.int32), (744, 1))
    base_scores = np.full((744, 8), np.float32(1.0 / 8), dtype=np.float32)

    def fake_candidate_runner(
        _source,
        prompts,
        *,
        pending_prompt_indices,
        checkpoint_dir,
        checkpoint_identities,
        phase_boundary,
        sink,
        capture_route_trace,
    ):
        del checkpoint_identities
        assert capture_route_trace is True
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "run-owned.partial").write_text("partial")
        phase_boundary()
        for index in pending_prompt_indices:
            sink(
                index,
                np.tile(
                    np.arange(teacher_contract.vocab_size, dtype=np.float32),
                    (prompts[index].token_count - 1, 1),
                ),
            )
        return producer_api.SourceRunEvidence(
            False,
            "7" * 64,
            route_traces=tuple(
                SimpleNamespace(
                    layer_index=layer_index,
                    expert_ids=base_ids.copy(),
                    scores=base_scores.copy(),
                    valid_assignment_count=744 * 8,
                    padded_assignment_count=0,
                )
                for layer_index in range(3, 78)
            ),
        )

    def strict_audit(*_args, label, **_kwargs):
        if (
            label == "candidate"
            and attempts == 1
            and mutation_timing == "immediately-before-writer"
        ):
            recovery_audit.mutated = True
        return SimpleNamespace(
            valid=True,
            release_eligible=True,
            cache_content_sha256=("9" if label == "teacher" else "a") * 64,
            manifest_body_sha256=("b" if label == "teacher" else "c") * 64,
        )

    def trace_writer(root, _layers, **_kwargs):
        nonlocal writer_calls
        writer_calls += 1
        root.mkdir(parents=True, exist_ok=True)
        (root / "trace.partial").write_text("partial")
        if attempts == 1 and mutation_timing == "during-writer":
            recovery_audit.mutated = True
        (root / "trace.partial").replace(root / "trace.json")
        return {"published": True}

    class Locks:
        @contextmanager
        def heavy_job_lock(self, _path):
            yield

        @contextmanager
        def run_lock(self, _path):
            yield

    def transactional_produce(**producer_kwargs):
        return producer_api.produce_glm52_teacher_cache(
            **producer_kwargs,
            lock_provider=Locks(),
            memory_counter_reader=lambda: {"pageouts": 0, "swapouts": 0},
            wired_policy_probe=lambda: True,
            buffer_releaser=lambda: None,
        )

    original_import = candidate_module.importlib.import_module
    monkeypatch.setattr(
        candidate_module.importlib,
        "import_module",
        lambda name: composite
        if name == "ramp.models.glm52_composite_loader"
        else original_import(name),
    )
    monkeypatch.setattr(
        candidate_module,
        "_load_teacher_producer_api",
        lambda: SimpleNamespace(
            SourceRunEvidence=producer_api.SourceRunEvidence,
            produce_glm52_teacher_cache=transactional_produce,
        ),
    )
    monkeypatch.setattr(
        candidate_module,
        "_load_recovery_audit_api",
        lambda: SimpleNamespace(
            audit_glm52_recovery_mixed_artifact=audit_recovery
        ),
    )
    monkeypatch.setattr(
        candidate_module,
        "_load_route_diagnostics_api",
        lambda: SimpleNamespace(
            RECOVERY_CANDIDATE_TRACE_SCHEMA_VERSION=2,
            write_route_trace_artifact=trace_writer,
        ),
    )
    monkeypatch.setattr(
        candidate_module,
        "run_glm52_candidate_to_sink",
        fake_candidate_runner,
    )
    monkeypatch.setattr(
        candidate_module,
        "_load_json_object",
        lambda *_args, **_kwargs: {"prompt": "authenticated"},
    )
    monkeypatch.setattr(
        candidate_module,
        "_contract_from_cache_manifest",
        lambda *_args, **_kwargs: teacher_contract,
    )
    monkeypatch.setattr(candidate_module, "_strict_audit", strict_audit)
    monkeypatch.setattr(
        candidate_module,
        "_cache_identity_envelope",
        lambda **_kwargs: {"identity_sha256": "d" * 64},
    )

    with pytest.raises(RuntimeError, match="recovery identity mutated"):
        candidate_module.produce_glm52_candidate_cache(**kwargs)

    transaction_paths = (
        Path(kwargs["cache_root"]),
        Path(kwargs["ledger_path"]),
        Path(f"{kwargs['ledger_path']}.candidate-state"),
        Path(kwargs["route_trace_root"]),
    )
    assert all(not path.exists() for path in transaction_paths)

    result = candidate_module.produce_glm52_candidate_cache(**kwargs)

    assert result.completed is True
    assert result.manifest_path is not None and result.manifest_path.is_file()
    assert Path(kwargs["route_trace_root"], "trace.json").is_file()
    assert attempts == 2
    assert writer_calls == (1 if mutation_timing == "immediately-before-writer" else 2)


def test_candidate_production_api_has_no_raw_accounting_authority() -> None:
    parameters = inspect.signature(
        candidate_module.produce_glm52_candidate_cache
    ).parameters
    assert "non_routed_tensor_payload_bytes" not in parameters
    assert "logical_payload_limit_bytes" not in parameters
    assert "expected_composite_audit_sha256" in parameters


@pytest.mark.parametrize(
    ("candidate_identity", "manifest_identity", "message"),
    [
        ("f" * 64, "6" * 64, "mixed artifact identity"),
        ("5" * 64, "f" * 64, "manifest body identity"),
    ],
)
def test_candidate_producer_rejects_recovery_identity_mismatch_before_load(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
    monkeypatch: pytest.MonkeyPatch,
    candidate_identity: str,
    manifest_identity: str,
    message: str,
) -> None:
    testing_teacher, _testing_candidate = contracts
    teacher_contract = GLM52TeacherCacheContract.from_frozen_prompt_pack(
        PROMPT_PACK_PATH,
        producer=testing_teacher.producer,
        source_evidence=testing_teacher.source_evidence,
        non_vq_package=testing_teacher.non_vq_package,
    )
    candidate_contract = build_glm52_candidate_cache_contract(teacher_contract)
    events: list[str] = []
    validated = SimpleNamespace(
        profile=SimpleNamespace(name="glm52-test"),
        artifact_identity=SimpleNamespace(sha256=GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256),
    )
    composite = SimpleNamespace(
        validate_glm52_production_inputs=lambda **_kwargs: validated,
        load_authenticated_glm52_composite=lambda _value: events.append("load"),
    )

    def fake_produce(**kwargs):
        kwargs["non_vq_package_auditor"]()
        kwargs["source_loader"]()

    original_import = candidate_module.importlib.import_module
    monkeypatch.setattr(candidate_module.importlib, "import_module", lambda name: composite if name == "ramp.models.glm52_composite_loader" else original_import(name))
    monkeypatch.setattr(candidate_module, "_load_teacher_producer_api", lambda: SimpleNamespace(produce_glm52_teacher_cache=fake_produce))
    monkeypatch.setattr(candidate_module, "_contract_from_cache_manifest", lambda *_args, **_kwargs: teacher_contract)
    monkeypatch.setattr(candidate_module, "_load_json_object", lambda *_args, **_kwargs: {"prompt": "authenticated"})
    monkeypatch.setattr(candidate_module, "_strict_audit", lambda *_args, **_kwargs: SimpleNamespace(valid=True, release_eligible=True, cache_content_sha256="9" * 64, manifest_body_sha256="b" * 64))
    monkeypatch.setattr(
        candidate_module,
        "_load_recovery_audit_api",
        lambda: SimpleNamespace(
            audit_glm52_recovery_mixed_artifact=lambda *_args, **_kwargs: SimpleNamespace(
                audit_pass=True,
                candidate_identity_sha256=candidate_identity,
                manifest_body_sha256=manifest_identity,
                recovery_dir=str(tmp_path / "recovery"),
            )
        ),
        raising=False,
    )

    with pytest.raises(ValueError, match=message):
        candidate_module.produce_glm52_candidate_cache(
            **_recovery_producer_kwargs(tmp_path, candidate_contract)
        )

    assert events == []


def test_candidate_route_capture_requires_complete_frozen_pack_before_production(
    tmp_path: Path,
    contracts: tuple[GLM52TeacherCacheContract, GLM52TeacherCacheContract],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _teacher_contract, candidate_contract = contracts
    calls: list[str] = []
    monkeypatch.setattr(
        candidate_module,
        "_load_teacher_producer_api",
        lambda: SimpleNamespace(
            produce_glm52_teacher_cache=lambda **_kwargs: calls.append("produce")
        ),
    )

    with pytest.raises(ValueError, match="complete frozen 66-prompt, 744-position pack"):
        candidate_module.produce_glm52_candidate_cache(
            **_recovery_producer_kwargs(tmp_path, candidate_contract)
        )

    assert calls == []


def test_recovery_leaf_audit_loads_without_validate_package_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "keep.validate", raising=False)
    monkeypatch.delitem(
        sys.modules,
        "keep.validate.glm52_recovery_artifact",
        raising=False,
    )

    api = candidate_module._load_recovery_audit_api()

    assert api.audit_glm52_recovery_mixed_artifact.__name__ == (
        "audit_glm52_recovery_mixed_artifact"
    )
    assert "keep.validate" not in sys.modules


def test_eval_cli_has_produce_and_compare_without_tuning_hooks() -> None:
    cli_path = REPO_ROOT / "benchmarks/eval_glm52_candidate_full_vocab.py"
    spec = importlib.util.spec_from_file_location("glm52_candidate_eval_cli", cli_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    parser = module._build_parser()
    help_text = parser.format_help()
    assert "produce" in help_text
    assert "compare" in help_text
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    options = {
        option
        for command_parser in subparsers.choices.values()
        for action in command_parser._actions
        for option in action.option_strings
    }
    assert "--threshold" not in options
    assert "--split" not in options
    assert "--limit-prompts" not in options
    assert "--prompt" not in options
    assert all(
        "--allow-non-release-teacher-cache" in {
            option
            for action in command_parser._actions
            for option in action.option_strings
        }
        for command_parser in subparsers.choices.values()
    )
