from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "benchmarks/eval_glm52_candidate_full_vocab.py"
BASELINE_IDENTITY = "a" * 64
RECOVERY_CANDIDATE_IDENTITY = "b" * 64
RECOVERY_MANIFEST_IDENTITY = "c" * 64
ACCEPTED_AUDIT_IDENTITY = "d" * 64
RECOVERED_BOUND_IDENTITY = "e" * 64


@pytest.fixture
def cli() -> Any:
    module_name = "glm52_candidate_eval_cli_task6_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _produce_argv(tmp_path: Path) -> list[str]:
    values = {
        "profile-path": "profile.json",
        "config-path": "config.json",
        "source-index-path": "index.json",
        "tokenizer-dir": "tokenizer",
        "tokenizer-readiness-json": "tokenizer-readiness.json",
        "family-policy-json": "family-policy.json",
        "prompt-pack-json": "prompt-pack.json",
        "teacher-cache-root": "teacher-cache",
        "non-vq-artifact-dir": "non-vq",
        "non-vq-evidence-json": "non-vq-evidence.json",
        "routed-artifact-dir": "routed",
        "composite-audit-json": "composite-audit.json",
        "materialization-runs-jsonl": "runs.jsonl",
        "full-bind-preflight-json": "preflight.json",
        "candidate-cache-root": "candidate-cache",
        "ledger-path": str(tmp_path / "ledger.jsonl"),
    }
    return ["produce"] + [item for key, value in values.items() for item in (f"--{key}", value)]


def _compare_argv(tmp_path: Path) -> list[str]:
    values = {
        "teacher-cache-root": "teacher-cache",
        "candidate-cache-root": "candidate-cache",
        "prompt-pack-json": "prompt-pack.json",
        "family-policy-json": "family-policy.json",
        "output-json": str(tmp_path / "comparison.json"),
    }
    return ["compare"] + [item for key, value in values.items() for item in (f"--{key}", value)]


def _recovery_produce_argv(tmp_path: Path, policy_path: Path) -> list[str]:
    values = {
        "route-trace-root": "route-traces",
        "recovery-dir": "recovery",
        "expected-seed-manifest-sha256": "1" * 64,
        "expected-stats-manifest-sha256": "2" * 64,
        "expected-full-source-blob-inventory-sha256": "3" * 64,
        "expected-routed-source-blob-inventory-sha256": "4" * 64,
        "expected-recovery-lever": "reap_rounding",
        "expected-recovery-policy-json": str(policy_path),
        "expected-accepted-composite-audit-sha256": ACCEPTED_AUDIT_IDENTITY,
        "expected-recovery-candidate-identity-sha256": RECOVERY_CANDIDATE_IDENTITY,
        "expected-recovery-manifest-body-sha256": RECOVERY_MANIFEST_IDENTITY,
    }
    return _produce_argv(tmp_path) + [
        item for key, value in values.items() for item in (f"--{key}", value)
    ]


def _result() -> Any:
    manifest = {
        "payload_integrity_pass": True,
        "all_producer_memory_counters_known": True,
        "all_producer_memory_clean": True,
        "system_wired_default": True,
        "release_eligible": True,
    }
    return SimpleNamespace(
        completed=True,
        manifest=SimpleNamespace(to_dict=lambda: manifest),
        manifest_path=Path("candidate-cache/manifest.json"),
        produced_prompt_ids=("p0",),
        resumed_prompt_ids=(),
        shards=(object(),),
    )


def test_help_lists_recovery_capture_and_comparison_authorities(cli: Any) -> None:
    parser = cli._build_parser()
    subparsers = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )
    produce_help = subparsers.choices["produce"].format_help()
    compare_help = subparsers.choices["compare"].format_help()

    for option in (
        "--route-trace-root",
        "--recovery-dir",
        "--expected-seed-manifest-sha256",
        "--expected-stats-manifest-sha256",
        "--expected-full-source-blob-inventory-sha256",
        "--expected-routed-source-blob-inventory-sha256",
        "--expected-recovery-lever",
        "--expected-recovery-policy-json",
        "--expected-accepted-composite-audit-sha256",
        "--expected-recovery-candidate-identity-sha256",
        "--expected-recovery-manifest-body-sha256",
    ):
        assert option in produce_help
    for option in (
        "--expected-recovery-candidate-identity-sha256",
        "--expected-recovery-manifest-body-sha256",
        "--expected-accepted-composite-audit-sha256",
    ):
        assert option in compare_help


def test_recovered_produce_builds_contract_forwards_authority_and_reports_identity(
    cli: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy = {"rounding_objective": "minimize_weight_error", "selection": "accepted"}
    policy_path = tmp_path / "recovery-policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    calls: dict[str, Any] = {}
    teacher_contract = SimpleNamespace(bound_identity_sha256="teacher")
    recovered_contract = SimpleNamespace(bound_identity_sha256=RECOVERED_BOUND_IDENTITY)

    def build_from_teacher(*args: Any, **kwargs: Any) -> Any:
        calls["teacher"] = (args, kwargs)
        return teacher_contract

    def build_candidate(contract: Any, **kwargs: Any) -> Any:
        calls["contract"] = (contract, kwargs)
        return recovered_contract

    def produce(**kwargs: Any) -> Any:
        calls["produce"] = kwargs
        return _result()

    cli.candidate_api = SimpleNamespace(
        GLM52_CANDIDATE_KIND="production_composite",
        GLM52_RECOVERED_CANDIDATE_KIND="recovered_composite",
        GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256=BASELINE_IDENTITY,
        build_glm52_candidate_contract_from_teacher_cache=build_from_teacher,
        build_glm52_candidate_cache_contract=build_candidate,
        produce_glm52_candidate_cache=produce,
    )

    assert cli.run_cli(_recovery_produce_argv(tmp_path, policy_path)) == 0

    assert calls["contract"] == (
        teacher_contract,
        {
            "recovery_candidate_identity_sha256": RECOVERY_CANDIDATE_IDENTITY,
            "recovery_manifest_body_sha256": RECOVERY_MANIFEST_IDENTITY,
            "accepted_composite_audit_sha256": ACCEPTED_AUDIT_IDENTITY,
        },
    )
    forwarded = calls["produce"]
    assert forwarded["contract"] is recovered_contract
    assert forwarded["route_trace_root"] == "route-traces"
    assert forwarded["recovery_dir"] == "recovery"
    assert forwarded["expected_seed_manifest_sha256"] == "1" * 64
    assert forwarded["expected_stats_manifest_sha256"] == "2" * 64
    assert forwarded["expected_full_source_blob_inventory_sha256"] == "3" * 64
    assert forwarded["expected_routed_source_blob_inventory_sha256"] == "4" * 64
    assert forwarded["expected_recovery_lever"] == "reap_rounding"
    assert forwarded["expected_recovery_policy"] == policy
    assert forwarded["expected_composite_audit_sha256"] == ACCEPTED_AUDIT_IDENTITY
    assert (
        forwarded["expected_recovery_mixed_artifact_identity_sha256"]
        == RECOVERY_CANDIDATE_IDENTITY
    )
    assert forwarded["expected_recovery_manifest_body_sha256"] == RECOVERY_MANIFEST_IDENTITY

    output = json.loads(capsys.readouterr().out)
    assert output["candidate_kind"] == "recovered_composite"
    assert output["candidate_artifact_identity_sha256"] == RECOVERED_BOUND_IDENTITY
    assert output["recovery_candidate_identity_sha256"] == RECOVERY_CANDIDATE_IDENTITY
    assert output["recovery_manifest_body_sha256"] == RECOVERY_MANIFEST_IDENTITY
    assert output["accepted_composite_audit_sha256"] == ACCEPTED_AUDIT_IDENTITY


@pytest.mark.parametrize(
    "extra",
    [
        ["--route-trace-root", "route-traces"],
        ["--expected-recovery-candidate-identity-sha256", RECOVERY_CANDIDATE_IDENTITY],
    ],
)
def test_partial_produce_recovery_authority_is_rejected_before_build_or_produce(
    cli: Any,
    tmp_path: Path,
    extra: list[str],
) -> None:
    calls: list[str] = []
    cli.candidate_api = SimpleNamespace(
        build_glm52_candidate_contract_from_teacher_cache=lambda *_args, **_kwargs: calls.append("build"),
        produce_glm52_candidate_cache=lambda **_kwargs: calls.append("produce"),
    )

    with pytest.raises(ValueError, match="all recovery audit authorities"):
        cli.run_cli(_produce_argv(tmp_path) + extra)

    assert calls == []


@pytest.mark.parametrize(
    "contents, message",
    [
        ("[]", "must contain a JSON object"),
        ('{"selection":"first","selection":"second"}', "duplicate JSON key"),
        ('{"selection":NaN}', "non-finite JSON constant"),
    ],
)
def test_recovery_policy_is_loaded_as_a_strict_json_object_before_production(
    cli: Any,
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    policy_path = tmp_path / "recovery-policy.json"
    policy_path.write_text(contents, encoding="utf-8")
    calls: list[str] = []
    cli.candidate_api = SimpleNamespace(
        build_glm52_candidate_contract_from_teacher_cache=lambda *_args, **_kwargs: calls.append("build"),
        build_glm52_candidate_cache_contract=lambda *_args, **_kwargs: calls.append("contract"),
        produce_glm52_candidate_cache=lambda **_kwargs: calls.append("produce"),
    )

    with pytest.raises(ValueError, match=message):
        cli.run_cli(_recovery_produce_argv(tmp_path, policy_path))

    assert calls == []


def test_baseline_produce_preserves_contract_and_reports_baseline_identity(
    cli: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {}
    baseline_contract = SimpleNamespace(bound_identity_sha256=BASELINE_IDENTITY)

    def produce(**kwargs: Any) -> Any:
        calls.update(kwargs)
        return _result()

    cli.candidate_api = SimpleNamespace(
        GLM52_CANDIDATE_KIND="production_composite",
        GLM52_RECOVERED_CANDIDATE_KIND="recovered_composite",
        GLM52_CANDIDATE_ARTIFACT_IDENTITY_SHA256=BASELINE_IDENTITY,
        build_glm52_candidate_contract_from_teacher_cache=lambda *_args, **_kwargs: baseline_contract,
        build_glm52_candidate_cache_contract=lambda *_args, **_kwargs: pytest.fail(
            "baseline must not rebuild a recovered contract"
        ),
        produce_glm52_candidate_cache=produce,
    )

    assert cli.run_cli(_produce_argv(tmp_path)) == 0

    assert calls["contract"] is baseline_contract
    assert calls["route_trace_root"] is None
    assert calls["expected_recovery_policy"] is None
    output = json.loads(capsys.readouterr().out)
    assert output["candidate_kind"] == "production_composite"
    assert output["candidate_artifact_identity_sha256"] == BASELINE_IDENTITY
    assert output["recovery_candidate_identity_sha256"] is None
    assert output["recovery_manifest_body_sha256"] is None
    assert output["accepted_composite_audit_sha256"] is None


def test_recovered_compare_forwards_all_three_identities(cli: Any, tmp_path: Path) -> None:
    calls: dict[str, Any] = {}

    def compare(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["compare"] = (args, kwargs)
        return {"family_eval_gate_pass": True, "checks": {"identity": True}}

    cli.candidate_api = SimpleNamespace(compare_glm52_candidate_caches=compare)
    argv = _compare_argv(tmp_path) + [
        "--expected-recovery-candidate-identity-sha256",
        RECOVERY_CANDIDATE_IDENTITY,
        "--expected-recovery-manifest-body-sha256",
        RECOVERY_MANIFEST_IDENTITY,
        "--expected-accepted-composite-audit-sha256",
        ACCEPTED_AUDIT_IDENTITY,
    ]

    assert cli.run_cli(argv) == 0

    assert calls["compare"][1]["expected_recovery_candidate_identity_sha256"] == (
        RECOVERY_CANDIDATE_IDENTITY
    )
    assert calls["compare"][1]["expected_recovery_manifest_body_sha256"] == (
        RECOVERY_MANIFEST_IDENTITY
    )
    assert calls["compare"][1]["expected_accepted_composite_audit_sha256"] == (
        ACCEPTED_AUDIT_IDENTITY
    )


def test_partial_compare_recovery_identity_is_rejected_before_compare(
    cli: Any,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    cli.candidate_api = SimpleNamespace(
        compare_glm52_candidate_caches=lambda *_args, **_kwargs: calls.append("compare")
    )

    with pytest.raises(ValueError, match="all three recovered comparison identities"):
        cli.run_cli(
            _compare_argv(tmp_path)
            + [
                "--expected-recovery-candidate-identity-sha256",
                RECOVERY_CANDIDATE_IDENTITY,
            ]
        )

    assert calls == []


def test_baseline_compare_forwards_no_recovered_identity(cli: Any, tmp_path: Path) -> None:
    calls: dict[str, Any] = {}

    def compare(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.update(kwargs)
        return {"family_eval_gate_pass": True, "checks": {"identity": True}}

    cli.candidate_api = SimpleNamespace(compare_glm52_candidate_caches=compare)

    assert cli.run_cli(_compare_argv(tmp_path)) == 0

    assert calls["expected_recovery_candidate_identity_sha256"] is None
    assert calls["expected_recovery_manifest_body_sha256"] is None
    assert calls["expected_accepted_composite_audit_sha256"] is None
