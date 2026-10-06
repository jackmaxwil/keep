from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "benchmarks/prove_glm52_source_teacher_bounded.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "prove_glm52_source_teacher_bounded",
        CLI_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arg_parsing_supports_pinned_defaults_and_incremental_steps(tmp_path: Path) -> None:
    cli = _load_cli()

    args = cli.parse_args(
        [
            "--output-json",
            str(tmp_path / "proof.json"),
            "--skip-steps",
            "9,11",
        ]
    )

    assert args.output_json == str(tmp_path / "proof.json")
    assert args.profile_yaml.endswith("models/glm52-reap-504b-v2.yaml")
    assert args.prompt_pack_json.endswith(
        "artifacts/quality/glm52-family-eval-prompts-20260709-v2.json"
    )
    assert cli.select_steps(args.only_step, args.skip_steps) == (10, 12)

    only = cli.parse_args(
        ["--output-json", str(tmp_path / "step-11.json"), "--only-step", "11"]
    )
    assert cli.select_steps(only.only_step, only.skip_steps) == (11,)


def test_evidence_schema_has_all_proofs_phases_and_pinned_identities(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    args = cli.parse_args(
        ["--output-json", str(tmp_path / "proof.json"), "--only-step", "9"]
    )

    evidence = cli.new_evidence(args, selected_steps=(9,))

    assert evidence["schema_version"] == 2
    assert evidence["record_type"] == "glm52_source_teacher_bounded_proof_v1"
    assert evidence["oracle_independence"] == {
        "independent_moe_oracle": True,
        "independent_route_slot_scatter": True,
        "independent_route_rank_reduction": True,
        "checkpoint_resume_from_persisted_bytes": True,
        "shared_attention_modules": True,
        "shared_norm_and_lm_head_modules": True,
        "shared_embedding_and_dense_mlp_modules": True,
        "shared_router_and_shared_expert_modules": True,
        "shared_expert_weight_resolver": True,
    }
    assert evidence["bounded_proof_pass"] is False
    assert evidence["requested_steps"] == [9]
    assert evidence["ordered_steps"] == [9, 10, 11, 12]
    assert evidence["phases"] == []
    assert list(evidence["proofs"]) == ["step_9", "step_10", "step_11", "step_12"]
    assert evidence["proofs"]["step_9"]["proof_pass"] is None
    assert evidence["proofs"]["step_10"]["status"] == "skipped"
    assert evidence["pinned_identities"]["model_id"] == cli.MODEL_ID
    assert evidence["pinned_identities"]["revision"] == cli.REVISION
    assert evidence["pinned_identities"]["config_sha256"] == cli.CONFIG_SHA256
    assert evidence["pinned_identities"]["index_sha256"] == cli.INDEX_SHA256
    assert evidence["pinned_identities"]["profile_sha256"] == cli.PROFILE_SHA256
    assert (
        evidence["pinned_identities"]["profile_contract_sha256"]
        == cli.PROFILE_CONTRACT_SHA256
    )

    encoded = json.dumps(evidence, sort_keys=True, allow_nan=False)
    assert json.loads(encoded) == evidence


def test_oracle_mismatch_fails_loud_with_fake_resolver() -> None:
    cli = _load_cli()
    oracle = cli.DecodedOracle(
        layer=10,
        expert=0,
        projection="gate_proj",
        sha256="0" * 64,
    )

    with pytest.raises(cli.OracleMismatchError, match="layer 10 expert 0 gate_proj"):
        cli.verify_decoded_oracles(
            (oracle,),
            decoded_bytes_resolver=lambda _oracle: b"definitely-not-the-oracle",
        )


def test_step_selection_and_execution_preserve_proof_ladder_order() -> None:
    cli = _load_cli()

    assert cli.select_steps(None, "12,10") == (9, 11)
    assert cli.select_steps(11, "") == (11,)
    with pytest.raises(ValueError, match="cannot also be skipped"):
        cli.select_steps(11, "11")

    called: list[int] = []
    results = cli.execute_selected_steps(
        (9, 11, 12),
        {
            12: lambda: called.append(12) or {"proof_pass": True},
            9: lambda: called.append(9) or {"proof_pass": True},
            11: lambda: called.append(11) or {"proof_pass": True},
        },
    )

    assert called == [9, 11, 12]
    assert list(results) == [9, 11, 12]


def test_independent_route_slot_oracle_catches_buggy_shared_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli()
    route_outputs = np.array([[[2.0], [10.0]]], dtype=np.float32)
    route_scores = np.array([[0.25, 0.75]], dtype=np.float32)

    def buggy_shared_reduction(outputs: np.ndarray, scores: np.ndarray) -> np.ndarray:
        del scores
        return outputs.sum(axis=1, dtype=np.float32)

    monkeypatch.setattr(
        cli,
        "_production_route_reduction_for_fixture",
        buggy_shared_reduction,
    )

    with pytest.raises(AssertionError, match="independent route-slot oracle"):
        cli.verify_route_reduction_fixture(route_outputs, route_scores)


def test_checkpoint_probe_interrupts_then_resumes_strictly_from_bytes(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    checkpoint = tmp_path / "layer-00040.bin"
    events: list[str] = []
    live_hidden = {"value": b"not-persisted"}

    def interrupting_run() -> None:
        checkpoint.write_bytes(b"persisted-layer-40")
        live_hidden["value"] = b"in-memory-layer-40"
        events.append("interrupted")
        raise cli.SimulatedCheckpointInterruption("after layer 40")

    def clear_live_state() -> None:
        live_hidden.clear()
        events.append("cleared")

    def resumed_run() -> bytes:
        assert live_hidden == {}
        events.append("resumed-from-bytes")
        return checkpoint.read_bytes()

    result = cli.run_checkpoint_reload_probe(
        interrupting_run=interrupting_run,
        resumed_run=resumed_run,
        interruption_type=cli.SimulatedCheckpointInterruption,
        clear_live_state=clear_live_state,
    )

    assert result == b"persisted-layer-40"
    assert events == ["interrupted", "cleared", "resumed-from-bytes"]
