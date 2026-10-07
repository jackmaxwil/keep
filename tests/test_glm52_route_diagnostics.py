from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "keep"
    / "quality"
    / "glm52_route_diagnostics.py"
)
POLICY_PATH = Path(__file__).parents[1] / "artifacts/quality/glm52-family-policy-20260709-v2.json"
PROMPT_PACK_SHA256 = "697677a4949f4ee7e370ac5e9a55e9631b0a1385f95a07dfd3a3f2b87d8edf31"
CANDIDATE_IDENTITY = "ef9d2e49d4a9d113a13d8b8e6c6ce7ebe60a7e9c7fb7b1b7784357d3efee5067"
BASELINE_IDENTITY = "d" * 64
CANDIDATE_CACHE_IDENTITY = "e" * 64
RECOVERY_ARTIFACT_IDENTITY = "f" * 64
RECOVERY_MANIFEST_BODY_SHA256 = "1" * 64


def _api():
    module_name = "_test_glm52_route_diagnostics"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _scores(n_valid: int) -> np.ndarray:
    row = np.array(
        [0.50, 0.45, 0.40, 0.35, 0.30, 0.20, 0.18, 0.12],
        dtype=np.float32,
    )
    assert float(row.sum(dtype=np.float64)) == pytest.approx(2.5)
    return np.tile(row, (n_valid, 1))


def _layers(*, permute_ranks: bool = False, bad_count: bool = False):
    api = _api()
    layers = []
    n_valid = 3
    base_ids = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
        ],
        dtype=np.int32,
    )
    if permute_ranks:
        base_ids = base_ids[:, [1, 0, 2, 3, 4, 5, 6, 7]]
    for layer_index in (3, 4):
        layers.append(
            api.RouteTraceLayer(
                layer_index=layer_index,
                expert_ids=base_ids + layer_index,
                scores=_scores(n_valid),
                valid_assignment_count=n_valid * 8 - int(bad_count),
                padded_assignment_count=0,
            )
        )
    return tuple(layers)


def _authority(tmp_path: Path, side: str, *, evidence_class: str = "release"):
    return {
        "teacher_cache_path": str(tmp_path / "teacher-cache.json"),
        "teacher_cache_content_sha256": "a" * 64,
        "candidate_composite_path": str(tmp_path / "candidate"),
        "candidate_composite_identity_sha256": CANDIDATE_IDENTITY,
        "prompt_pack_path": str(tmp_path / "prompts.json"),
        "prompt_pack_sha256": PROMPT_PACK_SHA256,
        "model_id": "zai-org/GLM-5.2",
        "model_revision": "pinned-revision",
        "producer_implementation_id": "keep.glm52.route-capture.v1",
        "capture_output_path": str(tmp_path / f"raw-{side}"),
        "capture_output_sha256": ("b" if side == "source" else "c") * 64,
        "evidence_class": evidence_class,
    }


def _recovery_candidate_authority(
    tmp_path: Path, *, evidence_class: str = "fixture_only"
):
    authority = _authority(tmp_path, "candidate", evidence_class=evidence_class)
    authority.pop("candidate_composite_path")
    authority.pop("candidate_composite_identity_sha256")
    authority.update(
        {
            "accepted_baseline_composite_path": str(tmp_path / "accepted-baseline"),
            "accepted_baseline_composite_identity_sha256": BASELINE_IDENTITY,
            "candidate_cache_identity_sha256": CANDIDATE_CACHE_IDENTITY,
            "recovery_mixed_artifact_identity_sha256": RECOVERY_ARTIFACT_IDENTITY,
            "recovery_manifest_body_sha256": RECOVERY_MANIFEST_BODY_SHA256,
        }
    )
    return authority


def _write_pair(tmp_path: Path, *, candidate_layers=None):
    api = _api()
    source_root = tmp_path / "source"
    candidate_root = tmp_path / "candidate"
    source_authority = _authority(tmp_path, "source")
    candidate_authority = _authority(tmp_path, "candidate")
    api.write_route_trace_artifact(
        source_root, _layers(), side="source", authority=source_authority
    )
    api.write_route_trace_artifact(
        candidate_root,
        candidate_layers if candidate_layers is not None else _layers(),
        side="candidate",
        authority=candidate_authority,
    )
    return source_root, candidate_root, source_authority, candidate_authority


def _compare(api, pair, **kwargs):
    source_root, candidate_root, source_authority, candidate_authority = pair
    return api.compare_route_trace_artifacts(
        source_root,
        candidate_root,
        expected_source_authority=source_authority,
        expected_candidate_authority=candidate_authority,
        **kwargs,
    )


def test_synthetic_route_traces_round_trip_as_authenticated_safetensors(
    tmp_path: Path,
) -> None:
    api = _api()
    root = tmp_path / "source"

    manifest = api.write_route_trace_artifact(
        root, _layers(), side="source", authority=_authority(tmp_path, "source")
    )
    audited = api.audit_route_trace_artifact(root, expected_side="source")

    assert manifest["record_type"] == "glm52_route_trace_v1"
    assert audited.manifest["manifest_body_sha256"] == manifest["manifest_body_sha256"]
    assert [layer.layer_index for layer in audited.layers] == [3, 4]
    assert all(layer.expert_ids.dtype == np.int32 for layer in audited.layers)
    assert all(layer.scores.dtype == np.float32 for layer in audited.layers)
    assert sorted(path.name for path in (root / "layers").iterdir()) == [
        "layer-00003.safetensors",
        "layer-00004.safetensors",
    ]


def test_v1_candidate_trace_remains_explicitly_supported(tmp_path: Path) -> None:
    api = _api()
    pair = _write_pair(tmp_path)

    evidence = _compare(api, pair)

    candidate_manifest = api.audit_route_trace_artifact(
        pair[1], expected_side="candidate"
    ).manifest
    assert candidate_manifest["schema_version"] == 1
    assert candidate_manifest["record_type"] == "glm52_route_trace_v1"
    assert evidence["checks"]["structural_math_pass"] is True


def test_v2_recovery_candidate_requires_distinct_authenticated_identities(
    tmp_path: Path,
) -> None:
    api = _api()
    root = tmp_path / "candidate-v2"
    authority = _recovery_candidate_authority(tmp_path)

    manifest = api.write_route_trace_artifact(
        root,
        _layers(),
        side="candidate",
        authority=authority,
        schema_version=2,
        created_at="2026-07-10T00:00:00Z",
    )
    audited = api.audit_route_trace_artifact(root, expected_side="candidate")

    assert manifest["schema_version"] == 2
    assert manifest["record_type"] == "glm52_recovery_candidate_route_trace_v2"
    assert audited.manifest["authority"] == authority
    assert set(authority) == set(api.RECOVERY_CANDIDATE_AUTHORITY_FIELDS)
    assert "candidate_composite_identity_sha256" not in authority
    body = dict(manifest)
    submitted_hash = body.pop("manifest_body_sha256")
    assert submitted_hash == api.canonical_sha256(body)


@pytest.mark.parametrize(
    "field",
    [
        "accepted_baseline_composite_identity_sha256",
        "candidate_cache_identity_sha256",
        "recovery_mixed_artifact_identity_sha256",
        "recovery_manifest_body_sha256",
    ],
)
def test_v2_recovery_candidate_rejects_missing_or_malformed_identity(
    tmp_path: Path, field: str
) -> None:
    api = _api()
    missing = _recovery_candidate_authority(tmp_path)
    missing.pop(field)
    with pytest.raises(ValueError, match="authority key inventory"):
        api.write_route_trace_artifact(
            tmp_path / f"missing-{field}",
            _layers(),
            side="candidate",
            authority=missing,
            schema_version=2,
        )

    malformed = _recovery_candidate_authority(tmp_path)
    malformed[field] = "A" * 64
    with pytest.raises(ValueError, match=field):
        api.write_route_trace_artifact(
            tmp_path / f"malformed-{field}",
            _layers(),
            side="candidate",
            authority=malformed,
            schema_version=2,
        )


def test_v2_schema_is_candidate_only(tmp_path: Path) -> None:
    api = _api()
    with pytest.raises(ValueError, match="candidate side"):
        api.write_route_trace_artifact(
            tmp_path / "source-v2",
            _layers(),
            side="source",
            authority=_recovery_candidate_authority(tmp_path),
            schema_version=2,
        )


def test_v2_expected_cache_and_recovery_identity_swaps_fail_structural_evidence(
    tmp_path: Path,
) -> None:
    api = _api()
    source_root = tmp_path / "source"
    candidate_root = tmp_path / "candidate-v2"
    source_authority = _authority(tmp_path, "source")
    candidate_authority = _recovery_candidate_authority(tmp_path)
    api.write_route_trace_artifact(
        source_root, _layers(), side="source", authority=source_authority
    )
    api.write_route_trace_artifact(
        candidate_root,
        _layers(),
        side="candidate",
        authority=candidate_authority,
        schema_version=2,
    )
    expected_candidate = dict(candidate_authority)
    expected_candidate["candidate_cache_identity_sha256"], expected_candidate[
        "recovery_mixed_artifact_identity_sha256"
    ] = (
        expected_candidate["recovery_mixed_artifact_identity_sha256"],
        expected_candidate["candidate_cache_identity_sha256"],
    )

    evidence = api.compare_route_trace_artifacts(
        source_root,
        candidate_root,
        expected_source_authority=source_authority,
        expected_candidate_authority=expected_candidate,
    )

    assert evidence["checks"]["candidate_capture_authority_pass"] is False
    assert evidence["checks"]["structural_math_pass"] is False


def test_v2_comparison_evidence_surfaces_recovery_lineage_and_tuning_scope(
    tmp_path: Path,
) -> None:
    api = _api()
    source_root = tmp_path / "source"
    candidate_root = tmp_path / "candidate-v2"
    source_authority = _authority(tmp_path, "source")
    candidate_authority = _recovery_candidate_authority(tmp_path)
    api.write_route_trace_artifact(
        source_root, _layers(), side="source", authority=source_authority
    )
    api.write_route_trace_artifact(
        candidate_root,
        _layers(),
        side="candidate",
        authority=candidate_authority,
        schema_version=2,
    )

    evidence = api.compare_route_trace_artifacts(
        source_root,
        candidate_root,
        expected_source_authority=source_authority,
        expected_candidate_authority=candidate_authority,
    )

    candidate_trace = evidence["candidate_trace"]
    assert candidate_trace["schema_version"] == 2
    assert candidate_trace["candidate_cache_identity_sha256"] == CANDIDATE_CACHE_IDENTITY
    assert (
        candidate_trace["recovery_mixed_artifact_identity_sha256"]
        == RECOVERY_ARTIFACT_IDENTITY
    )
    assert (
        candidate_trace["recovery_manifest_body_sha256"]
        == RECOVERY_MANIFEST_BODY_SHA256
    )
    assert (
        candidate_trace["accepted_baseline_composite_identity_sha256"]
        == BASELINE_IDENTITY
    )
    assert evidence["evaluation"]["recovery_tuning_scope"] == "per_layer_only"
    assert evidence["evaluation"]["global_route_metrics_allowed_for_recovery_tuning"] is False
    assert evidence["evaluation"]["diagnostic_only"] is True
    assert evidence["checks"]["release_gate_pass"] is False


def test_identical_traces_have_perfect_agreement_and_diagnostic_only_policy(
    tmp_path: Path,
) -> None:
    api = _api()
    pair = _write_pair(tmp_path)
    evidence = _compare(api, pair, policy_json=POLICY_PATH)

    assert evidence["metrics"]["global"]["top8_set_agreement"] == 1.0
    assert evidence["metrics"]["global"]["rank_exact_agreement"] == 1.0
    assert evidence["metrics"]["global"]["score_correlation"] == 1.0
    assert evidence["checks"]["structural_math_pass"] is True
    assert evidence["evaluation"]["diagnostic_only"] is True
    assert evidence["checks"]["release_thresholds_pass"] is False
    assert len(evidence["evidence_body_sha256"]) == 64


def test_permuted_ranks_keep_set_agreement_but_reduce_rank_agreement(
    tmp_path: Path,
) -> None:
    api = _api()
    pair = _write_pair(
        tmp_path,
        candidate_layers=_layers(permute_ranks=True),
    )

    evidence = _compare(api, pair)

    assert evidence["metrics"]["global"]["top8_set_agreement"] == 1.0
    assert evidence["metrics"]["global"]["rank_exact_agreement"] < 1.0


def test_assignment_count_violation_fails_structural_math_check(tmp_path: Path) -> None:
    api = _api()
    pair = _write_pair(
        tmp_path,
        candidate_layers=_layers(bad_count=True),
    )

    evidence = _compare(api, pair)

    candidate = evidence["checks"]["candidate"]
    assert candidate["assignment_count_pass"] is False
    assert evidence["checks"]["structural_math_pass"] is False
    assert evidence["checks"]["diagnostics_pass"] is False


def test_mismatched_capture_authority_fails_structural_comparison(tmp_path: Path) -> None:
    api = _api()
    source_root = tmp_path / "source"
    candidate_root = tmp_path / "candidate"
    source_authority = _authority(tmp_path, "source")
    candidate_authority = _authority(tmp_path, "candidate")
    api.write_route_trace_artifact(source_root, _layers(), side="source", authority=source_authority)
    api.write_route_trace_artifact(
        candidate_root,
        _layers(),
        side="candidate",
        authority=candidate_authority,
    )

    wrong_candidate_authority = dict(candidate_authority)
    wrong_candidate_authority["capture_output_sha256"] = "d" * 64
    evidence = api.compare_route_trace_artifacts(
        source_root,
        candidate_root,
        expected_source_authority=source_authority,
        expected_candidate_authority=wrong_candidate_authority,
    )

    assert evidence["checks"]["capture_authority_match_pass"] is False
    assert evidence["checks"]["structural_math_pass"] is False


def test_tampered_trace_file_fails_authentication(tmp_path: Path) -> None:
    api = _api()
    root = tmp_path / "source"
    api.write_route_trace_artifact(
        root, _layers(), side="source", authority=_authority(tmp_path, "source")
    )
    layer_path = root / "layers" / "layer-00003.safetensors"
    raw = bytearray(layer_path.read_bytes())
    raw[-1] ^= 1
    layer_path.write_bytes(raw)

    with pytest.raises(ValueError, match="SHA-256"):
        api.audit_route_trace_artifact(root)


def test_candidate_gate_tuple_adapter_preserves_ids_scores_and_counts() -> None:
    api = _api()
    ids = np.arange(16, dtype=np.int32).reshape(2, 8)
    scores = _scores(2)

    trace = api.candidate_gate_output_to_trace(7, (ids, scores))

    np.testing.assert_array_equal(trace.expert_ids, ids)
    np.testing.assert_array_equal(trace.scores, scores)
    assert trace.valid_assignment_count == 16
    assert trace.padded_assignment_count == 0


def test_writer_rejects_empty_capture_authority(tmp_path: Path) -> None:
    api = _api()
    with pytest.raises(ValueError, match="authority key inventory"):
        api.write_route_trace_artifact(tmp_path / "trace", _layers(), side="source")


def test_caller_supplied_route_thresholds_policy_is_rejected(tmp_path: Path) -> None:
    api = _api()
    policy = json.loads(POLICY_PATH.read_text())
    policy["route_math_diagnostics"] = {"top8_set_agreement_min": 0.0}
    policy_path = tmp_path / "invented-policy.json"
    policy_path.write_text(json.dumps(policy))

    with pytest.raises(ValueError, match="pinned file SHA-256"):
        _compare(api, _write_pair(tmp_path), policy_json=policy_path)


def test_cli_captures_both_sides_and_writes_comparison_evidence(
    tmp_path: Path,
) -> None:
    cli = Path(__file__).parents[1] / "benchmarks" / "check_glm52_route_math.py"
    raw_source = tmp_path / "raw-source"
    raw_candidate = tmp_path / "raw-candidate"
    raw_source.mkdir()
    raw_candidate.mkdir()
    layer = _layers()[0]
    for root in (raw_source, raw_candidate):
        np.savez(
            root / "layer-00003.npz",
            expert_ids=layer.expert_ids,
            scores=layer.scores,
            valid_assignment_count=np.array(layer.valid_assignment_count),
            padded_assignment_count=np.array(layer.padded_assignment_count),
        )
    source_root = tmp_path / "source"
    candidate_root = tmp_path / "candidate"
    evidence_path = tmp_path / "evidence.json"

    for side, input_root, trace_root in (
        ("source", raw_source, source_root),
        ("candidate", raw_candidate, candidate_root),
    ):
        authority_path = tmp_path / f"{side}-authority.json"
        authority_path.write_text(
            json.dumps(_authority(tmp_path, side, evidence_class="fixture_only"))
        )
        subprocess.run(
            [
                sys.executable,
                str(cli),
                "capture",
                "--side",
                side,
                "--input-root",
                str(input_root),
                "--trace-root",
                str(trace_root),
                "--authority-json",
                str(authority_path),
                "--allow-nonfrozen-fixture",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(cli),
            "compare",
            "--source-trace-root",
            str(source_root),
            "--candidate-trace-root",
            str(candidate_root),
            "--source-authority-json",
            str(tmp_path / "source-authority.json"),
            "--candidate-authority-json",
            str(tmp_path / "candidate-authority.json"),
            "--evidence-json",
            str(evidence_path),
            "--allow-nonfrozen-fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    evidence = json.loads(evidence_path.read_text())
    assert evidence["checks"]["diagnostics_pass"] is True
    assert evidence["evaluation"]["diagnostic_only"] is True
    assert evidence["evaluation"]["release_eligible"] is False
    assert evidence["checks"]["release_gate_pass"] is False


def test_cli_capture_serializes_v2_recovery_candidate_schema(tmp_path: Path) -> None:
    api = _api()
    cli = Path(__file__).parents[1] / "benchmarks" / "check_glm52_route_math.py"
    raw_candidate = tmp_path / "raw-candidate"
    raw_candidate.mkdir()
    layer = _layers()[0]
    np.savez(
        raw_candidate / "layer-00003.npz",
        expert_ids=layer.expert_ids,
        scores=layer.scores,
        valid_assignment_count=np.array(layer.valid_assignment_count),
        padded_assignment_count=np.array(layer.padded_assignment_count),
    )
    authority_path = tmp_path / "candidate-v2-authority.json"
    authority_path.write_text(json.dumps(_recovery_candidate_authority(tmp_path)))
    trace_root = tmp_path / "candidate-v2"

    subprocess.run(
        [
            sys.executable,
            str(cli),
            "capture",
            "--side",
            "candidate",
            "--input-root",
            str(raw_candidate),
            "--trace-root",
            str(trace_root),
            "--authority-json",
            str(authority_path),
            "--schema-version",
            "2",
            "--allow-nonfrozen-fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = api.audit_route_trace_artifact(
        trace_root, expected_side="candidate"
    ).manifest
    assert manifest["schema_version"] == 2
    assert manifest["authority"]["candidate_cache_identity_sha256"] == CANDIDATE_CACHE_IDENTITY
