from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks import bench_dsv4_mtp_headline as benchmark
from benchmarks.bench_dsv4_mtp_headline import summarize_dsv4_mtp_rows
from mlx_vq.benchmark import metrics

FIX1_ROOT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/benchmarks/dsv4-task4-fix1-20260819"
)
BIND_PROOF = FIX1_ROOT / "bind-proof.json"
BIND_PROOF_SHA256 = "647501aa70a4531c3c0cedabc866025f7c2072c5c4bf671e70c83c8605df6115"
R3_COMPACT = FIX1_ROOT / "headline-quiet60-r3/headline-evidence.json"
R2_COMPACT = FIX1_ROOT / "headline-quiet30-r2/headline-evidence.json"


def _authority_args():
    return {
        "bind_proof_path": BIND_PROOF,
        "expected_bind_proof_sha256": BIND_PROOF_SHA256,
    }


def _artifact_identity():
    return copy.deepcopy(json.loads(BIND_PROOF.read_text())["artifact_identity"])


@pytest.fixture
def report_pack(tmp_path):
    rows = []
    for prompt_id, length in (("long-report", 96), ("short-report", 80)):
        token_ids = list(range(length))
        rows.append(
            {
                "prompt_id": prompt_id,
                "campaign_split": "report",
                "encoded_token_ids": token_ids,
                "token_count": length,
                "token_ids_sha256": benchmark._canonical_sha256(token_ids),
                "positions": [1],
                "target_token_ids": [2],
            }
        )
    pack = tmp_path / "report-pack.json"
    pack.write_text(
        json.dumps(
            {
                "record_type": "dsv4_coding_agent_corpus",
                "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
                "prompt_row_count": len(rows),
                "prompt_rows": rows,
            }
        )
    )
    return pack


def _dispatch(implementation: str, *, token_rows: int = 6):
    return {
        "token_rows": token_rows,
        "route_count": token_rows * 6,
        "has_lhs_indices": implementation == "nax_e8p_m32n64",
        "input_dims": 4096,
        "output_dims": 2048,
        "group_size": 512,
        "code_bits": 16,
        "strategy": "sorted_tiled" if implementation == "nax_e8p_m32n64" else "direct",
        "implementation": implementation,
    }


def _rows(pack: Path):
    rows = []
    pid = 100
    artifact_identity = _artifact_identity()
    artifact_identity_sha256 = benchmark._canonical_sha256(artifact_identity)
    _session, _prompt_tokens, prompt_meta = benchmark._prompt(pack, 64)
    prompt_token_ids = prompt_meta["prompt_token_ids"]
    prompt_sha256 = prompt_meta["prompt_sha256"]
    output_token_ids = list(range(100, 116))
    token_ids_sha256 = benchmark._canonical_sha256(output_token_ids)
    harness_identity = copy.deepcopy(
        json.loads(BIND_PROOF.read_text())["harness_identity"]
    )
    generation_config = {
        "campaign_split": "report",
        "holdout_used": False,
        "max_new_tokens": 16,
        "prompt_id": prompt_meta["prompt_id"],
        "prompt_sha256": prompt_sha256,
        "prompt_tokens": 64,
        "scoring": "greedy_argmax",
    }
    for pair, order in enumerate(
        ("baseline_first", "speculative_first", "baseline_first")
    ):
        for role, elapsed in (("baseline", 2.0), ("speculative", 1.0)):
            rows.append(
                {
                    "record_type": "dsv4_mtp_headline_row_v1",
                    "pair_id": pair,
                    "pair_order": order,
                    "role": role,
                    "pid": pid,
                    "fresh_process": True,
                    "artifact_identity": artifact_identity,
                    "artifact_identity_sha256": artifact_identity_sha256,
                    "harness_identity": harness_identity,
                    "harness_identity_sha256": benchmark._canonical_sha256(
                        harness_identity
                    ),
                    "generation_config": generation_config,
                    "generation_config_sha256": benchmark._canonical_sha256(
                        generation_config
                    ),
                    "pack": str(pack),
                    "pack_sha256": benchmark._sha256(pack),
                    "campaign_split": "report",
                    "holdout_used": False,
                    "prompt_id": prompt_meta["prompt_id"],
                    "prompt_token_ids": prompt_token_ids,
                    "prompt_tokens": len(prompt_token_ids),
                    "prompt_sha256": prompt_sha256,
                    "source_session_tokens": prompt_meta["source_session_tokens"],
                    "selection": prompt_meta["selection"],
                    "max_new_tokens": 16,
                    "token_ids": output_token_ids,
                    "token_ids_sha256": token_ids_sha256,
                    "elapsed_seconds": elapsed,
                    "emitted_tokens": 16,
                    "quiet": {
                        "attempts": 1,
                        "available": True,
                        "enabled": True,
                        "pageouts_delta": 0,
                        "quiet": True,
                        "swapouts_delta": 0,
                        "window_seconds": 5.0,
                    },
                    "pageouts_delta": 0,
                    "swapouts_delta": 0,
                    "thermal": {
                        "output": "no warnings",
                        "sha256": benchmark._canonical_sha256("no warnings"),
                    },
                    "wired_limit_environment_absent": True,
                    "speculative_stats": (
                        None
                        if role == "baseline"
                        else {
                            "drafted_tokens": 15,
                            "accepted_tokens": 3,
                            "rejected_tokens": 12,
                            "verify_passes": 12,
                            "emitted_tokens": 16,
                            "verify_kernel_calls": 3,
                            "acceptance_rate": 0.2,
                            "accepted_per_verify_pass": 0.25,
                            "emitted_per_verify_pass": 16 / 12,
                            "verify_kernel_implementations": [
                                "metal",
                                "nax_e8p_m32n64",
                            ],
                            "verify_kernel_dispatches": [
                                _dispatch("nax_e8p_m32n64"),
                                _dispatch("metal"),
                                _dispatch("metal", token_rows=5),
                            ],
                        }
                    ),
                }
            )
            pid += 1
    return rows


def _row_with_quiet(quiet, *, pageouts_delta=0, swapouts_delta=0):
    return {
        "quiet": quiet,
        "pageouts_delta": pageouts_delta,
        "swapouts_delta": swapouts_delta,
    }


def _active_quiet(**changes):
    quiet = {
        "enabled": True,
        "available": True,
        "quiet": True,
        "attempts": 1,
        "window_seconds": 5.0,
        "pageouts_delta": 0,
        "swapouts_delta": 0,
    }
    quiet.update(changes)
    return quiet


def _unavailable_quiet(**changes):
    quiet = {
        "enabled": True,
        "available": False,
        "quiet": False,
        "attempts": 1,
        "window_seconds": 5.0,
        "pageouts_delta": None,
        "swapouts_delta": None,
    }
    quiet.update(changes)
    return quiet


def test_headline_summary_requires_paired_clean_fresh_token_identical_rows(report_pack):
    result = summarize_dsv4_mtp_rows(
        _rows(report_pack), min_clean_pairs=3, **_authority_args()
    )
    assert result["clean_pairs"] == 3
    assert result["dirty_pairs"] == 0
    assert result["median_baseline_over_speculative"] == pytest.approx(2.0)
    assert result["token_parity"] is True
    assert result["acceptance_consistency"] is True
    assert result["verdict"] == "PASS"


def test_headline_summary_allows_metal_for_ineligible_shapes_when_nax_is_present(
    report_pack,
):
    rows = _rows(report_pack)

    result = summarize_dsv4_mtp_rows(rows, min_clean_pairs=3, **_authority_args())

    assert result["speculative_stats"]["verify_kernel_calls"] == 3
    assert result["speculative_stats"]["verify_kernel_implementations"] == [
        "metal",
        "nax_e8p_m32n64",
    ]


def test_headline_summary_fails_closed_on_zero_real_verify_calls(report_pack):
    rows = copy.deepcopy(_rows(report_pack))
    rows[1]["speculative_stats"]["verify_kernel_calls"] = 0
    rows[1]["speculative_stats"]["verify_kernel_dispatches"] = []
    with pytest.raises(ValueError, match="zero verify-kernel calls"):
        summarize_dsv4_mtp_rows(rows, min_clean_pairs=3, **_authority_args())


def test_headline_summary_fails_closed_when_artifact_identity_mapping_is_missing(
    report_pack,
):
    rows = copy.deepcopy(_rows(report_pack))
    del rows[0]["artifact_identity"]

    with pytest.raises(TypeError, match="artifact identity"):
        summarize_dsv4_mtp_rows(rows, min_clean_pairs=3, **_authority_args())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("negative", "nonnegative"),
        ("bool", "nonnegative"),
        ("float", "nonnegative"),
        ("missing", "nonnegative"),
        ("rate", "acceptance rate"),
        ("emission", "emitted token"),
    ],
)
def test_headline_summary_fails_closed_on_invalid_acceptance_counters(
    report_pack, mutation, message
):
    rows = copy.deepcopy(_rows(report_pack))
    stats = rows[1]["speculative_stats"]
    if mutation == "negative":
        stats["accepted_tokens"] = -1
        stats["rejected_tokens"] = 16
        stats["acceptance_rate"] = -1 / 15
        stats["accepted_per_verify_pass"] = -1 / 12
    elif mutation == "bool":
        stats["accepted_tokens"] = False
    elif mutation == "float":
        stats["accepted_tokens"] = 3.0
    elif mutation == "missing":
        del stats["accepted_tokens"]
    elif mutation == "rate":
        stats["acceptance_rate"] = 0.99
    else:
        stats["emitted_tokens"] = 15

    with pytest.raises(ValueError, match=message):
        summarize_dsv4_mtp_rows(rows, min_clean_pairs=3, **_authority_args())


def test_headline_summary_rejects_fake_nax_claim_with_empty_dispatch_trace(report_pack):
    rows = copy.deepcopy(_rows(report_pack))
    stats = rows[1]["speculative_stats"]
    stats["verify_kernel_calls"] = 3
    stats["verify_kernel_implementations"] = ["nax_e8p_m32n64"]
    stats["verify_kernel_dispatches"] = []

    with pytest.raises(ValueError, match="dispatch trace"):
        summarize_dsv4_mtp_rows(rows, min_clean_pairs=3, **_authority_args())


def test_compact_evidence_recomputes_counts_and_rejects_fake_nax(report_pack):
    evidence = benchmark.build_dsv4_mtp_compact_evidence(
        _rows(report_pack), min_clean_pairs=3, **_authority_args()
    )
    validated = benchmark.validate_dsv4_mtp_compact_evidence(
        evidence, min_clean_pairs=3, **_authority_args()
    )
    assert validated["verdict"] == "PASS"
    compact_stats = evidence["rows"][1]["speculative_stats"]
    compact_stats["verify_kernel_calls"] = 3
    compact_stats["verify_kernel_implementation_counts"] = {"nax_e8p_m32n64": 3}
    compact_stats["verify_kernel_unique_dispatches"] = []

    with pytest.raises(ValueError, match="compact dispatch trace"):
        benchmark.validate_dsv4_mtp_compact_evidence(
            evidence, min_clean_pairs=3, **_authority_args()
        )


def test_compact_evidence_rejects_internally_consistent_fake_receipt():
    evidence = json.loads(R3_COMPACT.read_text())
    fake_receipt = "f" * 64
    for row in evidence["rows"]:
        row["artifact_identity"]["payload_content_receipt_sha256"] = fake_receipt
        row["artifact_identity_sha256"] = benchmark._canonical_sha256(
            row["artifact_identity"]
        )
    evidence["summary"]["artifact_identity"] = evidence["rows"][0]["artifact_identity"]
    evidence["summary"]["artifact_identity_sha256"] = evidence["rows"][0][
        "artifact_identity_sha256"
    ]

    with pytest.raises(ValueError, match="bind proof"):
        benchmark.validate_dsv4_mtp_compact_evidence(
            evidence,
            min_clean_pairs=3,
            **_authority_args(),
        )


def test_raw_evidence_rejects_internally_consistent_fake_receipt(report_pack):
    rows = _rows(report_pack)
    for row in rows:
        row["artifact_identity"]["payload_content_receipt_sha256"] = "f" * 64
        row["artifact_identity_sha256"] = benchmark._canonical_sha256(
            row["artifact_identity"]
        )

    with pytest.raises(ValueError, match="bind proof"):
        summarize_dsv4_mtp_rows(rows, min_clean_pairs=3, **_authority_args())


def test_compact_evidence_requires_expected_bind_proof_file_sha256():
    evidence = json.loads(R3_COMPACT.read_text())

    with pytest.raises(ValueError, match="bind proof file SHA-256"):
        benchmark.validate_dsv4_mtp_compact_evidence(
            evidence,
            min_clean_pairs=3,
            bind_proof_path=BIND_PROOF,
            expected_bind_proof_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("pageouts_delta", False),
        ("swapouts_delta", 0.0),
        ("pageouts_delta", -1),
        ("swapouts_delta", None),
    ],
)
def test_headline_summary_rejects_non_integer_or_missing_vm_deltas(
    report_pack, key, value
):
    rows = _rows(report_pack)
    if value is None:
        del rows[0][key]
    else:
        rows[0][key] = value

    with pytest.raises(ValueError, match="nonnegative integer"):
        summarize_dsv4_mtp_rows(rows, min_clean_pairs=3, **_authority_args())


@pytest.mark.parametrize(
    "mutation",
    ["not_mapping", "missing_counter", "bool_counter", "float_counter", "negative"],
)
def test_headline_summary_rejects_malformed_quiet_records(report_pack, mutation):
    rows = _rows(report_pack)
    quiet = rows[0]["quiet"]
    if mutation == "not_mapping":
        rows[0]["quiet"] = None
    elif mutation == "missing_counter":
        del quiet["attempts"]
    elif mutation == "bool_counter":
        quiet["attempts"] = False
    elif mutation == "float_counter":
        quiet["pageouts_delta"] = 0.0
    else:
        quiet["swapouts_delta"] = -1

    with pytest.raises((TypeError, ValueError), match="quiet"):
        summarize_dsv4_mtp_rows(rows, min_clean_pairs=3, **_authority_args())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("pageouts_delta", False),
        ("swapouts_delta", 0.0),
        ("quiet", None),
    ],
)
def test_dirty_baseline_does_not_hide_malformed_speculative_row(
    report_pack, key, value
):
    rows = _rows(report_pack)
    rows[0]["pageouts_delta"] = 1
    rows[1][key] = value

    with pytest.raises((TypeError, ValueError), match="headline"):
        summarize_dsv4_mtp_rows(rows, min_clean_pairs=3, **_authority_args())


@pytest.mark.parametrize(
    "key",
    ["pageouts_delta", "swapouts_delta"],
)
def test_well_formed_failed_quiet_gate_is_retained_as_dirty(report_pack, key):
    rows = _rows(report_pack)
    rows[0][key] = 1

    result = summarize_dsv4_mtp_rows(rows, min_clean_pairs=3, **_authority_args())

    assert result["clean_pairs"] == 2
    assert result["dirty_pairs"] == 1
    assert result["verdict"] == "INCOMPLETE"


def test_row_clean_accepts_exact_disabled_producer_record():
    quiet = metrics.wait_for_memory_quiet(window_seconds=0, max_attempts=3)

    assert quiet == {"enabled": False}
    assert benchmark._row_clean(_row_with_quiet(quiet)) is False


def test_row_clean_accepts_exact_unavailable_producer_record(monkeypatch):
    monkeypatch.setattr(metrics, "collect_vm_stat_counts", lambda: None)
    monkeypatch.setattr(metrics.time, "sleep", lambda _seconds: None)

    quiet = metrics.wait_for_memory_quiet(window_seconds=5, max_attempts=3)

    assert quiet == {
        "enabled": True,
        "available": False,
        "quiet": False,
        "attempts": 1,
        "window_seconds": 5.0,
        "pageouts_delta": None,
        "swapouts_delta": None,
    }
    assert benchmark._row_clean(_row_with_quiet(quiet)) is False


def test_row_clean_accepts_exact_active_dirty_producer_record(monkeypatch):
    snapshots = iter(
        [
            {"pageouts": 10, "swapouts": 20},
            {"pageouts": 11, "swapouts": 20},
        ]
    )
    monkeypatch.setattr(metrics, "collect_vm_stat_counts", lambda: next(snapshots))
    monkeypatch.setattr(metrics.time, "sleep", lambda _seconds: None)

    quiet = metrics.wait_for_memory_quiet(window_seconds=5, max_attempts=1)

    assert quiet == {
        "enabled": True,
        "available": True,
        "quiet": False,
        "attempts": 1,
        "window_seconds": 5.0,
        "pageouts_delta": 1,
        "swapouts_delta": 0,
    }
    assert benchmark._row_clean(_row_with_quiet(quiet)) is False


@pytest.mark.parametrize(
    "quiet",
    [
        {"enabled": False, "available": False},
        _unavailable_quiet(quiet=True),
        _unavailable_quiet(attempts=0),
        _unavailable_quiet(pageouts_delta=0),
        _active_quiet(pageouts_delta=1),
        _active_quiet(quiet=False),
        _active_quiet(unexpected=True),
    ],
)
def test_row_clean_rejects_impossible_quiet_state(quiet):
    with pytest.raises((TypeError, ValueError), match="quiet"):
        benchmark._row_clean(_row_with_quiet(quiet))


@pytest.mark.parametrize(
    ("quiet", "key", "value"),
    [
        ({"enabled": False}, "pageouts_delta", False),
        (_unavailable_quiet(), "swapouts_delta", 0.0),
    ],
)
def test_inactive_quiet_state_does_not_skip_timed_counter_validation(quiet, key, value):
    row = _row_with_quiet(quiet)
    row[key] = value

    with pytest.raises(ValueError, match="nonnegative integer"):
        benchmark._row_clean(row)


def test_r2_false_pageout_mutation_cannot_promote_incomplete_evidence_to_pass():
    evidence = json.loads(R2_COMPACT.read_text())
    dirty = next(row for row in evidence["rows"] if row["pageouts_delta"] == 69)
    dirty["pageouts_delta"] = False
    evidence["summary"].update(
        {
            "clean_pairs": 3,
            "dirty_pairs": 0,
            "orders": {"baseline_first": 2, "speculative_first": 1},
            "paired_baseline_over_speculative": [
                4.621752740058964,
                8.524364222270481,
                0.3748040037049363,
            ],
            "median_baseline_over_speculative": 4.621752740058964,
            "verdict": "PASS",
        }
    )

    with pytest.raises(ValueError, match="nonnegative integer"):
        benchmark.validate_dsv4_mtp_compact_evidence(
            evidence, min_clean_pairs=3, **_authority_args()
        )


def test_headline_summary_rejects_arbitrary_prompt_with_recomputed_hashes(report_pack):
    rows = _rows(report_pack)
    fake_prompt = [999] * 64
    fake_sha256 = benchmark._canonical_sha256(fake_prompt)
    for row in rows:
        row["prompt_id"] = "arbitrary-report-prefix"
        row["prompt_token_ids"] = fake_prompt
        row["prompt_sha256"] = fake_sha256
        row["generation_config"] = {
            **row["generation_config"],
            "prompt_id": row["prompt_id"],
            "prompt_sha256": fake_sha256,
        }
        row["generation_config_sha256"] = benchmark._canonical_sha256(
            row["generation_config"]
        )

    with pytest.raises(ValueError, match="prompt selection"):
        summarize_dsv4_mtp_rows(rows, min_clean_pairs=3, **_authority_args())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("source_session_tokens", 79),
        ("source_session_tokens", 80.0),
        ("selection", "arbitrary selection"),
        ("campaign_split", "holdout"),
        ("holdout_used", True),
        ("holdout_used", 0),
    ],
)
def test_headline_summary_rejects_prompt_policy_metadata_drift(report_pack, key, value):
    rows = _rows(report_pack)
    for row in rows:
        row[key] = value

    with pytest.raises(ValueError, match="prompt selection"):
        summarize_dsv4_mtp_rows(rows, min_clean_pairs=3, **_authority_args())
