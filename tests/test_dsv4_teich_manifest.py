"""Validate the committed DeepSeek-V4-Flash teich split manifest.

These tests read only the committed manifest JSON. They never load a tokenizer
and never touch the ~90 MB prompt pack, so they run anywhere the repo is
checked out. The manifest is produced by ``scripts/retokenize_teich_dsv4.py``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "recipes" / "dsv4_teich_split_manifest_v1_20260811.json"
)

EXPECTED_SPLITS = ("calibration", "mtp-train", "report", "selection", "holdout")
TRAIN_CAMPAIGN_SPLITS = frozenset({"calibration", "mtp-train"})
EVAL_CAMPAIGN_SPLITS = frozenset({"report", "selection", "holdout"})
V4_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
V4_TOKENIZER_SHA256 = "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"
GLM_TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    assert MANIFEST_PATH.exists(), f"missing committed manifest at {MANIFEST_PATH}"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_loads_with_required_structure(manifest: dict[str, Any]) -> None:
    assert manifest["record_type"] == "dsv4_teich_split_manifest"
    assert manifest["manifest_version"] == "v2"
    assert manifest["wave"] == 1
    assert manifest["pack_model_id"] == V4_MODEL_ID
    for key in ("pack_file", "pack_file_sha256", "retokenization", "totals", "splits", "rows"):
        assert key in manifest, f"manifest missing {key}"
    assert tuple(manifest["splits"]) == EXPECTED_SPLITS
    assert isinstance(manifest["exclusions"], list)


def test_all_sha256_fields_are_well_formed(manifest: dict[str, Any]) -> None:
    assert _is_sha256(manifest["pack_file_sha256"])
    retokenization = manifest["retokenization"]
    assert _is_sha256(retokenization["source_pack_sha256"])
    for name, digest in retokenization["glm_tokenizer"]["sha256"].items():
        assert _is_sha256(digest), f"glm {name}"
    for name, digest in retokenization["v4_tokenizer"]["sha256"].items():
        assert _is_sha256(digest), f"v4 {name}"
    assert retokenization["v4_tokenizer"]["sha256"]["tokenizer.json"] == V4_TOKENIZER_SHA256
    assert retokenization["glm_tokenizer"]["sha256"]["tokenizer.json"] == GLM_TOKENIZER_SHA256
    assert retokenization["v4_tokenizer"]["vocab_size"] == 129280
    assert retokenization["glm_tokenizer"]["vocab_size"] == 154856
    assert retokenization["glm_tokenizer"]["model_vocab_size"] == 154880
    for row in manifest["rows"]:
        assert _is_sha256(row["token_ids_sha256"]), row["prompt_id"]


def test_roundtrip_gate_is_exact_equality(manifest: dict[str, Any]) -> None:
    """Fix E: the tolerance is 0.0, and every session actually hit it."""
    assert manifest["retokenization"]["max_char_divergence"] == 0
    assert manifest["totals"]["max_char_divergence_observed"] == 0
    for row in manifest["rows"]:
        assert row["roundtrip_char_divergence"] == 0, row["prompt_id"]


def test_no_eval_split_row_is_tuning_eligible(manifest: dict[str, Any]) -> None:
    """Fix D: the holdout-leakage landmine must be disarmed.

    A stale consumer gating on ``split == "train" and tuning_eligible`` must not
    be handed a report/selection/holdout row.
    """
    for row in manifest["rows"]:
        prompt_id = row["prompt_id"]
        if row["campaign_split"] in EVAL_CAMPAIGN_SPLITS:
            assert row["tuning_eligible"] is False, f"{prompt_id} leaks: tuning_eligible"
            assert row["split"] != "train", f"{prompt_id} leaks: split={row['split']!r}"
        else:
            assert row["campaign_split"] in TRAIN_CAMPAIGN_SPLITS
            assert row["tuning_eligible"] is True, prompt_id
            assert row["split"] == "train", prompt_id

    eligible = [row for row in manifest["rows"] if row["tuning_eligible"]]
    assert eligible, "no row is tuning eligible; the pack would train on nothing"
    assert {row["campaign_split"] for row in eligible} == TRAIN_CAMPAIGN_SPLITS


def test_split_fields_are_recomputed_not_inherited(manifest: dict[str, Any]) -> None:
    """Fix D: GLM-era values survive only under non-gating names."""
    policy = manifest["retokenization"]["split_field_policy"]
    assert sorted(policy["train_campaign_splits"]) == sorted(TRAIN_CAMPAIGN_SPLITS)
    assert policy["train_split_label"] == "train"
    assert policy["eval_split_label"] == "holdout"
    assert policy["glm_era_fields_preserved"] == ["glm_era_split", "glm_era_tuning_eligible"]
    assert policy["consumer_gates_checked"]

    # The rename is load-bearing: the GLM-era labels really do disagree with the
    # recomputed ones, which is exactly why inheriting them would have leaked.
    disagreements = [
        row
        for row in manifest["rows"]
        if row["glm_era_tuning_eligible"] != row["tuning_eligible"]
    ]
    assert disagreements, "expected GLM-era eligibility to differ from the campaign's"
    for row in manifest["rows"]:
        assert isinstance(row["glm_era_split"], str) and row["glm_era_split"]
        assert isinstance(row["glm_era_tuning_eligible"], bool)


def test_streams_are_canonical_v4_encodings(manifest: dict[str, Any]) -> None:
    """Fix C: literal-text folding leaves no off-distribution tokenization drift."""
    totals = manifest["totals"]
    assert totals["canonical_stream_sessions"] == totals["sessions_included"]
    assert totals["non_canonical_token_overhead"] == 0
    assert totals["v4_canonical_raw_tokens"] == totals["v4_raw_tokens"]
    assert manifest["retokenization"]["literal_text_folding"]
    for row in manifest["rows"]:
        assert row["v4_canonical_token_count"] == row["v4_token_count"], row["prompt_id"]


def test_splits_are_disjoint_and_cover_every_included_row(manifest: dict[str, Any]) -> None:
    row_ids = [row["prompt_id"] for row in manifest["rows"]]
    assert len(row_ids) == len(set(row_ids)), "duplicate prompt_id in rows"

    seen: set[str] = set()
    for name in EXPECTED_SPLITS:
        members = manifest["splits"][name]["prompt_ids"]
        assert len(members) == len(set(members)), f"duplicate prompt_id inside {name}"
        overlap = seen & set(members)
        assert not overlap, f"{name} overlaps an earlier split: {sorted(overlap)[:5]}"
        seen |= set(members)

    assert seen == set(row_ids), "split membership does not exactly cover the row list"
    for row in manifest["rows"]:
        assert row["campaign_split"] in EXPECTED_SPLITS
        assert row["prompt_id"] in manifest["splits"][row["campaign_split"]]["prompt_ids"]


def test_no_source_session_spans_two_splits(manifest: dict[str, Any]) -> None:
    by_session: dict[str, set[str]] = defaultdict(set)
    for row in manifest["rows"]:
        by_session[row["source_session_id"]].add(row["campaign_split"])
    straddling = {key: sorted(value) for key, value in by_session.items() if len(value) > 1}
    assert not straddling, f"sessions span multiple splits: {straddling}"


def test_counts_match_manifest_totals(manifest: dict[str, Any]) -> None:
    totals = manifest["totals"]
    rows = manifest["rows"]

    assert totals["sessions_included"] == len(rows)
    assert totals["sessions_excluded"] == len(manifest["exclusions"])
    assert totals["sessions_in"] == totals["sessions_included"] + totals["sessions_excluded"]

    assert totals["v4_raw_tokens"] == sum(row["v4_token_count"] for row in rows)
    assert totals["v4_supervised_tokens"] == sum(row["v4_supervised_tokens"] for row in rows)
    assert totals["glm_raw_tokens"] == sum(row["glm_token_count"] for row in rows)
    assert totals["glm_supervised_tokens"] == sum(row["glm_supervised_tokens"] for row in rows)

    for name in EXPECTED_SPLITS:
        block = manifest["splits"][name]
        members = [row for row in rows if row["campaign_split"] == name]
        assert block["sessions"] == len(members) == len(block["prompt_ids"])
        assert block["v4_raw_tokens"] == sum(row["v4_token_count"] for row in members)
        assert block["v4_supervised_tokens"] == sum(row["v4_supervised_tokens"] for row in members)
        assert block["glm_raw_tokens"] == sum(row["glm_token_count"] for row in members)
        assert block["glm_supervised_tokens"] == sum(
            row["glm_supervised_tokens"] for row in members
        )

    assert sum(manifest["splits"][name]["sessions"] for name in EXPECTED_SPLITS) == len(rows)
    assert totals["v4_raw_ratio"] == pytest.approx(
        totals["v4_raw_tokens"] / totals["glm_raw_tokens"]
    )
    assert totals["v4_supervised_ratio"] == pytest.approx(
        totals["v4_supervised_tokens"] / totals["glm_supervised_tokens"]
    )


def test_split_sessions_match_declared_targets(manifest: dict[str, Any]) -> None:
    """With all 257 sessions surviving, quotas are hit exactly."""
    if manifest["totals"]["sessions_excluded"]:
        pytest.skip("quotas are scaled proportionally when sessions are excluded")
    for name in EXPECTED_SPLITS:
        block = manifest["splits"][name]
        assert block["sessions"] == block["target_sessions"], name


def test_per_row_token_counts_are_sane(manifest: dict[str, Any]) -> None:
    for row in manifest["rows"]:
        prompt_id = row["prompt_id"]
        assert row["v4_token_count"] > 1, prompt_id
        assert row["glm_token_count"] > 1, prompt_id
        assert 0 < row["v4_supervised_tokens"] < row["v4_token_count"], prompt_id
        assert 0 < row["glm_supervised_tokens"] < row["glm_token_count"], prompt_id
        assert row["roundtrip_char_divergence"] <= manifest["retokenization"][
            "max_char_divergence"
        ], prompt_id


def test_split_blocks_declare_their_gate_values(manifest: dict[str, Any]) -> None:
    for name in EXPECTED_SPLITS:
        block = manifest["splits"][name]
        expected_eligible = name in TRAIN_CAMPAIGN_SPLITS
        assert block["tuning_eligible"] is expected_eligible, name
        assert block["split_label"] == ("train" if expected_eligible else "holdout"), name
        members = [row for row in manifest["rows"] if row["campaign_split"] == name]
        assert block["sessions_over_65536_v4"] == sum(
            1 for row in members if row["v4_token_count"] > 65536
        ), name
        assert all(row["split"] == block["split_label"] for row in members), name


def test_context_window_overflow_is_recorded(manifest: dict[str, Any]) -> None:
    """The >64K overflow is a live campaign risk; keep it visible in the manifest."""
    totals = manifest["totals"]
    rows = manifest["rows"]
    assert totals["sessions_over_65536_v4"] == sum(
        1 for row in rows if row["v4_token_count"] > 65536
    )
    assert totals["sessions_over_65536_glm"] == sum(
        1 for row in rows if row["glm_token_count"] > 65536
    )
    assert totals["v4_session_tokens_max"] == max(row["v4_token_count"] for row in rows)
    assert totals["sessions_over_65536_v4"] == sum(
        manifest["splits"][name]["sessions_over_65536_v4"] for name in EXPECTED_SPLITS
    )


def test_exclusions_carry_a_reason(manifest: dict[str, Any]) -> None:
    included = {row["prompt_id"] for row in manifest["rows"]}
    for entry in manifest["exclusions"]:
        assert entry["prompt_id"] not in included
        assert isinstance(entry["reason"], str) and entry["reason"]


def test_special_token_map_is_complete_and_typed(manifest: dict[str, Any]) -> None:
    retokenization = manifest["retokenization"]
    bos = retokenization["bos_pair_rule"]
    assert bos["glm_ids"] == [154822, 154824]
    assert bos["v4_ids"] == [0]

    mapping = retokenization["special_token_map"]
    assert mapping, "special token map is empty"
    for glm_id, entry in mapping.items():
        assert glm_id.isdigit() and int(glm_id) >= 154820
        assert entry["glm_surface"]
        assert entry["v4_surface"]
        assert entry["v4_ids"], f"{glm_id} maps to no V4 ids (silent drop)"
        assert all(isinstance(i, int) and 0 <= i < 129280 for i in entry["v4_ids"])
        assert entry["tier"][0] in {"A", "B", "C"}
        assert entry["applied_as"] in {"v4_special_token", "literal_text"}
        if entry["applied_as"] == "literal_text":
            assert entry["v4_surface"] == entry["glm_surface"], (
                f"{glm_id} literal mapping must preserve the surface string"
            )
        assert entry["occurrences"] > 0, f"{glm_id} is mapped but never occurs"


def test_supervision_convention_is_recorded(manifest: dict[str, Any]) -> None:
    convention = manifest["retokenization"]["supervision_convention"]
    assert "positions[i] + 1" in convention
    assert manifest["retokenization"]["split_seed"]
    assert manifest["retokenization"]["script_version"].startswith("retokenize_teich_dsv4/")
    assert manifest["retokenization"]["tier_b_mode"] in {"native", "literal"}


# ---------------------------------------------------------------------------
# Opt-in: these need the ~90 MB pack on disk, so they skip when it is absent.
# Run with: pytest tests/test_dsv4_teich_manifest.py -m pack
# ---------------------------------------------------------------------------


def _pack_path(manifest: dict[str, Any]) -> Path:
    return Path(manifest["pack_file"])


def _require_pack(manifest: dict[str, Any]) -> Path:
    path = _pack_path(manifest)
    if not path.exists():
        pytest.skip(f"pack not present at {path}")
    return path


@pytest.mark.pack
def test_pack_file_digest_matches_manifest(manifest: dict[str, Any]) -> None:
    import hashlib

    path = _require_pack(manifest)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    assert digest.hexdigest() == manifest["pack_file_sha256"], (
        "pack on disk does not match the committed manifest digest; "
        "regenerate with scripts/retokenize_teich_dsv4.py"
    )


@pytest.mark.pack
def test_pack_rows_agree_with_manifest_and_hold_shift_invariant(
    manifest: dict[str, Any],
) -> None:
    path = _require_pack(manifest)
    pack = json.loads(path.read_text(encoding="utf-8"))
    assert pack["model_id"] == V4_MODEL_ID
    rows = pack["prompt_rows"]
    assert len(rows) == manifest["totals"]["sessions_included"]

    by_id = {row["prompt_id"]: row for row in manifest["rows"]}
    assert set(by_id) == {row["prompt_id"] for row in rows}

    for row in rows:
        prompt_id = row["prompt_id"]
        expected = by_id[prompt_id]
        ids = row["encoded_token_ids"]
        positions = row["positions"]
        targets = row["target_token_ids"]

        assert row["token_count"] == len(ids) == expected["v4_token_count"], prompt_id
        assert row["token_ids_sha256"] == expected["token_ids_sha256"], prompt_id
        assert row["campaign_split"] == expected["campaign_split"], prompt_id
        assert row["split"] == expected["split"], prompt_id
        assert row["tuning_eligible"] == expected["tuning_eligible"], prompt_id
        assert row["glm_era_split"] == expected["glm_era_split"], prompt_id

        # Output-side supervision convention: positions index the logit that
        # predicts the next token.
        assert len(positions) == len(targets) == expected["v4_supervised_tokens"], prompt_id
        assert positions == sorted(set(positions)), prompt_id
        assert positions[0] >= 0 and positions[-1] + 1 < len(ids), prompt_id
        assert all(ids[p + 1] == t for p, t in zip(positions, targets)), prompt_id
        assert ids[0] == 0, f"{prompt_id} does not open with the V4 BOS"
        assert max(ids) < 129280, prompt_id


@pytest.mark.pack
def test_pack_token_digests_use_the_corpus_recipe(manifest: dict[str, Any]) -> None:
    import hashlib

    path = _require_pack(manifest)
    pack = json.loads(path.read_text(encoding="utf-8"))
    for row in pack["prompt_rows"]:
        payload = json.dumps(
            row["encoded_token_ids"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        assert hashlib.sha256(payload).hexdigest() == row["token_ids_sha256"], row["prompt_id"]
