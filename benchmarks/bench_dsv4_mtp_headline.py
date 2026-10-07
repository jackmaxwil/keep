"""Fresh-process same-machine DSV4 compressed AR versus DSpark headline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

from ramp.benchmark.metrics import (
    collect_metric_snapshot,
    collect_vm_stat_counts,
    reset_mlx_peak_memory,
    wait_for_memory_quiet,
)
from ramp.models.dsv4_composite_loader import (
    load_authenticated_dsv4_composite,
    validate_dsv4_artifact_identity,
    validate_dsv4_composite_inputs,
    validate_dsv4_payload_receipt,
    validate_dsv4_vq_manifest_structure,
)
from keep.quality.dsv4_mtp_runtime import (
    generate_dsv4_autoregressive,
    generate_dsv4_speculative,
)
from keep.quality.dsv4_teacher_agreement import heavy_job_lock
from keep.quality.dsv4_teacher_runner import load_dsv4_teich_pack

REPO_ROOT = Path(__file__).resolve().parents[1]
HEAVY_JOB_LOCK = REPO_ROOT / ".keep-heavy-job.lock"
DEFAULT_SOURCE = Path.home() / "models/DeepSeek-V4-Flash-0731"
DEFAULT_RESIDENT = (
    Path.home() / "keep-artifacts/dsv4-residents-release-precision-20260819"
)
DEFAULT_VQ = Path.home() / "keep-artifacts/dsv4-vq-e8p-g512"
DEFAULT_PACK = Path.home() / "models/teich/dsv4-coding-agent-v1-20260811.json"
FORBIDDEN_WIRED_ENV = (
    "GLM_MLX_WIRED_LIMIT_GB",
    "GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB",
)
EXPECTED_VERIFY_IMPLEMENTATION = "nax_e8p_m32n64"
BENCHMARK_SOURCE_PATH = Path(__file__).resolve()
RUNTIME_SOURCE_PATH = (REPO_ROOT / "src/keep/quality/dsv4_mtp_runtime.py").resolve()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_harness_identity() -> dict[str, str]:
    return {
        "benchmark_source_sha256": _sha256(BENCHMARK_SOURCE_PATH),
        "runtime_source_sha256": _sha256(RUNTIME_SOURCE_PATH),
    }


def _validate_bind_proof_anchor(
    bind_proof_path: str | Path, expected_bind_proof_sha256: str
) -> dict[str, Any]:
    path = Path(bind_proof_path).resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256(path) != expected_bind_proof_sha256
    ):
        raise ValueError("bind proof file SHA-256 does not match external anchor")
    proof = json.loads(path.read_text())
    if (
        not isinstance(proof, Mapping)
        or proof.get("record_type") != "dsv4_mtp_composite_bind_proof_v1"
    ):
        raise ValueError("bind proof record is invalid")

    receipt = proof.get("payload_content_receipt")
    receipt_sha256 = proof.get("payload_content_receipt_sha256")
    identity = proof.get("artifact_identity")
    harness = proof.get("harness_identity")
    if not isinstance(receipt, Mapping) or not isinstance(identity, Mapping):
        raise TypeError("bind proof payload or artifact identity is missing")
    identity_sha256 = validate_dsv4_artifact_identity(identity)
    if (
        proof.get("artifact_identity_sha256") != identity_sha256
        or identity.get("payload_content_receipt_sha256") != receipt_sha256
    ):
        raise ValueError("bind proof composite identity is invalid")
    if not isinstance(harness, Mapping):
        raise TypeError("bind proof harness identity is missing")
    harness_sha256 = _canonical_sha256(harness)
    if proof.get("harness_identity_sha256") != harness_sha256:
        raise ValueError("bind proof harness identity is invalid")

    resident_dir = Path(str(receipt.get("resident_dir"))).resolve()
    vq_dir = Path(str(receipt.get("vq_dir"))).resolve()
    resident_manifest_path = resident_dir / "resident-manifest.json"
    vq_manifest_path = vq_dir / "manifest.json"
    if _sha256(resident_manifest_path) != identity.get(
        "resident_manifest_sha256"
    ) or _sha256(vq_manifest_path) != identity.get("vq_manifest_sha256"):
        raise ValueError("bind proof manifests do not match current bytes")
    resident_manifest = json.loads(resident_manifest_path.read_text())
    vq_manifest = json.loads(vq_manifest_path.read_text())
    vq_inventory = validate_dsv4_vq_manifest_structure(vq_manifest)
    if resident_manifest.get("package_set_sha256") != identity.get(
        "resident_package_set_sha256"
    ) or vq_inventory.inventory_sha256 != identity.get("vq_inventory_sha256"):
        raise ValueError("bind proof package inventory is invalid")
    validated_receipt_sha256 = validate_dsv4_payload_receipt(
        resident_dir=resident_dir,
        resident_manifest=resident_manifest,
        vq_dir=vq_dir,
        vq_inventory=vq_inventory,
        receipt=receipt,
        expected_receipt_sha256=str(receipt_sha256),
    )
    if validated_receipt_sha256 != receipt_sha256:
        raise ValueError("bind proof payload receipt is invalid")
    return {
        "artifact_identity": dict(identity),
        "artifact_identity_sha256": identity_sha256,
        "harness_identity": dict(harness),
        "harness_identity_sha256": harness_sha256,
        "payload_content_receipt_sha256": receipt_sha256,
    }


def _assert_runtime_policy() -> None:
    present = [name for name in FORBIDDEN_WIRED_ENV if name in os.environ]
    if present:
        raise RuntimeError(f"custom wired-limit variables must stay absent: {present}")


def _thermal_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        ["pmset", "-g", "therm"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    body = (result.stdout + result.stderr).strip()
    return {
        "command": ["pmset", "-g", "therm"],
        "returncode": result.returncode,
        "output": body,
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "warning_lines": [
            line for line in body.splitlines() if "warning" in line.lower()
        ],
    }


def _prompt(pack: Path, prompt_tokens: int) -> tuple[Any, mx.array, dict[str, Any]]:
    sessions = load_dsv4_teich_pack(pack, splits=["report"])
    session = sessions[0]
    if session.campaign_split != "report":
        raise RuntimeError("headline prompt must come from the report split")
    count = min(int(prompt_tokens), int(session.token_ids.size))
    if count < 8:
        raise ValueError("headline prompt must contain at least eight tokens")
    ids = session.token_ids[:count].astype("int32", copy=True)
    prompt_token_ids = [int(value) for value in ids.tolist()]
    metadata = {
        "pack": str(pack.resolve()),
        "pack_sha256": _sha256(pack),
        "campaign_split": session.campaign_split,
        "prompt_id": f"{session.prompt_id}:prefix-{count}",
        "source_session_tokens": int(session.token_ids.size),
        "prompt_tokens": count,
        "prompt_token_ids": prompt_token_ids,
        "prompt_sha256": _canonical_sha256(prompt_token_ids),
        "selection": "shortest report session, deterministic prefix",
        "holdout_used": False,
    }
    return session, mx.array(ids[None, :]), metadata


def _row_clean(row: Mapping[str, Any]) -> bool:
    quiet = row.get("quiet")
    if not isinstance(quiet, Mapping):
        raise TypeError("headline quiet record is missing or malformed")
    if type(quiet.get("enabled")) is not bool:
        raise ValueError("headline quiet enabled must be a boolean")
    for key in ("pageouts_delta", "swapouts_delta"):
        value = row.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"headline {key} must be a nonnegative integer")
    if quiet["enabled"] is False:
        if set(quiet) != {"enabled"}:
            raise ValueError("headline quiet disabled record is malformed")
        return False

    active_keys = {
        "enabled",
        "available",
        "quiet",
        "attempts",
        "window_seconds",
        "pageouts_delta",
        "swapouts_delta",
    }
    if set(quiet) != active_keys:
        raise ValueError("headline quiet active record is malformed")
    for key in ("available", "quiet"):
        if type(quiet[key]) is not bool:
            raise ValueError(f"headline quiet {key} must be a boolean")
    attempts = quiet["attempts"]
    if type(attempts) is not int or attempts <= 0:
        raise ValueError("headline quiet attempts must be a positive integer")
    window = quiet["window_seconds"]
    if (
        type(window) not in (int, float)
        or not math.isfinite(float(window))
        or window <= 0
    ):
        raise ValueError("headline quiet window must be a positive number")
    if quiet["available"] is False:
        if (
            quiet["quiet"] is not False
            or quiet["pageouts_delta"] is not None
            or quiet["swapouts_delta"] is not None
        ):
            raise ValueError("headline quiet unavailable record is malformed")
        return False

    for key in ("pageouts_delta", "swapouts_delta"):
        value = quiet[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"headline quiet {key} must be a nonnegative integer")
    preflight_clean = quiet["pageouts_delta"] == 0 and quiet["swapouts_delta"] == 0
    if quiet["quiet"] is not preflight_clean:
        raise ValueError("headline quiet flag does not match active deltas")
    return preflight_clean and row["pageouts_delta"] == 0 and row["swapouts_delta"] == 0


def _require_nonnegative_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"speculative {key} must be a nonnegative integer")
    return value


def _dispatch_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("speculative dispatch trace contains an invalid record")
    implementation = record.get("implementation")
    token_rows = record.get("token_rows")
    if (
        not isinstance(implementation, str)
        or type(token_rows) is not int
        or token_rows <= 1
    ):
        raise ValueError("speculative dispatch trace contains an invalid wide call")
    return dict(record)


def _derive_raw_dispatches(
    stats: Mapping[str, Any],
) -> tuple[Counter[str], list[dict[str, Any]]]:
    declared_calls = _require_nonnegative_int(stats, "verify_kernel_calls")
    if declared_calls == 0:
        raise ValueError("speculative row has zero verify-kernel calls")
    records = stats.get("verify_kernel_dispatches")
    if not isinstance(records, list) or not records:
        raise ValueError("speculative dispatch trace is empty")
    normalized = [_dispatch_record(record) for record in records]
    if declared_calls != len(normalized):
        raise ValueError("speculative verify-kernel calls do not match dispatch trace")
    counts = Counter(str(record["implementation"]) for record in normalized)
    return counts, normalized


def _derive_compact_dispatches(
    stats: Mapping[str, Any],
) -> tuple[Counter[str], list[dict[str, Any]]]:
    declared_calls = _require_nonnegative_int(stats, "verify_kernel_calls")
    if declared_calls == 0:
        raise ValueError("speculative row has zero verify-kernel calls")
    records = stats.get("verify_kernel_unique_dispatches")
    if not isinstance(records, list) or not records:
        raise ValueError("speculative compact dispatch trace is empty")
    counts: Counter[str] = Counter()
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    for item in records:
        if not isinstance(item, Mapping):
            raise TypeError("speculative compact dispatch trace is invalid")
        count = item.get("count")
        dispatch = item.get("dispatch")
        if type(count) is not int or count <= 0 or not isinstance(dispatch, Mapping):
            raise ValueError("speculative compact dispatch trace is invalid")
        record = _dispatch_record(dispatch)
        signature = json.dumps(record, sort_keys=True)
        if signature in seen:
            raise ValueError("speculative compact dispatch trace contains duplicates")
        seen.add(signature)
        implementation = str(record["implementation"])
        counts[implementation] += count
        total += count
        normalized.append({"count": count, "dispatch": record})
    if declared_calls != total:
        raise ValueError("speculative verify-kernel calls do not match compact trace")
    declared_counts = stats.get("verify_kernel_implementation_counts")
    if declared_counts != dict(sorted(counts.items())):
        raise ValueError("speculative implementation counts do not match compact trace")
    return counts, sorted(
        normalized, key=lambda item: json.dumps(item["dispatch"], sort_keys=True)
    )


def _validate_speculative_stats(
    stats: Mapping[str, Any],
    *,
    row_emitted_tokens: int,
    dispatch_mode: str,
) -> dict[str, Any]:
    if not isinstance(stats, Mapping):
        raise TypeError("speculative row statistics are missing")
    drafted = _require_nonnegative_int(stats, "drafted_tokens")
    accepted = _require_nonnegative_int(stats, "accepted_tokens")
    rejected = _require_nonnegative_int(stats, "rejected_tokens")
    passes = _require_nonnegative_int(stats, "verify_passes")
    emitted = _require_nonnegative_int(stats, "emitted_tokens")
    if drafted != accepted + rejected:
        raise ValueError("speculative acceptance counters do not conserve drafts")
    if passes == 0:
        raise ValueError("speculative verify passes must be positive")
    if emitted != row_emitted_tokens or emitted != 1 + accepted + passes:
        raise ValueError("speculative emitted token counters do not conserve output")

    expected_rates = {
        "acceptance_rate": accepted / drafted if drafted else 0.0,
        "accepted_per_verify_pass": accepted / passes,
        "emitted_per_verify_pass": emitted / passes,
    }
    for key, expected in expected_rates.items():
        value = stats.get(key)
        if (
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            label = key.replace("_", " ")
            raise ValueError(f"speculative {label} does not match counters")

    if dispatch_mode == "raw":
        counts, records = _derive_raw_dispatches(stats)
        grouped: dict[str, dict[str, Any]] = {}
        for record in records:
            signature = json.dumps(record, sort_keys=True)
            if signature not in grouped:
                grouped[signature] = {"count": 0, "dispatch": record}
            grouped[signature]["count"] += 1
        unique_dispatches = [grouped[key] for key in sorted(grouped)]
    elif dispatch_mode == "compact":
        counts, unique_dispatches = _derive_compact_dispatches(stats)
    else:
        raise ValueError(f"invalid DSV4 dispatch validation mode {dispatch_mode!r}")
    if counts[EXPECTED_VERIFY_IMPLEMENTATION] <= 0:
        raise ValueError("speculative dispatch trace has no production E8P verify call")
    implementations = sorted(counts)
    declared_implementations = stats.get("verify_kernel_implementations")
    if (
        declared_implementations is not None
        and declared_implementations != implementations
    ):
        raise ValueError("speculative implementation names do not match dispatch trace")

    return {
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "rejected_tokens": rejected,
        "verify_passes": passes,
        "emitted_tokens": emitted,
        **expected_rates,
        "verify_kernel_calls": sum(counts.values()),
        "verify_kernel_implementations": implementations,
        "verify_kernel_implementation_counts": dict(sorted(counts.items())),
        "verify_kernel_unique_dispatches": unique_dispatches,
    }


def _validate_prompt_pack(row: Mapping[str, Any]) -> dict[str, Any]:
    pack = row.get("pack")
    pack_sha256 = row.get("pack_sha256")
    if not isinstance(pack, str) or not isinstance(pack_sha256, str):
        raise TypeError("headline pack identity is missing")
    pack_path = Path(pack)
    if not pack_path.is_file() or _sha256(pack_path) != pack_sha256:
        raise ValueError("headline pack SHA-256 does not match current bytes")
    _session, _tokens, metadata = _prompt(pack_path, 64)
    if metadata["prompt_tokens"] != 64:
        raise ValueError("headline prompt selection did not produce exactly 64 tokens")
    return metadata


def _validate_row_identity(
    row: Mapping[str, Any],
    bind_proof: Mapping[str, Any],
    prompt_authority: Mapping[str, Any],
) -> tuple[Any, ...]:
    identity = row.get("artifact_identity")
    if not isinstance(identity, Mapping):
        raise TypeError("headline artifact identity mapping is missing")
    identity_sha256 = validate_dsv4_artifact_identity(identity)
    if (
        row.get("artifact_identity_sha256") != identity_sha256
        or identity != bind_proof["artifact_identity"]
        or identity_sha256 != bind_proof["artifact_identity_sha256"]
    ):
        raise ValueError("headline artifact identity does not match bind proof")

    harness = row.get("harness_identity")
    if harness != bind_proof["harness_identity"]:
        raise ValueError("headline harness identity does not match bind proof")
    harness_sha256 = _canonical_sha256(harness)
    if (
        row.get("harness_identity_sha256") != harness_sha256
        or harness_sha256 != bind_proof["harness_identity_sha256"]
    ):
        raise ValueError("headline harness identity does not match bind proof")

    prompt_keys = (
        "pack",
        "pack_sha256",
        "campaign_split",
        "prompt_id",
        "source_session_tokens",
        "prompt_tokens",
        "prompt_token_ids",
        "prompt_sha256",
        "selection",
        "holdout_used",
    )
    if (
        type(row.get("source_session_tokens")) is not int
        or row.get("holdout_used") is not False
        or any(row.get(key) != prompt_authority[key] for key in prompt_keys)
    ):
        raise ValueError("headline prompt selection does not match report pack")
    prompt_ids = row.get("prompt_token_ids")
    prompt_tokens = row.get("prompt_tokens")
    if (
        not isinstance(prompt_ids, list)
        or not prompt_ids
        or any(type(token) is not int for token in prompt_ids)
        or type(prompt_tokens) is not int
        or prompt_tokens != len(prompt_ids)
        or row.get("prompt_sha256") != _canonical_sha256(prompt_ids)
    ):
        raise ValueError("headline prompt digest does not match recorded token IDs")

    max_new = row.get("max_new_tokens")
    emitted = row.get("emitted_tokens")
    token_ids = row.get("token_ids")
    if (
        type(max_new) is not int
        or max_new <= 0
        or type(emitted) is not int
        or emitted != max_new
        or not isinstance(token_ids, list)
        or any(type(token) is not int for token in token_ids)
        or len(token_ids) != emitted
        or row.get("token_ids_sha256") != _canonical_sha256(token_ids)
    ):
        raise ValueError("headline emitted token digest/count is invalid")

    generation = {
        "campaign_split": "report",
        "holdout_used": False,
        "max_new_tokens": max_new,
        "prompt_id": row.get("prompt_id"),
        "prompt_sha256": row.get("prompt_sha256"),
        "prompt_tokens": prompt_tokens,
        "scoring": "greedy_argmax",
    }
    if row.get("generation_config") != generation or row.get(
        "generation_config_sha256"
    ) != _canonical_sha256(generation):
        raise ValueError("headline generation configuration digest is invalid")
    if row.get("wired_limit_environment_absent") is not True:
        raise ValueError("headline row did not prove wired-limit absence")
    return (
        identity_sha256,
        harness_sha256,
        prompt_authority["pack_sha256"],
        row.get("generation_config_sha256"),
    )


def _summarize_dsv4_mtp_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_clean_pairs: int,
    dispatch_mode: str,
    bind_proof: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no DSV4 MTP headline rows")
    if type(min_clean_pairs) is not int or min_clean_pairs <= 0:
        raise ValueError("minimum clean pairs must be a positive integer")
    pids: list[int] = []
    common: set[tuple[Any, ...]] = set()
    pairs: dict[int, dict[str, Mapping[str, Any]]] = {}
    cleanliness: dict[tuple[int, str], bool] = {}
    spec_signatures: set[str] = set()
    compact_stats: dict[int, dict[str, Any]] = {}
    prompt_authority = _validate_prompt_pack(rows[0])
    for row in rows:
        if row.get("record_type") != "dsv4_mtp_headline_row_v1":
            raise ValueError("unexpected DSV4 MTP headline row type")
        pid = row.get("pid")
        if row.get("fresh_process") is not True or type(pid) is not int:
            raise ValueError("every headline row must prove a fresh process")
        pids.append(pid)
        role = str(row.get("role"))
        if role not in ("baseline", "speculative"):
            raise ValueError(f"invalid headline role {role!r}")
        pair = row.get("pair_id")
        if type(pair) is not int or role in pairs.setdefault(pair, {}):
            raise ValueError("duplicate or invalid pair/role row")
        pairs[pair][role] = row
        cleanliness[(pair, role)] = _row_clean(row)
        common.add(_validate_row_identity(row, bind_proof, prompt_authority))
        if role == "baseline":
            if row.get("speculative_stats") is not None:
                raise ValueError("baseline row must not contain speculative statistics")
        else:
            normalized = _validate_speculative_stats(
                row.get("speculative_stats"),
                row_emitted_tokens=int(row["emitted_tokens"]),
                dispatch_mode=dispatch_mode,
            )
            compact_stats[pair] = normalized
            signature = {
                key: normalized[key]
                for key in (
                    "drafted_tokens",
                    "accepted_tokens",
                    "rejected_tokens",
                    "verify_passes",
                    "emitted_tokens",
                    "acceptance_rate",
                    "accepted_per_verify_pass",
                    "emitted_per_verify_pass",
                    "verify_kernel_calls",
                    "verify_kernel_implementation_counts",
                )
            }
            spec_signatures.add(json.dumps(signature, sort_keys=True))
    if len(pids) != len(set(pids)):
        raise ValueError("headline rows do not have distinct fresh-process PIDs")
    if len(common) != 1:
        raise ValueError("artifact, prompt, config, pack, or harness identity drifted")
    if any(set(pair) != {"baseline", "speculative"} for pair in pairs.values()):
        raise ValueError("each headline pair needs baseline and speculative rows")
    if len(spec_signatures) != 1:
        raise ValueError("deterministic speculative acceptance counters drifted")

    token_parity = all(
        pair["baseline"].get("token_ids_sha256")
        == pair["speculative"].get("token_ids_sha256")
        for pair in pairs.values()
    )
    if not token_parity:
        raise ValueError("greedy baseline/speculative token parity failed")
    clean_pairs = [
        pair
        for pair_id, pair in pairs.items()
        if cleanliness[(pair_id, "baseline")] and cleanliness[(pair_id, "speculative")]
    ]
    ratios = [
        float(pair["baseline"]["elapsed_seconds"])
        / float(pair["speculative"]["elapsed_seconds"])
        for pair in clean_pairs
    ]
    if any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios):
        raise ValueError("headline elapsed-time ratio is invalid")
    orders = Counter(str(pair["baseline"].get("pair_order")) for pair in clean_pairs)
    if clean_pairs and (
        not orders["baseline_first"] or not orders["speculative_first"]
    ):
        raise ValueError("clean pairs must include both execution orders")
    enough = len(clean_pairs) >= min_clean_pairs
    median_ratio = statistics.median(ratios) if ratios else None
    verdict = (
        "INCOMPLETE"
        if not enough
        else "PASS"
        if median_ratio is not None and median_ratio > 1.0
        else "STOP"
    )
    identity_sha, harness_sha, pack_sha, generation_sha = next(iter(common))
    representative_pair = min(compact_stats)
    representative_row = pairs[representative_pair]["speculative"]
    return {
        "record_type": "dsv4_mtp_headline_summary_v2",
        "schema_version": 2,
        "comparison": "compressed_ar_elapsed / compressed_speculative_elapsed",
        "rows": len(rows),
        "pairs": len(pairs),
        "clean_pairs": len(clean_pairs),
        "dirty_pairs": len(pairs) - len(clean_pairs),
        "min_clean_pairs": min_clean_pairs,
        "orders": dict(sorted(orders.items())),
        "artifact_identity": representative_row["artifact_identity"],
        "artifact_identity_sha256": identity_sha,
        "harness_identity": representative_row["harness_identity"],
        "harness_identity_sha256": harness_sha,
        "pack": representative_row["pack"],
        "pack_sha256": pack_sha,
        "generation_config": representative_row["generation_config"],
        "generation_config_sha256": generation_sha,
        "prompt_id": representative_row["prompt_id"],
        "prompt_sha256": representative_row["prompt_sha256"],
        "max_new_tokens": representative_row["max_new_tokens"],
        "token_parity": token_parity,
        "acceptance_consistency": len(spec_signatures) == 1,
        "speculative_stats": compact_stats[representative_pair],
        "paired_baseline_over_speculative": ratios,
        "median_baseline_over_speculative": median_ratio,
        "verdict": verdict,
        "claim_limit": (
            "Same-machine deterministic report-split prefix only; ratio is not "
            "portable across machines or prompt distributions."
        ),
    }


def summarize_dsv4_mtp_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bind_proof_path: str | Path,
    expected_bind_proof_sha256: str,
    min_clean_pairs: int = 3,
) -> dict[str, Any]:
    """Validate full raw paired evidence and return the decision-usable ratio."""

    bind_proof = _validate_bind_proof_anchor(
        bind_proof_path, expected_bind_proof_sha256
    )
    return _summarize_dsv4_mtp_rows(
        rows,
        min_clean_pairs=min_clean_pairs,
        dispatch_mode="raw",
        bind_proof=bind_proof,
    )


def build_dsv4_mtp_compact_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    bind_proof_path: str | Path,
    expected_bind_proof_sha256: str,
    min_clean_pairs: int = 3,
) -> dict[str, Any]:
    """Conserve authenticated rows while counting repeated dispatch records."""

    bind_proof = _validate_bind_proof_anchor(
        bind_proof_path, expected_bind_proof_sha256
    )
    _summarize_dsv4_mtp_rows(
        rows,
        min_clean_pairs=min_clean_pairs,
        dispatch_mode="raw",
        bind_proof=bind_proof,
    )
    compact_rows: list[dict[str, Any]] = []
    for row in rows:
        compact = json.loads(json.dumps(row))
        if compact.get("role") == "speculative":
            compact["speculative_stats"] = _validate_speculative_stats(
                row["speculative_stats"],
                row_emitted_tokens=int(row["emitted_tokens"]),
                dispatch_mode="raw",
            )
        compact_rows.append(compact)
    summary = _summarize_dsv4_mtp_rows(
        compact_rows,
        min_clean_pairs=min_clean_pairs,
        dispatch_mode="compact",
        bind_proof=bind_proof,
    )
    evidence = {
        "record_type": "dsv4_mtp_headline_compact_evidence_v2",
        "schema_version": 2,
        "rows": compact_rows,
        "summary": summary,
    }
    return evidence


def validate_dsv4_mtp_compact_evidence(
    evidence: Mapping[str, Any],
    *,
    bind_proof_path: str | Path,
    expected_bind_proof_sha256: str,
    min_clean_pairs: int = 3,
) -> dict[str, Any]:
    """Recompute a compact artifact from counted unique dispatch evidence."""

    bind_proof = _validate_bind_proof_anchor(
        bind_proof_path, expected_bind_proof_sha256
    )
    if evidence.get("record_type") != "dsv4_mtp_headline_compact_evidence_v2":
        raise ValueError("unexpected DSV4 compact headline evidence type")
    rows = evidence.get("rows")
    if not isinstance(rows, list):
        raise TypeError("DSV4 compact headline rows are missing")
    summary = _summarize_dsv4_mtp_rows(
        rows,
        min_clean_pairs=min_clean_pairs,
        dispatch_mode="compact",
        bind_proof=bind_proof,
    )
    if evidence.get("summary") != summary:
        raise ValueError("DSV4 compact headline summary does not recompute exactly")
    return summary


def _load(args: argparse.Namespace):
    if args.command == "bind-proof":
        inputs = validate_dsv4_composite_inputs(
            source_dir=args.source,
            resident_dir=args.resident_dir,
            vq_dir=args.vq_dir,
            authenticate_payloads=True,
        )
    else:
        receipt_path = args.content_receipt.resolve()
        receipt_container = json.loads(receipt_path.read_text())
        receipt = receipt_container.get("payload_content_receipt")
        if not isinstance(receipt, Mapping):
            raise ValueError("bind proof does not contain a payload content receipt")
        if (
            receipt_container.get("payload_content_receipt_sha256")
            != args.content_receipt_sha256
        ):
            raise ValueError(
                "bind proof payload receipt SHA-256 does not match command"
            )
        inputs = validate_dsv4_composite_inputs(
            source_dir=args.source,
            resident_dir=args.resident_dir,
            vq_dir=args.vq_dir,
            payload_receipt=receipt,
            expected_payload_receipt_sha256=args.content_receipt_sha256,
        )
    loaded = load_authenticated_dsv4_composite(inputs)
    mx.eval(loaded.model.parameters())
    return loaded


def _bind_proof(args: argparse.Namespace) -> dict[str, Any]:
    _assert_runtime_policy()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prior bind proof {output}")
    started = time.perf_counter()
    before = collect_vm_stat_counts()
    loaded = _load(args)
    params = list(tree_flatten(loaded.model.parameters()))
    names = [name for name, _ in params]
    values = [value for _, value in params]
    after = collect_metric_snapshot(previous_vm_stat_counts=before)
    non_vq = loaded.report.non_vq_bind_report
    result = {
        "record_type": "dsv4_mtp_composite_bind_proof_v1",
        "schema_version": 1,
        "pid": os.getpid(),
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_identity": loaded.inputs.identity,
        "artifact_identity_sha256": loaded.inputs.identity_sha256,
        "payload_content_receipt": loaded.inputs.payload_receipt,
        "payload_content_receipt_sha256": loaded.inputs.payload_receipt_sha256,
        "harness_identity": _current_harness_identity(),
        "harness_identity_sha256": _canonical_sha256(_current_harness_identity()),
        "resident_bound_count": non_vq.bound_count,
        "resident_backbone_bound_count": non_vq.backbone_bound_count,
        "resident_mtp_bound_count": non_vq.mtp_bound_count,
        "resident_bound_names_sha256": _canonical_sha256(
            sorted(non_vq.bound_backbone_parameters + non_vq.bound_mtp_parameters)
        ),
        "bound_backbone_layers": list(loaded.report.bound_backbone_layers),
        "bound_mtp_stages": list(loaded.report.bound_mtp_stages),
        "runtime_parameter_tensors": len(names),
        "runtime_parameter_names_sha256": _canonical_sha256(sorted(names)),
        "runtime_parameter_storage_bytes": sum(int(value.nbytes) for value in values),
        "dense_routed_parameter_names": list(
            loaded.report.dense_routed_parameter_names
        ),
        "unbound_backbone_vq_experts": loaded.report.unbound_backbone_vq_experts,
        "unbound_mtp_vq_experts": loaded.report.unbound_mtp_vq_experts,
        "wired_limit_environment_absent": True,
        **after,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _run_row(args: argparse.Namespace) -> dict[str, Any]:
    _assert_runtime_policy()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prior row {output}")
    loaded = _load(args)
    _session, prompt, prompt_meta = _prompt(args.pack.resolve(), args.prompt_tokens)
    harness_identity = _current_harness_identity()
    generation_config = {
        "campaign_split": prompt_meta["campaign_split"],
        "holdout_used": prompt_meta["holdout_used"],
        "max_new_tokens": args.max_new_tokens,
        "prompt_id": prompt_meta["prompt_id"],
        "prompt_sha256": prompt_meta["prompt_sha256"],
        "prompt_tokens": prompt_meta["prompt_tokens"],
        "scoring": "greedy_argmax",
    }

    # Compile each selected path before the quiet/timed window.
    if args.role == "baseline":
        generate_dsv4_autoregressive(loaded.model, prompt, max_new_tokens=3)
    else:
        generate_dsv4_speculative(loaded.model, prompt, max_new_tokens=4)
    gc.collect()
    clear_cache = getattr(mx, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()

    quiet = wait_for_memory_quiet(
        window_seconds=args.quiet_window_seconds,
        max_attempts=args.quiet_max_attempts,
    )
    thermal = _thermal_snapshot()
    before_vm = collect_vm_stat_counts()
    reset_mlx_peak_memory()
    started = time.perf_counter()
    if args.role == "baseline":
        result = generate_dsv4_autoregressive(
            loaded.model, prompt, max_new_tokens=args.max_new_tokens
        )
        spec_stats = None
    else:
        result = generate_dsv4_speculative(
            loaded.model, prompt, max_new_tokens=args.max_new_tokens
        )
        spec_stats = result.stats.to_dict()
        spec_stats["verify_kernel_implementations"] = sorted(
            {
                str(record["implementation"])
                for record in result.stats.verify_kernel_dispatches
            }
        )
    elapsed = time.perf_counter() - started
    after_vm = collect_vm_stat_counts()
    metrics = collect_metric_snapshot(previous_vm_stat_counts=before_vm)
    pageins_delta = (
        None
        if before_vm is None or after_vm is None
        else int(after_vm.get("pageins", 0)) - int(before_vm.get("pageins", 0))
    )
    tokens = list(result.token_ids)
    row = {
        "record_type": "dsv4_mtp_headline_row_v1",
        "schema_version": 1,
        "fresh_process": True,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "argv": sys.argv,
        "pair_id": args.pair_id,
        "pair_order": args.pair_order,
        "role": args.role,
        "artifact_identity_sha256": loaded.inputs.identity_sha256,
        "artifact_identity": loaded.inputs.identity,
        "harness_identity": harness_identity,
        "harness_identity_sha256": _canonical_sha256(harness_identity),
        "generation_config": generation_config,
        "generation_config_sha256": _canonical_sha256(generation_config),
        **prompt_meta,
        "max_new_tokens": args.max_new_tokens,
        "token_ids": tokens,
        "token_ids_sha256": _canonical_sha256(tokens),
        "emitted_tokens": len(tokens),
        "elapsed_seconds": elapsed,
        "tokens_per_second": len(tokens) / elapsed,
        "speculative_stats": spec_stats,
        "quiet": quiet,
        "thermal": thermal,
        "pageins_delta": pageins_delta,
        "wired_limit_environment_absent": True,
        **metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))
    return row


def _summarize(args: argparse.Namespace) -> dict[str, Any]:
    paths = sorted(args.rows_dir.glob("pair-*-*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    result = summarize_dsv4_mtp_rows(
        rows,
        bind_proof_path=args.bind_proof,
        expected_bind_proof_sha256=args.bind_proof_sha256,
        min_clean_pairs=args.min_clean_pairs,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite prior summary {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _validate_compact(args: argparse.Namespace) -> dict[str, Any]:
    evidence = json.loads(args.input.read_text())
    result = validate_dsv4_mtp_compact_evidence(
        evidence,
        bind_proof_path=args.bind_proof,
        expected_bind_proof_sha256=args.bind_proof_sha256,
        min_clean_pairs=args.min_clean_pairs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _series(args: argparse.Namespace) -> None:
    root = args.output_dir.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite prior evidence directory {root}")
    rows_dir = root / "rows"
    rows_dir.mkdir(parents=True)
    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run-row",
        "--source",
        str(args.source),
        "--resident-dir",
        str(args.resident_dir),
        "--vq-dir",
        str(args.vq_dir),
        "--content-receipt",
        str(args.content_receipt.resolve()),
        "--content-receipt-sha256",
        args.content_receipt_sha256,
        "--pack",
        str(args.pack),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--quiet-window-seconds",
        str(args.quiet_window_seconds),
        "--quiet-max-attempts",
        str(args.quiet_max_attempts),
    ]
    env = dict(os.environ)
    env["KEEP_TASK4_LOCK_HELD"] = "1"
    for pair in range(args.pairs):
        order = "baseline_first" if pair % 2 == 0 else "speculative_first"
        roles = ("baseline", "speculative")
        if order == "speculative_first":
            roles = tuple(reversed(roles))
        for role in roles:
            output = rows_dir / f"pair-{pair:02d}-{role}.json"
            command = [
                *common,
                "--pair-id",
                str(pair),
                "--pair-order",
                order,
                "--role",
                role,
                "--output",
                str(output),
            ]
            subprocess.run(command, check=True, env=env)
    summary_args = argparse.Namespace(
        rows_dir=rows_dir,
        min_clean_pairs=args.min_clean_pairs,
        output=root / "summary.json",
        bind_proof=args.content_receipt,
        bind_proof_sha256=args.bind_proof_sha256,
    )
    _summarize(summary_args)
    rows = [
        json.loads(path.read_text()) for path in sorted(rows_dir.glob("pair-*-*.json"))
    ]
    compact_output = root / "headline-evidence.json"
    if compact_output.exists():
        raise FileExistsError(
            f"refusing to overwrite compact evidence {compact_output}"
        )
    compact = build_dsv4_mtp_compact_evidence(
        rows,
        bind_proof_path=args.content_receipt,
        expected_bind_proof_sha256=args.bind_proof_sha256,
        min_clean_pairs=args.min_clean_pairs,
    )
    compact_output.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")


def _common_artifact_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--resident-dir", type=Path, default=DEFAULT_RESIDENT)
    parser.add_argument("--vq-dir", type=Path, default=DEFAULT_VQ)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bind = sub.add_parser("bind-proof")
    _common_artifact_args(bind)
    bind.add_argument("--output", type=Path, required=True)
    bind.set_defaults(func=_bind_proof, heavy=True)

    row = sub.add_parser("run-row")
    _common_artifact_args(row)
    row.add_argument("--content-receipt", type=Path, required=True)
    row.add_argument("--content-receipt-sha256", required=True)
    row.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    row.add_argument("--prompt-tokens", type=int, default=64)
    row.add_argument("--max-new-tokens", type=int, default=16)
    row.add_argument("--pair-id", type=int, required=True)
    row.add_argument(
        "--pair-order", choices=("baseline_first", "speculative_first"), required=True
    )
    row.add_argument("--role", choices=("baseline", "speculative"), required=True)
    row.add_argument("--quiet-window-seconds", type=float, default=5.0)
    row.add_argument("--quiet-max-attempts", type=int, default=3)
    row.add_argument("--output", type=Path, required=True)
    row.set_defaults(func=_run_row, heavy=True)

    summary = sub.add_parser("summarize")
    summary.add_argument("--rows-dir", type=Path, required=True)
    summary.add_argument("--bind-proof", type=Path, required=True)
    summary.add_argument("--bind-proof-sha256", required=True)
    summary.add_argument("--min-clean-pairs", type=int, default=3)
    summary.add_argument("--output", type=Path, required=True)
    summary.set_defaults(func=_summarize, heavy=False)

    compact = sub.add_parser("validate-compact")
    compact.add_argument("--input", type=Path, required=True)
    compact.add_argument("--bind-proof", type=Path, required=True)
    compact.add_argument("--bind-proof-sha256", required=True)
    compact.add_argument("--min-clean-pairs", type=int, default=3)
    compact.set_defaults(func=_validate_compact, heavy=False)

    series = sub.add_parser("series")
    _common_artifact_args(series)
    series.add_argument("--content-receipt", type=Path, required=True)
    series.add_argument("--content-receipt-sha256", required=True)
    series.add_argument("--bind-proof-sha256", required=True)
    series.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    series.add_argument("--prompt-tokens", type=int, default=64)
    series.add_argument("--max-new-tokens", type=int, default=16)
    series.add_argument("--pairs", type=int, default=3)
    series.add_argument("--min-clean-pairs", type=int, default=3)
    series.add_argument("--quiet-window-seconds", type=float, default=5.0)
    series.add_argument("--quiet-max-attempts", type=int, default=3)
    series.add_argument("--output-dir", type=Path, required=True)
    series.set_defaults(func=_series, heavy=True)

    args = parser.parse_args()
    lock_held = os.environ.get("KEEP_TASK4_LOCK_HELD") == "1"
    holder = f"dsv4-task4:{args.command}:{os.getpid()}"
    with heavy_job_lock(
        HEAVY_JOB_LOCK, enabled=bool(args.heavy and not lock_held), holder=holder
    ):
        result = args.func(args)
    if (
        args.command in {"summarize", "validate-compact"}
        and result["verdict"] == "INCOMPLETE"
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
