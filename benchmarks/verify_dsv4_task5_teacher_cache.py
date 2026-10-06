"""Replay the Task 5 terminal audit from external payload and pinned inputs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
PACK_SHA256 = "168eb4cb9251d1cb01a79a0d61205480f02252a0e6d2c96f596847b21e7f82a5"
CONFIG_SHA256 = "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
INDEX_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
COMPLETION_SHA256 = "68d973a8b79801a90f888c4edc5d7739721aaf2e5dc3698787964d64467cf4eb"
LEGACY_GENERATION_SHA256 = (
    "6cac8ee4e95efe949d9a7d3b0843e002ef0886d7e52be1d66fad210c9a195997"
)
HISTORICAL_MANIFEST_SHA256 = (
    "f3819b6ba69832a5aef1893bb82ebe1537f7eb420603c3a1a4ed8994b62a1bf5"
)
LEGACY_EVIDENCE_SHA256 = (
    "d2ac01f8fa02de942d38d108665ac7457d09b564295a643fc065100595bc801d"
)
LEGACY_SESSION_CONTRACT_SHA256 = (
    "c60bf865b639d37c78d56b637b3398834b7ee94793cfe36ba2ce812dcc379244"
)
LEGACY_AUDIT_CONTRACT_SHA256 = (
    "a6f9247a5ff6a799793df292b1ab49a01a25947bbd0c23905893168f1c729203"
)
PRODUCER_COMMIT = "433ec34da3a20538e0c7a7502bfbcdc225ce39f4"
PRODUCER_CLI_SHA256 = "0e9c536163ace0540587f564ae2e2f1dc31f5b66c47d9f4d3caaf5b59fe93ca1"
PRODUCER_RUNNER_SHA256 = (
    "c552cb8687f4e33258d639e752d9678fce18b42f26bdbcde1d87d45b7745cad6"
)

SOURCE_AUTHORITY_FIELDS = (
    "source_model_id",
    "source_revision",
    "config_sha256",
    "index_sha256",
    "download_completion_sha256",
)

SEMANTIC_CLAIMS = {
    "all_exact_array_keys_dtypes_shapes",
    "all_final_hidden_finite_and_nonconstant",
    "all_logit_values_finite_and_descending_with_fp16_tolerance",
    "all_logsumexp_finite_and_at_least_top_logit",
    "all_positions_and_next_targets_match_pack",
    "all_regular_uncompressed_npz",
    "all_source_generation_split_chunk_and_session_identity_match",
    "all_tail_mass_finite_in_half_open_unit_interval",
    "all_topk_ids_and_targets_in_vocab",
    "all_topk_ids_distinct_per_position_slot",
    "all_width5_targets_and_validity_rederived",
}

ARRAY_NAMES = {
    "positions",
    "target_token_ids",
    "mtp_draft_width",
    "mtp_target_token_ids",
    "mtp_target_valid",
    "mtp_topk_logit_ids",
    "mtp_topk_logit_values",
    "mtp_logsumexp",
    "mtp_tail_mass",
    "mtp_final_hidden",
    "record_type",
    "schema_version",
    "prompt_id",
    "campaign_split",
    "token_ids_sha256",
    "token_count",
    "prefill_chunk_tokens",
    "generation_config_sha256",
    "tokens_per_s",
    "session_seconds",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _scalar(value: np.ndarray) -> Any:
    return np.asarray(value).reshape(()).item()


def _exact_int(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be a JSON integer")
    return value


def _exact_int_list(value: Any, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not int for item in value)
    ):
        raise ValueError(f"{label} must be a non-empty list of JSON integers")
    return value


def _source_authority(manifest: Mapping[str, Any]) -> dict[str, str]:
    identities = manifest.get("identities")
    _require(isinstance(identities, Mapping), "historical manifest identities missing")
    _require(
        all(
            isinstance(identities.get(field), str) for field in SOURCE_AUTHORITY_FIELDS
        ),
        "historical manifest source authority is incomplete",
    )
    return {field: str(identities[field]) for field in SOURCE_AUTHORITY_FIELDS}


def summarize_legacy_evidence(
    payload: Mapping[str, Any], *, file_sha256: str
) -> dict[str, Any]:
    _require(file_sha256 == LEGACY_EVIDENCE_SHA256, "legacy evidence file hash")
    _require(
        payload.get("record_type") == "dsv4_task5_terminal_audit_v1",
        "legacy evidence record",
    )
    _require(payload.get("schema_version") == 1, "legacy evidence schema")
    _require(payload.get("result") == "PASS", "legacy evidence result")
    checks = payload.get("contract_checks")
    _require(isinstance(checks, Mapping), "legacy semantic claims")
    _require(set(checks) == SEMANTIC_CLAIMS, "legacy semantic claim inventory")
    _require(
        all(value is True for value in checks.values()), "legacy semantic claim value"
    )
    session_rows = payload.get("session_files")
    _require(isinstance(session_rows, list), "legacy session rows")
    claimed_session_sha = payload.get("session_files_contract_sha256")
    claimed_audit_sha = payload.get("audit_contract_sha256")
    _require(
        claimed_session_sha == LEGACY_SESSION_CONTRACT_SHA256,
        "legacy claimed session contract",
    )
    _require(
        claimed_audit_sha == LEGACY_AUDIT_CONTRACT_SHA256,
        "legacy claimed audit contract",
    )
    audit_preimage = dict(payload)
    audit_preimage.pop("audit_contract_sha256")
    return {
        "authority": "historical-record-only",
        "file_sha256": file_sha256,
        "session_contract": {
            "claimed_sha256": claimed_session_sha,
            "stored_preimage_sha256": canonical_sha256(session_rows),
        },
        "audit_contract": {
            "claimed_sha256": claimed_audit_sha,
            "stored_preimage_sha256": canonical_sha256(audit_preimage),
        },
        "semantic_claim_names": sorted(checks),
    }


def load_selected_pack(
    path: Path, *, split: str, vocab_size: int
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    _require(isinstance(payload, Mapping), "pack must be an object")
    _require(
        payload.get("record_type") == "dsv4_coding_agent_corpus", "pack record type"
    )
    _require(payload.get("model_id") == MODEL_ID, "pack model id")
    rows = payload.get("prompt_rows")
    _require(isinstance(rows, list) and rows, "pack prompt rows")
    _require(
        _exact_int(payload.get("prompt_row_count"), label="prompt_row_count")
        == len(rows),
        "pack prompt row count",
    )

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"prompt_rows[{index}] must be an object")
        prompt_id = row.get("prompt_id")
        campaign_split = row.get("campaign_split")
        _require(
            isinstance(prompt_id, str) and prompt_id, f"prompt_rows[{index}] prompt id"
        )
        _require(prompt_id not in seen, f"duplicate prompt id {prompt_id}")
        seen.add(prompt_id)
        _require(
            isinstance(campaign_split, str) and campaign_split,
            f"prompt_rows[{index}] campaign split",
        )
        if campaign_split != split:
            continue

        token_ids = _exact_int_list(
            row.get("encoded_token_ids"),
            label=f"prompt_rows[{index}] encoded_token_ids",
        )
        _require(len(token_ids) >= 2, f"prompt_rows[{index}] token count")
        _require(
            all(0 <= token < vocab_size for token in token_ids),
            f"prompt_rows[{index}] token outside vocabulary",
        )
        _require(
            _exact_int(
                row.get("token_count"), label=f"prompt_rows[{index}] token_count"
            )
            == len(token_ids),
            f"prompt_rows[{index}] token count mismatch",
        )
        token_sha = canonical_sha256(token_ids)
        _require(
            row.get("token_ids_sha256") == token_sha, f"prompt_rows[{index}] token hash"
        )
        positions = _exact_int_list(
            row.get("positions"), label=f"prompt_rows[{index}] positions"
        )
        targets = _exact_int_list(
            row.get("target_token_ids"), label=f"prompt_rows[{index}] target_token_ids"
        )
        _require(
            len(positions) == len(targets), f"prompt_rows[{index}] supervision shape"
        )
        _require(
            all(right > left for left, right in pairwise(positions)),
            f"prompt_rows[{index}] position ordering",
        )
        _require(
            positions[0] >= 0 and positions[-1] < len(token_ids) - 1,
            f"prompt_rows[{index}] position range",
        )
        _require(
            targets == [token_ids[position + 1] for position in positions],
            f"prompt_rows[{index}] target alignment",
        )
        selected.append(
            {
                "prompt_id": prompt_id,
                "campaign_split": campaign_split,
                "token_ids": token_ids,
                "positions": positions,
                "target_token_ids": targets,
                "token_ids_sha256": token_sha,
                "token_count": len(token_ids),
                "supervised": len(positions),
            }
        )
    _require(selected, f"no {split} sessions")
    return sorted(selected, key=lambda row: (row["token_count"], row["prompt_id"]))


def verify_session_file(
    path: Path,
    *,
    session: Mapping[str, Any],
    generation_sha256: str,
    chunk: int,
    top_k: int,
    width: int,
    hidden_size: int,
    vocab_size: int,
) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    with zipfile.ZipFile(path) as zipped:
        infos = zipped.infolist()
        names = [info.filename for info in infos]
        _require(len(names) == len(set(names)), f"duplicate zip member: {path}")
        _require(
            set(names) == {f"{name}.npy" for name in ARRAY_NAMES},
            f"zip inventory: {path}",
        )
        _require(
            all(info.compress_type == zipfile.ZIP_STORED for info in infos),
            f"compressed zip member: {path}",
        )

    with np.load(path, allow_pickle=False) as archive:
        _require(set(archive.files) == ARRAY_NAMES, f"array inventory: {path}")
        capture = {name: archive[name] for name in archive.files}

    supervised = len(session["positions"])
    shapes = {
        "positions": (np.dtype(np.int32), (supervised,)),
        "target_token_ids": (np.dtype(np.int32), (supervised,)),
        "mtp_draft_width": (np.dtype(np.int32), ()),
        "mtp_target_token_ids": (np.dtype(np.int32), (supervised, width)),
        "mtp_target_valid": (np.dtype(np.bool_), (supervised, width)),
        "mtp_topk_logit_ids": (np.dtype(np.int32), (supervised, width, top_k)),
        "mtp_topk_logit_values": (np.dtype(np.float16), (supervised, width, top_k)),
        "mtp_logsumexp": (np.dtype(np.float32), (supervised, width)),
        "mtp_tail_mass": (np.dtype(np.float16), (supervised, width)),
        "mtp_final_hidden": (np.dtype(np.float16), (supervised, width, hidden_size)),
        "schema_version": (np.dtype(np.int32), ()),
        "token_count": (np.dtype(np.int64), ()),
        "prefill_chunk_tokens": (np.dtype(np.int32), ()),
        "tokens_per_s": (np.dtype(np.float64), ()),
        "session_seconds": (np.dtype(np.float64), ()),
    }
    for name, (dtype, shape) in shapes.items():
        value = np.asarray(capture[name])
        _require(
            value.dtype == dtype and value.shape == shape, f"{path}: {name} contract"
        )
    for name in (
        "record_type",
        "prompt_id",
        "campaign_split",
        "token_ids_sha256",
        "generation_config_sha256",
    ):
        value = np.asarray(capture[name])
        _require(
            value.dtype.kind == "U" and value.shape == (), f"{path}: {name} contract"
        )

    _require(
        _scalar(capture["record_type"]) == "dsv4_teacher_mtp_targets_v1",
        f"record: {path}",
    )
    _require(int(_scalar(capture["schema_version"])) == 1, f"schema: {path}")
    _require(_scalar(capture["prompt_id"]) == session["prompt_id"], f"prompt: {path}")
    _require(
        _scalar(capture["campaign_split"]) == session["campaign_split"],
        f"split: {path}",
    )
    _require(
        _scalar(capture["token_ids_sha256"]) == session["token_ids_sha256"],
        f"token hash: {path}",
    )
    _require(
        int(_scalar(capture["token_count"])) == len(session["token_ids"]),
        f"tokens: {path}",
    )
    _require(int(_scalar(capture["prefill_chunk_tokens"])) == chunk, f"chunk: {path}")
    _require(
        _scalar(capture["generation_config_sha256"]) == generation_sha256,
        f"generation: {path}",
    )
    _require(int(_scalar(capture["mtp_draft_width"])) == width, f"width: {path}")
    rate = float(_scalar(capture["tokens_per_s"]))
    seconds = float(_scalar(capture["session_seconds"]))
    _require(
        math.isfinite(rate) and rate > 0 and math.isfinite(seconds) and seconds > 0,
        f"timing: {path}",
    )
    _require(
        abs(rate - len(session["token_ids"]) / seconds) <= max(1e-9, abs(rate) * 1e-12),
        f"timing conservation: {path}",
    )

    positions = np.asarray(session["positions"], dtype=np.int32)
    targets = np.asarray(session["target_token_ids"], dtype=np.int32)
    _require(np.array_equal(capture["positions"], positions), f"positions: {path}")
    _require(np.array_equal(capture["target_token_ids"], targets), f"targets: {path}")
    offsets = np.arange(1, width + 1, dtype=np.int64)
    wanted = positions.astype(np.int64)[:, None] + offsets[None, :]
    token_ids = np.asarray(session["token_ids"], dtype=np.int64)
    valid = wanted < token_ids.size
    block_targets = np.where(
        valid, token_ids[np.clip(wanted, 0, token_ids.size - 1)], -1
    )
    _require(
        np.array_equal(capture["mtp_target_valid"], valid), f"target validity: {path}"
    )
    _require(
        np.array_equal(capture["mtp_target_token_ids"], block_targets),
        f"block targets: {path}",
    )

    ids = capture["mtp_topk_logit_ids"]
    id_min = vocab_size
    id_max = -1
    for start in range(0, supervised, 32):
        block = ids[start : start + 32]
        id_min = min(id_min, int(block.min()))
        id_max = max(id_max, int(block.max()))
        _require(id_min >= 0 and id_max < vocab_size, f"top-k id range: {path}")
        ordered = np.sort(block, axis=-1)
        duplicates = np.any(np.diff(ordered, axis=-1) <= 0, axis=(1, 2))
        if np.any(duplicates):
            row = start + int(np.flatnonzero(duplicates)[0])
            raise ValueError(f"duplicate top-k id: {path} row {row}")

    values = capture["mtp_topk_logit_values"]
    logsumexp = capture["mtp_logsumexp"]
    _require(np.all(np.isfinite(logsumexp)), f"logsumexp finite: {path}")
    for start in range(0, supervised, 32):
        block = values[start : start + 32].astype(np.float32)
        _require(np.all(np.isfinite(block)), f"logit finite: {path} row {start}")
        tolerance = 1e-2 * np.maximum(np.abs(block[..., :-1]), 1.0)
        _require(
            np.all(np.diff(block, axis=-1) <= tolerance),
            f"logit ordering: {path} row {start}",
        )
        lse = logsumexp[start : start + 32]
        _require(
            np.all(block[..., 0] - lse <= 1e-2 * np.maximum(np.abs(lse), 1.0)),
            f"logsumexp bound: {path} row {start}",
        )

    tail = capture["mtp_tail_mass"]
    _require(
        np.all(np.isfinite(tail)) and np.all(tail >= 0) and np.all(tail < 1),
        f"tail mass: {path}",
    )
    hidden = capture["mtp_final_hidden"]
    hidden_min = math.inf
    hidden_max = -math.inf
    for start in range(0, supervised, 8):
        block = hidden[start : start + 8]
        _require(np.all(np.isfinite(block)), f"hidden finite: {path} row {start}")
        hidden_min = min(hidden_min, float(block.min()))
        hidden_max = max(hidden_max, float(block.max()))
    _require(hidden_min < hidden_max, f"hidden constant: {path}")

    array_contract = {
        name: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for name, value in capture.items()
    }
    return {
        "prompt_id": session["prompt_id"],
        "path": f"sessions/{path.name}",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "array_contract_sha256": canonical_sha256(array_contract),
        "token_count": len(session["token_ids"]),
        "supervised": supervised,
        "tokens_per_s": rate,
        "session_seconds": seconds,
        "hidden_min": hidden_min,
        "hidden_max": hidden_max,
        "topk_id_min": id_min,
        "topk_id_max": id_max,
    }


def _git_blob_sha256(repo: Path, commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def verify_task5_cache(
    *,
    repo: Path,
    out_dir: Path,
    pack: Path,
    checkpoint: Path,
    historical_manifest_path: Path,
    legacy_evidence_path: Path,
) -> dict[str, Any]:
    historical_manifest_sha = sha256_file(historical_manifest_path)
    _require(
        historical_manifest_sha == HISTORICAL_MANIFEST_SHA256,
        "historical manifest hash",
    )
    current_manifest_path = out_dir / "run-manifest.json"
    _require(
        sha256_file(current_manifest_path) == historical_manifest_sha,
        "current manifest differs from authenticated historical manifest",
    )
    manifest = json.loads(historical_manifest_path.read_text())
    _require(
        manifest.get("record_type") == "dsv4_teacher_capture_run", "manifest record"
    )
    _require(manifest.get("schema_version") == 1, "manifest schema")
    _require(manifest.get("mode") == "mtp-targets", "manifest mode")
    _require(manifest.get("splits") == ["mtp-train"], "manifest split")
    _require(
        Path(str(manifest.get("pack_path"))).resolve() == pack.resolve(),
        "manifest pack path",
    )
    _require(
        sha256_file(pack) == PACK_SHA256 == manifest.get("pack_sha256"), "pack hash"
    )

    source_authority = _source_authority(manifest)
    _require(source_authority["source_model_id"] == MODEL_ID, "source model")
    _require(source_authority["source_revision"] == REVISION, "source revision")
    source_hashes = {
        "config_sha256": sha256_file(checkpoint / "config.json"),
        "index_sha256": sha256_file(checkpoint / "model.safetensors.index.json"),
        "download_completion_sha256": sha256_file(
            checkpoint / "_KEEP_DOWNLOAD_COMPLETE.json"
        ),
    }
    _require(
        source_hashes
        == {
            "config_sha256": CONFIG_SHA256,
            "index_sha256": INDEX_SHA256,
            "download_completion_sha256": COMPLETION_SHA256,
        },
        "checkpoint source hash",
    )
    _require(
        all(source_authority[key] == value for key, value in source_hashes.items()),
        "manifest checkpoint authority",
    )
    config = json.loads((checkpoint / "config.json").read_text())
    vocab_size = _exact_int(config.get("vocab_size"), label="checkpoint vocab_size")
    hidden_size = _exact_int(config.get("hidden_size"), label="checkpoint hidden_size")

    sessions = load_selected_pack(pack, split="mtp-train", vocab_size=vocab_size)
    _require(len(sessions) == 120, "selected session count")
    _require(
        sum(row["token_count"] for row in sessions) == 5_552_257, "selected tokens"
    )
    _require(
        sum(row["supervised"] for row in sessions) == 1_502_378, "selected supervision"
    )
    session_manifest = {
        row["prompt_id"]: {
            "campaign_split": row["campaign_split"],
            "token_count": row["token_count"],
            "supervised": row["supervised"],
            "token_ids_sha256": row["token_ids_sha256"],
        }
        for row in sessions
    }
    _require(manifest.get("session_count") == 120, "manifest session count")
    _require(manifest.get("sessions") == session_manifest, "manifest session inventory")

    legacy_generation = manifest.get("generation_config")
    _require(isinstance(legacy_generation, Mapping), "legacy generation preimage")
    _require(
        legacy_generation.get("record_type") == "dsv4_teacher_generation_config_v1",
        "legacy generation record",
    )
    _require(
        canonical_sha256(legacy_generation)
        == LEGACY_GENERATION_SHA256
        == manifest.get("generation_config_sha256"),
        "legacy generation hash",
    )
    top_k = _exact_int(legacy_generation.get("top_k"), label="generation top_k")
    width = _exact_int(
        legacy_generation.get("mtp_draft_width"), label="generation mtp_draft_width"
    )
    chunk = _exact_int(
        legacy_generation.get("prefill_chunk_tokens"), label="generation prefill chunk"
    )
    bound_generation = {
        **legacy_generation,
        "record_type": "dsv4_teacher_generation_config_v2",
        "source_authority": source_authority,
        "pack_sha256": PACK_SHA256,
        "session_inventory_sha256": canonical_sha256(session_manifest),
    }
    bound_generation_sha256 = canonical_sha256(bound_generation)

    session_dir = out_dir / "sessions"
    expected_paths = {
        session_dir / f"{row['prompt_id'].replace('/', '__')}.npz" for row in sessions
    }
    _require(set(session_dir.iterdir()) == expected_paths, "session path inventory")
    per_file: list[dict[str, Any]] = []
    for index, session in enumerate(sessions, 1):
        path = session_dir / f"{session['prompt_id'].replace('/', '__')}.npz"
        per_file.append(
            verify_session_file(
                path,
                session=session,
                generation_sha256=LEGACY_GENERATION_SHA256,
                chunk=chunk,
                top_k=top_k,
                width=width,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
            )
        )
        if index % 10 == 0:
            print(f"verified {index}/120", file=sys.stderr, flush=True)
    session_files_contract_sha256 = canonical_sha256(per_file)

    legacy_evidence_sha = sha256_file(legacy_evidence_path)
    _require(legacy_evidence_sha == LEGACY_EVIDENCE_SHA256, "legacy evidence hash")
    legacy_evidence = json.loads(legacy_evidence_path.read_text())
    legacy_evidence_summary = summarize_legacy_evidence(
        legacy_evidence,
        file_sha256=legacy_evidence_sha,
    )

    status_path = out_dir / "status.json"
    status_log_path = out_dir / "status.log"
    summary_path = out_dir / "run-summary.json"
    status = json.loads(status_path.read_text())
    status_rows = [
        json.loads(line) for line in status_log_path.read_text().splitlines()
    ]
    completions = [row for row in status_rows if row.get("phase") == "session-complete"]
    _require(
        status.get("phase") == "complete" and status == status_rows[-1],
        "terminal status",
    )
    _require(len(completions) == 120, "completion row count")
    _require(
        [row.get("session") for row in completions]
        == [row["prompt_id"] for row in sessions],
        "completion inventory",
    )
    summary = json.loads(summary_path.read_text())
    _require(
        summary.get("sessions_produced") == 40
        and summary.get("sessions_skipped") == 80
        and summary.get("sessions_selected") == 120,
        "terminal resume summary",
    )

    producer_sources = {
        "commit": PRODUCER_COMMIT,
        "cli_sha256": _git_blob_sha256(
            repo, PRODUCER_COMMIT, "benchmarks/produce_dsv4_teacher_cache.py"
        ),
        "teacher_runner_sha256": _git_blob_sha256(
            repo, PRODUCER_COMMIT, "src/mlx_vq/quality/dsv4_teacher_runner.py"
        ),
    }
    _require(
        producer_sources["cli_sha256"] == PRODUCER_CLI_SHA256, "producer CLI source"
    )
    _require(
        producer_sources["teacher_runner_sha256"] == PRODUCER_RUNNER_SHA256,
        "producer runner source",
    )

    session_bytes = sum(row["bytes"] for row in per_file)
    _require(session_bytes == 153_938_907_002, "session logical bytes")
    result: dict[str, Any] = {
        "record_type": "dsv4_task5_replay_audit_v2",
        "schema_version": 2,
        "result": "PASS",
        "scope": {
            "sessions": 120,
            "input_tokens": 5_552_257,
            "supervised_positions": 1_502_378,
            "logical_session_bytes": session_bytes,
            "missing": [],
            "extra": [],
            "duplicates": [],
            "temporary_or_partial": [],
            "other_split_sessions": 0,
        },
        "legacy_payload_authentication": {
            "strategy": "complete-cache historical-manifest bridge",
            "historical_manifest_sha256": historical_manifest_sha,
            "legacy_generation_config": legacy_generation,
            "legacy_generation_config_sha256": LEGACY_GENERATION_SHA256,
            "source_pack_session_bound_generation_config": bound_generation,
            "source_pack_session_bound_generation_config_sha256": bound_generation_sha256,
            "complete_cache_required": True,
            "mixed_legacy_and_v2_files_allowed": False,
        },
        "authenticated_source": {
            **source_authority,
            "pack_sha256": PACK_SHA256,
            "producer_sources": producer_sources,
            "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        },
        "replayed_semantic_contract": {
            "session_files_contract_sha256": session_files_contract_sha256,
            "semantic_claims_replayed": sorted(SEMANTIC_CLAIMS),
            "legacy_record": legacy_evidence_summary,
        },
        "terminal_artifact_sha256": {
            "run_manifest": sha256_file(current_manifest_path),
            "run_summary": sha256_file(summary_path),
            "status": sha256_file(status_path),
            "status_log": sha256_file(status_log_path),
        },
        "session_files": per_file,
    }
    result["audit_contract_sha256"] = canonical_sha256(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--legacy-evidence", type=Path, required=True)
    parser.add_argument("--expected-evidence", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = args.repo.resolve()
    with (repo / ".keep-heavy-job.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("another heavy job holds .keep-heavy-job.lock") from error
        result = verify_task5_cache(
            repo=repo,
            out_dir=args.out_dir.resolve(),
            pack=args.pack.resolve(),
            checkpoint=args.checkpoint.resolve(),
            historical_manifest_path=args.historical_manifest.resolve(),
            legacy_evidence_path=args.legacy_evidence.resolve(),
        )
    if args.expected_evidence is not None:
        expected = json.loads(args.expected_evidence.read_text())
        _require(result == expected, "replayed audit differs from expected evidence")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Task 5 audit failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
