#!/usr/bin/env python3
"""Run proof-ladder steps 9--12 for the pinned GLM-5.2 source teacher.

The module intentionally imports neither MLX nor the model stack at import time.
Synthetic tests can therefore exercise planning, evidence, and failure semantics on
headless hosts.  Every source snapshot read occurs only after both advisory locks
have been acquired.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
MODEL_ID = "0xSero/glm-5.2-reap-504B-v2"
REVISION = "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
PROFILE_NAME = "glm52-reap-504b-v2"
CONFIG_SHA256 = "5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b"
INDEX_SHA256 = "bb5b4fa9782aea5ffc66f9145d6e630f1045d385c30437c531bebe422c075f3f"
PROFILE_SHA256 = "ae8b852139ac26e63846b0475064e5b955a00f1ff75f628682a3d8494474114d"
PROFILE_CONTRACT_SHA256 = (
    "28d95f2f2e411ff886c38b0c773f335c34a1b044e401573e90b399c93db3dcb8"
)
PROMPT_PACK_SHA256 = (
    "697677a4949f4ee7e370ac5e9a55e9631b0a1385f95a07dfd3a3f2b87d8edf31"
)
VOCAB_SIZE = 154_880
HIDDEN_SIZE = 6_144
EXPERTS_PER_TOKEN = 8
MAIN_LAYER_COUNT = 78
SPARSE_PROOF_LAYER = 10
EXPERT_PROOF_INDEX = 0
PROOF_STEPS = (9, 10, 11, 12)
RECORD_TYPE = "glm52_source_teacher_bounded_proof_v1"

DEFAULT_SNAPSHOT_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2"
    / "snapshots"
    / REVISION
)
DEFAULT_NON_VQ_PACKAGE_DIR = (
    REPO_ROOT
    / "artifacts/build/glm52_reap_504b_materialization_probe_20260709"
    / "steps/non_vq_pack-a678bc3c/out"
)
DEFAULT_PROMPT_PACK_JSON = (
    REPO_ROOT / "artifacts/quality/glm52-family-eval-prompts-20260709-v2.json"
)
DEFAULT_PROFILE_YAML = REPO_ROOT / "models/glm52-reap-504b-v2.yaml"


class DecodedOracle(NamedTuple):
    layer: int
    expert: int
    projection: str
    sha256: str


DECODED_ORACLES = (
    DecodedOracle(
        10,
        0,
        "gate_proj",
        "19638efebbca55205804574337d71a87dd0217ef6322e6a399444ecf0d79bfbd",
    ),
    DecodedOracle(
        10,
        0,
        "up_proj",
        "b673bd500f468b16eab2d04eefab9deca16191aef3bc63f2a18d72a0c6a6df13",
    ),
    DecodedOracle(
        10,
        0,
        "down_proj",
        "5e9ce06d0db45f464464f10fcab8bd424ee81f58115e9bea693c87bea3f59dc6",
    ),
    DecodedOracle(
        29,
        58,
        "gate_proj",
        "0119f166fe1fb59f47faa5b0580538b7030d1cbaa691206a7f98568f4ec7a117",
    ),
)


class OracleMismatchError(AssertionError):
    """A pinned decoded-byte oracle did not match the real source payload."""


class SimulatedCheckpointInterruption(RuntimeError):
    """Synthetic interruption used by headless checkpoint-resume tests."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path} must contain a JSON object")
    return value


def _require_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return resolved


def _require_directory(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    return resolved


def _require_file_sha256(path: Path, expected: str, *, label: str) -> str:
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return actual


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded real-source GLM-5.2 proof-ladder steps 9--12 under "
            "the global and proof-specific locks."
        )
    )
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    parser.add_argument(
        "--non-vq-package-dir", default=str(DEFAULT_NON_VQ_PACKAGE_DIR)
    )
    parser.add_argument("--prompt-pack-json", default=str(DEFAULT_PROMPT_PACK_JSON))
    parser.add_argument("--profile-yaml", default=str(DEFAULT_PROFILE_YAML))
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--skip-steps",
        default="",
        help="Comma-separated proof-ladder step numbers to skip (9,10,11,12)",
    )
    parser.add_argument("--only-step", type=int, choices=PROOF_STEPS)
    parser.add_argument(
        "--seed",
        type=int,
        default=5202,
        help="Deterministic hidden-state seed used by steps 10 and 11",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _parse_skips(raw: str | Sequence[int]) -> set[int]:
    if isinstance(raw, str):
        pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
        try:
            skips = {int(piece) for piece in pieces}
        except ValueError as error:
            raise ValueError("--skip-steps must be comma-separated integers") from error
    else:
        skips = {int(value) for value in raw}
    invalid = sorted(skips - set(PROOF_STEPS))
    if invalid:
        raise ValueError(f"unsupported skipped proof steps: {invalid}")
    return skips


def select_steps(only_step: int | None, skip_steps: str | Sequence[int]) -> tuple[int, ...]:
    skips = _parse_skips(skip_steps)
    if only_step is not None:
        if only_step not in PROOF_STEPS:
            raise ValueError(f"unsupported --only-step {only_step}")
        if only_step in skips:
            raise ValueError(f"--only-step {only_step} cannot also be skipped")
        return (only_step,)
    selected = tuple(step for step in PROOF_STEPS if step not in skips)
    if not selected:
        raise ValueError("at least one bounded proof step must be selected")
    return selected


def new_evidence(
    args: argparse.Namespace,
    *,
    selected_steps: Sequence[int],
) -> dict[str, Any]:
    selected = set(selected_steps)
    return {
        "schema_version": 2,
        "record_type": RECORD_TYPE,
        "created_at": _utc_now(),
        "ordered_steps": list(PROOF_STEPS),
        "requested_steps": [step for step in PROOF_STEPS if step in selected],
        "skipped_steps": [step for step in PROOF_STEPS if step not in selected],
        "pinned_identities": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "profile": PROFILE_NAME,
            "config_sha256": CONFIG_SHA256,
            "index_sha256": INDEX_SHA256,
            "profile_sha256": PROFILE_SHA256,
            "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
            "prompt_pack_sha256": PROMPT_PACK_SHA256,
        },
        "inputs": {
            "snapshot_dir": str(Path(args.snapshot_dir).expanduser()),
            "non_vq_package_dir": str(Path(args.non_vq_package_dir).expanduser()),
            "prompt_pack_json": str(Path(args.prompt_pack_json).expanduser()),
            "profile_yaml": str(Path(args.profile_yaml).expanduser()),
        },
        "locks": {
            "global_heavy_job_lock": str(REPO_ROOT / ".keep-heavy-job.lock"),
            "proof_run_lock": f"{Path(args.output_json).expanduser()}.lock",
            "locks_acquired": False,
        },
        "oracle_independence": {
            "independent_moe_oracle": True,
            "independent_route_slot_scatter": True,
            "independent_route_rank_reduction": True,
            "checkpoint_resume_from_persisted_bytes": True,
            "shared_attention_modules": True,
            "shared_norm_and_lm_head_modules": True,
            "shared_embedding_and_dense_mlp_modules": True,
            "shared_router_and_shared_expert_modules": True,
            "shared_expert_weight_resolver": True,
        },
        "proofs": {
            f"step_{step}": {
                "step": step,
                "status": "pending" if step in selected else "skipped",
                "proof_pass": None,
            }
            for step in PROOF_STEPS
        },
        "phases": [],
        "all_phase_memory_counters_known": False,
        "all_phase_memory_clean": False,
        "bounded_proof_pass": False,
        "evidence_body_sha256": None,
    }


def execute_selected_steps(
    selected_steps: Sequence[int],
    proof_functions: Mapping[int, Callable[[], dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    selected = set(selected_steps)
    results: dict[int, dict[str, Any]] = {}
    for step in PROOF_STEPS:
        if step not in selected:
            continue
        if step not in proof_functions:
            raise ValueError(f"missing proof function for step {step}")
        result = proof_functions[step]()
        if not isinstance(result, dict) or result.get("proof_pass") is not True:
            raise AssertionError(f"bounded proof step {step} did not pass")
        results[step] = result
    return results


def verify_decoded_oracles(
    oracles: Sequence[DecodedOracle],
    *,
    decoded_bytes_resolver: Callable[[DecodedOracle], bytes],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for oracle in oracles:
        payload = decoded_bytes_resolver(oracle)
        if not isinstance(payload, bytes):
            raise TypeError("decoded oracle resolver must return bytes")
        actual = _sha256_bytes(payload)
        if actual != oracle.sha256:
            raise OracleMismatchError(
                f"layer {oracle.layer} expert {oracle.expert} {oracle.projection} "
                f"decoded-byte oracle mismatch: expected {oracle.sha256}, found {actual}"
            )
        results.append(
            {
                "layer": oracle.layer,
                "expert": oracle.expert,
                "projection": oracle.projection,
                "decoded_sha256": actual,
                "decoded_bytes": len(payload),
            }
        )
    return results


def _independent_route_rank_reduce(
    route_outputs: Any,
    route_scores: Any,
    *,
    float32_dtype: Any,
    output_dtype: Any,
) -> Any:
    """Reduce route slots in rank order without a shared reduction primitive."""

    if route_outputs.ndim != 3 or route_scores.ndim != 2:
        raise ValueError("route-slot oracle requires [tokens, top_k, hidden] outputs")
    if tuple(route_outputs.shape[:2]) != tuple(route_scores.shape):
        raise ValueError("route-slot outputs and scores do not align")
    if int(route_outputs.shape[1]) <= 0:
        raise ValueError("route-slot oracle requires at least one route rank")
    accumulator = route_outputs[:, 0, :].astype(float32_dtype) * route_scores[
        :, 0, None
    ].astype(float32_dtype)
    for route_rank in range(1, int(route_outputs.shape[1])):
        weighted = route_outputs[:, route_rank, :].astype(
            float32_dtype
        ) * route_scores[:, route_rank, None].astype(float32_dtype)
        accumulator = accumulator + weighted
    return accumulator.astype(output_dtype)


def _production_route_reduction_for_fixture(
    route_outputs: np.ndarray,
    route_scores: np.ndarray,
) -> np.ndarray:
    """Synthetic stand-in for the shared production reduction in headless tests."""

    return (route_outputs * route_scores[..., None]).sum(
        axis=1, dtype=np.float32
    )


def verify_route_reduction_fixture(
    route_outputs: np.ndarray,
    route_scores: np.ndarray,
) -> None:
    """Prove a production-style reducer disagrees with an independent oracle bug."""

    production = np.ascontiguousarray(
        _production_route_reduction_for_fixture(route_outputs, route_scores),
        dtype=np.float32,
    )
    oracle = np.ascontiguousarray(
        _independent_route_rank_reduce(
            route_outputs,
            route_scores,
            float32_dtype=np.float32,
            output_dtype=np.float32,
        ),
        dtype=np.float32,
    )
    if production.shape != oracle.shape or production.tobytes() != oracle.tobytes():
        raise AssertionError(
            "production reduction disagrees with independent route-slot oracle"
        )


def run_checkpoint_reload_probe(
    *,
    interrupting_run: Callable[[], Any],
    resumed_run: Callable[[], Any],
    interruption_type: type[BaseException],
    clear_live_state: Callable[[], None],
) -> Any:
    """Require an interruption, clear live state, then invoke the byte-resume path."""

    interrupted = False
    try:
        interrupting_run()
    except interruption_type:
        interrupted = True
    if not interrupted:
        raise AssertionError("checkpoint probe did not interrupt before resume")
    clear_live_state()
    return resumed_run()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("output JSON must not be a symlink")
    body = dict(payload)
    body["evidence_body_sha256"] = None
    body["evidence_body_sha256"] = _canonical_sha256(
        {key: value for key, value in body.items() if key != "evidence_body_sha256"}
    )
    if isinstance(payload, dict):
        payload["evidence_body_sha256"] = body["evidence_body_sha256"]
    encoded = (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("short evidence JSON write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _load_producer_api() -> Any:
    module_name = "mlx_vq.quality.glm52_teacher_cache_producer"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = SOURCE_ROOT / "mlx_vq/quality/glm52_teacher_cache_producer.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load GLM52 teacher-cache producer helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _clear_runtime_memory() -> None:
    gc.collect()
    mx = sys.modules.get("mlx.core")
    clear_cache = None if mx is None else getattr(mx, "clear_cache", None)
    if clear_cache is not None:
        clear_cache()
    gc.collect()


def _counter_delta(
    before: Mapping[str, object] | None,
    after: Mapping[str, object] | None,
    key: str,
) -> int | None:
    if before is None or after is None:
        return None
    start = before.get(key)
    end = after.get(key)
    if type(start) is not int or type(end) is not int or start < 0 or end < start:
        return None
    return end - start


def _run_phase(
    *,
    ordinal: int,
    step: int,
    reader: Callable[[], Mapping[str, object] | None],
    action: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = reader()
    started_at = _utc_now()
    started = time.perf_counter()
    error: BaseException | None = None
    result: dict[str, Any] | None = None
    try:
        result = action()
        if result.get("proof_pass") is not True:
            raise AssertionError(f"bounded proof step {step} returned proof_pass != true")
    except BaseException as caught:
        error = caught
    finally:
        _clear_runtime_memory()
    after = reader()
    pageouts = _counter_delta(before, after, "pageouts")
    swapouts = _counter_delta(before, after, "swapouts")
    known = pageouts is not None and swapouts is not None
    phase = {
        "ordinal": ordinal,
        "phase_id": f"proof-step-{step}",
        "proof_step": step,
        "started_at": started_at,
        "ended_at": _utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "pageouts_delta": pageouts,
        "swapouts_delta": swapouts,
        "memory_counters_known": known,
        "memory_clean": (pageouts == 0 and swapouts == 0) if known else None,
        "phase_pass": error is None,
    }
    if error is not None:
        phase["error"] = f"{type(error).__name__}: {error}"
        raise _PhaseFailure(error, phase) from error
    assert result is not None
    return result, phase


class _PhaseFailure(RuntimeError):
    def __init__(self, original: BaseException, phase: dict[str, Any]):
        super().__init__(str(original))
        self.original = original
        self.phase = phase


class RuntimeContext:
    def __init__(self, args: argparse.Namespace, producer_api: Any):
        self.args = args
        self.producer_api = producer_api
        self.snapshot_dir: Path | None = None
        self.non_vq_package_dir: Path | None = None
        self.prompt_pack_path: Path | None = None
        self.profile_path: Path | None = None
        self.prompt_pack: dict[str, Any] | None = None
        self.prompt_rows: tuple[dict[str, Any], ...] = ()
        self.config: dict[str, Any] | None = None
        self.profile: Any = None
        self.source_index: Any = None
        self.non_vq_index: Any = None
        self._non_vq_audit: Any = None

    def authenticate_inputs(self) -> dict[str, Any]:
        if str(SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(SOURCE_ROOT))
        snapshot = _require_directory(self.args.snapshot_dir, label="snapshot directory")
        non_vq = _require_directory(
            self.args.non_vq_package_dir,
            label="non-VQ package directory",
        )
        prompt_path = _require_file(self.args.prompt_pack_json, label="prompt pack")
        profile_path = _require_file(self.args.profile_yaml, label="profile YAML")
        config_path = _require_file(snapshot / "config.json", label="source config")
        index_path = _require_file(
            snapshot / "model.safetensors.index.json",
            label="source index",
        )
        non_vq_manifest_path = _require_file(
            non_vq / "non-vq-manifest.json", label="non-VQ manifest"
        )
        non_vq_index_path = _require_file(
            non_vq / "model.safetensors.index.json", label="non-VQ index"
        )
        _require_file_sha256(config_path, CONFIG_SHA256, label="source config")
        _require_file_sha256(index_path, INDEX_SHA256, label="source index")
        _require_file_sha256(profile_path, PROFILE_SHA256, label="profile YAML")
        _require_file_sha256(prompt_path, PROMPT_PACK_SHA256, label="prompt pack")

        config = _load_json_object(config_path, label="source config")
        prompt_pack = _load_json_object(prompt_path, label="prompt pack")
        rows = prompt_pack.get("prompt_rows")
        if not isinstance(rows, list) or len(rows) != 66 or not all(
            isinstance(row, dict) for row in rows
        ):
            raise ValueError("prompt pack must contain exactly 66 prompt_rows objects")
        expected_prompt_values = {
            "record_type": "glm52_family_eval_prompt_pack",
            "model_id": MODEL_ID,
            "revision": REVISION,
            "profile": PROFILE_NAME,
            "prompt_pack_ready": True,
            "prompt_row_count": 66,
            "expected_vocab_size": VOCAB_SIZE,
        }
        for field, expected in expected_prompt_values.items():
            if prompt_pack.get(field) != expected:
                raise ValueError(
                    f"prompt pack {field} mismatch: expected {expected!r}, "
                    f"found {prompt_pack.get(field)!r}"
                )
        predictor_positions = 0
        for index, row in enumerate(rows):
            token_ids = row.get("encoded_token_ids")
            if not isinstance(token_ids, list) or len(token_ids) < 2 or any(
                type(token) is not int or not 0 <= token < VOCAB_SIZE
                for token in token_ids
            ):
                raise ValueError(f"prompt row {index} has invalid encoded_token_ids")
            if row.get("token_count") != len(token_ids):
                raise ValueError(f"prompt row {index} token_count mismatch")
            predictor_positions += len(token_ids) - 1
        if predictor_positions != 744:
            raise ValueError(
                f"frozen prompt predictor positions must equal 744, found {predictor_positions}"
            )

        profiles = importlib.import_module("mlx_vq.models.profiles")
        profile = profiles.load_profile(profile_path)
        profile_contract = _canonical_sha256(asdict(profile))
        if profile_contract != PROFILE_CONTRACT_SHA256:
            raise ValueError(
                "profile contract SHA-256 mismatch: expected "
                f"{PROFILE_CONTRACT_SHA256}, found {profile_contract}"
            )
        if (
            profile.name != PROFILE_NAME
            or profile.hf_model_id != MODEL_ID
            or profile.revision != REVISION
            or profile.num_layers != MAIN_LAYER_COUNT
            or profile.hidden_size != HIDDEN_SIZE
            or profile.experts_per_tok != EXPERTS_PER_TOKEN
            or profile.vocab_size != VOCAB_SIZE
        ):
            raise ValueError("profile values do not match the pinned proof contract")

        manifest = _load_json_object(non_vq_manifest_path, label="non-VQ manifest")
        expected_manifest_values = {
            "model_id": MODEL_ID,
            "source_revision": REVISION,
            "profile": PROFILE_NAME,
            "config_sha256": CONFIG_SHA256,
            "index_sha256": INDEX_SHA256,
            "retained_tensor_count": 1_194,
            "tensor_payload_bytes": 37_121_488_608,
            "production_ready": True,
        }
        for field, expected in expected_manifest_values.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"non-VQ manifest {field} mismatch: expected {expected!r}, "
                    f"found {manifest.get(field)!r}"
                )

        stream_convert = importlib.import_module("mlx_vq.convert.stream_convert")
        self.snapshot_dir = snapshot
        self.non_vq_package_dir = non_vq
        self.prompt_pack_path = prompt_path
        self.profile_path = profile_path
        self.prompt_pack = prompt_pack
        self.prompt_rows = tuple(rows)
        self.config = config
        self.profile = profile
        self.source_index = stream_convert.load_safetensors_index(index_path)
        self.non_vq_index = stream_convert.load_safetensors_index(non_vq_index_path)
        return {
            "snapshot_dir": str(snapshot),
            "config_path": str(config_path),
            "config_sha256": CONFIG_SHA256,
            "index_path": str(index_path),
            "index_sha256": INDEX_SHA256,
            "profile_path": str(profile_path),
            "profile_sha256": PROFILE_SHA256,
            "profile_contract_sha256": profile_contract,
            "prompt_pack_path": str(prompt_path),
            "prompt_pack_sha256": PROMPT_PACK_SHA256,
            "prompt_pack_contract_sha256": prompt_pack.get(
                "prompt_pack_contract_sha256"
            ),
            "prompt_content_contract_sha256": prompt_pack.get(
                "prompt_content_contract_sha256"
            ),
            "prompt_text_sha256": prompt_pack.get("prompt_text_sha256"),
            "prompt_count": len(rows),
            "predictor_position_count": predictor_positions,
            "non_vq_package_dir": str(non_vq),
            "non_vq_manifest_path": str(non_vq_manifest_path),
            "non_vq_manifest_sha256": _sha256_file(non_vq_manifest_path),
            "non_vq_index_sha256": _sha256_file(non_vq_index_path),
            "non_vq_package_set_sha256": manifest.get("package_set_sha256"),
        }

    def audit_non_vq_package(self) -> Any:
        if self._non_vq_audit is not None:
            return self._non_vq_audit
        assert self.snapshot_dir is not None
        assert self.non_vq_package_dir is not None
        assert self.profile_path is not None
        audit = self.producer_api.audit_glm52_non_vq_package(
            self.non_vq_package_dir,
            source_dir=self.snapshot_dir,
            source_index=self.source_index,
            profile=self.profile,
            expected_model_id=MODEL_ID,
            expected_revision=REVISION,
            expected_config_sha256=CONFIG_SHA256,
            expected_index_sha256=INDEX_SHA256,
            config_path=self.snapshot_dir / "config.json",
            index_path=self.snapshot_dir / "model.safetensors.index.json",
            enforce_pinned_source=True,
        )
        if getattr(audit, "audit_pass", False) is not True:
            raise ValueError("authenticated non-VQ package byte audit did not pass")
        if (
            audit.retained_tensor_count != 1_194
            or audit.tensor_payload_bytes != 37_121_488_608
        ):
            raise ValueError("authenticated non-VQ package totals do not match")
        self._non_vq_audit = audit
        return audit

    def expert_resolver(self, layer: int) -> Callable[[int], Mapping[str, Any]]:
        assert self.snapshot_dir is not None
        source_api = importlib.import_module("mlx_vq.models.glm52_source_teacher")
        return source_api.make_modelopt_nvfp4_expert_weight_resolver(
            self.snapshot_dir,
            self.source_index.weight_map,
            layer_index=layer,
        )

    def load_full_workload(self) -> Any:
        self.audit_non_vq_package()
        assert self.config is not None
        assert self.non_vq_package_dir is not None
        import mlx.core as mx

        from mlx.utils import tree_map_with_path
        from mlx_vq.models.glm52_vq_adapter import (
            GLM52VQModel,
            bind_glm52_non_vq_weights,
            glm52_vq_args_from_config,
        )

        model = GLM52VQModel(glm52_vq_args_from_config(self.config))

        def cast_parameter(path: str, value: Any) -> Any:
            if model.cast_predicate(path) and mx.issubdtype(value.dtype, mx.floating):
                return value.astype(mx.bfloat16)
            return value

        model.update(tree_map_with_path(cast_parameter, model.parameters()))
        bind_report = bind_glm52_non_vq_weights(
            model,
            self.non_vq_package_dir,
            self.non_vq_index,
            strict=True,
        )
        if bind_report.loaded_count != 1_272:
            raise ValueError("strict full non-VQ bind did not load 1272 runtime targets")
        model.eval()
        mx.eval(model.parameters())
        return self.producer_api.GLM52SourceTeacherWorkload(
            model=model,
            expert_resolver_for_layer=self.expert_resolver,
            expected_num_layers=MAIN_LAYER_COUNT,
        )


def _bf16_bytes(value: Any) -> bytes:
    import mlx.core as mx

    if value.dtype != mx.bfloat16:
        raise ValueError("byte comparison requires a bfloat16 MLX array")
    words = np.array(mx.contiguous(value).view(mx.uint16))
    return np.ascontiguousarray(words.astype("<u2", copy=False)).tobytes()


def _step_9(context: RuntimeContext) -> dict[str, Any]:
    assert context.snapshot_dir is not None
    nvfp4 = importlib.import_module("mlx_vq.convert.nvfp4")

    def resolve(oracle: DecodedOracle) -> bytes:
        weight_name = (
            f"model.layers.{oracle.layer}.mlp.experts.{oracle.expert}."
            f"{oracle.projection}.weight"
        )
        bundle = nvfp4.resolve_modelopt_nvfp4_weight_bundle(
            context.source_index.weight_map,
            weight_name,
        )
        decoded = nvfp4.read_modelopt_nvfp4_weight(context.snapshot_dir, bundle)
        if decoded.dtype != np.float32 or not decoded.flags.c_contiguous:
            raise ValueError("decoded oracle weight must be C-contiguous float32")
        return decoded.tobytes(order="C")

    oracle_results = verify_decoded_oracles(
        DECODED_ORACLES,
        decoded_bytes_resolver=resolve,
    )
    return {
        "proof_pass": True,
        "proof": "pinned_decoded_byte_oracles",
        "oracle_count": len(oracle_results),
        "oracles": oracle_results,
    }


def _step_10(context: RuntimeContext) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn

    source_api = importlib.import_module("mlx_vq.models.glm52_source_teacher")
    rng = np.random.default_rng(context.args.seed)
    hidden_fp32 = rng.normal(0.0, 0.02, size=(4, HIDDEN_SIZE)).astype(np.float32)
    hidden = mx.array(hidden_fp32).astype(mx.bfloat16)
    mx.eval(hidden)

    def one_expert_gate(value: Any) -> tuple[Any, Any]:
        return (
            mx.zeros((value.shape[0], 1), dtype=mx.int32),
            mx.ones((value.shape[0], 1), dtype=mx.float32),
        )

    streamed = source_api.stream_selected_expert_moe(
        hidden,
        gate=one_expert_gate,
        expert_weight_resolver=context.expert_resolver(SPARSE_PROOF_LAYER),
        shared_expert=None,
        num_routed_experts=168,
    )
    mx.eval(streamed.output)

    independent_sources = context.expert_resolver(SPARSE_PROOF_LAYER)(
        EXPERT_PROOF_INDEX
    )
    decoded = {
        projection: independent_sources[projection]()
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    if any(value.dtype != np.float32 for value in decoded.values()):
        raise ValueError("independent dense reference must decode all weights to FP32")
    weights = {
        projection: mx.array(value).astype(mx.bfloat16)
        for projection, value in decoded.items()
    }
    gate_output = hidden @ weights["gate_proj"].T
    up_output = hidden @ weights["up_proj"].T
    activated = nn.silu(gate_output) * up_output
    dense_reference = activated @ weights["down_proj"].T
    mx.eval(dense_reference)
    streamed_bytes = _bf16_bytes(streamed.output)
    reference_bytes = _bf16_bytes(dense_reference)
    if streamed_bytes != reference_bytes:
        raise AssertionError(
            "real expert streamed projection output is not byte-identical to "
            "the independent dense BF16 composition"
        )
    result = {
        "proof_pass": True,
        "proof": "one_real_expert_bundle_streamed_vs_dense",
        "layer": SPARSE_PROOF_LAYER,
        "expert": EXPERT_PROOF_INDEX,
        "seed": context.args.seed,
        "hidden_shape": list(hidden.shape),
        "hidden_dtype": str(hidden.dtype),
        "output_shape": list(streamed.output.shape),
        "output_dtype": str(streamed.output.dtype),
        "output_sha256": _sha256_bytes(streamed_bytes),
        "byte_identical": True,
        "decoded_weight_dtypes": list(streamed.precision.decoded_weight_dtypes),
        "matmul_weight_dtypes": list(streamed.precision.matmul_weight_dtypes),
    }
    del decoded, weights, gate_output, up_output, activated, dense_reference, streamed
    return result


def _load_single_sparse_layer(context: RuntimeContext) -> tuple[Any, Any, Any]:
    context.audit_non_vq_package()
    assert context.config is not None
    assert context.non_vq_package_dir is not None
    import mlx.core as mx

    from mlx.utils import tree_map_with_path
    from mlx_vq.models.glm52_vq_adapter import (
        Glm52VQDecoderLayer,
        bind_glm52_decoder_layer_non_vq_weights,
        glm52_vq_args_from_config,
    )

    model_args = glm52_vq_args_from_config(context.config)
    layer = Glm52VQDecoderLayer(model_args, SPARSE_PROOF_LAYER)

    def cast_parameter(path: str, value: Any) -> Any:
        if "e_score_correction_bias" not in path and mx.issubdtype(
            value.dtype, mx.floating
        ):
            return value.astype(mx.bfloat16)
        return value

    layer.update(tree_map_with_path(cast_parameter, layer.parameters()))
    bind_report = bind_glm52_decoder_layer_non_vq_weights(
        layer,
        context.non_vq_package_dir,
        context.non_vq_index,
        model_args=model_args,
        strict=True,
    )
    if bind_report.missing_model_parameters or bind_report.skipped_unmatched_tensors:
        raise ValueError("single-layer strict non-VQ bind was incomplete")
    layer.eval()
    mx.eval(layer.parameters())
    return layer, model_args, bind_report


def _independent_project_bf16(
    hidden: Any,
    source: Any,
    *,
    projection: str,
) -> Any:
    """Decode FP32 and perform one BF16 projection without runner helpers."""

    import mlx.core as mx

    decoded = source() if callable(source) else source
    if (
        not isinstance(decoded, np.ndarray)
        or decoded.dtype != np.float32
        or decoded.ndim != 2
        or not decoded.flags.c_contiguous
        or not np.isfinite(decoded).all()
    ):
        raise ValueError(f"{projection} source must be finite C-contiguous FP32")
    weight = mx.array(decoded).astype(mx.bfloat16)
    output = hidden @ weight.T
    mx.eval(output)
    del weight, decoded
    mx.clear_cache()
    return output


def _independent_selected_expert_moe(
    hidden: Any,
    *,
    gate: Callable[[Any], tuple[Any, Any]],
    resolver: Callable[[int], Mapping[str, Any]],
    shared_expert: Any,
) -> tuple[Any, Any, Any, tuple[int, ...]]:
    """Dense route-slot oracle independent of runner stream/scatter/reduction."""

    import mlx.core as mx
    import mlx.nn as nn

    if hidden.ndim != 2 or hidden.dtype != mx.bfloat16:
        raise ValueError("independent MoE input must be rank-2 bfloat16")
    expert_indices, route_scores = gate(hidden)
    mx.eval(expert_indices, route_scores)
    indices_np = np.array(expert_indices)
    if (
        route_scores.dtype != mx.float32
        or indices_np.ndim != 2
        or tuple(route_scores.shape) != tuple(indices_np.shape)
        or int(indices_np.shape[0]) != int(hidden.shape[0])
        or not np.issubdtype(indices_np.dtype, np.integer)
        or np.any(indices_np < 0)
    ):
        raise ValueError(
            "independent oracle requires aligned integer IDs and FP32 scores"
        )
    token_count, top_k = indices_np.shape
    hidden_size = int(hidden.shape[-1])
    flat_outputs = mx.zeros(
        (token_count * top_k, hidden_size), dtype=mx.bfloat16
    )
    expert_order = tuple(sorted(int(value) for value in np.unique(indices_np)))
    for expert_index in expert_order:
        slots = np.flatnonzero(indices_np.reshape(-1) == expert_index)
        expert_hidden = mx.take(
            hidden,
            mx.array((slots // top_k).astype(np.int32)),
            axis=0,
        )
        sources = resolver(expert_index)
        if set(sources) != {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError("independent oracle resolver returned an invalid bundle")
        gate_output = _independent_project_bf16(
            expert_hidden, sources["gate_proj"], projection="gate_proj"
        )
        up_output = _independent_project_bf16(
            expert_hidden, sources["up_proj"], projection="up_proj"
        )
        activated = nn.silu(gate_output) * up_output
        if activated.dtype != mx.bfloat16:
            raise ValueError("independent oracle activation must use bfloat16")
        expert_output = _independent_project_bf16(
            activated, sources["down_proj"], projection="down_proj"
        )
        element_indices = (
            slots[:, None] * hidden_size
            + np.arange(hidden_size, dtype=np.int64)[None, :]
        ).reshape(-1)
        flat_outputs = mx.put_along_axis(
            flat_outputs,
            mx.array(element_indices),
            expert_output.reshape(-1),
            axis=None,
        )
        mx.eval(flat_outputs)
        del sources, gate_output, up_output, activated, expert_output
        mx.clear_cache()
    route_outputs = flat_outputs.reshape(token_count, top_k, int(hidden.shape[-1]))
    output = _independent_route_rank_reduce(
        route_outputs,
        route_scores,
        float32_dtype=mx.float32,
        output_dtype=mx.bfloat16,
    )
    if shared_expert is not None:
        shared_output = shared_expert(hidden)
        if shared_output.shape != hidden.shape or shared_output.dtype != mx.bfloat16:
            raise ValueError("independent shared expert must return aligned bfloat16")
        output = output + shared_output
        del shared_output
    output = output.astype(mx.bfloat16)
    mx.eval(output)
    return output, expert_indices, route_scores, expert_order


def _independent_scatter_valid_rows(
    valid_output: Any,
    *,
    valid_flat_indices: np.ndarray,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
) -> Any:
    """Scatter valid rows without importing the runner's scatter primitive."""

    import mlx.core as mx

    flat = mx.zeros((batch_size * sequence_length, hidden_size), dtype=mx.bfloat16)
    element_indices = (
        valid_flat_indices[:, None] * hidden_size
        + np.arange(hidden_size, dtype=np.int64)[None, :]
    ).reshape(-1)
    scattered = mx.put_along_axis(
        flat,
        mx.array(element_indices),
        valid_output.reshape(-1),
        axis=None,
    )
    return scattered.reshape(batch_size, sequence_length, hidden_size)


def _production_single_layer_pass(
    *,
    layer: Any,
    hidden: Any,
    causal_mask: Any,
    valid_flat_indices: np.ndarray,
    resolver: Callable[[int], Mapping[str, Any]],
) -> tuple[Any, Any, Any, tuple[int, ...]]:
    import mlx.core as mx

    source_api = importlib.import_module("mlx_vq.models.glm52_source_teacher")
    attention_output, topk_indices = layer.self_attn(
        layer.input_layernorm(hidden), causal_mask, None, None
    )
    if topk_indices is not None:
        raise ValueError("short single-layer proof requires absent IndexShare state")
    attention_hidden = hidden + attention_output
    mlp_hidden = layer.post_attention_layernorm(attention_hidden)
    valid_hidden = mx.take(
        mlp_hidden.reshape(-1, HIDDEN_SIZE),
        mx.array(valid_flat_indices),
        axis=0,
    )
    streamed = source_api.stream_selected_expert_moe(
        valid_hidden,
        gate=layer.mlp.gate,
        expert_weight_resolver=resolver,
        shared_expert=layer.mlp.get("shared_experts"),
        num_routed_experts=168,
    )
    batch_size, sequence_length = hidden.shape[:2]
    scattered = source_api._scatter_valid_rows(
        streamed.output,
        valid_flat_indices=valid_flat_indices,
        batch_size=int(batch_size),
        sequence_length=int(sequence_length),
        hidden_size=HIDDEN_SIZE,
    )
    output = mx.contiguous(attention_hidden + scattered)
    mx.eval(output)
    selected = tuple(
        sorted(int(value) for value in np.unique(np.array(streamed.expert_indices)))
    )
    return output, streamed.expert_indices, streamed.route_scores, selected


def _independent_single_layer_pass(
    *,
    layer: Any,
    hidden: Any,
    causal_mask: Any,
    valid_flat_indices: np.ndarray,
    resolver: Callable[[int], Mapping[str, Any]],
) -> tuple[Any, Any, Any, tuple[int, ...]]:
    import mlx.core as mx

    attention_output, topk_indices = layer.self_attn(
        layer.input_layernorm(hidden), causal_mask, None, None
    )
    if topk_indices is not None:
        raise ValueError("independent single-layer proof requires absent IndexShare state")
    attention_hidden = hidden + attention_output
    mlp_hidden = layer.post_attention_layernorm(attention_hidden)
    valid_hidden = mx.take(
        mlp_hidden.reshape(-1, HIDDEN_SIZE),
        mx.array(valid_flat_indices),
        axis=0,
    )
    oracle, expert_ids, scores, order = _independent_selected_expert_moe(
        valid_hidden,
        gate=layer.mlp.gate,
        resolver=resolver,
        shared_expert=layer.mlp.get("shared_experts"),
    )
    batch_size, sequence_length = hidden.shape[:2]
    scattered = _independent_scatter_valid_rows(
        oracle,
        valid_flat_indices=valid_flat_indices,
        batch_size=int(batch_size),
        sequence_length=int(sequence_length),
        hidden_size=HIDDEN_SIZE,
    )
    output = mx.contiguous(attention_hidden + scattered)
    mx.eval(output)
    return output, expert_ids, scores, order


def _step_11(context: RuntimeContext) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.models.base import create_causal_mask

    source_api = importlib.import_module("mlx_vq.models.glm52_source_teacher")
    layer, model_args, bind_report = _load_single_sparse_layer(context)
    encoded = tuple(tuple(row["encoded_token_ids"]) for row in context.prompt_rows)
    token_batch, valid_mask, right_padding = source_api._prepare_predictor_batch(
        encoded,
        vocab_size=VOCAB_SIZE,
        pad_token_id=0,
    )
    valid_flat_indices = np.flatnonzero(valid_mask.reshape(-1)).astype(np.int64)
    if len(valid_flat_indices) != 744:
        raise AssertionError("complete frozen prompt batch must contain 744 valid rows")
    rng = np.random.default_rng(context.args.seed + 11)
    hidden_values = rng.normal(
        0.0,
        0.02,
        size=(token_batch.shape[0], token_batch.shape[1], HIDDEN_SIZE),
    ).astype(np.float32)
    hidden_values[~valid_mask] = 0.0
    hidden = mx.array(hidden_values).astype(mx.bfloat16)
    causal_mask = create_causal_mask(
        token_batch.shape[1], right_padding=mx.array(right_padding)
    )
    mx.eval(hidden, causal_mask)
    resolver = context.expert_resolver(SPARSE_PROOF_LAYER)
    production, production_ids, production_scores, production_order = (
        _production_single_layer_pass(
            layer=layer,
            hidden=hidden,
            causal_mask=causal_mask,
            valid_flat_indices=valid_flat_indices,
            resolver=resolver,
        )
    )
    oracle, oracle_ids, oracle_scores, oracle_order = _independent_single_layer_pass(
        layer=layer,
        hidden=hidden,
        causal_mask=causal_mask,
        valid_flat_indices=valid_flat_indices,
        resolver=resolver,
    )
    production_ids_np = np.array(production_ids)
    oracle_ids_np = np.array(oracle_ids)
    production_scores_np = np.ascontiguousarray(
        np.array(production_scores).astype(np.float32)
    )
    oracle_scores_np = np.ascontiguousarray(
        np.array(oracle_scores).astype(np.float32)
    )
    if not np.array_equal(production_ids_np, oracle_ids_np):
        raise AssertionError("production and independent oracle route IDs differ")
    if production_scores_np.tobytes() != oracle_scores_np.tobytes():
        raise AssertionError("production and independent oracle route scores differ")
    assignments = int(production_ids_np.size)
    expected_assignments = len(valid_flat_indices) * EXPERTS_PER_TOKEN
    if assignments != expected_assignments or assignments != 5_952:
        raise AssertionError("single sparse layer route assignments are not n_valid * 8")
    production_bytes = _bf16_bytes(production)
    oracle_bytes = _bf16_bytes(oracle)
    if production_bytes != oracle_bytes:
        raise AssertionError(
            "production sparse layer is not byte-identical to the independent "
            "dense route-slot oracle"
        )
    return {
        "proof_pass": True,
        "proof": "one_complete_real_sparse_layer_vs_independent_dense_route_slots",
        "layer": SPARSE_PROOF_LAYER,
        "seed": context.args.seed + 11,
        "prompt_count": len(context.prompt_rows),
        "predictor_batch_shape": list(token_batch.shape),
        "valid_hidden_rows": len(valid_flat_indices),
        "experts_per_token": int(model_args.num_experts_per_tok),
        "valid_route_assignments": assignments,
        "expected_route_assignments": expected_assignments,
        "padded_route_assignments": 0,
        "selected_expert_count": len(production_order),
        "production_expert_order": list(production_order),
        "oracle_expert_order": list(oracle_order),
        "independent_moe_oracle": True,
        "independent_route_slot_scatter": True,
        "independent_route_rank_reduction": True,
        "shared_attention_modules": True,
        "shared_router_and_shared_expert_modules": True,
        "shared_expert_weight_resolver": True,
        "precision_contract": (
            "FP32 decode -> BF16 matmul -> FP32 route scores -> "
            "route-rank sum -> BF16"
        ),
        "non_vq_loaded_parameter_count": bind_report.loaded_count,
        "output_shape": list(production.shape),
        "output_dtype": str(production.dtype),
        "output_sha256": _sha256_bytes(production_bytes),
        "byte_identical": True,
    }


def _independent_prepare_predictor_batch(
    encoded_token_ids: tuple[tuple[int, ...], ...],
    *,
    vocab_size: int,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare a right-padded predictor batch without runner prompt helpers."""

    if not encoded_token_ids:
        raise ValueError("independent forward requires at least one prompt")
    predictor_lengths: list[int] = []
    for prompt_index, row in enumerate(encoded_token_ids):
        if len(row) < 2:
            raise ValueError(f"independent prompt {prompt_index} is too short")
        if any(
            type(token) is not int or not 0 <= token < vocab_size for token in row
        ):
            raise ValueError(f"independent prompt {prompt_index} has an invalid token")
        predictor_lengths.append(len(row) - 1)
    lengths = np.array(predictor_lengths, dtype=np.int32)
    max_length = int(lengths.max())
    token_batch = np.full(
        (len(encoded_token_ids), max_length), pad_token_id, dtype=np.int32
    )
    valid_mask = np.zeros(token_batch.shape, dtype=np.bool_)
    for prompt_index, row in enumerate(encoded_token_ids):
        predictor = row[:-1]
        token_batch[prompt_index, : len(predictor)] = predictor
        valid_mask[prompt_index, : len(predictor)] = True
    return token_batch, valid_mask, max_length - lengths


def _run_independent_moe_source_teacher(
    workload: Any,
    encoded_token_ids: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, ...]:
    """Full forward sharing model modules but replacing every sparse MoE path."""

    import mlx.core as mx
    from mlx_lm.models.base import create_causal_mask
    from mlx_vq.models.glm52_vq_adapter import Glm52VQMoE

    model = workload.model
    if (
        model.args.num_hidden_layers != MAIN_LAYER_COUNT
        or model.model.start_idx != 0
        or model.model.end_idx != MAIN_LAYER_COUNT
        or len(model.model.layers) != MAIN_LAYER_COUNT
    ):
        raise ValueError("independent source teacher requires all 78 main layers")
    token_batch, valid_mask, right_padding = _independent_prepare_predictor_batch(
        encoded_token_ids,
        vocab_size=model.args.vocab_size,
        pad_token_id=workload.pad_token_id,
    )
    batch_size, sequence_length = token_batch.shape
    if sequence_length > model.args.index_topk:
        raise ValueError("independent forward requires absent IndexShare state")
    hidden = mx.contiguous(model.model.embed_tokens(mx.array(token_batch)))
    if hidden.dtype != mx.bfloat16:
        raise ValueError("independent embedded hidden state must use bfloat16")
    causal_mask = create_causal_mask(
        sequence_length,
        right_padding=mx.array(right_padding),
    )
    mx.eval(hidden, causal_mask)
    valid_flat_indices = np.flatnonzero(valid_mask.reshape(-1)).astype(np.int64)
    prev_topk_indices = None
    for layer_number, layer in enumerate(model.model.layers):
        if int(layer.layer_idx) != layer_number:
            raise ValueError("independent decoder layer identity mismatch")
        attention_output, next_topk_indices = layer.self_attn(
            layer.input_layernorm(hidden),
            causal_mask,
            None,
            prev_topk_indices,
        )
        if next_topk_indices is not None:
            raise ValueError("independent forward observed unexpected IndexShare state")
        prev_topk_indices = next_topk_indices
        attention_hidden = hidden + attention_output
        mlp_hidden = layer.post_attention_layernorm(attention_hidden)
        if isinstance(layer.mlp, Glm52VQMoE):
            valid_hidden = mx.take(
                mlp_hidden.reshape(-1, model.args.hidden_size),
                mx.array(valid_flat_indices),
                axis=0,
            )
            oracle_output, expert_ids, _scores, _order = (
                _independent_selected_expert_moe(
                    valid_hidden,
                    gate=layer.mlp.gate,
                    resolver=workload.expert_resolver_for_layer(layer_number),
                    shared_expert=layer.mlp.get("shared_experts"),
                )
            )
            expected_assignments = (
                len(valid_flat_indices) * model.args.num_experts_per_tok
            )
            if int(expert_ids.size) != expected_assignments:
                raise AssertionError(
                    "independent sparse assignments do not equal n_valid * top_k"
                )
            sparse_output = _independent_scatter_valid_rows(
                oracle_output,
                valid_flat_indices=valid_flat_indices,
                batch_size=int(batch_size),
                sequence_length=int(sequence_length),
                hidden_size=int(model.args.hidden_size),
            )
            hidden = attention_hidden + sparse_output
        else:
            hidden = attention_hidden + layer.mlp(mlp_hidden)
        hidden = mx.contiguous(hidden)
        if hidden.dtype != mx.bfloat16:
            raise ValueError("independent post-layer hidden state must use bfloat16")
        mx.eval(hidden)

    normalized = model.model.norm(hidden)
    if normalized.dtype != mx.bfloat16:
        raise ValueError("independent final norm must return bfloat16")
    predictor_lengths = valid_mask.sum(axis=1).astype(np.int64)
    logits_by_prompt: list[np.ndarray] = []
    for prompt_index, predictor_length in enumerate(predictor_lengths):
        prompt_hidden = normalized[
            prompt_index : prompt_index + 1,
            : int(predictor_length),
            :,
        ]
        prompt_logits = model.lm_head(prompt_hidden).astype(mx.float32).reshape(
            int(predictor_length), model.args.vocab_size
        )
        mx.eval(prompt_logits)
        logits_by_prompt.append(
            np.ascontiguousarray(
                np.array(prompt_logits).astype(np.float32, copy=False)
            )
        )
    return tuple(logits_by_prompt)


def _step_12(context: RuntimeContext) -> dict[str, Any]:
    source_api = importlib.import_module("mlx_vq.models.glm52_source_teacher")
    workload = context.load_full_workload()
    row = context.prompt_rows[0]
    prompt = context.producer_api.GLM52TeacherCachePrompt.from_prompt_pack_row(row)
    identities = source_api.CheckpointIdentities(
        prompt_sha256=PROMPT_PACK_SHA256,
        source_sha256=_canonical_sha256(
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "config_sha256": CONFIG_SHA256,
                "index_sha256": INDEX_SHA256,
            }
        ),
        profile_sha256=PROFILE_SHA256,
        config_sha256=CONFIG_SHA256,
        policy_sha256=_sha256_bytes(
            b"glm52_source_teacher_bounded_proof_step12_policy_v1"
        ),
        precision_sha256=_sha256_bytes(
            b"glm52_source_teacher_bounded_proof_bf16_compute_fp32_logits_v1"
        ),
    )
    output_parent = (
        Path(context.args.output_json).expanduser().resolve(strict=False).parent
    )
    output_parent.mkdir(parents=True, exist_ok=True)
    midpoint_layer = 40
    midpoint_checkpoint_file_count = 0
    with tempfile.TemporaryDirectory(
        prefix="glm52-bounded-step12-checkpoints-", dir=output_parent
    ) as checkpoint_dir:
        checkpoint_root = Path(checkpoint_dir)

        def interrupting_run() -> None:
            nonlocal midpoint_checkpoint_file_count
            try:
                source_api.run_glm52_source_teacher(
                    workload.model,
                    (tuple(prompt.encoded_token_ids),),
                    expert_resolver_for_layer=workload.expert_resolver_for_layer,
                    expected_num_layers=MAIN_LAYER_COUNT,
                    pad_token_id=workload.pad_token_id,
                    capture_route_trace=False,
                    checkpoint_dir=checkpoint_root,
                    checkpoint_identities=identities,
                    interrupt_after_layer=midpoint_layer,
                )
            finally:
                midpoint_checkpoint_file_count = len(
                    tuple(checkpoint_root.glob("layer-*.safetensors"))
                )

        def resumed_run() -> tuple[tuple[np.ndarray, ...], Any]:
            return source_api.run_glm52_source_teacher(
                workload.model,
                (tuple(prompt.encoded_token_ids),),
                expert_resolver_for_layer=workload.expert_resolver_for_layer,
                expected_num_layers=MAIN_LAYER_COUNT,
                pad_token_id=workload.pad_token_id,
                capture_route_trace=False,
                checkpoint_dir=checkpoint_root,
                checkpoint_identities=identities,
            )

        checkpointed, checkpointed_traces = run_checkpoint_reload_probe(
            interrupting_run=interrupting_run,
            resumed_run=resumed_run,
            interruption_type=source_api.SourceTeacherInterrupted,
            clear_live_state=_clear_runtime_memory,
        )
        if checkpointed_traces is not None or len(checkpointed) != 1:
            raise AssertionError("resumed checkpoint path returned unexpected outputs")
        if midpoint_checkpoint_file_count != midpoint_layer + 1:
            raise AssertionError("interrupted path did not persist through layer 40")
        checkpoint_files = sorted(checkpoint_root.glob("layer-*.safetensors"))
        if len(checkpoint_files) != MAIN_LAYER_COUNT:
            raise AssertionError(
                "checkpointed full path did not write exactly 78 layers"
            )
        checkpoint_file_count = len(checkpoint_files)
        checkpoint_output_identity_sha256 = _sha256_file(checkpoint_files[-1])
    _clear_runtime_memory()
    independent = _run_independent_moe_source_teacher(
        workload,
        (tuple(prompt.encoded_token_ids),),
    )
    if len(independent) != 1:
        raise AssertionError("independent full path returned unexpected outputs")
    first = np.ascontiguousarray(checkpointed[0], dtype=np.float32)
    second = np.ascontiguousarray(independent[0], dtype=np.float32)
    expected_shape = (len(prompt.encoded_token_ids) - 1, VOCAB_SIZE)
    if first.shape != expected_shape or second.shape != expected_shape:
        raise AssertionError(
            f"step 12 logits shape must be {list(expected_shape)}, found "
            f"{list(first.shape)} and {list(second.shape)}"
        )
    if first.dtype != np.float32 or second.dtype != np.float32:
        raise AssertionError("step 12 logits must use float32")
    if first.tobytes(order="C") != second.tobytes(order="C"):
        raise AssertionError(
            "reloaded checkpoint logits are not byte-identical to the independent "
            "MoE-oracle execution"
        )
    if not np.isfinite(first).all():
        raise AssertionError("step 12 logits contain non-finite values")
    non_constant_rows = [bool(np.ptp(row_values) > 0) for row_values in first]
    if not all(non_constant_rows):
        raise AssertionError("step 12 logits contain a constant vocabulary row")
    logits_sha256 = _sha256_bytes(first.tobytes(order="C"))
    return {
        "proof_pass": True,
        "proof": "one_frozen_prompt_reloaded_checkpoint_vs_independent_moe_oracle",
        "prompt_index": 0,
        "prompt_id": prompt.prompt_id,
        "token_count": prompt.token_count,
        "predictor_position_count": prompt.token_count - 1,
        "main_layer_count": MAIN_LAYER_COUNT,
        "excluded_mtp_layer": 78,
        "checkpoint_file_count": checkpoint_file_count,
        "checkpoint_output_identity_sha256": checkpoint_output_identity_sha256,
        "checkpoint_boundary_calls": 0,
        "checkpoint_resume_boundary_count": 1,
        "simulated_interruption_after_layer": midpoint_layer,
        "midpoint_checkpoint_file_count": midpoint_checkpoint_file_count,
        "resumed_from_persisted_checkpoint_bytes": True,
        "independent_moe_oracle": True,
        "independent_route_slot_scatter": True,
        "independent_route_rank_reduction": True,
        "shared_attention_modules": True,
        "shared_norm_and_lm_head_modules": True,
        "shared_embedding_and_dense_mlp_modules": True,
        "shared_router_and_shared_expert_modules": True,
        "shared_expert_weight_resolver": True,
        "shared_prompt_preparation": False,
        "logits_shape": list(first.shape),
        "logits_dtype": str(first.dtype),
        "logits_sha256": logits_sha256,
        "all_finite": True,
        "non_constant_rows": non_constant_rows,
        "byte_identical": True,
    }


def _refresh_memory_summary(evidence: dict[str, Any]) -> None:
    phases = evidence["phases"]
    evidence["all_phase_memory_counters_known"] = bool(phases) and all(
        phase["memory_counters_known"] is True for phase in phases
    )
    evidence["all_phase_memory_clean"] = bool(phases) and all(
        phase["memory_clean"] is True for phase in phases
    )


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected_steps = select_steps(args.only_step, args.skip_steps)
    output_path = Path(args.output_json).expanduser().resolve(strict=False)
    evidence = new_evidence(args, selected_steps=selected_steps)
    producer_api = _load_producer_api()
    locks = producer_api.FileProducerLockProvider()
    run_lock_path = Path(f"{output_path}.lock")
    try:
        with locks.heavy_job_lock(producer_api.HEAVY_JOB_LOCK_PATH):
            with locks.run_lock(run_lock_path):
                evidence["locks"]["locks_acquired"] = True
                context = RuntimeContext(args, producer_api)
                authenticated = context.authenticate_inputs()
                evidence["authenticated_inputs"] = authenticated
                evidence["input_identity_sha256"] = _canonical_sha256(authenticated)
                _write_json_atomic(output_path, evidence)
                proof_functions: dict[int, Callable[[], dict[str, Any]]] = {
                    9: lambda: _step_9(context),
                    10: lambda: _step_10(context),
                    11: lambda: _step_11(context),
                    12: lambda: _step_12(context),
                }
                for ordinal, step in enumerate(selected_steps, start=1):
                    try:
                        result, phase = _run_phase(
                            ordinal=ordinal,
                            step=step,
                            reader=producer_api.collect_vm_stat_counts,
                            action=proof_functions[step],
                        )
                    except _PhaseFailure as failure:
                        evidence["phases"].append(failure.phase)
                        evidence["proofs"][f"step_{step}"] = {
                            "step": step,
                            "status": "failed",
                            "proof_pass": False,
                            "error": (
                                f"{type(failure.original).__name__}: {failure.original}"
                            ),
                        }
                        for later in selected_steps[ordinal:]:
                            evidence["proofs"][f"step_{later}"]["status"] = (
                                "not_run_due_to_prior_failure"
                            )
                        _refresh_memory_summary(evidence)
                        evidence["bounded_proof_pass"] = False
                        _write_json_atomic(output_path, evidence)
                        raise failure.original
                    evidence["phases"].append(phase)
                    evidence["proofs"][f"step_{step}"] = {
                        "step": step,
                        "status": "passed",
                        **result,
                    }
                    _refresh_memory_summary(evidence)
                    _write_json_atomic(output_path, evidence)
                evidence["bounded_proof_pass"] = all(
                    evidence["proofs"][f"step_{step}"]["proof_pass"] is True
                    for step in selected_steps
                )
                evidence["completed_at"] = _utc_now()
                _refresh_memory_summary(evidence)
                _write_json_atomic(output_path, evidence)
    except Exception as error:
        if "error" not in evidence:
            evidence["error"] = f"{type(error).__name__}: {error}"
        evidence["bounded_proof_pass"] = False
        try:
            _write_json_atomic(output_path, evidence)
        except Exception:
            pass
        raise
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_cli(argv)
    except Exception as error:
        print(
            json.dumps(
                {
                    "bounded_proof_pass": False,
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
