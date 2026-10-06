"""Read-only observation of a declared GLM-5.2 recovery campaign."""

from __future__ import annotations

import errno
import ctypes
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from mlx_vq.io.schema import codebook_metadata_for_bits

from .models import (
    CampaignConfig,
    CampaignObservation,
    Experiment,
    ExperimentObservation,
    ProcessObservation,
)


_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_ARTIFACT_GROUP_COUNT = 225
_OPEN_BASE = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
_OPEN_NOFOLLOW = _OPEN_BASE | getattr(os, "O_NOFOLLOW", 0)
_OPEN_DIRECTORY = _OPEN_NOFOLLOW | getattr(os, "O_DIRECTORY", 0)
_E8_TENSOR_PAYLOAD_BYTES = 272_499_712
_E8P_TENSOR_PAYLOAD_BYTES = 536_740_864
_NON_ROUTED_PAYLOAD_BYTES = 37_121_488_608
_RECOVERY_LEVER = "selection_diagonal_hessian_importance_weighted_reround_v1"
_SCALE_ESTIMATOR = "importance_weighted_least_squares"
_ROUNDING_OBJECTIVE = "selection_diagonal_hessian_weighted_squared_error"
_SOURCE_PROFILE = "glm52-reap-504b-v2"


@dataclass(frozen=True, slots=True)
class _ManifestObservation:
    state: str
    file_sha256: str | None
    body_sha256: str | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _LockProbe:
    state: str
    held: bool
    device: int | None = None
    inode: int | None = None
    error: str | None = None


def _probe_lock(path: Path) -> _LockProbe:
    """Inspect an existing advisory lock without ever creating it."""

    try:
        before = path.lstat()
    except FileNotFoundError:
        return _LockProbe("absent", False)
    except OSError as error:
        return _LockProbe("invalid", False, error=str(error))
    if not stat.S_ISREG(before.st_mode):
        return _LockProbe("invalid", False, error="lock path is not a regular file")
    try:
        descriptor = os.open(path, _OPEN_NOFOLLOW)
    except OSError as error:
        return _LockProbe("invalid", False, error=str(error))
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            return _LockProbe(
                "invalid",
                False,
                error="lock identity changed while opening",
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                return _LockProbe(
                    "held",
                    True,
                    device=opened.st_dev,
                    inode=opened.st_ino,
                )
            return _LockProbe("invalid", False, error=str(error))
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            return _LockProbe(
                "free",
                False,
                device=opened.st_dev,
                inode=opened.st_ino,
            )
    finally:
        os.close(descriptor)


def _elapsed_seconds(value: str) -> float | None:
    try:
        day_text, clock = value.split("-", 1) if "-" in value else ("0", value)
        fields = clock.split(":")
        if len(fields) == 2:
            hours, minutes, seconds = 0, int(fields[0]), int(fields[1])
        elif len(fields) == 3:
            hours, minutes, seconds = map(int, fields)
        else:
            return None
        days = int(day_text)
        if min(days, hours, minutes, seconds) < 0 or minutes >= 60 or seconds >= 60:
            return None
        return float(days * 86_400 + hours * 3_600 + minutes * 60 + seconds)
    except ValueError:
        return None


def _read_ps_output() -> str:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,etime="],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _parse_lsof_openers(output: str, *, device: int, inode: int) -> tuple[int, ...]:
    matches: set[int] = set()
    pid: int | None = None
    observed_device: int | None = None
    observed_inode: int | None = None

    def commit() -> None:
        if (
            pid is not None
            and observed_device == device
            and observed_inode == inode
        ):
            matches.add(pid)

    for raw in output.splitlines():
        if not raw:
            continue
        field, value = raw[0], raw[1:]
        if field == "p":
            commit()
            try:
                pid = int(value)
            except ValueError:
                pid = None
            observed_device = None
            observed_inode = None
        elif field == "D":
            try:
                observed_device = int(value, 0)
            except ValueError:
                observed_device = None
        elif field == "i":
            try:
                observed_inode = int(value)
            except ValueError:
                observed_inode = None
    commit()
    return tuple(sorted(matches))


def _read_lock_openers(
    path: Path,
    *,
    device: int,
    inode: int,
) -> tuple[int, ...] | None:
    try:
        completed = subprocess.run(
            ["lsof", "-F", "pDi", "--", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode not in {0, 1}:
        return None
    return _parse_lsof_openers(completed.stdout, device=device, inode=inode)


def _read_macos_process_argv(pid: int) -> tuple[str, ...] | None:
    libc = ctypes.CDLL(None, use_errno=True)
    sysctl = libc.sysctl
    sysctl.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    mib = (ctypes.c_int * 3)(1, 49, pid)  # CTL_KERN, KERN_PROCARGS2, pid
    size = ctypes.c_size_t(0)
    if sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0 or size.value < 4:
        return None
    buffer = ctypes.create_string_buffer(size.value)
    if sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
        return None
    payload = buffer.raw[: size.value]
    argc = int.from_bytes(payload[:4], byteorder=sys.byteorder, signed=True)
    if argc <= 0:
        return None
    cursor = 4
    executable_end = payload.find(b"\0", cursor)
    if executable_end < 0:
        return None
    cursor = executable_end + 1
    while cursor < len(payload) and payload[cursor] == 0:
        cursor += 1
    argv: list[str] = []
    for _ in range(argc):
        end = payload.find(b"\0", cursor)
        if end < 0:
            return None
        argv.append(os.fsdecode(payload[cursor:end]))
        cursor = end + 1
    return tuple(argv)


def _read_process_argv(pid: int) -> tuple[str, ...] | None:
    if sys.platform == "darwin":
        return _read_macos_process_argv(pid)
    if sys.platform.startswith("linux"):
        try:
            payload = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return None
        return tuple(os.fsdecode(item) for item in payload.split(b"\0") if item)
    return None


def _matching_processes(
    config: CampaignConfig, ps_output: str
) -> tuple[ProcessObservation, ...]:
    transitions = {transition.argv: transition for transition in config.transitions}
    observations: list[ProcessObservation] = []
    for raw_line in ps_output.splitlines():
        fields = raw_line.strip().split(None, 2)
        if len(fields) < 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        command = _read_process_argv(pid)
        if command is None:
            continue
        transition = transitions.get(command)
        if transition is None:
            continue
        observations.append(
            ProcessObservation(
                pid=pid,
                command=command,
                elapsed_seconds=_elapsed_seconds(fields[1]),
                experiment_name=transition.experiment_name,
                transition_name=transition.name,
            )
        )
    return tuple(sorted(observations, key=lambda observation: observation.pid))


def _open_directory(root: Path, relative: PurePosixPath) -> int:
    descriptor = os.open(root, _OPEN_BASE | getattr(os, "O_DIRECTORY", 0))
    try:
        for component in relative.parts:
            next_descriptor = os.open(component, _OPEN_DIRECTORY, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _stable_regular_bytes(
    root: Path, relative: PurePosixPath
) -> tuple[str, bytes | None, str | None]:
    """Read a regular descendant through no-follow directory descriptors."""

    try:
        parent = _open_directory(root, relative.parent)
    except FileNotFoundError:
        return "absent", None, None
    except OSError as error:
        return "invalid", None, str(error)
    try:
        try:
            descriptor = os.open(relative.name, _OPEN_NOFOLLOW, dir_fd=parent)
        except FileNotFoundError:
            return "absent", None, None
        except OSError as error:
            return "invalid", None, str(error)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                return "invalid", None, "manifest is not a regular file"
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            visible = os.stat(relative.name, dir_fd=parent, follow_symlinks=False)
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            identity_visible = (
                visible.st_dev,
                visible.st_ino,
                visible.st_size,
                visible.st_mtime_ns,
            )
            if identity_before != identity_after or identity_after != identity_visible:
                return "invalid", None, "manifest changed while it was read"
            return "present", b"".join(chunks), None
        except OSError as error:
            return "invalid", None, str(error)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON value {value}")


def _require_exact_fields(label: str, value: object, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} must have exact fields; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_contract(config: CampaignConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    lineage = {
        "source_model_id": config.model_id,
        "source_revision": config.model_revision,
        "source_config_sha256": config.config_sha256,
        "source_index_sha256": config.index_sha256,
        "source_profile": _SOURCE_PROFILE,
    }
    verification = {
        "source_blob_inventory_sha256": config.full_source_blob_inventory_sha256,
        "routed_source_blob_inventory_sha256": config.routed_source_blob_inventory_sha256,
        "shard_count": None,
    }
    return lineage, verification


def _expected_recovery_policy(source_lineage: dict[str, Any]) -> dict[str, Any]:
    return {
        "decoded_expert_working_set": "one_per_worker",
        "recovery_lever": _RECOVERY_LEVER,
        "rounding_objective": _ROUNDING_OBJECTIVE,
        "scale_estimator": _SCALE_ESTIMATOR,
        "source_config_sha256": source_lineage["source_config_sha256"],
        "source_decoder": "modelopt_nvfp4_v1",
        "source_index_sha256": source_lineage["source_index_sha256"],
        "source_model_id": source_lineage["source_model_id"],
        "source_profile": source_lineage["source_profile"],
        "source_revision": source_lineage["source_revision"],
        "source_weight_encoding": "modelopt_nvfp4",
    }


def _effective_layer_bits(experiment: Experiment) -> dict[int, int]:
    bits = {layer: 8 for layer in experiment.recovered_layers}
    for rates in experiment.projection_rates:
        if len(set(rates.values)) != 1 or rates.values[0] not in {8, 16}:
            raise ValueError("experiment projection rates are not complete E8/E8P layers")
        bits[rates.layer] = rates.values[0]
    return bits


def _validate_manifest_contract(
    manifest: dict[str, Any],
    *,
    config: CampaignConfig,
    experiment: Experiment,
    root: Path,
) -> None:
    schema_version = manifest.get("schema_version")
    top_fields = {
        "schema_version",
        "record_type",
        "status",
        "resumable",
        "selected_groups",
        "source_lineage",
        "source_verification",
        "expected_full_source_blob_inventory_sha256",
        "expected_routed_source_blob_inventory_sha256",
        "stats_manifest_sha256",
        "accepted_composite_audit_sha256",
        "recovery_policy",
        "lever_provenance",
        "groups",
        "mixed_artifact",
        "accounting",
        "manifest_body_sha256",
    }
    if schema_version == 2:
        top_fields.add("rate_policy")
    _require_exact_fields("recovery manifest", manifest, top_fields)
    expected_groups = [
        (layer, projection)
        for layer in experiment.recovered_layers
        for projection in _PROJECTIONS
    ]
    if manifest.get("resumable") is not True:
        raise ValueError("manifest is not resumable")
    if manifest.get("selected_groups") != [
        f"{layer}:{projection}" for layer, projection in expected_groups
    ]:
        raise ValueError("manifest selected groups do not match the experiment")
    source_lineage = _require_exact_fields(
        "manifest source lineage",
        manifest.get("source_lineage"),
        {
            "source_model_id",
            "source_revision",
            "source_config_sha256",
            "source_index_sha256",
            "source_profile",
        },
    )
    expected_lineage, expected_verification = _source_contract(config)
    for field, expected in expected_lineage.items():
        actual = source_lineage.get(field)
        if expected is None:
            if not _is_sha256(actual):
                raise ValueError(f"manifest source lineage {field} is invalid")
        elif actual != expected:
            raise ValueError(f"manifest source lineage {field} is invalid")
    source_verification = _require_exact_fields(
        "manifest source verification",
        manifest.get("source_verification"),
        {
            "source_blob_inventory_sha256",
            "routed_source_blob_inventory_sha256",
            "shard_count",
        },
    )
    for field, expected in expected_verification.items():
        actual = source_verification.get(field)
        if expected is None:
            if type(actual) is not int or actual <= 0:
                raise ValueError("manifest source verification shard count is invalid")
        elif actual != expected:
            raise ValueError(f"manifest source verification {field} is invalid")
    authorities = {authority.name: authority for authority in config.authorities}
    stats_sha256 = authorities["recovery-stats"].sha256
    expected_provenance = {
        "holdout_used_for_tuning": False,
        "lever": _RECOVERY_LEVER,
        "report_used_for_tuning": False,
        "rounding_objective": _ROUNDING_OBJECTIVE,
        "scale_estimator": _SCALE_ESTIMATOR,
        "source_lineage": source_lineage,
        "source_verification": source_verification,
        "stats_manifest_sha256": stats_sha256,
    }
    if manifest.get("lever_provenance") != expected_provenance:
        raise ValueError("manifest provenance is invalid or uses non-selection tuning")
    expected_policy = _expected_recovery_policy(source_lineage)
    if manifest.get("recovery_policy") != expected_policy:
        raise ValueError("manifest recovery policy is invalid")

    records = manifest.get("groups")
    if not isinstance(records, list) or len(records) != len(expected_groups):
        raise ValueError("manifest group records are incomplete")
    common_record_fields = {
        "layer",
        "projection",
        "status",
        "artifact_path",
        "artifact_bytes",
        "artifact_sha256",
        "source_lineage",
        "recovery_policy",
        "lever_provenance",
    }
    schema2_record_fields = common_record_fields | {
        "code_bits",
        "codes_dtype",
        "codebook_name",
        "codebook_sha256",
        "tensor_payload_bytes",
        "zero_importance_policy",
    }
    layer_bits = _effective_layer_bits(experiment)
    record_groups: list[tuple[int, str]] = []
    artifact_bytes_total = 0
    for record, (expected_layer, expected_projection) in zip(
        records, expected_groups, strict=True
    ):
        record = _require_exact_fields(
            f"manifest group {expected_layer}:{expected_projection}",
            record,
            schema2_record_fields if schema_version == 2 else common_record_fields,
        )
        layer = record.get("layer")
        projection = record.get("projection")
        if type(layer) is not int or not isinstance(projection, str):
            raise ValueError("manifest group record identity is invalid")
        if record.get("status") not in {"materialized", "resumed"}:
            raise ValueError("manifest group record is not complete")
        expected_name = (
            f"layer-{expected_layer:05d}-{expected_projection}.safetensors"
        )
        if record.get("artifact_path") != expected_name:
            raise ValueError("manifest group artifact path is invalid")
        artifact_bytes = record.get("artifact_bytes")
        if type(artifact_bytes) is not int or artifact_bytes <= 0:
            raise ValueError("manifest group artifact bytes are invalid")
        artifact_bytes_total += artifact_bytes
        artifact_sha256 = record.get("artifact_sha256")
        if (
            not isinstance(artifact_sha256, str)
            or len(artifact_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in artifact_sha256
            )
        ):
            raise ValueError("manifest group artifact SHA-256 is invalid")
        if record.get("source_lineage") != source_lineage:
            raise ValueError("manifest group source lineage is invalid")
        if record.get("recovery_policy") != expected_policy:
            raise ValueError("manifest group recovery policy is invalid")
        if record.get("lever_provenance") != expected_provenance:
            raise ValueError("manifest group provenance is invalid")
        if schema_version == 2:
            bits = layer_bits[expected_layer]
            codebook_name, codebook_sha256 = codebook_metadata_for_bits(bits)
            expected_rate_fields = {
                "code_bits": bits,
                "codes_dtype": "uint16" if bits == 16 else "uint8",
                "codebook_name": codebook_name,
                "codebook_sha256": codebook_sha256,
                "tensor_payload_bytes": (
                    _E8P_TENSOR_PAYLOAD_BYTES
                    if bits == 16
                    else _E8_TENSOR_PAYLOAD_BYTES
                ),
                "zero_importance_policy": (
                    "source_rtn_e8p" if bits == 16 else "preserve_seed_expert_bytes"
                ),
            }
            for field, expected in expected_rate_fields.items():
                if record.get(field) != expected:
                    raise ValueError(f"manifest group {field} is invalid for E{bits}")
        record_groups.append((layer, projection))
    if record_groups != expected_groups:
        raise ValueError("manifest group record identities do not match the experiment")
    expected_rate_policy = {
        "complete_layer_rates_required": True,
        "default_code_bits": 8,
        "layer_code_bits": {
            str(layer): 16
            for layer, bits in layer_bits.items()
            if bits == 16
        },
    }
    if schema_version == 1:
        if any(bits != 8 for bits in layer_bits.values()):
            raise ValueError("manifest schema v1 cannot represent mixed rates")
        if "rate_policy" in manifest:
            raise ValueError("manifest schema v1 must not declare a rate policy")
    elif schema_version == 2:
        if manifest.get("rate_policy") != expected_rate_policy:
            raise ValueError("manifest rate policy does not match the experiment")
    else:
        raise ValueError("manifest schema version is invalid")
    bindings = (
        (
            "expected_full_source_blob_inventory_sha256",
            config.full_source_blob_inventory_sha256,
        ),
        (
            "expected_routed_source_blob_inventory_sha256",
            config.routed_source_blob_inventory_sha256,
        ),
        ("stats_manifest_sha256", authorities["recovery-stats"].sha256),
        (
            "accepted_composite_audit_sha256",
            authorities["accepted-composite-audit"].sha256,
        ),
    )
    for field, expected in bindings:
        if manifest.get(field) != expected:
            raise ValueError(f"manifest {field} does not match the campaign authority")
    mixed = manifest.get("mixed_artifact")
    mixed = _require_exact_fields(
        "manifest mixed artifact",
        mixed,
        {
            "seed_artifact_dir",
            "seed_manifest_sha256",
            "output_dir",
            "recovered_groups_dir",
            "replacement_count",
            "inherited_group_count",
        },
    )
    expected_count = len(expected_groups)
    expected_output = root.absolute() / experiment.output_path
    expected_seed = root.absolute() / Path(
        authorities["accepted-seed-manifest"].path
    ).parent
    expected_mixed = (
        ("replacement_count", expected_count),
        ("inherited_group_count", _ARTIFACT_GROUP_COUNT - expected_count),
        (
            "seed_manifest_sha256",
            authorities["accepted-seed-manifest"].sha256,
        ),
        ("output_dir", str(expected_output / "artifact")),
        ("recovered_groups_dir", str(expected_output / "recovered-groups")),
        ("seed_artifact_dir", str(expected_seed)),
    )
    for field, expected in expected_mixed:
        if mixed.get(field) != expected:
            raise ValueError(f"manifest mixed artifact {field} is invalid")

    accounting_fields = {
        "logical_whole_model_tensor_payload_bytes",
        "incremental_disk_bytes",
        "logical_payload_limit_bytes",
    }
    if schema_version == 1:
        accounting_fields.add("non_routed_tensor_payload_bytes")
    else:
        accounting_fields.update(
            {
                "routed_codes_scales_bytes",
                "routed_codebook_bytes",
                "main_non_routed_tensor_payload_bytes",
            }
        )
    accounting = _require_exact_fields(
        "manifest accounting",
        manifest.get("accounting"),
        accounting_fields,
    )
    expected_accounting: dict[str, int] = {
        "logical_whole_model_tensor_payload_bytes": experiment.expected_payload_bytes,
        "incremental_disk_bytes": artifact_bytes_total,
        "logical_payload_limit_bytes": config.budget.payload_limit_bytes,
    }
    if schema_version == 1:
        expected_accounting["non_routed_tensor_payload_bytes"] = (
            _NON_ROUTED_PAYLOAD_BYTES
        )
    else:
        expected_accounting.update(
            {
                "routed_codes_scales_bytes": (
                    experiment.expected_payload_bytes
                    - _NON_ROUTED_PAYLOAD_BYTES
                    - _ARTIFACT_GROUP_COUNT * 1024
                ),
                "routed_codebook_bytes": _ARTIFACT_GROUP_COUNT * 1024,
                "main_non_routed_tensor_payload_bytes": _NON_ROUTED_PAYLOAD_BYTES,
            }
        )
    for field, expected in expected_accounting.items():
        actual = accounting.get(field)
        if type(actual) is not int or actual != expected:
            raise ValueError(f"manifest accounting {field} is invalid")


def _manifest_observation(
    root: Path,
    experiment: Experiment,
    config: CampaignConfig,
) -> _ManifestObservation:
    relative = PurePosixPath(experiment.output_path) / "conversion-manifest.json"
    state, payload, read_error = _stable_regular_bytes(root, relative)
    if state == "absent":
        return _ManifestObservation("absent", None, None)
    if state == "invalid" or payload is None:
        return _ManifestObservation("invalid", None, None, read_error)
    file_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        manifest = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if not isinstance(manifest, dict):
            raise ValueError("manifest root is not an object")
        recorded = manifest.get("manifest_body_sha256")
        if (
            not isinstance(recorded, str)
            or len(recorded) != 64
            or any(character not in "0123456789abcdef" for character in recorded)
        ):
            raise ValueError("manifest body SHA-256 is not canonical")
        body = dict(manifest)
        body.pop("manifest_body_sha256")
        canonical = hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if recorded != canonical:
            raise ValueError("manifest body SHA-256 mismatch")
        if manifest.get("record_type") != "glm52_recovery_conversion_manifest":
            raise ValueError("manifest record type is invalid")
        if manifest.get("schema_version") not in {1, 2}:
            raise ValueError("manifest schema version is invalid")
        if manifest.get("status") != "complete":
            raise ValueError("manifest is not complete")
        _validate_manifest_contract(
            manifest,
            config=config,
            experiment=experiment,
            root=root,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as error:
        return _ManifestObservation("invalid", file_sha256, None, str(error))
    return _ManifestObservation("valid", file_sha256, canonical)


def _select_experiment(
    config: CampaignConfig,
    root: Path,
    processes: tuple[ProcessObservation, ...],
) -> Experiment | None:
    if processes:
        name = processes[0].experiment_name
        return config.experiment(name) if name is not None else None
    for experiment in config.experiments:
        if experiment.transition_name is None:
            continue
        if _manifest_observation(root, experiment, config).state != "valid":
            return experiment
    return None


def _scan_directory(
    root: Path, relative: PurePosixPath
) -> tuple[list[tuple[str, bool, bool, str | None]], str, str | None]:
    try:
        descriptor = _open_directory(root, relative)
    except FileNotFoundError:
        return [], "absent", None
    except OSError as error:
        return [], "invalid", str(error)
    try:
        try:
            with os.scandir(descriptor) as entries:
                observed = [
                    (
                        entry.name,
                        entry.is_file(follow_symlinks=False),
                        entry.is_symlink(),
                        (
                            os.readlink(entry.name, dir_fd=descriptor)
                            if entry.is_symlink()
                            else None
                        ),
                    )
                    for entry in entries
                ]
                return sorted(observed), "present", None
        except OSError as error:
            return [], "invalid", str(error)
    finally:
        os.close(descriptor)


def _group_filename(name: str) -> tuple[int, str] | None:
    prefix = "layer-"
    suffix = ".safetensors"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    core = name[len(prefix) : -len(suffix)]
    try:
        layer_text, projection = core.split("-", 1)
        layer = int(layer_text)
    except ValueError:
        return None
    if len(layer_text) != 5 or layer_text != f"{layer:05d}" or projection not in _PROJECTIONS:
        return None
    return layer, projection


def _count_groups(
    root: Path, experiment: Experiment
) -> tuple[int, str, tuple[str, ...]]:
    entries, directory_state, scan_error = _scan_directory(
        root, PurePosixPath(experiment.output_path) / "recovered-groups"
    )
    contradictions: list[str] = []
    if scan_error is not None:
        contradictions.append(f"recovered-groups directory is invalid: {scan_error}")
        return 0, directory_state, tuple(contradictions)
    expected = {
        (layer, projection)
        for layer in experiment.recovered_layers
        for projection in _PROJECTIONS
    }
    count = 0
    for name, is_file, _is_symlink, _target in entries:
        group = _group_filename(name)
        if group is None:
            contradictions.append(f"noncanonical recovered-group entry: {name}")
            continue
        if not is_file:
            contradictions.append(
                f"canonical recovered-group is not a regular file: {name}"
            )
            continue
        count += 1
        if group not in expected:
            contradictions.append(
                f"noncanonical recovered-group for experiment {experiment.name}: {name}"
            )
    return count, directory_state, tuple(contradictions)


def _count_artifact_links(
    root: Path,
    experiment: Experiment,
    config: CampaignConfig,
    *,
    require_complete: bool,
) -> tuple[int, str, tuple[str, ...]]:
    entries, directory_state, scan_error = _scan_directory(
        root, PurePosixPath(experiment.output_path) / "artifact"
    )
    if scan_error is not None:
        return 0, directory_state, (
            f"artifact directory is invalid: {scan_error}",
        )
    canonical = {
        f"layer-{layer:05d}-{projection}.safetensors"
        for layer in range(3, 78)
        for projection in _PROJECTIONS
    }
    selected = {
        f"layer-{layer:05d}-{projection}.safetensors"
        for layer in experiment.recovered_layers
        for projection in _PROJECTIONS
    }
    authorities = {authority.name: authority for authority in config.authorities}
    seed_root = root.absolute() / Path(
        authorities["accepted-seed-manifest"].path
    ).parent
    artifact_root = root.absolute() / experiment.output_path / "artifact"
    contradictions: list[str] = []
    links = 0
    names = {name for name, _is_file, _is_symlink, _target in entries}
    for name, _is_file, is_symlink, target in entries:
        if name not in canonical:
            contradictions.append(f"artifact has noncanonical or extra entry: {name}")
            continue
        if not is_symlink or target is None:
            contradictions.append(f"artifact canonical entry is not a symlink: {name}")
            continue
        links += 1
        expected_target = (
            str(Path("../recovered-groups") / name)
            if name in selected
            else os.path.relpath(seed_root / name, artifact_root)
        )
        if os.path.isabs(target) or target != expected_target:
            contradictions.append(
                f"artifact symlink target is invalid for {name}: {target!r}"
            )
    if require_complete and names != canonical:
        missing = sorted(canonical - names)
        extra = sorted(names - canonical)
        contradictions.append(
            "valid manifest requires exact 225-name artifact inventory: "
            f"missing={missing}, extra={extra}"
        )
    return links, directory_state, tuple(contradictions)


def _other_manifest_contradictions(
    config: CampaignConfig,
    root: Path,
    *,
    selected: Experiment | None,
) -> tuple[str, ...]:
    """Reject corrupt or physically incomplete published evidence campaign-wide."""

    contradictions: list[str] = []
    for experiment in config.experiments:
        if experiment.transition_name is None or experiment is selected:
            continue
        manifest = _manifest_observation(root, experiment, config)
        if manifest.state == "absent":
            continue
        if manifest.state == "invalid":
            contradictions.append(
                f"{experiment.name} conversion manifest is invalid: "
                f"{manifest.error or 'unknown error'}"
            )
            continue
        recovered_groups, _groups_state, group_contradictions = _count_groups(
            root, experiment
        )
        contradictions.extend(
            f"{experiment.name}: {item}" for item in group_contradictions
        )
        artifact_links, _links_state, artifact_contradictions = _count_artifact_links(
            root,
            experiment,
            config,
            require_complete=True,
        )
        contradictions.extend(
            f"{experiment.name}: {item}" for item in artifact_contradictions
        )
        expected_groups = len(experiment.recovered_layers) * len(_PROJECTIONS)
        if recovered_groups > expected_groups:
            contradictions.append(
                f"{experiment.name} recovered groups exceed expected count: "
                f"{recovered_groups}>{expected_groups}"
            )
        if recovered_groups != expected_groups:
            contradictions.append(
                f"{experiment.name} valid manifest contradicts incomplete recovered "
                f"groups: {recovered_groups}/{expected_groups}"
            )
        if artifact_links != _ARTIFACT_GROUP_COUNT:
            contradictions.append(
                f"{experiment.name} valid manifest contradicts incomplete artifact "
                f"links: {artifact_links}/{_ARTIFACT_GROUP_COUNT}"
            )
    return tuple(contradictions)


def _dependency_contradictions(
    config: CampaignConfig,
    root: Path,
    experiment: Experiment,
) -> tuple[str, ...]:
    contradictions: list[str] = []
    for dependency_name in experiment.depends_on:
        dependency = config.experiment(dependency_name)
        manifest = _manifest_observation(root, dependency, config)
        if manifest.state != "valid":
            contradictions.append(
                f"dependency {dependency_name} manifest is {manifest.state}"
            )
            continue
        recovered, recovered_state, recovered_errors = _count_groups(root, dependency)
        links, links_state, link_errors = _count_artifact_links(
            root,
            dependency,
            config,
            require_complete=True,
        )
        expected = len(dependency.recovered_layers) * len(_PROJECTIONS)
        if (
            recovered_state != "present"
            or recovered != expected
            or recovered_errors
            or links_state != "present"
            or links != _ARTIFACT_GROUP_COUNT
            or link_errors
        ):
            contradictions.append(
                f"dependency {dependency_name} physical evidence is incomplete"
            )
    return tuple(contradictions)


def _observe_experiment_physical(
    config: CampaignConfig,
    root: Path,
    experiment: Experiment,
) -> ExperimentObservation:
    expected_groups = len(experiment.recovered_layers) * len(_PROJECTIONS)
    recovered_groups, groups_state, group_errors = _count_groups(root, experiment)
    manifest = _manifest_observation(root, experiment, config)
    artifact_links, links_state, link_errors = _count_artifact_links(
        root,
        experiment,
        config,
        require_complete=manifest.state == "valid",
    )
    contradictions = [*group_errors, *link_errors]
    if manifest.state == "invalid":
        contradictions.append(
            f"conversion manifest is invalid: {manifest.error or 'unknown error'}"
        )
    if manifest.state == "valid" and recovered_groups != expected_groups:
        contradictions.append(
            "valid manifest contradicts incomplete recovered groups: "
            f"{recovered_groups}/{expected_groups}"
        )
    if recovered_groups > expected_groups:
        contradictions.append(
            f"recovered groups exceed expected count: {recovered_groups}>{expected_groups}"
        )
    if manifest.state == "valid" and artifact_links != _ARTIFACT_GROUP_COUNT:
        contradictions.append(
            "valid manifest contradicts incomplete artifact links: "
            f"{artifact_links}/{_ARTIFACT_GROUP_COUNT}"
        )
    return ExperimentObservation(
        experiment_name=experiment.name,
        recovered_groups=recovered_groups,
        expected_groups=expected_groups,
        recovered_groups_state=groups_state,
        artifact_links=artifact_links,
        expected_artifact_links=_ARTIFACT_GROUP_COUNT,
        artifact_links_state=links_state,
        manifest_state=manifest.state,
        manifest_file_sha256=manifest.file_sha256,
        manifest_body_sha256=manifest.body_sha256,
        contradictions=tuple(contradictions),
    )


def observe_campaign(
    config: CampaignConfig,
    repo_root: str | Path,
    ps_output: str | None = None,
) -> CampaignObservation:
    """Observe campaign state without creating or mutating any filesystem path."""

    root = Path(repo_root)
    contradictions: list[str] = []
    lock_path = root / ".keep-heavy-job.lock"
    lock = _probe_lock(lock_path)
    lock_state = lock.state
    lock_held = lock.held
    if lock_state == "invalid":
        contradictions.append(f"heavy lock is invalid: {lock.error or 'unknown error'}")
    try:
        process_text = _read_ps_output() if ps_output is None else ps_output
        processes = _matching_processes(config, process_text)
    except (OSError, subprocess.SubprocessError) as error:
        processes = ()
        contradictions.append(f"process observation failed: {error}")
    if len(processes) > 1:
        contradictions.append("more than one matching heavy process is active")
    if processes and lock_state != "held":
        contradictions.append(
            f"declared active process has a {lock_state} lock instead of a held lock"
        )
    lock_owner_known = False
    if (
        lock_state == "held"
        and len(processes) == 1
        and lock.device is not None
        and lock.inode is not None
    ):
        openers = _read_lock_openers(
            lock_path,
            device=lock.device,
            inode=lock.inode,
        )
        if openers == (processes[0].pid,):
            lock_owner_known = True
        elif openers and (
            processes[0].pid not in openers or any(pid != processes[0].pid for pid in openers)
        ):
            contradictions.append(
                "declared active process has proven different opener(s) for the held lock: "
                + ", ".join(str(pid) for pid in openers)
            )
    if len(processes) > 1:
        experiment = None
    else:
        experiment = _select_experiment(config, root, processes)
    experiment_observations = tuple(
        _observe_experiment_physical(config, root, declared)
        for declared in config.experiments
    )
    snapshots = {
        snapshot.experiment_name: snapshot
        for snapshot in experiment_observations
    }
    for snapshot in experiment_observations:
        prefix = "" if experiment is not None and snapshot.experiment_name == experiment.name else f"{snapshot.experiment_name}: "
        contradictions.extend(prefix + item for item in snapshot.contradictions)

    recovered_groups = 0
    recovered_groups_state = "absent"
    expected_groups = 0
    artifact_links = 0
    artifact_links_state = "absent"
    manifest = _ManifestObservation("absent", None, None)
    if experiment is not None:
        selected = snapshots[experiment.name]
        expected_groups = selected.expected_groups
        recovered_groups = selected.recovered_groups
        recovered_groups_state = selected.recovered_groups_state
        artifact_links = selected.artifact_links
        artifact_links_state = selected.artifact_links_state
        manifest = _ManifestObservation(
            selected.manifest_state,
            selected.manifest_file_sha256,
            selected.manifest_body_sha256,
        )
        if len(processes) == 1 and processes[0].experiment_name == experiment.name:
            for dependency_name in experiment.depends_on:
                dependency = snapshots[dependency_name]
                if dependency.manifest_state != "valid":
                    contradictions.append(
                        f"dependency {dependency_name} manifest is "
                        f"{dependency.manifest_state}"
                    )
                elif not dependency.physically_complete:
                    contradictions.append(
                        f"dependency {dependency_name} physical evidence is incomplete"
                    )

    throughput: float | None = None
    eta: float | None = None
    if (
        experiment is not None
        and len(processes) == 1
        and lock_owner_known
        and processes[0].experiment_name == experiment.name
        and processes[0].elapsed_seconds is not None
        and processes[0].elapsed_seconds > 0
        and recovered_groups > 0
    ):
        throughput = recovered_groups / processes[0].elapsed_seconds
        eta = max(expected_groups - recovered_groups, 0) / throughput

    return CampaignObservation(
        lock_held=lock_held,
        lock_owner_known=lock_owner_known,
        active_processes=processes,
        recovered_groups=recovered_groups,
        expected_groups=expected_groups,
        manifest_sha256=manifest.file_sha256,
        contradictions=tuple(contradictions),
        campaign=config.campaign,
        experiment_name=experiment.name if experiment is not None else None,
        lock_state=lock_state,
        artifact_links=artifact_links,
        expected_artifact_links=_ARTIFACT_GROUP_COUNT,
        manifest_state=manifest.state,
        manifest_file_sha256=manifest.file_sha256,
        manifest_body_sha256=manifest.body_sha256,
        throughput_groups_per_second=throughput,
        eta_seconds=eta,
        recovered_groups_state=recovered_groups_state,
        artifact_links_state=artifact_links_state,
        experiment_observations=experiment_observations,
    )


__all__ = ["observe_campaign"]
